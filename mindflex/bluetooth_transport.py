from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass

from .parser import ThinkGearParser
from .transport import EndpointInfo, TransportKind

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - real backend is Windows-only
    import winrt.windows.foundation.collections  # noqa: F401
    from winrt.windows.devices.bluetooth import BluetoothAdapter, BluetoothCacheMode, BluetoothDevice
    from winrt.windows.devices.bluetooth.rfcomm import RfcommServiceId
    from winrt.windows.devices.enumeration import DeviceInformation, DeviceInformationKind
    from winrt.windows.networking.sockets import StreamSocket
    from winrt.windows.storage.streams import DataReader, InputStreamOptions
    WINRT_AVAILABLE = True
except ImportError:  # pragma: no cover
    BluetoothAdapter = BluetoothCacheMode = BluetoothDevice = None
    RfcommServiceId = DeviceInformation = DeviceInformationKind = None
    StreamSocket = DataReader = InputStreamOptions = None
    WINRT_AVAILABLE = False


@dataclass(frozen=True, slots=True)
class _ServiceCandidate:
    route: str
    service: object


class BluetoothWorker:
    """Bluetooth Classic transport with one responsibility: deliver bytes.

    Discovery, pairing and RFCOMM live in this worker because WinRT objects are
    apartment/loop sensitive. ThinkGear interpretation does *not* live here.
    The only parser created here is a short-lived probe parser used to reject a
    non-TGAM RFCOMM channel; the application controller remains the sole owner
    of persistent EEG state.
    """

    backend_available = WINRT_AVAILABLE
    CLASSIC_BT_AQS = 'System.Devices.Aep.ProtocolId:="{e0cbf06c-cd8b-4647-bb8a-263b43f0f974}"'
    REQUESTED_PROPERTIES = [
        "System.Devices.Aep.DeviceAddress",
        "System.Devices.Aep.IsConnected",
        "System.Devices.Aep.IsPresent",
        "System.Devices.Aep.SignalStrength",
    ]
    PROBE_PACKETS = 4
    PROBE_TIMEOUT = 4.0
    READ_SIZE = 4096
    DISCOVERY_TIMEOUT = 3.0
    CONNECT_TIMEOUT = 5.0
    DISCOVERY_ROUNDS = 2

    def __init__(
        self,
        gui_queue,
        *,
        on_connecting=None,
        on_connected=None,
        on_bytes=None,
        on_disconnected=None,
        on_error=None,
    ) -> None:
        self.gui_queue = gui_queue
        self.on_connecting = on_connecting or (lambda endpoint: None)
        self.on_connected = on_connected or (lambda endpoint: None)
        self.on_bytes = on_bytes or (lambda data: None)
        self.on_disconnected = on_disconnected or (lambda: None)
        self.on_error = on_error or (lambda exc: None)

        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.loop_ready = threading.Event()
        self._closed = False

        self.watcher = None
        self.watcher_tokens: list[tuple[str, object]] = []
        self.devices: dict[str, object] = {}

        self.bt_device = None
        self.service = None
        self.socket = None
        self.reader = None
        self.receive_task: asyncio.Task | None = None
        self.current_endpoint: EndpointInfo | None = None
        self._pending_endpoint: EndpointInfo | None = None

    # ------------------------------------------------------------------
    # Worker/event-loop lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self._closed = False
        self.loop_ready.clear()
        self.thread = threading.Thread(target=self._thread_main, name="MindFlexBluetooth", daemon=True)
        self.thread.start()

    def _thread_main(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop_ready.set()
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()

    def submit(self, coro):
        if self._closed:
            coro.close()
            return None
        self.start()
        if not self.loop_ready.wait(timeout=2.0):
            coro.close()
            self.emit("error", "Bluetooth worker failed to initialize.")
            return None
        if self.loop is None or self.loop.is_closed():
            coro.close()
            self.emit("error", "Bluetooth worker is unavailable.")
            return None
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def emit(self, event: str, data=None) -> None:
        self.gui_queue.put((event, data))

    def emit_status(self, key: str, **values) -> None:
        """Emit a translatable status message without coupling transport to i18n."""
        self.emit("status", {"key": key, "values": values})

    # ------------------------------------------------------------------
    # DeviceWatcher
    # ------------------------------------------------------------------

    @staticmethod
    def _prop(info, name: str, default=None):
        try:
            value = info.properties.get(name)
            return default if value is None else value
        except (AttributeError, KeyError, TypeError, RuntimeError, OSError):
            return default

    def device_to_dict(self, info) -> dict:
        name = (getattr(info, "name", "") or "").strip() or "(unnamed device)"
        address = self._prop(info, "System.Devices.Aep.DeviceAddress", "unknown")
        connected = bool(self._prop(info, "System.Devices.Aep.IsConnected", False))
        present = self._prop(info, "System.Devices.Aep.IsPresent", None)
        signal = self._prop(info, "System.Devices.Aep.SignalStrength", None)
        try:
            paired = bool(info.pairing.is_paired)
        except (AttributeError, RuntimeError, OSError):
            paired = False
        device_id = str(info.id)
        endpoint = EndpointInfo(
            id=f"winbt:{device_id}",
            kind=TransportKind.BLUETOOTH,
            backend="winrt-rfcomm",
            address=str(address),
            description=name,
            system_id=device_id,
            paired=paired,
            present=present,
            connected=connected,
            signal=signal,
            detected_now=True,
        )
        return {
            "id": device_id,
            "name": name,
            "address": str(address),
            "paired": paired,
            "connected": connected,
            "present": present,
            "signal": signal,
            "endpoint": endpoint,
        }

    async def start_scan(self) -> None:
        if not WINRT_AVAILABLE:
            self.emit("error", "WinRT Bluetooth backend is unavailable in this environment.")
            return
        # Scanning and an active RFCOMM stream are mutually exclusive. This is
        # one state transition, not a hidden retry rule.
        await self.disconnect_internal(notify=False)
        self.stop_scan()
        self.emit("scan_started")
        self.emit_status("connection.bluetooth.status.adapter_check")
        try:
            adapter = await BluetoothAdapter.get_default_async()
            if adapter is None:
                raise RuntimeError("No Bluetooth adapter was found.")
            if not adapter.is_classic_supported:
                raise RuntimeError("The adapter does not support Bluetooth Classic.")
        except (OSError, RuntimeError) as exc:
            self.emit("error", str(exc))
            self.emit("scan_stopped")
            return

        self.devices.clear()
        self.emit("clear_devices")
        self.emit_status("connection.bluetooth.status.live_list")
        try:
            self.watcher = DeviceInformation.create_watcher_with_kind_aqs_filter_and_additional_properties(
                self.CLASSIC_BT_AQS,
                self.REQUESTED_PROPERTIES,
                DeviceInformationKind.ASSOCIATION_ENDPOINT,
            )
        except (OSError, RuntimeError, TypeError) as exc:
            self.emit("error", f"Could not create DeviceWatcher:\n\n{exc}")
            self.emit("scan_stopped")
            return

        def on_added(sender, info):
            del sender
            try:
                self.devices[info.id] = info
                self.emit("device_upsert", self.device_to_dict(info))
            except Exception as exc:  # WinRT callbacks must never escape
                self.emit("debug", f"DeviceWatcher added: {exc}")

        def on_updated(sender, update):
            del sender
            try:
                info = self.devices.get(update.id)
                if info is not None:
                    info.update(update)
                    self.emit("device_upsert", self.device_to_dict(info))
            except Exception as exc:
                self.emit("debug", f"DeviceWatcher updated: {exc}")

        def on_removed(sender, update):
            del sender
            self.devices.pop(update.id, None)
            self.emit("device_removed", update.id)

        def on_completed(sender, obj):
            del sender, obj
            self.emit_status("connection.bluetooth.status.initial_complete", count=len(self.devices))

        def on_stopped(sender, obj):
            del sender, obj
            self.emit("scan_stopped")

        self.watcher_tokens = [
            ("added", self.watcher.add_added(on_added)),
            ("updated", self.watcher.add_updated(on_updated)),
            ("removed", self.watcher.add_removed(on_removed)),
            ("enumeration_completed", self.watcher.add_enumeration_completed(on_completed)),
            ("stopped", self.watcher.add_stopped(on_stopped)),
        ]
        self.watcher.start()

    def stop_scan(self) -> None:
        watcher, self.watcher = self.watcher, None
        if watcher is None:
            return
        try:
            watcher.stop()
        except (OSError, RuntimeError):
            pass
        for kind, token in self.watcher_tokens:
            try:
                getattr(watcher, f"remove_{kind}")(token)
            except (AttributeError, OSError, RuntimeError, ValueError):
                pass
        self.watcher_tokens = []

    # ------------------------------------------------------------------
    # RFCOMM discovery / validation
    # ------------------------------------------------------------------

    async def _bounded(self, awaitable, timeout: float, label: str):
        try:
            return await asyncio.wait_for(awaitable, timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            self.emit("debug", f"{label}: timeout after {timeout:.1f}s")
            return None

    @staticmethod
    def _service_key(service) -> tuple[str, str, str]:
        host = getattr(service, "connection_host_name", None)
        host_name = getattr(host, "raw_name", None) or (str(host) if host is not None else "")
        channel = str(getattr(service, "connection_service_name", ""))
        service_id = getattr(service, "service_id", None)
        try:
            service_uuid = str(service_id.as_string()) if service_id is not None else ""
        except Exception:
            service_uuid = str(service_id) if service_id is not None else ""
        return str(host_name), channel, service_uuid

    @staticmethod
    def _is_serial_port_service(service) -> bool:
        service_id = getattr(service, "service_id", None)
        if service_id is None:
            return False
        try:
            return int(service_id.as_short_id()) == 0x1101
        except Exception:
            try:
                return "00001101-0000-1000-8000-00805f9b34fb" in str(service_id.as_string()).lower()
            except Exception:
                return False

    @staticmethod
    def _close_service(service) -> None:
        if service is None:
            return
        try:
            service.close()
        except Exception:
            pass

    async def _discover_candidates(self) -> list[_ServiceCandidate]:
        """One deterministic discovery policy; no persistent cache path.

        A live UNCACHED query is authoritative. The default WinRT query is the
        fallback for stacks that do not implement UNCACHED well. If the targeted
        Serial Port query returns nothing, enumerate all RFCOMM and filter UUID
        0x1101. CACHED is intentionally absent: a stale cache must never be part
        of the connection state machine.
        """
        if self.bt_device is None:
            return []
        serial_id = RfcommServiceId.serial_port
        candidates: list[_ServiceCandidate] = []
        seen: set[tuple[str, str, str]] = set()

        attempts = [
            ("SPP UNCACHED", lambda: self.bt_device.get_rfcomm_services_for_id_with_cache_mode_async(serial_id, BluetoothCacheMode.UNCACHED)),
            ("SPP default", lambda: self.bt_device.get_rfcomm_services_for_id_async(serial_id)),
        ]
        for label, operation in attempts:
            self.emit_status("connection.bluetooth.status.discovering", route=label)
            try:
                result = await self._bounded(operation(), self.DISCOVERY_TIMEOUT, label)
            except (OSError, RuntimeError, AttributeError) as exc:
                self.emit("debug", f"{label}: {exc}")
                continue
            if result is None:
                continue
            for service in list(result.services):
                key = self._service_key(service)
                if key in seen:
                    self._close_service(service)
                    continue
                seen.add(key)
                candidates.append(_ServiceCandidate(label, service))
            if candidates:
                return candidates

        all_attempts = [
            ("RFCOMM UNCACHED", lambda: self.bt_device.get_rfcomm_services_with_cache_mode_async(BluetoothCacheMode.UNCACHED)),
            ("RFCOMM default", lambda: self.bt_device.get_rfcomm_services_async()),
        ]
        for label, operation in all_attempts:
            self.emit_status("connection.bluetooth.status.discovering", route=label)
            try:
                result = await self._bounded(operation(), self.DISCOVERY_TIMEOUT, label)
            except (OSError, RuntimeError, AttributeError) as exc:
                self.emit("debug", f"{label}: {exc}")
                continue
            if result is None:
                continue
            for service in list(result.services):
                if not self._is_serial_port_service(service):
                    self._close_service(service)
                    continue
                key = self._service_key(service)
                if key in seen:
                    self._close_service(service)
                    continue
                seen.add(key)
                candidates.append(_ServiceCandidate(label, service))
            if candidates:
                return candidates
        return candidates

    async def _probe(self) -> tuple[bool, str]:
        if self.socket is None:
            return False, "socket unavailable"
        parser = ThinkGearParser()
        self.reader = DataReader(self.socket.input_stream)
        self.reader.input_stream_options = InputStreamOptions.PARTIAL
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.PROBE_TIMEOUT
        bytes_received = 0
        while self.socket is not None and parser.packets < self.PROBE_PACKETS:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                count = await asyncio.wait_for(self.reader.load_async(1024), timeout=remaining)
            except asyncio.TimeoutError:
                break
            if count == 0:
                break
            data = bytearray(count)
            self.reader.read_bytes(data)
            bytes_received += count
            parser.feed(data)
            # Probe bytes are real EEG bytes; preserve them in the single
            # controller pipeline instead of discarding the first eSense rows.
            self.on_bytes(bytes(data))
            self.emit("thinkgear_probe", {
                "bytes": bytes_received,
                "sync": parser.sync_count,
                "packets": parser.packets,
                "errors": parser.bad_checksums,
                "checksum_errors": parser.bad_checksums,
            })
        ok = parser.packets >= self.PROBE_PACKETS
        return ok, f"bytes={bytes_received}; sync={parser.sync_count}; packets={parser.packets}; checksum_err={parser.bad_checksums}"

    async def _close_channel(self, *, close_service: bool = True) -> None:
        if self.reader is not None:
            try:
                self.reader.detach_stream()
            except Exception:
                pass
            try:
                self.reader.close()
            except Exception:
                pass
            self.reader = None
        if self.socket is not None:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None
        if close_service and self.service is not None:
            self._close_service(self.service)
            self.service = None
        await asyncio.sleep(0)

    async def connect(self, device_id: str) -> None:
        self.stop_scan()
        await self.disconnect_internal(notify=False)
        info = self.devices.get(device_id)
        if info is None:
            self.emit("connect_failed", "The device is no longer available. Search again.")
            return
        data = self.device_to_dict(info)
        endpoint: EndpointInfo = data["endpoint"]
        self._pending_endpoint = endpoint
        try:
            self.on_connecting(endpoint)
        except Exception:
            pass
        self.emit("connecting", data["name"])

        try:
            try:
                if not info.pairing.is_paired and info.pairing.can_pair:
                    self.emit_status("connection.bluetooth.status.pairing", device=data["name"])
                    await self._bounded(info.pairing.pair_async(), 15.0, "Pairing")
                    await asyncio.sleep(1.0)
            except (OSError, RuntimeError, AttributeError) as exc:
                self.emit("debug", f"Pairing: {exc}")

            last_diagnostic = "no channel tested"
            for round_index in range(self.DISCOVERY_ROUNDS):
                if round_index:
                    await asyncio.sleep(0.7)
                self.emit_status("connection.bluetooth.status.opening", attempt=round_index + 1, total=self.DISCOVERY_ROUNDS)
                self.bt_device = await self._bounded(BluetoothDevice.from_id_async(device_id), self.CONNECT_TIMEOUT, "Open BluetoothDevice")
                if self.bt_device is None:
                    continue
                candidates = await self._discover_candidates()
                for index, candidate in enumerate(candidates, 1):
                    self.service = candidate.service
                    self.socket = StreamSocket()
                    channel = str(getattr(self.service, "connection_service_name", "?"))
                    self.emit_status("connection.bluetooth.status.testing_channel", index=index, total=len(candidates), route=candidate.route, channel=channel)
                    try:
                        result = self.socket.connect_async(
                            self.service.connection_host_name,
                            self.service.connection_service_name,
                        )
                        await self._bounded(result, self.CONNECT_TIMEOUT, "Open RFCOMM")
                    except (OSError, RuntimeError, AttributeError) as exc:
                        last_diagnostic = f"{candidate.route}: RFCOMM could not open: {exc}"
                        await self._close_channel(close_service=True)
                        continue
                    try:
                        valid, diagnostic = await self._probe()
                    except (OSError, RuntimeError, AttributeError) as exc:
                        valid, diagnostic = False, f"probe error: {exc}"
                    last_diagnostic = f"{candidate.route}: {diagnostic}"
                    if not valid:
                        await self._close_channel(close_service=True)
                        continue

                    for unused in candidates[index:]:
                        self._close_service(unused.service)
                    self.current_endpoint = endpoint
                    self._pending_endpoint = None
                    try:
                        self.on_connected(endpoint)
                    except Exception:
                        pass
                    self.emit("bluetooth_connected", data)
                    self.emit("mindflex_confirmed", data)
                    self.emit_status("connection.bluetooth.status.confirmed_detail", diagnostic=diagnostic)
                    self.receive_task = asyncio.create_task(self.receive_loop())
                    return

                for candidate in candidates:
                    if candidate.service is not self.service:
                        self._close_service(candidate.service)
                if self.bt_device is not None:
                    try:
                        self.bt_device.close()
                    except Exception:
                        pass
                    self.bt_device = None

            raise RuntimeError(f"No RFCOMM channel delivered valid ThinkGear data. {last_diagnostic}")
        except Exception as exc:
            await self.disconnect_internal(notify=False)
            self._pending_endpoint = None
            try:
                self.on_error(exc)
            except Exception:
                pass
            self.emit("connect_failed", str(exc))

    async def receive_loop(self) -> None:
        try:
            if self.reader is None:
                if self.socket is None:
                    raise RuntimeError("RFCOMM socket is unavailable after validation.")
                self.reader = DataReader(self.socket.input_stream)
                self.reader.input_stream_options = InputStreamOptions.PARTIAL
            while self.socket is not None:
                count = await self.reader.load_async(self.READ_SIZE)
                if count == 0:
                    break
                data = bytearray(count)
                self.reader.read_bytes(data)
                self.on_bytes(bytes(data))
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self.emit_status("connection.bluetooth.status.interrupted", error=str(exc))
        finally:
            await self.disconnect_internal(notify=True, from_receive=True)

    # ------------------------------------------------------------------
    # Disconnect/close
    # ------------------------------------------------------------------

    async def disconnect(self) -> None:
        self.stop_scan()
        await self.disconnect_internal(notify=True)

    async def disconnect_internal(self, notify: bool = True, from_receive: bool = False) -> None:
        task, self.receive_task = self.receive_task, None
        if task is not None and not from_receive and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self._close_channel(close_service=True)
        if self.bt_device is not None:
            try:
                self.bt_device.close()
            except Exception:
                pass
            self.bt_device = None
        was_connected = self.current_endpoint is not None
        self.current_endpoint = None
        self._pending_endpoint = None
        if was_connected:
            try:
                self.on_disconnected()
            except Exception:
                pass
        if notify:
            self.emit("disconnected")
            self.emit_status("connection.disconnected")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop, thread = self.loop, self.thread
        if loop is not None and not loop.is_closed():
            try:
                future = asyncio.run_coroutine_threadsafe(self.disconnect_internal(notify=False), loop)
                future.result(timeout=2.5)
            except Exception:
                pass
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self.thread = None
