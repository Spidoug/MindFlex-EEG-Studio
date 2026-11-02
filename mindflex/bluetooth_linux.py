"""BlueZ / native RFCOMM Bluetooth Classic backend for Linux.

Uses bluetoothctl for discovery and optional pairing, sdptool (when present)
for Serial Port Profile channel lookup, and Python's Linux RFCOMM sockets for
ThinkGear bytes. No WinRT, BLE-only backend or root access is required.
"""
from __future__ import annotations

import asyncio
import logging
import re
import shutil
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass

from .parser import ThinkGearParser
from .transport import EndpointInfo, TransportKind

LOGGER = logging.getLogger(__name__)
ADDRESS = re.compile(r'(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f:])')
DEVICE_LINE = re.compile(r'^\s*(?:\[[^]]+\]\s*)?Device\s+((?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2})(?:\s+(.*))?$', re.I)
CHANNEL_LINE = re.compile(r'^\s*Channel:\s*(\d+)\s*$', re.I | re.M)


@dataclass(frozen=True)
class _Device:
    address: str
    name: str
    paired: bool = False


class BluetoothWorker:
    """Event-compatible Linux implementation of the Windows BluetoothWorker."""

    backend_available = (
        sys.platform.startswith('linux')
        and hasattr(socket, 'AF_BLUETOOTH')
        and hasattr(socket, 'BTPROTO_RFCOMM')
    )
    CONNECT_TIMEOUT = 5.0
    PROBE_TIMEOUT = 4.0
    PROBE_PACKETS = 4
    READ_SIZE = 4096
    DISCOVERY_SECONDS = 7
    FALLBACK_CHANNELS = (1, 2, 3, 4, 5)

    def __init__(
        self, gui_queue, *, on_connecting=None, on_connected=None,
        on_bytes=None, on_disconnected=None, on_error=None,
    ):
        self.gui_queue = gui_queue
        self.on_connecting = on_connecting or (lambda endpoint: None)
        self.on_connected = on_connected or (lambda endpoint: None)
        self.on_bytes = on_bytes or (lambda chunk: None)
        self.on_disconnected = on_disconnected or (lambda: None)
        self.on_error = on_error or (lambda exc: None)
        self.loop = None
        self.thread = None
        self.loop_ready = threading.Event()
        self._closed = False
        self._scan_revision = 0
        self.devices: dict[str, _Device] = {}
        self.socket: socket.socket | None = None
        self.receive_task: asyncio.Task | None = None
        self.current_endpoint: EndpointInfo | None = None

    def emit(self, kind, payload=None):
        self.gui_queue.put((kind, payload))

    def emit_status(self, key, **values):
        self.emit('status', {'key': key, 'values': values})

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self._closed = False
        self.loop_ready.clear()
        self.thread = threading.Thread(target=self._thread_main, name='MindFlexBlueZ', daemon=True)
        self.thread.start()

    def _thread_main(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop_ready.set()
        try:
            self.loop.run_forever()
        finally:
            for task in asyncio.all_tasks(self.loop):
                task.cancel()
            pending = asyncio.all_tasks(self.loop)
            if pending:
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()

    def submit(self, coroutine):
        if self._closed:
            coroutine.close()
            return None
        self.start()
        if not self.loop_ready.wait(timeout=2) or self.loop is None or self.loop.is_closed():
            coroutine.close()
            self.emit('error', 'The BlueZ worker could not initialize.')
            return None
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    @staticmethod
    def _command(args, timeout):
        """No shell expansion; bounded subprocess, including missing binaries."""
        try:
            completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
            return completed.returncode, (completed.stdout + '\n' + completed.stderr).strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            return -1, str(exc)

    async def _call(self, args, timeout=4):
        return await asyncio.to_thread(self._command, args, timeout)

    @staticmethod
    def _parse_devices(output):
        result = {}
        for line in output.splitlines():
            # An event like "[CHG] Device ... Connected: yes" is NOT a name.
            if line.lstrip().startswith(('[CHG]', '[DEL]')):
                continue
            match = DEVICE_LINE.match(line)
            if match:
                address = match.group(1).upper()
                name = (match.group(2) or '').strip()
                if name.startswith('('):
                    name = ''
                result[address] = name or address
        return result

    @staticmethod
    def _serial_channels(output):
        """Extract channels belonging to an SDP Serial Port service, not OBEX."""
        channels = []
        for block in re.split(r'(?=^Service Name:)', output, flags=re.M | re.I):
            if not re.search(r'(?:Serial Port|0x1101)', block, re.I):
                continue
            for channel in CHANNEL_LINE.findall(block):
                number = int(channel)
                if 1 <= number <= 30 and number not in channels:
                    channels.append(number)
        return channels

    def _endpoint(self, device):
        return EndpointInfo(
            id='linuxbt:' + device.address,
            kind=TransportKind.BLUETOOTH,
            backend='bluez-rfcomm',
            address=device.address,
            description=device.name,
            system_id=device.address,
            paired=device.paired,
            present=None,
            connected=False,
            detected_now=True,
        )

    def _present_device(self, device):
        endpoint = self._endpoint(device)
        return {
            'id': device.address,
            'name': device.name,
            'address': device.address,
            'paired': device.paired,
            'connected': False,
            'present': None,
            'signal': None,
            'endpoint': endpoint,
        }

    def stop_scan(self):
        # Incrementing invalidates any in-flight scan without affecting sockets.
        self._scan_revision += 1

    async def start_scan(self):
        self.stop_scan()
        revision = self._scan_revision
        await self.disconnect_internal(notify=False)
        self.emit('scan_started')
        self.emit('clear_devices')
        try:
            if not self.backend_available:
                raise RuntimeError('The Python interpreter lacks Linux RFCOMM socket support. Use the distribution Python.')
            if not shutil.which('bluetoothctl'):
                raise RuntimeError('BlueZ is missing. Install the bluez package and enable the bluetooth service.')
            status, output = await self._call(['bluetoothctl', 'show'])
            if status != 0 or not re.search(r'Powered:\s+yes', output, re.I):
                raise RuntimeError('Bluetooth adapter is unavailable or powered off. Check bluetoothctl show and systemctl status bluetooth.')
            self.devices.clear()
            await self._refresh_devices(revision)
            # A bounded BR/EDR scan discovers Classic headsets without BLE-only APIs.
            await self._call(['bluetoothctl', '--timeout', str(self.DISCOVERY_SECONDS), 'scan', 'bredr'], timeout=self.DISCOVERY_SECONDS + 4)
            if revision == self._scan_revision:
                await self._refresh_devices(revision)
                if not self.devices:
                    self.emit('debug', 'No Bluetooth Classic devices found; pair the MindFlex in Bluetooth settings and search again.')
        except Exception as exc:
            if revision == self._scan_revision:
                self.emit('scan_stopped')
                self.emit('error', str(exc))
        else:
            if revision == self._scan_revision:
                self.emit('scan_stopped')

    async def _refresh_devices(self, revision):
        _, paired_output = await self._call(['bluetoothctl', 'devices', 'Paired'])
        _, discovered_output = await self._call(['bluetoothctl', 'devices'])
        if revision != self._scan_revision:
            return
        paired = self._parse_devices(paired_output)
        found = {**paired, **self._parse_devices(discovered_output)}
        for address, name in sorted(found.items()):
            device = _Device(address=address, name=name, paired=address in paired)
            self.devices[address] = device
            self.emit('device_upsert', self._present_device(device))

    @staticmethod
    def _open_socket(address, channel, timeout):
        sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        try:
            sock.settimeout(timeout)
            sock.connect((address, channel))
            sock.setblocking(False)
            return sock
        except BaseException:
            sock.close()
            raise

    async def _channels(self, address):
        candidates = []
        if shutil.which('sdptool'):
            code, output = await self._call(['sdptool', 'browse', address], timeout=7)
            if code == 0:
                candidates.extend(self._serial_channels(output))
            else:
                self.emit('debug', f'SDP channel discovery failed: {output[:160]}')
        for channel in self.FALLBACK_CHANNELS:
            if channel not in candidates:
                candidates.append(channel)
        return candidates

    async def _probe(self, sock):
        parser = ThinkGearParser()
        loop = asyncio.get_running_loop()
        until = loop.time() + self.PROBE_TIMEOUT
        collected = []
        total = 0
        raw_count = 0
        while (parser.packets < self.PROBE_PACKETS or raw_count < self.PROBE_PACKETS) and loop.time() < until:
            try:
                data = await asyncio.wait_for(loop.sock_recv(sock, 1024), timeout=until - loop.time())
            except (asyncio.TimeoutError, TimeoutError):
                break
            if not data:
                break
            collected.append(data)
            total += len(data)
            raw_count += sum(len(item.raw) for item in parser.feed(data))
            self.emit('thinkgear_probe', {
                'bytes': total, 'sync': parser.sync_count,
                'packets': parser.packets, 'errors': parser.bad_checksums,
                'checksum_errors': parser.bad_checksums,
            })
        diagnostic = f'bytes={total}; sync={parser.sync_count}; packets={parser.packets}; raw={raw_count}; checksum_err={parser.bad_checksums}'
        # No unvalidated stream bytes are ever routed to the EEG controller.
        return (parser.packets >= self.PROBE_PACKETS and raw_count >= self.PROBE_PACKETS), collected, diagnostic

    async def connect(self, device_id):
        self.stop_scan()
        await self.disconnect_internal(notify=False)
        device = self.devices.get(device_id)
        if device is None:
            self.emit('connect_failed', 'The Bluetooth device is no longer listed. Search again.')
            return
        endpoint = self._endpoint(device)
        try:
            self.on_connecting(endpoint)
            self.emit('connecting', device.name)
            if not device.paired:
                self.emit_status('connection.bluetooth.status.pairing', device=device.name)
                code, output = await self._call(['bluetoothctl', '--timeout', '15', 'pair', device.address], timeout=17)
                if code != 0 or not ('Pairing successful' in output or 'AlreadyExists' in output):
                    raise RuntimeError(
                        f'Could not pair {device.address}. Pair the MindFlex in Linux Bluetooth Settings '
                        f'(PIN if requested) and search again. {output[:180]}'
                    )
                device = _Device(device.address, device.name, paired=True)
                self.devices[device.address] = device
                self.emit('device_upsert', self._present_device(device))
            channels = await self._channels(device.address)
            last_error = 'no RFCOMM channel responded'
            for index, channel in enumerate(channels, 1):
                self.emit_status('connection.bluetooth.status.testing_channel',
                                 index=index, total=len(channels), route='BlueZ RFCOMM', channel=channel)
                try:
                    sock = await asyncio.to_thread(self._open_socket, device.address, channel, self.CONNECT_TIMEOUT)
                    self.socket = sock
                    valid, initial_data, diagnostic = await self._probe(sock)
                    if valid:
                        # The EEG controller drops bytes until connection confirmation.
                        # Confirm first, then deliver the exact probe bytes once.
                        self.current_endpoint = endpoint
                        self.on_connected(endpoint)
                        for chunk in initial_data:
                            self.on_bytes(chunk)
                        data = self._present_device(device)
                        data['endpoint'] = endpoint
                        self.emit('bluetooth_connected', data)
                        self.emit('mindflex_confirmed', data)
                        self.emit_status('connection.bluetooth.status.confirmed_detail', diagnostic=diagnostic)
                        self.receive_task = asyncio.create_task(self.receive_loop())
                        return
                    last_error = f'channel {channel}: {diagnostic}'
                except (OSError, RuntimeError, asyncio.TimeoutError) as exc:
                    last_error = f'channel {channel}: {exc}'
                finally:
                    # Successful socket remains owned by receive_loop.
                    if self.current_endpoint is None:
                        self._close_socket()
            raise RuntimeError('No RFCOMM channel delivered valid ThinkGear packets. ' + last_error)
        except Exception as exc:
            await self.disconnect_internal(notify=False)
            try:
                self.on_error(exc)
            except Exception:
                LOGGER.exception('Bluetooth error callback failed')
            self.emit('connect_failed', str(exc))

    def _close_socket(self):
        sock, self.socket = self.socket, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                LOGGER.debug('Bluetooth socket close failed', exc_info=True)

    async def receive_loop(self):
        try:
            while self.socket is not None:
                chunk = await asyncio.get_running_loop().sock_recv(self.socket, self.READ_SIZE)
                if not chunk:
                    break
                self.on_bytes(chunk)
        except asyncio.CancelledError:
            return
        except (OSError, RuntimeError) as exc:
            self.emit_status('connection.bluetooth.status.interrupted', error=str(exc))
        finally:
            await self.disconnect_internal(notify=True, from_receive=True)

    async def disconnect(self):
        self.stop_scan()
        await self.disconnect_internal(notify=True)

    async def disconnect_internal(self, notify=True, from_receive=False):
        task, self.receive_task = self.receive_task, None
        if task is not None and not from_receive and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._close_socket()
        connected = self.current_endpoint is not None
        self.current_endpoint = None
        if connected:
            try:
                self.on_disconnected()
            except Exception:
                LOGGER.exception('Bluetooth disconnected callback failed')
        if notify:
            self.emit('disconnected')
            self.emit_status('connection.disconnected')

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.stop_scan()
        if self.loop is not None and not self.loop.is_closed():
            try:
                future = asyncio.run_coroutine_threadsafe(self.disconnect_internal(notify=False), self.loop)
                future.result(timeout=3)
            except Exception:
                LOGGER.warning('BlueZ worker could not disconnect during shutdown', exc_info=True)
            try:
                self.loop.call_soon_threadsafe(self.loop.stop)
            except RuntimeError:
                pass
        if self.thread is not None and self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)
        self.thread = None
