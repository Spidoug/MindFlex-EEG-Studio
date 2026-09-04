from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .settings import MINDFLEX_BAUDRATE

try:
    import serial
    from serial.tools import list_ports
except ImportError:  # pragma: no cover
    serial = None
    list_ports = None


class TransportKind(str, Enum):
    SERIAL = "serial"
    BLUETOOTH = "bluetooth"


class TransportMode(str, Enum):
    SERIAL = "serial"
    BLUETOOTH = "bluetooth"


@dataclass(frozen=True, slots=True)
class EndpointInfo:
    id: str
    kind: TransportKind
    backend: str
    address: str
    description: str = ""
    hwid: str = ""
    channel: int = 1
    system_id: str = ""
    paired: bool | None = None
    present: bool | None = None
    connected: bool = False
    signal: int | None = None
    detected_now: bool = False

    @property
    def display_name(self) -> str:
        prefix = "Bluetooth" if self.kind is TransportKind.BLUETOOTH else "Serial/USB"
        detail = self.description.strip()
        if detail and detail != self.address:
            return f"{prefix} · {self.address} — {detail}"
        return f"{prefix} · {self.address}"


_BLUETOOTH_MARKERS = ("bluetooth", "bthenum", "rfcomm", "standard serial over bluetooth")


def _is_bluetooth_serial(device: str, description: str, hwid: str) -> bool:
    text = " ".join((device, description, hwid)).lower()
    return any(marker in text for marker in _BLUETOOTH_MARKERS) or device.lower().startswith("/dev/rfcomm")


def _serial_endpoints() -> list[EndpointInfo]:
    """Return physical/USB serial endpoints only.

    Bluetooth virtual COM ports are deliberately excluded. Bluetooth selection
    is owned exclusively by the live DeviceWatcher in
    ``bluetooth_transport.py``.
    """
    if list_ports is None:
        return []
    endpoints: list[EndpointInfo] = []
    for port in list_ports.comports():
        device = str(port.device)
        description = str(port.description or device)
        hwid = str(port.hwid or "")
        if _is_bluetooth_serial(device, description, hwid):
            continue
        endpoints.append(
            EndpointInfo(
                id=f"serial:{device}",
                kind=TransportKind.SERIAL,
                backend="serial",
                address=device,
                description=description,
                hwid=hwid,
            )
        )
    return endpoints


def discover_endpoints(mode: TransportMode | str = TransportMode.SERIAL) -> list[EndpointInfo]:
    mode = TransportMode(mode)
    if mode is TransportMode.BLUETOOTH:
        # Bluetooth devices are live AssociationEndpoint events, not static serial endpoints.
        return []
    return _serial_endpoints()


def manual_endpoint(value: str, mode: TransportMode | str = TransportMode.SERIAL) -> EndpointInfo:
    mode = TransportMode(mode)
    text = value.strip()
    if not text:
        raise ValueError("An endpoint is required")
    if mode is TransportMode.BLUETOOTH:
        raise ValueError("Bluetooth devices must be selected from the live Bluetooth Classic list")
    return EndpointInfo(f"serial:{text}", TransportKind.SERIAL, "serial", text, text)


def endpoint_from_id(endpoint_id: str) -> EndpointInfo | None:
    text = (endpoint_id or "").strip()
    if text.startswith("serial:"):
        address = text.split(":", 1)[1]
        return EndpointInfo(text, TransportKind.SERIAL, "serial", address, address)
    # Bluetooth AssociationEndpoint IDs are intentionally not reconstructed
    # from settings. Bluetooth selection always comes from a live DeviceWatcher
    # row before connection.
    return None


class _SerialBackend:
    def __init__(self, on_bytes: Callable[[bytes], None], on_error: Callable[[Exception], None]) -> None:
        self._on_bytes = on_bytes
        self._on_error = on_error
        self._serial = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()

    @property
    def connected(self) -> bool:
        with self._lock:
            return bool(self._serial and getattr(self._serial, "is_open", False))

    def connect(self, endpoint: EndpointInfo, baudrate: int) -> None:
        if serial is None:
            raise RuntimeError("pyserial is not installed")
        self.disconnect()
        ser = serial.Serial(port=endpoint.address, baudrate=baudrate, timeout=0.05)
        with self._lock:
            self._serial = ser
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="MindFlexSerialTransport", daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        self._stop.set()
        with self._lock:
            ser, self._serial = self._serial, None
        if ser is not None:
            try:
                ser.close()
            except OSError:
                pass
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.6)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                ser = self._serial
            if ser is None:
                return
            try:
                waiting = int(getattr(ser, "in_waiting", 0) or 0)
                chunk = ser.read(max(1, min(4096, waiting)))
                if chunk:
                    self._on_bytes(chunk)
                else:
                    time.sleep(0.002)
            except Exception as exc:
                if not self._stop.is_set():
                    self._on_error(exc)
                self._stop.set()
                with self._lock:
                    failed, self._serial = self._serial, None
                if failed is not None:
                    try:
                        failed.close()
                    except OSError:
                        pass
                return


class TransportReader:
    """Serial/USB byte transport.

    Bluetooth deliberately does not pass through this class. The Bluetooth architecture uses one persistent ``BluetoothWorker`` for watcher + pairing +
    RFCOMM + ThinkGear reading, so keeping a second Bluetooth backend here would
    reintroduce the split-worker behaviour that this release removes.
    """

    def __init__(self, on_bytes: Callable[[bytes], None], on_error: Callable[[Exception], None] | None = None) -> None:
        self._on_bytes = on_bytes
        self._on_error = on_error or (lambda exc: None)
        self._backend: _SerialBackend | None = None
        self._endpoint: EndpointInfo | None = None
        self._lock = threading.RLock()

    @property
    def connected(self) -> bool:
        with self._lock:
            backend = self._backend
        return bool(backend and backend.connected)

    @property
    def endpoint(self) -> EndpointInfo | None:
        with self._lock:
            return self._endpoint

    def connect(self, endpoint: EndpointInfo, baudrate: int = MINDFLEX_BAUDRATE) -> None:
        if int(baudrate) != MINDFLEX_BAUDRATE:
            raise ValueError("MindFlex EEG Studio accepts only the fixed MindFlex RAW baud rate")
        if endpoint.kind is TransportKind.BLUETOOTH:
            raise RuntimeError("Bluetooth is handled by the persistent BluetoothWorker, not TransportReader")
        if endpoint.backend != "serial":
            raise RuntimeError("Serial/USB transport requires a serial endpoint")
        self.disconnect()
        backend = _SerialBackend(self._on_bytes, self._on_error)
        with self._lock:
            self._backend = backend
            self._endpoint = endpoint
        try:
            backend.connect(endpoint, MINDFLEX_BAUDRATE)
        except Exception:
            with self._lock:
                if self._backend is backend:
                    self._backend = None
                    self._endpoint = None
            backend.disconnect()
            raise

    def disconnect(self) -> None:
        with self._lock:
            backend, self._backend = self._backend, None
            self._endpoint = None
        if backend is not None:
            backend.disconnect()

