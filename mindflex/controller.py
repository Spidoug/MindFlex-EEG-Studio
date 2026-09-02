from __future__ import annotations

import math
import random
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable

from .cognitive import CognitiveEstimator
from .parser import EEG_BANDS, ThinkGearData, ThinkGearParser, make_packet
from .transport import EndpointInfo, TransportKind, TransportReader

MINDFLEX_BAUDRATE = 57600
MINDFLEX_RAW_SAMPLE_RATE = 512
NATIVE_METRIC_FRESH_SECONDS = 4.0


@dataclass(slots=True)
class EEGSnapshot:
    timestamp: float = 0.0
    connected: bool = False
    receiving: bool = False
    source: str = ""
    transport: str = ""
    endpoint_id: str = ""
    baudrate: int = MINDFLEX_BAUDRATE

    # Effective values consumed by every subsystem.  Source says whether the
    # value is native TGAM or a RAW-derived continuity estimate.
    poor_signal: int | None = None
    attention: int | None = None
    meditation: int | None = None
    attention_source: str = ""
    meditation_source: str = ""
    blink: int | None = None
    bands: dict[str, int] = field(default_factory=dict)
    bands_source: str = ""

    # Native values are exposed only for diagnostics/inspection; they never
    # create a second workflow path.
    native_attention: int | None = None
    native_meditation: int | None = None
    native_bands: dict[str, int] = field(default_factory=dict)
    derived_attention: int | None = None
    derived_meditation: int | None = None

    raw_samples: int = 0
    raw_rate_hz: float = 0.0
    packets: int = 0
    bad_checksums: int = 0
    bytes_received: int = 0
    signal_age: float = math.inf
    esense_age: float = math.inf
    attention_age: float = math.inf
    meditation_age: float = math.inf
    bands_age: float = math.inf
    raw_age: float = math.inf
    stream_age: float = 0.0
    signal_updates: int = 0
    attention_updates: int = 0
    meditation_updates: int = 0
    bands_updates: int = 0
    raw_total_samples: int = 0
    error: str = ""


class EEGController:
    """Single-source EEG state machine for every transport.

    General rules:
    * transports only deliver bytes;
    * exactly one persistent ThinkGear parser owns protocol interpretation;
    * native summary rows and RAW-derived continuity metrics are merged here;
    * UI, diagnostics, recorder and BCI only read EEGSnapshot;
    * no transport-specific signal/eSense rules exist downstream.
    """

    def __init__(self, raw_capacity: int = 32768, raw_sample_rate: int = MINDFLEX_RAW_SAMPLE_RATE) -> None:
        del raw_sample_rate
        self.parser = ThinkGearParser()
        self.cognitive = CognitiveEstimator()
        self.raw_sample_rate = MINDFLEX_RAW_SAMPLE_RATE
        self._lock = threading.RLock()
        self._snapshot = EEGSnapshot()
        self._raw: deque[int] = deque(maxlen=raw_capacity)
        self._raw_arrivals: deque[tuple[float, int]] = deque(maxlen=8192)
        self._metric_history: dict[str, deque[tuple[float, float]]] = {
            "attention": deque(maxlen=4096),
            "meditation": deque(maxlen=4096),
        }
        self._listeners: list[Callable[[EEGSnapshot], None]] = []

        self._last_packet_at = 0.0
        self._first_packet_at = 0.0
        self._last_signal_at = 0.0
        self._last_raw_at = 0.0
        self._last_attention_effective_at = 0.0
        self._last_meditation_effective_at = 0.0
        self._last_bands_effective_at = 0.0
        self._last_native_attention_at = 0.0
        self._last_native_meditation_at = 0.0
        self._last_native_bands_at = 0.0
        self._native_attention: int | None = None
        self._native_meditation: int | None = None
        self._native_bands: dict[str, int] = {}

        # All byte transports share one bounded/coalescing ingestion path.  The
        # transport thread never waits for FFT, parser listeners or UI work.
        self._ingest_cv = threading.Condition()
        self._ingest_buffer = bytearray()
        self._ingest_closed = False
        self._ingest_thread = threading.Thread(target=self._ingest_loop, name="MindFlexEEGIngest", daemon=True)
        self._ingest_thread.start()

        self.reader = TransportReader(self.submit_bytes, self._on_error)
        self._sim_stop = threading.Event()
        self._sim_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    def snapshot(self) -> EEGSnapshot:
        now = time.monotonic()
        with self._lock:
            self._refresh_effective_locked(now, record_history=False)
            snap = replace(self._snapshot)
            snap.bands = dict(self._snapshot.bands)
            snap.native_bands = dict(self._snapshot.native_bands)
            if snap.connected:
                snap.receiving = self._last_packet_at > 0 and (now - self._last_packet_at) < 2.0
            snap.stream_age = max(0.0, now - self._first_packet_at) if self._first_packet_at else 0.0
            snap.signal_age = (now - self._last_signal_at) if self._last_signal_at else math.inf
            snap.attention_age = (now - self._last_attention_effective_at) if self._last_attention_effective_at else math.inf
            snap.meditation_age = (now - self._last_meditation_effective_at) if self._last_meditation_effective_at else math.inf
            snap.esense_age = max(snap.attention_age, snap.meditation_age)
            snap.bands_age = (now - self._last_bands_effective_at) if self._last_bands_effective_at else math.inf
            snap.raw_age = max(0.0, now - self._last_raw_at) if self._last_raw_at else math.inf
            count, rate, _spread = self._raw_stats_locked(2.0, now)
            snap.raw_samples = count
            snap.raw_rate_hz = rate
            return snap

    def add_listener(self, callback: Callable[[EEGSnapshot], None]) -> None:
        with self._lock:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[EEGSnapshot], None]) -> None:
        with self._lock:
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Transport lifecycle
    # ------------------------------------------------------------------

    def connect(self, endpoint: EndpointInfo, baudrate: int = MINDFLEX_BAUDRATE) -> None:
        if int(baudrate) != MINDFLEX_BAUDRATE:
            raise ValueError("MindFlex RAW mode is fixed at 57600 bps")
        if endpoint.kind is TransportKind.BLUETOOTH:
            raise RuntimeError("Bluetooth is connected by BluetoothWorker and feeds the same controller byte pipeline")
        self.stop_simulation()
        self.reader.disconnect()
        self._reset_stream_state(source=endpoint.address, transport=endpoint.kind.value, endpoint_id=endpoint.id, connected=False)
        self.reader.connect(endpoint, MINDFLEX_BAUDRATE)
        with self._lock:
            self._snapshot.connected = True
            self._snapshot.baudrate = MINDFLEX_BAUDRATE
            self._snapshot.error = ""
        self._notify()

    def disconnect(self) -> None:
        self.reader.disconnect()
        self.stop_simulation()
        with self._lock:
            self._snapshot.connected = False
            self._snapshot.receiving = False
            self._last_packet_at = 0.0
            self._first_packet_at = 0.0
        self._notify()

    def prepare_bluetooth_connection(self, endpoint: EndpointInfo) -> None:
        self.stop_simulation()
        self.reader.disconnect()
        self._reset_stream_state(
            source=endpoint.address,
            transport=TransportKind.BLUETOOTH.value,
            endpoint_id=endpoint.id,
            connected=False,
        )
        self._notify()

    def confirm_bluetooth_connection(self, endpoint: EndpointInfo) -> None:
        with self._lock:
            self._snapshot.source = endpoint.address
            self._snapshot.transport = TransportKind.BLUETOOTH.value
            self._snapshot.endpoint_id = endpoint.id
            self._snapshot.connected = True
            self._snapshot.baudrate = MINDFLEX_BAUDRATE
            self._snapshot.error = ""
        self._notify()

    def feed_bluetooth_bytes(self, chunk: bytes) -> None:
        self.submit_bytes(chunk)

    def bluetooth_disconnected(self) -> None:
        with self._lock:
            if self._snapshot.transport != TransportKind.BLUETOOTH.value:
                return
            self._snapshot.connected = False
            self._snapshot.receiving = False
            self._last_packet_at = 0.0
            self._first_packet_at = 0.0
        self._notify()

    def bluetooth_error(self, exc: Exception) -> None:
        with self._lock:
            self._snapshot.error = str(exc)
            if self._snapshot.transport == TransportKind.BLUETOOTH.value:
                self._snapshot.connected = False
                self._snapshot.receiving = False
                self._last_packet_at = 0.0
        self._notify()

    # ------------------------------------------------------------------
    # Unified byte ingestion
    # ------------------------------------------------------------------

    def submit_bytes(self, chunk: bytes | bytearray | memoryview) -> None:
        if not chunk:
            return
        payload = bytes(chunk)
        with self._ingest_cv:
            if self._ingest_closed:
                return
            self._ingest_buffer.extend(payload)
            self._ingest_cv.notify()

    def _ingest_loop(self) -> None:
        while True:
            with self._ingest_cv:
                while not self._ingest_buffer and not self._ingest_closed:
                    self._ingest_cv.wait()
                if self._ingest_closed and not self._ingest_buffer:
                    return
                # A few milliseconds of coalescing is enough to absorb WinRT
                # PARTIAL byte fragments without adding meaningful latency.
                if len(self._ingest_buffer) < 4096 and not self._ingest_closed:
                    self._ingest_cv.wait(timeout=0.010)
                batch = bytes(self._ingest_buffer)
                self._ingest_buffer.clear()
            if batch:
                self._on_bytes(batch)

    def _on_bytes(self, chunk: bytes) -> None:
        """Synchronous parser entry used by the ingest thread and deterministic tests."""
        now = time.monotonic()
        packets = self.parser.feed(chunk)
        with self._lock:
            self._snapshot.bytes_received += len(chunk)
            self._snapshot.packets = self.parser.packets
            self._snapshot.bad_checksums = self.parser.bad_checksums
            if packets:
                if not self._first_packet_at:
                    self._first_packet_at = now
                raw_before = self._snapshot.raw_total_samples
                for data in packets:
                    self._apply_locked(data, now)
                if self._snapshot.raw_total_samples > raw_before:
                    self._raw_arrivals.append((now, self._snapshot.raw_total_samples))
                self._refresh_effective_locked(now, record_history=True)
                self._snapshot.timestamp = time.time()
                self._snapshot.receiving = True
                self._snapshot.error = ""
                self._last_packet_at = now
        if packets:
            self._notify()

    def _apply_locked(self, data: ThinkGearData, now: float) -> None:
        snap = self._snapshot
        if data.poor_signal is not None:
            snap.poor_signal = max(0, min(200, int(data.poor_signal)))
            self._last_signal_at = now
            snap.signal_updates += 1

        if data.attention is not None:
            self._native_attention = max(0, min(100, int(data.attention)))
            self._last_native_attention_at = now
            snap.native_attention = self._native_attention

        if data.meditation is not None:
            self._native_meditation = max(0, min(100, int(data.meditation)))
            self._last_native_meditation_at = now
            snap.native_meditation = self._native_meditation

        if data.blink is not None:
            snap.blink = max(0, min(255, int(data.blink)))

        if data.bands:
            self._native_bands = {name: max(0, int(data.bands.get(name, 0))) for name in EEG_BANDS if name in data.bands}
            self._last_native_bands_at = now
            snap.native_bands = dict(self._native_bands)

        if data.raw:
            # RAW is a fixed-rate 512 Hz sequence. Host arrival timestamps are
            # kept separately for liveness/rate diagnostics; sample time for
            # plotting and epochs comes from sequence position, so Bluetooth
            # buffering can never create overlapping or future timestamps.
            self._raw.extend(int(raw) for raw in data.raw)
            self._last_raw_at = now
            snap.raw_total_samples += len(data.raw)
            self.cognitive.feed(data.raw, now)

    @staticmethod
    def _fresh(last_at: float, now: float, seconds: float) -> bool:
        return last_at > 0 and (now - last_at) <= seconds

    def _refresh_effective_locked(self, now: float, *, record_history: bool) -> None:
        snap = self._snapshot
        derived = self.cognitive.state
        derived_fresh = derived.updated_at > 0 and (now - derived.updated_at) <= self.cognitive.FRESH_SECONDS
        snap.derived_attention = None if derived.attention is None else int(round(derived.attention))
        snap.derived_meditation = None if derived.meditation is None else int(round(derived.meditation))

        def choose_metric(native: int | None, native_at: float, derived_value: float | None) -> tuple[int | None, str, float]:
            native_fresh = native is not None and self._fresh(native_at, now, NATIVE_METRIC_FRESH_SECONDS)
            if native_fresh and native is not None and native > 0:
                return int(native), "thinkgear", native_at
            if derived_fresh and derived_value is not None and math.isfinite(derived_value):
                return max(0, min(100, int(round(derived_value)))), "raw", derived.updated_at
            if native_fresh and native is not None:
                return int(native), "thinkgear", native_at
            return None, "", 0.0

        old_attention_at = self._last_attention_effective_at
        old_meditation_at = self._last_meditation_effective_at
        old_bands_at = self._last_bands_effective_at

        attention, attention_source, attention_at = choose_metric(
            self._native_attention, self._last_native_attention_at, derived.attention
        )
        meditation, meditation_source, meditation_at = choose_metric(
            self._native_meditation, self._last_native_meditation_at, derived.meditation
        )

        native_bands_fresh = (
            len(self._native_bands) == len(EEG_BANDS)
            and self._fresh(self._last_native_bands_at, now, NATIVE_METRIC_FRESH_SECONDS)
        )
        if native_bands_fresh:
            bands = dict(self._native_bands)
            bands_source = "thinkgear"
            bands_at = self._last_native_bands_at
        elif derived_fresh and len(derived.bands) == len(EEG_BANDS):
            bands = dict(derived.bands)
            bands_source = "raw"
            bands_at = derived.updated_at
        else:
            bands = dict(self._native_bands) if self._native_bands else {}
            bands_source = "thinkgear" if bands else ""
            bands_at = self._last_native_bands_at if bands else 0.0

        snap.attention = attention
        snap.attention_source = attention_source
        snap.meditation = meditation
        snap.meditation_source = meditation_source
        snap.bands = bands
        snap.bands_source = bands_source
        self._last_attention_effective_at = attention_at
        self._last_meditation_effective_at = meditation_at
        self._last_bands_effective_at = bands_at

        # History follows measurement events, not numerical changes.  A stable
        # value is still a new sample and must advance the graph/timestamps.
        if record_history and attention is not None and attention_at > old_attention_at + 1e-9:
            snap.attention_updates += 1
            self._metric_history["attention"].append((attention_at, float(attention)))
        if record_history and meditation is not None and meditation_at > old_meditation_at + 1e-9:
            snap.meditation_updates += 1
            self._metric_history["meditation"].append((meditation_at, float(meditation)))
        if record_history and bands and bands_at > old_bands_at + 1e-9:
            snap.bands_updates += 1

    # ------------------------------------------------------------------
    # Windows/statistics
    # ------------------------------------------------------------------

    def raw_values(self, seconds: float, max_samples: int | None = None) -> list[int]:
        """Return the trailing RAW epoch in acquisition order.

        The MindFlex RAW stream is fixed at 512 Hz, so epoch duration is defined
        by sample count rather than host packet-arrival timing.
        """
        wanted = max(1, int(round(max(0.2, float(seconds)) * MINDFLEX_RAW_SAMPLE_RATE)))
        if max_samples is not None:
            wanted = min(wanted, max(1, int(max_samples)))
        with self._lock:
            data = list(self._raw)
        return data[-wanted:]

    def raw_window(self, seconds: float, max_points: int = 2400) -> tuple[list[float], list[int]]:
        wanted = max(1, int(round(max(0.2, float(seconds)) * MINDFLEX_RAW_SAMPLE_RATE)))
        with self._lock:
            data = list(self._raw)[-wanted:]
        if not data:
            return [], []
        indices = list(range(len(data)))
        if len(data) > max_points:
            stride = max(1, math.ceil(len(data) / max_points))
            indices = list(range(0, len(data), stride))
            if indices[-1] != len(data) - 1:
                indices.append(len(data) - 1)
        end_index = len(data) - 1
        x = [(index - end_index) / float(MINDFLEX_RAW_SAMPLE_RATE) for index in indices]
        y = [data[index] for index in indices]
        return x, y

    def raw_stats(self, seconds: float = 2.0) -> tuple[int, float, float]:
        with self._lock:
            return self._raw_stats_locked(seconds, time.monotonic())

    def _raw_stats_locked(self, seconds: float, now: float) -> tuple[int, float, float]:
        duration_requested = max(0.2, float(seconds))
        cutoff = now - duration_requested
        arrivals = list(self._raw_arrivals)
        window: list[tuple[float, int]] = []
        previous: tuple[float, int] | None = None
        for item in arrivals:
            if item[0] < cutoff:
                previous = item
                continue
            window.append(item)
        if previous is not None:
            window.insert(0, previous)

        count = 0
        rate = 0.0
        if len(window) >= 2:
            t0, total0 = window[0]
            t1, total1 = window[-1]
            count = max(0, int(total1 - total0))
            elapsed = t1 - t0
            if elapsed > 0:
                rate = count / elapsed
        elif window:
            # A single arrival cannot establish a host-time rate, but the RAW
            # amount is still useful for readiness/flat-signal diagnostics.
            count = min(len(self._raw), int(round(duration_requested * MINDFLEX_RAW_SAMPLE_RATE)))

        spread_count = min(len(self._raw), max(2, int(round(duration_requested * MINDFLEX_RAW_SAMPLE_RATE))))
        recent = list(self._raw)[-spread_count:] if spread_count else []
        spread = statistics.pstdev(recent) if len(recent) > 1 else 0.0
        return count, rate, spread

    def metric_window(self, metric: str, seconds: float, max_points: int = 1200) -> tuple[list[float], list[float]]:
        cutoff = time.monotonic() - max(1.0, seconds)
        with self._lock:
            data = [item for item in self._metric_history.get(metric, ()) if item[0] >= cutoff]
        if len(data) > max_points:
            stride = max(1, math.ceil(len(data) / max_points))
            data = data[::stride]
        if not data:
            return [], []
        t0 = data[-1][0]
        return [t - t0 for t, _ in data], [value for _, value in data]

    # ------------------------------------------------------------------
    # Reset/error/notify
    # ------------------------------------------------------------------

    def _reset_stream_state(self, *, source: str, transport: str, endpoint_id: str, connected: bool) -> None:
        with self._lock:
            self.parser.reset()
            self.cognitive.reset()
            self._snapshot = EEGSnapshot(
                connected=connected,
                source=source,
                transport=transport,
                endpoint_id=endpoint_id,
                baudrate=MINDFLEX_BAUDRATE,
            )
            self._last_packet_at = 0.0
            self._first_packet_at = 0.0
            self._last_signal_at = 0.0
            self._last_raw_at = 0.0
            self._last_attention_effective_at = 0.0
            self._last_meditation_effective_at = 0.0
            self._last_bands_effective_at = 0.0
            self._last_native_attention_at = 0.0
            self._last_native_meditation_at = 0.0
            self._last_native_bands_at = 0.0
            self._native_attention = None
            self._native_meditation = None
            self._native_bands = {}
            self._raw.clear()
            self._raw_arrivals.clear()
            for series in self._metric_history.values():
                series.clear()
        with self._ingest_cv:
            self._ingest_buffer.clear()

    def _on_error(self, exc: Exception) -> None:
        with self._lock:
            self._snapshot.error = str(exc)
            self._snapshot.connected = False
            self._snapshot.receiving = False
            self._last_packet_at = 0.0
        self._notify()

    def _notify(self) -> None:
        with self._lock:
            listeners = tuple(self._listeners)
        if not listeners:
            return
        snap = self.snapshot()
        for callback in listeners:
            try:
                callback(snap)
            except Exception:
                continue

    # ------------------------------------------------------------------
    # Simulator
    # ------------------------------------------------------------------

    def start_simulation(self, sample_rate: int = MINDFLEX_RAW_SAMPLE_RATE) -> None:
        del sample_rate
        self.reader.disconnect()
        self.stop_simulation()
        self._sim_stop.clear()
        self._reset_stream_state(source="SIMULATOR", transport="simulator", endpoint_id="simulator", connected=True)
        self._sim_thread = threading.Thread(target=self._simulate, name="MindFlexSimulator", daemon=True)
        self._sim_thread.start()

    def stop_simulation(self) -> None:
        self._sim_stop.set()
        thread = self._sim_thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.5)
        self._sim_thread = None

    def _simulate(self) -> None:
        rng = random.Random(7341)
        phase = 0.0
        raw_per_packet = 16
        period = raw_per_packet / float(MINDFLEX_RAW_SAMPLE_RATE)
        next_at = time.perf_counter()
        while not self._sim_stop.is_set():
            payload = bytearray()
            for _ in range(raw_per_packet):
                phase += 2.0 * math.pi * (9.5 + 0.5 * math.sin(phase / 80.0)) / MINDFLEX_RAW_SAMPLE_RATE
                raw = int(480 * math.sin(phase) + 120 * math.sin(phase * 0.27) + rng.gauss(0, 55))
                payload += bytes((0x80, 0x02)) + int(raw).to_bytes(2, "big", signed=True)
            tick = int(time.monotonic() * 2)
            attention = int(55 + 25 * math.sin(tick / 9.0))
            meditation = int(52 + 22 * math.sin(tick / 13.0 + 1.2))
            payload += bytes((0x02, 0x00, 0x04, max(0, min(100, attention)), 0x05, max(0, min(100, meditation))))
            bands = [max(1, int(90000 * (1 + 0.7 * math.sin(phase * (i + 1) * 0.01)))) for i in range(8)]
            band_payload = b"".join(int(v).to_bytes(3, "big") for v in bands)
            payload += bytes((0x83, len(band_payload))) + band_payload
            self.submit_bytes(make_packet(payload))
            next_at += period
            delay = next_at - time.perf_counter()
            if delay > 0:
                self._sim_stop.wait(delay)
            else:
                next_at = time.perf_counter()

    def close(self) -> None:
        self.disconnect()
        with self._ingest_cv:
            self._ingest_closed = True
            self._ingest_cv.notify_all()
        if self._ingest_thread.is_alive() and self._ingest_thread is not threading.current_thread():
            self._ingest_thread.join(timeout=1.0)
