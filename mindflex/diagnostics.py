from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .controller import EEGController, MINDFLEX_BAUDRATE
from .parser import EEG_BANDS
from .rules import SignalPolicy


class DiagnosticStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


@dataclass(slots=True)
class DiagnosticResult:
    key: str
    status: DiagnosticStatus
    detail_key: str
    value: str = ""


@dataclass(frozen=True, slots=True)
class DiagnosticTest:
    key: str
    check: Callable[[EEGController, SignalPolicy], DiagnosticResult]


def _connection(controller: EEGController, policy: SignalPolicy) -> DiagnosticResult:
    del policy
    snap = controller.snapshot()
    ok = snap.connected
    return DiagnosticResult(
        "connection", DiagnosticStatus.PASSED if ok else DiagnosticStatus.FAILED,
        "diag.detail.connection.ok" if ok else "diag.detail.connection.fail", snap.source,
    )


def _baud(controller: EEGController, policy: SignalPolicy) -> DiagnosticResult:
    del policy
    snap = controller.snapshot()
    ok = snap.baudrate == MINDFLEX_BAUDRATE
    return DiagnosticResult(
        "baud", DiagnosticStatus.PASSED if ok else DiagnosticStatus.FAILED,
        "diag.detail.baud.ok" if ok else "diag.detail.baud.fail", f"{snap.baudrate} bps",
    )


def _stream(controller: EEGController, policy: SignalPolicy) -> DiagnosticResult:
    del policy
    snap = controller.snapshot()
    ok = snap.receiving and snap.packets > 0
    return DiagnosticResult(
        "stream", DiagnosticStatus.PASSED if ok else DiagnosticStatus.FAILED,
        "diag.detail.stream.ok" if ok else "diag.detail.stream.fail",
        f"{snap.packets} packets · {snap.bytes_received} bytes",
    )


def _checksum(controller: EEGController, policy: SignalPolicy) -> DiagnosticResult:
    del policy
    snap = controller.snapshot()
    total = snap.packets + snap.bad_checksums
    ratio = (snap.bad_checksums / total) if total else 1.0
    if not total:
        status = DiagnosticStatus.FAILED
    elif ratio <= 0.02:
        status = DiagnosticStatus.PASSED
    elif ratio <= 0.10:
        status = DiagnosticStatus.WARNING
    else:
        status = DiagnosticStatus.FAILED
    detail = {
        DiagnosticStatus.PASSED: "diag.detail.checksum.ok",
        DiagnosticStatus.WARNING: "diag.detail.checksum.warn",
        DiagnosticStatus.FAILED: "diag.detail.checksum.fail",
    }[status]
    return DiagnosticResult("checksum", status, detail, f"{ratio * 100:.2f}%")


def _raw(controller: EEGController, policy: SignalPolicy) -> DiagnosticResult:
    count, rate, spread = controller.raw_stats(2.0)
    enough = count >= policy.minimum_raw_samples
    rate_ok = policy.minimum_raw_rate_hz <= rate <= policy.maximum_raw_rate_hz
    active = spread >= policy.minimum_raw_spread
    ok = enough and rate_ok and active
    if not enough:
        detail = "diag.detail.raw.fail"
    elif not rate_ok:
        detail = "diag.detail.raw.rate"
    elif not active:
        detail = "diag.detail.raw.flat"
    else:
        detail = "diag.detail.raw.ok"
    snap = controller.snapshot()
    return DiagnosticResult(
        "raw", DiagnosticStatus.PASSED if ok else DiagnosticStatus.FAILED, detail,
        f"{count} samples/2s · {rate:.1f} Hz · σ={spread:.1f} · total={snap.raw_total_samples}",
    )


def _metric(controller: EEGController, policy: SignalPolicy, metric: str) -> DiagnosticResult:
    snap = controller.snapshot()
    value = getattr(snap, metric)
    age = getattr(snap, f"{metric}_age")
    source = getattr(snap, f"{metric}_source")
    valid = value is not None and 0 <= int(value) <= 100
    fresh = age <= policy.metric_stale_after_seconds
    ok = valid and fresh
    if ok:
        detail = f"diag.detail.{metric}.raw" if source == "raw" else f"diag.detail.{metric}.ok"
        status = DiagnosticStatus.PASSED
    elif snap.receiving and age == float("inf") and snap.stream_age < policy.metric_stale_after_seconds:
        detail = f"diag.detail.{metric}.wait"
        status = DiagnosticStatus.WARNING
    else:
        detail = f"diag.detail.{metric}.fail"
        status = DiagnosticStatus.FAILED
    age_text = "∞" if age == float("inf") else f"{age:.1f}s"
    value_text = "—" if value is None else str(value)
    source_text = source or "—"
    return DiagnosticResult(metric, status, detail, f"value={value_text} · source={source_text} · age={age_text}")


def _attention(controller: EEGController, policy: SignalPolicy) -> DiagnosticResult:
    return _metric(controller, policy, "attention")


def _meditation(controller: EEGController, policy: SignalPolicy) -> DiagnosticResult:
    return _metric(controller, policy, "meditation")


def _bands(controller: EEGController, policy: SignalPolicy) -> DiagnosticResult:
    snap = controller.snapshot()
    present = sum(1 for name in EEG_BANDS if name in snap.bands)
    fresh = snap.bands_age <= policy.metric_stale_after_seconds
    valid = present == len(EEG_BANDS) and all(
        isinstance(snap.bands.get(name), int) and snap.bands[name] >= 0 for name in EEG_BANDS
    )
    ok = valid and fresh
    if ok:
        detail = "diag.detail.bands.raw" if snap.bands_source == "raw" else "diag.detail.bands.ok"
        status = DiagnosticStatus.PASSED
    elif snap.receiving and snap.bands_age == float("inf") and snap.stream_age < policy.metric_stale_after_seconds:
        detail = "diag.detail.bands.wait"
        status = DiagnosticStatus.WARNING
    else:
        detail = "diag.detail.bands.fail"
        status = DiagnosticStatus.FAILED
    age = "∞" if snap.bands_age == float("inf") else f"{snap.bands_age:.1f}s"
    return DiagnosticResult(
        "bands", status, detail,
        f"{present}/8 fields · source={snap.bands_source or '—'} · age={age}",
    )


# POOR_SIGNAL is intentionally not a diagnostic test. A RAW stream must never
# be rejected because that optional summary byte is absent, stale or non-zero.
DEFAULT_TESTS = (
    DiagnosticTest("connection", _connection),
    DiagnosticTest("baud", _baud),
    DiagnosticTest("stream", _stream),
    DiagnosticTest("checksum", _checksum),
    DiagnosticTest("raw", _raw),
    DiagnosticTest("attention", _attention),
    DiagnosticTest("meditation", _meditation),
    DiagnosticTest("bands", _bands),
)


class DiagnosticSequence:
    """Ordered diagnostics where each subsystem is tested independently."""

    def __init__(self, controller: EEGController, policy: SignalPolicy, tests=DEFAULT_TESTS) -> None:
        self.controller = controller
        self.policy = policy
        self.tests = tuple(tests)
        self.results: list[DiagnosticResult] = [
            DiagnosticResult(test.key, DiagnosticStatus.PENDING, "") for test in self.tests
        ]
        self.index = 0

    @property
    def complete(self) -> bool:
        return self.index >= len(self.tests)

    def reset(self) -> None:
        self.results = [DiagnosticResult(test.key, DiagnosticStatus.PENDING, "") for test in self.tests]
        self.index = 0

    def run_next(self) -> DiagnosticResult | None:
        if self.complete:
            return None
        test = self.tests[self.index]
        result = test.check(self.controller, self.policy)
        self.results[self.index] = result
        self.index += 1
        return result

    def run_all(self) -> list[DiagnosticResult]:
        self.reset()
        executed: list[DiagnosticResult] = []
        while not self.complete:
            result = self.run_next()
            if result is not None:
                executed.append(result)
        return executed
