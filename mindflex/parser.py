from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
import math
import struct

SYNC = 0xAA
EXCODE = 0x55

EEG_BANDS = (
    "delta",
    "theta",
    "low_alpha",
    "high_alpha",
    "low_beta",
    "high_beta",
    "low_gamma",
    "mid_gamma",
)


@dataclass(slots=True)
class ThinkGearData:
    poor_signal: int | None = None
    attention: int | None = None
    meditation: int | None = None
    blink: int | None = None
    raw: list[int] = field(default_factory=list)
    bands: dict[str, int] = field(default_factory=dict)


class ThinkGearParser:
    """Incremental parser for Neurosky/ThinkGear packets used by MindFlex."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.bad_checksums = 0
        self.packets = 0
        self.sync_count = 0

    def reset(self) -> None:
        self._buffer.clear()
        self.bad_checksums = 0
        self.packets = 0
        self.sync_count = 0

    def discard_partial(self) -> None:
        """Drop only an incomplete framing fragment while preserving diagnostics."""
        self._buffer.clear()

    def feed(self, data: bytes | bytearray | memoryview) -> list[ThinkGearData]:
        if data:
            self._buffer.extend(data)
        out: list[ThinkGearData] = []
        while True:
            start = self._find_sync()
            if start < 0:
                if len(self._buffer) > 1:
                    del self._buffer[:-1]
                break
            if start:
                del self._buffer[:start]
            if len(self._buffer) < 3:
                break
            length = self._buffer[2]
            if length > 169:
                del self._buffer[0]
                continue
            total = 3 + length + 1
            if len(self._buffer) < total:
                break
            self.sync_count += 1
            payload = bytes(self._buffer[3 : 3 + length])
            checksum = self._buffer[3 + length]
            del self._buffer[:total]
            expected = (~(sum(payload) & 0xFF)) & 0xFF
            if checksum != expected:
                self.bad_checksums += 1
                continue
            self.packets += 1
            parsed = self._parse_payload(payload)
            if parsed is not None:
                out.append(parsed)
        return out

    def _find_sync(self) -> int:
        b = self._buffer
        for i in range(max(0, len(b) - 1)):
            if b[i] == SYNC and b[i + 1] == SYNC:
                return i
        return -1

    @staticmethod
    def _parse_payload(payload: bytes) -> ThinkGearData:
        result = ThinkGearData()
        i = 0
        n = len(payload)
        while i < n:
            excode_level = 0
            while i < n and payload[i] == EXCODE:
                excode_level += 1
                i += 1
            if i >= n:
                break
            code = payload[i]
            i += 1
            if code < 0x80:
                if i >= n:
                    break
                value = payload[i]
                i += 1
                if excode_level != 0:
                    continue
                if code == 0x02:
                    result.poor_signal = value
                elif code == 0x04:
                    result.attention = value
                elif code == 0x05:
                    result.meditation = value
                elif code == 0x16:
                    result.blink = value
                continue
            if i >= n:
                break
            value_len = payload[i]
            i += 1
            if i + value_len > n:
                break
            value = payload[i : i + value_len]
            i += value_len
            if excode_level != 0:
                continue
            if code == 0x80 and value_len == 2:
                raw = int.from_bytes(value, byteorder="big", signed=True)
                result.raw.append(raw)
            elif code == 0x81 and value_len >= 32:
                # EEG_POWER: eight big-endian IEEE-754 floats. Some ThinkGear
                # modules use this representation instead of ASIC_EEG_POWER.
                values = struct.unpack(">8f", value[:32])
                # A valid checksum does not guarantee semantically valid float
                # payloads.  Reject NaN/Inf instead of letting int(NaN/Inf)
                # raise inside the acquisition thread.
                if all(math.isfinite(v) for v in values):
                    result.bands = {name: max(0, int(v)) for name, v in zip(EEG_BANDS, values)}
            elif code == 0x83 and value_len >= 24:
                # ASIC_EEG_POWER used by TGAT/TGAM1: 8 x unsigned 24-bit values.
                result.bands = {
                    name: int.from_bytes(value[j : j + 3], "big")
                    for name, j in zip(EEG_BANDS, range(0, 24, 3))
                }
        return result


def make_packet(payload: Iterable[int]) -> bytes:
    """Build a valid ThinkGear packet for simulation and development fixtures."""
    payload_bytes = bytes(payload)
    if len(payload_bytes) > 169:
        raise ValueError("ThinkGear payload cannot exceed 169 bytes")
    checksum = (~(sum(payload_bytes) & 0xFF)) & 0xFF
    return bytes((SYNC, SYNC, len(payload_bytes))) + payload_bytes + bytes((checksum,))
