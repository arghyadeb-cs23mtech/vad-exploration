from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VadSegment:
    start_s: float
    end_s: float
    confidence: float | None = None
