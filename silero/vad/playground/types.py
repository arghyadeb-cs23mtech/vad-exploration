from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class VadSegment:
    start_s: float
    end_s: float
    confidence: Optional[float] = None
