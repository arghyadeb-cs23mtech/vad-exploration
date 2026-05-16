from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import VadSegment


@runtime_checkable
class StreamingVad(Protocol):
    """Common shape every streaming VAD backend exposes."""

    name: str
    sample_rate: int
    chunk_samples: int

    def reset(self) -> None: ...

    def process_chunk(self, chunk: np.ndarray) -> list[VadSegment]: ...

    def flush(self) -> list[VadSegment]: ...


class SegmentBuilder:
    """Turns a stream of (is_speech, frame_duration_s) decisions into
    [start, end] speech segments.

    Shared by frame-based backends (WebRTC, TenVAD). Silero emits start/end
    events directly so it doesn't use this.
    """

    def __init__(self, *, min_silence_ms: int, padding_ms: int) -> None:
        self._min_silence_s = min_silence_ms / 1000.0
        self._pad_s = padding_ms / 1000.0
        self.reset()

    def reset(self) -> None:
        self._offset_s = 0.0
        self._in_speech = False
        self._speech_start_s: float | None = None
        self._last_speech_end_s = 0.0
        self._silence_run_s = 0.0

    @property
    def offset_s(self) -> float:
        return self._offset_s

    def feed(self, *, is_speech: bool, frame_s: float) -> VadSegment | None:
        frame_start = self._offset_s
        frame_end = frame_start + frame_s
        self._offset_s = frame_end

        if is_speech:
            if not self._in_speech:
                self._speech_start_s = max(0.0, frame_start - self._pad_s)
                self._in_speech = True
            self._last_speech_end_s = frame_end
            self._silence_run_s = 0.0
            return None

        if not self._in_speech:
            return None

        self._silence_run_s += frame_s
        if self._silence_run_s < self._min_silence_s:
            return None

        # Close the segment; trailing padding is bounded by what we've already advanced past.
        end = min(frame_end, self._last_speech_end_s + self._pad_s)
        segment = VadSegment(
            start_s=float(self._speech_start_s or 0.0),
            end_s=float(end),
            confidence=None,
        )
        self._in_speech = False
        self._speech_start_s = None
        self._silence_run_s = 0.0
        return segment

    def flush(self) -> VadSegment | None:
        if not self._in_speech or self._speech_start_s is None:
            return None
        end = min(self._offset_s, self._last_speech_end_s + self._pad_s)
        segment = VadSegment(
            start_s=float(self._speech_start_s),
            end_s=float(end),
            confidence=None,
        )
        self._in_speech = False
        self._speech_start_s = None
        return segment
