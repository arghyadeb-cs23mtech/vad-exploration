from __future__ import annotations

from typing import Any, cast

import numpy as np

from .types import VadSegment


class SileroStreamingVad:
    def __init__(
            self,
            threshold: float = 0.5,
            min_silence_ms: int = 100,
            padding_ms: int = 30,
            sample_rate: int = 16000
        ):
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        self.padding_ms = padding_ms
        self._model: Any = None
        self._utils: tuple[Any, ...] | None = None
        self._iterator: Any = None
        self._sample_rate: int = sample_rate
        self._stream_offset_s: float = 0.0
        self._open_segment_start_s: float | None = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("Silero VAD requires 'torch'.") from exc

        loaded = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
            onnx=False,
        )
        self._model, self._utils = cast(tuple[Any, tuple[Any, ...]], loaded)

    def _ensure_iterator(self, sample_rate: int) -> None:
        if self._utils is None:
            raise RuntimeError("Silero utilities are not initialized")

        if self._iterator is not None and self._sample_rate == sample_rate:
            return

        vad_iterator_cls = self._utils[3]
        self._iterator = vad_iterator_cls(
            self._model,
            threshold=self.threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=self.padding_ms,
        )
        self._sample_rate = sample_rate

    def reset(self) -> None:
        if self._iterator is not None and hasattr(self._iterator, "reset_states"):
            self._iterator.reset_states()
        self._iterator = None
        self._stream_offset_s = 0.0
        self._open_segment_start_s = None

    def process_chunk(self, chunk: np.ndarray) -> list[VadSegment]:
        self._ensure_iterator(self._sample_rate)
        if self._iterator is None:
            return []

        chunk_f32 = np.asarray(chunk, dtype=np.float32)
        if len(chunk_f32) == 0:
            return []

        event = self._iterator(chunk_f32, return_seconds=True)
        self._stream_offset_s += len(chunk_f32) / float(self._sample_rate)
        if not event:
            return []

        segments: list[VadSegment] = []
        if "start" in event:
            self._open_segment_start_s = float(event["start"])
        if "end" in event and self._open_segment_start_s is not None:
            segments.append(
                VadSegment(
                    start_s=self._open_segment_start_s,
                    end_s=float(event["end"]),
                    confidence=None,
                )
            )
            self._open_segment_start_s = None
        return segments

    def flush(self) -> list[VadSegment]:
        if self._open_segment_start_s is None:
            return []

        final_segment = VadSegment(
            start_s=self._open_segment_start_s,
            end_s=self._stream_offset_s,
            confidence=None,
        )
        self._open_segment_start_s = None
        return [final_segment]
