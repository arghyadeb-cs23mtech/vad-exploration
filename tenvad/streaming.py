from __future__ import annotations

import numpy as np

from vadcore.base import SegmentBuilder
from vadcore.types import VadSegment


class TenVadStreamingVad:
    """Streaming VAD wrapper around TEN VAD (ten-vad PyPI package).

    TEN VAD operates on 16 kHz int16 audio in fixed hop blocks (256 samples =
    16 ms by default). Each `process` call returns a (probability, flag) pair;
    we group flagged frames into segments via SegmentBuilder.
    """

    name = "tenvad"

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        hop_size: int = 256,
        threshold: float = 0.5,
        min_silence_ms: int = 100,
        padding_ms: int = 30,
    ) -> None:
        if sample_rate != 16000:
            raise ValueError(f"TEN VAD only supports 16 kHz audio, got {sample_rate}")
        if hop_size <= 0:
            raise ValueError("TEN VAD hop_size must be positive")

        from ten_vad import TenVad

        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.chunk_samples = hop_size
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        self.padding_ms = padding_ms
        self._frame_s = hop_size / float(sample_rate)
        self._vad = TenVad(hop_size=hop_size, threshold=threshold)
        self._builder = SegmentBuilder(
            min_silence_ms=min_silence_ms, padding_ms=padding_ms
        )

    def reset(self) -> None:
        self._builder.reset()

    def process_chunk(self, chunk: np.ndarray) -> list[VadSegment]:
        if len(chunk) == 0:
            return []

        frame = np.asarray(chunk, dtype=np.float32)
        if len(frame) != self.chunk_samples:
            padded = np.zeros(self.chunk_samples, dtype=np.float32)
            padded[: min(len(frame), self.chunk_samples)] = frame[: self.chunk_samples]
            frame = padded

        # TEN VAD requires a contiguous int16 array of exactly hop_size samples.
        pcm16 = np.ascontiguousarray(
            (np.clip(frame, -1.0, 1.0) * 32767.0).astype(np.int16)
        )
        _prob, flag = self._vad.process(pcm16)
        is_speech = bool(flag)

        segment = self._builder.feed(is_speech=is_speech, frame_s=self._frame_s)
        return [segment] if segment is not None else []

    def flush(self) -> list[VadSegment]:
        segment = self._builder.flush()
        return [segment] if segment is not None else []
