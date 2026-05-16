from __future__ import annotations

import numpy as np

from vadcore.base import SegmentBuilder
from vadcore.types import VadSegment

SUPPORTED_SAMPLE_RATES = (8000, 16000, 32000, 48000)
SUPPORTED_FRAME_MS = (10, 20, 30)


class WebRTCStreamingVad:
    """Streaming VAD wrapper around Google's WebRTC VAD (py-webrtcvad).

    WebRTC VAD is frame-by-frame: each call returns a bool. This wrapper
    groups consecutive speech frames into segments using SegmentBuilder.
    """

    name = "webrtc"

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        aggressiveness: int = 2,
        frame_ms: int = 30,
        min_silence_ms: int = 100,
        padding_ms: int = 30,
    ) -> None:
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"WebRTC VAD only supports sample rates {SUPPORTED_SAMPLE_RATES}, got {sample_rate}"
            )
        if frame_ms not in SUPPORTED_FRAME_MS:
            raise ValueError(
                f"WebRTC VAD frame_ms must be one of {SUPPORTED_FRAME_MS}, got {frame_ms}"
            )
        if not 0 <= aggressiveness <= 3:
            raise ValueError("WebRTC VAD aggressiveness must be in [0, 3]")

        import webrtcvad

        self.sample_rate = sample_rate
        self.aggressiveness = aggressiveness
        self.frame_ms = frame_ms
        self.min_silence_ms = min_silence_ms
        self.padding_ms = padding_ms
        self.chunk_samples = int(sample_rate * frame_ms / 1000)
        self._frame_s = frame_ms / 1000.0
        self._vad = webrtcvad.Vad(aggressiveness)
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

        pcm16 = np.clip(frame, -1.0, 1.0) * 32767.0
        pcm16_bytes = pcm16.astype(np.int16).tobytes()
        is_speech = self._vad.is_speech(pcm16_bytes, self.sample_rate)

        segment = self._builder.feed(is_speech=is_speech, frame_s=self._frame_s)
        return [segment] if segment is not None else []

    def flush(self) -> list[VadSegment]:
        segment = self._builder.flush()
        return [segment] if segment is not None else []
