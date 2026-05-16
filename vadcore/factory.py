from __future__ import annotations

from typing import Any

from .base import StreamingVad

BACKENDS = ("silero", "webrtc", "tenvad")


def create_vad(
    name: str,
    *,
    sample_rate: int = 16000,
    threshold: float = 0.5,
    min_silence_ms: int = 100,
    padding_ms: int = 30,
    **extra: Any,
) -> StreamingVad:
    """Create a streaming VAD backend by name.

    Extra kwargs are forwarded to the backend (e.g. WebRTC `aggressiveness`,
    WebRTC `frame_ms`, TenVAD `hop_size`).
    """
    key = name.lower()
    if key == "silero":
        from silero.vad.silero_streaming import SileroStreamingVad
        return SileroStreamingVad(
            threshold=threshold,
            min_silence_ms=min_silence_ms,
            padding_ms=padding_ms,
            sample_rate=sample_rate,
        )
    if key == "webrtc":
        from webrtc_vad.streaming import WebRTCStreamingVad
        return WebRTCStreamingVad(
            sample_rate=sample_rate,
            min_silence_ms=min_silence_ms,
            padding_ms=padding_ms,
            **extra,
        )
    if key == "tenvad":
        from tenvad.streaming import TenVadStreamingVad
        return TenVadStreamingVad(
            sample_rate=sample_rate,
            threshold=threshold,
            min_silence_ms=min_silence_ms,
            padding_ms=padding_ms,
            **extra,
        )
    raise ValueError(f"Unknown VAD backend: {name!r}. Expected one of {BACKENDS}.")
