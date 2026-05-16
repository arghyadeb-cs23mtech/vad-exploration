"""Standalone WebRTC streaming VAD demo.

Usage:
    python -m webrtc_vad.main path/to/audio.wav
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from silero.vad.audio import load_wav_mono_float32, resample_mono_float32
from webrtc_vad.streaming import WebRTCStreamingVad

SUPPORTED_SAMPLE_RATES = (8000, 16000, 32000, 48000)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone WebRTC streaming VAD demo")
    p.add_argument("audio", type=Path, help="Path to WAV input")
    p.add_argument("--aggressiveness", type=int, default=2, choices=[0, 1, 2, 3],
                   help="WebRTC VAD aggressiveness (0..3, default 2)")
    p.add_argument("--frame-ms", type=int, default=30, choices=[10, 20, 30],
                   help="WebRTC VAD frame size in ms (default 30)")
    p.add_argument("--min-silence-ms", type=int, default=100,
                   help="Minimum silence in ms to close a segment (default 100)")
    p.add_argument("--padding-ms", type=int, default=30,
                   help="Padding in ms around each segment (default 30)")
    p.add_argument("--sample-rate", type=int, default=16000,
                   choices=list(SUPPORTED_SAMPLE_RATES),
                   help="Target sample rate (default 16000)")
    return p


def main() -> None:
    args = build_parser().parse_args()

    audio, sr = load_wav_mono_float32(args.audio)
    if sr != args.sample_rate:
        print(f"[webrtc] resampling {sr} Hz -> {args.sample_rate} Hz")
        audio = resample_mono_float32(audio, sr, args.sample_rate)
        sr = args.sample_rate

    vad = WebRTCStreamingVad(
        sample_rate=sr,
        aggressiveness=args.aggressiveness,
        frame_ms=args.frame_ms,
        min_silence_ms=args.min_silence_ms,
        padding_ms=args.padding_ms,
    )

    segments = []
    for start in range(0, len(audio), vad.chunk_samples):
        chunk = audio[start : start + vad.chunk_samples]
        if len(chunk) < vad.chunk_samples:
            padded = np.zeros(vad.chunk_samples, dtype=np.float32)
            padded[: len(chunk)] = chunk
            chunk = padded
        segments.extend(vad.process_chunk(chunk))
    segments.extend(vad.flush())

    print(f"[webrtc] detected {len(segments)} speech segments")
    for seg in segments:
        print(f"{seg.start_s:.3f} {seg.end_s:.3f}")


if __name__ == "__main__":
    main()
