"""Standalone TEN VAD streaming demo.

Usage:
    python -m tenvad.main path/to/audio.wav
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from silero.vad.audio import load_wav_mono_float32, resample_mono_float32
from tenvad.streaming import TenVadStreamingVad


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone TEN VAD streaming demo")
    p.add_argument("audio", type=Path, help="Path to WAV input")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="TEN VAD speech probability threshold (default 0.5)")
    p.add_argument("--hop-size", type=int, default=256,
                   help="TEN VAD hop size in samples (default 256 = 16 ms @ 16 kHz)")
    p.add_argument("--min-silence-ms", type=int, default=100,
                   help="Minimum silence in ms to close a segment (default 100)")
    p.add_argument("--padding-ms", type=int, default=30,
                   help="Padding in ms around each segment (default 30)")
    return p


def main() -> None:
    args = build_parser().parse_args()

    audio, sr = load_wav_mono_float32(args.audio)
    if sr != 16000:
        print(f"[tenvad] resampling {sr} Hz -> 16000 Hz")
        audio = resample_mono_float32(audio, sr, 16000)
        sr = 16000

    vad = TenVadStreamingVad(
        sample_rate=sr,
        hop_size=args.hop_size,
        threshold=args.threshold,
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

    print(f"[tenvad] detected {len(segments)} speech segments")
    for seg in segments:
        print(f"{seg.start_s:.3f} {seg.end_s:.3f}")


if __name__ == "__main__":
    main()
