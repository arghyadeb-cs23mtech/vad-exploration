#!/usr/bin/env python3
"""Streaming VAD benchmark — batch-process a directory of WAV files and
report chunk-duration statistics for the selected VAD backend
(silero, webrtc, or tenvad).

Usage (direct):
    python main.py ./my_audio_dir --vad-model webrtc

Usage (module):
    python -m benchmark.main ./my_audio_dir --vad-model tenvad
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from stats import (compute_stats, print_summary, save_results_csv,
                   save_segments_csv)
from vad import process_directory

DEFAULT_OUTPUT = "benchmark_results.csv"
SUPPORTED_MODELS = ("silero", "webrtc", "tenvad")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="benchmark",
        description="Batch streaming VAD benchmark — compute speech-chunk statistics over a directory of WAV files.",
    )
    p.add_argument("input_dir", type=Path,
                   help="Directory containing WAV files (searched recursively).")
    p.add_argument("-o", "--output", type=Path, default=Path(DEFAULT_OUTPUT),
                   help=f"Path for the output CSV file (default: {DEFAULT_OUTPUT}).")
    p.add_argument("--vad-model", choices=SUPPORTED_MODELS, default="silero",
                   help="VAD backend to use (default: silero).")

    # Common VAD hyper-parameters (interpretation depends on backend).
    g = p.add_argument_group("Common VAD hyper-parameters")
    g.add_argument("--threshold", type=float, default=0.5,
                   help="Speech probability threshold (silero, tenvad). WebRTC ignores this. Default 0.5.")
    g.add_argument("--min-silence-ms", type=int, default=100,
                   help="Minimum silence duration in ms to close a segment (default 100).")
    g.add_argument("--padding-ms", type=int, default=30,
                   help="Speech padding in ms around detected segments (default 30).")
    g.add_argument("--sample-rate", type=int, default=16000, choices=[8000, 16000],
                   help="Target sample rate — files at other rates are resampled (default 16000).")

    # Backend-specific knobs.
    g2 = p.add_argument_group("WebRTC-specific")
    g2.add_argument("--webrtc-aggressiveness", type=int, default=2, choices=[0, 1, 2, 3],
                    help="WebRTC VAD aggressiveness (default 2).")
    g2.add_argument("--webrtc-frame-ms", type=int, default=30, choices=[10, 20, 30],
                    help="WebRTC VAD frame size in ms (default 30).")

    g3 = p.add_argument_group("TenVAD-specific")
    g3.add_argument("--tenvad-hop-size", type=int, default=256,
                    help="TEN VAD hop size in samples (default 256 = 16 ms @ 16 kHz).")

    p.add_argument("-v", "--verbose", action="store_true",
                   help="Enable DEBUG-level logging.")
    return p


def run(
    input_dir: Path,
    output: Path = Path(DEFAULT_OUTPUT),
    vad_model: str = "silero",
    threshold: float = 0.5,
    min_silence_ms: int = 100,
    padding_ms: int = 0,
    sample_rate: int = 16000,
    webrtc_aggressiveness: int = 2,
    webrtc_frame_ms: int = 30,
    tenvad_hop_size: int = 256,
    verbose: bool = False,
) -> Path:
    """Programmatic entry point — returns the path to the generated CSV."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    extra: dict = {}
    if vad_model == "webrtc":
        extra = {
            "aggressiveness": webrtc_aggressiveness,
            "frame_ms": webrtc_frame_ms,
        }
    elif vad_model == "tenvad":
        extra = {"hop_size": tenvad_hop_size}

    results = process_directory(
        input_dir,
        vad_model=vad_model,
        threshold=threshold,
        min_silence_ms=min_silence_ms,
        padding_ms=padding_ms,
        target_sr=sample_rate,
        **extra,
    )

    stats = compute_stats(results)
    print_summary(stats)
    output = Path(output)
    csv_path = save_results_csv(results, stats, output)
    segments_path = output.with_stem(output.stem + "_segments")
    save_segments_csv(results, segments_path)
    return csv_path


def main() -> None:
    args = build_parser().parse_args()
    try:
        csv_path = run(
            input_dir=args.input_dir,
            output=args.output,
            vad_model=args.vad_model,
            threshold=args.threshold,
            min_silence_ms=args.min_silence_ms,
            padding_ms=args.padding_ms,
            sample_rate=args.sample_rate,
            webrtc_aggressiveness=args.webrtc_aggressiveness,
            webrtc_frame_ms=args.webrtc_frame_ms,
            tenvad_hop_size=args.tenvad_hop_size,
            verbose=args.verbose,
        )
        print(f"Results saved to {csv_path}")
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        logging.getLogger(__name__).exception("Benchmark failed")
        sys.exit(2)


if __name__ == "__main__":
    main()
