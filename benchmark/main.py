#!/usr/bin/env python3
"""Silero VAD benchmark — batch-process a directory of WAV files and
report chunk-duration statistics.

Usage (direct):
    python -m benchmark.main ./my_audio_dir

Usage (CLI):
    python -m benchmark.main ./my_audio_dir --threshold 0.4 --output results.csv
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="benchmark",
        description="Batch Silero VAD benchmark — compute speech-chunk statistics over a directory of WAV files.",
    )
    p.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing WAV files (searched recursively).",
    )
    p.add_argument(
        "-o", "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Path for the output CSV file (default: {DEFAULT_OUTPUT}).",
    )

    # Silero hyper-parameters
    g = p.add_argument_group("Silero VAD hyper-parameters")
    g.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Speech probability threshold (default: 0.5).",
    )
    g.add_argument(
        "--min-silence-ms",
        type=int,
        default=100,
        help="Minimum silence duration in ms to split segments (default: 100).",
    )
    g.add_argument(
        "--padding-ms",
        type=int,
        default=30,
        help="Speech padding in ms added around detected segments (default: 30).",
    )
    g.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        choices=[8000, 16000],
        help="Target sample rate — files at other rates are resampled (default: 16000).",
    )

    # Logging
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p


def run(
    input_dir: Path,
    output: Path = Path(DEFAULT_OUTPUT),
    threshold: float = 0.5,
    min_silence_ms: int = 100,
    padding_ms: int = 0,
    sample_rate: int = 16000,
    verbose: bool = False,
) -> Path:
    """Programmatic entry point — call this from notebooks or other scripts.

    Returns the path to the generated CSV.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    results = process_directory(
        input_dir,
        threshold=threshold,
        min_silence_ms=min_silence_ms,
        padding_ms=padding_ms,
        target_sr=sample_rate,
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
            threshold=args.threshold,
            min_silence_ms=args.min_silence_ms,
            padding_ms=args.padding_ms,
            sample_rate=args.sample_rate,
            verbose=args.verbose,
        )
        print(f"Results saved to {csv_path}")
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        logging.getLogger(__name__).exception("Benchmark failed")
        sys.exit(2)


if __name__ == "__main__":
    # THRESHOLD = 0.5
    # MIN_SILENCE_MS = 120
    # PADDING_MS = 0
    # SAMPLE_RATE = 16000
    # run(
    #     input_dir=Path("../recordings"),
    #     output=Path(f"{DEFAULT_OUTPUT}_thr{THRESHOLD}_sil{MIN_SILENCE_MS}ms_pad{PADDING_MS}.csv"),
    #     threshold=THRESHOLD,
    #     min_silence_ms=MIN_SILENCE_MS,
    #     padding_ms=PADDING_MS,
    #     sample_rate=SAMPLE_RATE,
    #     verbose=False,
    # )

    main()
