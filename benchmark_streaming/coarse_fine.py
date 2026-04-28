#!/usr/bin/env python3
"""Coarse-Fine VAD benchmark — batch-process a directory of WAV files using two VAD layers.

The coarse layer processes the entire audio to identify initial segments.
Segments longer than the threshold are further processed by the fine layer for more precise segmentation.

Usage (direct):
    python coarse_fine.py ./my_audio_dir

Usage (CLI):
    python coarse_fine.py ./my_audio_dir --coarse-threshold 0.4 --output results.csv --fine-threshold 0.3
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from stats import (compute_stats, print_summary, save_results_csv,
                   save_segments_csv)
from audio import TARGET_SAMPLE_RATE, discover_wav_files, load_and_prepare

DEFAULT_OUTPUT = "coarse_fine_results.csv"
DEFAULT_CHUNK_MS = 32
DEFAULT_SEGMENT_LENGTH_S = 30  # 30 seconds default

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Silero model loading (done once, shared across files)
# ---------------------------------------------------------------------------

def _load_silero_model():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Silero VAD requires 'torch'. Install it first.") from exc

    loaded = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
        onnx=False,
    )
    model, utils = loaded
    return model, utils


def _create_iterator(
    model,
    utils,
    *,
    threshold: float,
    min_silence_ms: int,
    padding_ms: int,
    sample_rate: int,
):
    vad_iterator_cls = utils[3]
    return vad_iterator_cls(
        model,
        threshold=threshold,
        sampling_rate=sample_rate,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=padding_ms,
    )


def _process_segment_with_vad(
    audio_segment: np.ndarray,
    sr: int,
    iterator,
    chunk_ms: int = DEFAULT_CHUNK_MS,
) -> list[tuple[float, float]]:
    """Process a single audio segment with VAD and return segments."""
    chunk_size = int(sr * (chunk_ms / 1000.0))
    segments: list[tuple[float, float]] = []
    open_start: float | None = None
    offset_s = 0.0
    for start in range(0, len(audio_segment), chunk_size):
        chunk = audio_segment[start : start + chunk_size]
        if len(chunk) < chunk_size:
            padded = np.zeros(chunk_size, dtype=np.float32)
            padded[: len(chunk)] = chunk
            chunk = padded

        event = iterator(chunk, return_seconds=True)
        offset_s += len(chunk) / float(sr)

        if event:
            if "start" in event:
                open_start = float(event["start"])
            if "end" in event and open_start is not None:
                segments.append((open_start, float(event["end"])))
                open_start = None

    # flush open segment
    if open_start is not None:
        segments.append((open_start, offset_s))

    return segments


def _process_file_coarse_fine(
    path: Path,
    model,
    utils,
    *,
    coarse_threshold: float,
    coarse_min_silence_ms: int,
    padding_ms: int,
    target_sr: int,
    chunk_ms: int = DEFAULT_CHUNK_MS,
    fine_threshold: float,
    fine_min_silence_ms: int,
    segment_length_s: float = DEFAULT_SEGMENT_LENGTH_S,
) -> tuple[list[tuple[float, float]], int]:
    """Process a single file with coarse-fine VAD and return segments and count."""
    audio, sr = load_and_prepare(path, target_sr=target_sr)
    duration_s = len(audio) / float(sr)

    # Coarse VAD
    coarse_iterator = _create_iterator(
        model, utils,
        threshold=coarse_threshold,
        min_silence_ms=coarse_min_silence_ms,
        padding_ms=padding_ms,
        sample_rate=sr,
    )
    coarse_segments = _process_segment_with_vad(audio, sr, coarse_iterator, chunk_ms)

    # Fine VAD for long segments
    fine_iterator = _create_iterator(
        model, utils,
        threshold=fine_threshold,
        min_silence_ms=fine_min_silence_ms,
        padding_ms=padding_ms,
        sample_rate=sr,
    )

    final_segments: list[tuple[float, float]] = []
    total_segments = 0

    for start_s, end_s in coarse_segments:
        duration = end_s - start_s
        if duration > segment_length_s:
            # Extract the segment
            start_idx = int(start_s * sr)
            end_idx = int(end_s * sr)
            segment_audio = audio[start_idx:end_idx]

            # Run fine VAD on this segment
            fine_segments = _process_segment_with_vad(segment_audio, sr, fine_iterator, chunk_ms)

            # Adjust timestamps to global time
            adjusted_fine_segments = [(s + start_s, e + start_s) for s, e in fine_segments]
            final_segments.extend(adjusted_fine_segments)
            total_segments += len(adjusted_fine_segments)
        else:
            final_segments.append((start_s, end_s))
            total_segments += 1

    return final_segments, total_segments


def process_directory_coarse_fine(
    input_dir: Path,
    *,
    coarse_threshold: float = 0.5,
    coarse_min_silence_ms: int = 100,
    padding_ms: int = 30,
    target_sr: int = TARGET_SAMPLE_RATE,
    chunk_ms: int = DEFAULT_CHUNK_MS,
    fine_threshold: float = 0.3,
    fine_min_silence_ms: int = 50,
    segment_length_s: float = DEFAULT_SEGMENT_LENGTH_S,
) -> tuple[list, int]:
    """Discover all WAV files under *input_dir* and run coarse-fine VAD on each.

    Returns results and total segment count.
    """
    from vad import FileVadResult  # Import from vad.py

    files = discover_wav_files(input_dir)
    if not files:
        return [], 0

    total = len(files)
    logger.info("Found %d WAV file(s) in %s", total, input_dir)
    logger.info(
        "Coarse VAD params — threshold=%.2f  min_silence_ms=%d  padding_ms=%d",
        coarse_threshold, coarse_min_silence_ms, padding_ms,
    )
    logger.info(
        "Fine VAD params — threshold=%.2f  min_silence_ms=%d  padding_ms=%d",
        fine_threshold, fine_min_silence_ms, padding_ms,
    )
    logger.info("Segment length threshold: %.1f s", segment_length_s)

    logger.info("Loading Silero VAD model …")
    model, utils = _load_silero_model()
    logger.info("Model loaded.")

    results: list[FileVadResult] = []
    total_segments = 0
    last_pct_logged = -1

    for idx, fpath in enumerate(files, start=1):
        final_segments, file_segments = _process_file_coarse_fine(
            fpath, model, utils,
            coarse_threshold=coarse_threshold,
            coarse_min_silence_ms=coarse_min_silence_ms,
            padding_ms=padding_ms,
            target_sr=target_sr,
            chunk_ms=chunk_ms,
            fine_threshold=fine_threshold,
            fine_min_silence_ms=fine_min_silence_ms,
            segment_length_s=segment_length_s,
        )

        # Create FileVadResult
        audio, sr = load_and_prepare(fpath, target_sr=target_sr)
        duration_s = len(audio) / float(sr)
        result = FileVadResult(
            path=fpath,
            sample_rate=sr,
            duration_s=duration_s,
            segments=final_segments,
        )
        results.append(result)
        total_segments += file_segments

        pct = int(idx / total * 100)
        pct_bucket = (pct // 10) * 10
        if pct_bucket > last_pct_logged:
            last_pct_logged = pct_bucket
            logger.info(
                "Progress: %3d%% (%d/%d)  —  %s  [%d segments, %.1fs]",
                pct_bucket, idx, total, fpath.name,
                len(result.segments), result.duration_s,
            )

    logger.info("Finished processing %d file(s). Total segments: %d", total, total_segments)
    return results, total_segments


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="coarse_fine",
        description="Batch Coarse-Fine VAD benchmark — compute speech-chunk statistics over a directory of WAV files using two VAD layers.",
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

    # Coarse VAD hyper-parameters
    g = p.add_argument_group("Coarse VAD hyper-parameters")
    g.add_argument(
        "--coarse-threshold",
        type=float,
        default=0.5,
        help="Speech probability threshold for coarse VAD (default: 0.5).",
    )
    g.add_argument(
        "--coarse-min-silence-ms",
        type=int,
        default=100,
        help="Minimum silence duration in ms to split segments for coarse VAD (default: 100).",
    )

    # Fine VAD hyper-parameters
    g2 = p.add_argument_group("Fine VAD hyper-parameters")
    g2.add_argument(
        "--fine-threshold",
        type=float,
        default=0.3,
        help="Speech probability threshold for fine VAD (default: 0.3).",
    )
    g2.add_argument(
        "--fine-min-silence-ms",
        type=int,
        default=50,
        help="Minimum silence duration in ms to split segments for fine VAD (default: 50).",
    )

    # Shared parameters
    p.add_argument(
        "--padding-ms",
        type=int,
        default=30,
        help="Speech padding in ms added around detected segments for both coarse and fine VAD (default: 30).",
    )
    p.add_argument(
        "--segment-length-s",
        type=float,
        default=DEFAULT_SEGMENT_LENGTH_S,
        help=f"Segment length threshold in seconds to trigger fine VAD (default: {DEFAULT_SEGMENT_LENGTH_S}).",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        choices=[8000, 16000],
        help="Target sample rate — files at other rates are resampled (default: 16000).",
    )
    p.add_argument(
        "--chunk-ms",
        type=int,
        default=DEFAULT_CHUNK_MS,
        help=f"Streaming chunk size for Silero VAD in milliseconds (default: {DEFAULT_CHUNK_MS}).",
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
    coarse_threshold: float = 0.5,
    coarse_min_silence_ms: int = 100,
    padding_ms: int = 30,
    sample_rate: int = 16000,
    chunk_ms: int = DEFAULT_CHUNK_MS,
    fine_threshold: float = 0.3,
    fine_min_silence_ms: int = 50,
    segment_length_s: float = DEFAULT_SEGMENT_LENGTH_S,
    verbose: bool = False,
) -> tuple[Path, int]:
    """Programmatic entry point — call this from notebooks or other scripts.

    Returns the path to the generated CSV and total segment count.
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

    results, total_segments = process_directory_coarse_fine(
        input_dir,
        coarse_threshold=coarse_threshold,
        coarse_min_silence_ms=coarse_min_silence_ms,
        padding_ms=padding_ms,
        target_sr=sample_rate,
        chunk_ms=chunk_ms,
        fine_threshold=fine_threshold,
        fine_min_silence_ms=fine_min_silence_ms,
        segment_length_s=segment_length_s,
    )

    stats = compute_stats(results)
    print_summary(stats)
    print(f"Total segments across all files: {total_segments}")
    output = Path(output)
    csv_path = save_results_csv(results, stats, output)
    segments_path = output.with_stem(output.stem + "_segments")
    save_segments_csv(results, segments_path)
    return csv_path, total_segments


def main() -> None:
    args = build_parser().parse_args()
    try:
        csv_path, total_segments = run(
            input_dir=args.input_dir,
            output=args.output,
            coarse_threshold=args.coarse_threshold,
            coarse_min_silence_ms=args.coarse_min_silence_ms,
            padding_ms=args.padding_ms,
            sample_rate=args.sample_rate,
            chunk_ms=args.chunk_ms,
            fine_threshold=args.fine_threshold,
            fine_min_silence_ms=args.fine_min_silence_ms,
            segment_length_s=args.segment_length_s,
            verbose=args.verbose,
        )
        print(f"Results saved to {csv_path}")
        print(f"Total segments: {total_segments}")
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        logging.getLogger(__name__).exception("Benchmark failed")
        sys.exit(2)


if __name__ == "__main__":
    main()