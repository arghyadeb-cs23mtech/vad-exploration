from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from audio import TARGET_SAMPLE_RATE, discover_wav_files, load_and_prepare

logger = logging.getLogger(__name__)

STREAMING_CHUNK_MS = 32


@dataclass
class FileVadResult:
    """VAD results for a single audio file."""
    path: Path
    sample_rate: int
    duration_s: float
    segments: list[tuple[float, float]]  # (start_s, end_s)

    @property
    def chunk_durations(self) -> list[float]:
        return [end - start for start, end in self.segments]


# ---------------------------------------------------------------------------
# Silero model loading (done once, shared across files)
# ---------------------------------------------------------------------------

def _load_silero_model() -> tuple[Any, tuple[Any, ...]]:
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
    model, utils = cast(tuple[Any, tuple[Any, ...]], loaded)
    return model, utils


def _create_iterator(
    model: Any,
    utils: tuple[Any, ...],
    *,
    threshold: float,
    min_silence_ms: int,
    padding_ms: int,
    sample_rate: int,
) -> Any:
    vad_iterator_cls = utils[3]
    return vad_iterator_cls(
        model,
        threshold=threshold,
        sampling_rate=sample_rate,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=padding_ms,
    )


# ---------------------------------------------------------------------------
# Process a single file
# ---------------------------------------------------------------------------

def _process_file(
    path: Path,
    model: Any,
    utils: tuple[Any, ...],
    *,
    threshold: float,
    min_silence_ms: int,
    padding_ms: int,
    target_sr: int,
) -> FileVadResult:
    audio, sr = load_and_prepare(path, target_sr=target_sr)
    duration_s = len(audio) / float(sr)

    iterator = _create_iterator(
        model, utils,
        threshold=threshold,
        min_silence_ms=min_silence_ms,
        padding_ms=padding_ms,
        sample_rate=sr,
    )

    chunk_size = int(sr * (STREAMING_CHUNK_MS / 1000.0))
    segments: list[tuple[float, float]] = []
    open_start: float | None = None

    offset_s = 0.0
    for start in range(0, len(audio), chunk_size):
        chunk = audio[start : start + chunk_size]
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

    return FileVadResult(
        path=path,
        sample_rate=sr,
        duration_s=duration_s,
        segments=segments,
    )


# ---------------------------------------------------------------------------
# Batch processing with progress
# ---------------------------------------------------------------------------

def process_directory(
    input_dir: Path,
    *,
    threshold: float = 0.5,
    min_silence_ms: int = 100,
    padding_ms: int = 30,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> list[FileVadResult]:
    """Discover all WAV files under *input_dir* and run Silero VAD on each.

    Progress is logged at every 10 % boundary.
    """
    files = discover_wav_files(input_dir)
    if not files:
        return []

    total = len(files)
    logger.info("Found %d WAV file(s) in %s", total, input_dir)
    logger.info(
        "Silero params — threshold=%.2f  min_silence_ms=%d  padding_ms=%d  sample_rate=%d",
        threshold, min_silence_ms, padding_ms, target_sr,
    )

    logger.info("Loading Silero VAD model …")
    model, utils = _load_silero_model()
    logger.info("Model loaded.")

    results: list[FileVadResult] = []
    last_pct_logged = -1

    for idx, fpath in enumerate(files, start=1):
        result = _process_file(
            fpath, model, utils,
            threshold=threshold,
            min_silence_ms=min_silence_ms,
            padding_ms=padding_ms,
            target_sr=target_sr,
        )
        results.append(result)

        pct = int(idx / total * 100)
        # Log at every 10% boundary (and always at 100%)
        pct_bucket = (pct // 10) * 10
        if pct_bucket > last_pct_logged:
            last_pct_logged = pct_bucket
            logger.info(
                "Progress: %3d%% (%d/%d)  —  %s  [%d segments, %.1fs]",
                pct_bucket, idx, total, fpath.name,
                len(result.segments), result.duration_s,
            )

    logger.info("Finished processing %d file(s).", total)
    return results
