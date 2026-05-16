from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from audio import TARGET_SAMPLE_RATE, discover_wav_files, load_and_prepare  # noqa: E402

from vadcore.factory import create_vad  # noqa: E402

logger = logging.getLogger(__name__)

STREAMING_CHUNK_MS = 32


@dataclass
class FileVadResult:
    """VAD results for a single audio file."""
    path: Path
    sample_rate: int
    duration_s: float
    segments: list[tuple[float, float]]

    @property
    def chunk_durations(self) -> list[float]:
        return [end - start for start, end in self.segments]


def _process_file(path: Path, vad: Any, *, target_sr: int) -> FileVadResult:
    audio, sr = load_and_prepare(path, target_sr=target_sr)
    duration_s = len(audio) / float(sr)

    vad.reset()
    chunk_size = vad.chunk_samples
    segments: list[tuple[float, float]] = []

    for start in range(0, len(audio), chunk_size):
        chunk = audio[start : start + chunk_size]
        if len(chunk) < chunk_size:
            padded = np.zeros(chunk_size, dtype=np.float32)
            padded[: len(chunk)] = chunk
            chunk = padded

        for seg in vad.process_chunk(chunk):
            segments.append((seg.start_s, seg.end_s))

    for seg in vad.flush():
        segments.append((seg.start_s, seg.end_s))

    return FileVadResult(
        path=path,
        sample_rate=sr,
        duration_s=duration_s,
        segments=segments,
    )


def process_directory(
    input_dir: Path,
    *,
    vad_model: str = "silero",
    threshold: float = 0.5,
    min_silence_ms: int = 100,
    padding_ms: int = 30,
    target_sr: int = TARGET_SAMPLE_RATE,
    chunk_ms: int = STREAMING_CHUNK_MS,  # accepted for CLI compat; only Silero is fixed at 32 ms today
    **vad_kwargs: Any,
) -> list[FileVadResult]:
    """Discover all WAV files under *input_dir* and run the selected streaming VAD on each."""
    files = discover_wav_files(input_dir)
    if not files:
        return []

    total = len(files)
    logger.info("Found %d WAV file(s) in %s", total, input_dir)
    logger.info(
        "VAD params — model=%s  threshold=%.2f  min_silence_ms=%d  padding_ms=%d  sample_rate=%d  chunk_ms_hint=%d",
        vad_model, threshold, min_silence_ms, padding_ms, target_sr, chunk_ms,
    )
    if vad_kwargs:
        logger.info("VAD extra — %s", vad_kwargs)

    logger.info("Loading %s VAD …", vad_model)
    vad = create_vad(
        vad_model,
        sample_rate=target_sr,
        threshold=threshold,
        min_silence_ms=min_silence_ms,
        padding_ms=padding_ms,
        **vad_kwargs,
    )
    actual_chunk_ms = 1000.0 * vad.chunk_samples / target_sr
    logger.info(
        "Model loaded — chunk_samples=%d (%.1f ms @ %d Hz).",
        vad.chunk_samples, actual_chunk_ms, target_sr,
    )

    results: list[FileVadResult] = []
    last_pct_logged = -1

    for idx, fpath in enumerate(files, start=1):
        result = _process_file(fpath, vad, target_sr=target_sr)
        results.append(result)

        pct = int(idx / total * 100)
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
