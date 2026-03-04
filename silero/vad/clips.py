from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio import save_wav_mono_float32
from .types import VadSegment


def build_padded_slices(
    segments: list[VadSegment],
    total_duration_s: float,
) -> list[tuple[float, float, list[VadSegment]]]:

    padded_segments: list[tuple[float, float, VadSegment]] = []
    for segment in segments:
        start_s = max(0.0, segment.start_s)
        end_s = min(total_duration_s, segment.end_s)
        if end_s > start_s:
            padded_segments.append((start_s, end_s, segment))

    if not padded_segments:
        return []

    padded_segments.sort(key=lambda item: item[0])
    merged: list[tuple[float, float, list[VadSegment]]] = [
        (padded_segments[0][0], padded_segments[0][1], [padded_segments[0][2]])
    ]

    for start_s, end_s, segment in padded_segments[1:]:
        current_start_s, current_end_s, current_segments = merged[-1]
        if start_s <= current_end_s:
            merged[-1] = (
                current_start_s,
                max(current_end_s, end_s),
                [*current_segments, segment],
            )
        else:
            merged.append((start_s, end_s, [segment]))

    return merged


def build_padded_ranges(
    segments: list[VadSegment],
    total_duration_s: float,
) -> list[tuple[float, float]]:
    slices = build_padded_slices(
        segments=segments,
        total_duration_s=total_duration_s,
    )
    return [(start_s, end_s) for start_s, end_s, _ in slices]


def export_vad_clips(
    audio: np.ndarray,
    sample_rate: int,
    segments: list[VadSegment],
    output_dir: str | Path,
    prefix: str,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    audio_f32 = np.asarray(audio, dtype=np.float32)
    total_duration_s = len(audio_f32) / float(sample_rate)
    slices = build_padded_slices(
        segments=segments,
        total_duration_s=total_duration_s,
    )

    written_paths: list[Path] = []
    for index, (start_s, end_s, _) in enumerate(slices, start=1):
        start_idx = int(round(start_s * sample_rate))
        end_idx = int(round(end_s * sample_rate))
        clip = audio_f32[start_idx:end_idx]
        if len(clip) == 0:
            continue

        file_path = output_path / f"{prefix}_clip_{index:03d}.wav"
        save_wav_mono_float32(file_path, clip, sample_rate)
        written_paths.append(file_path)

    return written_paths
