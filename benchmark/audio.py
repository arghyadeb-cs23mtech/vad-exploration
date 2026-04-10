from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".wav"}
TARGET_SAMPLE_RATE = 16000


def discover_wav_files(directory: Path) -> list[Path]:
    """Recursively find all WAV files under *directory*, sorted by name."""
    files = sorted(
        p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        logger.warning("No WAV files found under %s", directory)
    return files


def load_wav_mono_float32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width * 8} bits")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio, sample_rate


def resample_mono_float32(
    audio: np.ndarray, orig_sr: int, target_sr: int
) -> np.ndarray:
    audio_f32 = np.asarray(audio, dtype=np.float32)
    if len(audio_f32) == 0 or orig_sr == target_sr:
        return audio_f32

    duration_s = len(audio_f32) / float(orig_sr)
    target_len = max(1, int(round(duration_s * target_sr)))

    orig_time = np.linspace(0.0, duration_s, num=len(audio_f32), endpoint=False, dtype=np.float64)
    target_time = np.linspace(0.0, duration_s, num=target_len, endpoint=False, dtype=np.float64)
    resampled = np.interp(target_time, orig_time, audio_f32.astype(np.float64, copy=False))
    return resampled.astype(np.float32)


def load_and_prepare(path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load a WAV file and resample to *target_sr* if necessary.

    Returns (audio_float32, sample_rate).
    """
    audio, sr = load_wav_mono_float32(path)
    if sr != target_sr:
        logger.warning(
            "Resampling %s from %d Hz to %d Hz", path.name, sr, target_sr
        )
        audio = resample_mono_float32(audio, sr, target_sr)
        sr = target_sr
    return audio, sr
