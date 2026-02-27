from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def load_wav_mono_float32(path: str | Path) -> tuple[np.ndarray, int]:
    wav_path = Path(path)
    with wave.open(str(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.getnframes()
        raw = wav_file.readframes(frames)

    if sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width * 8} bits")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio, sample_rate


def iter_audio_chunks(
    audio: np.ndarray, sample_rate: int, chunk_ms: int
) -> list[np.ndarray]:
    chunk_size = int(sample_rate * (chunk_ms / 1000.0))
    if chunk_size <= 0:
        raise ValueError("chunk_ms must be > 0")

    chunks: list[np.ndarray] = []
    for start in range(0, len(audio), chunk_size):
        chunk = audio[start : start + chunk_size]
        if len(chunk) < chunk_size:
            padded = np.zeros(chunk_size, dtype=np.float32)
            padded[: len(chunk)] = chunk.astype(np.float32, copy=False)
            chunks.append(padded)
            continue
        chunks.append(chunk.astype(np.float32, copy=False))
    return chunks


def resample_mono_float32(
    audio: np.ndarray, orig_sample_rate: int, target_sample_rate: int
) -> np.ndarray:
    if orig_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("Sample rates must be positive")

    audio_f32 = np.asarray(audio, dtype=np.float32)
    if len(audio_f32) == 0 or orig_sample_rate == target_sample_rate:
        return audio_f32

    duration_s = len(audio_f32) / float(orig_sample_rate)
    target_len = max(1, int(round(duration_s * target_sample_rate)))

    orig_time = np.linspace(
        0.0, duration_s, num=len(audio_f32), endpoint=False, dtype=np.float64
    )
    target_time = np.linspace(
        0.0, duration_s, num=target_len, endpoint=False, dtype=np.float64
    )
    resampled = np.interp(
        target_time, orig_time, audio_f32.astype(np.float64, copy=False)
    )
    return resampled.astype(np.float32)


def save_wav_mono_float32(
    path: str | Path, audio: np.ndarray, sample_rate: int
) -> None:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    audio_f32 = np.asarray(audio, dtype=np.float32)
    audio_clipped = np.clip(audio_f32, -1.0, 1.0)
    pcm16 = (audio_clipped * 32767.0).astype(np.int16)

    with wave.open(str(path_obj), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
