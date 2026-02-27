from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from vad.audio import (iter_audio_chunks, load_wav_mono_float32,
                       resample_mono_float32)
from vad.clips import build_padded_slices, export_vad_clips
from vad.silero_streaming import SileroStreamingVad

STREAMING_CHUNK_MS = 32
STREAMING_SUPPORTED_SAMPLE_RATES = {8000, 16000}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone Silero streaming VAD cleanup demo"
    )
    parser.add_argument("audio", type=Path, help="Path to WAV input")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Silero speech threshold"
    )
    parser.add_argument(
        "--padding-ms", type=int, default=120, help="Clip padding in ms"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output root (optional)"
    )
    parser.add_argument(
        "--chunk-output",
        action="store_true",
        help="Enable verbose chunk-level output",
    )
    return parser


def save_config_file(
    output_path: Path,
    audio_path: Path,
    threshold: float,
    padding_ms: int,
    sample_rate: int,
    segments_count: int,
    clips_count: int,
) -> Path:
    config_path = output_path / "config.txt"
    config_content = "\n".join(
        [
            f"audio={audio_path}",
            f"threshold={threshold}",
            f"padding_ms={padding_ms}",
            f"sample_rate={sample_rate}",
            f"segments_detected={segments_count}",
            f"clips_written={clips_count}",
        ]
    )
    config_path.write_text(config_content + "\n", encoding="utf-8")
    return config_path


def main() -> None:
    args = build_parser().parse_args()
    if args.chunk_output:
        print(
            "[run] "
            f"audio={args.audio} threshold={args.threshold} "
            f"padding_ms={args.padding_ms} output_dir={args.output_dir}"
        )

    audio, sample_rate = load_wav_mono_float32(args.audio)
    if sample_rate not in STREAMING_SUPPORTED_SAMPLE_RATES:
        target_sr = 16000
        audio = resample_mono_float32(audio, sample_rate, target_sr)
        if args.chunk_output:
            print(f"[streaming] auto-resampled from {sample_rate} Hz to {target_sr} Hz")
        sample_rate = target_sr

    vad = SileroStreamingVad(threshold=args.threshold)
    vad.reset()

    segments = []
    for chunk_index, chunk in enumerate(
        iter_audio_chunks(audio, sample_rate, STREAMING_CHUNK_MS), start=1
    ):
        chunk_segments = vad.process_chunk(chunk, sample_rate)
        if args.chunk_output and chunk_segments:
            for segment in chunk_segments:
                print(
                    f"[chunk {chunk_index:05d}] "
                    f"start={segment.start_s:.3f}s end={segment.end_s:.3f}s"
                )
        segments.extend(chunk_segments)
    segments.extend(vad.flush())

    total_duration_s = len(audio) / float(sample_rate)
    slices = build_padded_slices(
        segments=segments,
        total_duration_s=total_duration_s,
        padding_ms=args.padding_ms,
    )

    if args.chunk_output:
        print(f"[silero-streaming] detected {len(segments)} speech segments")
    for _, _, slice_segments in slices:
        line = ", ".join(
            f"{segment.start_s:.3f} {segment.end_s:.3f}" for segment in slice_segments
        )
        print(line)

    if args.output_dir is not None:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = args.output_dir / "silero" / run_timestamp
        written = export_vad_clips(
            audio=audio,
            sample_rate=sample_rate,
            segments=segments,
            output_dir=output_path,
            prefix="silero_streaming",
            padding_ms=args.padding_ms,
        )
        config_path = save_config_file(
            output_path=output_path,
            audio_path=args.audio,
            threshold=args.threshold,
            padding_ms=args.padding_ms,
            sample_rate=sample_rate,
            segments_count=len(segments),
            clips_count=len(written),
        )
        if args.chunk_output:
            print(f"[clips] wrote {len(written)} clips to {output_path}")
            print(f"[clips] wrote run config to {config_path}")


if __name__ == "__main__":
    main()
