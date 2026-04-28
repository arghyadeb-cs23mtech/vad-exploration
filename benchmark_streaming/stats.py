from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from vad import FileVadResult

logger = logging.getLogger(__name__)


def compute_stats(results: list[FileVadResult]) -> dict[str, object]:
    """Compute aggregate chunk-duration statistics across all files.

    Returns a dict with scalar summary values and per-file rows suitable for
    DataFrame construction.
    """
    all_durations: list[float] = []
    per_file_rows: list[dict[str, object]] = []

    for r in results:
        durations = r.chunk_durations
        all_durations.extend(durations)
        per_file_rows.append({
            "file": str(r.path),
            "duration_s": round(r.duration_s, 4),
            "num_segments": len(r.segments),
            "total_speech_s": round(sum(durations), 4) if durations else 0.0,
        })

    count_over_30s = sum(1 for d in all_durations if d > 30.0)
    if not all_durations:
        logger.warning("No speech segments detected across any file.")
        return {
            "per_file": per_file_rows,
            "mean": 0.0,
            "median": 0.0,
            "max": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "total_segments": 0,
            "count_over_30s": 0,
        }

    arr = np.array(all_durations, dtype=np.float64)
    return {
        "per_file": per_file_rows,
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "total_segments": len(arr),
        "count_over_30s": count_over_30s,
    }


def save_results_csv(
    results: list[FileVadResult],
    stats: dict[str, object],
    output_path: Path,
) -> Path:
    """Write per-file results and aggregate statistics to a CSV file.

    Uses only the stdlib csv module so pandas is not a hard dependency.
    A pandas-friendly CSV is produced so it can be loaded with
    ``pd.read_csv(path)`` for further analysis.
    """
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_file: list[dict[str, object]] = stats.get("per_file", [])  # type: ignore[assignment]

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        # --- Section 1: per-file rows ---
        fieldnames = ["file", "duration_s", "num_segments", "total_speech_s"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_file:
            writer.writerow(row)

        # --- Section 2: aggregate summary (appended after a blank line) ---
        fh.write("\n")
        fh.write("# Aggregate chunk-duration statistics\n")
        for key in ("mean", "median", "max", "p90", "p99", "total_segments", "count_over_30s"):
            value = stats.get(key, "")
            if isinstance(value, float):
                fh.write(f"# {key},{value:.6f}\n")
            else:
                fh.write(f"# {key},{value}\n")

    logger.info("Results written to %s", output_path)
    return output_path


def save_segments_csv(
    results: list[FileVadResult],
    output_path: Path,
) -> Path:
    """Write one row per detected speech segment across all files.

    Columns: file, segment_index, start_s, end_s, duration_s

    Loadable with:
        df = pd.read_csv(path)
    """
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file", "segment_index", "start_s", "end_s", "duration_s"]

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            for i, (start, end) in enumerate(r.segments):
                writer.writerow({
                    "file": str(r.path),
                    "segment_index": i,
                    "start_s": round(start, 6),
                    "end_s": round(end, 6),
                    "duration_s": round(end - start, 6),
                })

    logger.info("Segment-level data written to %s", output_path)
    return output_path


def print_summary(stats: dict[str, object]) -> None:
    """Pretty-print aggregate statistics to stdout."""
    print("\n===== Benchmark Summary =====")
    print(f"  Total speech segments : {stats['total_segments']}")
    print(f"  Mean chunk duration   : {stats['mean']:.4f} s")
    print(f"  Median chunk duration : {stats['median']:.4f} s")
    print(f"  Max chunk duration    : {stats['max']:.4f} s")
    print(f"  90th percentile       : {stats['p90']:.4f} s")
    print(f"  99th percentile       : {stats['p99']:.4f} s")
    print("=============================\n")
