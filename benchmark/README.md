# Silero VAD Benchmark

Batch-process a directory of WAV files with [Silero VAD](https://github.com/snakers4/silero-vad), then compute and export speech-chunk statistics.

## What it does

1. **Discovers** all `.wav` files recursively under a given directory.
2. **Loads & resamples** each file to the target sample rate (default 16 kHz). A warning is logged whenever resampling is required.
3. **Runs Silero VAD** on every file in streaming mode (32 ms chunks) and collects speech segments.
4. **Logs progress** at every 10 % boundary so you can monitor long runs.
5. **Computes aggregate statistics** on the detected speech-chunk durations:
   - Mean, Median, Max
   - 90th and 99th percentile
6. **Saves results** to a CSV file that can be reloaded with `pandas.read_csv()` for further analysis.

## Quick start

```bash
# From the repository root (with the venv activated)
python -m benchmark.main ./my_audio_dir
```

This writes `benchmark_results.csv` in the current directory and prints a summary to stdout.

## CLI options

```
usage: benchmark [-h] [-o OUTPUT] [--threshold THRESHOLD]
                 [--min-silence-ms MIN_SILENCE_MS] [--padding-ms PADDING_MS]
                 [--sample-rate {8000,16000}] [-v]
                 input_dir

positional arguments:
  input_dir             Directory containing WAV files (searched recursively).

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   Path for the output CSV file (default: benchmark_results.csv).
  -v, --verbose         Enable DEBUG-level logging.

Silero VAD hyper-parameters:
  --threshold THRESHOLD
                        Speech probability threshold (default: 0.5).
  --min-silence-ms MIN_SILENCE_MS
                        Minimum silence duration in ms to split segments (default: 100).
  --padding-ms PADDING_MS
                        Speech padding in ms added around detected segments (default: 30).
  --sample-rate {8000,16000}
                        Target sample rate — files at other rates are resampled (default: 16000).
```

### Examples

```bash
# Custom threshold and output path
python -m benchmark.main ./recordings --threshold 0.4 -o vad_stats.csv

# 8 kHz mode with longer silence gap
python -m benchmark.main ./calls --sample-rate 8000 --min-silence-ms 300

# Verbose logging
python -m benchmark.main ./data -v
```

## Programmatic usage

You can call the benchmark directly from Python without the CLI:

```python
from benchmark.main import run

csv_path = run(
    input_dir="./my_audio_dir",
    output="results.csv",
    threshold=0.5,
    min_silence_ms=100,
    padding_ms=30,
    sample_rate=16000,
)

# Load for further analysis
import pandas as pd
df = pd.read_csv(csv_path, comment="#")
print(df)
```

## Output format

The CSV contains one row per file:

| Column          | Description                              |
|-----------------|------------------------------------------|
| `file`          | Full path to the WAV file                |
| `duration_s`    | Total audio duration in seconds          |
| `num_segments`  | Number of speech segments detected       |
| `total_speech_s`| Sum of all speech-segment durations (s)  |

Aggregate statistics are appended as comment lines (`# key,value`) at the bottom of the file:

```
# Aggregate chunk-duration statistics
# mean,1.234567
# median,0.987654
# max,5.432100
# p90,3.210000
# p99,4.890000
# total_segments,42
```

## Dependencies

Same as the main project — `torch`, `torchaudio`, `numpy`. No additional packages are required. (`pandas` is optional, only needed for reloading the CSV.)
