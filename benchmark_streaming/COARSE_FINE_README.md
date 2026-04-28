# Coarse-Fine VAD Benchmark

This script implements a two-layer Voice Activity Detection (VAD) system for batch-processing WAV files. It uses Silero VAD with a coarse layer for initial segmentation and a fine layer for refining long segments.

## How It Works

1. **Coarse Layer**: Processes the entire audio file to identify initial speech segments.
2. **Fine Layer**: For segments longer than the specified threshold (default 30 seconds), applies a more sensitive VAD to split them into smaller, more precise segments.
3. **Output**: Combines all segments and counts the total number across all processed files.

## Usage Example

```bash
# Basic usage with default parameters
python coarse_fine.py /path/to/audio/directory

# Custom parameters
python coarse_fine.py /path/to/audio/directory \
    --coarse-threshold 0.4 \
    --fine-threshold 0.2 \
    --padding-ms 20 \
    --segment-length-s 25 \
    --output my_results.csv \
    --verbose
```

## Arguments

- `input_dir`: Directory containing WAV files (required)
- `--output`, `-o`: Output CSV file path (default: coarse_fine_results.csv)
- `--coarse-threshold`: Speech probability threshold for coarse VAD (default: 0.5)
- `--coarse-min-silence-ms`: Minimum silence duration for coarse VAD (default: 100)
- `--fine-threshold`: Speech probability threshold for fine VAD (default: 0.3)
- `--fine-min-silence-ms`: Minimum silence duration for fine VAD (default: 50)
- `--padding-ms`: Speech padding in ms for both layers (default: 30)
- `--segment-length-s`: Threshold to trigger fine VAD (default: 30.0)
- `--sample-rate`: Target sample rate (default: 16000, choices: 8000, 16000)
- `--chunk-ms`: Streaming chunk size (default: 32)
- `--verbose`, `-v`: Enable debug logging

## Output

- CSV file with segment statistics
- Segments CSV with detailed segment information
- Console output with summary and total segment count
