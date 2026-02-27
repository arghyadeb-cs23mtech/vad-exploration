# Cleanup: standalone Silero streaming demo

This is an independent mini-project containing only:
- Silero VAD **streaming** detection
- speech clip splitting + WAV saving with padding

## Structure

```text
cleanup/
├── pyproject.toml
├── main.py
├── README.md
└── cleanup_vad/
    ├── __init__.py
    ├── audio.py
    ├── clips.py
    ├── silero_streaming.py
    └── types.py
```

## Run

```bash
cd cleanup
uv sync
uv run python main.py ../news.wav --threshold 0.5 --padding-ms 120
```

Default terminal output prints only timestamps (`start_s end_s`).

To write clips + config files, provide `--output-dir`:

```bash
uv run python main.py ../news.wav --threshold 0.5 --padding-ms 120 --output-dir demo_clips
```

Enable verbose chunk-level logs with:

```bash
uv run python main.py ../news.wav --threshold 0.5 --padding-ms 120 --output-dir demo_clips --chunk-output
```

Output clips are written to:
- `demo_clips/silero/<run_timestamp>/`

Each run output folder also includes:
- `config.txt` (run configuration + summary)
