# How much do synthetic callers lie?

A local, reproducible experiment on the sim-to-real gap in voice-agent evaluation: do synthetic callers (TTS voices) give the same ASR scores, and above all the same ranking between ASR models, as real human speech? Public ASR models stand in for the voice agent's ear.

## Setup

```
uv sync                      # Python 3.12, deps pinned in uv.lock
uv run python scripts/smoke_test.py
```

Data (gitignored, in `data/`): LibriSpeech test-clean (openslr.org, CC BY 4.0), ESC-50 (GitHub, CC BY-NC), Piper voices. Models live in the Hugging Face cache.

## Components

| Role | ID |
|---|---|
| TTS | hexgrad/Kokoro-82M (af_heart, am_michael) |
| TTS | Piper en_US-lessac-medium, en_US-ryan-medium (rhasspy/piper-voices) |
| TTS | facebook/mms-tts-eng |
| TTS | macOS `say` (Samantha en_US, Daniel en_GB) |
| ASR | mlx-community/whisper-large-v3-turbo (mlx-whisper) |
| ASR | facebook/wav2vec2-base-960h (transformers, MPS) |
| ASR | mlx-community/parakeet-tdt-0.6b-v2 (parakeet-mlx) |
| Channel | ffmpeg 8 kHz G.711 mu-law round trip |

## Run the experiment

```
uv run python scripts/01_make_corpus.py          # 250 LibriSpeech sentences x (human + 7 TTS), 122 caller sentences x 7 TTS
uv run python scripts/02_channel_selftest.py     # channel checks (G.711, SNR, loss, muffle)
uv run python scripts/03_asr_selftest.py         # ASR checks
uv run python scripts/04_run_experiment.py       # every voice x condition x model (cached, resumable)
uv run python scripts/05_analyze.py              # tables, figures, summary.json
```

`04_run_experiment.py --model whisper --conditions clean,realistic` runs one worker on a subset; `--merge-only` rebuilds `results/utterances.csv` from the transcript shards. Results: `results/utterances.csv` (one row per sentence x voice x condition x model), `results/tables/`, `results/figures/`, `results/FINDINGS.md`.

If `import sim2real` fails under `uv run`, macOS may have flagged the editable `.pth` file in `.venv` as hidden (Python 3.12 skips hidden `.pth` files). The scripts add `src/` to `sys.path` themselves; `chflags nohidden .venv/lib/python3.12/site-packages/*.pth` also fixes it.
