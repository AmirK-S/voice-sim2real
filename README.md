# How much do synthetic callers lie?

A local, reproducible experiment on the acoustic half of the sim-to-real gap in voice-agent evaluation: when the same scripts are read by TTS voices instead of humans, do public ASR models (standing in for the voice agent's ear) give the same scores, and above all the same ranking between models? Behaviour (what real callers say, panic, disfluency, turn-taking) is not measured. Results and caveats: `results/FINDINGS.md`.

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
| Channel | channel v2 in `src/sim2real/channel.py`: P.56-style AGC to -26 dBov, IRS-like handset filter, numpy G.711 mu-law (bit-identical decoder to ffmpeg), Appendix I-style packet-loss concealment, libopus SILK narrowband (system libopus via ctypes, real decoder PLC), G.722 wideband through ffmpeg, ESC-50 noise calibrated to the post-channel SNR |

## Run the experiment

```
uv run python scripts/01_make_corpus.py          # 250 LibriSpeech sentences x (human + 7 TTS), 122 caller sentences x 7 TTS
uv run python scripts/01b_rate_matched.py        # rate-matched variant of every TTS voice (LibriSpeech only)
uv run python scripts/02_channel_selftest.py     # channel checks (G.711, levels, SNR, loss and PLC, Opus, G.722)
uv run python scripts/03_asr_selftest.py         # ASR checks
scripts/run_v2.sh whisper & scripts/run_v2.sh others   # the v2 grid in two chains (cached, resumable)
uv run python scripts/04_run_experiment.py --merge-only
uv run python scripts/05_analyze.py              # tables, figures, summary.json
uv run --with pytest python -m pytest tests      # stats unit tests
```

`04_run_experiment.py --model whisper --conditions clean,realistic` runs one worker on a subset (`--group rate` for the rate-matched voices, `--noise-only` for the hallucination check); `--merge-only` rebuilds `results/utterances.csv` from the transcript shards. Results: `results/utterances.csv` (one row per sentence x voice x condition x model), `results/tables/`, `results/figures/`, `results/FINDINGS.md`.

If `import sim2real` fails under `uv run`, macOS may have flagged the editable `.pth` file in `.venv` as hidden (Python 3.12 skips hidden `.pth` files). The scripts add `src/` to `sys.path` themselves; `chflags nohidden .venv/lib/python3.12/site-packages/*.pth` also fixes it.

Scoring uses Whisper's English normalizer with its British-to-American spelling map, vendored as `src/sim2real/whisper_english_spelling.json` (from openai-whisper 20250625, MIT licence). The Opus conditions need the system libopus (`brew install opus`; ffmpeg already links it).

Licences: LibriSpeech CC BY 4.0; ESC-50 CC BY-NC 3.0 (non-commercial: mention it if results are shared).
