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
