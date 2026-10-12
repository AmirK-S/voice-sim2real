#!/usr/bin/env bash
# Reproduce the published run end to end. Transcripts are cached by audio hash,
# so the script can be stopped and resumed at any point.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="uv run python"

$PY scripts/smoke_test.py
$PY scripts/01_make_corpus.py
$PY scripts/01b_rate_matched.py

# Grid used in the published run (Whisper is the slowest model, so it runs fewer channels).
$PY scripts/04_run_experiment.py --model parakeet
$PY scripts/04_run_experiment.py --model wav2vec2
$PY scripts/04_run_experiment.py --model whisper --conditions clean,phone,phone+noise10,realistic
for m in whisper parakeet wav2vec2; do
  $PY scripts/04_run_experiment.py --model "$m" --group rate --conditions clean
  $PY scripts/04_run_experiment.py --model "$m" --noise-only
done

$PY scripts/05_analyze.py
