#!/bin/zsh
# Re-run the noise conditions after the noise-bed fix (silent ESC-50 padding); the transcript
# cache makes this cheap: only clips whose noise bed changed are transcribed again.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R=scripts/04_run_experiment.py
$PY $R --model parakeet --conditions phone+noise10,realistic,phone+noise5,phone+loss5,opus-nb+loss5
$PY $R --model wav2vec2 --conditions phone+noise10,realistic,phone+noise5
until grep -q "whisper-large-v3-turbo\] phone:" results/raw/logs/v2_whisper.log; do sleep 20; done
$PY $R --model whisper --conditions phone+noise10,realistic
echo "FIX DONE"
