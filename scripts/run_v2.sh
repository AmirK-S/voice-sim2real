#!/bin/zsh
# Channel v2 rerun in two chains that share the GPU (Whisper; Parakeet then wav2vec2).
# The grid is cut to fit about 4 GPU-hours on an M-series Mac (Whisper large-v3-turbo decodes
# about 1.3 clips/s): Whisper runs 3 conditions, Parakeet 7, wav2vec2 all 10; every model runs the
# rate-matched voices on clean audio and the noise-only check. Shards are cached, so the
# "optional" tail of each chain can be resumed later.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R=scripts/04_run_experiment.py
wait_rate() { until grep -q "@rate" data/corpus/manifest.csv; do sleep 30; done; }
if [[ "$1" == "whisper" ]]; then
  $PY $R --model whisper --conditions clean,realistic,phone+noise10
  wait_rate
  $PY $R --model whisper --group rate --conditions clean
  $PY $R --model whisper --noise-only
  echo "CHAIN whisper CORE DONE"
  $PY $R --model whisper --conditions phone,opus-nb,g722-wb   # optional
else
  $PY $R --model parakeet --conditions clean,phone,realistic,phone+noise10,opus-nb,g722-wb,phone+muffle
  $PY $R --model wav2vec2
  wait_rate
  for m in parakeet wav2vec2; do
    $PY $R --model $m --group rate --conditions clean
    $PY $R --model $m --noise-only
  done
  echo "CHAIN others CORE DONE"
  $PY $R --model parakeet --conditions phone+noise5,phone+loss5,opus-nb+loss5   # optional
fi
echo "CHAIN $1 DONE"
