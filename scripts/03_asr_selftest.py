"""Self-test for sim2real.asr.

1. normalize() sanity checks.
2. For each model: 5 LibriSpeech test-clean files (5 different speakers), uncached run,
   corpus WER and seconds of compute per second of audio (model load excluded by a warm-up).
3. Cache: a second call must return identical text from disk, much faster.
4. Whisper: the batched decoding path must match mlx_whisper.transcribe on these files.

Writes results/raw/asr_selftest.json. Run: uv run python scripts/03_asr_selftest.py
"""
from __future__ import annotations

import json
import time

import numpy as np

from sim2real import asr
from sim2real.datasets import librispeech_index
from sim2real.paths import RAW

N = 5

# 1. normalizer
cases = {
    "Help! He's NOT breathing...": "help he is not breathing",
    "Chest-pain, at twenty two Main St.": "chest pain at 22 main saint",  # Whisper rule: st -> saint
    "  Uh, there's a FIRE  ": "there is a fire",
}
for raw, want in cases.items():
    got = asr.normalize(raw)
    print(f"normalize({raw!r}) = {got!r}")
    assert got == want, (got, want)
assert asr.wer("help he is not breathing", "Help, he's not breathing.") == 0.0
assert asr.resolve("whisper") == "whisper-large-v3-turbo"

# 2. pick 5 files from 5 speakers, deterministic
idx = librispeech_index()
pick = idx.groupby("speaker").head(1).sort_values("id").iloc[:: max(1, len(idx.speaker.unique()) // N)].head(N)
paths, refs = pick["path"].tolist(), pick["text"].tolist()
audio = [asr._load16k(p) for p in paths]
dur = sum(len(a) for a in audio) / asr.SR
print(f"\n{N} files, {dur:.1f} s of audio: {[p.split('/')[-1] for p in paths]}")

report = {"files": paths, "audio_s": round(dur, 2), "models": {}}
for mid in asr.MODELS:
    asr._run_batch(mid, [np.zeros(asr.SR, dtype=np.float32)])  # load + warm up
    t0 = time.perf_counter()
    hyps = asr.transcribe(paths, mid, use_cache=False)
    t = time.perf_counter() - t0
    w = asr.wer(refs, hyps)
    # cache round trip
    asr.transcribe(paths, mid)  # fill
    t1 = time.perf_counter()
    cached = asr.transcribe(paths, mid)
    tc = time.perf_counter() - t1
    assert cached == hyps, f"{mid}: cached transcript differs"
    rep = {"wer": round(w, 4), "sec_per_audio_sec": round(t / dur, 4), "wall_s": round(t, 2),
           "cached_wall_s": round(tc, 4), "hyps": hyps}
    if mid == asr.MODELS[0]:
        ref_path = [asr._whisper_long(a) for a in audio]
        rep["batched_matches_transcribe"] = [asr.normalize(a) == asr.normalize(b) for a, b in zip(hyps, ref_path)]
        rep["transcribe_wer"] = round(asr.wer(refs, ref_path), 4)
    report["models"][mid] = rep
    print(f"{mid:24s} WER {w:.3f}  {t / dur:.3f} s/audio-s  cached {tc * 1000:.0f} ms"
          + (f"  batched==transcribe {rep['batched_matches_transcribe']}" if "batched_matches_transcribe" in rep else ""))
    for r, h in zip(refs, hyps):
        if asr.normalize(r) != asr.normalize(h):
            print(f"   REF {asr.normalize(r)}\n   HYP {asr.normalize(h)}")

for mid, rep in report["models"].items():
    assert rep["wer"] < 0.15, f"{mid} WER {rep['wer']} too high on clean read speech"

RAW.mkdir(parents=True, exist_ok=True)
(RAW / "asr_selftest.json").write_text(json.dumps(report, indent=2))
print("\nOK, wrote", RAW / "asr_selftest.json")
