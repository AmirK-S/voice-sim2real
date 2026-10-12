"""Run the full grid: every voice x every channel condition x every ASR model.

Sets
- librispeech: 250 sentences, voice `human` (real read speech) + 7 TTS voices.
- emergency:   122 written caller sentences, 7 TTS voices (no human audio).

Each channel condition is applied on the fly (sim2real.channel.apply) with a seed
derived from the sentence id only, so the same sentence gets the same noise clip,
noise offset and loss pattern under every voice: comparisons stay paired.

Transcripts are cached by audio hash (sim2real.asr), so the script is safe to
stop and re-run. Raw transcripts land in results/raw/transcripts/<model>/<condition>.csv;
the merge step scores them into results/utterances.csv (one row per
set x sentence x voice x condition x model).

Usage
  python scripts/04_run_experiment.py                 # all models in parallel, then merge
  python scripts/04_run_experiment.py --model whisper # one worker
  python scripts/04_run_experiment.py --merge-only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # robust to a hidden editable .pth on macOS

import numpy as np
import pandas as pd
import soundfile as sf

from sim2real import asr, channel, data
from sim2real.paths import RAW, RESULTS

MANIFEST = ROOT / "data" / "corpus" / "manifest.csv"
TRANSCRIPTS = RAW / "transcripts"
# Priority order: the most informative conditions first, so a partial run is still useful.
CONDITION_ORDER = ["clean", "phone", "realistic", "phone+noise10", "phone+noise5",
                   "phone+muffle", "phone+loss10", "phone+noise20", "phone+loss5"]
assert set(CONDITION_ORDER) == set(channel.CONDITIONS), "condition list out of sync"


def utt_seed(utt_id: str) -> int:
    return zlib.crc32(utt_id.encode()) & 0x7FFFFFFF


def load_manifest() -> pd.DataFrame:
    m = pd.read_csv(MANIFEST, keep_default_na=False)
    m["duration_s"] = m["duration_s"].astype(float)
    return m


def read16k(path: str) -> np.ndarray:
    x, sr = sf.read(ROOT / path, dtype="float32")
    assert sr == channel.SR, (path, sr)
    return x.mean(axis=1) if x.ndim > 1 else x


def worker(model: str, conditions: list[str], sets: list[str]) -> None:
    model = asr.resolve(model)
    m = load_manifest()
    m = m[m["set"].isin(sets)]
    out_dir = TRANSCRIPTS / model
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    for cond in conditions:
        shard = out_dir / f"{cond}.csv"
        if shard.exists():
            done = pd.read_csv(shard, keep_default_na=False)
            if len(done) == len(m):
                print(f"[{model}] {cond}: already done", flush=True)
                continue
        t0, rows = time.time(), []
        for (s, voice), g in m.groupby(["set", "voice"], sort=True):
            arrays = [channel.apply(cond, read16k(p), seed=utt_seed(i))
                      for p, i in zip(g["path"], g["id"])]
            hyps = asr.transcribe_arrays(arrays, model)
            rows += [{"set": s, "id": i, "voice": voice, "hyp": h}
                     for i, h in zip(g["id"], hyps)]
        tmp = shard.with_suffix(".tmp")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.replace(shard)
        print(f"[{model}] {cond}: {len(rows)} clips in {time.time() - t0:.0f} s "
              f"(total {time.time() - t_all:.0f} s)", flush=True)


def _kw_hits(ref_kws: list[str], hyp: str) -> list[str]:
    found = set(data.find_keywords(hyp))
    return [k for k in ref_kws if k in found]


def merge() -> pd.DataFrame:
    import jiwer

    m = load_manifest()
    shards = []
    for model_dir in sorted(TRANSCRIPTS.iterdir()):
        for f in sorted(model_dir.glob("*.csv")):
            d = pd.read_csv(f, keep_default_na=False)
            d["model"], d["condition"] = model_dir.name, f.stem
            shards.append(d)
    t = pd.concat(shards, ignore_index=True)
    t = t.merge(m[["set", "id", "voice", "text", "keywords", "category", "speaker", "duration_s"]],
                on=["set", "id", "voice"], how="left", validate="many_to_one")
    assert t["text"].notna().all()
    t = t.rename(columns={"text": "ref"})

    ref_n = {r: asr.normalize(r) for r in t["ref"].unique()}
    errs, nref = [], []
    for r, h in zip(t["ref"], t["hyp"]):
        rn, hn = ref_n[r], asr.normalize(h)
        n = len(rn.split())
        if not hn:
            e = n
        else:
            o = jiwer.process_words(rn, hn)
            e = o.substitutions + o.deletions + o.insertions
        errs.append(e)
        nref.append(n)
    t["errors"], t["n_ref"] = errs, nref
    t["wer"] = t["errors"] / t["n_ref"]
    # Keywords from the reference (whole-word match, sim2real.data), and which survive in the hypothesis.
    ref_kws = [data.find_keywords(r) for r in t["ref"]]
    hits = [_kw_hits(k, h) for k, h in zip(ref_kws, t["hyp"])]
    t["kw_ref"] = ["|".join(k) for k in ref_kws]
    t["kw_hit"] = ["|".join(k) for k in hits]
    t["kw_total"] = [len(k) for k in ref_kws]
    t["kw_hits"] = [len(k) for k in hits]
    t["kw_detected_any"] = [len(data.find_keywords(h)) > 0 for h in t["hyp"]]
    cols = ["set", "id", "speaker", "category", "voice", "condition", "model", "duration_s",
            "ref", "hyp", "errors", "n_ref", "wer", "kw_ref", "kw_hit", "kw_total", "kw_hits",
            "kw_detected_any"]
    t = t[cols].sort_values(["set", "model", "condition", "voice", "id"]).reset_index(drop=True)
    t.to_csv(RESULTS / "utterances.csv", index=False)
    print(f"merged {len(t)} rows -> results/utterances.csv")
    print(t.groupby(["set", "model", "condition"]).size().unstack(0).to_string())
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="run one worker for this model (whisper, wav2vec2, parakeet)")
    ap.add_argument("--conditions", default=",".join(CONDITION_ORDER))
    ap.add_argument("--sets", default="librispeech,emergency")
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args()
    conditions = a.conditions.split(",")
    sets = a.sets.split(",")
    if a.merge_only:
        merge()
        return
    if a.model:
        worker(a.model, conditions, sets)
        return
    log_dir = RAW / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for mdl in asr.MODELS:
        log = open(log_dir / f"run_{mdl}.log", "a")
        procs.append(subprocess.Popen(
            [sys.executable, __file__, "--model", mdl, "--conditions", a.conditions, "--sets", a.sets],
            stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    if any(codes):
        sys.exit(f"worker failed: {codes}, see {log_dir}")
    merge()


if __name__ == "__main__":
    main()
