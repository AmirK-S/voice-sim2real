"""Run the full grid: every voice x every channel condition x every ASR model.

Sets
- librispeech: 250 sentences, voice `human` (real read speech) + 7 TTS voices.
- emergency:   122 written caller sentences, 7 TTS voices (no human audio).

Every clip first goes through sim2real.channel.prepare (edge silence trimmed, 250 ms
lead and trail, -26 dBov active level, -65 dBFS noise floor), then the channel
condition is applied on the fly (sim2real.channel.apply) with a seed derived from the
sentence id only, so the same sentence gets the same noise bed, noise offset and loss
pattern under every voice: comparisons stay paired. Per-clip channel measurements
(post-channel SNR, noise group, loss rate, Opus bitrate) are kept in the shards.

Groups
- main: every voice except the rate-matched variants, both sets, every condition.
- rate: the "<voice>@rate" variants (scripts/01b_rate_matched.py), LibriSpeech only;
  run with --group rate --conditions clean,realistic.
- noise-only: --noise-only transcribes 100 clips of pure noise through the phone line
  (hallucination check), written to <model>/noise_only.csv.

Transcripts are cached by audio hash (sim2real.asr), so the script is safe to
stop and re-run. Raw transcripts land in results/raw/transcripts/<model>/<condition>.csv;
the merge step scores them into results/utterances.csv (one row per
set x sentence x voice x condition x model). v1 shards (channel v1) are kept in
results/raw/transcripts_v1/ for the record and are not merged.

Usage
  python scripts/04_run_experiment.py                 # all models in parallel, then merge
  python scripts/04_run_experiment.py --model whisper # one worker
  python scripts/04_run_experiment.py --model whisper --group rate --conditions clean,realistic
  python scripts/04_run_experiment.py --model whisper --noise-only
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
TRANSCRIPTS = RAW / "transcripts_v2"
# Priority order: the most informative conditions first, so a partial run is still useful.
CONDITION_ORDER = ["clean", "phone", "realistic", "phone+noise10", "phone+noise5", "opus-nb",
                   "g722-wb", "phone+muffle", "phone+loss5", "opus-nb+loss5"]
META_COLS = ["noise_group", "snr_post_channel_db", "snr_mic_inband_db", "snr_fullband_whole_db",
             "loss_rate", "opus_kbps"]
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


def select(m: pd.DataFrame, group: str, sets: list[str]) -> pd.DataFrame:
    rate = m["voice"].str.endswith("@rate")
    if group == "rate":
        return m[rate & (m["set"] == "librispeech")]
    return m[~rate & m["set"].isin(sets)]


def worker(model: str, conditions: list[str], sets: list[str], group: str = "main") -> None:
    model = asr.resolve(model)
    m = select(load_manifest(), group, sets)
    out_dir = TRANSCRIPTS / model
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    suffix = "" if group == "main" else f"@{group}"
    for cond in conditions:
        shard = out_dir / f"{cond}{suffix}.csv"
        if shard.exists():
            done = pd.read_csv(shard, keep_default_na=False)
            if len(done) == len(m):
                print(f"[{model}] {cond}: already done", flush=True)
                continue
        t0, rows = time.time(), []
        for (s, voice), g in m.groupby(["set", "voice"], sort=True):
            arrays, metas = [], []
            for p, i in zip(g["path"], g["id"]):
                meta: dict = {}
                x = channel.prepare(read16k(p), seed=utt_seed(i))
                arrays.append(channel.apply(cond, x, seed=utt_seed(i), meta=meta))
                metas.append(meta)
            hyps = asr.transcribe_arrays(arrays, model)
            rows += [{"set": s, "id": i, "voice": voice, "hyp": h, **{k: mt.get(k, "") for k in META_COLS}}
                     for i, h, mt in zip(g["id"], hyps, metas)]
        tmp = shard.with_suffix(".tmp")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.replace(shard)
        print(f"[{model}] {cond}: {len(rows)} clips in {time.time() - t0:.0f} s "
              f"(total {time.time() - t_all:.0f} s)", flush=True)


def noise_only(model: str, n: int = 100) -> None:
    """Hallucination check: pure noise (no speech) through the phone line.

    The noise is set to the level it has in phone+noise10 (speech at -26 dBov, 10 dB
    post-channel SNR), i.e. -36 dB post-channel power, 4 s per clip, groups street, rain,
    crowd and home in turn. Any non-empty transcript is a hallucination."""
    model = asr.resolve(model)
    groups = ["street", "rain", "crowd", "home"]
    arrays, recs = [], []
    for k in range(n):
        g = groups[k % len(groups)]
        bed = channel.noise_bed(g, 4 * channel.SR, seed=10_000 + k)
        lvl = channel.db(channel.power(channel.phone_8k(bed)))
        bed = bed * 10 ** ((-36.0 - lvl) / 20) + channel._floor_noise(len(bed), k)
        arrays.append(channel.telephone(bed.astype(np.float32)))
        recs.append({"clip": k, "noise_group": g})
    hyps = asr.transcribe_arrays(arrays, model)
    out = TRANSCRIPTS / model
    out.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(recs)
    d["hyp"] = hyps
    d.to_csv(out / "noise_only.csv", index=False)
    print(f"[{model}] noise-only: {sum(bool(asr.normalize(h)) for h in hyps)}/{n} non-empty")


def _kw_hits(ref_kws: list[str], hyp: str) -> list[str]:
    found = set(data.find_keywords(hyp))
    return [k for k in ref_kws if k in found]


def audio_stats(m: pd.DataFrame) -> pd.DataFrame:
    """Per clip (raw TTS or human file): speech duration with edge silence trimmed, active level,
    in-band (300-3400 Hz) share of speech energy. Cached in data/corpus/audio_stats.csv."""
    path = ROOT / "data" / "corpus" / "audio_stats.csv"
    have = pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame(
        columns=["set", "id", "voice", "speech_s", "raw_level_db", "inband_share"])
    key = set(zip(have["set"], have["id"], have["voice"]))
    new = []
    for r in m.itertuples():
        if (r.set, r.id, r.voice) in key:
            continue
        x = read16k(r.path)
        s, e = channel.trim_bounds(x)
        xs = x[s:e].astype(np.float64)
        bl = channel._bandlimit_16k(xs)
        new.append({"set": r.set, "id": r.id, "voice": r.voice, "speech_s": (e - s) / channel.SR,
                    "raw_level_db": channel.active_level(xs)[0],
                    "inband_share": channel.power(bl) / max(channel.power(xs), 1e-20)})
    if new:
        have = pd.concat([have, pd.DataFrame(new)], ignore_index=True)
        have.to_csv(path, index=False)
    have["speech_s"] = have["speech_s"].astype(float)
    return have


def merge() -> pd.DataFrame:
    import jiwer

    m = load_manifest()
    stats_ = audio_stats(m)
    shards, noise = [], []
    for model_dir in sorted(TRANSCRIPTS.iterdir()):
        for f in sorted(model_dir.glob("*.csv")):
            d = pd.read_csv(f, keep_default_na=False)
            d["model"] = model_dir.name
            if f.stem == "noise_only":
                noise.append(d)
                continue
            d["condition"] = f.stem.split("@")[0]
            shards.append(d)
    if noise:
        nz = pd.concat(noise, ignore_index=True)
        nz["hyp_norm"] = [asr.normalize(h) for h in nz["hyp"]]
        nz["n_words"] = [len(h.split()) for h in nz["hyp_norm"]]
        nz.to_csv(RESULTS / "noise_only.csv", index=False)
    t = pd.concat(shards, ignore_index=True)
    t = t.merge(m[["set", "id", "voice", "text", "keywords", "category", "speaker", "duration_s"]],
                on=["set", "id", "voice"], how="left", validate="many_to_one")
    assert t["text"].notna().all()
    t = t.rename(columns={"text": "ref"})

    ref_n = {r: asr.normalize(r) for r in t["ref"].unique()}
    errs, nref, nins, nhyp, eidx = [], [], [], [], []
    for r, h in zip(t["ref"], t["hyp"]):
        rn, hn = ref_n[r], asr.normalize(h)
        n = len(rn.split())
        if not hn:
            e, ins, bad = n, 0, list(range(n))
        else:
            o = jiwer.process_words(rn, hn)
            e, ins = o.substitutions + o.deletions + o.insertions, o.insertions
            bad = [k for ch in o.alignments[0] if ch.type in ("substitute", "delete")
                   for k in range(ch.ref_start_idx, ch.ref_end_idx)]
        errs.append(e)
        nref.append(n)
        nins.append(ins)
        nhyp.append(len(hn.split()))
        eidx.append(" ".join(map(str, bad)))
    t["errors"], t["n_ref"], t["n_ins"], t["n_hyp"], t["err_ref_idx"] = errs, nref, nins, nhyp, eidx
    t = t.merge(stats_, on=["set", "id", "voice"], how="left", validate="many_to_one")
    t["words_per_s"] = t["n_ref"] / t["speech_s"]
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
            "speech_s", "words_per_s", "ref", "hyp", "errors", "n_ref", "n_ins", "n_hyp", "err_ref_idx",
            "wer", "kw_ref", "kw_hit", "kw_total", "kw_hits", "kw_detected_any"] + META_COLS
    for c in META_COLS:
        if c not in t.columns:
            t[c] = ""
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
    ap.add_argument("--group", default="main", choices=["main", "rate"])
    ap.add_argument("--noise-only", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args()
    conditions = a.conditions.split(",")
    sets = a.sets.split(",")
    if a.merge_only:
        merge()
        return
    if a.model and a.noise_only:
        noise_only(a.model)
        return
    if a.model:
        worker(a.model, conditions, sets, a.group)
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
