"""Follow-up: phantom speech on non-speech audio, and clip-level guards against it.

Question 1. How often do the three ASR models write text when nobody speaks?
  300 non-speech clips of 3 to 10 s (sim2real.guards.nonspeech_plan: 6 ESC-50 kinds, line hiss,
  comfort noise, digital silence, near-silence; 30 each), each through the clean and the phone
  (G.711) channel: 600 clips per model.

Question 2. Can a guard remove the phantoms without dropping real words?
  Guards: py-webrtcvad (modes 0-3) and a simple energy VAD (seconds of detected speech),
  Whisper no_speech_prob and avg_logprob (and Whisper's own default rule), Parakeet token
  confidence, wav2vec2 CTC posterior confidence, and VAD + model-score combinations.
  Speech side: the real human clips of the main run (250 LibriSpeech, 250 FLEURS) and the
  122 caller scripts x 7 TTS voices, under phone+noise10 and realistic, prepared and degraded
  exactly as in scripts/04_run_experiment.py and scripts/08_fleurs.py (same seeds).

Steps
  features         build the clip list and the VAD features (CPU) -> raw/.../features.csv
  run --model M    decode every clip with scores (GPU), one shard per set x condition
  length --model M re-create the main run's 100-clip noise-only check at several exact lengths
  analyze          tables, figures, numbers for NOTES.md

Usage
  uv run python scripts/09_phantom.py features
  uv run python scripts/09_phantom.py run --model parakeet   # then wav2vec2, then whisper
  uv run python scripts/09_phantom.py length --model parakeet  # then wav2vec2, then whisper
  uv run python scripts/09_phantom.py analyze
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import soundfile as sf

from sim2real import asr, channel, data, fleurs, guards, stats
from sim2real.paths import RAW, RESULTS

OUT = RESULTS / "followups" / "phantom"
SHARDS = RAW / "followups" / "phantom"
MAIN_MANIFEST = ROOT / "data" / "corpus" / "manifest.csv"
FLEURS_MANIFEST = RESULTS / "followups" / "fleurs" / "manifest.csv"
SPEECH_CONDS = ["phone+noise10", "realistic"]
NS_CHANNELS = ["clean", "phone"]
MODELS = list(asr.MODELS)
SHORT = {asr.MODELS[0]: "whisper", asr.MODELS[1]: "wav2vec2", asr.MODELS[2]: "parakeet"}
B = 2000
SEED = 0
BUDGET = 0.0025  # max share of reference words lost on speech clips (set before any result)


def utt_seed(utt_id: str) -> int:  # identical to scripts/04_run_experiment.py
    return zlib.crc32(utt_id.encode()) & 0x7FFFFFFF


def read16k(path: str) -> np.ndarray:
    x, sr = sf.read(ROOT / path, dtype="float32")
    assert sr == channel.SR, (path, sr)
    return x.mean(axis=1) if x.ndim > 1 else x


# --------------------------------------------------------------------------- clip list

def clip_table() -> pd.DataFrame:
    """One row per (clip, channel/condition). Groups: nonspeech, ls_human, fl_human, callers."""
    rows = []
    for r in guards.nonspeech_plan():
        for ch in NS_CHANNELS:
            rows.append({"group": "nonspeech", "id": r["clip"], "voice": "", "cond": ch, "kind": r["kind"],
                         "cluster": r["clip"], "ref": "", "path": "", "level_db": r["level_db"],
                         "duration_s": r["duration_s"], "seed": r["seed"]})
    m = pd.read_csv(MAIN_MANIFEST, keep_default_na=False)
    ls = m[(m["set"] == "librispeech") & (m["voice"] == "human")]
    em = m[(m["set"] == "emergency") & ~m["voice"].str.endswith("@rate")]
    fl = pd.read_csv(FLEURS_MANIFEST, keep_default_na=False)
    fl = fl[fl["voice"] == "human"]
    for cond in SPEECH_CONDS:
        for _, r in ls.iterrows():
            rows.append({"group": "ls_human", "id": r["id"], "voice": "human", "cond": cond, "kind": "speech",
                         "cluster": f"ls{r['speaker']}", "ref": r["text"], "path": r["path"]})
        for _, r in fl.iterrows():
            rows.append({"group": "fl_human", "id": r["id"], "voice": "human", "cond": cond, "kind": "speech",
                         "cluster": r["id"], "ref": fleurs.scoring_ref(r["text"]), "path": r["path"]})
        for _, r in em.iterrows():
            rows.append({"group": "callers", "id": r["id"], "voice": r["voice"], "cond": cond,
                         "kind": "speech", "cluster": r["id"], "ref": r["text"], "path": r["path"],
                         "category": r["category"]})
    d = pd.DataFrame(rows)
    d["key"] = d["group"] + "|" + d["id"] + "|" + d["voice"] + "|" + d["cond"]
    assert d["key"].is_unique
    return d


def clip_audio(r) -> np.ndarray:
    """The exact array the ASR receives."""
    if r["group"] == "nonspeech":
        x = guards.nonspeech_clip({"kind": r["kind"], "duration_s": r["duration_s"],
                                   "level_db": r["level_db"], "seed": int(r["seed"])})
        return x if r["cond"] == "clean" else channel.telephone(x)
    s = utt_seed(r["id"])
    x = channel.prepare(read16k(r["path"]), seed=s)
    return channel.apply(r["cond"], x, seed=s, meta={})


# --------------------------------------------------------------------------- steps

def features() -> None:
    d = clip_table()
    SHARDS.mkdir(parents=True, exist_ok=True)
    t0, rows = time.time(), []
    for i, (_, r) in enumerate(d.iterrows()):
        x = clip_audio(r)
        v = guards.webrtc_voiced_s(x)
        rows.append({"key": r["key"], **{f"webrtc{m}_s": v[m] for m in v},
                     "energy_s": guards.energy_voiced_s(x), "audio_s": len(x) / channel.SR})
        if i % 500 == 0:
            print(f"features {i}/{len(d)} ({time.time() - t0:.0f} s)", flush=True)
    d.merge(pd.DataFrame(rows), on="key").to_csv(SHARDS / "features.csv", index=False)
    print(f"features: {len(d)} clips in {time.time() - t0:.0f} s")


def run(model: str) -> None:
    model = asr.resolve(model)
    d = clip_table()
    out = SHARDS / SHORT[model]
    out.mkdir(parents=True, exist_ok=True)
    for (g, cond), part in d.groupby(["group", "cond"], sort=False):
        shard = out / f"{g}__{cond}.csv"
        if shard.exists():
            print(f"[{model}] {g} {cond}: cached", flush=True)
            continue
        t0 = time.time()
        arrays = [clip_audio(r) for _, r in part.iterrows()]
        res = guards.scored(model, arrays)
        # does the scored decode give the same text as the main run's cache (same audio hash)?
        cached = [asr._cache_get(model, asr._key(model, asr._as16k(a).tobytes())) for a in arrays]
        rows = [{"key": k, **r, "cache_text": c if c is not None else ""
                 , "in_cache": c is not None} for k, r, c in zip(part["key"], res, cached)]
        tmp = shard.with_suffix(".tmp")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.replace(shard)
        print(f"[{model}] {g} {cond}: {len(rows)} clips in {time.time() - t0:.0f} s", flush=True)


# --------------------------------------------------------------------------- analysis helpers

def load_model(model: str, feats: pd.DataFrame) -> pd.DataFrame:
    parts = [pd.read_csv(p, keep_default_na=False) for p in sorted((SHARDS / SHORT[model]).glob("*__*.csv"))]
    t = pd.concat(parts, ignore_index=True).merge(feats, on="key", how="inner")
    assert len(t) == len(feats), (model, len(t), len(feats))
    for c in ("no_speech_prob", "avg_logprob", "compression_ratio", "conf_geo", "conf_min", "conf"):
        if c in t:  # NaN only when the transcript is empty (nothing to drop then)
            t[c] = pd.to_numeric(t[c], errors="coerce")
    if "avg_logprob" in t:
        t["avg_logprob"] = t["avg_logprob"].fillna(0.0)
        t["no_speech_prob"] = t["no_speech_prob"].fillna(0.0)
    t["hyp_norm"] = [asr.normalize(x) for x in t["text"]]
    t["nonempty"] = t["hyp_norm"].str.len() > 0
    refs = [asr.normalize(x) for x in t["ref"]]
    hyps = [asr.normalize(fleurs.scoring_ref(h)) if g == "fl_human" else h
            for h, g in zip(t["hyp_norm"], t["group"])]
    t["hyp_score"] = hyps
    n_ref, hits, errs = [], [], []
    import jiwer
    for r, h in zip(refs, hyps):
        if not r:
            n_ref.append(0); hits.append(0); errs.append(len(h.split()))
            continue
        if not h:
            n_ref.append(len(r.split())); hits.append(0); errs.append(len(r.split()))
            continue
        o = jiwer.process_words(r, h)
        n_ref.append(len(r.split())); hits.append(o.hits)
        errs.append(o.substitutions + o.deletions + o.insertions)
    t["n_ref"], t["hits"], t["errors"] = n_ref, hits, errs
    ref_kw = [data.find_keywords(r) if g == "callers" else [] for r, g in zip(t["ref"], t["group"])]
    hyp_kw = [data.find_keywords(h) for h in t["text"]]
    t["kw_ref"] = [len(k) for k in ref_kw]
    t["kw_hit"] = [len(set(k) & set(h)) for k, h in zip(ref_kw, hyp_kw)]
    t["kw_invented"] = ["; ".join(h) if g == "nonspeech" else "" for h, g in zip(hyp_kw, t["group"])]
    t["call_detected"] = [(len(k) > 0) and (len(h) > 0) for k, h in zip(ref_kw, hyp_kw)]
    t["fold"] = [zlib.crc32(c.encode()) % 2 for c in t["cluster"]]
    return t


def guard_scores(t: pd.DataFrame, model: str) -> dict[str, np.ndarray]:
    """Every 1-D guard as a score where low means 'no speech' (drop when score < thr)."""
    g = {f"webrtc mode {m}": t[f"webrtc{m}_s"].to_numpy(float) for m in range(4)}
    g["energy VAD"] = t["energy_s"].to_numpy(float)
    if model == asr.MODELS[0]:
        # large-v3-turbo's no_speech_prob is ~1e-10 on every clip, so it is used on a log scale
        g["no_speech_prob"] = -np.log10(t["no_speech_prob"].to_numpy(float) + 1e-30)
        g["avg_logprob"] = t["avg_logprob"].to_numpy(float)
    elif model == asr.MODELS[2]:
        g["token confidence"] = t["conf_geo"].to_numpy(float)
    else:
        g["CTC confidence"] = t["conf"].to_numpy(float)
    return g


def thresholds(s: np.ndarray, k: int = 120) -> np.ndarray:
    u = np.unique(np.quantile(s, np.linspace(0, 1, k)))
    mids = (u[1:] + u[:-1]) / 2 if len(u) > 1 else u
    return np.concatenate([[-np.inf], mids, [u[-1] + 1e-9]])


class Evaluator:
    """Phantom and speech-loss metrics of a drop mask, on a subset of rows."""

    def __init__(self, t: pd.DataFrame):
        self.ns = (t["group"] == "nonspeech").to_numpy()
        self.sp = ~self.ns
        self.hum = t["group"].isin(["ls_human", "fl_human"]).to_numpy()
        self.cal = (t["group"] == "callers").to_numpy()
        self.ne = t["nonempty"].to_numpy()
        self.n_ref = t["n_ref"].to_numpy(float)
        self.hits = t["hits"].to_numpy(float)
        self.errors = t["errors"].to_numpy(float)
        self.kw_ref = t["kw_ref"].to_numpy(float)
        self.kw_hit = t["kw_hit"].to_numpy(float)
        self.call = t["call_detected"].to_numpy()
        self.is_call = (t["kw_ref"] > 0).to_numpy()

    def metrics(self, drop: np.ndarray, rows: np.ndarray | None = None) -> dict:
        r = np.ones(len(drop), bool) if rows is None else rows
        ns, sp, hum, cal = self.ns & r, self.sp & r, self.hum & r, self.cal & r
        ph0 = self.ne[ns].mean()
        ph1 = (self.ne & ~drop)[ns].mean()
        lost = (self.hits * drop)
        return {
            "phantom_before": ph0, "phantom_after": ph1,
            "phantom_removed": 1 - ph1 / ph0 if ph0 > 0 else np.nan,
            "words_lost": lost[sp].sum() / self.n_ref[sp].sum(),
            "words_lost_human": lost[hum].sum() / self.n_ref[hum].sum(),
            "words_lost_callers": lost[cal].sum() / self.n_ref[cal].sum(),
            "speech_clips_dropped": drop[sp].mean(),
            "wer_before_human": self.errors[hum].sum() / self.n_ref[hum].sum(),
            "wer_after_human": (np.where(drop, self.n_ref, self.errors))[hum].sum() / self.n_ref[hum].sum(),
            "kw_lost": (self.kw_hit * drop)[cal].sum() / self.kw_ref[cal].sum(),
            "calls_lost": (self.call & drop)[cal & self.is_call].sum(),
            "calls_detected": self.call[cal & self.is_call].sum(),
        }


def candidates(t: pd.DataFrame, model: str) -> list[tuple[str, str, np.ndarray]]:
    """(family, setting, drop mask) for every guard family and threshold, incl. combinations."""
    gs = guard_scores(t, model)
    out = [("none", "", np.zeros(len(t), bool))]
    for name, s in gs.items():
        for th in thresholds(s):
            out.append((name, f"{th:.4g}", guards.drop_low(s, th)))
    model_keys = [k for k in gs if not (k.startswith("webrtc") or k == "energy VAD")]
    vad_keys = ["webrtc mode 3", "energy VAD"]
    for v in vad_keys:
        for mk in model_keys:
            for tv in thresholds(gs[v], 25):
                dv = guards.drop_low(gs[v], tv)
                for tm in thresholds(gs[mk], 25):
                    out.append((f"{v} OR {mk}", f"{tv:.4g} | {tm:.4g}", dv | guards.drop_low(gs[mk], tm)))
    if model == asr.MODELS[0]:
        nsp, lp = t["no_speech_prob"].to_numpy(float), t["avg_logprob"].to_numpy(float)
        out.append(("whisper default rule", "nsp>0.6 & logprob<-1", guards.whisper_builtin(nsp, lp)))
        for a in np.unique(np.quantile(nsp, np.linspace(0.02, 0.98, 25))):
            for b in np.linspace(-2.0, 0.0, 21):
                out.append(("no_speech_prob AND avg_logprob", f"nsp>{a:.3g} & lp<{b:.2f}",
                            guards.whisper_builtin(nsp, lp, a, b)))
    return out


def pick(ev: Evaluator, cands, rows: np.ndarray, budget: float = BUDGET):
    """Fewest remaining phantoms with words_lost <= budget on `rows`; ties -> fewer words lost."""
    best, key = None, None
    for fam, setting, drop in cands:
        m = ev.metrics(drop, rows)
        if m["words_lost"] > budget:
            continue
        k = (m["phantom_after"], m["words_lost"])
        if key is None or k < key:
            best, key = (fam, setting, drop), k
    return best


def boot_ci(ev: Evaluator, drop: np.ndarray, t: pd.DataFrame, keys: list[str]) -> dict:
    """Cluster bootstrap (non-speech clusters and speech clusters resampled independently)."""
    ns_idx = np.flatnonzero(ev.ns)
    sp_idx = np.flatnonzero(ev.sp)
    Wn = stats.cluster_weights(t["cluster"].to_numpy()[ns_idx], B, seed=SEED)
    Ws = stats.cluster_weights(t["cluster"].to_numpy()[sp_idx], B, seed=SEED + 1)
    ne, keep = ev.ne[ns_idx].astype(float), (ev.ne & ~drop)[ns_idx].astype(float)
    ph0 = Wn @ ne / Wn.sum(1)
    ph1 = Wn @ keep / Wn.sum(1)
    out = {"phantom_before": ph0, "phantom_after": ph1,
           "phantom_removed": 1 - ph1 / np.where(ph0 > 0, ph0, np.nan)}
    lost = (ev.hits * drop)[sp_idx]
    hum = ev.hum[sp_idx].astype(float)
    cal = ev.cal[sp_idx].astype(float)
    nref = ev.n_ref[sp_idx]
    out["words_lost"] = Ws @ lost / (Ws @ nref)
    out["words_lost_human"] = Ws @ (lost * hum) / (Ws @ (nref * hum))
    out["words_lost_callers"] = Ws @ (lost * cal) / (Ws @ (nref * cal))
    kwh, kwr = (ev.kw_hit * drop)[sp_idx], ev.kw_ref[sp_idx]
    out["kw_lost"] = Ws @ kwh / (Ws @ kwr)
    res = {}
    for k in keys:
        lo, hi = np.nanquantile(out[k], [0.025, 0.975])
        res[f"{k}_lo"], res[f"{k}_hi"] = float(lo), float(hi)
    return res


# --------------------------------------------------------------------------- analyze

def analyze() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    feats = pd.read_csv(SHARDS / "features.csv", keep_default_na=False)
    tabs = {SHORT[m]: load_model(m, feats) for m in MODELS}
    summary: dict = {"budget_words_lost": BUDGET}

    # ---- sanity: scored decode vs main-run cache
    agree = []
    for name, t in tabs.items():
        c = t[t["in_cache"].astype(str) == "True"]
        agree.append({"model": name, "clips_in_cache": len(c),
                      "same_text": int((c["text"] == c["cache_text"]).sum()),
                      "same_normalized": int((c["hyp_norm"] == [asr.normalize(x) for x in c["cache_text"]]).sum())})
    pd.DataFrame(agree).to_csv(OUT / "cache_agreement.csv", index=False)
    summary["cache_agreement"] = agree

    # ---- Q1 phantom rates
    rate_rows, phr_rows, kw_rows = [], [], []
    for name, t in tabs.items():
        ns = t[t["group"] == "nonspeech"]
        for (kind, ch), g in list(ns.groupby(["kind", "cond"])) + [(("ALL", ch), g) for ch, g in ns.groupby("cond")] \
                + [((k, "both"), g) for k, g in ns.groupby("kind")] + [(("ALL", "both"), ns)]:
            k, n = int(g["nonempty"].sum()), len(g)
            lo, hi = stats.clopper_pearson(k, n)
            rate_rows.append({"model": name, "kind": kind, "channel": ch, "n": n, "with_text": k,
                              "rate": k / n, "lo": lo, "hi": hi,
                              "mean_words": g["hyp_norm"].str.split().str.len().where(g["nonempty"]).mean()})
        cnt = Counter(ns.loc[ns["nonempty"], "hyp_norm"])
        for phrase, c in cnt.most_common(15):
            phr_rows.append({"model": name, "phrase": phrase, "count": c})
        inv = ns[ns["kw_invented"] != ""]
        for _, r in inv.iterrows():
            kw_rows.append({"model": name, "clip": r["id"], "kind": r["kind"], "channel": r["cond"],
                            "keywords": r["kw_invented"], "text": r["text"]})
    rates = pd.DataFrame(rate_rows)
    rates.to_csv(OUT / "phantom_rates.csv", index=False)
    pd.DataFrame(phr_rows).to_csv(OUT / "phantom_phrases.csv", index=False)
    pd.DataFrame(kw_rows, columns=["model", "clip", "kind", "channel", "keywords", "text"]).to_csv(
        OUT / "phantom_emergency_keywords.csv", index=False)
    # cluster bootstrap CI for overall rate (clean and phone versions of a clip are one cluster)
    summary["phantom_overall"] = {}
    for name, t in tabs.items():
        ns = t[t["group"] == "nonspeech"]
        W = stats.cluster_weights(ns["cluster"].to_numpy(), B, seed=SEED)
        b = W @ ns["nonempty"].to_numpy(float) / W.sum(1)
        lo, hi = np.quantile(b, [0.025, 0.975])
        summary["phantom_overall"][name] = {"rate": float(ns["nonempty"].mean()), "lo": float(lo), "hi": float(hi),
                                            "n": len(ns), "with_text": int(ns["nonempty"].sum()),
                                            "invented_keyword_clips": int((ns["kw_invented"] != "").sum())}

    # ---- Q2 guards: curves (in-sample), cross-fitted operating points with CIs
    curve_rows, op_rows, fam_rows, resid_rows = [], [], [], []
    for m in MODELS:
        name = SHORT[m]
        t = tabs[name].reset_index(drop=True)
        ev = Evaluator(t)
        cands = candidates(t, m)
        allrows = np.ones(len(t), bool)
        combo_pts: dict[str, list] = {}
        for fam, setting, drop in cands:
            mt = ev.metrics(drop)
            if " OR " in fam or " AND " in fam:
                combo_pts.setdefault(fam, []).append((mt["words_lost"], mt["phantom_after"], setting, mt))
                continue
            curve_rows.append({"model": name, "guard": fam, "setting": setting,
                               **{k: mt[k] for k in ("phantom_after", "phantom_removed", "words_lost",
                                                     "words_lost_human", "words_lost_callers", "kw_lost",
                                                     "calls_lost", "speech_clips_dropped")}})
        for fam, pts in combo_pts.items():  # 2-D guards: keep the in-sample Pareto front only
            best = np.inf
            for wl, pa, setting, mt in sorted(pts, key=lambda x: (x[0], x[1])):
                if pa < best - 1e-12:
                    best = pa
                    curve_rows.append({"model": name, "guard": fam, "setting": setting,
                                       **{k: mt[k] for k in ("phantom_after", "phantom_removed", "words_lost",
                                                             "words_lost_human", "words_lost_callers", "kw_lost",
                                                             "calls_lost", "speech_clips_dropped")}})
        fold = t["fold"].to_numpy()
        keys = ["phantom_before", "phantom_after", "phantom_removed", "words_lost", "words_lost_human",
                "words_lost_callers", "kw_lost"]
        families = sorted({c[0] for c in cands} - {"none", "whisper default rule"})

        def crossfit(cs):
            drop_cf = np.zeros(len(t), bool)
            chosen = []
            for f in (0, 1):
                p = pick(ev, cs, fold != f)
                chosen.append(p[0] + " @ " + p[1])
                drop_cf[fold == f] = p[2][fold == f]
            return drop_cf, chosen

        # one cross-fitted point per guard family (each family alone), and the best over all
        for fam in families + ["ALL"]:
            cs = [c for c in cands if c[0] == fam or c[0] == "none"] if fam != "ALL" else cands
            drop_cf, chosen = crossfit(cs)
            full = pick(ev, cs, allrows)
            mt = ev.metrics(drop_cf)
            row = {"model": name, "guard": fam, "applied_to_fold0": chosen[0], "applied_to_fold1": chosen[1],
                   "full_data_setting": full[0] + " @ " + full[1], **mt}
            row.update(boot_ci(ev, drop_cf, t, keys))
            (op_rows if fam == "ALL" else fam_rows).append(row)
            if fam == "ALL":
                ns = t["group"] == "nonspeech"
                for (kind, ch), g in t[ns].groupby(["kind", "cond"]):
                    ix = g.index.to_numpy()
                    resid_rows.append({"model": name, "kind": kind, "channel": ch, "n": len(ix),
                                       "with_text_before": int(t.loc[ix, "nonempty"].sum()),
                                       "with_text_after": int((t.loc[ix, "nonempty"].to_numpy() & ~drop_cf[ix]).sum())})
        # Whisper default rule as a fixed point (no tuning)
        if m == asr.MODELS[0]:
            d = [c for c in cands if c[0] == "whisper default rule"][0][2]
            row = {"model": name, "guard": "whisper default rule (fixed)", "applied_to_fold0": "", "applied_to_fold1": "",
                   "full_data_setting": "nsp>0.6 & logprob<-1", **ev.metrics(d)}
            row.update(boot_ci(ev, d, t, keys))
            fam_rows.append(row)
    pd.DataFrame(curve_rows).to_csv(OUT / "operating_curves.csv", index=False)
    fam = pd.DataFrame(fam_rows)
    fam.to_csv(OUT / "guards_crossfit.csv", index=False)
    ops = pd.DataFrame(op_rows)
    ops.to_csv(OUT / "recommended_points.csv", index=False)
    summary["recommended"] = ops.to_dict("records")
    pd.DataFrame(resid_rows).to_csv(OUT / "residual_phantoms_by_kind.csv", index=False)
    lv = []
    for name, t in tabs.items():
        e = t[(t["group"] == "nonspeech") & t["kind"].isin(list(guards.ESC_KINDS))].copy()
        e["level_bin"] = pd.cut(e["level_db"].astype(float), [-46, -41, -36, -31, -26], include_lowest=True).astype(str)
        for (b, ch), g in e.groupby(["level_bin", "cond"]):
            k, n = int(g["nonempty"].sum()), len(g)
            lo, hi = stats.clopper_pearson(k, n)
            lv.append({"model": name, "level_db_bin": b, "channel": ch, "n": n, "with_text": k, "rate": k / n,
                       "lo": lo, "hi": hi})
    pd.DataFrame(lv).to_csv(OUT / "phantom_by_level.csv", index=False)

    # per-clip table (small, for the record)
    keep = ["key", "group", "id", "voice", "cond", "kind", "cluster", "level_db", "duration_s", "webrtc0_s",
            "webrtc1_s", "webrtc2_s", "webrtc3_s", "energy_s", "audio_s"]
    per = tabs["whisper"][keep].copy()
    for name, t in tabs.items():
        t = t.set_index("key").loc[per["key"]]
        per[f"{name}_text"] = t["text"].to_numpy()
        for c in ("no_speech_prob", "avg_logprob", "conf_geo", "conf_min", "conf"):
            if c in t:
                per[f"{name}_{c}"] = t[c].astype(float).to_numpy()
        per[f"{name}_hits"] = t["hits"].to_numpy()
        per[f"{name}_n_ref"] = t["n_ref"].to_numpy()
    per.to_csv(OUT / "clips.csv.gz", index=False, compression="gzip")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    figures(rates, pd.DataFrame(curve_rows), fam, ops)
    lt = length_table()
    if len(lt):
        summary["phantom_by_exact_length"] = lt.to_dict("records")
        (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary["phantom_overall"], indent=1))
    print(ops[["model", "full_data_setting", "phantom_before", "phantom_after", "phantom_removed",
               "phantom_removed_lo", "phantom_removed_hi", "words_lost", "words_lost_lo", "words_lost_hi",
               "kw_lost", "calls_lost"]].to_string())


def figures(rates: pd.DataFrame, curves: pd.DataFrame, fam: pd.DataFrame, ops: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # fig1: phantom rate by kind and model, both channels pooled
    r = rates[(rates["channel"] == "both")].copy()
    kinds = [k for k in guards.KINDS] + ["ALL"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    cols = {"whisper": "#3b6fb6", "parakeet": "#d08a2e", "wav2vec2": "#6a9e5b"}
    w = 0.27
    for j, name in enumerate(["whisper", "parakeet", "wav2vec2"]):
        rr = r[r["model"] == name].set_index("kind").loc[kinds]
        x = np.arange(len(kinds)) + (j - 1) * w
        ax.bar(x, 100 * rr["rate"], w, color=cols[name], label=name)
        ax.errorbar(x, 100 * rr["rate"], yerr=[100 * (rr["rate"] - rr["lo"]), 100 * (rr["hi"] - rr["rate"])],
                    fmt="none", ecolor="#333", lw=0.8, capsize=2)
    ax.set_xticks(np.arange(len(kinds)))
    ax.set_xticklabels([k.replace("_", " ") for k in kinds], rotation=30, ha="right")
    ax.set_ylabel("clips with any text (%)")
    ax.set_title("Phantom text on non-speech clips (clean + phone channel, 60 clips per bar, 95 % CI)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_phantom_rate_by_kind.png", dpi=150)
    plt.close(fig)

    # fig2: operating curves per model
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    show = {"whisper": ["webrtc mode 3", "energy VAD", "no_speech_prob", "avg_logprob",
                        "energy VAD OR avg_logprob"],
            "parakeet": ["webrtc mode 3", "energy VAD", "token confidence"],
            "wav2vec2": ["webrtc mode 3", "energy VAD", "CTC confidence"]}
    pal = ["#3b6fb6", "#6a9e5b", "#d08a2e", "#9b4f96", "#c44e52"]
    for ax, name in zip(axes, ["whisper", "parakeet", "wav2vec2"]):
        for c, g in zip(pal, show[name]):
            cc = curves[(curves["model"] == name) & (curves["guard"] == g)].sort_values("words_lost")
            ax.plot(100 * cc["words_lost"], 100 * cc["phantom_after"], "-", color=c, lw=1.4, label=g)
        o = ops[ops["model"] == name].iloc[0]
        ax.errorbar([100 * o["words_lost"]], [100 * o["phantom_after"]],
                    xerr=[[100 * (o["words_lost"] - o["words_lost_lo"])], [100 * (o["words_lost_hi"] - o["words_lost"])]],
                    yerr=[[100 * (o["phantom_after"] - o["phantom_after_lo"])], [100 * (o["phantom_after_hi"] - o["phantom_after"])]],
                    fmt="o", color="black", ms=5, capsize=2, label="recommended (cross-fitted)")
        ax.axvline(100 * BUDGET, color="#999", ls=":", lw=1)
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlim(0, 100)
        ax.set_title(name)
        ax.set_xlabel("real words dropped on speech clips (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=7.5)
    axes[0].set_ylabel("non-speech clips still with text (%)")
    fig.suptitle("Guard operating curves (in-sample); dotted line = 0.25 % word-loss budget", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_operating_curves.png", dpi=150)
    plt.close(fig)


LENGTHS = [4.0, 4.01, 4.02, 4.1, 4.5, 5.0, 5.37]


def length_check(model: str) -> None:
    """Does the phantom rate depend on the exact clip length? Re-creates the main run's noise-only
    check (scripts/04_run_experiment.noise_only: 100 clips, street/rain/crowd/home in turn, -36 dB
    after the phone line, seeds 10000+k) at several lengths; 4.00 s is the main run's setting."""
    model = asr.resolve(model)
    shard = SHARDS / SHORT[model] / "length_check.csv"
    if shard.exists():
        print(f"[{model}] length check: cached")
        return
    groups = ["street", "rain", "crowd", "home"]
    rows, arrays = [], []
    for dur in LENGTHS:
        for k in range(100):
            g = groups[k % len(groups)]
            bed = channel.noise_bed(g, int(round(dur * channel.SR)), seed=10_000 + k)
            lvl = channel.db(channel.power(channel.phone_8k(bed)))
            bed = bed * 10 ** ((-36.0 - lvl) / 20) + channel._floor_noise(len(bed), k)
            arrays.append(channel.telephone(bed.astype(np.float32)))
            rows.append({"duration_s": dur, "clip": k, "noise_group": g})
    res = guards.scored(model, arrays)
    d = pd.DataFrame(rows)
    d["text"] = [r["text"] for r in res]
    d.to_csv(shard, index=False)
    print(f"[{model}] length check: {len(d)} clips")


def length_table() -> pd.DataFrame:
    out = []
    for m in MODELS:
        p = SHARDS / SHORT[m] / "length_check.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, keep_default_na=False)
        d["ne"] = [bool(asr.normalize(x)) for x in d["text"]]
        base = d[d["duration_s"] == 4.0].set_index("clip")["ne"]
        for dur, g in d.groupby("duration_s"):
            k, n = int(g["ne"].sum()), len(g)
            lo, hi = stats.clopper_pearson(k, n)
            ne = g.set_index("clip")["ne"].loc[base.index]
            mc = stats.mcnemar(int((ne & ~base).sum()), int((~ne & base).sum()))
            out.append({"model": SHORT[m], "duration_s": dur, "n": n, "with_text": k, "rate": k / n,
                        "lo": lo, "hi": hi,
                        "thank_you": int((g["text"].map(asr.normalize) == "thank you").sum()),
                        "vs_4s_text_only_here": mc["b01"], "vs_4s_text_only_at_4s": mc["b10"],
                        "mcnemar_p_vs_4s": mc["p_value"]})
    t = pd.DataFrame(out)
    t.to_csv(OUT / "phantom_by_exact_length.csv", index=False)
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["features", "run", "length", "analyze"])
    ap.add_argument("--model")
    a = ap.parse_args()
    if a.step == "features":
        features()
    elif a.step == "run":
        run(a.model)
    elif a.step == "length":
        length_check(a.model)
    else:
        analyze()


if __name__ == "__main__":
    main()
