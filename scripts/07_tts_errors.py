"""Follow-up: detect simulator-caused (TTS rendering) word errors automatically and measure how much
of the sim-to-real gap they explain. CPU only, reads results/utterances.csv (no ASR is run).

Flag rules (LibriSpeech subset, clean audio, word level, per voice):
- R3 (strict, as in v2): the reference word is wrong (substituted or deleted) for all 3 ASR models on
  the voice's clean clip, and right for all 3 on the same sentence's human clean clip.
- R2 (lenient): wrong for both strong models (Whisper, Parakeet) on the voice, right for both on human;
  wav2vec2 is ignored. R2 contains R3.
A word flagged for voice v on clean audio is removed from v and from human (same sentence) in every
channel condition, since all conditions are made from the same TTS clip.

Caller set (no human audio): a keyword instance (voice, script, keyword) is flagged when all models of
the rule miss it on the voice's clean clip while at least 4 of the 6 other voices have it recognized by
all models of the rule. The other voices stand in for the missing human reference.

Outputs: results/followups/tts_errors/ (CSV tables, figures, hand-check sample). Hand labels, written by
reading the reference text, the TTS input and the transcripts (nobody listened to audio), live in
hand_check_labels.csv and are merged if present.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jiwer  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sim2real import stats  # noqa: E402
from sim2real.asr import normalize  # noqa: E402
from sim2real.paths import DATA, RESULTS  # noqa: E402

_spec = importlib.util.spec_from_file_location("analyze", ROOT / "scripts" / "05_analyze.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

OUT = RESULTS / "followups" / "tts_errors"
SYNTH, RATE, MODELS, CONDS = A.SYNTH, A.RATE, A.MODELS, A.CONDS
WH, PK, W2V = "whisper-large-v3-turbo", "parakeet-tdt-0.6b-v2", "wav2vec2-base-960h"
RULES = {"R3": [WH, PK, W2V], "R2": [WH, PK]}
CLEAN_RULES = ["raw", "R3", "R2", "R3sym", "R2sym"]
SEED = 0
N_HAND = 30


def idx_set(s: str) -> set[int]:
    return set(map(int, s.split())) if s else set()


def align_tokens(ref_n: str, hyp_n: str) -> dict[int, str]:
    """ref word index -> what the hypothesis has there ('' for a deletion); correct words map to themselves."""
    r = ref_n.split()
    if not hyp_n:
        return {k: "" for k in range(len(r))}
    o = jiwer.process_words(ref_n, hyp_n)
    h = hyp_n.split()
    out = {}
    for ch in o.alignments[0]:
        for k in range(ch.ref_start_idx, ch.ref_end_idx):
            if ch.type == "delete":
                out[k] = ""
            else:
                out[k] = h[ch.hyp_start_idx + (k - ch.ref_start_idx)]
    return out


# --------------------------------------------------------------------------- flags

def librispeech_flags(u: pd.DataFrame, ids: list[str]) -> dict:
    """{rule: {voice: {id: set(word idx)}}}"""
    cl = u[(u["set"] == "librispeech") & (u["condition"] == "clean")]
    err = {(r.voice, r.id, r.model): idx_set(r.err_ref_idx) for r in cl.itertuples()}
    out = {}
    for rule, ms in RULES.items():
        out[rule], out[rule + "h"], out[rule + "sym"] = {}, {}, {}
        for v in SYNTH + RATE:
            fv, fh = {}, {}
            for i in ids:
                s = set.intersection(*[err[(v, i, m)] for m in ms])
                s -= set.union(*[err[("human", i, m)] for m in ms])
                if s:
                    fv[i] = s
                # mirror image: wrong for every model of the rule on human, right on the voice
                # (reader deviations from the text or reference errors); used as a selection-bias check
                h = set.intersection(*[err[("human", i, m)] for m in ms])
                h -= set.union(*[err[(v, i, m)] for m in ms])
                if h:
                    fh[i] = h
            out[rule][v] = fv
            out[rule + "h"][v] = fh
            out[rule + "sym"][v] = {i: fv.get(i, set()) | fh.get(i, set()) for i in set(fv) | set(fh)}
    return out


def keyword_instances(u: pd.DataFrame) -> pd.DataFrame:
    em = u[(u["set"] == "emergency") & (u["kw_total"] > 0) & u["category"].isin(A.CALL_CATS)]
    rows = []
    for r in em.itertuples():
        hit = set(r.kw_hit.split("|")) if r.kw_hit else set()
        for kw in r.kw_ref.split("|"):
            rows.append((r.id, kw, r.voice, r.condition, r.model, kw in hit, r.hyp, r.ref))
    return pd.DataFrame(rows, columns=["id", "keyword", "voice", "condition", "model", "hit", "hyp", "ref"])


def keyword_flags(inst: pd.DataFrame) -> dict:
    """{rule: set((voice, id, keyword))}"""
    cl = inst[inst["condition"] == "clean"]
    hit = cl.set_index(["voice", "id", "keyword", "model"])["hit"]
    keys = cl[["id", "keyword"]].drop_duplicates().itertuples(index=False)
    keys = list(keys)
    out = {}
    for rule, ms in RULES.items():
        fl = set()
        for i, k in keys:
            allhit = {v: all(hit[(v, i, k, m)] for m in ms) for v in SYNTH}
            allmiss = {v: not any(hit[(v, i, k, m)] for m in ms) for v in SYNTH}
            for v in SYNTH:
                if allmiss[v] and sum(allhit[o] for o in SYNTH if o != v) >= 4:
                    fl.add((v, i, k))
        out[rule] = fl
    return out


# --------------------------------------------------------------------------- cleaned error matrices

class Cleaner:
    def __init__(self, u: pd.DataFrame, L):
        self.L = L
        self.pos = {i: k for k, i in enumerate(L.ids)}
        ls = u[u["set"] == "librispeech"]
        self.err = {(r.voice, r.condition, r.model, r.id): idx_set(r.err_ref_idx) for r in ls.itertuples()}

    def errors(self, voice: str, cond: str, model: str, fv: dict) -> np.ndarray:
        e = self.L.E[(voice, cond, model)].copy()
        for i, s in fv.items():
            e[self.pos[i]] -= len(s & self.err.get((voice, cond, model, i), set()))
        return e

    def flagged_errors(self, voice: str, cond: str, model: str, fv: dict) -> np.ndarray:
        return self.L.E[(voice, cond, model)] - self.errors(voice, cond, model, fv)

    def n(self, fv: dict) -> np.ndarray:
        n = self.L.n.copy()
        for i, s in fv.items():
            n[self.pos[i]] -= len(s)
        return n


def union_flags(fdict: dict, voices: list[str]) -> dict:
    out: dict = {}
    for v in voices:
        for i, s in fdict[v].items():
            out[i] = out.get(i, set()) | s
    return out


def ci_ratio(num: np.ndarray, den: np.ndarray, W: np.ndarray):
    pt = num.sum() / den.sum() if den.sum() else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (W @ num) / (W @ den)
    b = b[np.isfinite(b)]
    return float(pt), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# --------------------------------------------------------------------------- analyses

def flag_summary(flags, C, L, u) -> pd.DataFrame:
    """Per rule x voice x model, clean audio: flagged words, share of synthetic WER and of the gap."""
    refs = u[u["set"] == "librispeech"].drop_duplicates("id").set_index("id")["ref"]
    rows = []
    for rule in RULES:
        fd = flags[rule]
        for v in SYNTH + RATE:
            fv = fd[v]
            words = []
            for i in sorted(fv):
                toks = normalize(refs[i]).split()
                words += [toks[k] for k in sorted(fv[i])]
            for m in MODELS:
                ev = L.E[(v, "clean", m)]
                eh = L.E[("human", "clean", m)]
                fe_v = C.flagged_errors(v, "clean", m, fv)
                fe_h = C.flagged_errors("human", "clean", m, fv)
                sh, sh_lo, sh_hi = ci_ratio(fe_v, ev, L.W)
                gap = A.paired_diff(ev, eh, L.n, L.W)
                gs, gs_lo, gs_hi = ci_ratio(fe_v - fe_h, ev - eh, L.W)
                gap_sig = A.excl0(gap["lo"], gap["hi"]) and gap["diff"] > 0
                rows.append({
                    "rule": rule, "voice": v, "model": m,
                    "n_flagged_words": int(sum(len(s) for s in fv.values())), "n_sentences": len(fv),
                    "n_mirror_human_side_words": int(sum(len(s) for s in flags[rule + "h"][v].values())),
                    "share_of_words": sum(len(s) for s in fv.values()) / L.n.sum(),
                    "wer": ev.sum() / L.n.sum(), "synth_errors": int(ev.sum()),
                    "errors_on_flagged_words": int(fe_v.sum()),
                    "human_errors_on_flagged_words": int(fe_h.sum()),
                    "share_of_synth_wer": sh, "share_lo": sh_lo, "share_hi": sh_hi,
                    "gap_raw": gap["diff"], "gap_raw_lo": gap["lo"], "gap_raw_hi": gap["hi"],
                    # only meaningful when the raw gap is positive and its CI excludes 0
                    "share_of_gap": gs if gap_sig else np.nan,
                    "share_of_gap_lo": gs_lo if gap_sig else np.nan,
                    "share_of_gap_hi": gs_hi if gap_sig else np.nan,
                    "examples": " ".join(words[:30]),
                })
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "flag_summary.csv", index=False, float_format="%.5f")
    return t


def gap_tables(flags, C, L) -> pd.DataFrame:
    """Synthetic minus human WER per voice x condition x model, raw and after removing flagged words."""
    rows = []
    for rule in CLEAN_RULES:
        for v in SYNTH + RATE:
            fv = {} if rule == "raw" else flags[rule][v]
            n = C.n(fv)
            for c in CONDS:
                for m in MODELS:
                    if not L.has((v, c, m), ("human", c, m)):
                        continue
                    ev, eh = C.errors(v, c, m, fv), C.errors("human", c, m, fv)
                    g = A.paired_diff(ev, eh, n, L.W)
                    rows.append({"rule": rule, "voice": v, "condition": c, "model": m,
                                 "wer_synth": ev.sum() / n.sum(), "wer_human": eh.sum() / n.sum(),
                                 "gap": g["diff"], "lo": g["lo"], "hi": g["hi"], "p": g["p"],
                                 "n_words": int(n.sum())})
    t = pd.DataFrame(rows)
    for rule, g in t[t["voice"].isin(SYNTH)].groupby("rule"):
        t.loc[g.index, "p_bh"] = stats.bh(g["p"])
    t.to_csv(OUT / "gap_cleaned.csv", index=False, float_format="%.5f")
    return t


def decisions(flags, C, L) -> pd.DataFrame:
    """Lab gap vs real gap between two ASR models (DiD), raw and cleaned. Holm within each model pair
    over the voice-level decisions, as in v2."""
    rows = []
    for rule in CLEAN_RULES:
        for a, b in itertools.combinations(MODELS, 2):
            for c in CONDS:
                if not L.has(("human", c, a), ("human", c, b)):
                    continue
                for v in SYNTH + ["synthetic_mean"]:
                    if v == "synthetic_mean":
                        fv = {} if rule == "raw" else union_flags(flags[rule], SYNTH)
                        if not all(L.has((s, c, a), (s, c, b)) for s in SYNTH):
                            continue
                        sa = np.mean([C.errors(s, c, a, fv) for s in SYNTH], axis=0)
                        sb = np.mean([C.errors(s, c, b, fv) for s in SYNTH], axis=0)
                    else:
                        fv = {} if rule == "raw" else flags[rule][v]
                        if not L.has((v, c, a), (v, c, b)):
                            continue
                        sa, sb = C.errors(v, c, a, fv), C.errors(v, c, b, fv)
                    ha, hb = C.errors("human", c, a, fv), C.errors("human", c, b, fv)
                    n = C.n(fv)
                    hd = A.paired_diff(ha, hb, n, L.W)
                    sd = A.paired_diff(sa, sb, n, L.W)
                    did = A.paired_diff(sa - sb, ha - hb, n, L.W)
                    h_sig = A.excl0(hd["lo"], hd["hi"])
                    rows.append({
                        "rule": rule, "model_a": a, "model_b": b, "condition": c, "voice": v,
                        "human_gap": hd["diff"], "human_gap_lo": hd["lo"], "human_gap_hi": hd["hi"],
                        "synth_gap": sd["diff"], "synth_gap_lo": sd["lo"], "synth_gap_hi": sd["hi"],
                        "did": did["diff"], "did_lo": did["lo"], "did_hi": did["hi"], "did_p": did["p"],
                        "flip": bool(h_sig and np.sign(sd["diff"]) != np.sign(hd["diff"])),
                        "gap_ratio_lab_over_real": sd["diff"] / hd["diff"] if h_sig else np.nan,
                    })
    t = pd.DataFrame(rows)
    for key, g in t[t["voice"] != "synthetic_mean"].groupby(["rule", "model_a", "model_b"]):
        t.loc[g.index, "did_p_holm"] = stats.holm(g["did_p"])
    t.to_csv(OUT / "decisions_cleaned.csv", index=False, float_format="%.5f")
    return t


def spread_and_damage(flags, C, L) -> tuple[pd.DataFrame, pd.DataFrame]:
    sp, dm = [], []
    for rule in CLEAN_RULES:
        for m in [WH, PK, W2V]:
            boots, pts = [], []
            for v in SYNTH:
                fv = {} if rule == "raw" else flags[rule][v]
                e, n = C.errors(v, "clean", m, fv), C.n(fv)
                pts.append(e.sum() / n.sum())
                boots.append((L.W @ e) / (L.W @ n))
            pts, boots = np.array(pts), np.stack(boots)
            rb = boots.max(0) / boots.min(0)
            sp.append({"rule": rule, "model": m, "min": pts.min(), "argmin": SYNTH[int(pts.argmin())],
                       "max": pts.max(), "argmax": SYNTH[int(pts.argmax())], "max_over_min": pts.max() / pts.min(),
                       "lo": float(np.percentile(rb, 2.5)), "hi": float(np.percentile(rb, 97.5))})
        for v in SYNTH:
            fv = {} if rule == "raw" else flags[rule][v]
            n = C.n(fv)
            for c in CONDS:
                for m in MODELS:
                    if c == "clean" or not L.has((v, c, m), ("human", c, m)):
                        continue
                    ds = C.errors(v, c, m, fv) - C.errors(v, "clean", m, fv)
                    dh = C.errors("human", c, m, fv) - C.errors("human", "clean", m, fv)
                    r = A.paired_diff(ds, dh, n, L.W)
                    dm.append({"rule": rule, "voice": v, "condition": c, "model": m,
                               "human_damage": dh.sum() / n.sum(), "synth_damage": ds.sum() / n.sum(),
                               "damage_error": r["diff"], "lo": r["lo"], "hi": r["hi"], "p": r["p"]})
    sp, dm = pd.DataFrame(sp), pd.DataFrame(dm)
    for rule, g in dm.groupby("rule"):
        dm.loc[g.index, "p_bh"] = stats.bh(g["p"])
    sp.to_csv(OUT / "voice_spread_cleaned.csv", index=False, float_format="%.5f")
    dm.to_csv(OUT / "channel_damage_cleaned.csv", index=False, float_format="%.5f")
    return sp, dm


def keyword_analysis(inst: pd.DataFrame, kflags: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # flagged instances with what each model heard (clean)
    cl = inst[inst["condition"] == "clean"].set_index(["voice", "id", "keyword", "model"])
    fl_rows = []
    for rule, fs in kflags.items():
        for (v, i, k) in sorted(fs):
            r = {"rule": rule, "voice": v, "id": i, "keyword": k, "ref": cl.loc[(v, i, k, WH), "ref"]}
            for m in MODELS:
                r[f"hyp_{m.split('-')[0]}"] = cl.loc[(v, i, k, m), "hyp"]
            r["heard_hell"] = any(" hell " in f" {normalize(r[f'hyp_{m.split('-')[0]}'])} " for m in RULES[rule])
            fl_rows.append(r)
    ft = pd.DataFrame(fl_rows)
    ft.to_csv(OUT / "keyword_flags.csv", index=False)

    rows = []
    for rule in ["raw", "R3", "R2"]:
        fs = set() if rule == "raw" else kflags[rule]
        keep = ~pd.Series([(v, i, k) in fs for v, i, k in zip(inst["voice"], inst["id"], inst["keyword"])],
                          index=inst.index)
        d = inst[keep]
        for (v, c, m), g in d.groupby(["voice", "condition", "model"]):
            k, n = int(g["hit"].sum()), len(g)
            lo, hi = stats.clopper_pearson(k, n)
            rows.append({"rule": rule, "voice": v, "condition": c, "model": m, "level": "voice",
                         "hits": k, "total": n, "recall": k / n, "lo": lo, "hi": hi})
        for (c, m), g in d.groupby(["condition", "model"]):
            Wc = stats.cluster_weights(g["id"].to_numpy(), 4000, seed=SEED)
            boot = (Wc @ g["hit"].to_numpy(float)) / Wc.sum(1)
            rows.append({"rule": rule, "voice": "pooled", "condition": c, "model": m, "level": "pooled",
                         "hits": int(g["hit"].sum()), "total": len(g), "recall": g["hit"].mean(),
                         "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5))})
        h = d[d["keyword"] == "help"]
        for (v, c, m), g in h.groupby(["voice", "condition", "model"]):
            rows.append({"rule": rule, "voice": v, "condition": c, "model": m, "level": "help",
                         "hits": int(g["hit"].sum()), "total": len(g), "recall": g["hit"].mean(),
                         "lo": np.nan, "hi": np.nan})
    kr = pd.DataFrame(rows)
    kr.to_csv(OUT / "keyword_recall_cleaned.csv", index=False, float_format="%.5f")

    # McNemar between voice pairs (clean, realistic); cleaned: drop a (script, keyword) pair when either
    # voice's instance is flagged. p_holm: Holm over the 126 voice-pair tests of a rule (3 models x 42),
    # which is what 05_analyze.py computes for v2; p_holm_per_model: Holm over each model's 42 tests.
    key = inst.set_index(["id", "keyword", "voice", "condition", "model"])["hit"]
    mc = []
    for rule in ["raw", "R3", "R2"]:
        fs = set() if rule == "raw" else kflags[rule]
        for m in MODELS:
            for c in ["clean", "realistic"]:
                for va, vb in itertools.combinations(SYNTH, 2):
                    try:
                        a = key.xs((va, c, m), level=["voice", "condition", "model"])
                        b = key.xs((vb, c, m), level=["voice", "condition", "model"])
                    except KeyError:
                        continue
                    a, b = a.align(b, join="inner")
                    ok = [((va, i, k) not in fs) and ((vb, i, k) not in fs) for i, k in a.index]
                    a, b = a[ok], b[ok]
                    b01, b10 = int((a & ~b).sum()), int((~a & b).sum())
                    mc.append({"rule": rule, "model": m, "condition": c, "a": va, "b": vb, "n_pairs": len(a),
                               "recall_a": a.mean(), "recall_b": b.mean(), "a_hit_b_miss": b01,
                               "a_miss_b_hit": b10, "p": stats.mcnemar(b01, b10)["p_value"]})
    mct = pd.DataFrame(mc)
    for key_, g in mct.groupby("rule"):
        mct.loc[g.index, "p_holm"] = stats.holm(g["p"])
    for key_, g in mct.groupby(["rule", "model"]):
        mct.loc[g.index, "p_holm_per_model"] = stats.holm(g["p"])
    mct.to_csv(OUT / "keyword_mcnemar_cleaned.csv", index=False, float_format="%.6g")
    return ft, kr, mct


# --------------------------------------------------------------------------- hand check

def hand_check_sample(flags, u) -> pd.DataFrame:
    cl = u[(u["set"] == "librispeech") & (u["condition"] == "clean")]
    hyp = {(r.voice, r.id, r.model): r.hyp for r in cl.itertuples()}
    sub = pd.read_csv(DATA / "corpus" / "librispeech_subset.csv").set_index("id")
    items = [(v, i, k) for v in SYNTH for i, s in flags["R2"][v].items() for k in sorted(s)]
    rng = np.random.default_rng(SEED)
    by_v = {v: [it for it in items if it[0] == v] for v in SYNTH}
    tot = len(items)
    alloc = {v: max(2, int(round(N_HAND * len(by_v[v]) / tot))) for v in SYNTH}
    while sum(alloc.values()) > N_HAND:
        vmax = max(alloc, key=alloc.get)
        alloc[vmax] -= 1
    while sum(alloc.values()) < N_HAND:
        vmax = max(SYNTH, key=lambda v: len(by_v[v]) - alloc[v])
        alloc[vmax] += 1
    rows = []
    for v in SYNTH:
        pick = rng.permutation(len(by_v[v]))[: alloc[v]]
        for j in sorted(pick):
            _, i, k = by_v[v][j]
            ref_n = normalize(sub.loc[i, "text"])
            toks = ref_n.split()
            r = {"voice": v, "id": i, "word_idx": k, "word": toks[k],
                 "context": " ".join(toks[max(0, k - 4):k] + [f"[{toks[k]}]"] + toks[k + 1:k + 5]),
                 "tts_input": sub.loc[i, "tts_text"], "in_R3": k in flags["R3"][v].get(i, set())}
            for m in MODELS:
                al = align_tokens(ref_n, normalize(hyp[(v, i, m)]))
                r[f"heard_{m.split('-')[0]}"] = al.get(k, "")
            for m in MODELS:
                r[f"hyp_{m.split('-')[0]}"] = hyp[(v, i, m)]
            r["hyp_human_whisper"] = hyp[("human", i, WH)]
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "hand_check_sample.csv", index=False)
    return t


def hand_check_precision(sample: pd.DataFrame) -> dict:
    p = OUT / "hand_check_labels.csv"
    if not p.exists():
        return {}
    lab = pd.read_csv(p)
    d = sample.merge(lab, on=["voice", "id", "word_idx"], how="left")
    assert d["label"].notna().all(), "unlabelled hand-check items"
    n = len(d)
    k = int((d["label"] == "tts").sum())
    unclear = int((d["label"] == "unclear").sum())
    lo, hi = stats.clopper_pearson(k, n)
    r = {"n": n, "tts": k, "unclear": unclear, "not_tts": int((d["label"] == "not_tts").sum()),
         "precision_strict": k / n, "lo": lo, "hi": hi}
    if n - unclear:
        lo2, hi2 = stats.clopper_pearson(k, n - unclear)
        r.update({"precision_excl_unclear": k / (n - unclear), "lo_excl_unclear": lo2, "hi_excl_unclear": hi2})
    k3 = d[d["in_R3"]]
    if len(k3):
        lo3, hi3 = stats.clopper_pearson(int((k3["label"] == "tts").sum()), len(k3))
        r.update({"R3_n": len(k3), "R3_tts": int((k3["label"] == "tts").sum()), "R3_lo": lo3, "R3_hi": hi3})
    d.to_csv(OUT / "hand_check.csv", index=False)
    return r


# --------------------------------------------------------------------------- figures

def fig_share(fs: pd.DataFrame) -> None:
    d = fs[fs["voice"].isin(SYNTH) & fs["model"].isin([WH, PK])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, m in zip(axes, [WH, PK]):
        y = np.arange(len(SYNTH))
        for j, (rule, col, lab) in enumerate([("R3", "#2a78d6", "R3: all 3 models wrong"),
                                              ("R2", "#eb6834", "R2: both strong models wrong")]):
            g = d[(d["model"] == m) & (d["rule"] == rule)].set_index("voice").reindex(SYNTH)
            yy = y + (j - 0.5) * 0.36
            ax.barh(yy, g["share_of_synth_wer"] * 100, height=0.32, color=col, label=lab)
            ax.errorbar(g["share_of_synth_wer"] * 100, yy,
                        xerr=[(g["share_of_synth_wer"] - g["share_lo"]) * 100, (g["share_hi"] - g["share_of_synth_wer"]) * 100],
                        fmt="none", ecolor=A.INK2, elinewidth=0.9)
        ax.set_title(A.MSHORT[m], fontsize=9, loc="left")
        ax.set_xlabel("share of the voice's clean WER on flagged words (%), 95% CI")
    axes[0].set_yticks(range(len(SYNTH)), [A.SHORT[v] for v in SYNTH])
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("How much of each synthetic voice's errors are likely TTS rendering errors (clean audio)",
                 fontsize=11, x=0.02, ha="left")
    fig.savefig(OUT / "fig1_flagged_share.png")
    plt.close(fig)


def fig_gaps(gt: pd.DataFrame) -> None:
    d = gt[(gt["condition"] == "clean") & gt["voice"].isin(SYNTH) & gt["model"].isin([WH, PK])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    cols = {"raw": "#9a9994", "R3": "#2a78d6", "R2": "#eb6834"}
    labs = {"raw": "all words", "R3": "R3 words removed", "R2": "R2 words removed"}
    for ax, m in zip(axes, [WH, PK]):
        for j, rule in enumerate(["raw", "R3", "R2"]):
            g = d[(d["model"] == m) & (d["rule"] == rule)].set_index("voice").reindex(SYNTH)
            y = np.arange(len(SYNTH)) + (j - 1) * 0.25
            ax.errorbar(g["gap"] * 100, y, xerr=[(g["gap"] - g["lo"]) * 100, (g["hi"] - g["gap"]) * 100],
                        fmt="o", ms=5, color=cols[rule], elinewidth=1.2, label=labs[rule])
        ax.axvline(0, color=A.INK, lw=0.8)
        ax.set_title(A.MSHORT[m], fontsize=9, loc="left")
        ax.set_xlabel("WER synthetic minus WER human (points), 95% CI")
    axes[0].set_yticks(range(len(SYNTH)), [A.SHORT[v] for v in SYNTH])
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Clean-audio gap per voice before and after removing flagged rendering errors",
                 fontsize=11, x=0.02, ha="left")
    fig.savefig(OUT / "fig2_gap_cleaned.png")
    plt.close(fig)


def fig_keywords(kr: pd.DataFrame) -> None:
    d = kr[(kr["level"] == "voice") & (kr["condition"] == "clean") & kr["model"].isin([WH, PK])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cols = {"raw": "#9a9994", "R2": "#eb6834"}
    labs = {"raw": "all keyword instances", "R2": "flagged instances removed (R2)"}
    for ax, m in zip(axes, [WH, PK]):
        for j, rule in enumerate(["raw", "R2"]):
            g = d[(d["model"] == m) & (d["rule"] == rule)].set_index("voice").reindex(SYNTH)
            y = np.arange(len(SYNTH)) + (j - 0.5) * 0.3
            ax.errorbar(g["recall"] * 100, y, xerr=[(g["recall"] - g["lo"]) * 100, (g["hi"] - g["recall"]) * 100],
                        fmt="o", ms=5, color=cols[rule], elinewidth=1.2, label=labs[rule])
        ax.set_title(A.MSHORT[m], fontsize=9, loc="left")
        ax.set_xlabel("keyword recall, clean audio (%), Clopper-Pearson 95% CI")
    axes[0].set_yticks(range(len(SYNTH)), [A.SHORT[v] for v in SYNTH])
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Emergency keyword recall per voice, before and after removing flagged rendering errors",
                 fontsize=11, x=0.02, ha="left")
    fig.savefig(OUT / "fig3_keyword_recall_cleaned.png")
    plt.close(fig)


# --------------------------------------------------------------------------- main

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    u = pd.read_csv(RESULTS / "utterances.csv", keep_default_na=False, dtype={"speaker": str})
    for c in ["errors", "n_ref", "kw_total", "kw_hits"]:
        u[c] = u[c].astype(int)
    L = A.LS(u[u["set"] == "librispeech"])
    C = Cleaner(u, L)
    flags = librispeech_flags(u, L.ids)
    fs = flag_summary(flags, C, L, u)
    gt = gap_tables(flags, C, L)
    dec = decisions(flags, C, L)
    sp, dm = spread_and_damage(flags, C, L)
    inst = keyword_instances(u)
    kflags = keyword_flags(inst)
    ft, kr, mct = keyword_analysis(inst, kflags)
    sample = hand_check_sample(flags, u)
    prec = hand_check_precision(sample)
    fig_share(fs)
    fig_gaps(gt)
    fig_keywords(kr)

    s = {"hand_check": prec, "n_keyword_flags": {r: len(v) for r, v in kflags.items()}}
    wp = dec[(dec["model_a"] == WH) & (dec["model_b"] == PK) & (dec["voice"] != "synthetic_mean")]
    s["whisper_vs_parakeet"] = {r: {"n": int(len(g)), "did_nominal": int((g["did_p"] < 0.05).sum()),
                                    "did_holm": int((g["did_p_holm"] < 0.05).sum()),
                                    "min_holm_p": float(g["did_p_holm"].min()), "flips": int(g["flip"].sum())}
                                for r, g in wp.groupby("rule")}
    lg = dec[(dec["model_b"] == W2V) & (dec["voice"] != "synthetic_mean")]
    s["large_gap_pairs"] = {r: {"ratio_min": float(g["gap_ratio_lab_over_real"].min()),
                                "ratio_max": float(g["gap_ratio_lab_over_real"].max()),
                                "did_holm": {f"{a}": int((gg["did_p_holm"] < 0.05).sum())
                                             for a, gg in g.groupby("model_a")},
                                "n": {f"{a}": int(len(gg)) for a, gg in g.groupby("model_a")}}
                            for r, g in lg.groupby("rule")}
    gm = gt[gt["voice"].isin(SYNTH)]
    s["gap_cells_bh_sig"] = {r: {"harder": int(((g["p_bh"] < 0.05) & (g["gap"] > 0)).sum()),
                                 "easier": int(((g["p_bh"] < 0.05) & (g["gap"] < 0)).sum()), "n": int(len(g))}
                             for r, g in gm.groupby("rule")}
    s["damage_cells_bh_sig"] = {r: int((g["p_bh"] < 0.05).sum()) for r, g in dm.groupby("rule")}
    s["kw_mcnemar_holm_sig"] = {f"{r}|{m}": [int((g["p_holm"] < 0.05).sum()), int((g["p_holm_per_model"] < 0.05).sum())]
                                for (r, m), g in mct.groupby(["rule", "model"])}
    (OUT / "summary.json").write_text(json.dumps(s, indent=1, default=float))
    print(json.dumps(s, indent=1, default=float))


if __name__ == "__main__":
    main()
