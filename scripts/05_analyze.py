"""Analyze results/utterances.csv into summary tables and charts (analysis v2).

Outputs (results/)
- tables/wer_by_cell.csv            WER per set x model x condition x voice, CI, gap vs human (raw and
                                    without TTS rendering errors), BH-adjusted p
- tables/gap_by_voice.csv           synthetic minus human WER per voice, wide view
- tables/voice_spread.csv           best vs worst voice per model x condition, CI of the max/min ratio
- tables/channel_damage.csv         lab vs real damage of each channel, BH-adjusted p
- tables/decision_flips.csv         per ASR pair x condition x voice: human gap, lab gap, difference in
                                    differences (DiD) with CI, Holm and BH within each model pair
- tables/same_family.csv            MMS-TTS (Meta) -> wav2vec2 (Meta) vs cross-family, absolute and log-ratio DiD
- tables/rate.csv                   speaking-rate confound: gap regressed on rate, default vs rate-matched voices
- tables/tts_rendering_errors.csv   words every ASR gets wrong on a voice's clean audio but none on human audio
- tables/keyword_recall.csv         keyword recall per voice x condition x model (emergency + fall_help sentences)
- tables/keyword_recall_pooled.csv  pooled over voices and per keyword, sentence-cluster bootstrap CI
- tables/keyword_mcnemar.csv        exact McNemar between voices and between channels, Holm per family
- tables/keyword_help.csv           "help" misses and what was heard instead
- tables/decision_level.csv         call-level detection (any keyword) and false alarms per category
- tables/ppi.csv                    finite-population PPI / PPI++ / cross-fit PPI++ vs human-only (t intervals)
- tables/snr_achieved.csv           post-channel SNR actually delivered, per condition x voice
- tables/hallucination.csv          insertion rate, long hypotheses, noise-only transcripts
- tables/bootstrap_coverage.csv     coverage of sentence vs speaker-cluster bootstrap CIs (simulation)
- figures/*.png
- summary.json                      headline numbers used in FINDINGS.md

WER is corpus WER (total edits / total reference words) after sim2real.asr.normalize.
LibriSpeech CIs use a speaker-cluster bootstrap (40 speakers resampled with replacement, all their
sentences kept together; 10k replicates, percentile). The emergency set has one item per sentence and
uses an ordinary bootstrap over sentences. p-values are CI-inversion bootstrap p-values,
2 * min(P(boot <= 0), P(boot >= 0)) with the (count + 1) / (B + 1) convention, so the smallest
reportable value is about 2e-4.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sim2real import stats
from sim2real.paths import RESULTS

TABLES = RESULTS / "tables"
FIGS = RESULTS / "figures"
B = 10000
SEED = 0
CONDS = ["clean", "phone", "g722-wb", "opus-nb", "phone+noise10", "phone+noise5",
         "phone+muffle", "phone+loss5", "opus-nb+loss5", "realistic"]
STRESS = {"phone+noise5", "phone+loss5", "opus-nb+loss5"}
SYNTH = ["kokoro:af_heart", "kokoro:am_michael", "piper:en_US-lessac-medium",
         "piper:en_US-ryan-medium", "mms:eng", "say:Samantha", "say:Daniel"]
RATE = [f"{v}@rate" for v in SYNTH]
SHORT = {"human": "human (LibriSpeech)", "kokoro:af_heart": "Kokoro af_heart",
         "kokoro:am_michael": "Kokoro am_michael", "piper:en_US-lessac-medium": "Piper lessac",
         "piper:en_US-ryan-medium": "Piper ryan", "mms:eng": "Meta MMS-TTS",
         "say:Samantha": "macOS Samantha", "say:Daniel": "macOS Daniel",
         "synthetic_mean": "mean of 7 TTS voices"}
SHORT.update({f"{k}@rate": f"{v} (rate-matched)" for k, v in list(SHORT.items())})
MODELS = ["whisper-large-v3-turbo", "parakeet-tdt-0.6b-v2", "wav2vec2-base-960h"]
MSHORT = {"whisper-large-v3-turbo": "Whisper large-v3-turbo",
          "parakeet-tdt-0.6b-v2": "Parakeet TDT 0.6B v2",
          "wav2vec2-base-960h": "wav2vec2-base-960h (in-domain control)"}
W2V = "wav2vec2-base-960h"
CALL_CATS = ("emergency", "fall_help")      # keyword recall is scored on these
BENIGN_CATS = ("fall_ok", "non_emergency")  # false alarms are scored on these
# Reference categorical palette (dataviz skill, light mode), fixed order.
C_MODEL = {"whisper-large-v3-turbo": "#2a78d6", "parakeet-tdt-0.6b-v2": "#eb6834",
           "wav2vec2-base-960h": "#1baf7a"}
INK, INK2, GRID, RED, PURPLE = "#0b0b0b", "#52514e", "#e4e3df", "#e34948", "#4a3aa7"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False,
})


# --------------------------------------------------------------------------- helpers

def ratio_ci(e, n, W):
    w = e.sum() / n.sum()
    boot = (W @ e) / (W @ n)
    return float(w), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), boot


def paired_diff(ea, eb, n, W, na=None, nb=None) -> dict:
    """Corpus WER(a) - WER(b) on the same sentences, ratio bootstrap with weights W.
    na/nb: optional per-sentence word counts when a and b exclude different words."""
    na = n if na is None else na
    nb = n if nb is None else nb
    d = ea.sum() / na.sum() - eb.sum() / nb.sum()
    boot = (W @ ea) / (W @ na) - (W @ eb) / (W @ nb)
    return {"diff": float(d), "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
            "p": stats.boot_p(boot), "boot": boot}


def excl0(lo, hi) -> bool:
    return bool(lo > 0 or hi < 0)


class LS:
    """LibriSpeech matrices: errors per (voice, condition, model) over a fixed sentence order."""

    def __init__(self, d: pd.DataFrame):
        self.ids = sorted(d["id"].unique())
        first = d.drop_duplicates("id").set_index("id").loc[self.ids]
        self.n = first["n_ref"].to_numpy(float)
        self.speaker = first["speaker"].astype(str).to_numpy()
        piv = d.pivot_table(index="id", columns=["voice", "condition", "model"], values="errors",
                            aggfunc="first").reindex(self.ids)
        self.E = {c: piv[c].to_numpy(float) for c in piv.columns if not piv[c].isna().any()}
        self.W = stats.cluster_weights(self.speaker, B, seed=SEED)
        self.Ws = stats.cluster_weights(np.arange(len(self.ids)), B, seed=SEED)  # sentence-level, for contrast

    def has(self, *keys):
        return all(k in self.E for k in keys)


def add_synth_mean(E: dict, voices: list[str]) -> None:
    for c, m in {(c, mo) for (_, c, mo) in E}:
        arrs = [E[(v, c, m)] for v in voices if (v, c, m) in E]
        if len(arrs) == len(voices):
            E[("synthetic_mean", c, m)] = np.mean(arrs, axis=0)


# --------------------------------------------------------------------------- TTS rendering errors

def rendering_errors(u: pd.DataFrame, L: LS) -> tuple[dict, pd.DataFrame]:
    """Words a TTS voice mis-rendered: on that voice's clean audio all three ASR models get the
    reference word wrong (substitution or deletion), on human clean audio none does.
    Returns {voice: {id: set(word idx)}} and a per-voice summary table."""
    ls = u[(u["set"] == "librispeech") & (u["condition"] == "clean")]
    errs = {(r.voice, r.id, r.model): set(map(int, r.err_ref_idx.split())) if r.err_ref_idx else set()
            for r in ls.itertuples()}
    flagged, rows = {}, []
    refs = ls.drop_duplicates("id").set_index("id")["ref"]
    from sim2real.asr import normalize
    for v in SYNTH + RATE:
        if not all((v, i, m) in errs for i in L.ids[:1] for m in MODELS):
            continue
        fv, words = {}, []
        for i in L.ids:
            s = set.intersection(*[errs.get((v, i, m), set()) for m in MODELS])
            s -= set.union(*[errs.get(("human", i, m), set()) for m in MODELS])
            if s:
                fv[i] = s
                toks = normalize(refs[i]).split()
                words += [toks[k] for k in sorted(s) if k < len(toks)]
        flagged[v] = fv
        rows.append({"voice": v, "n_flagged_words": sum(len(s) for s in fv.values()),
                     "n_sentences": len(fv), "share_of_words": sum(len(s) for s in fv.values()) / L.n.sum(),
                     "examples": " ".join(words[:25])})
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "tts_rendering_errors.csv", index=False, float_format="%.5f")
    return flagged, t


def adjusted(u: pd.DataFrame, L: LS, flagged: dict) -> dict:
    """For each synthetic voice v: errors of (v, c, m) and of (human, c, m) with the words flagged
    for v removed, and the reduced word count. {(v, c, m): (e_v_adj, e_h_adj, n_adj)}."""
    ls = u[u["set"] == "librispeech"]
    pos = {i: k for k, i in enumerate(L.ids)}
    err = {}
    for r in ls.itertuples():
        err[(r.voice, r.condition, r.model, r.id)] = set(map(int, r.err_ref_idx.split())) if r.err_ref_idx else set()
    out = {}
    for v, fv in flagged.items():
        nflag = np.zeros(len(L.ids))
        for i, s in fv.items():
            nflag[pos[i]] = len(s)
        for (vv, c, m) in list(L.E):
            if vv != v or ("human", c, m) not in L.E:
                continue
            ev, eh = L.E[(v, c, m)].copy(), L.E[("human", c, m)].copy()
            for i, s in fv.items():
                ev[pos[i]] -= len(s & err.get((v, c, m, i), set()))
                eh[pos[i]] -= len(s & err.get(("human", c, m, i), set()))
            out[(v, c, m)] = (ev, eh, L.n - nflag)
    return out


# --------------------------------------------------------------------------- WER tables

def wer_tables(u: pd.DataFrame, L: LS, adj: dict, summary: dict) -> pd.DataFrame:
    rows = []
    for s in ["librispeech", "emergency"]:
        d = u[u["set"] == s]
        if s == "librispeech":
            E, n, W = L.E, L.n, L.W
        else:
            ids = sorted(d["id"].unique())
            piv = d.pivot_table(index="id", columns=["voice", "condition", "model"], values="errors",
                                aggfunc="first").reindex(ids)
            E = {c: piv[c].to_numpy(float) for c in piv.columns if not piv[c].isna().any()}
            n = d.drop_duplicates("id").set_index("id").loc[ids, "n_ref"].to_numpy(float)
            W = stats.cluster_weights(np.arange(len(ids)), B, seed=SEED)
        for (v, c, m), e in sorted(E.items()):
            w, lo, hi, _ = ratio_ci(e, n, W)
            r = {"set": s, "model": m, "condition": c, "voice": v, "wer": w, "lo": lo, "hi": hi,
                 "n_utts": len(e), "n_words": int(n.sum()), "share_utts_perfect": float((e == 0).mean())}
            if s == "librispeech" and v != "human" and ("human", c, m) in E:
                g = paired_diff(e, E[("human", c, m)], n, W)
                gs = paired_diff(e, E[("human", c, m)], n, L.Ws)
                r.update({"gap_vs_human": g["diff"], "gap_lo": g["lo"], "gap_hi": g["hi"], "gap_p": g["p"],
                          "gap_lo_sentence_boot": gs["lo"], "gap_hi_sentence_boot": gs["hi"],
                          "ratio_vs_human": w / max(E[("human", c, m)].sum() / n.sum(), 1e-9)})
                if (v, c, m) in adj:
                    ev, eh, na = adj[(v, c, m)]
                    ga = paired_diff(ev, eh, n, W, na, na)
                    r.update({"gap_no_render": ga["diff"], "gap_no_render_lo": ga["lo"],
                              "gap_no_render_hi": ga["hi"], "gap_no_render_p": ga["p"]})
            rows.append(r)
    t = pd.DataFrame(rows)
    mask = t["gap_p"].notna()
    t.loc[mask, "gap_p_bh"] = stats.bh(t.loc[mask, "gap_p"])
    t.to_csv(TABLES / "wer_by_cell.csv", index=False, float_format="%.5f")

    ls = t[t["set"] == "librispeech"]
    g = ls[ls["voice"].isin(SYNTH + RATE + ["synthetic_mean"])].pivot_table(
        index=["model", "voice"], columns="condition", values="gap_vs_human")
    g = g.reindex(columns=[c for c in CONDS if c in g.columns])
    g.to_csv(TABLES / "gap_by_voice.csv", float_format="%.5f")

    # spread of synthetic voices with a bootstrap CI of the max/min ratio
    sp = []
    for (m, c), _ in ls.groupby(["model", "condition"]):
        keys = [(v, c, m) for v in SYNTH]
        if not all(k in L.E for k in keys):
            continue
        boots = np.stack([(L.W @ L.E[k]) / (L.W @ L.n) for k in keys])
        pts = np.array([L.E[k].sum() / L.n.sum() for k in keys])
        rb = boots.max(0) / np.maximum(boots.min(0), 1e-9)
        sp.append({"model": m, "condition": c, "min": pts.min(), "argmin": SYNTH[int(pts.argmin())],
                   "max": pts.max(), "argmax": SYNTH[int(pts.argmax())],
                   "human": L.E[("human", c, m)].sum() / L.n.sum() if ("human", c, m) in L.E else np.nan,
                   "max_over_min": pts.max() / pts.min(), "ratio_lo": float(np.percentile(rb, 2.5)),
                   "ratio_hi": float(np.percentile(rb, 97.5))})
    sp = pd.DataFrame(sp)
    sp.to_csv(TABLES / "voice_spread.csv", index=False, float_format="%.5f")
    summary["voice_spread"] = sp.to_dict("records")
    return t


# --------------------------------------------------------------------------- channel damage

def channel_damage(L: LS) -> pd.DataFrame:
    """damage = WER(condition) - WER(clean), per voice; error = synthetic damage - human damage."""
    rows = []
    for (v, c, m) in sorted(L.E):
        if v in ("human",) or c == "clean" or not L.has(("human", c, m), (v, "clean", m), ("human", "clean", m)):
            continue
        ds = L.E[(v, c, m)] - L.E[(v, "clean", m)]
        dh = L.E[("human", c, m)] - L.E[("human", "clean", m)]
        r = paired_diff(ds, dh, L.n, L.W)
        rows.append({"model": m, "condition": c, "voice": v, "stress": c in STRESS,
                     "human_damage": dh.sum() / L.n.sum(), "synth_damage": ds.sum() / L.n.sum(),
                     "damage_error": r["diff"], "lo": r["lo"], "hi": r["hi"], "p": r["p"]})
    t = pd.DataFrame(rows)
    main = t["voice"].isin(SYNTH)
    t.loc[main, "p_bh"] = stats.bh(t.loc[main, "p"])
    t.to_csv(TABLES / "channel_damage.csv", index=False, float_format="%.5f")
    return t


# --------------------------------------------------------------------------- decision flips

def flips(L: LS) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(MODELS, 2):
        for c in CONDS:
            if not L.has(("human", c, a), ("human", c, b)):
                continue
            hgap = L.E[("human", c, a)] - L.E[("human", c, b)]
            hd = paired_diff(L.E[("human", c, a)], L.E[("human", c, b)], L.n, L.W)
            for v in SYNTH + ["synthetic_mean"] + RATE:
                if not L.has((v, c, a), (v, c, b)):
                    continue
                sgap = L.E[(v, c, a)] - L.E[(v, c, b)]
                sd = paired_diff(L.E[(v, c, a)], L.E[(v, c, b)], L.n, L.W)
                did = paired_diff(sgap, hgap, L.n, L.W)
                h_sig = excl0(hd["lo"], hd["hi"])
                s_sig = excl0(sd["lo"], sd["hi"])
                opposite = np.sign(sd["diff"]) != np.sign(hd["diff"])
                rows.append({
                    "model_a": a, "model_b": b, "condition": c, "voice": v,
                    "kind": "mean" if v == "synthetic_mean" else ("rate" if v.endswith("@rate") else "voice"),
                    "human_gap": hd["diff"], "human_gap_lo": hd["lo"], "human_gap_hi": hd["hi"],
                    "human_gap_ci_excludes_0": h_sig,
                    "synth_gap": sd["diff"], "synth_gap_lo": sd["lo"], "synth_gap_hi": sd["hi"],
                    "synth_gap_ci_excludes_0": s_sig,
                    "did": did["diff"], "did_lo": did["lo"], "did_hi": did["hi"], "did_p": did["p"],
                    # a flip needs a real winner: human CI excludes 0 and the lab's point estimate disagrees
                    "flip": bool(h_sig and opposite),
                    "confident_flip": bool(h_sig and opposite and s_sig),
                    "share_sign_disagree": float((np.sign(sd["boot"]) != np.sign(hd["boot"])).mean()),
                    "gap_ratio_lab_over_real": sd["diff"] / hd["diff"] if h_sig else np.nan,
                })
    t = pd.DataFrame(rows)
    for (a, b), g in t[t["kind"] == "voice"].groupby(["model_a", "model_b"]):
        t.loc[g.index, "did_p_holm"] = stats.holm(g["did_p"])
        t.loc[g.index, "did_p_bh"] = stats.bh(g["did_p"])
    t.to_csv(TABLES / "decision_flips.csv", index=False, float_format="%.5f")
    return t


# --------------------------------------------------------------------------- same family

def same_family(L: LS) -> pd.DataFrame:
    rows = []
    n = L.n
    for c in CONDS:
        if not all(L.has((v, c, m)) for v in SYNTH + ["human"] for m in MODELS):
            continue
        gaps = {m: {v: (L.E[(v, c, m)].sum() - L.E[("human", c, m)].sum()) / n.sum() for v in SYNTH}
                for m in MODELS}
        for other in [m for m in MODELS if m != W2V]:
            def did_for(v):
                ea = L.E[(v, c, W2V)] - L.E[("human", c, W2V)]
                eb = L.E[(v, c, other)] - L.E[("human", c, other)]
                return paired_diff(ea, eb, n, L.W)

            def logdid_for(v):
                Wn = L.W @ n
                w = lambda vv, mm: (L.W @ L.E[(vv, c, mm)]) / Wn
                boot = (np.log(w(v, W2V) / w("human", W2V)) - np.log(w(v, other) / w("human", other)))
                pt = (np.log(L.E[(v, c, W2V)].sum() / L.E[("human", c, W2V)].sum())
                      - np.log(L.E[(v, c, other)].sum() / L.E[("human", c, other)].sum()))
                return float(pt), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

            d = did_for("mms:eng")
            ld = logdid_for("mms:eng")
            ref = [did_for(v)["diff"] for v in SYNTH if v != "mms:eng"]
            lref = [logdid_for(v)[0] for v in SYNTH if v != "mms:eng"]
            rows.append({
                "condition": c, "cross_model": other,
                "w2v_gap_mms": gaps[W2V]["mms:eng"], "cross_gap_mms": gaps[other]["mms:eng"],
                "did": d["diff"], "did_lo": d["lo"], "did_hi": d["hi"], "did_p": d["p"],
                "did_other_voices_median": float(np.median(ref)),
                "did_other_voices_min": min(ref), "did_other_voices_max": max(ref),
                "log_did": ld[0], "log_did_lo": ld[1], "log_did_hi": ld[2],
                "log_did_other_voices_median": float(np.median(lref)),
                "mms_rank_for_w2v_gap": int(sorted(gaps[W2V].values()).index(gaps[W2V]["mms:eng"]) + 1),
                "mms_rank_for_cross_gap": int(sorted(gaps[other].values()).index(gaps[other]["mms:eng"]) + 1),
            })
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "same_family.csv", index=False, float_format="%.5f")
    return t


# --------------------------------------------------------------------------- speaking rate

def rate_tables(u: pd.DataFrame, L: LS) -> pd.DataFrame:
    """(a) Regression of the per-sentence gap on the speaking-rate difference, pooled over voices,
    speaker-cluster bootstrap for the slope. (b) Default vs rate-matched voice gaps."""
    ls = u[u["set"] == "librispeech"]
    wps = ls.drop_duplicates(["id", "voice"]).set_index(["voice", "id"])["words_per_s"].astype(float)
    pos = np.arange(len(L.ids))
    rows = []
    for m in MODELS:
        for c in ["clean", "realistic"]:
            if ("human", c, m) not in L.E:
                continue
            X, Y, G = [], [], []
            for v in SYNTH:
                if (v, c, m) not in L.E:
                    continue
                dx = np.array([wps[(v, i)] - wps[("human", i)] for i in L.ids])
                dy = (L.E[(v, c, m)] - L.E[("human", c, m)]) / L.n
                X.append(dx), Y.append(dy), G.append(pos)
            X, Y, G = np.concatenate(X), np.concatenate(Y), np.concatenate(G)
            spk = L.speaker[G]
            beta = np.polyfit(X, Y, 1)[0]
            Wg = stats.cluster_weights(spk, 2000, seed=SEED)
            xm = (Wg @ X) / Wg.sum(1)
            ym = (Wg @ Y) / Wg.sum(1)
            sxy = (Wg @ (X * Y)) / Wg.sum(1) - xm * ym
            sxx = (Wg @ (X * X)) / Wg.sum(1) - xm ** 2
            bb = sxy / sxx
            rows.append({"analysis": "slope_gap_per_word_per_s", "model": m, "condition": c, "voice": "pooled",
                         "value": beta, "lo": float(np.percentile(bb, 2.5)), "hi": float(np.percentile(bb, 97.5))})
            for v in SYNTH:
                rv = f"{v}@rate"
                if not L.has((v, c, m), (rv, c, m)):
                    continue
                g0 = paired_diff(L.E[(v, c, m)], L.E[("human", c, m)], L.n, L.W)
                g1 = paired_diff(L.E[(rv, c, m)], L.E[("human", c, m)], L.n, L.W)
                dd = paired_diff(L.E[(rv, c, m)], L.E[(v, c, m)], L.n, L.W)
                rows.append({"analysis": "rate_matched", "model": m, "condition": c, "voice": v,
                             "gap_default": g0["diff"], "gap_default_lo": g0["lo"], "gap_default_hi": g0["hi"],
                             "gap_rate_matched": g1["diff"], "gap_rm_lo": g1["lo"], "gap_rm_hi": g1["hi"],
                             "value": dd["diff"], "lo": dd["lo"], "hi": dd["hi"], "p": dd["p"],
                             "wps_default": float(np.mean([wps[(v, i)] for i in L.ids])),
                             "wps_rate_matched": float(np.mean([wps[(rv, i)] for i in L.ids])),
                             "wps_human": float(np.mean([wps[("human", i)] for i in L.ids]))})
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "rate.csv", index=False, float_format="%.5f")
    return t


# --------------------------------------------------------------------------- keywords

def keyword_tables(u: pd.DataFrame, summary: dict):
    em_all = u[u["set"] == "emergency"].copy()
    em = em_all[(em_all["kw_total"] > 0) & em_all["category"].isin(CALL_CATS)].copy()
    benign_kw = em_all[(em_all["kw_total"] > 0) & em_all["category"].isin(BENIGN_CATS)]
    summary["keyword_instances_scored"] = int(em.drop_duplicates("id")["kw_total"].sum())
    summary["keyword_sentences_scored"] = int(em["id"].nunique())
    summary["keyword_instances_benign_excluded"] = int(benign_kw.drop_duplicates("id")["kw_total"].sum())

    # instances
    inst = []
    for r in em.itertuples():
        hit = set(r.kw_hit.split("|")) if r.kw_hit else set()
        for kw in r.kw_ref.split("|"):
            inst.append((r.id, kw, r.voice, r.condition, r.model, kw in hit, r.hyp))
    inst = pd.DataFrame(inst, columns=["id", "keyword", "voice", "condition", "model", "hit", "hyp"])

    rows = []
    for (v, c, m), g in inst.groupby(["voice", "condition", "model"]):
        k, nn = int(g["hit"].sum()), len(g)
        lo, hi = stats.clopper_pearson(k, nn)
        r = {"voice": v, "condition": c, "model": m, "hits": k, "total": nn, "misses": nn - k,
             "recall": k / nn, "lo": lo, "hi": hi, "n_sentences": g["id"].nunique()}
        if k == nn:
            r["miss_rate_upper95_rule_of_three"] = stats.rule_of_three(nn)
            r["miss_rate_upper95_exact_one_sided"] = stats.clopper_pearson_upper_zero(nn)
            r["miss_rate_upper95_two_sided_cp"] = 1 - lo
        rows.append(r)
    kr = pd.DataFrame(rows)
    kr.to_csv(TABLES / "keyword_recall.csv", index=False, float_format="%.5f")

    # pooled over voices: cluster bootstrap over sentence id (instances of one sentence share audio)
    def cluster_ci(g):
        sid = g["id"].to_numpy()
        hits = g["hit"].to_numpy(float)
        Wc = stats.cluster_weights(sid, 4000, seed=SEED)
        boot = (Wc @ hits) / Wc.sum(1)
        return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    pooled = []
    for (c, m), g in inst[inst["voice"].isin(SYNTH)].groupby(["condition", "model"]):
        lo, hi = cluster_ci(g)
        pooled.append({"level": "pooled_voices", "keyword": "ALL", "condition": c, "model": m,
                       "hits": int(g["hit"].sum()), "total": len(g), "recall": g["hit"].mean(), "lo": lo, "hi": hi})
    for (kw, c, m), g in inst[inst["voice"].isin(SYNTH)].groupby(["keyword", "condition", "model"]):
        lo, hi = cluster_ci(g) if g["id"].nunique() > 1 else stats.clopper_pearson(int(g["hit"].sum()), len(g))
        pooled.append({"level": "per_keyword", "keyword": kw, "condition": c, "model": m,
                       "hits": int(g["hit"].sum()), "total": len(g), "recall": g["hit"].mean(), "lo": lo, "hi": hi})
    kp = pd.DataFrame(pooled)
    kp.to_csv(TABLES / "keyword_recall_pooled.csv", index=False, float_format="%.5f")

    # "help": what is heard instead
    hp = inst[(inst["keyword"] == "help") & inst["voice"].isin(SYNTH)].copy()
    hp["heard_hell"] = hp["hyp"].str.lower().str.contains(r"\bhell\b", regex=True)
    hrows = []
    for (c, m, v), g in hp.groupby(["condition", "model", "voice"]):
        miss = g[~g["hit"]]
        hrows.append({"condition": c, "model": m, "voice": v, "help_total": len(g), "help_hits": int(g["hit"].sum()),
                      "help_misses": len(miss), "misses_with_hell": int(miss["heard_hell"].sum())})
    hk = pd.DataFrame(hrows)
    hk.to_csv(TABLES / "keyword_help.csv", index=False)

    # McNemar, paired by (sentence, keyword)
    key = inst.set_index(["id", "keyword", "voice", "condition", "model"])["hit"]
    voices = [v for v in SYNTH if v in set(inst["voice"])]
    mc = []

    def compare(kind, m, sel_a, sel_b, label_a, label_b):
        try:
            a = key.xs(sel_a, level=["voice", "condition", "model"])
            b = key.xs(sel_b, level=["voice", "condition", "model"])
        except KeyError:
            return
        a, b = a.align(b, join="inner")
        if len(a) == 0:  # a cell that was not run
            return
        b01 = int((a & ~b).sum())  # a hit, b missed
        b10 = int((~a & b).sum())  # a missed, b hit
        r = stats.mcnemar(b01, b10)
        mc.append({"comparison": kind, "model": m, "a": label_a, "b": label_b, "n_pairs": len(a),
                   "recall_a": float(a.mean()), "recall_b": float(b.mean()),
                   "a_hit_b_miss": b01, "a_miss_b_hit": b10, "p_value": r["p_value"]})

    for m in MODELS:
        for v in voices:
            for c in [c for c in CONDS if c != "clean"]:
                compare("channel", m, (v, "clean", m), (v, c, m), f"{v}|clean", f"{v}|{c}")
        for c in ["clean", "realistic"]:
            for va, vb in itertools.combinations(voices, 2):
                compare("voice", m, (va, c, m), (vb, c, m), f"{va}|{c}", f"{vb}|{c}")
    mct = pd.DataFrame(mc)
    for fam, g in mct.groupby("comparison"):
        mct.loc[g.index, "p_holm"] = stats.holm(g["p_value"])
    mct.to_csv(TABLES / "keyword_mcnemar.csv", index=False, float_format="%.6g")

    # call-level decision: any keyword detected in the hypothesis
    dl = []
    for (v, c, m), g in em_all.groupby(["voice", "condition", "model"]):
        for cat, gc in g.groupby("category"):
            if cat in CALL_CATS:
                # only scripts that contain a keyword can be detected by keyword spotting;
                # call scripts without one are counted in summary["call_scripts_without_keyword"]
                gc = gc[gc["kw_total"] > 0]
                k = int(gc["kw_detected_any"].sum())
                metric = "detected"
            else:
                gc = gc[gc["kw_total"] == 0]  # benign sentences with no keyword in the script
                k = int(gc["kw_detected_any"].sum())
                metric = "false_alarm"
            if len(gc) == 0:
                continue
            lo, hi = stats.clopper_pearson(k, len(gc))
            dl.append({"voice": v, "condition": c, "model": m, "category": cat, "metric": metric,
                       "k": k, "n": len(gc), "rate": k / len(gc), "lo": lo, "hi": hi})
    dlt = pd.DataFrame(dl)
    calls = em_all[em_all["category"].isin(CALL_CATS)].drop_duplicates("id")
    summary["call_scripts_without_keyword"] = calls.loc[calls["kw_total"] == 0, "ref"].tolist()
    dlt.to_csv(TABLES / "decision_level.csv", index=False, float_format="%.5f")
    return kr, kp, mct, hk, dlt


def librispeech_keywords(u: pd.DataFrame) -> pd.DataFrame:
    ls = u[(u["set"] == "librispeech") & (u["kw_total"] > 0)]
    t = ls.groupby(["voice", "condition", "model"])[["kw_hits", "kw_total"]].sum().reset_index()
    t["recall"] = t["kw_hits"] / t["kw_total"]
    t.to_csv(TABLES / "keyword_recall_librispeech.csv", index=False, float_format="%.4f")
    return t


# --------------------------------------------------------------------------- PPI

def ppi_table(L: LS, reps: int = 1000) -> pd.DataFrame:
    """Finite population of the 250 sentences: estimate the human corpus WER from the synthetic
    result on all 250 plus the human result on n sentences drawn without replacement.
    Per-sentence scores are errors / mean words, so their mean is the corpus WER."""
    mean_n = L.n.mean()
    N = len(L.n)
    rows = []
    rng = np.random.default_rng(SEED)
    draws = {k: np.array([rng.choice(N, size=k, replace=False) for _ in range(reps)]) for k in (10, 25, 50)}
    for m in MODELS:
        for c in ["clean", "phone", "phone+noise10", "phone+noise5", "realistic"]:
            if ("human", c, m) not in L.E:
                continue
            y_all = L.E[("human", c, m)] / mean_n
            truth = float(y_all.mean())
            for proxy in SYNTH + ["synthetic_mean"]:
                if (proxy, c, m) not in L.E:
                    continue
                f_all = L.E[(proxy, c, m)] / mean_n
                corr = float(np.corrcoef(y_all, f_all)[0, 1])
                for k, lab in draws.items():
                    res = {meth: stats.ppi_mean_finite(y_all[lab], f_all[lab], f_all.mean(), N, method=meth)
                           for meth in ("naive", "ppi", "ppi++", "ppi++cf")}
                    for meth, r in res.items():
                        rows.append({
                            "model": m, "condition": c, "proxy": proxy, "n_labeled": k, "method": meth,
                            "truth_human_wer": truth, "synth_only_wer": float(f_all.mean()),
                            "median_width": float(np.median(r["width"])),
                            "coverage": float(np.mean((r["lo"] <= truth) & (truth <= r["hi"]))),
                            "mean_abs_error": float(np.mean(np.abs(r["estimate"] - truth))),
                            "share_all_zero_labels": float(np.mean((y_all[lab] == 0).all(1))),
                            "median_lambda": float(np.median(r["lam"])),
                            "corr_human_synth": corr,
                        })
    t = pd.DataFrame(rows)
    naive = t[t["method"] == "naive"].set_index(["model", "condition", "proxy", "n_labeled"])["median_width"]
    t["width_vs_naive"] = [w / naive[(r.model, r.condition, r.proxy, r.n_labeled)]
                           for w, r in zip(t["median_width"], t.itertuples())]
    t.to_csv(TABLES / "ppi.csv", index=False, float_format="%.5f")
    return t


# --------------------------------------------------------------------------- channel checks, hallucination

def snr_achieved(u: pd.DataFrame) -> pd.DataFrame:
    d = u[u["snr_post_channel_db"].astype(str) != ""].copy()
    d["snr_post_channel_db"] = d["snr_post_channel_db"].astype(float)
    d["snr_fullband_whole_db"] = pd.to_numeric(d["snr_fullband_whole_db"], errors="coerce")
    d = d.drop_duplicates(["set", "id", "voice", "condition"])
    t = d.groupby(["condition", "voice"]).agg(
        post_channel_mean=("snr_post_channel_db", "mean"), post_channel_sd=("snr_post_channel_db", "std"),
        fullband_whole_mean=("snr_fullband_whole_db", "mean"), n=("id", "size")).reset_index()
    t.to_csv(TABLES / "snr_achieved.csv", index=False, float_format="%.3f")
    return t


def hallucination(u: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, m, c), g in u.groupby(["set", "model", "condition"]):
        rows.append({"kind": "utterances", "set": s, "model": m, "condition": c,
                     "insertion_rate": g["n_ins"].sum() / g["n_ref"].sum(),
                     "share_hyp_over_1_5x_ref": float((g["n_hyp"] > 1.5 * g["n_ref"]).mean()),
                     "max_hyp_over_ref": float((g["n_hyp"] / g["n_ref"]).max()), "n": len(g)})
    p = RESULTS / "noise_only.csv"
    if p.exists():
        nz = pd.read_csv(p, keep_default_na=False)
        for m, g in nz.groupby("model"):
            rows.append({"kind": "noise_only", "set": "noise", "model": m, "condition": "phone (noise only)",
                         "share_nonempty": float((g["n_words"] > 0).mean()),
                         "mean_words": float(g["n_words"].mean()), "max_words": int(g["n_words"].max()),
                         "n": len(g),
                         "examples": " | ".join(g.loc[g["n_words"] > 0, "hyp_norm"].astype(str).head(6))})
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "hallucination.csv", index=False, float_format="%.5f")
    return t


def bootstrap_coverage(L: LS, reps: int = 400, nb: int = 1000) -> pd.DataFrame:
    """Simulation: treat the 40 speakers as the population; draw 40 speakers with replacement,
    build a sentence-level and a speaker-cluster percentile CI, check coverage of the population value."""
    rng = np.random.default_rng(SEED + 1)
    spk_codes, uniq = stats._codes(L.speaker)
    k = len(uniq)
    rows = []
    cases = [("human WER", ("human", c, W2V), None) for c in ["clean", "realistic"]]
    cases += [("Whisper - Parakeet gap, human", ("human", c, MODELS[0]), ("human", c, MODELS[1]))
              for c in ["clean", "realistic"]]
    for label, ka, kb in cases:
        if ka not in L.E or (kb and kb not in L.E):
            continue
        e = L.E[ka] - (L.E[kb] if kb else 0)
        truth = e.sum() / L.n.sum()
        hit_s = hit_c = 0
        for _ in range(reps):
            pick = rng.integers(0, k, k)
            items = np.concatenate([np.flatnonzero(spk_codes == p) for p in pick])
            es, ns, sp = e[items], L.n[items], np.repeat(np.arange(k), [np.sum(spk_codes == p) for p in pick])
            Ws = stats.cluster_weights(np.arange(len(items)), nb, seed=int(rng.integers(1 << 30)))
            Wc = stats.cluster_weights(sp, nb, seed=int(rng.integers(1 << 30)))
            for W, name in ((Ws, "s"), (Wc, "c")):
                bt = (W @ es) / (W @ ns)
                ok = np.percentile(bt, 2.5) <= truth <= np.percentile(bt, 97.5)
                if name == "s":
                    hit_s += ok
                else:
                    hit_c += ok
        rows.append({"quantity": label, "condition": ka[1], "reps": reps,
                     "coverage_sentence_bootstrap": hit_s / reps, "coverage_speaker_bootstrap": hit_c / reps})
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "bootstrap_coverage.csv", index=False, float_format="%.3f")
    return t


# --------------------------------------------------------------------------- figures

def fig_wer_dots(t: pd.DataFrame) -> None:
    ls = t[(t["set"] == "librispeech")]
    order = ["human"] + SYNTH
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, m in zip(axes, MODELS):
        for j, (c, mk, fill) in enumerate((("clean", "o", True), ("realistic", "s", False))):
            d = ls[(ls["model"] == m) & (ls["condition"] == c)].set_index("voice").reindex(order)
            y = np.arange(len(order)) + (j - 0.5) * 0.28
            col = C_MODEL[m]
            ax.errorbar(d["wer"] * 100, y, xerr=[(d["wer"] - d["lo"]) * 100, (d["hi"] - d["wer"]) * 100],
                        fmt=mk, ms=6, color=col, mfc=col if fill else "white", mec=col, ecolor=col,
                        elinewidth=1.2, capsize=0, label=f"{c}")
        hv = ls[(ls["model"] == m) & (ls["voice"] == "human")].set_index("condition")["wer"]
        for c, ls_ in (("clean", "-"), ("realistic", "--")):
            if c in hv:
                ax.axvline(hv[c] * 100, color=INK2, lw=0.8, ls=ls_)
        ax.set_title(MSHORT[m], fontsize=9, color=INK)
        ax.set_xlabel("WER (%), 95% speaker-cluster bootstrap CI")
        ax.set_xlim(left=0)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_yticks(range(len(order)), [SHORT[v] for v in order])
    axes[0].invert_yaxis()
    fig.suptitle("Same 250 sentences: WER on real read speech vs 7 synthetic voices "
                 "(clean and realistic phone channel; gray lines = human)", fontsize=11, x=0.02, ha="left")
    fig.savefig(FIGS / "fig1_wer_by_voice.png")
    plt.close(fig)


def fig_wer_vs_condition(t: pd.DataFrame) -> None:
    ls = t[t["set"] == "librispeech"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, m in zip(axes, MODELS):
        d = ls[ls["model"] == m]
        conds = [c for c in CONDS if c in set(d["condition"])]
        x = np.arange(len(conds))
        for v in SYNTH:
            dv = d[d["voice"] == v].set_index("condition").reindex(conds)
            ax.plot(x, dv["wer"] * 100, color="#b9b8b2", lw=1, zorder=1)
        dm = d[d["voice"] == "synthetic_mean"].set_index("condition").reindex(conds)
        dh = d[d["voice"] == "human"].set_index("condition").reindex(conds)
        ax.plot(x, dm["wer"] * 100, color=C_MODEL[m], lw=2, marker="o", ms=4, label="mean of 7 TTS voices")
        ax.fill_between(x, dh["lo"] * 100, dh["hi"] * 100, color=INK, alpha=0.08, lw=0)
        ax.plot(x, dh["wer"] * 100, color=INK, lw=2, marker="o", ms=4, label="human (95% CI band)")
        ax.plot([], [], color="#b9b8b2", lw=1, label="each TTS voice")
        ax.set_xticks(x, [c + (" *" if c in STRESS else "") for c in conds], rotation=45, ha="right", fontsize=8)
        ax.set_title(MSHORT[m], fontsize=9)
        ax.set_ylabel("WER (%)")
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("WER by channel condition: human vs synthetic voices (LibriSpeech subset, n = 250; "
                 "* = stress condition)", fontsize=11, x=0.02, ha="left")
    fig.savefig(FIGS / "fig2_wer_by_condition.png")
    plt.close(fig)


def fig_gap_forest(t: pd.DataFrame) -> None:
    ls = t[(t["set"] == "librispeech") & (t["voice"].isin(SYNTH)) & (t["condition"] == "clean")]
    if "gap_no_render" not in ls.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (col, lo, hi, title) in zip(axes, [
            ("gap_vs_human", "gap_lo", "gap_hi", "all word errors"),
            ("gap_no_render", "gap_no_render_lo", "gap_no_render_hi",
             "without words every ASR misses on that voice only (TTS rendering errors)")]):
        for j, m in enumerate(MODELS):
            d = ls[ls["model"] == m].set_index("voice").reindex(SYNTH)
            y = np.arange(len(SYNTH)) + (j - 1) * 0.25
            ax.errorbar(d[col] * 100, y, xerr=[(d[col] - d[lo]) * 100, (d[hi] - d[col]) * 100],
                        fmt="o", ms=5, color=C_MODEL[m], elinewidth=1.2, label=MSHORT[m])
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_xlabel("WER synthetic minus WER human (points), 95% CI")
        ax.set_title(title, fontsize=9, loc="left")
    axes[0].set_yticks(range(len(SYNTH)), [SHORT[v] for v in SYNTH])
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Sim-to-real gap per voice, clean audio (right of 0: synthetic is harder)",
                 fontsize=11, x=0.02, ha="left")
    fig.savefig(FIGS / "fig3_gap_by_voice.png")
    plt.close(fig)


def fig_decision(fl: pd.DataFrame, a: str = "whisper-large-v3-turbo", b: str = "parakeet-tdt-0.6b-v2") -> None:
    """Difference in differences (lab gap minus real gap) for the closest pair of ASR models."""
    d = fl[(fl["model_a"] == a) & (fl["model_b"] == b) & (fl["kind"] == "voice")]
    conds = [c for c in CONDS if c in set(d["condition"])]
    fig, axes = plt.subplots(2, (len(conds) + 1) // 2, figsize=(3.0 * ((len(conds) + 1) // 2) + 1.5, 7.4),
                             sharey=True, squeeze=False)
    for ax, c in zip(axes.ravel(), conds):
        dc = d[d["condition"] == c].set_index("voice").reindex(SYNTH)
        h = dc.iloc[0]
        ax.axvline(0, color=INK2, lw=0.8)
        for i, v in enumerate(SYNTH):
            r = dc.loc[v]
            col = RED if r["did_p_holm"] < 0.05 else ("#2a78d6" if r["did_p"] < 0.05 else "#9a9994")
            ax.errorbar([r["did"] * 100], [i], xerr=[[(r["did"] - r["did_lo"]) * 100], [(r["did_hi"] - r["did"]) * 100]],
                        fmt="o", ms=5, color=col, elinewidth=1.2)
        sig = "real winner: " + ("Parakeet" if h["human_gap_lo"] > 0 else ("Whisper" if h["human_gap_hi"] < 0 else "none"))
        ax.set_title(f"{c}\nreal gap {h['human_gap']*100:+.2f} [{h['human_gap_lo']*100:+.2f}, "
                     f"{h['human_gap_hi']*100:+.2f}]\n{sig}", fontsize=8)
        ax.set_xlabel("lab gap minus real gap (points)", fontsize=8)
    for ax in axes.ravel()[len(conds):]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_yticks(range(len(SYNTH)), [SHORT[v] for v in SYNTH], fontsize=8)
    axes[0][0].invert_yaxis()
    fig.suptitle("Does a synthetic voice change the Whisper vs Parakeet comparison?\nDifference in differences "
                 "(gray: p >= 0.05; blue: nominal p < 0.05; red: Holm p < 0.05)",
                 fontsize=10, x=0.02, ha="left", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_decision_whisper_vs_parakeet.png")
    plt.close(fig)


def fig_keyword_recall(kr: pd.DataFrame) -> None:
    conds = [c for c in CONDS if c in set(kr["condition"])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
    for ax, m in zip(axes, MODELS):
        mat = kr[kr["model"] == m].pivot_table(index="voice", columns="condition",
                                               values="recall").reindex(index=SYNTH, columns=conds)
        mat = mat.dropna(axis=1, how="all")
        im = ax.imshow(mat.to_numpy() * 100, cmap="Blues_r", vmin=0, vmax=100, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.iat[i, j] * 100
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v < 55 else INK)
        ax.set_xticks(range(mat.shape[1]), list(mat.columns), rotation=45, ha="right", fontsize=8)
        ax.set_title(MSHORT[m], fontsize=9)
        ax.grid(False)
    axes[0].set_yticks(range(len(SYNTH)), [SHORT[v] for v in SYNTH], fontsize=8)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label("Emergency keyword recall (%)")
    n_kw = int(kr["total"].iloc[0])
    fig.suptitle(f"Emergency keyword recall by synthetic voice and channel ({n_kw} keyword instances per cell, "
                 "emergency and fall-help scripts; no human audio for this set)", fontsize=11, x=0.02, ha="left")
    fig.savefig(FIGS / "fig5_keyword_recall.png")
    plt.close(fig)


def fig_ppi(pp: pd.DataFrame) -> None:
    conds = ["clean", "realistic"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=False)
    styles = {"naive": ("human labels only", INK2, "o", "--"),
              "ppi": ("PPI (synthetic + human)", "#2a78d6", "s", "-"),
              "ppi++cf": ("PPI++ cross-fit", "#eb6834", "^", "-")}
    for r, c in enumerate(conds):
        for ax, m in zip(axes[r], MODELS):
            dm = pp[(pp["proxy"] == "synthetic_mean") & (pp["condition"] == c) & (pp["model"] == m)]
            if dm.empty:
                ax.set_visible(False)
                continue
            for meth, (lab, col, mk, ls_) in styles.items():
                x = dm[dm["method"] == meth].sort_values("n_labeled")
                ax.plot(x["n_labeled"], x["median_width"] * 100, color=col, marker=mk, ls=ls_, lw=2, ms=6, label=lab)
            ax.set_xticks([10, 25, 50])
            ax.set_xlabel("human-labeled sentences")
            ax.set_ylabel("median 95% CI width (WER points)")
            ax.set_ylim(bottom=0)
            cov = dm.pivot_table(index="method", columns="n_labeled", values="coverage")
            rho = dm["corr_human_synth"].iloc[0]
            ax.set_title(f"{MSHORT[m]}, {c} (corr {rho:.2f})\ncoverage n=10/25: human-only "
                         f"{cov.loc['naive', 10]:.0%}/{cov.loc['naive', 25]:.0%}, PPI {cov.loc['ppi', 10]:.0%}/"
                         f"{cov.loc['ppi', 25]:.0%}", fontsize=8)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Estimating human WER from 250 synthetic results plus n human-labeled sentences "
                 "(proxy: mean of 7 TTS voices; finite population, t intervals; 1000 draws)",
                 fontsize=11, x=0.02, ha="left", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_ppi_ci_width.png")
    plt.close(fig)


def fig_rate(rt: pd.DataFrame) -> None:
    d = rt[(rt["analysis"] == "rate_matched")]
    if d.empty:
        return
    conds = [c for c in ["clean", "realistic"] if c in set(d["condition"])]
    fig, axes = plt.subplots(1, len(conds), figsize=(6 * len(conds), 4.4), sharey=True, squeeze=False)
    for ax, c in zip(axes[0], conds):
        for j, m in enumerate(MODELS):
            dm = d[(d["model"] == m) & (d["condition"] == c)].set_index("voice").reindex(SYNTH)
            y = np.arange(len(SYNTH)) + (j - 1) * 0.25
            ax.scatter(dm["gap_default"] * 100, y, marker="o", facecolor="white", edgecolor=C_MODEL[m], s=28, zorder=3)
            ax.errorbar(dm["gap_rate_matched"] * 100, y, xerr=[(dm["gap_rate_matched"] - dm["gap_rm_lo"]) * 100,
                        (dm["gap_rm_hi"] - dm["gap_rate_matched"]) * 100], fmt="o", ms=5, color=C_MODEL[m],
                        elinewidth=1.1, label=MSHORT[m], zorder=4)
            for yy, a, b in zip(y, dm["gap_default"] * 100, dm["gap_rate_matched"] * 100):
                ax.plot([a, b], [yy, yy], color=C_MODEL[m], lw=0.8, alpha=0.6)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(f"{c}: open = default rate, filled = rate-matched to the human recording", fontsize=9, loc="left")
        ax.set_xlabel("WER synthetic minus WER human (points)")
    axes[0][0].set_yticks(range(len(SYNTH)), [SHORT[v] for v in SYNTH])
    axes[0][0].invert_yaxis()
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.savefig(FIGS / "fig7_rate_matched.png")
    plt.close(fig)


# --------------------------------------------------------------------------- main

def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    u = pd.read_csv(RESULTS / "utterances.csv", keep_default_na=False, dtype={"speaker": str})
    u["kw_detected_any"] = u["kw_detected_any"].astype(str) == "True"
    for c in ["errors", "n_ref", "n_ins", "n_hyp", "kw_total", "kw_hits"]:
        u[c] = u[c].astype(int)
    summary: dict = {"n_rows": len(u)}
    ls = u[u["set"] == "librispeech"]
    L = LS(ls)
    add_synth_mean(L.E, SYNTH)
    flagged, rend = rendering_errors(u, L)
    adj = adjusted(u, L, flagged)
    t = wer_tables(u, L, adj, summary)
    fl = flips(L)
    channel_damage(L)
    same_family(L)
    rt = rate_tables(u, L)
    kr, kp, mct, hk, dlt = keyword_tables(u, summary)
    librispeech_keywords(u)
    pp = ppi_table(L)
    snr_achieved(u)
    hallucination(u)
    bootstrap_coverage(L)
    fig_wer_dots(t)
    fig_wer_vs_condition(t)
    fig_gap_forest(t)
    fig_decision(fl)
    fig_keyword_recall(kr)
    fig_ppi(pp)
    fig_rate(rt)

    lsr = t[t["set"] == "librispeech"]
    summary["human_wer"] = lsr[lsr["voice"] == "human"][["model", "condition", "wer", "lo", "hi"]].to_dict("records")
    v = fl[fl["kind"] == "voice"]
    summary["decisions_voice_level"] = int(len(v))
    summary["did_nominal_p05"] = int((v["did_p"] < 0.05).sum())
    summary["did_holm_p05"] = int((v["did_p_holm"] < 0.05).sum())
    summary["flips"] = int(v["flip"].sum())
    summary["confident_flips"] = int(v["confident_flip"].sum())
    summary["cells_run"] = u.groupby("model")["condition"].nunique().to_dict()
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("tables:", sorted(p.name for p in TABLES.iterdir()))
    print("figures:", sorted(p.name for p in FIGS.iterdir()))


if __name__ == "__main__":
    main()
