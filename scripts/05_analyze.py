"""Analyze results/utterances.csv into summary tables and charts.

Outputs (results/)
- tables/wer_by_cell.csv          WER per set x model x condition x voice, bootstrap CI, paired gap vs human
- tables/gap_by_voice.csv         synthetic minus human WER per voice, wide view
- tables/decision_flips.csv       for each ASR pair: winner on human audio vs winner under each voice
- tables/same_family.csv          MMS-TTS (Meta) -> wav2vec2 (Meta) vs cross-family pairs
- tables/keyword_recall.csv       emergency keyword recall per voice x condition x model, exact CI, rule of three
- tables/keyword_recall_pooled.csv recall pooled over voices, and per keyword
- tables/keyword_mcnemar.csv      exact McNemar between key conditions and between voices
- tables/ppi.csv                  PPI / PPI++ estimates of human WER from synthetic + few human labels
- figures/*.png
- summary.json                    headline numbers used in FINDINGS.md

WER is corpus WER (total edits / total reference words) after sim2real.asr.normalize.
Paired CIs resample sentences with all conditions kept together (percentile, 10k replicates).
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
CONDS = ["clean", "phone", "phone+noise20", "phone+noise10", "phone+noise5",
         "phone+loss5", "phone+loss10", "phone+muffle", "realistic"]
SYNTH = ["kokoro:af_heart", "kokoro:am_michael", "piper:en_US-lessac-medium",
         "piper:en_US-ryan-medium", "mms:eng", "say:Samantha", "say:Daniel"]
SHORT = {"human": "human (LibriSpeech)", "kokoro:af_heart": "Kokoro af_heart",
         "kokoro:am_michael": "Kokoro am_michael", "piper:en_US-lessac-medium": "Piper lessac",
         "piper:en_US-ryan-medium": "Piper ryan", "mms:eng": "Meta MMS-TTS",
         "say:Samantha": "macOS Samantha", "say:Daniel": "macOS Daniel",
         "synthetic_mean": "mean of 7 TTS voices"}
MODELS = ["whisper-large-v3-turbo", "parakeet-tdt-0.6b-v2", "wav2vec2-base-960h"]
MSHORT = {"whisper-large-v3-turbo": "Whisper large-v3-turbo",
          "parakeet-tdt-0.6b-v2": "Parakeet TDT 0.6B",
          "wav2vec2-base-960h": "wav2vec2-base (Meta)"}
# Reference categorical palette (dataviz skill, light mode), fixed order.
C_MODEL = {"whisper-large-v3-turbo": "#2a78d6", "parakeet-tdt-0.6b-v2": "#eb6834",
           "wav2vec2-base-960h": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False,
})


# --------------------------------------------------------------------------- helpers

def ratio_ci(e: np.ndarray, n: np.ndarray, idx: np.ndarray) -> tuple[float, float, float]:
    w = e.sum() / n.sum()
    boot = e[idx].sum(1) / n[idx].sum(1)
    return float(w), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_diff(ea: np.ndarray, eb: np.ndarray, n: np.ndarray, idx: np.ndarray) -> dict:
    """Corpus WER(a) - WER(b), same sentences, ratio bootstrap over sentences."""
    d = (ea.sum() - eb.sum()) / n.sum()
    boot = (ea[idx].sum(1) - eb[idx].sum(1)) / n[idx].sum(1)
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"diff": float(d), "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)), "p_two_sided": float(min(1.0, p))}


def fmt_ci(v, lo, hi, pct=True):
    k = 100 if pct else 1
    return f"{v * k:.1f} [{lo * k:.1f}, {hi * k:.1f}]"


def matrices(df: pd.DataFrame, value: str) -> tuple[list[str], dict, np.ndarray]:
    """{(voice, condition, model): array over sentences in a fixed id order}."""
    ids = sorted(df["id"].unique())
    piv = df.pivot_table(index="id", columns=["voice", "condition", "model"], values=value,
                         aggfunc="first").reindex(ids)
    n = df.drop_duplicates("id").set_index("id").loc[ids, "n_ref"].to_numpy(float)
    return ids, {c: piv[c].to_numpy(float) for c in piv.columns}, n


def add_synth_mean(E: dict, voices: list[str]) -> None:
    for c, m in {(c, mo) for (_, c, mo) in E}:
        arrs = [E[(v, c, m)] for v in voices if (v, c, m) in E]
        if len(arrs) == len(voices):
            E[("synthetic_mean", c, m)] = np.mean(arrs, axis=0)


# --------------------------------------------------------------------------- WER tables

def wer_tables(u: pd.DataFrame, summary: dict) -> tuple[pd.DataFrame, dict, np.ndarray]:
    rows = []
    out_E = None
    for s in ["librispeech", "emergency"]:
        d = u[u["set"] == s]
        if d.empty:
            continue
        ids, E, n = matrices(d, "errors")
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(ids), size=(B, len(ids)))
        voices = sorted({v for v, _, _ in E})
        if s == "librispeech":
            add_synth_mean(E, SYNTH)
            out_E, out_n, out_idx = E, n, idx
        for (v, c, m), e in sorted(E.items()):
            w, lo, hi = ratio_ci(e, n, idx)
            uw = e / n
            r = {"set": s, "model": m, "condition": c, "voice": v, "wer": w, "lo": lo, "hi": hi,
                 "n_utts": len(ids), "n_words": int(n.sum()), "median_utt_wer": float(np.median(uw)),
                 "share_utts_perfect": float((e == 0).mean())}
            if s == "librispeech" and v != "human" and ("human", c, m) in E:
                pd_ = paired_diff(e, E[("human", c, m)], n, idx)
                r.update({"gap_vs_human": pd_["diff"], "gap_lo": pd_["lo"], "gap_hi": pd_["hi"],
                          "gap_p": pd_["p_two_sided"],
                          "ratio_vs_human": w / max(E[("human", c, m)].sum() / n.sum(), 1e-9)})
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "wer_by_cell.csv", index=False, float_format="%.5f")

    ls = t[t["set"] == "librispeech"]
    g = ls[ls["voice"] != "human"].pivot_table(index=["model", "voice"], columns="condition",
                                               values="gap_vs_human").reindex(columns=CONDS)
    g["mean_over_conditions"] = g.mean(axis=1)
    g.to_csv(TABLES / "gap_by_voice.csv", float_format="%.5f")

    # spread of synthetic voices: how much does the vendor choice move the lab number?
    sp = ls[ls["voice"].isin(SYNTH)].groupby(["model", "condition"])["wer"].agg(["min", "max"])
    hum = ls[ls["voice"] == "human"].set_index(["model", "condition"])["wer"]
    sp["human"] = hum
    sp["max_over_min"] = sp["max"] / sp["min"]
    sp.to_csv(TABLES / "voice_spread.csv", float_format="%.5f")
    summary["voice_spread"] = sp.reset_index().to_dict("records")
    return t, (out_E, out_n, out_idx)


# --------------------------------------------------------------------------- channel damage

def channel_damage(E: dict, n: np.ndarray, idx: np.ndarray) -> pd.DataFrame:
    """Does the lab measure the same damage from a channel as real speech does?
    damage = WER(condition) - WER(clean), per voice; error = synthetic damage - human damage."""
    rows = []
    for (v, c, m) in sorted(E):
        if v == "human" or c == "clean" or ("human", c, m) not in E or (v, "clean", m) not in E:
            continue
        ds = E[(v, c, m)] - E[(v, "clean", m)]
        dh = E[("human", c, m)] - E[("human", "clean", m)]
        r = paired_diff(ds, dh, n, idx)
        rows.append({"model": m, "condition": c, "voice": v,
                     "human_damage": dh.sum() / n.sum(), "synth_damage": ds.sum() / n.sum(),
                     "damage_error": r["diff"], "lo": r["lo"], "hi": r["hi"], "p_two_sided": r["p_two_sided"]})
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "channel_damage.csv", index=False, float_format="%.5f")
    return t


# --------------------------------------------------------------------------- decision flips

def flips(E: dict, n: np.ndarray, idx: np.ndarray) -> pd.DataFrame:
    mean_n = n.mean()
    rows = []
    for a, b in itertools.combinations(MODELS, 2):
        for c in CONDS:
            if ("human", c, a) not in E or ("human", c, b) not in E:
                continue
            sc = {"human": {a: E[("human", c, a)] / mean_n, b: E[("human", c, b)] / mean_n}}
            for v in SYNTH + ["synthetic_mean"]:
                if (v, c, a) in E and (v, c, b) in E:
                    sc[v] = {a: E[(v, c, a)] / mean_n, b: E[(v, c, b)] / mean_n}
            r = stats.decision_flip(sc, systems=(a, b), n=B, seed=SEED)
            h = r["human"]
            hd = paired_diff(E[("human", c, a)], E[("human", c, b)], n, idx)
            for v, s in r["conditions"].items():
                sd = paired_diff(E[(v, c, a)], E[(v, c, b)], n, idx)
                rows.append({
                    "model_a": a, "model_b": b, "condition": c, "voice": v,
                    "human_wer_a": h["mean"][a], "human_wer_b": h["mean"][b],
                    "human_gap": hd["diff"], "human_gap_lo": hd["lo"], "human_gap_hi": hd["hi"],
                    "human_winner": h["winner"], "human_p_winner_stable": h["p_winner_stable"],
                    "synth_wer_a": s["mean"][a], "synth_wer_b": s["mean"][b],
                    "synth_gap": sd["diff"], "synth_gap_lo": sd["lo"], "synth_gap_hi": sd["hi"],
                    "synth_winner": s["winner"], "flip": s["flip"], "p_flip": s["p_flip"],
                    "synth_gap_ci_excludes_0": bool(sd["lo"] > 0 or sd["hi"] < 0),
                    "human_gap_ci_excludes_0": bool(hd["lo"] > 0 or hd["hi"] < 0),
                })
    t = pd.DataFrame(rows)
    # A "confident flip": both gaps individually significant and in opposite directions.
    t["confident_flip"] = t["flip"] & t["synth_gap_ci_excludes_0"] & t["human_gap_ci_excludes_0"]
    # "False certainty": the lab declares a significant winner where real speech shows no clear one.
    t["false_certainty"] = t["synth_gap_ci_excludes_0"] & ~t["human_gap_ci_excludes_0"]
    # "Missed difference": real speech separates the two models, the lab cannot.
    t["missed_difference"] = ~t["synth_gap_ci_excludes_0"] & t["human_gap_ci_excludes_0"]
    # Size distortion: lab gap divided by real gap (same sign > 1 = lab exaggerates).
    t["gap_ratio_lab_over_real"] = t["synth_gap"] / t["human_gap"].where(t["human_gap"] != 0)
    t.to_csv(TABLES / "decision_flips.csv", index=False, float_format="%.5f")
    return t


# --------------------------------------------------------------------------- same family

def same_family(E: dict, n: np.ndarray, idx: np.ndarray) -> pd.DataFrame:
    w2v = "wav2vec2-base-960h"
    rows = []
    for c in CONDS:
        if not all((v, c, m) in E for v in SYNTH + ["human"] for m in MODELS):
            continue
        gaps = {}
        for m in MODELS:
            gaps[m] = {v: (E[(v, c, m)].sum() - E[("human", c, m)].sum()) / n.sum() for v in SYNTH}
        for other in [m for m in MODELS if m != w2v]:
            # difference in differences: (w2v on MMS - w2v on human) - (other on MMS - other on human)
            ea = E[("mms:eng", c, w2v)] - E[("human", c, w2v)]
            eb = E[("mms:eng", c, other)] - E[("human", c, other)]
            did = paired_diff(ea, eb, n, idx)
            # same contrast for every other voice, as a reference distribution
            ref = [((E[(v, c, w2v)] - E[("human", c, w2v)]).sum()
                    - (E[(v, c, other)] - E[("human", c, other)]).sum()) / n.sum()
                   for v in SYNTH if v != "mms:eng"]
            rows.append({
                "condition": c, "cross_model": other,
                "w2v_gap_mms": gaps[w2v]["mms:eng"], "cross_gap_mms": gaps[other]["mms:eng"],
                "did": did["diff"], "did_lo": did["lo"], "did_hi": did["hi"], "did_p": did["p_two_sided"],
                "did_other_voices_min": min(ref), "did_other_voices_max": max(ref),
                "did_other_voices_median": float(np.median(ref)),
                "mms_rank_for_w2v_gap": int(sorted(gaps[w2v].values()).index(gaps[w2v]["mms:eng"]) + 1),
                "mms_rank_for_cross_gap": int(sorted(gaps[other].values()).index(gaps[other]["mms:eng"]) + 1),
            })
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "same_family.csv", index=False, float_format="%.5f")
    return t


# --------------------------------------------------------------------------- keywords

def keyword_tables(u: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    em = u[(u["set"] == "emergency") & (u["kw_total"] > 0)].copy()
    rows = []
    for (v, c, m), g in em.groupby(["voice", "condition", "model"]):
        k, nn = int(g["kw_hits"].sum()), int(g["kw_total"].sum())
        lo, hi = stats.clopper_pearson(k, nn)
        r = {"voice": v, "condition": c, "model": m, "hits": k, "total": nn, "misses": nn - k,
             "recall": k / nn, "lo": lo, "hi": hi, "n_utts": len(g)}
        if k == nn:
            r["miss_rate_upper95_rule_of_three"] = stats.rule_of_three(nn)
            r["miss_rate_upper95_exact"] = stats.clopper_pearson_upper_zero(nn)
        rows.append(r)
    kr = pd.DataFrame(rows)
    kr.to_csv(TABLES / "keyword_recall.csv", index=False, float_format="%.5f")

    pooled = []
    for (c, m), g in em.groupby(["condition", "model"]):
        k, nn = int(g["kw_hits"].sum()), int(g["kw_total"].sum())
        lo, hi = stats.clopper_pearson(k, nn)
        pooled.append({"level": "pooled_voices", "keyword": "ALL", "condition": c, "model": m,
                       "hits": k, "total": nn, "recall": k / nn, "lo": lo, "hi": hi})
    # per keyword, pooled over voices
    inst = []
    for _, r in em.iterrows():
        hit = set(r["kw_hit"].split("|")) if r["kw_hit"] else set()
        for kw in r["kw_ref"].split("|"):
            inst.append((r["id"], kw, r["voice"], r["condition"], r["model"], kw in hit))
    inst = pd.DataFrame(inst, columns=["id", "keyword", "voice", "condition", "model", "hit"])
    for (kw, c, m), g in inst.groupby(["keyword", "condition", "model"]):
        k, nn = int(g["hit"].sum()), len(g)
        lo, hi = stats.clopper_pearson(k, nn)
        pooled.append({"level": "per_keyword", "keyword": kw, "condition": c, "model": m,
                       "hits": k, "total": nn, "recall": k / nn, "lo": lo, "hi": hi})
    kp = pd.DataFrame(pooled)
    kp.to_csv(TABLES / "keyword_recall_pooled.csv", index=False, float_format="%.5f")

    # McNemar on keyword instances, paired by (sentence, keyword).
    key = inst.set_index(["id", "keyword", "voice", "condition", "model"])["hit"]
    # one row per instance; a keyword can repeat in a sentence only once in find_keywords output
    mc = []
    voices = sorted(inst["voice"].unique())
    models = sorted(inst["model"].unique())

    def compare(kind, m, sel_a, sel_b, label_a, label_b):
        try:
            a = key.xs(sel_a, level=["voice", "condition", "model"])
            b = key.xs(sel_b, level=["voice", "condition", "model"])
        except KeyError:  # cell not run (e.g. Whisper on a reduced condition set)
            return
        a, b = a.align(b, join="inner")
        b01 = int((a & ~b).sum())  # a hit, b missed
        b10 = int((~a & b).sum())  # a missed, b hit
        r = stats.mcnemar(b01, b10)
        mc.append({"comparison": kind, "model": m, "a": label_a, "b": label_b, "n_pairs": len(a),
                   "recall_a": float(a.mean()), "recall_b": float(b.mean()),
                   "a_hit_b_miss": b01, "a_miss_b_hit": b10, "p_value": r["p_value"]})

    for m in models:
        for v in voices:
            for c in ["phone", "phone+noise10", "phone+noise5", "phone+muffle", "phone+loss10", "realistic"]:
                compare("channel", m, (v, "clean", m), (v, c, m), f"{v}|clean", f"{v}|{c}")
        for c in ["clean", "realistic"]:
            for va, vb in itertools.combinations(voices, 2):
                compare("voice", m, (va, c, m), (vb, c, m), f"{va}|{c}", f"{vb}|{c}")
    mct = pd.DataFrame(mc)
    # Holm correction within each comparison family
    for fam, g in mct.groupby("comparison"):
        order = g["p_value"].sort_values().index
        mtests = len(order)
        adj, running = {}, 0.0
        for i, ix in enumerate(order):
            running = max(running, min(1.0, (mtests - i) * mct.loc[ix, "p_value"]))
            adj[ix] = running
        mct.loc[list(adj), "p_holm"] = pd.Series(adj)
    mct.to_csv(TABLES / "keyword_mcnemar.csv", index=False, float_format="%.6g")
    return kr, kp, mct


def librispeech_keywords(u: pd.DataFrame) -> pd.DataFrame:
    ls = u[(u["set"] == "librispeech") & (u["kw_total"] > 0)]
    t = ls.groupby(["voice", "condition", "model"])[["kw_hits", "kw_total"]].sum().reset_index()
    t["recall"] = t["kw_hits"] / t["kw_total"]
    t.to_csv(TABLES / "keyword_recall_librispeech.csv", index=False, float_format="%.4f")
    return t


# --------------------------------------------------------------------------- PPI

def ppi_table(E: dict, n: np.ndarray, reps: int = 1000) -> pd.DataFrame:
    mean_n = n.mean()
    N = len(n)
    rows = []
    rng = np.random.default_rng(SEED)
    for m in MODELS:
        for c in ["clean", "realistic", "phone+noise5"]:
            if ("human", c, m) not in E:
                continue
            y_all = E[("human", c, m)] / mean_n
            truth = float(y_all.mean())  # = corpus WER on all 250 human sentences
            draws = [rng.choice(N, size=k, replace=False) for k in (10, 25, 50) for _ in range(reps)]
            for proxy in SYNTH + ["synthetic_mean"]:
                if (proxy, c, m) not in E:
                    continue
                f_all = E[(proxy, c, m)] / mean_n
                for k in (10, 25, 50):
                    res = {"ppi": [], "ppi++": [], "naive": []}
                    for lab in draws[(10, 25, 50).index(k) * reps:((10, 25, 50).index(k) + 1) * reps]:
                        mask = np.zeros(N, bool)
                        mask[lab] = True
                        r = stats.ppi_mean(y_all[mask], f_all[mask], f_all[~mask])
                        r2 = stats.ppi_mean(y_all[mask], f_all[mask], f_all[~mask], power_tuning=True)
                        for name, x in (("ppi", r), ("ppi++", r2), ("naive", r["naive"])):
                            res[name].append((x["estimate"], x["lo"], x["hi"]))
                    for name, arr in res.items():
                        a = np.array(arr)
                        rows.append({
                            "model": m, "condition": c, "proxy": proxy, "n_labeled": k, "method": name,
                            "truth_human_wer": truth, "synth_only_wer": float(f_all.mean()),
                            "median_width": float(np.median(a[:, 2] - a[:, 1])),
                            "coverage": float(((a[:, 1] <= truth) & (truth <= a[:, 2])).mean()),
                            "mean_abs_error": float(np.mean(np.abs(a[:, 0] - truth))),
                            "rmse": float(np.sqrt(np.mean((a[:, 0] - truth) ** 2))),
                            "corr_human_synth": float(np.corrcoef(y_all, f_all)[0, 1]),
                        })
    t = pd.DataFrame(rows)
    naive = t[t["method"] == "naive"].set_index(["model", "condition", "proxy", "n_labeled"])["median_width"]
    t["width_vs_naive"] = [w / naive[(r.model, r.condition, r.proxy, r.n_labeled)]
                           for w, r in zip(t["median_width"], t.itertuples())]
    t.to_csv(TABLES / "ppi.csv", index=False, float_format="%.5f")
    return t


# --------------------------------------------------------------------------- figures

def fig_wer_dots(t: pd.DataFrame) -> None:
    ls = t[(t["set"] == "librispeech") & (t["voice"] != "synthetic_mean")]
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
        ax.set_title(MSHORT[m], fontsize=10, color=INK)
        ax.set_xlabel("WER (%), 95% bootstrap CI")
        ax.set_xlim(left=0)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_yticks(range(len(order)), [SHORT[v] for v in order])
    axes[0].invert_yaxis()
    fig.suptitle("Same 250 sentences: WER on real speech vs 7 synthetic voices "
                 "(clean and realistic phone channel; gray lines = human)", fontsize=11, x=0.02, ha="left")
    fig.savefig(FIGS / "fig1_wer_by_voice.png")
    plt.close(fig)


def fig_wer_vs_condition(t: pd.DataFrame) -> None:
    ls = t[t["set"] == "librispeech"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
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
        ax.set_xticks(x, conds, rotation=45, ha="right", fontsize=8)
        ax.set_title(MSHORT[m], fontsize=10)
        ax.set_ylabel("WER (%)")
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("WER by channel condition: human vs synthetic voices (LibriSpeech subset, n = 250)",
                 fontsize=11, x=0.02, ha="left")
    fig.savefig(FIGS / "fig2_wer_by_condition.png")
    plt.close(fig)


def fig_gap_forest(t: pd.DataFrame) -> None:
    ls = t[(t["set"] == "librispeech") & (t["voice"].isin(SYNTH)) & (t["condition"] == "clean")]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for j, m in enumerate(MODELS):
        d = ls[ls["model"] == m].set_index("voice").reindex(SYNTH)
        y = np.arange(len(SYNTH)) + (j - 1) * 0.25
        ax.errorbar(d["gap_vs_human"] * 100, y,
                    xerr=[(d["gap_vs_human"] - d["gap_lo"]) * 100, (d["gap_hi"] - d["gap_vs_human"]) * 100],
                    fmt="o", ms=5, color=C_MODEL[m], elinewidth=1.2, label=MSHORT[m])
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(SYNTH)), [SHORT[v] for v in SYNTH])
    ax.invert_yaxis()
    ax.set_xlabel("WER synthetic minus WER human (percentage points), paired 95% CI")
    ax.set_title("Sim-to-real gap per voice, clean audio (right of 0: synthetic is harder)",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(FIGS / "fig3_gap_by_voice.png")
    plt.close(fig)


def fig_decision(fl: pd.DataFrame, a: str = "whisper-large-v3-turbo", b: str = "parakeet-tdt-0.6b-v2") -> None:
    """The lab-vs-reality decision for the closest pair of ASR models, one panel per condition."""
    d = fl[(fl["model_a"] == a) & (fl["model_b"] == b)]
    conds = [c for c in CONDS if c in set(d["condition"])]
    voices = SYNTH + ["synthetic_mean"]
    fig, axes = plt.subplots(1, len(conds), figsize=(3.6 * len(conds) + 1.5, 4.6), sharey=True, squeeze=False)
    for ax, c in zip(axes[0], conds):
        dc = d[d["condition"] == c].set_index("voice")
        h = dc.iloc[0]
        ax.axvspan(h["human_gap_lo"] * 100, h["human_gap_hi"] * 100, color=INK, alpha=0.08, lw=0)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.errorbar([h["human_gap"] * 100], [0], xerr=[[(h["human_gap"] - h["human_gap_lo"]) * 100],
                    [(h["human_gap_hi"] - h["human_gap"]) * 100]], fmt="o", ms=7, color=INK, elinewidth=1.4)
        for i, v in enumerate(voices, start=1):
            r = dc.loc[v]
            col = "#e34948" if r["flip"] else ("#2a78d6" if v != "synthetic_mean" else "#4a3aa7")
            ax.errorbar([r["synth_gap"] * 100], [i], xerr=[[(r["synth_gap"] - r["synth_gap_lo"]) * 100],
                        [(r["synth_gap_hi"] - r["synth_gap"]) * 100]], fmt="o", ms=6, color=col, elinewidth=1.2)
            ax.annotate(f"p_flip {r['p_flip']:.2f}", (1.0, i), xycoords=("axes fraction", "data"),
                        xytext=(4, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=7, color=INK2)
        ax.set_title(f"{c}", fontsize=10)
        ax.set_xlabel("WER Whisper minus WER Parakeet (points)\n> 0: Parakeet better")
    fig.subplots_adjust(wspace=0.45)
    axes[0][0].set_yticks(range(len(voices) + 1), [SHORT["human"]] + [SHORT[v] for v in voices])
    axes[0][0].invert_yaxis()
    fig.suptitle("Which ASR wins? Real speech (black, gray band = 95% CI) vs each synthetic voice "
                 "(red = opposite winner)", fontsize=11, x=0.02, ha="left", y=1.02)
    fig.savefig(FIGS / "fig4_decision_whisper_vs_parakeet.png")
    plt.close(fig)


def fig_keyword_recall(kr: pd.DataFrame) -> None:
    conds = [c for c in CONDS if c in set(kr["condition"])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True,
                             gridspec_kw={"width_ratios": [max(1, kr[kr["model"] == m]["condition"].nunique())
                                                           for m in MODELS]})
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
        ax.set_title(MSHORT[m], fontsize=10)
        ax.grid(False)
    axes[0].set_yticks(range(len(SYNTH)), [SHORT[v] for v in SYNTH], fontsize=8)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label("Emergency keyword recall (%)")
    n_kw = int(kr["total"].iloc[0])
    fig.suptitle(f"Emergency keyword recall by synthetic voice and channel ({n_kw} keyword instances per cell; "
                 "written caller sentences)", fontsize=11, x=0.02, ha="left")
    fig.savefig(FIGS / "fig5_keyword_recall.png")
    plt.close(fig)


def fig_ppi(pp: pd.DataFrame) -> None:
    d = pp[(pp["proxy"] == "synthetic_mean") & (pp["condition"] == "clean")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
    styles = {"naive": ("human labels only", INK2, "o", "--"),
              "ppi": ("PPI (synthetic + human)", "#2a78d6", "s", "-"),
              "ppi++": ("PPI++ (power-tuned)", "#eb6834", "^", "-")}
    for ax, m in zip(axes, MODELS):
        dm = d[d["model"] == m]
        if dm.empty:
            ax.set_visible(False)
            continue
        for meth, (lab, col, mk, ls_) in styles.items():
            x = dm[dm["method"] == meth].sort_values("n_labeled")
            ax.plot(x["n_labeled"], x["median_width"] * 100, color=col, marker=mk, ls=ls_, lw=2, ms=6,
                    label=lab)
        ax.set_xticks([10, 25, 50])
        ax.set_xlabel("human-labeled sentences")
        ax.set_ylabel("median 95% CI width (WER points)")
        ax.set_ylim(bottom=0)
        cov = dm.pivot_table(index="method", columns="n_labeled", values="coverage")
        ax.set_title(f"{MSHORT[m]}\ncoverage at n=10 / 50: human-only {cov.loc['naive', 10]:.0%} / "
                     f"{cov.loc['naive', 50]:.0%}, PPI {cov.loc['ppi', 10]:.0%} / {cov.loc['ppi', 50]:.0%}, "
                     f"PPI++ {cov.loc['ppi++', 10]:.0%} / {cov.loc['ppi++', 50]:.0%}", fontsize=8)
    axes[0].legend(fontsize=8)
    fig.suptitle("Estimating human WER (clean) from 250 synthetic results plus a few human-labeled sentences "
                 "(proxy: mean of 7 TTS voices; 1000 random draws)", fontsize=11, x=0.02, ha="left", y=1.06)
    fig.savefig(FIGS / "fig6_ppi_ci_width.png")
    plt.close(fig)


# --------------------------------------------------------------------------- main

def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    u = pd.read_csv(RESULTS / "utterances.csv", keep_default_na=False, dtype={"speaker": str})
    u["kw_detected_any"] = u["kw_detected_any"].astype(str) == "True"
    summary: dict = {"n_rows": len(u)}
    t, (E, n, idx) = wer_tables(u, summary)
    fl = flips(E, n, idx)
    channel_damage(E, n, idx)
    sf_ = same_family(E, n, idx)
    kr, kp, mct = keyword_tables(u)
    lk = librispeech_keywords(u)
    pp = ppi_table(E, n)
    fig_wer_dots(t)
    fig_wer_vs_condition(t)
    fig_gap_forest(t)
    fig_decision(fl)
    fig_keyword_recall(kr)
    fig_ppi(pp)

    # headline numbers
    ls = t[t["set"] == "librispeech"]
    summary["human_wer"] = ls[ls["voice"] == "human"][["model", "condition", "wer", "lo", "hi"]].to_dict("records")
    summary["n_flips"] = int(fl["flip"].sum())
    summary["n_confident_flips"] = int(fl["confident_flip"].sum())
    summary["n_flip_cells"] = len(fl)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("tables:", sorted(p.name for p in TABLES.iterdir()))
    print("figures:", sorted(p.name for p in FIGS.iterdir()))


if __name__ == "__main__":
    main()
