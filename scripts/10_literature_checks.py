"""Literature checks on existing results (CPU only, no new transcription).

Applies the cheap items of results/followups/LITERATURE-ACTIONS.md to the transcripts already scored:

1. BOOT-BOTH (Deutsch et al. [28] in RELATED-WORK.md): CIs for the 7-voice mean that resample the
   voices as well as the speakers (LibriSpeech) or sentences (FLEURS). Two voice schemes:
     flat    7 voices drawn with replacement from the 7;
     nested  4 engines drawn with replacement from (Kokoro, Piper, MMS, macOS), then, inside each
             drawn engine, as many voices as it has, drawn with replacement from its own voices.
   The speaker-only CI (the v2 method) is reported next to them. Voice draws are independent of
   the speaker draws. Applied to the Whisper vs Parakeet lab gap and DiD (E5), plus a TOST
   equivalence check and a minimum detectable effect on the human gap (E7).
2. Pairwise decision accuracy (Kocmi et al. [26]) and a Type I / Type II decision table
   (Mathur et al. [27]) per voice, on LibriSpeech and FLEURS.
3. Cross-check of the PPI / PPI++ numbers against the reference implementation ppi_py
   (Angelopoulos et al. [34], PyPI ppi-python) when it is installed.

Outputs: results/followups/literature_checks/*.csv, summary.json. Read NOTES.md there.
The speaker weights use the same seed and replicate count as scripts/05_analyze.py, so the
speaker-only numbers reproduce results/tables/decision_flips.csv exactly (checked at run time).
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import stats as st

from sim2real import stats
from sim2real.paths import RESULTS

OUT = RESULTS / "followups" / "literature_checks"
B = 10000
SEED = 0          # speaker / sentence weights: same as 05_analyze.py and 08_fleurs.py
VSEED = 1         # voice resampling stream
SYNTH = ["kokoro:af_heart", "kokoro:am_michael", "piper:en_US-lessac-medium",
         "piper:en_US-ryan-medium", "mms:eng", "say:Samantha", "say:Daniel"]
ENGINE = {"kokoro:af_heart": "kokoro", "kokoro:am_michael": "kokoro",
          "piper:en_US-lessac-medium": "piper", "piper:en_US-ryan-medium": "piper",
          "mms:eng": "mms", "say:Samantha": "macos", "say:Daniel": "macos"}
WHISPER, PARAKEET, W2V = "whisper-large-v3-turbo", "parakeet-tdt-0.6b-v2", "wav2vec2-base-960h"
MODELS = [WHISPER, PARAKEET, W2V]
LS_CONDS = ["clean", "phone", "g722-wb", "opus-nb", "phone+noise10", "phone+noise5",
            "phone+muffle", "phone+loss5", "opus-nb+loss5", "realistic"]
FL_CONDS = ["clean", "phone", "phone+noise10", "realistic"]
DELTA = 0.005     # TOST smallest effect of interest: 0.5 WER points (illustrative, not pre-registered)


# --------------------------------------------------------------------------- data

def load_ls():
    u = pd.read_csv(RESULTS / "utterances.csv", keep_default_na=False, dtype={"speaker": str},
                    usecols=["set", "id", "speaker", "voice", "condition", "model", "errors", "n_ref"])
    d = u[u["set"] == "librispeech"]
    ids = sorted(d["id"].unique())
    first = d.drop_duplicates("id").set_index("id").loc[ids]
    n = first["n_ref"].to_numpy(float)
    piv = d.pivot_table(index="id", columns=["voice", "condition", "model"], values="errors",
                        aggfunc="first").reindex(ids)
    E = {c: piv[c].to_numpy(float) for c in piv.columns if not piv[c].isna().any()}
    W = stats.cluster_weights(first["speaker"].astype(str).to_numpy(), B, seed=SEED)
    return E, n, W


def load_fleurs():
    u = pd.read_csv(RESULTS / "followups" / "fleurs" / "utterances.csv.gz", keep_default_na=False)
    ids = sorted(u["id"].unique())
    first = u.drop_duplicates("id").set_index("id").loc[ids]
    n = first["n_ref"].to_numpy(float)
    piv = u.pivot_table(index="id", columns=["voice", "condition", "model"], values="errors",
                        aggfunc="first").reindex(ids)
    E = {c: piv[c].to_numpy(float) for c in piv.columns if not piv[c].isna().any()}
    W = stats.cluster_weights(np.arange(len(ids)), B, seed=SEED)
    return E, n, W


def voice_weights(scheme: str, rng) -> np.ndarray:
    """(B x 7) weights summing to 1 per replicate; column order = SYNTH."""
    k = len(SYNTH)
    if scheme == "flat":
        draws = rng.integers(0, k, size=(B, k))
        cnt = np.zeros((B, k))
        np.add.at(cnt, (np.repeat(np.arange(B), k), draws.ravel()), 1)
        return cnt / k
    engines = sorted(set(ENGINE.values()))
    members = {e: [i for i, v in enumerate(SYNTH) if ENGINE[v] == e] for e in engines}
    cnt = np.zeros((B, k))
    eidx = rng.integers(0, len(engines), size=(B, len(engines)))
    for b in range(B):
        for j in eidx[b]:
            mem = members[engines[j]]
            for i in rng.integers(0, len(mem), size=len(mem)):
                cnt[b, mem[i]] += 1
    return cnt / cnt.sum(1, keepdims=True)


def ci(boot, alpha=0.05):
    return float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))


def excl0(lo, hi):
    return bool(lo > 0 or hi < 0)


# --------------------------------------------------------------------------- 1. BOOT-BOTH

def per_voice_boot(E, n, W, a, b, c):
    """Per-voice lab gap (WER a - WER b): point (7,) and speaker/sentence bootstrap (B x 7)."""
    Wn = W @ n
    X = np.stack([E[(v, c, a)] - E[(v, c, b)] for v in SYNTH])          # (7, S)
    return X.sum(1) / n.sum(), (W @ X.T) / Wn[:, None]


def human_boot(E, n, W, a, b, c):
    h = E[("human", c, a)] - E[("human", c, b)]
    return h.sum() / n.sum(), (W @ h) / (W @ n)


def mean_rows(corpus, E, n, W, conds, VW):
    rows, boots = [], {}
    for a, b in [(WHISPER, PARAKEET), (WHISPER, W2V), (PARAKEET, W2V)]:
        for c in conds:
            keys = [("human", c, a), ("human", c, b)] + [(v, c, m) for v in SYNTH for m in (a, b)]
            if not all(k in E for k in keys):
                continue
            sp, sb = per_voice_boot(E, n, W, a, b, c)
            hp, hb = human_boot(E, n, W, a, b, c)
            mp = float(sp.mean())
            schemes = {"speaker_only": sb.mean(1),
                       "voices_only": VW["flat"] @ sp,
                       "both_flat": (sb * VW["flat"]).sum(1),
                       "both_nested": (sb * VW["nested"]).sum(1)}
            hlo, hhi = ci(hb)
            for s, gb in schemes.items():
                db = gb - (hp if s == "voices_only" else hb)
                glo, ghi = ci(gb)
                dlo, dhi = ci(db)
                rows.append({"corpus": corpus, "model_a": a, "model_b": b, "condition": c, "scheme": s,
                             "human_gap": hp, "human_lo": hlo, "human_hi": hhi,
                             "human_ci_excludes_0": excl0(hlo, hhi),
                             "mean_gap": mp, "mean_gap_lo": glo, "mean_gap_hi": ghi,
                             "mean_gap_ci_excludes_0": excl0(glo, ghi), "mean_gap_p": stats.boot_p(gb),
                             "did": mp - hp, "did_lo": dlo, "did_hi": dhi, "did_p": stats.boot_p(db),
                             "width_vs_speaker_only": np.nan})
                boots[(a, b, c, s)] = (gb, hb)
    t = pd.DataFrame(rows)
    ref = t[t.scheme == "speaker_only"].set_index(["model_a", "model_b", "condition"])
    w0 = ref["mean_gap_hi"] - ref["mean_gap_lo"]
    t["width_vs_speaker_only"] = [(r.mean_gap_hi - r.mean_gap_lo) / w0[(r.model_a, r.model_b, r.condition)]
                                  for r in t.itertuples()]
    return t, boots


def mean_vs_human_rows(corpus, E, n, W, conds, VW):
    """7-voice mean minus human, per model x condition: the WER gap (synthetic - human) and the
    channel-damage error ((synthetic channel - clean) - (human channel - clean)), under each scheme."""
    rows = []
    Wn = W @ n
    for m in MODELS:
        for c in conds:
            for qty in ("gap", "damage_error"):
                if qty == "damage_error" and c == "clean":
                    continue
                need = [("human", c, m)] + [(v, c, m) for v in SYNTH]
                if qty == "damage_error":
                    need += [("human", "clean", m)] + [(v, "clean", m) for v in SYNTH]
                if not all(k in E for k in need):
                    continue
                if qty == "gap":
                    X = np.stack([E[(v, c, m)] for v in SYNTH]); h = E[("human", c, m)]
                else:
                    X = np.stack([E[(v, c, m)] - E[(v, "clean", m)] for v in SYNTH])
                    h = E[("human", c, m)] - E[("human", "clean", m)]
                sp = X.sum(1) / n.sum(); sb = (W @ X.T) / Wn[:, None]
                hp = h.sum() / n.sum(); hb = (W @ h) / Wn
                schemes = {"speaker_only": sb.mean(1) - hb, "voices_only": VW["flat"] @ sp - hp,
                           "both_flat": (sb * VW["flat"]).sum(1) - hb,
                           "both_nested": (sb * VW["nested"]).sum(1) - hb}
                for s_, bt in schemes.items():
                    lo, hi = ci(bt)
                    rows.append({"corpus": corpus, "model": m, "condition": c, "quantity": qty, "scheme": s_,
                                 "human": hp, "mean_minus_human": float(sp.mean() - hp), "lo": lo, "hi": hi,
                                 "ci_excludes_0": excl0(lo, hi), "p": stats.boot_p(bt),
                                 "voice_min": float((sp - hp).min()), "voice_max": float((sp - hp).max())})
    return pd.DataFrame(rows)


def tost_rows(corpus, E, n, W, conds):
    rows = []
    for c in conds:
        if not (("human", c, WHISPER) in E and ("human", c, PARAKEET) in E):
            continue
        hp, hb = human_boot(E, n, W, WHISPER, PARAKEET, c)
        lo90, hi90 = ci(hb, 0.10)
        lo95, hi95 = ci(hb)
        se = float(hb.std(ddof=1))
        rows.append({"corpus": corpus, "condition": c, "human_gap": hp, "ci95_lo": lo95, "ci95_hi": hi95,
                     "ci90_lo": lo90, "ci90_hi": hi90, "delta": DELTA,
                     "equivalent_within_delta": bool(lo90 > -DELTA and hi90 < DELTA),
                     "boot_se": se, "mde80_unadjusted": 2.80 * se})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 2. decisions

def decision_rows(corpus, E, n, W, conds, boots_mean):
    """One row per (voice or mean scheme) x model pair x condition."""
    rows, agree_boot = [], {}
    for a, b in [(WHISPER, PARAKEET), (WHISPER, W2V), (PARAKEET, W2V)]:
        for c in conds:
            if not all(k in E for k in [("human", c, a), ("human", c, b)]):
                continue
            hp, hb = human_boot(E, n, W, a, b, c)
            hlo, hhi = ci(hb)
            cands = {}
            for v in SYNTH:
                if (v, c, a) in E and (v, c, b) in E:
                    x = E[(v, c, a)] - E[(v, c, b)]
                    cands[v] = (x.sum() / n.sum(), (W @ x) / (W @ n))
            for s in ("speaker_only", "both_flat", "both_nested"):
                if (a, b, c, s) in boots_mean:
                    gb, _ = boots_mean[(a, b, c, s)]
                    sp = float(np.mean([cands[v][0] for v in SYNTH]))
                    cands[f"mean[{s}]"] = (sp, gb)
            for v, (sp, sb) in cands.items():
                slo, shi = ci(sb)
                h_sig, s_sig = excl0(hlo, hhi), excl0(slo, shi)
                same = np.sign(sp) == np.sign(hp)
                rows.append({"corpus": corpus, "voice": v, "model_a": a, "model_b": b, "condition": c,
                             "pair_type": "close" if (a, b) == (WHISPER, PARAKEET) else "large",
                             "human_gap": hp, "human_sig": h_sig, "lab_gap": sp, "lab_sig": s_sig,
                             "sign_agree": bool(same),
                             "type_I": bool(s_sig and not h_sig),
                             "type_II": bool(h_sig and not s_sig),
                             "confident_wrong_sign": bool(s_sig and h_sig and not same),
                             "both_sig_right_sign": bool(s_sig and h_sig and same),
                             "both_null": bool(not s_sig and not h_sig)})
                agree_boot[(v, a, b, c)] = (np.sign(sb) == np.sign(hb)).astype(float)
    return pd.DataFrame(rows), agree_boot


def accuracy_table(dec, agree_boot):
    rows = []
    for (corpus, v), g in dec.groupby(["corpus", "voice"], sort=False):
        for scope, sel in [("all pairs", g.index == g.index),
                           ("close pair (Whisper vs Parakeet)", (g.pair_type == "close").to_numpy()),
                           ("large-gap pairs", (g.pair_type == "large").to_numpy())]:
            for subset, mask in [("all rows", np.ones(len(g), bool)), ("human CI excludes 0", g.human_sig.to_numpy())]:
                gg = g[sel & mask]
                if gg.empty:
                    continue
                bt = np.mean([agree_boot[(r.voice, r.model_a, r.model_b, r.condition)] for r in gg.itertuples()], 0)
                lo, hi = ci(bt)
                rows.append({"corpus": corpus, "voice": v, "scope": scope, "rows": subset, "n_decisions": len(gg),
                             "pairwise_accuracy": float(gg.sign_agree.mean()), "acc_lo": lo, "acc_hi": hi,
                             "swap_rate": float(1 - gg.sign_agree.mean()),
                             "type_I": int(gg.type_I.sum()), "type_II": int(gg.type_II.sum()),
                             "confident_wrong_sign": int(gg.confident_wrong_sign.sum()),
                             "both_sig_right_sign": int(gg.both_sig_right_sign.sum()),
                             "both_null": int(gg.both_null.sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 3. PPI reference check

def ppi_check(E, n):
    try:
        import ppi_py
        from ppi_py import ppi_mean_ci, ppi_mean_pointestimate, classical_mean_ci
        from ppi_py.ppi import _calc_lam_glm
    except ImportError:
        return None, {"ppi_py": "not installed"}
    mean_n = n.mean()
    N = len(n)
    rng = np.random.default_rng(SEED)        # same draws as 05_analyze.ppi_table
    draws = {k: np.array([rng.choice(N, size=k, replace=False) for _ in range(1000)]) for k in (10, 25, 50)}
    ref = pd.read_csv(RESULTS / "tables" / "ppi.csv")
    ref = ref[ref.proxy == "synthetic_mean"].set_index(["model", "condition", "n_labeled", "method"])
    rows = []
    maxdiff = {"point_lam1": 0.0, "se_lam1_ddof0": 0.0, "lam_unclipped_abs": 0.0, "ours_vs_ppi_csv": 0.0}
    for m in MODELS:
        for c in ["clean", "phone", "phone+noise10", "phone+noise5", "realistic"]:
            if ("human", c, m) not in E:
                continue
            y_all = E[("human", c, m)] / mean_n
            f_all = np.mean([E[(v, c, m)] for v in SYNTH], 0) / mean_n
            truth = float(y_all.mean())
            for k, lab in draws.items():
                ours = {meth: stats.ppi_mean_finite(y_all[lab], f_all[lab], f_all.mean(), N, method=meth)
                        for meth in ("naive", "ppi", "ppi++")}
                for meth, r in ours.items():
                    cov = float(np.mean((r["lo"] <= truth) & (truth <= r["hi"])))
                    maxdiff["ours_vs_ppi_csv"] = max(maxdiff["ours_vs_ppi_csv"],
                                                     abs(cov - ref.loc[(m, c, k, meth), "coverage"]))
                res = {"py_classical": [], "py_ppi": [], "py_ppi++": [], "py_ppi_others": []}
                for idx in lab:
                    y, f = y_all[idx], f_all[idx]
                    others = np.setdiff1d(np.arange(N), idx)
                    res["py_classical"].append(classical_mean_ci(y, alpha=0.05))
                    res["py_ppi"].append(ppi_mean_ci(y, f, f_all, alpha=0.05, lam=1))
                    res["py_ppi++"].append(ppi_mean_ci(y, f, f_all, alpha=0.05))
                    res["py_ppi_others"].append(ppi_mean_ci(y, f, f_all[others], alpha=0.05, lam=1))
                    # exactness checks against our formulas
                    pe = float(np.squeeze(ppi_mean_pointestimate(y, f, f_all, lam=1)))
                    pe_ours = f_all.mean() + (y - f).mean()
                    maxdiff["point_lam1"] = max(maxdiff["point_lam1"], abs(pe - pe_ours))
                    lo, hi = ppi_mean_ci(y, f, f_all, alpha=0.05, lam=1)
                    se_py = (float(np.squeeze(hi)) - float(np.squeeze(lo))) / (2 * st.norm.ppf(0.975))
                    se_ours0 = np.sqrt(f_all.var() / N + (y - f).var() / k)      # ddof 0, no fpc, unl term
                    maxdiff["se_lam1_ddof0"] = max(maxdiff["se_lam1_ddof0"], abs(se_py - se_ours0))
                    g, gh, gu = y - pe, f - pe, f_all - pe
                    lam_py = float(_calc_lam_glm(g, gh, gu, np.eye(1), clip=False))
                    vf = np.var(np.concatenate([f, f_all]), ddof=1)
                    lam_ours = np.cov(y, f, ddof=1)[0, 1] / ((1 + k / N) * vf) if vf > 0 else 0.0
                    maxdiff["lam_unclipped_abs"] = max(maxdiff["lam_unclipped_abs"],
                                                       abs(lam_py - lam_ours * (k - 1) / k))
                naive_w = np.median(ours["naive"]["width"])
                for meth in ("naive", "ppi", "ppi++"):
                    r = ours[meth]
                    rows.append({"model": m, "condition": c, "n_labeled": k, "implementation": "ours (finite, t)",
                                 "method": meth,
                                 "coverage": float(np.mean((r["lo"] <= truth) & (truth <= r["hi"]))),
                                 "median_width": float(np.median(r["width"])),
                                 "width_vs_ours_naive": float(np.median(r["width"]) / naive_w)})
                for meth, lst in res.items():
                    a = np.array([[float(np.squeeze(lo)), float(np.squeeze(hi))] for lo, hi in lst])
                    w = a[:, 1] - a[:, 0]
                    rows.append({"model": m, "condition": c, "n_labeled": k, "implementation": "ppi_py 0.2.3",
                                 "method": meth,
                                 "coverage": float(np.mean((a[:, 0] <= truth) & (truth <= a[:, 1]))),
                                 "median_width": float(np.median(w)),
                                 "width_vs_ours_naive": float(np.median(w) / naive_w)})
    info = {"ppi_py_version": getattr(ppi_py, "__version__", "0.2.3"), **{f"max_abs_{k}": v for k, v in maxdiff.items()}}
    return pd.DataFrame(rows), info


# --------------------------------------------------------------------------- main

def check_reproduces(dec_ls, path, keys=("model_a", "model_b", "condition", "voice")):
    ref = pd.read_csv(path).set_index(list(keys))
    v = dec_ls[dec_ls.voice.isin(SYNTH)]
    diffs = [abs(r.lab_gap - ref.loc[(r.model_a, r.model_b, r.condition, r.voice), "synth_gap"]) for r in v.itertuples()]
    return float(max(diffs))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(VSEED)
    VW = {"flat": voice_weights("flat", rng), "nested": voice_weights("nested", rng)}
    summary: dict = {"B": B, "voice_engines": ENGINE, "tost_delta_points": DELTA * 100}

    all_mean, all_tost, all_dec, all_agree, all_mvh = [], [], [], {}, []
    for corpus, loader, conds in [("librispeech", load_ls, LS_CONDS), ("fleurs", load_fleurs, FL_CONDS)]:
        E, n, W = loader()
        mt, boots = mean_rows(corpus, E, n, W, conds, VW)
        all_mean.append(mt)
        all_tost.append(tost_rows(corpus, E, n, W, conds))
        all_mvh.append(mean_vs_human_rows(corpus, E, n, W, conds, VW))
        dec, agree = decision_rows(corpus, E, n, W, conds, boots)
        all_dec.append(dec)
        all_agree.update({(corpus,) + k: v for k, v in agree.items()})
        ref = RESULTS / "tables" / "decision_flips.csv" if corpus == "librispeech" else \
            RESULTS / "followups" / "fleurs" / "decisions.csv"
        summary[f"{corpus}_max_abs_diff_vs_published_synth_gap"] = check_reproduces(dec, ref)
        if corpus == "librispeech":
            pp, info = ppi_check(E, n)
            summary["ppi_reference"] = info
            if pp is not None:
                pp.to_csv(OUT / "ppi_vs_ppi_py.csv", index=False, float_format="%.5f")

    mt = pd.concat(all_mean, ignore_index=True)
    mt.to_csv(OUT / "mean_boot_both.csv", index=False, float_format="%.5f")
    mvh = pd.concat(all_mvh, ignore_index=True)
    mvh.to_csv(OUT / "mean_vs_human_boot_both.csv", index=False, float_format="%.5f")
    print(mvh.groupby(["corpus", "quantity", "scheme"])["ci_excludes_0"].agg(["size", "sum"]).to_string())
    tt = pd.concat(all_tost, ignore_index=True)
    tt.to_csv(OUT / "human_gap_tost_mde.csv", index=False, float_format="%.5f")
    dec = pd.concat(all_dec, ignore_index=True)
    dec.to_csv(OUT / "decisions_by_voice.csv", index=False, float_format="%.5f")
    acc_parts = []
    for corpus in ("librispeech", "fleurs"):
        d = dec[dec.corpus == corpus]
        ab = {k[1:]: v for k, v in all_agree.items() if k[0] == corpus}
        acc_parts.append(accuracy_table(d, ab))
    acc = pd.concat(acc_parts, ignore_index=True)
    acc.to_csv(OUT / "pairwise_accuracy_type_I_II.csv", index=False, float_format="%.4f")

    # headline extracts
    wp = mt[(mt.model_a == WHISPER) & (mt.model_b == PARAKEET)]
    summary["whisper_vs_parakeet_mean"] = wp[["corpus", "condition", "scheme", "human_gap", "human_lo", "human_hi",
                                              "mean_gap", "mean_gap_lo", "mean_gap_hi", "mean_gap_ci_excludes_0",
                                              "did", "did_lo", "did_hi", "width_vs_speaker_only"]].to_dict("records")
    summary["tost"] = tt.to_dict("records")
    print(wp[["corpus", "condition", "scheme", "mean_gap", "mean_gap_lo", "mean_gap_hi", "did_lo", "did_hi",
              "width_vs_speaker_only"]].round(4).to_string())
    print(tt.round(4).to_string())
    print(acc[(acc.scope != "large-gap pairs")].round(3).to_string())
    print(json.dumps({k: v for k, v in summary.items() if k not in ("whisper_vs_parakeet_mean", "tost")},
                     indent=1, default=float))
    if (OUT / "ppi_vs_ppi_py.csv").exists():
        pp = pd.read_csv(OUT / "ppi_vs_ppi_py.csv")
        s = pp[pp.model != W2V].pivot_table(index=["method", "n_labeled"], columns="implementation",
                                            values=["coverage", "width_vs_ours_naive"], aggfunc=["min", "max"])
        print(s.round(3).to_string())
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
