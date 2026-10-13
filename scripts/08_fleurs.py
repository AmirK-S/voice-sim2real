"""Follow-up: FLEURS en_us as an out-of-domain human baseline (do the v2 conclusions hold?).

Steps (each cached and resumable)
  corpus   download the en_us test split (TSV + audio archive only), select 250 sentences
           (sim2real.fleurs.select_fleurs), extract the human wavs, synthesize the same
           transcripts with the 7 TTS voices of the main run. Audio: data/audio/fleurs/<voice>/.
  run      --model M --conditions a,b  transcribe every clip through the same preparation and
           channel code as scripts/04_run_experiment.py (seed from the sentence id, so voices stay
           paired). Shards: results/raw/followups/fleurs/<model>/<condition>.csv (gitignored).
  merge    score the shards into results/followups/fleurs/utterances.csv.gz
  analyze  tables, figures and numbers for NOTES.md (see analyze() docstring)

Grid: Parakeet and wav2vec2 on clean, phone, phone+noise10, realistic; Whisper on clean and
realistic (GPU time). Run the models one after the other; the GPU is shared.

Usage
  uv run python scripts/08_fleurs.py corpus
  uv run python scripts/08_fleurs.py run --model parakeet --conditions clean,phone,phone+noise10,realistic
  uv run python scripts/08_fleurs.py run --model wav2vec2 --conditions clean,phone,phone+noise10,realistic
  uv run python scripts/08_fleurs.py run --model whisper --conditions clean,realistic
  uv run python scripts/08_fleurs.py merge
  uv run python scripts/08_fleurs.py analyze
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import soundfile as sf

from sim2real import asr, channel, fleurs, stats, tts
from sim2real.audio import resample
from sim2real.paths import DATA, RAW, RESULTS

OUT = RESULTS / "followups" / "fleurs"
SHARDS = RAW / "followups" / "fleurs"
MANIFEST = OUT / "manifest.csv"
SUBSET = OUT / "fleurs_subset.csv"
SYNTH = tts.VOICES
assert SYNTH == ["kokoro:af_heart", "kokoro:am_michael", "piper:en_US-lessac-medium",
                 "piper:en_US-ryan-medium", "mms:eng", "say:Samantha", "say:Daniel"]


def utt_seed(utt_id: str) -> int:  # identical to scripts/04_run_experiment.py
    return zlib.crc32(utt_id.encode()) & 0x7FFFFFFF


# --------------------------------------------------------------------------- corpus

def corpus(n: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fleurs.download()
    sub = fleurs.select_fleurs(n=n)
    sub[["id", "sentence_id", "file", "gender", "duration_s", "text", "tts_text", "speaker"]].to_csv(
        SUBSET, index=False)
    print(f"selected {len(sub)} sentences ({sub['gender'].value_counts().to_dict()}), "
          f"{sub['duration_s'].sum() / 60:.1f} min of human audio")
    wavs = fleurs.extract(list(sub["file"]))
    rows = []
    for r in sub.itertuples():
        out = fleurs.voice_dir("human") / f"{r.id}.wav"
        if not out.exists():
            a, sr = sf.read(wavs[r.file], dtype="float32")
            a = a.mean(axis=1) if a.ndim > 1 else a
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, resample(a, sr), 16000, subtype="PCM_16")
        rows.append(dict(id=r.id, voice="human", path=str(out.relative_to(DATA.parent)), text=r.text,
                         speaker=r.speaker, gender=r.gender))
    errors = []
    for voice in SYNTH:
        t0 = time.perf_counter()
        jobs = [(r, fleurs.voice_dir(voice) / f"{r.id}.wav") for r in sub.itertuples()]
        workers = 6 if voice.startswith("say:") else 1
        with ThreadPoolExecutor(workers) as ex:
            status = list(ex.map(lambda j: fleurs.synth_one(j[0].tts_text, voice, j[1]), jobs))
        for (r, path), st in zip(jobs, status):
            if st in ("ok", "skip"):
                rows.append(dict(id=r.id, voice=voice, path=str(path.relative_to(DATA.parent)), text=r.text,
                                 speaker=r.speaker, gender=r.gender))
            else:
                errors.append(dict(id=r.id, voice=voice, error=st))
        print(f"{voice}: {status.count('ok')} new, {status.count('skip')} cached, "
              f"{len(jobs) - status.count('ok') - status.count('skip')} failed, "
              f"{time.perf_counter() - t0:.0f} s", flush=True)
    man = pd.DataFrame(rows)
    man["duration_s"] = [round(sf.info(DATA.parent / p).duration, 3) for p in man["path"]]
    man.to_csv(MANIFEST, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(OUT / "tts_errors.csv", index=False)
        print(f"{len(errors)} TTS failures, see {OUT / 'tts_errors.csv'}")
    print(man.groupby("voice")["duration_s"].agg(["count", "mean"]).round(2).to_string())


# --------------------------------------------------------------------------- run

def read16k(path: str) -> np.ndarray:
    x, sr = sf.read(ROOT / path, dtype="float32")
    assert sr == channel.SR, (path, sr)
    return x.mean(axis=1) if x.ndim > 1 else x


def run(model: str, conditions: list[str]) -> None:
    model = asr.resolve(model)
    m = pd.read_csv(MANIFEST, keep_default_na=False)
    out_dir = SHARDS / model
    out_dir.mkdir(parents=True, exist_ok=True)
    for cond in conditions:
        shard = out_dir / f"{cond}.csv"
        if shard.exists() and len(pd.read_csv(shard)) == len(m):
            print(f"[{model}] {cond}: already done", flush=True)
            continue
        t0, rows = time.time(), []
        for voice, g in m.groupby("voice", sort=True):
            arrays, metas = [], []
            for p, i in zip(g["path"], g["id"]):
                meta: dict = {}
                x = channel.prepare(read16k(p), seed=utt_seed(i))
                arrays.append(channel.apply(cond, x, seed=utt_seed(i), meta=meta))
                metas.append(meta)
            hyps = asr.transcribe_arrays(arrays, model)
            rows += [{"id": i, "voice": voice, "hyp": h, "snr_post_channel_db": mt.get("snr_post_channel_db", ""),
                      "noise_group": mt.get("noise_group", ""), "loss_rate": mt.get("loss_rate", "")}
                     for i, h, mt in zip(g["id"], hyps, metas)]
        tmp = shard.with_suffix(".tmp")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.replace(shard)
        print(f"[{model}] {cond}: {len(rows)} clips in {time.time() - t0:.0f} s", flush=True)


# --------------------------------------------------------------------------- merge

def merge() -> pd.DataFrame:
    import jiwer

    m = pd.read_csv(MANIFEST, keep_default_na=False)
    shards = []
    for model_dir in sorted(SHARDS.iterdir()):
        for f in sorted(model_dir.glob("*.csv")):
            d = pd.read_csv(f, keep_default_na=False)
            d["model"], d["condition"] = model_dir.name, f.stem
            shards.append(d)
    t = pd.concat(shards, ignore_index=True)
    t = t.merge(m[["id", "voice", "text", "speaker", "gender", "duration_s"]], on=["id", "voice"],
                how="left", validate="many_to_one").rename(columns={"text": "ref"})
    assert t["ref"].notna().all()
    t["ref"] = [fleurs.scoring_ref(r) for r in t["ref"]]  # keep text inside parentheses
    ref_n = {r: asr.normalize(r) for r in t["ref"].unique()}
    errs, nref, eidx = [], [], []
    for r, h in zip(t["ref"], t["hyp"]):
        rn, hn = ref_n[r], asr.normalize(fleurs.scoring_ref(h))
        n = len(rn.split())
        if not hn:
            e, bad = n, list(range(n))
        else:
            o = jiwer.process_words(rn, hn)
            e = o.substitutions + o.deletions + o.insertions
            bad = [k for ch in o.alignments[0] if ch.type in ("substitute", "delete")
                   for k in range(ch.ref_start_idx, ch.ref_end_idx)]
        errs.append(e)
        nref.append(n)
        eidx.append(" ".join(map(str, bad)))
    t["errors"], t["n_ref"], t["err_ref_idx"] = errs, nref, eidx
    t["wer"] = t["errors"] / t["n_ref"]
    t = t.sort_values(["model", "condition", "voice", "id"]).reset_index(drop=True)
    t.to_csv(OUT / "utterances.csv.gz", index=False)
    print(f"merged {len(t)} rows")
    print(t.groupby(["model", "condition"]).size().to_string())
    return t


# --------------------------------------------------------------------------- analysis

B = 10000
SEED = 0
CONDS = ["clean", "phone", "phone+noise10", "realistic"]
MODELS = ["whisper-large-v3-turbo", "parakeet-tdt-0.6b-v2", "wav2vec2-base-960h"]
WHISPER, PARAKEET, W2V = MODELS
PAIRS = [(WHISPER, PARAKEET), (WHISPER, W2V), (PARAKEET, W2V)]
MEAN = "synthetic_mean"
SHORT = {"human": "human", "kokoro:af_heart": "Kokoro af_heart", "kokoro:am_michael": "Kokoro am_michael",
         "piper:en_US-lessac-medium": "Piper lessac", "piper:en_US-ryan-medium": "Piper ryan",
         "mms:eng": "Meta MMS-TTS", "say:Samantha": "macOS Samantha", "say:Daniel": "macOS Daniel",
         MEAN: "mean of 7 voices"}
MSHORT = {WHISPER: "Whisper", PARAKEET: "Parakeet", W2V: "wav2vec2"}
LS_TABLES = RESULTS / "tables"
INK, INK2, GRID, BLUE, ORANGE, GRAY, RED = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#9a9994", "#e34948"


def _paired(ea, eb, n, W, na=None, nb=None) -> dict:
    """Corpus WER(a) - WER(b) on the same sentences, ratio bootstrap with weights W."""
    na = n if na is None else na
    nb = n if nb is None else nb
    d = ea.sum() / na.sum() - eb.sum() / nb.sum()
    boot = (W @ ea) / (W @ na) - (W @ eb) / (W @ nb)
    return {"diff": float(d), "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
            "p": stats.boot_p(boot), "boot": boot}


def _ratio(e, n, W):
    boot = (W @ e) / (W @ n)
    return float(e.sum() / n.sum()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _ls_lookup(path: Path, keys: list[str]) -> dict:
    t = pd.read_csv(path)
    return {tuple(r[k] for k in keys): r for _, r in t.iterrows()}


def _flag_mismatch(u: pd.DataFrame, ids: list[str]) -> tuple[dict, pd.DataFrame]:
    """Reference words that look like human reading or recording mismatches: on human clean audio
    all three models get the word wrong (substitution or deletion), while at most 2 of the 21
    synthetic clean decodes (7 voices x 3 models) do. TTS reads the reference text exactly, so a
    word only humans 'fail' is either a deviation of the recording from the reference or a word that
    is genuinely hard in that speaker's accent; the filter cannot tell them apart."""
    c = u[u["condition"] == "clean"]
    models = [m for m in MODELS if m in set(c["model"])]
    errs = {(r.voice, r.id, r.model): set(map(int, r.err_ref_idx.split())) if r.err_ref_idx else set()
            for r in c.itertuples()}
    refs = c.drop_duplicates("id").set_index("id")["ref"]
    flagged, rows = {}, []
    for i in ids:
        hum = set.intersection(*[errs[("human", i, m)] for m in models])
        keep = {k for k in hum if sum(k in errs[(v, i, m)] for v in SYNTH for m in models) <= 2 * len(models) // 3}
        if keep:
            flagged[i] = keep
            toks = asr.normalize(refs[i]).split()
            hyp = c[(c["id"] == i) & (c["voice"] == "human") & (c["model"] == PARAKEET)]["hyp"].iloc[0]
            rows.append({"id": i, "n_words": len(keep), "words": " ".join(toks[k] for k in sorted(keep)),
                         "ref": refs[i], "parakeet_hyp_human_clean": hyp})
    return flagged, pd.DataFrame(rows)


def analyze() -> None:
    """Tables (results/followups/fleurs/):
    - human_wer.csv        human WER per model x condition, FLEURS vs LibriSpeech
    - gaps.csv             synthetic minus human WER per voice (and 7-voice mean), FLEURS vs LibriSpeech,
                           BH over the FLEURS voice cells, plus a sensitivity without suspected reference
                           mismatches
    - voice_spread.csv     worst / best voice WER ratio per model x condition
    - channel_damage.csv   synthetic damage minus human damage, FLEURS vs LibriSpeech
    - decisions.csv        per model pair x condition x voice: human gap, lab gap, DiD, Holm within each
                           model pair over the voice-level decisions (the mean gets its own Holm family)
    - reference_mismatch.csv  words flagged by _flag_mismatch
    - gap_by_sentence_type.csv  gaps on sentences with proper names or digits vs plain sentences
    - summary.json
    CIs: 95 % percentile bootstrap, 10,000 replicates, resampling sentences (no speaker ids in FLEURS).
    LibriSpeech numbers are read from results/tables/ (speaker-cluster bootstrap)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150,
        "savefig.bbox": "tight", "legend.frameon": False})

    u = pd.read_csv(OUT / "utterances.csv.gz", keep_default_na=False)
    ids = sorted(u["id"].unique())
    first = u.drop_duplicates("id").set_index("id").loc[ids]
    n = first["n_ref"].to_numpy(float)
    piv = u.pivot_table(index="id", columns=["voice", "condition", "model"], values="errors",
                        aggfunc="first").reindex(ids)
    E = {c: piv[c].to_numpy(float) for c in piv.columns if not piv[c].isna().any()}
    for c, m in {(c, m) for (_, c, m) in E}:
        if all((v, c, m) in E for v in SYNTH):
            E[(MEAN, c, m)] = np.mean([E[(v, c, m)] for v in SYNTH], axis=0)
    W = stats.cluster_weights(np.arange(len(ids)), B, seed=SEED)
    VOICES = SYNTH + [MEAN]
    summary: dict = {"n_sentences": len(ids), "n_words": int(n.sum()),
                     "gender": first["gender"].value_counts().to_dict(),
                     "human_minutes": float(first["duration_s"].sum() / 60) if "duration_s" in first else None}

    ls_cell = _ls_lookup(LS_TABLES / "wer_by_cell.csv", ["set", "model", "condition", "voice"])
    ls_dmg = _ls_lookup(LS_TABLES / "channel_damage.csv", ["model", "condition", "voice"])
    ls_dec = _ls_lookup(LS_TABLES / "decision_flips.csv", ["model_a", "model_b", "condition", "voice"])

    # ---- human WER
    rows = []
    for m in MODELS:
        for c in CONDS:
            if ("human", c, m) not in E:
                continue
            w, lo, hi = _ratio(E[("human", c, m)], n, W)
            ls = ls_cell[("librispeech", m, c, "human")]
            rows.append({"model": m, "condition": c, "fleurs_wer": w, "fleurs_lo": lo, "fleurs_hi": hi,
                         "ls_wer": ls["wer"], "ls_lo": ls["lo"], "ls_hi": ls["hi"]})
    hw = pd.DataFrame(rows)
    hw.to_csv(OUT / "human_wer.csv", index=False, float_format="%.5f")

    # ---- reference mismatch sensitivity
    flagged, mm = _flag_mismatch(u, ids)
    mm.to_csv(OUT / "reference_mismatch.csv", index=False)
    pos = {i: k for k, i in enumerate(ids)}
    nflag = np.zeros(len(ids))
    for i, s in flagged.items():
        nflag[pos[i]] = len(s)
    errsets = {(r.voice, r.condition, r.model, r.id): set(map(int, r.err_ref_idx.split())) if r.err_ref_idx else set()
               for r in u.itertuples()}

    def adj(v, c, m):
        if v == MEAN:
            return np.mean([adj(x, c, m) for x in SYNTH], axis=0)
        e = E[(v, c, m)].copy()
        for i, s in flagged.items():
            e[pos[i]] -= len(s & errsets.get((v, c, m, i), set()))
        return e

    summary["mismatch_words"] = int(nflag.sum())
    summary["mismatch_sentences"] = len(flagged)
    summary["mismatch_share_words"] = float(nflag.sum() / n.sum())

    # ---- gaps
    rows = []
    for (v, c, m) in sorted(E):
        if v == "human" or ("human", c, m) not in E:
            continue
        g = _paired(E[(v, c, m)], E[("human", c, m)], n, W)
        ga = _paired(adj(v, c, m), adj("human", c, m), n, W, n - nflag, n - nflag)
        w, lo, hi = _ratio(E[(v, c, m)], n, W)
        r = {"model": m, "condition": c, "voice": v, "wer": w, "wer_lo": lo, "wer_hi": hi,
             "gap": g["diff"], "gap_lo": g["lo"], "gap_hi": g["hi"], "gap_p": g["p"],
             "gap_no_mismatch": ga["diff"], "gap_no_mismatch_lo": ga["lo"], "gap_no_mismatch_hi": ga["hi"]}
        ls = ls_cell.get(("librispeech", m, c, v))
        if ls is not None:
            r.update({"ls_gap": ls["gap_vs_human"], "ls_gap_lo": ls["gap_lo"], "ls_gap_hi": ls["gap_hi"]})
        rows.append(r)
    gp = pd.DataFrame(rows)
    mask = gp["voice"].isin(SYNTH)
    gp.loc[mask, "gap_p_bh"] = stats.bh(gp.loc[mask, "gap_p"])
    for pre in ("gap", "ls_gap"):
        gp[f"{pre}_sig"] = np.where(gp[f"{pre}_lo"] > 0, "harder", np.where(gp[f"{pre}_hi"] < 0, "easier", "ns"))
    gp.to_csv(OUT / "gaps.csv", index=False, float_format="%.5f")

    # ---- names and digits: is the gap only about rare words that TTS reads from the spelling?
    import re

    def has_names_or_digits(t: str) -> bool:
        toks = re.findall(r"[A-Za-z0-9][\w'-]*", t)
        return bool(re.search(r"\d", t)) or any(w[0].isupper() for w in toks[1:])

    kind = np.array([has_names_or_digits(first.loc[i, "ref"]) for i in ids])
    rows = []
    for label, sel in (("names_or_digits", kind), ("plain", ~kind)):
        Wk = stats.cluster_weights(np.arange(int(sel.sum())), B, seed=SEED)
        for (v, c, m) in sorted(E):
            if v == "human" or m == W2V or ("human", c, m) not in E:
                continue
            g = _paired(E[(v, c, m)][sel], E[("human", c, m)][sel], n[sel], Wk)
            rows.append({"subset": label, "n_sentences": int(sel.sum()), "model": m, "condition": c, "voice": v,
                         "human_wer": E[("human", c, m)][sel].sum() / n[sel].sum(),
                         "gap": g["diff"], "gap_lo": g["lo"], "gap_hi": g["hi"]})
    pd.DataFrame(rows).to_csv(OUT / "gap_by_sentence_type.csv", index=False, float_format="%.5f")
    summary["n_names_or_digits"] = int(kind.sum())

    # ---- voice spread
    rows = []
    for m in MODELS:
        for c in CONDS:
            keys = [(v, c, m) for v in SYNTH]
            if not all(k in E for k in keys):
                continue
            boots = np.stack([(W @ E[k]) / (W @ n) for k in keys])
            pts = np.array([E[k].sum() / n.sum() for k in keys])
            rb = boots.max(0) / np.maximum(boots.min(0), 1e-9)
            ls = pd.read_csv(LS_TABLES / "voice_spread.csv").set_index(["model", "condition"])
            rows.append({"model": m, "condition": c, "best": SYNTH[int(pts.argmin())], "best_wer": pts.min(),
                         "worst": SYNTH[int(pts.argmax())], "worst_wer": pts.max(),
                         "worst_over_best": pts.max() / pts.min(), "lo": float(np.percentile(rb, 2.5)),
                         "hi": float(np.percentile(rb, 97.5)),
                         "ls_worst_over_best": ls.loc[(m, c), "max_over_min"] if (m, c) in ls.index else np.nan})
    sp = pd.DataFrame(rows)
    sp.to_csv(OUT / "voice_spread.csv", index=False, float_format="%.4f")

    # ---- channel damage
    rows = []
    for (v, c, m) in sorted(E):
        if v == "human" or c == "clean" or not all(k in E for k in [("human", c, m), (v, "clean", m), ("human", "clean", m)]):
            continue
        ds = E[(v, c, m)] - E[(v, "clean", m)]
        dh = E[("human", c, m)] - E[("human", "clean", m)]
        r = _paired(ds, dh, n, W)
        row = {"model": m, "condition": c, "voice": v, "human_damage": dh.sum() / n.sum(),
               "synth_damage": ds.sum() / n.sum(), "damage_error": r["diff"], "lo": r["lo"], "hi": r["hi"], "p": r["p"]}
        ls = ls_dmg.get((m, c, v))
        if ls is not None:
            row.update({"ls_human_damage": ls["human_damage"], "ls_damage_error": ls["damage_error"],
                        "ls_lo": ls["lo"], "ls_hi": ls["hi"]})
        rows.append(row)
    dm = pd.DataFrame(rows)
    mask = dm["voice"].isin(SYNTH)
    dm.loc[mask, "p_bh"] = stats.bh(dm.loc[mask, "p"])
    dm.to_csv(OUT / "channel_damage.csv", index=False, float_format="%.5f")

    # ---- decisions
    rows = []
    for a, b in PAIRS:
        for c in CONDS:
            if not all(k in E for k in [("human", c, a), ("human", c, b)]):
                continue
            hgap = E[("human", c, a)] - E[("human", c, b)]
            hd = _paired(E[("human", c, a)], E[("human", c, b)], n, W)
            h_sig = hd["lo"] > 0 or hd["hi"] < 0
            for v in VOICES:
                if not all(k in E for k in [(v, c, a), (v, c, b)]):
                    continue
                sd = _paired(E[(v, c, a)], E[(v, c, b)], n, W)
                did = _paired(E[(v, c, a)] - E[(v, c, b)], hgap, n, W)
                s_sig = sd["lo"] > 0 or sd["hi"] < 0
                opposite = np.sign(sd["diff"]) != np.sign(hd["diff"])
                r = {"model_a": a, "model_b": b, "condition": c, "voice": v,
                     "kind": "mean" if v == MEAN else "voice",
                     "human_gap": hd["diff"], "human_gap_lo": hd["lo"], "human_gap_hi": hd["hi"],
                     "human_gap_ci_excludes_0": h_sig,
                     "synth_gap": sd["diff"], "synth_gap_lo": sd["lo"], "synth_gap_hi": sd["hi"],
                     "synth_gap_ci_excludes_0": s_sig,
                     "did": did["diff"], "did_lo": did["lo"], "did_hi": did["hi"], "did_p": did["p"],
                     "flip": bool(h_sig and opposite), "confident_flip": bool(h_sig and opposite and s_sig),
                     "lab_says_winner_real_does_not": bool(s_sig and not h_sig),
                     "gap_ratio_lab_over_real": sd["diff"] / hd["diff"] if h_sig else np.nan}
                ls = ls_dec.get((a, b, c, v))
                if ls is not None:
                    r.update({"ls_human_gap": ls["human_gap"], "ls_human_gap_lo": ls["human_gap_lo"],
                              "ls_human_gap_hi": ls["human_gap_hi"], "ls_synth_gap": ls["synth_gap"],
                              "ls_synth_gap_lo": ls["synth_gap_lo"], "ls_synth_gap_hi": ls["synth_gap_hi"],
                              "ls_did": ls["did"], "ls_did_lo": ls["did_lo"], "ls_did_hi": ls["did_hi"],
                              "ls_did_p_holm": ls["did_p_holm"]})
                rows.append(r)
    dc = pd.DataFrame(rows)
    for (a, b, k), g in dc.groupby(["model_a", "model_b", "kind"]):
        dc.loc[g.index, "did_p_holm"] = stats.holm(g["did_p"])
    dc.to_csv(OUT / "decisions.csv", index=False, float_format="%.5f")

    # ---- agreement of FLEURS and LibriSpeech gaps across matching cells
    both = gp[gp["voice"].isin(SYNTH) & gp["ls_gap"].notna()]
    from scipy.stats import spearmanr

    rho = spearmanr(both["gap"], both["ls_gap"])
    summary["gap_agreement"] = {
        "n_cells": len(both), "spearman": float(rho.statistic), "spearman_p": float(rho.pvalue),
        "same_sign": int((np.sign(both["gap"]) == np.sign(both["ls_gap"])).sum()),
        "fleurs_sig_harder": int((both["gap_sig"] == "harder").sum()),
        "fleurs_sig_easier": int((both["gap_sig"] == "easier").sum()),
        "ls_sig_harder": int((both["ls_gap_sig"] == "harder").sum()),
        "ls_sig_easier": int((both["ls_gap_sig"] == "easier").sum()),
        "median_fleurs_minus_ls": float((both["gap"] - both["ls_gap"]).median()),
        "share_fleurs_below_ls": float((both["gap"] < both["ls_gap"]).mean())}
    # how much wider a speaker-cluster CI was than a sentence CI on LibriSpeech (same cells)
    lsc = pd.read_csv(LS_TABLES / "wer_by_cell.csv")
    lsc = lsc[(lsc["set"] == "librispeech") & lsc["voice"].isin(SYNTH) & lsc["gap_lo"].notna()]
    wr = (lsc["gap_hi"] - lsc["gap_lo"]) / (lsc["gap_hi_sentence_boot"] - lsc["gap_lo_sentence_boot"])
    summary["ls_speaker_over_sentence_ci_width"] = {"median": float(wr.median()), "q10": float(wr.quantile(.1)),
                                                     "q90": float(wr.quantile(.9))}
    summary["holm"] = {f"{MSHORT[a]} vs {MSHORT[b]}": int((g["did_p_holm"] < 0.05).sum()) for (a, b), g in
                       dc[dc["kind"] == "voice"].groupby(["model_a", "model_b"])}
    summary["holm_n"] = {f"{MSHORT[a]} vs {MSHORT[b]}": int(len(g)) for (a, b), g in
                         dc[dc["kind"] == "voice"].groupby(["model_a", "model_b"])}
    big = dc[(dc["model_b"] == W2V) & (dc["kind"] == "voice")]
    summary["large_gap_ratio_range"] = [float(big["gap_ratio_lab_over_real"].min()),
                                        float(big["gap_ratio_lab_over_real"].max())]
    summary["large_gap_flips"] = int(big["flip"].sum())
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))

    # ---- figures
    _fig_gaps(gp, plt)
    _fig_decision(dc, plt)
    _fig_human(hw, plt)
    print(json.dumps(summary, indent=1, default=float))


def _fig_gaps(gp: pd.DataFrame, plt) -> None:
    order = SYNTH + [MEAN]
    panels = [(WHISPER, "clean"), (WHISPER, "realistic"), (PARAKEET, "clean"), (PARAKEET, "realistic")]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), sharey=True)
    for ax, (m, c) in zip(axes.ravel(), panels):
        d = gp[(gp["model"] == m) & (gp["condition"] == c)].set_index("voice").reindex(order)
        y = np.arange(len(order))[::-1]
        ax.axvline(0, color=INK2, lw=0.8)
        ax.errorbar(d["ls_gap"] * 100, y + 0.15, xerr=[(d["ls_gap"] - d["ls_gap_lo"]) * 100, (d["ls_gap_hi"] - d["ls_gap"]) * 100],
                    fmt="o", ms=5, color=GRAY, lw=1.5, capsize=0, label="LibriSpeech humans (in-domain)")
        ax.errorbar(d["gap"] * 100, y - 0.15, xerr=[(d["gap"] - d["gap_lo"]) * 100, (d["gap_hi"] - d["gap"]) * 100],
                    fmt="o", ms=5, color=BLUE, lw=1.5, capsize=0, label="FLEURS humans (out-of-domain)")
        ax.set_yticks(y, [SHORT[v] for v in order])
        ax.set_title(f"{MSHORT[m]}, {c}", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("synthetic minus human WER (points)")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Synthetic minus human WER: against FLEURS humans, every voice but MMS looks easier than real speech", x=0.02, ha="left",
                 fontsize=11, color=INK)
    fig.text(0.02, -0.01, "95 % CIs. FLEURS: sentence bootstrap (no speaker ids). LibriSpeech: speaker-cluster bootstrap. "
             "Negative = the voice is easier than real speech.", fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_gap_fleurs_vs_librispeech.png")
    plt.close(fig)


def _fig_decision(dc: pd.DataFrame, plt) -> None:
    order = SYNTH + [MEAN]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), sharey=True)
    for ax, c in zip(axes, ["clean", "realistic"]):
        d = dc[(dc["model_a"] == WHISPER) & (dc["model_b"] == PARAKEET) & (dc["condition"] == c)].set_index("voice").reindex(order)
        y = np.arange(len(order))[::-1]
        h = d["human_gap"].iloc[0]
        hl, hh = d["human_gap_lo"].iloc[0], d["human_gap_hi"].iloc[0]
        lh, ll, lhh = d["ls_human_gap"].iloc[0], d["ls_human_gap_lo"].iloc[0], d["ls_human_gap_hi"].iloc[0]
        ax.axvspan(hl * 100, hh * 100, color=BLUE, alpha=0.10, lw=0)
        ax.axvline(h * 100, color=BLUE, lw=1.2)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.errorbar(d["ls_synth_gap"] * 100, y + 0.15, xerr=[(d["ls_synth_gap"] - d["ls_synth_gap_lo"]) * 100,
                    (d["ls_synth_gap_hi"] - d["ls_synth_gap"]) * 100], fmt="o", ms=5, color=GRAY, lw=1.5,
                    label="lab gap, LibriSpeech sentences")
        ax.errorbar(d["synth_gap"] * 100, y - 0.15, xerr=[(d["synth_gap"] - d["synth_gap_lo"]) * 100,
                    (d["synth_gap_hi"] - d["synth_gap"]) * 100], fmt="o", ms=5, color=ORANGE, lw=1.5,
                    label="lab gap, FLEURS sentences")
        ax.set_yticks(y, [SHORT[v] for v in order])
        ax.set_title(f"{c}: real gap FLEURS {h * 100:+.2f} [{hl * 100:+.2f}, {hh * 100:+.2f}], "
                     f"LibriSpeech {lh * 100:+.2f}", loc="left", fontsize=8.5, color=INK)
        ax.set_xlabel("Whisper WER minus Parakeet WER (points)")
    fig.suptitle("Whisper vs Parakeet: what each voice says (dots) vs FLEURS humans (blue band)", x=0.02, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=8)
    fig.text(0.02, -0.12, "Positive = Parakeet better. Blue line and band: FLEURS human gap and 95 % CI. "
             "Grey dots: the same voices on the LibriSpeech sentences (main run).", fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_decision_whisper_vs_parakeet.png")
    plt.close(fig)


def _fig_human(hw: pd.DataFrame, plt) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.4))
    labels, x = [], 0
    for m in MODELS:
        d = hw[hw["model"] == m]
        for r in d.itertuples():
            ax.errorbar([x - 0.15], [r.ls_wer * 100], yerr=[[(r.ls_wer - r.ls_lo) * 100], [(r.ls_hi - r.ls_wer) * 100]],
                        fmt="o", color=GRAY, ms=5, lw=1.5, label="LibriSpeech humans" if x == 0 else None)
            ax.errorbar([x + 0.15], [r.fleurs_wer * 100], yerr=[[(r.fleurs_wer - r.fleurs_lo) * 100], [(r.fleurs_hi - r.fleurs_wer) * 100]],
                        fmt="o", color=BLUE, ms=5, lw=1.5, label="FLEURS humans" if x == 0 else None)
            labels.append(f"{MSHORT[m]}\n{r.condition.replace('phone+', '+')}")
            x += 1
        x += 0.5
    ticks, x = [], 0
    for m in MODELS:
        for _ in range(int((hw["model"] == m).sum())):
            ticks.append(x)
            x += 1
        x += 0.5
    ax.set_xticks(ticks, labels, fontsize=7.5)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    ax.set_ylabel("human WER, % (log scale)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Human WER: FLEURS vs LibriSpeech, same models and channels", loc="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_human_wer.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["corpus", "run", "merge", "analyze"])
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--model")
    ap.add_argument("--conditions", default="clean,phone,phone+noise10,realistic")
    a = ap.parse_args()
    if a.step == "corpus":
        corpus(a.n)
    elif a.step == "run":
        run(a.model, a.conditions.split(","))
    elif a.step == "merge":
        merge()
    else:
        analyze()


if __name__ == "__main__":
    main()
