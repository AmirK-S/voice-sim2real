"""Phantom follow-up, v2 add-on: Whisper's default pipeline and Silero VAD as baseline guards.

Why (results/followups/LITERATURE-ACTIONS.md, A2, A4, A4b, A6)
- The main run's "33 of 100 noise clips with text" was measured with the raw decoder
  (mlx_whisper.decoding.decode, temperature 0, no fallback, no no-speech gate). An off-the-shelf
  user calls mlx_whisper.transcribe with its defaults: no-speech gate (no_speech_prob > 0.6 and
  avg_logprob < -1), temperature fallback (0.0 to 1.0 when compression ratio > 2.4 or
  avg_logprob < -1), condition_on_previous_text, language detection. Fallback cannot be
  simulated post hoc, so it is run.
- Silero VAD [134] is the published simple guard of Barański et al. [71]; it was missing.

Sets
  main     the main run's noise-only clips (scripts/04_run_experiment.py::noise_only): 100 clips,
           4.00 s, street/rain/crowd/home at -36 dB after the phone line, seeds 10000+k
  randlen  the same recipe with lengths drawn from U(3, 6) s (scripts/09b_phantom_v2.py::randlen)
  v2ns     the 1,320 v2 non-speech clips (scripts/09b_phantom_v2.py)

Steps
  silero              Silero VAD (bundled jit weights, CPU) on every v2 clip and on main/randlen
  defaults --set S    Whisper arms on set S (GPU): raw decoder with scores; transcribe() all
                      defaults; transcribe() defaults with language="en";
                      (main/randlen only) language="en" + word timestamps +
                      hallucination_silence_threshold=2 s
  analyze             tables in results/followups/phantom/v2/ (gate_*.csv, silero_*.csv,
                      summary_gate_silero.json)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from sim2real import asr, channel, data, guards, stats

spec = importlib.util.spec_from_file_location("ph2", ROOT / "scripts" / "09b_phantom_v2.py")
ph2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ph2)

SR = channel.SR
OUT, SHARDS = ph2.OUT, ph2.SHARDS
WSH = SHARDS / "whisper_defaults"  # not under whisper/: 09b globs whisper/*__*.csv
SILERO_DEFAULT = dict(threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=100, speech_pad_ms=30)


# ----------------------------------------------------------------------------- audio sets

def main_set() -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Exactly scripts/04_run_experiment.py::noise_only."""
    groups = ["street", "rain", "crowd", "home"]
    arrays, recs = [], []
    for k in range(100):
        g = groups[k % 4]
        bed = channel.noise_bed(g, 4 * SR, seed=10_000 + k)
        lvl = channel.db(channel.power(channel.phone_8k(bed)))
        bed = bed * 10 ** ((-36.0 - lvl) / 20) + channel._floor_noise(len(bed), k)
        arrays.append(channel.telephone(bed.astype(np.float32)))
        recs.append({"clip": k, "noise_group": g, "duration_s": 4.0})
    return pd.DataFrame(recs), arrays


def randlen_set() -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Exactly scripts/09b_phantom_v2.py::randlen."""
    rng = np.random.default_rng(4242)
    groups = ["street", "rain", "crowd", "home"]
    rows, arrays = [], []
    for k in range(100):
        dur = float(np.round(rng.uniform(3.0, 6.0), 2))
        g = groups[k % 4]
        bed = channel.noise_bed(g, int(round(dur * SR)), seed=10_000 + k)
        bed = bed * 10 ** ((-36.0 - ph2.band_level(bed)) / 20) + channel._floor_noise(len(bed), k)
        arrays.append(channel.telephone(bed.astype(np.float32)))
        rows.append({"clip": k, "noise_group": g, "duration_s": dur})
    return pd.DataFrame(rows), arrays


def v2ns_set() -> tuple[pd.DataFrame, list[np.ndarray]]:
    d = ph2.clip_table()
    d = d[d["group"] == "nonspeech"].reset_index(drop=True)
    return d, [ph2.clip_audio(r) for _, r in d.iterrows()]


SETS = {"main": main_set, "randlen": randlen_set, "v2ns": v2ns_set}


# ----------------------------------------------------------------------------- silero

def _silero():
    import torch
    from silero_vad import load_silero_vad

    torch.set_num_threads(2)
    return load_silero_vad()  # bundled jit weights, no download


def silero_feats(model, x: np.ndarray) -> dict:
    import torch
    from silero_vad.utils_vad import get_speech_timestamps_from_probs

    x = np.ascontiguousarray(np.asarray(x, np.float32))
    if len(x) < 512:
        x = np.pad(x, (0, 512 - len(x)))
    with torch.no_grad():
        p = model.audio_forward(torch.from_numpy(x), sr=SR)[0].numpy()
    ts = get_speech_timestamps_from_probs(list(map(float, p)), sampling_rate=SR, return_seconds=False,
                                          **SILERO_DEFAULT)
    speech = sum(t["end"] - t["start"] for t in ts) / SR
    return {"silero_max_p": float(p.max()), "silero_mean_p": float(p.mean()),
            "silero_speech_s": float(speech), "silero_segments": len(ts)}


def silero_step() -> None:
    model = _silero()
    t0 = time.time()
    d = ph2.clip_table()
    rows = []
    for i, (_, r) in enumerate(d.iterrows()):
        rows.append({"key": r["key"], **silero_feats(model, ph2.clip_audio(r))})
        if i % 1000 == 0:
            print(f"silero v2 {i}/{len(d)} ({time.time() - t0:.0f} s)", flush=True)
    pd.DataFrame(rows).to_csv(SHARDS / "silero.csv", index=False)
    for name in ("main", "randlen"):
        meta, arrays = SETS[name]()
        meta = meta.assign(**pd.DataFrame([silero_feats(model, a) for a in arrays]))
        meta.to_csv(SHARDS / f"silero_{name}.csv", index=False)
    print(f"silero done in {time.time() - t0:.0f} s")


# ----------------------------------------------------------------------------- whisper arms

def _transcribe(a: np.ndarray, **kw) -> dict:
    import mlx_whisper

    o = mlx_whisper.transcribe(np.asarray(a, np.float32), path_or_hf_repo=asr.WHISPER_ID, verbose=None, **kw)
    segs = o.get("segments", [])
    return {"text": o["text"].strip(), "language": o.get("language", ""), "n_segments": len(segs),
            "max_temperature": max((s["temperature"] for s in segs), default=np.nan),
            "seg_no_speech_prob_max": max((s["no_speech_prob"] for s in segs), default=np.nan),
            "seg_avg_logprob_min": min((s["avg_logprob"] for s in segs), default=np.nan),
            "seg_compression_ratio_max": max((s["compression_ratio"] for s in segs), default=np.nan)}


def raw_melpad(arrays: list[np.ndarray]) -> list[dict]:
    """Raw decoder (temperature 0, no gate, no fallback) fed the window the way
    mlx_whisper.transcribe builds it: log-mel of the clip only, then zero-padded in the log-mel
    domain to 3,000 frames. asr._whisper_batch instead keeps the log-mel of 30 s of zero audio
    after the clip. This arm isolates the input difference from fallback and prompting."""
    import mlx.core as mx
    from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
    from mlx_whisper.decoding import DecodingOptions, decode

    model = asr._whisper_model()
    opts = DecodingOptions(language="en", task="transcribe", temperature=0.0, fp16=True)
    out = []
    for s in range(0, len(arrays), asr.WHISPER_BATCH):
        mels = []
        for a in arrays[s : s + asr.WHISPER_BATCH]:
            mel = log_mel_spectrogram(asr._as16k(a), n_mels=model.dims.n_mels, padding=N_SAMPLES)
            content = mel.shape[-2] - N_FRAMES
            mels.append(pad_or_trim(mel[:content], N_FRAMES, axis=-2).astype(mx.float16))
        for r in decode(model, mx.stack(mels), opts):
            out.append({"text": r.text.strip(), "no_speech_prob": float(r.no_speech_prob),
                        "avg_logprob": float(r.avg_logprob), "compression_ratio": float(r.compression_ratio)})
    return out


ARMS = {
    "transcribe_defaults": {},
    "transcribe_defaults_en": {"language": "en"},
    "transcribe_en_hst2": {"language": "en", "word_timestamps": True, "hallucination_silence_threshold": 2.0},
}


def defaults_step(name: str) -> None:
    WSH.mkdir(parents=True, exist_ok=True)
    meta, arrays = SETS[name]()
    raw_p = WSH / f"{name}_raw.csv"
    if not raw_p.exists():
        res = guards.whisper_scored(arrays)
        meta.assign(**pd.DataFrame(res)).to_csv(raw_p, index=False)
        print(f"[{name}] raw done", flush=True)
    mp = WSH / f"{name}_raw_melpad.csv"
    if not mp.exists():
        meta.assign(**pd.DataFrame(raw_melpad(arrays))).to_csv(mp, index=False)
        print(f"[{name}] raw_melpad done", flush=True)
    arms = ARMS if name != "v2ns" else {k: v for k, v in ARMS.items() if k != "transcribe_en_hst2"}
    for arm, kw in arms.items():
        p = WSH / f"{name}_{arm}.csv"
        if p.exists():
            continue
        t0 = time.time()
        res = [_transcribe(a, **kw) for a in arrays]
        tmp = p.with_suffix(".tmp")
        meta.assign(**pd.DataFrame(res)).to_csv(tmp, index=False)
        tmp.replace(p)
        print(f"[{name}] {arm}: {len(arrays)} clips in {time.time() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------- analysis

def nonempty(texts) -> np.ndarray:
    return np.array([bool(asr.normalize(str(x))) for x in texts])


def rate_row(label: str, ne: np.ndarray, extra: dict | None = None, clusters=None) -> dict:
    k, n = int(ne.sum()), len(ne)
    lo, hi = stats.clopper_pearson(k, n)
    row = {"arm": label, "n": n, "with_text": k, "rate": k / n, "cp_lo": lo, "cp_hi": hi}
    if clusters is not None:  # cluster bootstrap when one clip feeds several channel rows
        W = stats.cluster_weights(np.asarray(clusters), 2000, seed=ph2.SEED)
        b = W @ ne.astype(float) / W.sum(1)
        row.update({"boot_lo": float(np.quantile(b, 0.025)), "boot_hi": float(np.quantile(b, 0.975))})
    row.update(extra or {})
    return row


def analyze() -> None:
    summary: dict = {}
    # ---- A. noise-only sets: raw decoder, raw + default gate post hoc, transcribe arms, Silero
    rows = []
    for name in ("main", "randlen", "v2ns"):
        rp = WSH / f"{name}_raw.csv"
        if not rp.exists():
            continue
        raw = pd.read_csv(rp, keep_default_na=False)
        clusters = raw["id"].to_numpy() if name == "v2ns" else None
        ne_raw = nonempty(raw["text"])
        nsp = pd.to_numeric(raw["no_speech_prob"]).to_numpy()
        lp = pd.to_numeric(raw["avg_logprob"]).to_numpy()
        cr = pd.to_numeric(raw["compression_ratio"]).to_numpy()
        gate = guards.whisper_builtin(nsp, lp)
        info = {"set": name, "nsp_median": float(np.median(nsp)), "nsp_max": float(np.max(nsp)),
                "gate_fires": int(gate.sum()), "share_logprob_below_-1": float((lp < -1).mean()),
                "share_compression_above_2.4": float((cr > 2.4).mean())}
        rows.append({"set": name, **rate_row("raw decoder (no gate, no fallback)", ne_raw, info, clusters)})
        rows.append({"set": name, **rate_row("raw decoder + default gate (post hoc)", ne_raw & ~gate, None, clusters)})
        sil = None
        if name in ("main", "randlen"):
            sil = pd.read_csv(SHARDS / f"silero_{name}.csv")
        else:
            s = pd.read_csv(SHARDS / "silero.csv")
            sil = raw[["key"]].merge(s, on="key", how="left")
        sdrop = sil["silero_speech_s"].to_numpy() <= 0
        rows.append({"set": name, **rate_row("raw decoder + Silero default (drop if no speech segment)",
                                              ne_raw & ~sdrop, {"silero_triggers_on_noise": int((~sdrop).sum())},
                                              clusters)})
        mp = WSH / f"{name}_raw_melpad.csv"
        if mp.exists():
            d = pd.read_csv(mp, keep_default_na=False)
            rows.append({"set": name, **rate_row("raw decoder, transcribe-style log-mel zero padding", nonempty(d["text"]),
                                                  {"same_as_raw": int((nonempty(d["text"]) == ne_raw).sum())}, clusters)})
        for arm in ARMS:
            p = WSH / f"{name}_{arm}.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p, keep_default_na=False)
            ne = nonempty(d["text"])
            lang = d["language"].value_counts().to_dict()
            ex = {"languages": json.dumps(lang), "fallback_used": int((pd.to_numeric(d["max_temperature"]) > 0).sum()),
                  "same_as_raw": int((ne == ne_raw).sum())}
            rows.append({"set": name, **rate_row(f"mlx_whisper.transcribe {arm}", ne, ex, clusters)})
            rows.append({"set": name, **rate_row(f"mlx_whisper.transcribe {arm} + Silero default", ne & ~sdrop, None,
                                                  clusters)})
            # emergency keywords invented
            kw = [data.find_keywords(str(t)) for t in d["text"]]
            rows[-2]["kw_clips"] = int(sum(bool(k) for k in kw))
        kw_raw = [data.find_keywords(str(t)) for t in raw["text"]]
        rows[[i for i, r in enumerate(rows) if r["set"] == name and r["arm"].startswith("raw decoder (")][0]]["kw_clips"] = \
            int(sum(bool(k) for k in kw_raw))
        hours = float(raw["duration_s"].astype(float).sum() / 3600) if "duration_s" in raw else np.nan
        summary.setdefault("hours_of_noise", {})[name] = hours
    gate_tab = pd.DataFrame(rows)
    gate_tab.to_csv(OUT / "gate_noise_only_arms.csv", index=False)
    summary["noise_only_arms"] = gate_tab.to_dict("records")
    # top phrases per arm on main / randlen
    phr = []
    for name in ("main", "randlen", "v2ns"):
        for arm in ["raw", "raw_melpad"] + list(ARMS):
            p = WSH / f"{name}_{arm}.csv"
            if p.exists():
                d = pd.read_csv(p, keep_default_na=False)
                h = pd.Series([asr.normalize(str(x)) for x in d["text"]])
                for t, c in h[h != ""].value_counts().head(8).items():
                    phr.append({"set": name, "arm": arm, "text": t[:80], "clips": int(c)})
    pd.DataFrame(phr).to_csv(OUT / "gate_top_phrases.csv", index=False)

    # ---- B. v2 full design: Whisper gate and Silero on speech and non-speech, every model
    feats = pd.read_csv(SHARDS / "features.csv", keep_default_na=False)
    for c in ("level_db", "gain_db", "duration_s", "t0", "t1"):
        feats[c] = pd.to_numeric(feats[c], errors="coerce")
    feats["short"] = feats["short"].astype(str) == "True"
    sil = pd.read_csv(SHARDS / "silero.csv")
    n_shards = feats.groupby(["group", "cond"]).ngroups
    present = [m for m in ph2.MODELS if len(list((SHARDS / ph2.SHORT[m]).glob("*__*.csv"))) == n_shards]
    grows = []
    for m in present:
        name = ph2.SHORT[m]
        t = ph2.load_model(m, feats).merge(sil, on="key", how="left").reset_index(drop=True)
        assert t["silero_speech_s"].notna().all()
        Wb = ph2.boot_weights(t, ph2.B, ph2.SEED + 11)
        rules = {"none": np.zeros(len(t), bool),
                 "Silero default (no speech segment)": t["silero_speech_s"].to_numpy() <= 0}
        if m == asr.MODELS[0]:
            g = guards.whisper_builtin(t["no_speech_prob"].to_numpy(float), t["avg_logprob"].to_numpy(float))
            rules["Whisper default gate (nsp>0.6 & logprob<-1)"] = g
            rules["Whisper default gate OR Silero default"] = g | rules["Silero default (no speech segment)"]
            sp = t["group"] != "nonspeech"
            summary.setdefault("whisper_v2_scores", {}).update({
                "nsp_median_nonspeech": float(t.loc[~sp, "no_speech_prob"].median()),
                "nsp_max_nonspeech": float(t.loc[~sp, "no_speech_prob"].max()),
                "nsp_median_speech": float(t.loc[sp, "no_speech_prob"].median()),
                "nsp_max_all": float(t["no_speech_prob"].max()),
                "gate_fires_nonspeech": int(g[~sp.to_numpy()].sum()), "gate_fires_speech": int(g[sp.to_numpy()].sum())})
        for rname, drop in rules.items():
            mt = ph2.metrics(t, drop)
            Db = np.repeat(drop[:, None], ph2.B, axis=1)
            mb = ph2.metrics(t, Db, Wb)
            row = {"model": name, "guard": rname, **mt}
            for k, v in mb.items():
                row[f"{k}_lo"], row[f"{k}_hi"] = float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))
            ns = t["group"].to_numpy() == "nonspeech"
            row["nonspeech_clips_dropped"] = float(drop[ns].mean())
            c = t[(t["group"] == "callers") & (t["kw_ref"] > 0)]
            lost = c["call_detected"].to_numpy() & drop[c.index]
            row.update({"calls_detected": int(c["call_detected"].sum()), "calls_lost": int(lost.sum()),
                        "scripts_with_a_lost_call": int(c.loc[lost, "id"].nunique()),
                        "keyword_scripts": int(c["id"].nunique())})
            ow = t[t["group"] == "oneword"]
            okd = ow["word_ok"].to_numpy()
            k_, n_ = int((okd & drop[ow.index]).sum()), int(okd.sum())
            row.update({"oneword_correct_n": n_, "oneword_correct_dropped_n": k_})
            grows.append(row)
        # cross-fitted Silero threshold (max speech prob, speech seconds), v2 selection rule, point estimates
        for sname, col in (("Silero max prob (tuned)", "silero_max_p"), ("Silero speech s (tuned)", "silero_speech_s")):
            s = t[col].to_numpy(float)
            for vname, sg in ph2.VARIANTS_PUBLIC.items():
                dd = np.zeros(len(t), bool)
                chosen = []
                for f in (0, 1):
                    train = (t["fold"].to_numpy() != f) & t["group"].isin(["nonspeech", *sg]).to_numpy()
                    cs = [("none", "", np.zeros(len(t), bool))] + [(sname, f"{th:.4g}", s < th)
                                                                     for th in ph2.grid(s, train, 120)]
                    p = ph2.pick_simple(t, cs, t["fold"].to_numpy() != f, sg)
                    chosen.append(p[1])
                    app = t["fold"].to_numpy() == f
                    dd[app] = p[2][app]
                grows.append({"model": name, "guard": f"{sname}, chosen on {vname}", "thresholds": " | ".join(chosen),
                              **ph2.metrics(t, dd), "nonspeech_clips_dropped": float(dd[t["group"].to_numpy() == "nonspeech"].mean())})
        # Silero false triggers by kind
        ns = t[t["group"] == "nonspeech"]
        for (kind, sh), g in ns.groupby(["kind", "short"]):
            summary.setdefault("silero_trigger_by_kind", []).append(
                {"model": name, "kind": kind, "short": bool(sh), "n": len(g),
                 "silero_triggers": int((g["silero_speech_s"] > 0).sum()),
                 "phantoms": int(g["nonempty"].sum()),
                 "phantoms_left_after_silero": int((g["nonempty"] & (g["silero_speech_s"] > 0)).sum())})
    gt = pd.DataFrame(grows)
    gt.to_csv(OUT / "silero_and_gate_v2.csv", index=False)
    pd.DataFrame(summary.get("silero_trigger_by_kind", [])).to_csv(OUT / "silero_by_kind_v2.csv", index=False)
    summary.pop("silero_trigger_by_kind", None)
    summary["v2_guards"] = gt.to_dict("records")
    (OUT / "summary_gate_silero.json").write_text(json.dumps(summary, indent=1, default=float))
    pd.set_option("display.width", 250)
    print(gate_tab.drop(columns=[c for c in gate_tab if c.startswith("languages")]).to_string())
    cols = ["model", "guard", "phantom_before", "phantom_after", "phantom_after_lo", "phantom_after_hi",
            "words_lost_human", "words_lost_human_hi", "words_lost_segments", "oneword_correct_dropped",
            "kw_lost_callers", "calls_lost", "thresholds"]
    print(gt[[c for c in cols if c in gt]].round(4).to_string())


ph2.VARIANTS_PUBLIC = {"human full clips": ("ls_human", "fl_human"),
                       "human full clips + 1-3 word segments": ("ls_human", "fl_human", "segments")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["silero", "defaults", "analyze"])
    ap.add_argument("--set", default="main", choices=list(SETS))
    a = ap.parse_args()
    {"silero": silero_step, "analyze": analyze, "defaults": lambda: defaults_step(a.set)}[a.step]()


if __name__ == "__main__":
    main()
