"""Phantom follow-up, v2 after review: matched non-speech, random speech level, short utterances.

What changed from scripts/09_phantom.py (v1), and why
- Matched noise-only clips (review B1): the exact speech-side chain with the speech removed:
  250 ms padding and the stationary floor of channel.prepare, the same random gain as the speech
  clips, then the noise bed of phone+noise10 (street/rain/crowd) or realistic (street) at the level
  a talker at that gain would get at 10 dB post-channel SNR, then G.711 or G.711 + 3 % bursty loss
  with concealment. A guard can no longer separate speech from noise by channel or level alone.
- Random speech level (B1): every speech clip gets a gain drawn from U(-20, +10) dB after
  channel.prepare (active level -46 to -16 dBov instead of always -26), before the channel.
- Channels (B1, I9): non-speech clips go through "phone" (G.711) and "phone+loss" (the line part
  of the realistic chain: G.711, 3 % bursty loss, Appendix I concealment, no added noise). No
  clean channel on either side. Comfort noise goes through the phone channel only.
- More kinds (I9): ESC-50 breathing, sneezing, crying_baby; residual echo of the bot's own voice
  (TTS prompts, -45 to -30 dB, echo path); DTMF digit tones; line clicks; comfort noise at -78 to
  -60 dB. Each kind has 20 long clips (3 to 10 s) and 20 short ones (0.5 to 2 s, streaming
  VAD segment lengths).
- Leakage (M2): ESC-50 sources are split by their official fold (1, 3, 5 vs 2, 4) before chaining;
  a clip only uses sources of its own half, and its cross-fitting fold is that half. Threshold
  grids are built from the training fold only.
- Selection on human speech only (I5): the guard is chosen on non-speech vs the human LibriSpeech
  and FLEURS clips. TTS callers are reported separately.
- Short utterances (I4, I6): (a) one-word TTS clips ("help", "fire", "police", "ambulance", "yes",
  "no", "hello", "stop") from the 7 voices, 2 draws each (gain, noise) per condition; (b) 1 to 3 word
  segments cut from every human LibriSpeech and FLEURS clip with Parakeet token timestamps (span
  of words Parakeet got right on the clean clip, +-0.1 s), re-prepared like any clip.
- Nested bootstrap (M3): the cluster bootstrap re-runs the whole cross-fitted selection in each
  replicate, so the CIs include the variability of choosing the rule.
- Script-level bound on calls lost (I5 of review 1): rule of three over keyword scripts, not
  over script x voice x condition readings.
- Noise-only length check at random lengths (I10): the main run's 100-clip check (4.00 s) with
  lengths drawn from U(3, 6) s.

Steps
  segments          Parakeet word timestamps on the clean human clips (GPU, short)
  features          clip list, audio, VAD features (CPU)
  run --model M     decode with scores (GPU), one shard per group x condition
  randlen --model M noise-only check at random lengths
  nsp               check the no-speech token id and probability in mlx-whisper (one clip)
  analyze           tables, figures, numbers for NOTES-v2 in results/followups/phantom/v2/
"""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import re
import sys
import time
import zlib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import signal

from sim2real import asr, channel, data, guards, stats, tts
from sim2real.audio import resample
from sim2real.datasets import esc50_meta, load
from sim2real.paths import RAW, RESULTS

spec = importlib.util.spec_from_file_location("ph1", ROOT / "scripts" / "09_phantom.py")
ph1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ph1)

OUT = RESULTS / "followups" / "phantom" / "v2"
SHARDS = RAW / "followups" / "phantom_v2"
SR = channel.SR
SPEECH_CONDS = ["phone+noise10", "realistic"]
NS_CHANNELS = ["phone", "phone+loss"]
MODELS = list(asr.MODELS)
SHORT = ph1.SHORT
B = 1000
SEED = 1
BUDGET = 0.0025
GAIN_DB = (-20.0, 10.0)
PER_KIND = 40  # 20 long + 20 short

ESC_KINDS = dict(guards.ESC_KINDS, breathing=["breathing"], sneezing=["sneezing"], crying_baby=["crying_baby"])
SYNTH_KINDS = ["line_hiss", "near_silence", "digital_silence", "comfort_noise", "bot_echo", "dtmf", "line_clicks"]
KINDS = list(ESC_KINDS) + SYNTH_KINDS + ["matched_noise"]
LEVELS = {"esc": (-46.0, -26.0), "line_hiss": (-55.0, -35.0), "near_silence": (-80.0, -60.0),
          "comfort_noise": (-78.0, -60.0), "bot_echo": (-45.0, -30.0), "dtmf": (-20.0, -10.0),
          "line_clicks": (-30.0, -10.0), "digital_silence": (None, None), "matched_noise": (None, None)}
BOT_PROMPTS = [
    "Nine one one, what is the address of your emergency?",
    "Can you tell me exactly what happened?",
    "Is the person breathing?",
    "Stay on the line with me, help is on the way.",
    "What is the phone number you are calling from?",
    "Is anyone hurt?",
    "Okay, I am sending an ambulance now.",
    "Can you see any smoke or flames?",
    "Are you in a safe place right now?",
    "Please repeat the street name for me.",
]
ONE_WORDS = ["help", "fire", "police", "ambulance", "yes", "no", "hello", "stop"]
DTMF = {"1": (697, 1209), "2": (697, 1336), "3": (697, 1477), "4": (770, 1209), "5": (770, 1336),
        "6": (770, 1477), "7": (852, 1209), "8": (852, 1336), "9": (852, 1477), "0": (941, 1336),
        "*": (941, 1209), "#": (941, 1477)}


def crc(s: str) -> int:
    return zlib.crc32(s.encode()) & 0x7FFFFFFF


# ----------------------------------------------------------------------------- non-speech audio

def nonspeech_plan() -> list[dict]:
    rng = np.random.default_rng(SEED)
    out = []
    for kind in KINDS:
        lo, hi = LEVELS["esc" if kind in ESC_KINDS else kind]
        for k in range(PER_KIND):
            short = k % 2 == 1
            dur = float(np.round(rng.uniform(0.5, 2.0) if short else rng.uniform(3.0, 10.0), 2))
            lvl = float(np.round(rng.uniform(lo, hi), 1)) if lo is not None else float("nan")
            out.append({"clip": f"{kind}-{k:02d}", "kind": kind, "duration_s": dur, "short": short,
                        "level_db": lvl, "gain_db": float(np.round(rng.uniform(*GAIN_DB), 2)),
                        "half": (k // 2) % 2, "seed": int(rng.integers(0, 2**31 - 1))})
    return out


@lru_cache(maxsize=1)
def esc_bank() -> dict[tuple[str, int], list[np.ndarray]]:
    """ESC-50 clips per (kind, half): half 0 = official folds 1, 3, 5; half 1 = folds 2, 4."""
    meta = esc50_meta()
    bank: dict = {}
    for kind, cats in ESC_KINDS.items():
        for half in (0, 1):
            folds = [1, 3, 5] if half == 0 else [2, 4]
            sel = meta[meta["category"].isin(cats) & meta["fold"].isin(folds)]
            clips = []
            for p in sorted(sel["path"]):
                a, sr = load(p)
                a = channel._strip_silence(resample(a, sr, SR))
                if len(a) >= SR // 4:
                    clips.append(a.astype(np.float64))
            bank[(kind, half)] = clips
    return bank


@lru_cache(maxsize=None)
def bot_prompt_audio(i: int) -> np.ndarray:
    """Prompt i % 10 read by voice i % 7 (cached in data/audio/bot_prompts/)."""
    import soundfile as sf

    p = ROOT / "data" / "audio" / "bot_prompts" / f"{i:02d}.wav"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        a = tts.synthesize(BOT_PROMPTS[i % len(BOT_PROMPTS)], tts.VOICES[i % len(tts.VOICES)])
        sf.write(p, a, SR, subtype="PCM_16")
    return ph1.read16k(str(p.relative_to(ROOT))).astype(np.float64)


def band_level(x: np.ndarray) -> float:
    return channel.db(channel.power(channel.phone_8k(x)))


def nonspeech_audio(rec: dict) -> np.ndarray:
    """16 kHz audio before the line (for matched_noise: the whole pre-line mix)."""
    kind, n, rng = rec["kind"], int(round(rec["duration_s"] * SR)), np.random.default_rng(rec["seed"])
    s = rec["seed"]
    if kind in ESC_KINDS:
        bed = guards._chain(esc_bank()[(kind, rec["half"])], n, rng, xf=min(800, n // 4))
        bed = bed * 10 ** ((rec["level_db"] - band_level(bed)) / 20) + channel._floor_noise(n, s)
    elif kind == "digital_silence":
        return np.zeros(n, np.float32)
    elif kind == "near_silence":
        bed = guards._rms_to(channel._floor_noise(n, s), rec["level_db"])
    elif kind == "line_hiss":
        bed = guards._rms_to(rng.standard_normal(n), rec["level_db"])
    elif kind == "comfort_noise":  # shaped like a street background (low-passed pink), very low level
        y = signal.lfilter([1.0], [1.0, -0.97], rng.standard_normal(n))
        bed = guards._rms_to(y, rec["level_db"])
    elif kind == "bot_echo":  # the bot's prompt coming back: 30 to 120 ms delay, band-limited, faint
        a = bot_prompt_audio(int(rng.integers(0, 70)))
        a = channel._bandlimit_16k(a)
        d = int(rng.uniform(0.03, 0.12) * SR)
        a = np.concatenate([np.zeros(d), a])
        reps = int(np.ceil(n / len(a)))
        a = np.tile(a, reps)[:n] if len(a) < n else a[int(rng.integers(0, len(a) - n + 1)):][:n]
        a = a * 10 ** ((rec["level_db"] - channel.active_level(a)[0]) / 20) if channel.power(a) > 0 else a
        bed = a + channel._floor_noise(n, s)
    elif kind == "dtmf":
        bed = channel._floor_noise(n, s)
        t, pos = np.arange(int(0.12 * SR)) / SR, int(rng.uniform(0.05, 0.3) * SR)
        while pos < n - int(0.07 * SR):
            dur = int(rng.uniform(0.07, 0.12) * SR)
            f1, f2 = DTMF[str(rng.choice(list(DTMF)))]
            tone = np.sin(2 * np.pi * f1 * t[:dur]) + np.sin(2 * np.pi * f2 * t[:dur])
            tone *= 10 ** (rec["level_db"] / 20) / np.sqrt(np.mean(tone ** 2))
            seg = slice(pos, min(n, pos + dur))
            bed[seg] += tone[: seg.stop - seg.start]
            pos += dur + int(rng.uniform(0.06, 0.4) * SR)
    elif kind == "line_clicks":
        bed = channel._floor_noise(n, s) + guards._rms_to(rng.standard_normal(n), rng.uniform(-62, -50))
        for _ in range(int(rng.integers(1, 6))):
            at = int(rng.integers(0, max(1, n - 400)))
            L = int(rng.integers(40, 400))
            click = rng.standard_normal(L) * np.exp(-np.arange(L) / (L / 5))
            click *= 10 ** ((rec["level_db"] + rng.uniform(-6, 0)) / 20) / max(np.max(np.abs(click)), 1e-9)
            bed[at : at + L] += click[: len(bed[at : at + L])]
    elif kind == "matched_noise":
        return matched_noise_premix(rec, rec["cond"])
    else:
        raise ValueError(kind)
    peak = float(np.max(np.abs(bed)))
    if peak > 0.99:
        bed = bed * 0.99 / peak
    return bed.astype(np.float32)


def matched_noise_premix(rec: dict, cond: str) -> np.ndarray:
    """Speech-side chain without the speech: pad + floor (channel.prepare on silence), random gain,
    noise at the level a talker at that gain gets at 10 dB post-channel SNR."""
    s = rec["seed"]
    n = int(round(rec["duration_s"] * SR))  # total length incl. the 2 x 250 ms padding
    g = 10 ** (rec["gain_db"] / 20)
    x = channel._floor_noise(n, s) * g
    group = channel.noise_group(s) if cond == "phone+noise10" else "street"
    bed = channel.noise_bed(group, n, s)
    target = channel.TARGET_DBOV + rec["gain_db"] - 10.0  # post-line noise power
    bed = bed * 10 ** ((target - band_level(bed)) / 20)
    y = x + bed
    peak = float(np.max(np.abs(y)))
    if peak > 0.99:
        y *= 0.99 / peak
    return y.astype(np.float32)


def line(x: np.ndarray, ch: str, seed: int) -> np.ndarray:
    if ch in ("phone", "phone+noise10"):
        return channel.telephone(x)
    return channel.phone_with_loss(x, 0.03, seed, burst=3, meta={})


# ----------------------------------------------------------------------------- speech audio

def gained(x: np.ndarray, gain_db: float) -> np.ndarray:
    y = np.asarray(x, np.float64) * 10 ** (gain_db / 20)
    peak = float(np.max(np.abs(y)))
    if peak > 0.99:
        y *= 0.99 / peak
    return y.astype(np.float32)


def speech_clip(x16: np.ndarray, cond: str, seed: int, gain_db: float) -> np.ndarray:
    x = gained(channel.prepare(x16, seed=seed), gain_db)
    return channel.apply(cond, x, seed=seed, meta={})


def segment_table() -> pd.DataFrame:
    return pd.read_csv(SHARDS / "segments.csv", keep_default_na=False)


def human_rows() -> pd.DataFrame:
    m = pd.read_csv(ph1.MAIN_MANIFEST, keep_default_na=False)
    ls = m[(m["set"] == "librispeech") & (m["voice"] == "human")].copy()
    ls["group"], ls["cluster"] = "ls_human", "ls" + ls["speaker"].astype(str)
    fl = pd.read_csv(ph1.FLEURS_MANIFEST, keep_default_na=False)
    fl = fl[fl["voice"] == "human"].copy()
    fl["group"], fl["cluster"] = "fl_human", fl["id"]
    fl["text"] = [ph1.fleurs.scoring_ref(t) for t in fl["text"]]
    return pd.concat([ls[["group", "id", "cluster", "text", "path"]], fl[["group", "id", "cluster", "text", "path"]]],
                     ignore_index=True)


def segments() -> None:
    """Parakeet word timestamps on the prepared clean human clips; one 1-3 word span per clip,
    chosen among words Parakeet recognized correctly (seeded by the clip id)."""
    import mlx.core as mx
    from parakeet_mlx.audio import get_logmel

    SHARDS.mkdir(parents=True, exist_ok=True)
    model = asr._parakeet()
    rows = []
    norm = lambda w: re.sub(r"[^a-z0-9']", "", w.lower())
    for r in human_rows().itertuples():
        x = channel.prepare(ph1.read16k(r.path), seed=ph1.utt_seed(r.id))
        res = model.generate(get_logmel(mx.array(asr._as16k(x)), model.preprocessor_config))[0]
        words = []  # (text, start, end)
        for t in res.tokens:
            if t.text.startswith(" ") or not words:
                words.append([t.text.strip(), t.start, t.end])
            else:
                words[-1][0] += t.text
                words[-1][2] = t.end
        hw = [norm(w[0]) for w in words]
        rw = [norm(w) for w in r.text.split()]
        sm = difflib.SequenceMatcher(a=rw, b=hw, autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
        rng = np.random.default_rng(crc(r.id))
        k = int(rng.integers(1, 4))
        cands = [(b.b + j, kk) for b in blocks for kk in (k, 2, 1) if b.size >= kk for j in range(b.size - kk + 1)]
        if not cands:
            continue
        kk = cands[0][1]
        starts = [c[0] for c in cands if c[1] == kk]
        j = int(rng.choice(starts))
        span = words[j : j + kk]
        t0 = max(0.0, span[0][1] - 0.1)
        t1 = min(len(x) / SR, span[-1][2] + 0.1)
        if j > 0:  # do not reach into the previous word
            t0 = max(t0, words[j - 1][2])
        if j + kk < len(words):
            t1 = min(t1, words[j + kk][1])
        rows.append({"group": r.group, "src_id": r.id, "cluster": r.cluster, "n_words": kk,
                     "ref": " ".join(w[0] for w in span), "t0": round(t0, 3), "t1": round(t1, 3),
                     "path": r.path})
    pd.DataFrame(rows).to_csv(SHARDS / "segments.csv", index=False)
    print(f"segments: {len(rows)}")


def oneword_path(word: str, voice: str) -> Path:
    return ROOT / "data" / "audio" / "oneword" / voice.replace(":", "_") / f"{word}.wav"


def oneword_audio(word: str, voice: str) -> np.ndarray:
    import soundfile as sf

    p = oneword_path(word, voice)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        text = word.capitalize() + "!"
        sf.write(p, tts.synthesize(text, voice), SR, subtype="PCM_16")
    return ph1.read16k(str(p.relative_to(ROOT)))


# ----------------------------------------------------------------------------- clip table

def clip_table() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED + 7)
    for r in nonspeech_plan():
        chans = (["phone"] if r["kind"] == "comfort_noise" else
                 SPEECH_CONDS if r["kind"] == "matched_noise" else NS_CHANNELS)
        for ch in chans:
            rows.append({"group": "nonspeech", "id": r["clip"], "voice": "", "cond": ch, "kind": r["kind"],
                         "cluster": r["clip"], "fold": r["half"], "ref": "", "path": "", "level_db": r["level_db"],
                         "duration_s": r["duration_s"], "short": r["short"], "gain_db": r["gain_db"],
                         "seed": r["seed"], "t0": np.nan, "t1": np.nan})
    hum = human_rows()
    m = pd.read_csv(ph1.MAIN_MANIFEST, keep_default_na=False)
    em = m[(m["set"] == "emergency") & ~m["voice"].str.endswith("@rate")]
    seg = segment_table()
    for cond in SPEECH_CONDS:
        for r in hum.itertuples():
            rows.append({"group": r.group, "id": r.id, "voice": "human", "cond": cond, "kind": "speech",
                         "cluster": r.cluster, "ref": r.text, "path": r.path,
                         "gain_db": float(np.round(rng.uniform(*GAIN_DB), 2)), "seed": ph1.utt_seed(r.id)})
        for r in em.itertuples():
            rows.append({"group": "callers", "id": r.id, "voice": r.voice, "cond": cond, "kind": "speech",
                         "cluster": r.id, "ref": r.text, "path": r.path, "category": r.category,
                         "gain_db": float(np.round(rng.uniform(*GAIN_DB), 2)), "seed": ph1.utt_seed(r.id)})
        for r in seg.itertuples():
            rows.append({"group": "segments", "id": f"{r.src_id}~seg", "voice": "human", "cond": cond,
                         "kind": f"segment{r.n_words}", "cluster": r.cluster, "ref": r.ref, "path": r.path,
                         "t0": r.t0, "t1": r.t1, "src_id": r.src_id,
                         "gain_db": float(np.round(rng.uniform(*GAIN_DB), 2)), "seed": ph1.utt_seed(r.src_id)})
        for w in ONE_WORDS:
            for v in tts.VOICES:
                for draw in (0, 1):
                    i = f"{w}|{draw}"
                    rows.append({"group": "oneword", "id": i, "voice": v, "cond": cond, "kind": "oneword",
                                 "cluster": w, "ref": w, "path": "",
                                 "gain_db": float(np.round(rng.uniform(*GAIN_DB), 2)),
                                 "seed": crc(f"{w}|{v}|{draw}|{cond}")})
    d = pd.DataFrame(rows)
    d["key"] = d["group"] + "|" + d["id"] + "|" + d["voice"] + "|" + d["cond"]
    assert d["key"].is_unique
    sp = d["group"] != "nonspeech"
    d.loc[sp, "fold"] = [crc(c) % 2 for c in d.loc[sp, "cluster"]]
    d["fold"] = d["fold"].astype(int)
    return d


def clip_audio(r) -> np.ndarray:
    g = r["group"]
    if g == "nonspeech":
        rec = {k: r[k] for k in ("kind", "duration_s", "level_db", "gain_db", "seed")}
        rec["seed"], rec["cond"], rec["half"] = int(rec["seed"]), r["cond"], int(r["fold"])
        x = nonspeech_audio(rec)
        if r["kind"] == "matched_noise":
            return line(x, r["cond"], int(r["seed"]))
        return line(x, r["cond"], int(r["seed"]))
    if g == "oneword":
        x16 = oneword_audio(r["ref"], r["voice"])
    elif g == "segments":
        prep = channel.prepare(ph1.read16k(r["path"]), seed=int(r["seed"]))
        x16 = prep[int(r["t0"] * SR) : int(r["t1"] * SR)]
    else:
        x16 = ph1.read16k(r["path"])
    return speech_clip(x16, r["cond"], int(r["seed"]), float(r["gain_db"]))


# ----------------------------------------------------------------------------- steps

def features() -> None:
    d = clip_table()
    t0, rows = time.time(), []
    for i, (_, r) in enumerate(d.iterrows()):
        x = clip_audio(r)
        v = guards.webrtc_voiced_s(x)
        rows.append({"key": r["key"], **{f"webrtc{m}_s": v[m] for m in v},
                     "energy_s": guards.energy_voiced_s(x), "audio_s": len(x) / SR,
                     "level_dbov": channel.active_level(x)[0] if channel.power(x) > 0 else -120.0})
        if i % 1000 == 0:
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
            continue
        t0 = time.time()
        arrays = [clip_audio(r) for _, r in part.iterrows()]
        res = guards.scored(model, arrays)
        tmp = shard.with_suffix(".tmp")
        pd.DataFrame([{"key": k, **r} for k, r in zip(part["key"], res)]).to_csv(tmp, index=False)
        tmp.replace(shard)
        print(f"[{model}] {g} {cond}: {len(part)} clips in {time.time() - t0:.0f} s", flush=True)


def randlen(model: str) -> None:
    """The main run's noise-only check (100 clips, street/rain/crowd/home, -36 dB after the phone
    line, seeds 10000+k) with lengths drawn from U(3, 6) s instead of exactly 4.00 s."""
    model = asr.resolve(model)
    shard = SHARDS / SHORT[model] / "randlen.csv"
    if shard.exists():
        return
    shard.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4242)
    groups = ["street", "rain", "crowd", "home"]
    rows, arrays = [], []
    for k in range(100):
        dur = float(np.round(rng.uniform(3.0, 6.0), 2))
        g = groups[k % 4]
        bed = channel.noise_bed(g, int(round(dur * SR)), seed=10_000 + k)
        bed = bed * 10 ** ((-36.0 - band_level(bed)) / 20) + channel._floor_noise(len(bed), k)
        arrays.append(channel.telephone(bed.astype(np.float32)))
        rows.append({"clip": k, "duration_s": dur, "noise_group": g})
    res = guards.scored(model, arrays)
    d = pd.DataFrame(rows)
    d["text"] = [r["text"] for r in res]
    d.to_csv(shard, index=False)


def nsp_check() -> dict:
    """Which token id mlx-whisper reads as no-speech for large-v3-turbo, and its probability."""
    import importlib.metadata as md

    import mlx.core as mx
    from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
    from mlx_whisper.tokenizer import get_tokenizer

    model = asr._whisper_model()
    tok = get_tokenizer(model.is_multilingual, num_languages=model.num_languages, language="en", task="transcribe")
    out = {"mlx_whisper_version": md.version("mlx-whisper"), "n_vocab": int(model.dims.n_vocab),
           "num_languages": int(model.num_languages), "no_speech_id": int(tok.no_speech),
           "no_speech_token": tok.decode_with_timestamps([tok.no_speech]) if hasattr(tok, "decode_with_timestamps") else "",
           "sot_sequence": list(map(int, tok.sot_sequence))}
    # probability of every special token at the start of transcript for 5 s of zeros
    mel = log_mel_spectrogram(np.zeros(5 * SR, np.float32), n_mels=model.dims.n_mels, padding=N_SAMPLES)
    mel = pad_or_trim(mel[:N_FRAMES], N_FRAMES, axis=-2).astype(mx.float16)[None]
    feats = model.encoder(mel)
    toks = mx.array([list(tok.sot_sequence) + [tok.no_timestamps]])
    logits = model.decoder(toks, feats)[0]
    logits = logits[0] if isinstance(logits, tuple) else logits
    sot_idx = list(tok.sot_sequence).index(tok.sot)
    p = mx.softmax(logits[0, sot_idx].astype(mx.float32), axis=-1)
    p = np.array(p)
    names = {i: tok.encoding.decode_single_token_bytes(i).decode(errors="replace") for i in range(tok.eot, len(p))}
    top = sorted(((float(p[i]), names[i]) for i in names), reverse=True)[:6]
    out.update({"p_no_speech_zeros_5s": float(p[tok.no_speech]), "top_special_after_sot": top,
                "token_at_no_speech_id": names.get(int(tok.no_speech), "")})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "nsp_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------------------- analysis

def load_model(model: str, feats: pd.DataFrame) -> pd.DataFrame:
    import jiwer

    parts = [pd.read_csv(p, keep_default_na=False) for p in sorted((SHARDS / SHORT[model]).glob("*__*.csv"))]
    t = pd.concat(parts, ignore_index=True).merge(feats, on="key", how="inner")
    assert len(t) == len(feats), (model, len(t), len(feats))
    for c in ("no_speech_prob", "avg_logprob", "conf_geo", "conf_min", "conf"):
        if c in t:
            t[c] = pd.to_numeric(t[c], errors="coerce")
    if "avg_logprob" in t:
        t["avg_logprob"] = t["avg_logprob"].fillna(0.0)
        t["no_speech_prob"] = t["no_speech_prob"].fillna(0.0)
    t["hyp_norm"] = [asr.normalize(x) for x in t["text"]]
    t["nonempty"] = t["hyp_norm"].str.len() > 0
    fl = t["group"].eq("fl_human") | (t["group"].eq("segments") & t["id"].str.startswith("fl"))
    hyps = [asr.normalize(ph1.fleurs.scoring_ref(h)) if f else h for h, f in zip(t["hyp_norm"], fl)]
    refs = [asr.normalize(x) for x in t["ref"]]
    n_ref, hits, errs = [], [], []
    for r, h in zip(refs, hyps):
        if not r:
            n_ref.append(0); hits.append(0); errs.append(len(h.split())); continue
        if not h:
            n_ref.append(len(r.split())); hits.append(0); errs.append(len(r.split())); continue
        o = jiwer.process_words(r, h)
        n_ref.append(len(r.split())); hits.append(o.hits); errs.append(o.substitutions + o.deletions + o.insertions)
    t["n_ref"], t["hits"], t["errors"] = n_ref, hits, errs
    is_cal = t["group"] == "callers"
    ref_kw = [data.find_keywords(r) if c else [] for r, c in zip(t["ref"], is_cal)]
    hyp_kw = [data.find_keywords(h) for h in t["text"]]
    t["kw_ref"] = [len(k) for k in ref_kw]
    t["kw_hit"] = [len(set(k) & set(h)) for k, h in zip(ref_kw, hyp_kw)]
    t["call_detected"] = [(len(k) > 0) and (len(h) > 0) for k, h in zip(ref_kw, hyp_kw)]
    t["word_ok"] = [(g == "oneword") and (r in h.split()) for g, r, h in zip(t["group"], refs, t["hyp_norm"])]
    t["kw_invented"] = ["; ".join(h) if g == "nonspeech" else "" for h, g in zip(hyp_kw, t["group"])]
    return t


def guard_scores(t: pd.DataFrame, model: str) -> dict[str, np.ndarray]:
    return ph1.guard_scores(t, model)


def grid(s: np.ndarray, rows: np.ndarray, k: int) -> np.ndarray:
    return ph1.thresholds(s[rows], k)


def candidates(t: pd.DataFrame, model: str, rows: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
    """Every guard family and threshold; grids from the quantiles of `rows` only (training fold)."""
    gs = guard_scores(t, model)
    out = [("none", "", np.zeros(len(t), bool))]
    for name, s in gs.items():
        for th in grid(s, rows, 120):
            out.append((name, f"{th:.4g}", s < th))
    model_keys = [k for k in gs if not (k.startswith("webrtc") or k == "energy VAD")]
    for v in ["webrtc mode 3", "energy VAD"]:
        for mk in model_keys:
            for tv in grid(gs[v], rows, 25):
                dv = gs[v] < tv
                for tm in grid(gs[mk], rows, 25):
                    out.append((f"{v} OR {mk}", f"{tv:.4g} | {tm:.4g}", dv | (gs[mk] < tm)))
    if model == asr.MODELS[0]:
        out.append(("whisper default rule", "nsp>0.6 & logprob<-1",
                    guards.whisper_builtin(t["no_speech_prob"].to_numpy(float), t["avg_logprob"].to_numpy(float))))
    return out


class Sel:
    """Matrix form of the selection rule, for fast nested bootstrap."""

    def __init__(self, t: pd.DataFrame, model: str, speech_groups=("ls_human", "fl_human")):
        self.t = t
        self.ns = (t["group"] == "nonspeech").to_numpy()
        self.hum = t["group"].isin(list(speech_groups)).to_numpy()
        self.ne = t["nonempty"].to_numpy()
        self.hits = t["hits"].to_numpy(float)
        self.nref = t["n_ref"].to_numpy(float)
        self.fold = t["fold"].to_numpy()
        self.cands = {}
        self.D = {}
        for f in (0, 1):
            train = (self.fold != f) & (self.ns | self.hum)
            cs = candidates(t, model, train)
            self.cands[f] = cs
            self.D[f] = np.stack([c[2] for c in cs]).astype(np.float32)  # (n_cand, n_rows)
        allr = self.ns | self.hum
        self.cands_full = candidates(t, model, allr)

    def pick_idx(self, f: int, w: np.ndarray) -> np.ndarray:
        """Best candidate index per bootstrap column of w (n_rows x B), choosing on fold != f."""
        train = (self.fold != f).astype(np.float32)
        D = self.D[f]
        wn = w * (self.ns * self.ne * train)[:, None]       # non-speech with text, training fold
        wd = w * (self.ns * train)[:, None]
        ph_after = (wn.sum(0)[None, :] - D @ wn) / np.maximum(wd.sum(0)[None, :], 1e-12)
        wh = w * (self.hum * train * self.hits)[:, None]
        den = (w * (self.hum * train * self.nref)[:, None]).sum(0)[None, :]
        lost = (D @ wh) / np.maximum(den, 1e-12)
        ph_after = np.where(lost <= BUDGET + 1e-12, ph_after, np.inf)
        key = ph_after + 1e-6 * lost  # ties -> fewer words lost
        return np.argmin(key, axis=0)

    def crossfit_drop(self, w: np.ndarray) -> np.ndarray:
        """Cross-fitted drop masks (n_rows x B)."""
        out = np.zeros(w.shape, bool)
        for f in (0, 1):
            idx = self.pick_idx(f, w)
            app = self.fold == f
            out[app] = self.D[f][idx][:, app].T.astype(bool)
        return out


def metrics(t: pd.DataFrame, drop: np.ndarray, w: np.ndarray | None = None) -> dict:
    """Point metrics (w None) or per-replicate arrays (w: n_rows x B, drop: n_rows x B)."""
    if w is None:
        w = np.ones((len(t), 1))
        drop = drop[:, None]
    g = t["group"].to_numpy()
    ne, hits, nref = t["nonempty"].to_numpy(), t["hits"].to_numpy(float), t["n_ref"].to_numpy(float)
    out = {}

    def share(mask_num, mask_den):
        return (w * mask_num).sum(0) / np.maximum((w * mask_den).sum(0), 1e-12)

    ns = (g == "nonspeech")[:, None]
    out["phantom_before"] = share(ns & ne[:, None], ns)
    out["phantom_after"] = share(ns & ne[:, None] & ~drop, ns)
    for name, grp in (("human", ["ls_human", "fl_human"]), ("callers", ["callers"]), ("segments", ["segments"]),
                      ("oneword", ["oneword"])):
        m = np.isin(g, grp)[:, None]
        out[f"words_lost_{name}"] = (w * m * drop * hits[:, None]).sum(0) / np.maximum((w * m * nref[:, None]).sum(0), 1e-12)
        out[f"clips_dropped_{name}"] = share(m & drop, m)
    ok = t["word_ok"].to_numpy()[:, None]
    out["oneword_correct_dropped"] = share(ok & drop, ok)
    cal = (g == "callers")[:, None]
    out["kw_lost_callers"] = (w * cal * drop * t["kw_hit"].to_numpy(float)[:, None]).sum(0) / np.maximum(
        (w * cal * t["kw_ref"].to_numpy(float)[:, None]).sum(0), 1e-12)
    hum = np.isin(g, ["ls_human", "fl_human"])[:, None]
    e = t["errors"].to_numpy(float)[:, None]
    out["wer_human_before"] = (w * hum * e).sum(0) / (w * hum * nref[:, None]).sum(0)
    out["wer_human_after"] = (w * hum * np.where(drop, nref[:, None], e)).sum(0) / (w * hum * nref[:, None]).sum(0)
    return {k: (float(v[0]) if v.shape[0] == 1 else v) for k, v in out.items()}


def boot_weights(t: pd.DataFrame, b: int, seed: int) -> np.ndarray:
    """Cluster bootstrap weights (n_rows x b); non-speech clips, human clusters, caller scripts,
    one-word words resampled independently per group."""
    w = np.zeros((len(t), b), np.float32)
    for k, grp in enumerate([["nonspeech"], ["ls_human", "segments_ls"], ["fl_human", "segments_fl"], ["callers"], ["oneword"]]):
        g = t["group"].where(t["group"] != "segments", np.where(t["id"].str.startswith("fl"), "segments_fl", "segments_ls"))
        idx = np.flatnonzero(g.isin(grp).to_numpy())
        if len(idx) == 0:
            continue
        W = stats.cluster_weights(t["cluster"].to_numpy()[idx], b, seed=seed + k)
        w[idx] = W.T
    return w


def analyze() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    feats = pd.read_csv(SHARDS / "features.csv", keep_default_na=False)
    for c in ("level_db", "gain_db", "duration_s", "t0", "t1"):
        feats[c] = pd.to_numeric(feats[c], errors="coerce")
    feats["short"] = feats["short"].astype(str) == "True"
    n_shards = feats.groupby(["group", "cond"]).ngroups
    present = [m for m in MODELS if len(list((SHARDS / SHORT[m]).glob("*__*.csv"))) == n_shards]
    tabs = {SHORT[m]: load_model(m, feats).reset_index(drop=True) for m in present}
    summary: dict = {"budget_words_lost": BUDGET, "gain_db_range": GAIN_DB, "n_clips": len(feats),
                     "n_by_group": feats.groupby("group").size().to_dict()}

    # ---- phantom rates (non-speech v2) by kind, length and channel
    rr = []
    for name, t in tabs.items():
        ns = t[t["group"] == "nonspeech"]
        sets = [(("ALL", "all", "all"), ns)]
        sets += [((k, "all", "all"), g) for k, g in ns.groupby("kind")]
        sets += [((k, "all", ch), g) for (k, ch), g in ns.groupby(["kind", "cond"])]
        sets += [(("ALL", "short" if s else "long", "all"), g) for s, g in ns.groupby("short")]
        sets += [((k, "short" if s else "long", "all"), g) for (k, s), g in ns.groupby(["kind", "short"])]
        for (kind, length, ch), g in sets:
            k, n = int(g["nonempty"].sum()), len(g)
            lo, hi = stats.clopper_pearson(k, n)
            rr.append({"model": name, "kind": kind, "length": length, "channel": ch, "n": n, "with_text": k,
                       "rate": k / n, "lo": lo, "hi": hi})
        for label, sub in (("excluding bot_echo", ns[ns["kind"] != "bot_echo"]), ("long clips (3-10 s)", ns[~ns["short"]]),
                           ("short clips (0.5-2 s)", ns[ns["short"]])):
            k_, n_ = int(sub["nonempty"].sum()), len(sub)
            lo_, hi_ = stats.clopper_pearson(k_, n_)
            summary.setdefault("phantom_subsets", {}).setdefault(name, {})[label] = {
                "with_text": k_, "n": n_, "rate": k_ / n_, "cp_lo": lo_, "cp_hi": hi_}
        W = stats.cluster_weights(ns["cluster"].to_numpy(), 2000, seed=SEED)
        b = W @ ns["nonempty"].to_numpy(float) / W.sum(1)
        summary.setdefault("phantom_overall", {})[name] = {
            "rate": float(ns["nonempty"].mean()), "lo": float(np.quantile(b, 0.025)), "hi": float(np.quantile(b, 0.975)),
            "with_text": int(ns["nonempty"].sum()), "n": len(ns),
            "invented_keyword_clips": int((ns["kw_invented"] != "").sum())}
    rates = pd.DataFrame(rr)
    rates.to_csv(OUT / "phantom_rates_v2.csv", index=False)
    inv = pd.concat([t.loc[(t["group"] == "nonspeech") & (t["kw_invented"] != ""), ["id", "kind", "cond", "kw_invented", "text"]]
                     .assign(model=n) for n, t in tabs.items()])
    inv.to_csv(OUT / "phantom_emergency_keywords_v2.csv", index=False)

    # ---- level giveaway check: speech vs matched noise, active level and VAD seconds
    t = next(iter(tabs.values()))
    lv = t[t["group"].isin(["ls_human", "fl_human", "nonspeech"])].copy()
    lv["side"] = np.where(lv["group"] == "nonspeech", lv["kind"], "human speech")
    lvt = lv.groupby("side").agg(n=("key", "size"), level_dbov_median=("level_dbov", "median"),
                                 level_q10=("level_dbov", lambda s: s.quantile(.1)), level_q90=("level_dbov", lambda s: s.quantile(.9)),
                                 energy_s_median=("energy_s", "median"), webrtc3_s_median=("webrtc3_s", "median")).reset_index()
    lvt.to_csv(OUT / "level_by_kind_v2.csv", index=False)

    # ---- guards: nested cross-fitted selection, human speech only
    ops, fams = [], []
    resid = []
    short_rows = []
    VARIANTS = {"human full clips": ("ls_human", "fl_human"),
                "human full clips + 1-3 word segments": ("ls_human", "fl_human", "segments")}
    for m, (vname, sgroups) in [(m, v) for m in present for v in VARIANTS.items()]:
        name = SHORT[m]
        t = tabs[name]
        sel = Sel(t, m, sgroups)
        w1 = np.ones((len(t), 1), np.float32)
        drop = sel.crossfit_drop(w1)[:, 0]
        chosen = {f: sel.cands[f][int(sel.pick_idx(f, w1)[0])] for f in (0, 1)}
        mt = metrics(t, drop)
        Wb = boot_weights(t, B, SEED + 11)
        # nested: re-run the cross-fitted selection in each replicate (chunks to bound memory)
        Db = np.concatenate([sel.crossfit_drop(Wb[:, s:s + 100]) for s in range(0, B, 100)], axis=1)
        mb = metrics(t, Db, Wb)
        row = {"model": name, "selection": vname, "fold0_rule": f"{chosen[0][0]} @ {chosen[0][1]}",
               "fold1_rule": f"{chosen[1][0]} @ {chosen[1][1]}", **mt}
        for k, v in mb.items():
            row[f"{k}_lo"], row[f"{k}_hi"] = float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))
        # calls lost, at script level
        c = t[(t["group"] == "callers") & (t["kw_ref"] > 0)]
        lost = c["call_detected"].to_numpy() & drop[c.index]
        scripts_kw = c["id"].nunique()
        row.update({"calls_detected": int(c["call_detected"].sum()), "calls_lost": int(lost.sum()),
                    "keyword_scripts": int(scripts_kw), "scripts_with_a_lost_call": int(c.loc[lost, "id"].nunique()),
                    "script_level_upper_bound": stats.rule_of_three(scripts_kw) if lost.sum() == 0 else np.nan})
        ow = t[(t["group"] == "oneword")]
        okd = ow["word_ok"].to_numpy()
        k_, n_ = int((okd & drop[ow.index]).sum()), int(okd.sum())
        row.update({"oneword_correct_n": n_, "oneword_correct_dropped_n": k_,
                    "oneword_drop_cp_lo": stats.clopper_pearson(k_, n_)[0], "oneword_drop_cp_hi": stats.clopper_pearson(k_, n_)[1]})
        ops.append(row)
        # short-utterance detail per word / per segment length
        for wd, g in ow.groupby("ref"):
            ok = g["word_ok"].to_numpy()
            dd = drop[g.index]
            short_rows.append({"model": name, "selection": vname, "set": "oneword", "item": wd, "n": len(g), "recognized": int(ok.sum()),
                               "recognized_and_dropped": int((ok & dd).sum()), "dropped_any": int(dd.sum()),
                               "energy_s_median": float(g["energy_s"].median())})
        sg = t[t["group"] == "segments"]
        for kind, g in sg.groupby("kind"):
            dd = drop[g.index]
            short_rows.append({"model": name, "selection": vname, "set": "segments", "item": kind, "n": len(g),
                               "recognized": int((g["hits"] > 0).sum()),
                               "recognized_and_dropped": int(((g["hits"] > 0).to_numpy() & dd).sum()),
                               "dropped_any": int(dd.sum()), "energy_s_median": float(g["energy_s"].median()),
                               "words_lost_share": float((g["hits"].to_numpy() * dd).sum() / g["n_ref"].sum())})
        nsx = t[t["group"] == "nonspeech"]
        for (kind, ch), g in nsx.groupby(["kind", "cond"]):
            resid.append({"model": name, "selection": vname, "kind": kind, "channel": ch, "n": len(g),
                          "with_text_before": int(g["nonempty"].sum()),
                          "with_text_after": int((g["nonempty"].to_numpy() & ~drop[g.index]).sum())})
        # each family alone (point estimates, cross-fitted, no nested CI)
        for fam in sorted({c[0] for c in sel.cands[0]} - {"none"}):
            dd = np.zeros(len(t), bool)
            for f in (0, 1):
                cs = [c for c in sel.cands[f] if c[0] in (fam, "none")]
                if fam == "whisper default rule":
                    p = cs[-1]
                else:
                    p = pick_simple(t, cs, t["fold"].to_numpy() != f, sgroups)
                dd[t["fold"].to_numpy() == f] = p[2][t["fold"].to_numpy() == f]
            fams.append({"model": name, "selection": vname, "guard": fam, **metrics(t, dd)})
    ops = pd.DataFrame(ops)
    ops.to_csv(OUT / "recommended_points_v2.csv", index=False)
    pd.DataFrame(fams).to_csv(OUT / "guards_crossfit_v2.csv", index=False)
    pd.DataFrame(resid).to_csv(OUT / "residual_phantoms_by_kind_v2.csv", index=False)
    pd.DataFrame(short_rows).to_csv(OUT / "short_utterances_v2.csv", index=False)
    summary["recommended"] = ops.to_dict("records")

    # ---- noise-only check at random lengths
    rl = []
    for m in MODELS:
        p = SHARDS / SHORT[m] / "randlen.csv"
        if p.exists():
            d = pd.read_csv(p, keep_default_na=False)
            k = int(sum(bool(asr.normalize(x)) for x in d["text"]))
            lo, hi = stats.clopper_pearson(k, len(d))
            whole = d["duration_s"].round(2).mod(1.0).abs() < 1e-9
            rl.append({"model": SHORT[m], "n": len(d), "with_text": k, "rate": k / len(d), "lo": lo, "hi": hi,
                       "n_whole_second": int(whole.sum())})
    pd.DataFrame(rl).to_csv(OUT / "noise_only_random_lengths.csv", index=False)
    summary["noise_only_random_lengths"] = rl
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    figures(rates, ops, tabs)
    pd.set_option("display.width", 250)
    print(json.dumps(summary["phantom_overall"], indent=1))
    print(ops.T.to_string())
    print(pd.DataFrame(fams).round(4).to_string())
    print(pd.DataFrame(short_rows).to_string())
    print(rl)


def pick_simple(t: pd.DataFrame, cs, rows: np.ndarray, speech_groups=("ls_human", "fl_human")):
    """The v2 rule (human speech only) for one family, point estimate."""
    g = t["group"].to_numpy()
    ns = (g == "nonspeech") & rows
    hum = np.isin(g, list(speech_groups)) & rows
    ne, hits, nref = t["nonempty"].to_numpy(), t["hits"].to_numpy(float), t["n_ref"].to_numpy(float)
    best, key = None, None
    for c in cs:
        d = c[2]
        lost = (hits * d)[hum].sum() / nref[hum].sum()
        if lost > BUDGET + 1e-12:
            continue
        k = ((ne & ~d)[ns].mean(), lost)
        if key is None or k < key:
            best, key = c, k
    return best


def figures(rates: pd.DataFrame, ops: pd.DataFrame, tabs: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = rates[(rates["length"] == "all") & (rates["channel"] == "all")]
    kinds = KINDS + ["ALL"]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    cols = {"whisper": "#3b6fb6", "parakeet": "#d08a2e", "wav2vec2": "#6a9e5b"}
    w = 0.27
    for j, name in enumerate([x for x in ["whisper", "parakeet", "wav2vec2"] if x in tabs]):
        q = r[r["model"] == name].set_index("kind").reindex(kinds)
        x = np.arange(len(kinds)) + (j - 1) * w
        ax.bar(x, 100 * q["rate"], w, color=cols[name], label=name)
        ax.errorbar(x, 100 * q["rate"], yerr=[100 * (q["rate"] - q["lo"]), 100 * (q["hi"] - q["rate"])],
                    fmt="none", ecolor="#333", lw=0.8, capsize=2)
    ax.set_xticks(np.arange(len(kinds)))
    ax.set_xticklabels([k.replace("_", " ") for k in kinds], rotation=35, ha="right")
    ax.set_ylabel("clips with any text (%)")
    ax.set_title("v2 non-speech clips: phantom text by kind (phone and phone+loss lines; matched noise: "
                 "phone+noise10 and realistic chains), 95 % CI", fontsize=9.5)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_v2_phantom_rate_by_kind.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    labels = ["phantoms left", "human words dropped", "caller words dropped", "segment words dropped",
              "one-word clips dropped"]
    keys = ["phantom_after", "words_lost_human", "words_lost_callers", "words_lost_segments", "oneword_correct_dropped"]
    for j, name in enumerate([x for x in ["whisper", "parakeet", "wav2vec2"] if x in tabs]):
        o = ops[(ops["model"] == name) & (ops["selection"] == "human full clips + 1-3 word segments")].iloc[0]
        x = np.arange(len(keys)) + (j - 1) * w
        v = np.array([o[k] for k in keys]) * 100
        lo = np.array([o.get(f"{k}_lo", np.nan) for k in keys]) * 100
        hi = np.array([o.get(f"{k}_hi", np.nan) for k in keys]) * 100
        ax.bar(x, v, w, color=cols[name], label=name)
        ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt="none", ecolor="#333", lw=0.8, capsize=2)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("% (symlog)")
    ax.set_title("v2 guard per model, chosen on human clips + short segments (cross-fitted, nested bootstrap CI)",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_v2_guard_tradeoffs.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["segments", "features", "run", "randlen", "nsp", "analyze"])
    ap.add_argument("--model")
    a = ap.parse_args()
    SHARDS.mkdir(parents=True, exist_ok=True)
    {"segments": segments, "features": features, "nsp": nsp_check, "analyze": analyze}.get(
        a.step, lambda: (run if a.step == "run" else randlen)(a.model))()


if __name__ == "__main__":
    main()
