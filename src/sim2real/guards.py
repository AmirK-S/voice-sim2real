"""Phantom speech: non-speech test clips, ASR decoding with confidence scores, and guards.

A "phantom" is any non-empty transcript (after sim2real.asr.normalize) produced on a clip that
contains no speech. A "guard" is a clip-level rule that discards the transcript (treats the clip
as silence). Everything here is offline and per clip: a streaming pipeline would apply the same
signals per segment, which this module does not test.

Contents
- Non-speech clips: ``nonspeech_plan`` (which clips, seeded), ``nonspeech_clip`` (the audio).
  ESC-50 categories rain, wind, engine, keyboard_typing, door_wood_knock and a "crowd events"
  group (laughing, clapping, coughing, footsteps); synthetic line hiss (white), comfort noise
  (pink, low level), digital silence (zeros) and near-silence (the stationary floor only).
  ESC-50 has no TV, music or babble class, so "TV-like" sound is not covered.
- Decoding with scores, same settings as sim2real.asr (so the text matches the main run):
  ``whisper_scored`` (text, no_speech_prob, avg_logprob, compression_ratio),
  ``parakeet_scored`` (text, geometric-mean and minimum token confidence; parakeet-mlx's
  entropy-based confidence, 1 - H/log V per token), ``wav2vec2_scored`` (text, mean max
  posterior over the non-blank frames of the greedy CTC path).
- Voice activity: ``webrtc_voiced_s`` (py-webrtcvad, 30 ms frames, mode 0..3) and
  ``energy_voiced_s`` (300-3400 Hz frame energy above max(floor estimate + 9 dB, -55 dBov)).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import signal

from . import asr, channel
from .audio import resample
from .datasets import esc50_meta, load

SR = 16000

ESC_KINDS: dict[str, list[str]] = {
    "rain": ["rain"],
    "wind": ["wind"],
    "engine": ["engine"],
    "keyboard": ["keyboard_typing"],
    "door_knock": ["door_wood_knock"],
    "crowd_events": ["laughing", "clapping", "coughing", "footsteps"],
}
SYNTH_KINDS = ("line_hiss", "comfort_noise", "digital_silence", "near_silence")
KINDS = tuple(ESC_KINDS) + SYNTH_KINDS
PER_KIND = 30
# Level ranges (dB re full scale, RMS). ESC noise: phone-band power -46 to -26 dB, i.e. what a
# -26 dBov talker would have at 20 to 0 dB SNR. Hiss and comfort noise: typical line and
# comfort-noise-generator levels. Near-silence: the stationary floor of channel.prepare and below.
LEVELS = {"esc": (-46.0, -26.0), "line_hiss": (-55.0, -35.0), "comfort_noise": (-70.0, -50.0),
          "near_silence": (-80.0, -60.0), "digital_silence": (None, None)}


# ----------------------------------------------------------------------------- non-speech clips

def nonspeech_plan(per_kind: int = PER_KIND, seed: int = 0) -> list[dict]:
    """The clip list: kind, duration (3 to 10 s), level and seed, all drawn from `seed`."""
    rng = np.random.default_rng(seed)
    out = []
    for kind in KINDS:
        lo, hi = LEVELS["esc" if kind in ESC_KINDS else kind]
        for k in range(per_kind):
            dur = float(np.round(rng.uniform(3.0, 10.0), 2))
            lvl = float(np.round(rng.uniform(lo, hi), 1)) if lo is not None else float("nan")
            out.append({"clip": f"{kind}-{k:02d}", "kind": kind, "duration_s": dur,
                        "level_db": lvl, "seed": int(rng.integers(0, 2**31 - 1))})
    return out


@lru_cache(maxsize=1)
def _esc_bank() -> dict[str, list[np.ndarray]]:
    """ESC-50 clips per kind at 16 kHz with their digitally silent 50 ms frames removed."""
    meta = esc50_meta()
    bank = {}
    for kind, cats in ESC_KINDS.items():
        clips = []
        for p in sorted(meta[meta["category"].isin(cats)]["path"]):
            a, sr = load(p)
            a = channel._strip_silence(resample(a, sr, SR))
            if len(a) >= SR // 2:
                clips.append(a.astype(np.float64))
        bank[kind] = clips
    return bank


def _chain(clips: list[np.ndarray], n: int, rng: np.random.Generator, xf: int = 800) -> np.ndarray:
    """Different clips chained with 50 ms equal-power crossfades, RMS-matched, random start."""
    order = rng.permutation(len(clips))
    first = clips[order[0]]
    bed = first[int(rng.integers(0, max(1, len(first) - xf))):]
    k = 1
    while len(bed) < n:
        nxt = clips[order[k % len(order)]].copy()
        k += 1
        nxt *= np.sqrt(channel.power(bed[-SR:]) / max(channel.power(nxt), 1e-20))
        t = np.linspace(0, np.pi / 2, xf)
        bed = np.concatenate([bed[:-xf], bed[-xf:] * np.cos(t) + nxt[:xf] * np.sin(t), nxt[xf:]])
    return bed[:n]


def _rms_to(x: np.ndarray, level_db: float) -> np.ndarray:
    return x * 10 ** (level_db / 20) / np.sqrt(np.mean(x ** 2) + 1e-20)


def nonspeech_clip(rec: dict) -> np.ndarray:
    """16 kHz float32 audio of one plan entry (before the channel)."""
    kind, n, rng = rec["kind"], int(round(rec["duration_s"] * SR)), np.random.default_rng(rec["seed"])
    if kind == "digital_silence":
        return np.zeros(n, np.float32)
    if kind == "near_silence":
        return _rms_to(channel._floor_noise(n, rec["seed"]), rec["level_db"]).astype(np.float32)
    if kind == "line_hiss":
        return _rms_to(rng.standard_normal(n), rec["level_db"]).astype(np.float32)
    if kind == "comfort_noise":  # pink-ish, like a comfort noise generator
        y = signal.lfilter([1.0], [1.0, -0.95], rng.standard_normal(n))
        return _rms_to(y, rec["level_db"]).astype(np.float32)
    bed = _chain(_esc_bank()[kind], n, rng)
    # level measured on the phone band, as in scripts/04_run_experiment.noise_only
    lvl = channel.db(channel.power(channel.phone_8k(bed)))
    bed = bed * 10 ** ((rec["level_db"] - lvl) / 20) + channel._floor_noise(n, rec["seed"])
    peak = float(np.max(np.abs(bed)))
    if peak > 0.99:
        bed *= 0.99 / peak
    return bed.astype(np.float32)


# ----------------------------------------------------------------------------- voice activity

def webrtc_voiced_s(x16: np.ndarray, modes=(0, 1, 2, 3)) -> dict[int, float]:
    """Seconds of 30 ms frames that py-webrtcvad marks as speech, per aggressiveness mode."""
    import webrtcvad

    pcm = channel._to_pcm16(np.asarray(x16, np.float64))
    f = 480
    frames = [pcm[i : i + f].tobytes() for i in range(0, len(pcm) - f + 1, f)]
    out = {}
    for m in modes:
        v = webrtcvad.Vad(m)
        out[m] = sum(v.is_speech(fr, SR) for fr in frames) * 0.03
    return out


def energy_voiced_s(x16: np.ndarray, margin_db: float = 9.0, abs_db: float = -55.0) -> float:
    """Seconds of 30 ms frames whose 300-3400 Hz energy exceeds both the clip's floor estimate
    (10th percentile of frame energies) + margin_db and an absolute -55 dB."""
    y = channel._bandlimit_16k(np.asarray(x16, np.float64))
    f = 480
    n = len(y) // f
    if n == 0:
        return 0.0
    e = 10 * np.log10(np.mean(y[: n * f].reshape(n, f) ** 2, axis=1) + 1e-20)
    thr = max(np.percentile(e, 10) + margin_db, abs_db)
    return float(np.sum(e > thr) * 0.03)


# ----------------------------------------------------------------------------- decoding + scores

def whisper_scored(arrays: list[np.ndarray], batch: int = asr.WHISPER_BATCH) -> list[dict]:
    """Same decoding as asr._whisper_batch (clips <= 30 s), keeping the decoder's scores."""
    import mlx.core as mx
    from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
    from mlx_whisper.decoding import DecodingOptions, decode

    model = asr._whisper_model()
    opts = DecodingOptions(language="en", task="transcribe", temperature=0.0, fp16=True)
    out = []
    for s in range(0, len(arrays), batch):
        mels = []
        for a in arrays[s : s + batch]:
            assert len(a) / SR <= asr.WHISPER_MAX_S
            mel = log_mel_spectrogram(asr._as16k(a), n_mels=model.dims.n_mels, padding=N_SAMPLES)
            mels.append(pad_or_trim(mel[:N_FRAMES], N_FRAMES, axis=-2).astype(mx.float16))
        for r in decode(model, mx.stack(mels), opts):
            out.append({"text": r.text.strip(), "no_speech_prob": float(r.no_speech_prob),
                        "avg_logprob": float(r.avg_logprob), "compression_ratio": float(r.compression_ratio)})
    return out


def parakeet_scored(a: np.ndarray) -> dict:
    """Same decoding as asr.parakeet, plus token confidences (1.0 when there is no token)."""
    import mlx.core as mx
    from parakeet_mlx.audio import get_logmel

    audio = asr._as16k(a)
    if len(audio) < 1600:
        return {"text": "", "conf_geo": 1.0, "conf_min": 1.0}
    model = asr._parakeet()
    res = model.generate(get_logmel(mx.array(audio), model.preprocessor_config))[0]
    c = np.array([t.confidence for t in res.tokens], float)
    if len(c) == 0:
        return {"text": res.text.strip(), "conf_geo": 1.0, "conf_min": 1.0}
    return {"text": res.text.strip(), "conf_geo": float(np.exp(np.mean(np.log(c + 1e-10)))),
            "conf_min": float(c.min())}


def wav2vec2_scored(a: np.ndarray) -> dict:
    """Same decoding as asr.wav2vec2, plus the mean max posterior over non-blank frames."""
    import torch

    audio = asr._as16k(a)
    if len(audio) < 400:
        return {"text": "", "conf": 1.0}
    proc, model, dev = asr._w2v()
    inp = proc(audio, sampling_rate=SR, return_tensors="pt").input_values.to(dev)
    with torch.no_grad():
        logits = model(inp).logits[0].float().cpu()
    p = torch.softmax(logits, -1)
    pmax, ids = p.max(-1)
    blank = proc.tokenizer.pad_token_id
    keep = ids != blank
    text = proc.batch_decode(ids[None])[0].strip()
    conf = float(pmax[keep].mean()) if bool(keep.any()) else 1.0
    return {"text": text, "conf": conf}


def scored(model_id: str, arrays: list[np.ndarray]) -> list[dict]:
    mid = asr.resolve(model_id)
    if mid == asr.MODELS[0]:
        return whisper_scored(arrays)
    fn = parakeet_scored if mid == asr.MODELS[2] else wav2vec2_scored
    return [fn(a) for a in arrays]


# ----------------------------------------------------------------------------- guard decisions

def drop_low(score: np.ndarray, thr: float) -> np.ndarray:
    """Discard the transcript when score < thr (every guard is oriented so that low = no speech)."""
    return np.asarray(score, float) < thr


def whisper_builtin(no_speech_prob: np.ndarray, avg_logprob: np.ndarray,
                    nsp_thr: float = 0.6, lp_thr: float = -1.0) -> np.ndarray:
    """The rule of openai-whisper / mlx_whisper.transcribe: skip when no_speech_prob > 0.6 and
    avg_logprob < -1.0 (both defaults)."""
    return (np.asarray(no_speech_prob) > nsp_thr) & (np.asarray(avg_logprob) < lp_thr)
