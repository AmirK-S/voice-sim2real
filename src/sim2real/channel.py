"""Phone-channel degradations for ASR input.

Every public function takes 16 kHz mono float32 audio and returns 16 kHz mono
float32 audio of the same length. Randomness always comes from an explicit
integer seed, so a condition applied to the same utterance with the same seed
gives bit-identical output.

Order used by the composite conditions follows the physical chain of a call:
acoustic scene at the caller (noise, muffling under clothing) -> handset and
telephone network (band-limit, G.711) -> packet loss in transport.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np
from scipy import signal

from .audio import resample
from .datasets import esc50_meta, load

SR = 16000
PHONE_SR = 8000

# ESC-50 has no "street" or "TV/crowd" class, so we use the closest ones.
NOISE_GROUPS: dict[str, list[str]] = {
    "street": ["engine", "car_horn", "siren", "train"],
    "rain": ["rain"],
    "wind": ["wind"],
    "crowd": ["laughing", "clapping", "crying_baby"],  # TV / people-in-the-room proxy
    "home": ["vacuum_cleaner", "washing_machine"],
}


# --------------------------------------------------------------------------
# G.711 mu-law (segmented 8-bit, decision levels of the CCITT/Sun reference
# g711.c). The decoder is bit-identical to ffmpeg. The encoder differs from
# ffmpeg by one adjacent code on about 1.3 % of int16 inputs near interval
# edges, because ffmpeg picks the nearest output level instead.
# --------------------------------------------------------------------------
_BIAS = 0x84
_CLIP = 32635


def mulaw_encode(pcm16: np.ndarray) -> np.ndarray:
    """int16 PCM -> uint8 G.711 mu-law codes."""
    x = pcm16.astype(np.int32)
    sign = (x < 0)
    mag = np.minimum(np.abs(x), _CLIP) + _BIAS
    exponent = np.floor(np.log2(mag)).astype(np.int32) - 7  # segment 0..7
    exponent = np.clip(exponent, 0, 7)
    mantissa = (mag >> (exponent + 3)) & 0x0F
    code = ~((sign.astype(np.int32) << 7) | (exponent << 4) | mantissa)
    return (code & 0xFF).astype(np.uint8)


def mulaw_decode(codes: np.ndarray) -> np.ndarray:
    """uint8 G.711 mu-law codes -> int16 PCM."""
    u = (~codes.astype(np.int32)) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    mag = (((mantissa << 3) + _BIAS) << exponent) - _BIAS
    return np.where(sign, -mag, mag).astype(np.int16)


def _to_pcm16(x: np.ndarray) -> np.ndarray:
    return np.round(np.clip(x, -1.0, 1.0 - 1 / 32768) * 32768).astype(np.int16)


def _fit(y: np.ndarray, n: int) -> np.ndarray:
    """Pad or trim to exactly n samples (resampling can be off by one)."""
    if len(y) >= n:
        return y[:n].astype(np.float32)
    return np.pad(y, (0, n - len(y))).astype(np.float32)


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def telephone(x: np.ndarray, band: tuple[float, float] = (300.0, 3400.0)) -> np.ndarray:
    """Narrowband phone line: 16k -> 8k, 300-3400 Hz band-pass, G.711 mu-law, -> 16k."""
    n = len(x)
    y = resample(x, SR, PHONE_SR)
    sos = signal.butter(4, band, btype="bandpass", fs=PHONE_SR, output="sos")
    y = signal.sosfiltfilt(sos, y).astype(np.float32)
    y = mulaw_decode(mulaw_encode(_to_pcm16(y))).astype(np.float32) / 32768.0
    return _fit(resample(y, PHONE_SR, SR), n)


def loss_mask(n_frames: int, rate: float, seed: int, burst: float | None = None) -> np.ndarray:
    """Boolean mask, True = frame lost.

    burst=None: independent (Bernoulli) losses at `rate`.
    burst=L: Gilbert-Elliott two-state chain (good state never loses, bad
    state always loses) with mean burst length L frames and stationary loss
    probability `rate`: r = 1/L, p = rate * r / (1 - rate).
    """
    rng = np.random.default_rng(seed)
    if rate <= 0:
        return np.zeros(n_frames, dtype=bool)
    if burst is None:
        return rng.random(n_frames) < rate
    r = 1.0 / max(burst, 1.0)
    p = min(rate * r / (1.0 - rate), 1.0)
    u = rng.random(n_frames)
    lost = np.zeros(n_frames, dtype=bool)
    bad = rng.random() < rate  # start from the stationary distribution
    for i in range(n_frames):
        lost[i] = bad
        bad = (u[i] >= r) if bad else (u[i] < p)
    return lost


def packet_loss(x: np.ndarray, rate: float, frame_ms: float = 20, seed: int = 0,
                burst: float | None = None, max_repeat: int = 2,
                atten: float = 0.5) -> np.ndarray:
    """Drop frames and conceal them like a basic jitter buffer.

    A lost frame is replaced by the last good frame scaled by atten**k (k-th
    consecutive loss); after `max_repeat` repeats the gap becomes silence.
    Short 2 ms ramps at concealment boundaries avoid clicks.
    """
    y = x.astype(np.float32).copy()
    flen = int(round(SR * frame_ms / 1000))
    n_frames = int(np.ceil(len(x) / flen))
    lost = loss_mask(n_frames, rate, seed, burst)
    ramp = np.linspace(0.0, 1.0, int(0.002 * SR), dtype=np.float32)
    last_good = np.zeros(flen, dtype=np.float32)
    k = 0
    for i in range(n_frames):
        s, e = i * flen, min((i + 1) * flen, len(x))
        if not lost[i]:
            seg = x[s:e]
            last_good = np.pad(seg, (0, flen - len(seg))).astype(np.float32)
            k = 0
            continue
        k += 1
        fill = last_good * (atten ** k) if k <= max_repeat else np.zeros(flen, np.float32)
        fill = fill[: e - s].copy()
        m = min(len(ramp), len(fill) // 2)
        if m:
            fill[:m] *= ramp[:m]
            fill[-m:] *= ramp[:m][::-1]
        y[s:e] = fill
    return y


def power(x: np.ndarray) -> float:
    """Mean power of a signal (whole-signal definition used for SNR)."""
    return float(np.mean(np.asarray(x, dtype=np.float64) ** 2))


def measured_snr(speech: np.ndarray, noise: np.ndarray) -> float:
    """SNR in dB between a speech signal and the noise actually added to it."""
    return 10.0 * np.log10(power(speech) / max(power(noise), 1e-20))


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float, seed: int = 0,
              return_noise: bool = False):
    """Mix `noise` (16 kHz) into `x` at `snr_db`.

    SNR is defined on mean power over the whole utterance (speech including
    its pauses), which is the convention most ASR robustness papers use. The
    noise is looped and taken from a seeded random offset. If the mix would
    clip, speech and noise are scaled together, which keeps the SNR exact.
    """
    rng = np.random.default_rng(seed)
    noise = noise.astype(np.float32)
    reps = int(np.ceil((len(x) + len(noise)) / len(noise)))
    long = np.tile(noise, reps)
    off = int(rng.integers(0, len(noise)))
    seg = long[off:off + len(x)].astype(np.float64)
    gain = np.sqrt(power(x) / (max(power(seg), 1e-20) * 10 ** (snr_db / 10)))
    seg = seg * gain
    mix = x.astype(np.float64) + seg
    peak = np.max(np.abs(mix)) if len(mix) else 0.0
    scale = 0.99 / peak if peak > 0.99 else 1.0
    mix, seg = mix * scale, seg * scale
    if return_noise:
        return mix.astype(np.float32), seg.astype(np.float32)
    return mix.astype(np.float32)


def muffle(x: np.ndarray, cutoff: float = 1800.0, drop_db: float = -8.0,
           rt60: float = 0.25, wet: float = 0.15, seed: int = 0) -> np.ndarray:
    """Pendant under clothing: low-pass near 1.8 kHz, level drop, light room reverb.

    The reverb is a synthetic exponentially decaying noise impulse response
    (seeded), mixed at `wet` relative to the dry signal.
    """
    sos = signal.butter(4, cutoff, btype="lowpass", fs=SR, output="sos")
    y = signal.sosfilt(sos, x).astype(np.float64)
    rng = np.random.default_rng(seed)
    t = np.arange(int(rt60 * SR)) / SR
    ir = rng.standard_normal(len(t)) * np.exp(-6.9078 * t / rt60)  # -60 dB at rt60
    ir /= np.sqrt(np.sum(ir ** 2))
    rev = signal.fftconvolve(y, ir)[: len(y)]
    y = (1 - wet) * y + wet * rev
    y *= 10 ** (drop_db / 20)
    return np.clip(y, -1, 1).astype(np.float32)


# --------------------------------------------------------------------------
# ESC-50 noise bank
# --------------------------------------------------------------------------
def _activity(clip: np.ndarray) -> float:
    """Fraction of 50 ms frames within 30 dB of the loudest frame."""
    f = 800
    n = len(clip) // f
    if n == 0:
        return 0.0
    e = np.mean(clip[: n * f].reshape(n, f) ** 2, axis=1) + 1e-12
    return float(np.mean(10 * np.log10(e / e.max()) > -30))


@lru_cache(maxsize=1)
def _noise_bank() -> dict[str, list[np.ndarray]]:
    """Load the ESC-50 clips of every group, resampled to 16 kHz.

    Clips that are mostly silence (ESC-50 pads short events with zeros) are
    dropped so the requested SNR is not driven by a single burst.
    """
    meta = esc50_meta()
    bank: dict[str, list[np.ndarray]] = {}
    for group, cats in NOISE_GROUPS.items():
        clips = []
        for path in sorted(meta[meta["category"].isin(cats)]["path"]):
            a, sr = load(path)
            a = resample(a, sr, SR)
            if _activity(a) >= 0.8:
                clips.append(a)
        bank[group] = clips
    return bank


def noise_clip(group: str, seed: int) -> np.ndarray:
    """Deterministically pick one 16 kHz ESC-50 clip of `group` for this seed."""
    clips = _noise_bank()[group]
    return clips[int(np.random.default_rng(seed).integers(0, len(clips)))]


def noise_any(seed: int, groups: tuple[str, ...] = ("street", "rain", "wind", "crowd")) -> np.ndarray:
    """Pick a group then a clip, both from the seed."""
    g = groups[int(np.random.default_rng(seed + 7919).integers(0, len(groups)))]
    return noise_clip(g, seed)


# --------------------------------------------------------------------------
# Conditions registry
# --------------------------------------------------------------------------
Condition = Callable[[np.ndarray, int], np.ndarray]


def _noisy_phone(snr: float) -> Condition:
    return lambda x, seed: telephone(add_noise(x, noise_any(seed), snr, seed=seed))


CONDITIONS: dict[str, Condition] = {
    "clean": lambda x, seed: x.astype(np.float32),
    "phone": lambda x, seed: telephone(x),
    "phone+loss5": lambda x, seed: packet_loss(telephone(x), 0.05, seed=seed),
    "phone+loss10": lambda x, seed: packet_loss(telephone(x), 0.10, seed=seed),
    "phone+noise20": _noisy_phone(20),
    "phone+noise10": _noisy_phone(10),
    "phone+noise5": _noisy_phone(5),
    "phone+muffle": lambda x, seed: telephone(muffle(x, seed=seed)),
    # Plausible bad-but-common call: street noise at 10 dB, phone line,
    # 3 % bursty loss (mean burst 3 frames = 60 ms).
    "realistic": lambda x, seed: packet_loss(
        telephone(add_noise(x, noise_clip("street", seed), 10, seed=seed)),
        0.03, seed=seed, burst=3),
}


def apply(name: str, x: np.ndarray, seed: int = 0) -> np.ndarray:
    """Apply condition `name` to 16 kHz audio with a deterministic seed."""
    return CONDITIONS[name](np.asarray(x, dtype=np.float32), seed)
