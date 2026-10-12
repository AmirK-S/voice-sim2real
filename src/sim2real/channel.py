"""Phone-channel degradations for ASR input (channel v2).

Pipeline for every clip, every condition (clean included):

    prepare()  trim edge silence, 250 ms lead and trail, normalize the active speech
               level to -26 dBov (P.56-style, like a handset AGC), add a -65 dBFS
               stationary noise floor so no voice has digital silence
    condition  acoustic scene (noise, muffling) -> handset send filter and codec
               -> packet loss with concealment

Every condition function takes 16 kHz mono float32 audio (already prepared) and
returns 16 kHz mono float32 audio of the same length. Randomness always comes
from an explicit integer seed, so a condition applied to the same utterance with
the same seed gives bit-identical output. ``apply(..., meta=dict)`` also reports
per-clip measurements (in-band SNR before and after the channel, noise group,
actual loss rate).

Changes from v1 (see results/FINDINGS.md, "Fixes"): SNR is now the in-band
(300-3400 Hz) active-speech SNR instead of whole-utterance full-band power; the
phone filter is a causal, IRS-like send response instead of a zero-phase
Butterworth; packet loss runs on the 8 kHz stream with pitch-synchronous
concealment modelled on G.711 Appendix I instead of attenuated frame repeats;
the noise bed chains several clips with crossfades instead of tiling one clip;
wind is no longer in the noise mix; Opus (libopus, real decoder concealment) and
G.722 wideband conditions are added.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import subprocess
from functools import lru_cache
from typing import Callable

import numpy as np
from scipy import signal

from .audio import resample
from .datasets import esc50_meta, load

SR = 16000
PHONE_SR = 8000
TARGET_DBOV = -26.0     # ITU-T P.56 active level commonly used for speech test material
FLOOR_DBFS = -65.0      # stationary noise floor (mic + preamp), RMS
PAD_S = 0.25

# ESC-50 has no "street" or "TV/crowd" class, so we use the closest ones.
# "crowd" and the horn in "street" are transient events rather than steady background.
NOISE_GROUPS: dict[str, list[str]] = {
    "street": ["engine", "car_horn", "siren", "train"],
    "rain": ["rain"],
    "wind": ["wind"],
    "crowd": ["laughing", "clapping", "crying_baby"],  # people-in-the-room proxy (events)
    "home": ["vacuum_cleaner", "washing_machine"],
}
# Wind is excluded from the mix: 82 % of its energy is below 300 Hz, so an
# in-band SNR target would boost it into an unrealistic rumble.
NOISE_MIX = ("street", "rain", "crowd")


# --------------------------------------------------------------------------
# Levels
# --------------------------------------------------------------------------
def power(x: np.ndarray) -> float:
    """Mean power of a signal."""
    return float(np.mean(np.asarray(x, dtype=np.float64) ** 2))


def db(p: float) -> float:
    return 10.0 * np.log10(max(p, 1e-20))


def active_level(x: np.ndarray, sr: int = SR, margin_db: float = 15.9) -> tuple[float, float]:
    """Active speech level in dB (0 dB = full-scale square wave) and activity factor.

    Simplified ITU-T P.56 method B: envelope = one-pole smoothed |x| (30 ms) with
    200 ms hangover; for a grid of thresholds c, the active level A(c) is the
    energy over samples whose envelope is >= c; the level is A where A(c) - c
    equals the 15.9 dB margin (linear interpolation on the grid).
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0 or not np.any(x):
        return -200.0, 0.0
    g = np.exp(-1.0 / (0.03 * sr))
    env = signal.lfilter([1 - g], [1, -g], np.abs(x))
    env = signal.lfilter([1 - g], [1, -g], env)
    # hangover: max filter over 200 ms trailing window
    h = int(0.2 * sr)
    from scipy.ndimage import maximum_filter1d
    env = maximum_filter1d(env, size=h, origin=(h - 1) // 2)  # hold for 200 ms after activity
    e = x ** 2
    thr_db = np.arange(-100.0, 0.5, 0.5)
    env_db = 20 * np.log10(np.maximum(env, 1e-10))
    srt = np.sort(env_db)
    csum = np.cumsum(e[np.argsort(env_db)][::-1])[::-1]  # energy of samples with env >= value
    pos = np.searchsorted(srt, thr_db, side="left")
    cnt = len(x) - pos
    ok = cnt > 0
    en = np.where(ok, csum[np.minimum(pos, len(x) - 1)], 0.0)
    a_db = np.where(ok, 10 * np.log10(np.maximum(en / np.maximum(cnt, 1), 1e-20)), -200.0)
    diff = a_db - thr_db
    k = np.flatnonzero(ok & (diff <= margin_db))
    if len(k) == 0:
        return float(db(power(x))), 1.0
    j = k[0]
    if j == 0:
        lvl, act = a_db[0], cnt[0] / len(x)
    else:
        d0, d1 = diff[j - 1], diff[j]
        w = (d0 - margin_db) / (d0 - d1) if d0 != d1 else 0.0
        lvl = a_db[j - 1] + w * (a_db[j] - a_db[j - 1])
        act = (cnt[j - 1] + w * (cnt[j] - cnt[j - 1])) / len(x)
    return float(lvl), float(act)


def _bandlimit_16k(x: np.ndarray, band=(300.0, 3400.0)) -> np.ndarray:
    sos = signal.butter(6, band, btype="bandpass", fs=SR, output="sos")
    return signal.sosfilt(sos, np.asarray(x, dtype=np.float64))


def inband_active_level(x: np.ndarray) -> float:
    return active_level(_bandlimit_16k(x))[0]


# --------------------------------------------------------------------------
# Preparation (every clip, every voice, every condition)
# --------------------------------------------------------------------------
def trim_bounds(x: np.ndarray, rel_db: float = -30.0, keep_s: float = 0.05) -> tuple[int, int]:
    """First and last sample of speech: 10 ms frames within `rel_db` of the active level."""
    lvl, _ = active_level(x)
    f = SR // 100
    n = len(x) // f
    if n == 0:
        return 0, len(x)
    e = np.mean(np.asarray(x[: n * f], dtype=np.float64).reshape(n, f) ** 2, axis=1)
    on = np.flatnonzero(10 * np.log10(e + 1e-20) > lvl + rel_db)
    if len(on) == 0:
        return 0, len(x)
    k = int(keep_s * SR)
    return max(0, on[0] * f - k), min(len(x), (on[-1] + 1) * f + k)


def speech_duration(x: np.ndarray) -> float:
    s, e = trim_bounds(x)
    return (e - s) / SR


def _floor_noise(n: int, seed: int) -> np.ndarray:
    """Pink-ish stationary noise (white through a one-pole low-pass), RMS FLOOR_DBFS."""
    rng = np.random.default_rng(seed + 104729)
    w = rng.standard_normal(n)
    y = signal.lfilter([1.0], [1.0, -0.9], w)
    y *= 10 ** (FLOOR_DBFS / 20) / np.sqrt(np.mean(y ** 2) + 1e-20)
    return y


def prepare(x: np.ndarray, seed: int = 0) -> np.ndarray:
    """Trim edge silence, pad 250 ms each side, AGC to -26 dBov active level, add noise floor."""
    x = np.asarray(x, dtype=np.float64)
    s, e = trim_bounds(x)
    x = x[s:e]
    lvl, _ = active_level(x)
    x = x * 10 ** ((TARGET_DBOV - lvl) / 20)
    pad = int(PAD_S * SR)
    y = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    y = y + _floor_noise(len(y), seed)
    peak = np.max(np.abs(y))
    if peak > 0.99:  # soft limiter is out of scope: scale down (rare at -26 dBov)
        y *= 0.99 / peak
    return y.astype(np.float32)


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
# Handset send filters (linear-phase FIR, causal, delay compensated)
# --------------------------------------------------------------------------
# IRS-like narrowband send response (dB re 1 kHz). An approximation of the
# modified IRS send characteristic (ITU-T P.48 / P.830): steep below 300 Hz,
# slight rise towards 3 kHz, steep above 3.4 kHz. Not the exact P.48 mask.
_IRS_NB = [(0, -60), (100, -40), (200, -20), (300, -6), (400, -3), (500, -1.5), (700, -0.5),
           (1000, 0), (1500, 1), (2000, 2), (2500, 3), (3000, 3.5), (3300, 2), (3400, 0),
           (3600, -15), (3800, -35), (4000, -60)]
_WB = [(0, -60), (50, -30), (100, -6), (150, -1), (300, 0), (1000, 0), (3000, 1.5), (5000, 2),
       (6500, 0), (7000, -6), (7500, -30), (8000, -60)]


@lru_cache(maxsize=4)
def _fir(table: tuple, fs: int, taps: int = 129) -> np.ndarray:
    f = np.array([p[0] for p in table], float) / (fs / 2)
    g = 10 ** (np.array([p[1] for p in table], float) / 20)
    return signal.firwin2(taps, f, g)


def _send_filter(x: np.ndarray, table, fs: int) -> np.ndarray:
    h = _fir(tuple(table), fs)
    d = (len(h) - 1) // 2
    y = signal.lfilter(h, [1.0], np.concatenate([x, np.zeros(d)]))
    return y[d:]


def phone_8k(x16: np.ndarray) -> np.ndarray:
    """16 kHz -> 8 kHz, IRS-like send filter, G.711 mu-law round trip. Returns 8 kHz float."""
    y = resample(np.asarray(x16, np.float32), SR, PHONE_SR).astype(np.float64)
    y = _send_filter(y, _IRS_NB, PHONE_SR)
    return mulaw_decode(mulaw_encode(_to_pcm16(y))).astype(np.float64) / 32768.0


def telephone(x: np.ndarray) -> np.ndarray:
    """Narrowband phone line at 16 kHz in and out."""
    return _fit(resample(phone_8k(x).astype(np.float32), PHONE_SR, SR), len(x))


# --------------------------------------------------------------------------
# Packet loss
# --------------------------------------------------------------------------
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


def _pitch_period(hist: np.ndarray, sr: int) -> int:
    """Pitch period (samples) of the end of `hist` by normalized cross-correlation, 5-15 ms."""
    lo, hi = int(0.005 * sr), int(0.015 * sr)
    w = int(0.02 * sr)
    seg = hist[-w:]
    best, best_c = lo, -np.inf
    for p in range(lo, hi + 1):
        ref = hist[-w - p:-p]
        den = np.sqrt(np.dot(ref, ref) * np.dot(seg, seg)) + 1e-12
        c = np.dot(ref, seg) / den
        if c > best_c:
            best, best_c = p, c
    return best


def plc_g711_appendix1(y: np.ndarray, lost: np.ndarray, frame: int, sr: int = PHONE_SR) -> np.ndarray:
    """Conceal lost frames in the spirit of ITU-T G.711 Appendix I.

    - Pitch period P from the last 20 ms of good signal (5-15 ms search).
    - The erasure is filled by cycling the last P samples of history; the tail
      of the pitch buffer is overlap-added (quarter period) with the samples
      before it so the cycle has no step, and the last quarter period of the
      already-output signal is replaced by the same OLA (Appendix I delays its
      output by 3.75 ms to allow this; we work offline).
    - After 10 ms the buffer grows to 2 periods, after 20 ms to 3 periods
      (crossfaded switch); no attenuation for the first 10 ms, then a linear
      decay of 20 % per 10 ms, silence from 60 ms on.
    - On the first good frame the synthetic continuation is overlap-added with
      the real signal over a quarter period plus 4 ms per extra 10 ms of erasure
      (capped at 10 ms).
    """
    x = np.asarray(y, dtype=np.float64)
    out = x.copy()
    n = len(x)
    ms10 = int(0.01 * sr)
    i = 0
    n_frames = len(lost)
    hist_need = int(0.02 * sr) + int(0.015 * sr) * 3 + 8
    while i < n_frames:
        if not lost[i]:
            i += 1
            continue
        j = i
        while j < n_frames and lost[j]:
            j += 1
        s, e = i * frame, min(j * frame, n)
        if s < hist_need:  # not enough history at the very start: silence
            out[s:e] = 0.0
            i = j
            continue
        hist = out[:s]
        P = _pitch_period(hist, sr)
        q = max(P // 4, 1)
        L = e - s
        tail = int(min(0.01 * sr, q + 0.004 * sr * max(0, int(np.ceil(L / ms10)) - 1)))
        total = L + (tail if e < n else 0)
        synth = np.zeros(total)
        pos = 0
        cur_k = 0
        buf = None
        ramp_q = np.linspace(0.0, 1.0, q + 2)[1:-1]
        for t in range(total):
            k = 1 if t < ms10 else (2 if t < 2 * ms10 else 3)
            if k != cur_k:
                nb = hist[-k * P:].copy()
                nb[-q:] = nb[-q:] * (1 - ramp_q) + hist[-k * P - q:-k * P] * ramp_q
                if buf is not None:  # crossfade old and new buffers over q samples
                    newpos = pos % len(nb)
                    fade = [(buf[(pos + u) % len(buf)], nb[(newpos + u) % len(nb)]) for u in range(q)]
                    for u, (a, b) in enumerate(fade):
                        if t + u < total:
                            w = (u + 1) / (q + 1)
                            synth[t + u] = (1 - w) * a + w * b
                    buf, pos, cur_k = nb, newpos, k
                    skip_until = t + q
                else:
                    buf, pos, cur_k, skip_until = nb, 0, k, t
                    # replace the last q output samples by the OLA tail (offline 3.75 ms delay)
                    out[s - q:s] = buf[-q:] if len(buf) >= q else out[s - q:s]
            if t >= skip_until:
                synth[t] = buf[pos % len(buf)]
            pos += 1
        # attenuation: flat for 10 ms, then -20 % per 10 ms, silent from 60 ms
        tt = np.arange(total) / sr
        gain = np.clip(1.0 - np.maximum(tt - 0.01, 0.0) * 20.0, 0.0, 1.0)
        gain[tt >= 0.06] = 0.0
        synth *= gain
        out[s:e] = synth[:L]
        if e < n and tail > 0:
            m = min(tail, n - e)
            w = np.linspace(0.0, 1.0, m + 2)[1:-1]
            out[e:e + m] = synth[L:L + m] * (1 - w) + x[e:e + m] * w
        i = j
    return out


def phone_with_loss(x16: np.ndarray, rate: float, seed: int, burst: float | None = None,
                    meta: dict | None = None) -> np.ndarray:
    """G.711 phone line with 20 ms packets lost at `rate` and Appendix I-style concealment."""
    y8 = phone_8k(x16)
    frame = int(0.02 * PHONE_SR)
    lost = loss_mask(int(np.ceil(len(y8) / frame)), rate, seed, burst)
    y8 = plc_g711_appendix1(y8, lost, frame)
    if meta is not None:
        meta["loss_rate"] = float(lost.mean())
    return _fit(resample(y8.astype(np.float32), PHONE_SR, SR), len(x16))


# --------------------------------------------------------------------------
# Opus (libopus through ctypes: SILK narrowband, real decoder concealment)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _libopus():
    path = ctypes.util.find_library("opus") or "/opt/homebrew/lib/libopus.dylib"
    lib = ctypes.CDLL(path)
    lib.opus_encoder_create.restype = ctypes.c_void_p
    lib.opus_encoder_create.argtypes = [ctypes.c_int32, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.opus_decoder_create.restype = ctypes.c_void_p
    lib.opus_decoder_create.argtypes = [ctypes.c_int32, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.opus_encode.restype = ctypes.c_int32
    lib.opus_encode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16), ctypes.c_int,
                                ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32]
    lib.opus_decode.restype = ctypes.c_int
    lib.opus_decode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32,
                                ctypes.POINTER(ctypes.c_int16), ctypes.c_int, ctypes.c_int]
    lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]
    lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
    return lib


OPUS_APPLICATION_VOIP = 2048


def opus_nb(x16: np.ndarray, rate: float = 0.0, seed: int = 0, meta: dict | None = None) -> np.ndarray:
    """IRS-like send filter at 8 kHz, Opus VOIP mode at 8 kHz input (libopus default bitrate for
    8 kHz mono 20 ms frames: 60*8000/160 + 8000 = 11 kb/s, SILK narrowband), optional
    Bernoulli packet loss concealed by the libopus decoder's own PLC."""
    lib = _libopus()
    y = resample(np.asarray(x16, np.float32), SR, PHONE_SR).astype(np.float64)
    pcm = _to_pcm16(_send_filter(y, _IRS_NB, PHONE_SR))
    frame = 160
    nfr = int(np.ceil(len(pcm) / frame))
    pcm = np.pad(pcm, (0, nfr * frame - len(pcm)))
    err = ctypes.c_int(0)
    enc = lib.opus_encoder_create(PHONE_SR, 1, OPUS_APPLICATION_VOIP, ctypes.byref(err))
    dec = lib.opus_decoder_create(PHONE_SR, 1, ctypes.byref(err))
    lost = loss_mask(nfr, rate, seed)
    out = np.zeros(nfr * frame, np.int16)
    pkt = (ctypes.c_ubyte * 4000)()
    obuf = (ctypes.c_int16 * frame)()
    nbytes = 0
    try:
        for i in range(nfr):
            fr = np.ascontiguousarray(pcm[i * frame:(i + 1) * frame])
            nb = lib.opus_encode(enc, fr.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)), frame, pkt, 4000)
            if nb < 0:
                raise RuntimeError(f"opus_encode error {nb}")
            nbytes += nb
            if lost[i]:
                got = lib.opus_decode(dec, None, 0, obuf, frame, 0)
            else:
                got = lib.opus_decode(dec, pkt, nb, obuf, frame, 0)
            if got < 0:
                raise RuntimeError(f"opus_decode error {got}")
            out[i * frame:(i + 1) * frame] = np.frombuffer(obuf, np.int16)[:frame]
    finally:
        lib.opus_encoder_destroy(enc)
        lib.opus_decoder_destroy(dec)
    if meta is not None:
        meta["opus_kbps"] = nbytes * 8 / (nfr * 0.02) / 1000
        meta["loss_rate"] = float(lost.mean())
    return _fit(resample(out.astype(np.float32) / 32768.0, PHONE_SR, SR), len(x16))


# --------------------------------------------------------------------------
# G.722 wideband (ffmpeg, 64 kb/s)
# --------------------------------------------------------------------------
def g722_wb(x16: np.ndarray) -> np.ndarray:
    """Wideband handset (100-7000 Hz send filter), G.722 64 kb/s round trip through ffmpeg."""
    y = _send_filter(np.asarray(x16, np.float64), _WB, SR)
    pcm = _to_pcm16(y).astype("<i2").tobytes()
    enc = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", "16000", "-ac", "1",
                          "-i", "pipe:0", "-c:a", "g722", "-f", "g722", "pipe:1"],
                         input=pcm, capture_output=True, check=True).stdout
    dec = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "g722",
                          "-i", "pipe:0", "-f", "s16le", "-ar", "16000", "pipe:1"],
                         input=enc, capture_output=True, check=True).stdout
    out = np.frombuffer(dec, "<i2").astype(np.float32) / 32768.0
    return _fit(out, len(x16))


# --------------------------------------------------------------------------
# Noise
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


def noise_group(seed: int, groups: tuple[str, ...] = NOISE_MIX) -> str:
    return groups[int(np.random.default_rng(seed + 7919).integers(0, len(groups)))]


def _silent_share(x: np.ndarray) -> float:
    f = len(x) // 800
    if f == 0:
        return 1.0
    e = np.mean(np.asarray(x[: f * 800], np.float64).reshape(f, 800) ** 2, axis=1)
    return float(np.mean(e < 1e-9))


def _strip_silence(clip: np.ndarray) -> np.ndarray:
    """Drop the 50 ms frames of digital silence (ESC-50 pads short events with zeros)."""
    f = len(clip) // 800
    fr = clip[: f * 800].reshape(f, 800)
    keep = np.mean(fr.astype(np.float64) ** 2, axis=1) >= 1e-9
    return fr[keep].ravel()


def noise_bed(group: str, n: int, seed: int, xfade_s: float = 0.05) -> np.ndarray:
    """n samples of `group` noise: different clips of the group chained with 50 ms
    equal-power crossfades (no loop seam), random start offset, all from the seed.

    If the bed contains more than 2 % of digitally silent 50 ms frames (ESC-50 zero
    padding), it is rebuilt from the same clips with their silent frames removed, so
    the noise is present over the whole utterance. Beds without silence are unchanged."""
    bed = _noise_bed(group, n, seed, xfade_s, strip=False)
    if _silent_share(bed) > 0.02:
        bed = _noise_bed(group, n, seed, xfade_s, strip=True)
    return bed


def _noise_bed(group: str, n: int, seed: int, xfade_s: float, strip: bool) -> np.ndarray:
    clips = _noise_bank()[group]
    if strip:
        clips = [_strip_silence(c) for c in clips]
    rng = np.random.default_rng(seed)
    xf = int(xfade_s * SR)
    order = rng.permutation(len(clips))
    first = clips[order[0]]
    off = int(rng.integers(0, max(1, len(first) - xf)))
    bed = first[off:].astype(np.float64)
    k = 1
    while len(bed) < n:
        nxt = clips[order[k % len(order)]].astype(np.float64)
        k += 1
        # match RMS of the incoming clip to the bed so the level does not jump
        nxt *= np.sqrt(power(bed[-SR:]) / max(power(nxt), 1e-20))
        t = np.linspace(0, np.pi / 2, xf)
        bed = np.concatenate([bed[:-xf], bed[-xf:] * np.cos(t) + nxt[:xf] * np.sin(t), nxt[xf:]])
    return bed[:n]


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float, meta: dict | None = None,
              return_noise: bool = False):
    """Mix `noise` into `x` so the in-band (300-3400 Hz) SNR is `snr_db`.

    Speech level = P.56-style active level of the band-limited speech (pauses
    excluded); noise level = mean power of the band-limited noise. Both are
    measured on the 300-3400 Hz band that survives the phone line, so the label
    means the same thing for every voice and every noise group. If the mix
    would clip, speech and noise are scaled together (the SNR is unchanged).
    """
    noise = np.asarray(noise[: len(x)], dtype=np.float64)
    s_lvl = inband_active_level(x)
    n_lvl = db(power(_bandlimit_16k(noise)))
    gain = 10 ** ((s_lvl - snr_db - n_lvl) / 20)
    seg = noise * gain
    mix = np.asarray(x, dtype=np.float64) + seg
    peak = np.max(np.abs(mix)) if len(mix) else 0.0
    scale = 0.99 / peak if peak > 0.99 else 1.0
    mix, seg = mix * scale, seg * scale
    if meta is not None:
        meta["snr_inband_target_db"] = float(snr_db)
        meta["snr_fullband_whole_db"] = db(power(x)) - db(power(noise * gain))
    if return_noise:
        return mix.astype(np.float32), seg.astype(np.float32)
    return mix.astype(np.float32)


def muffle(x: np.ndarray, cutoff: float = 1800.0, drop_db: float = -8.0,
           rt60: float = 0.25, wet: float = 0.15, seed: int = 0) -> np.ndarray:
    """Mild pendant-under-clothing simulation: low-pass near 1.8 kHz, -8 dB level drop
    (after the AGC, so not compensated), light synthetic room reverb (RT60 0.25 s, 15 % wet)."""
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
# Conditions registry
# --------------------------------------------------------------------------
Condition = Callable[[np.ndarray, int, dict], np.ndarray]


def post_channel_snr(speech: np.ndarray, noise: np.ndarray) -> float:
    """SNR actually reaching the ASR: speech and noise sent separately through the phone
    line (send filter + G.711), active speech level vs mean noise power, at 8 kHz."""
    return active_level(phone_8k(speech), PHONE_SR)[0] - db(power(phone_8k(noise)))


def _noisy(snr: float, group: str | None, then: Callable[[np.ndarray, int, dict], np.ndarray]):
    """Noise at `snr` dB as measured after the handset filter and codec.

    First pass: in-band (300-3400 Hz) active SNR at the microphone (add_noise).
    Second pass: the noise gain is corrected by the residual error measured
    after the line (post_channel_snr), so the label is the SNR the ASR receives,
    for every voice and noise group. The achieved value is logged per clip.
    """
    def f(x, seed, meta):
        g = group or noise_group(seed)
        meta["noise_group"] = g
        bed = noise_bed(g, len(x), seed)
        mix, seg = add_noise(x, bed, snr, meta=meta, return_noise=True)
        first = post_channel_snr(mix - seg, seg)
        corr = 10 ** ((first - snr) / 20)  # > 1: too clean, raise the noise
        mix = (mix - seg) + seg * corr
        seg = seg * corr
        peak = float(np.max(np.abs(mix)))
        if peak > 0.99:
            mix, seg = mix * (0.99 / peak), seg * (0.99 / peak)
        meta["snr_post_channel_db"] = post_channel_snr(mix - seg, seg)
        meta["snr_mic_inband_db"] = float(snr - 20 * np.log10(corr))
        meta["snr_fullband_whole_db"] = float(meta["snr_fullband_whole_db"] - 20 * np.log10(corr))
        return then(mix.astype(np.float32), seed, meta)
    return f


CONDITIONS: dict[str, Condition] = {
    "clean": lambda x, seed, meta: x.astype(np.float32),
    "phone": lambda x, seed, meta: telephone(x),
    "phone+noise10": _noisy(10, None, lambda x, s, m: telephone(x)),
    "phone+noise5": _noisy(5, None, lambda x, s, m: telephone(x)),
    "phone+muffle": lambda x, seed, meta: telephone(muffle(x, seed=seed)),
    "phone+loss5": lambda x, seed, meta: phone_with_loss(x, 0.05, seed, meta=meta),
    "opus-nb": lambda x, seed, meta: opus_nb(x, 0.0, seed, meta=meta),
    "opus-nb+loss5": lambda x, seed, meta: opus_nb(x, 0.05, seed, meta=meta),
    "g722-wb": lambda x, seed, meta: g722_wb(x),
    # Plausible bad-but-common call: street noise at 10 dB in-band SNR, G.711 line,
    # 3 % bursty loss (mean burst 3 packets = 60 ms) with Appendix I-style concealment.
    "realistic": _noisy(10, "street", lambda x, s, m: phone_with_loss(x, 0.03, s, burst=3, meta=m)),
}
STRESS_CONDITIONS = ("phone+noise5", "phone+loss5", "opus-nb+loss5")


def apply(name: str, x: np.ndarray, seed: int = 0, meta: dict | None = None) -> np.ndarray:
    """Apply condition `name` to prepared 16 kHz audio with a deterministic seed."""
    m = {} if meta is None else meta
    return CONDITIONS[name](np.asarray(x, dtype=np.float32), seed, m)
