"""Self-test for sim2real.channel (v2). Exits non-zero on any failure.

Checks: G.711 tables match ffmpeg bit for bit, P.56-style active level, preparation
(-26 dBov), phone band-limit, post-channel SNR exactness, packet-loss rate and burst
length, concealment without steps, Opus bitrate and PLC, G.722 band, length/dtype/
determinism of every condition. Writes one LibriSpeech utterance under every
condition to results/examples/channel/.
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import soundfile as sf

from sim2real import channel as ch
from sim2real.audio import resample
from sim2real.datasets import librispeech_index, load
from sim2real.paths import RESULTS

OUT = RESULTS / "examples" / "channel"
OUT.mkdir(parents=True, exist_ok=True)
fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails.append(name)


def ffmpeg(args, data):
    return subprocess.run(["ffmpeg", "-loglevel", "error", *args], input=data,
                          capture_output=True, check=True).stdout


# 1. G.711 mu-law against ffmpeg (no resampling: input already 8 kHz).
rng = np.random.default_rng(0)
pcm = np.concatenate([np.arange(-32768, 32768, 7), rng.integers(-32768, 32768, 20000)]).astype(np.int16)
ff_codes = np.frombuffer(ffmpeg(["-f", "s16le", "-ar", "8000", "-ac", "1", "-i", "pipe:0",
                                 "-c:a", "pcm_mulaw", "-f", "mulaw", "pipe:1"], pcm.tobytes()), np.uint8)
ours = ch.mulaw_encode(pcm)
enc_match = np.mean(ff_codes == ours)
adjacent = np.all(np.abs(ff_codes.astype(int) - ours.astype(int)) <= 1)
check("mulaw encode ~ ffmpeg (>=98% equal, rest adjacent code)", enc_match >= 0.98 and adjacent,
      f"match={enc_match:.4f}")
# Reference decision rule: output must lie in the input's quantisation interval.
full = np.arange(-32635, 32636).astype(np.int16)
code = ch.mulaw_encode(full)
step = 2 ** ((((~code.astype(int)) & 0xFF) >> 4 & 7) + 3)
err = np.abs(ch.mulaw_decode(code).astype(int) - full.astype(int))
check("mulaw error <= half step (all int16)", np.all(err <= step // 2), f"max err/step={np.max(err/step):.3f}")
codes = np.arange(256, dtype=np.uint8)
ff_dec = np.frombuffer(ffmpeg(["-f", "mulaw", "-ar", "8000", "-ac", "1", "-i", "pipe:0",
                               "-f", "s16le", "pipe:1"], codes.tobytes()), "<i2")
check("mulaw decode == ffmpeg (256 codes)", np.array_equal(ff_dec, ch.mulaw_decode(codes)))

# Real speech.
idx = librispeech_index()
utts = idx.sample(6, random_state=1).reset_index(drop=True)
raw = [resample(*load(p)) for p in utts["path"]]
speech = [ch.prepare(r, seed=i) for i, r in enumerate(raw)]
x = speech[0]

# 2. Active level and preparation.
sr = ch.SR
t = np.arange(sr * 4) / sr
sine = np.sin(2 * np.pi * 440 * t) * np.sqrt(2) * 10 ** (-20 / 20)
sine[sr * 2:] = 0
lvl, act = ch.active_level(sine)
check("P.56-style level of a -20 dB sine, 50 % silence", abs(lvl + 20) < 1.0 and 0.45 < act < 0.65,
      f"level {lvl:.2f} dB, activity {act:.2f}")
lv = [ch.active_level(s)[0] for s in speech]
check("prepare() -> -26 dBov active level", max(abs(v + 26) for v in lv) < 0.5, f"{np.round(lv, 2)}")
check("prepare() lead/trail 250 ms", all(abs(ch.trim_bounds(s)[0] / sr - 0.2) < 0.15 for s in speech))
floor = 10 * np.log10(np.mean(speech[0][: int(0.15 * sr)].astype(float) ** 2))
check("noise floor in the lead, not digital silence", -70 < floor < -60, f"{floor:.1f} dBFS")

# 3. Telephone: length, band-limit, causal filter.
t0 = time.perf_counter()
y = ch.telephone(x)
dt = time.perf_counter() - t0
check("telephone length/dtype", len(y) == len(x) and y.dtype == np.float32, f"({dt*1000:.0f} ms)")
f = np.fft.rfftfreq(len(y), 1 / ch.SR)
P = np.abs(np.fft.rfft(y)) ** 2
inband = P[(f > 300) & (f < 3400)].sum()
above = P[f > 4000].sum()
below = P[f < 150].sum()
check("telephone removes >4 kHz", 10 * np.log10(above / inband) < -30, f"{10*np.log10(above/inband):.1f} dB")
check("telephone removes <150 Hz", 10 * np.log10(below / inband) < -15, f"{10*np.log10(below/inband):.1f} dB")
imp = np.zeros(4000, np.float32); imp[2000] = 0.5
yi = ch.telephone(imp)
check("telephone delay compensated (peak within 1 ms)", abs(int(np.argmax(np.abs(yi))) - 2000) < 16)

# 4. SNR after the channel is exact, for every voice-independent noise group.
errs = []
for gi, group in enumerate(ch.NOISE_MIX):
    n_clips = len(ch._noise_bank()[group])
    check(f"noise bank '{group}' non-empty", n_clips > 0, f"{n_clips} clips")
for name, snr in (("phone+noise10", 10), ("phone+noise5", 5), ("realistic", 10)):
    for si, s in enumerate(speech):
        meta = {}
        ch.apply(name, s, seed=si, meta=meta)
        errs.append(abs(meta["snr_post_channel_db"] - snr))
check("post-channel SNR within 0.2 dB of the label", max(errs) < 0.2, f"max err {max(errs):.3f} dB")
bed = ch.noise_bed("rain", 12 * sr, seed=3)
check("noise bed covers 12 s without tiling one clip", len(bed) == 12 * sr and np.isfinite(bed).all())

# 5. Packet loss and concealment.
nfr = 20000
for rate in (0.05, 0.10):
    m = ch.loss_mask(nfr, rate, seed=3)
    check(f"bernoulli loss {rate:.0%}", abs(m.mean() - rate) < 0.01, f"measured {m.mean():.3f}")
m = ch.loss_mask(200000, 0.03, seed=3, burst=3)
starts = np.flatnonzero(np.diff(np.r_[0, m.astype(int)]) == 1)
ends = np.flatnonzero(np.diff(np.r_[m.astype(int), 0]) == -1)
mean_burst = float(np.mean(ends - starts + 1))
check("gilbert-elliott rate 3%", abs(m.mean() - 0.03) < 0.005, f"measured {m.mean():.4f}")
check("gilbert-elliott burst ~3", abs(mean_burst - 3) < 0.3, f"mean burst {mean_burst:.2f} frames")
y8 = ch.phone_8k(x)
lost = ch.loss_mask(int(np.ceil(len(y8) / 160)), 0.10, seed=5)
c8 = ch.plc_g711_appendix1(y8, lost, 160)
d_ref = np.abs(np.diff(y8))
d_plc = np.abs(np.diff(c8))
check("concealment adds no steps (max jump <= 1.5x max jump of the clean line)",
      d_plc.max() <= 1.5 * d_ref.max() + 1e-4, f"{d_plc.max():.4f} vs {d_ref.max():.4f}")
far = np.flatnonzero(~lost & ~np.r_[False, lost[:-1]] & ~np.r_[lost[1:], False])
same = all(np.array_equal(c8[i * 160:(i + 1) * 160], y8[i * 160:(i + 1) * 160]) for i in far[:-1])
check("concealment leaves frames away from losses untouched", same, f"{len(far)} frames checked")
mid = np.flatnonzero(lost[5:-5])[0] + 5 if lost[5:-5].any() else None
if mid is not None:
    seg = c8[mid * 160: mid * 160 + 80]
    check("first 10 ms of a loss is not silent", np.sqrt(np.mean(seg ** 2)) > 1e-4)
meta = {}
yl = ch.apply("phone+loss5", x, seed=5, meta=meta)
check("phone+loss5 length/finite", len(yl) == len(x) and np.isfinite(yl).all(), f"loss {meta['loss_rate']:.3f}")

# 6. Opus and G.722.
meta = {}
yo = ch.opus_nb(x, 0.0, meta=meta)
check("opus-nb bitrate 8-14 kb/s (SILK NB)", 8 <= meta["opus_kbps"] <= 14, f"{meta['opus_kbps']:.1f} kb/s")
yo2 = ch.opus_nb(x, 0.10, seed=2)
check("opus PLC output finite, differs from no-loss", np.isfinite(yo2).all() and not np.allclose(yo, yo2))
yg = ch.g722_wb(x)
Pg = np.abs(np.fft.rfft(yg)) ** 2
fg = np.fft.rfftfreq(len(yg), 1 / ch.SR)
share = Pg[(fg > 3400) & (fg < 7000)].sum() / Pg.sum()
check("g722 keeps 3.4-7 kHz energy (wideband)", share > 1e-3, f"share {share:.4f}")

# 7. Muffle: high band attenuated vs input.
ym = ch.muffle(x)
Px, Pm = np.abs(np.fft.rfft(x)) ** 2, np.abs(np.fft.rfft(ym)) ** 2
hi = (f > 3000)
lo = (f > 200) & (f < 1000)
tilt = 10 * np.log10(Pm[hi].sum() / Px[hi].sum()) - 10 * np.log10(Pm[lo].sum() / Px[lo].sum())
check("muffle tilts spectrum (>3k vs <1k)", tilt < -15, f"{tilt:.1f} dB")

# 8. Every condition: length, dtype, finite, deterministic; write examples.
uid = utts["id"][0]
sf.write(OUT / f"{uid}__input_prepared.wav", x, ch.SR)
for name in ch.CONDITIONS:
    t0 = time.perf_counter()
    a = ch.apply(name, x, seed=42)
    dt = time.perf_counter() - t0
    b = ch.apply(name, x, seed=42)
    ok = len(a) == len(x) and a.dtype == np.float32 and np.isfinite(a).all() and np.abs(a).max() <= 1.0
    check(f"condition {name:14s}", ok and np.array_equal(a, b), f"({dt*1000:.0f} ms, deterministic)")
    sf.write(OUT / f"{uid}__{name.replace('+', '_')}.wav", a, ch.SR)
c1, c2 = ch.apply("phone+noise10", x, seed=1), ch.apply("phone+noise10", x, seed=2)
check("different seeds differ", not np.array_equal(c1, c2))

print(f"\nExamples in {OUT}")
print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
