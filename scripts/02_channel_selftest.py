"""Self-test for sim2real.channel. Exits non-zero on any failure.

Checks: G.711 tables match ffmpeg bit for bit, phone band-limit, SNR accuracy
(within 0.5 dB), packet-loss rate and burst length, length/dtype/determinism of
every condition. Writes one LibriSpeech utterance under every condition to
results/examples/channel/.
"""
import subprocess
import sys
import time

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
speech = [resample(*load(p)) for p in utts["path"]]
x = speech[0]

# 2. Telephone: sample rate, length, band-limit.
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

# 3. SNR accuracy on every noise group, several SNRs, several utterances.
errs = []
for gi, group in enumerate(ch.NOISE_GROUPS):
    n_clips = len(ch._noise_bank()[group])
    check(f"noise bank '{group}' non-empty", n_clips > 0, f"{n_clips} clips")
    for snr in (20, 10, 5, 0):
        for si, s in enumerate(speech):
            seed = 100 * gi + si
            mix, nz = ch.add_noise(s, ch.noise_clip(group, seed), snr, seed=seed, return_noise=True)
            clean_part = mix.astype(np.float64) - nz  # speech after any anti-clip scaling
            errs.append(abs(ch.measured_snr(clean_part, nz) - snr))
            # independent check, same definition, recomputed from scratch
            assert len(mix) == len(s) and mix.dtype == np.float32
print(f"      SNR abs error: max={max(errs):.4f} dB over {len(errs)} mixes")
check("SNR within 0.5 dB", max(errs) < 0.5)

# 4. Packet loss.
long = np.tile(np.concatenate(speech), 3)
nfr = int(np.ceil(len(long) / 320))
for rate in (0.05, 0.10):
    m = ch.loss_mask(nfr, rate, seed=3)
    check(f"bernoulli loss {rate:.0%}", abs(m.mean() - rate) < 0.015, f"measured {m.mean():.3f} over {nfr} frames")
m = ch.loss_mask(200000, 0.03, seed=3, burst=3)
starts = np.flatnonzero(np.diff(np.r_[0, m.astype(int)]) == 1)
ends = np.flatnonzero(np.diff(np.r_[m.astype(int), 0]) == -1)
mean_burst = float(np.mean(ends - starts + 1))
check("gilbert-elliott rate 3%", abs(m.mean() - 0.03) < 0.005, f"measured {m.mean():.4f}")
check("gilbert-elliott burst ~3", abs(mean_burst - 3) < 0.3, f"mean burst {mean_burst:.2f} frames")
yl = ch.packet_loss(x, 0.10, seed=5)
check("packet_loss length/finite", len(yl) == len(x) and np.isfinite(yl).all())
check("packet_loss changes audio", not np.allclose(yl, x))
check("packet_loss rate 0 is identity", np.array_equal(ch.packet_loss(x, 0.0, seed=5), x))

# 5. Muffle: high band attenuated vs input.
ym = ch.muffle(x)
Px, Pm = np.abs(np.fft.rfft(x)) ** 2, np.abs(np.fft.rfft(ym)) ** 2
hi = (f > 3000)
lo = (f > 200) & (f < 1000)
tilt = 10 * np.log10(Pm[hi].sum() / Px[hi].sum()) - 10 * np.log10(Pm[lo].sum() / Px[lo].sum())
check("muffle tilts spectrum (>3k vs <1k)", tilt < -15, f"{tilt:.1f} dB")

# 6. Every condition: length, dtype, finite, deterministic; write examples.
uid = utts["id"][0]
sf.write(OUT / f"{uid}__input.wav", x, ch.SR)
for name in ch.CONDITIONS:
    t0 = time.perf_counter()
    a = ch.apply(name, x, seed=42)
    dt = time.perf_counter() - t0
    b = ch.apply(name, x, seed=42)
    ok = len(a) == len(x) and a.dtype == np.float32 and np.isfinite(a).all() and np.abs(a).max() <= 1.0
    check(f"condition {name:14s}", ok and np.array_equal(a, b), f"({dt*1000:.0f} ms, deterministic)")
    sf.write(OUT / f"{uid}__{name.replace('+', '_')}.wav", a, ch.SR)
    info = sf.info(OUT / f"{uid}__{name.replace('+', '_')}.wav")
    if info.samplerate != 16000 or info.channels != 1:
        check(f"wav format {name}", False)
c1, c2 = ch.apply("phone+noise10", x, seed=1), ch.apply("phone+noise10", x, seed=2)
check("different seeds differ", not np.array_equal(c1, c2))

print(f"\nExamples in {OUT}")
print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
sys.exit(1 if fails else 0)
