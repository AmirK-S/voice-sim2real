# Phantom speech and guards

Code: `scripts/09_phantom.py` (steps `features`, `run`, `length`, `analyze`) and `src/sim2real/guards.py`. Run of 13/10/2026, same Mac, one model on the GPU at a time. Extra package: `webrtcvad-wheels` 2.0.14.post1 (installed in the venv with `uv pip install webrtcvad-wheels`, not added to `pyproject.toml`).

A **phantom** is any non-empty transcript (after the scoring normalizer) on a clip with no speech. A **guard** is a rule that throws away the whole transcript of a clip. Everything here is offline and per clip: a streaming pipeline would apply the same signals per segment, which is not tested.

## Set-up

**Non-speech clips (300 per channel, 600 per model).** 30 clips each of:
- ESC-50 rain, wind, engine, keyboard typing, door knock, and "crowd events" (laughing, clapping, coughing, footsteps). Clips are chained with crossfades, set to a phone-band level of -46 to -26 dB (what a -26 dBov talker would have at 20 to 0 dB SNR), plus the -65 dBFS floor.
- line hiss (white, -55 to -35 dB), comfort noise (pink, -70 to -50 dB), digital silence (zeros), near-silence (the stationary floor only, -80 to -60 dB).

Each clip is 3 to 10 s long (random, two decimals) and is sent through the clean channel and the phone channel (G.711). ESC-50 has no TV, music or babble class, so "TV-like" sound is not covered. Real TV audio contains speech anyway, so it is a different problem.

**Speech clips (2,708 per model).** The real human recordings (250 LibriSpeech, 250 FLEURS) and the 122 caller scripts read by the 7 TTS voices, each under phone+noise10 and realistic. The preparation, channel code and seeds are those of the main run and the FLEURS follow-up.

**Check.** The decoder that also returns the scores gives exactly the cached main-run transcript on every clip that was already in the cache: 2,459 of 2,459 for Whisper, and 2,708 of 2,708 for Parakeet and for wav2vec2.

## 1. How often the models write text when nobody speaks

| Model | Clean channel | Phone channel | Both (95 % CI, clips clustered) |
|---|---|---|---|
| Whisper large-v3-turbo | 248 / 300 | 262 / 300 | **85.0 % [82.2, 87.8]** |
| wav2vec2-base-960h | 47 / 300 | 53 / 300 | 16.7 % [12.8, 20.8] |
| Parakeet TDT 0.6B v2 | 2 / 300 | 2 / 300 | 0.7 % [0.2, 1.3] |

**By kind** (`phantom_rates.csv`, `fig1_phantom_rate_by_kind.png`):
- **Whisper** writes text on every kind, including **digital silence (60 of 60)**, near-silence (53 of 60) and comfort noise (46 of 60).
- **wav2vec2** writes text mostly on door knocks (50 of 60) and crowd events (42 of 60), and never on silence, hiss or comfort noise.
- **Parakeet** writes text on 4 clips: wind, engine and two crowd-events clips.

**Most common invented phrases** (`phantom_phrases.csv`):
- **Whisper:** "thank you" (343 of 510 phantoms), "i am going to go" (25), "so" (24), "i am going to go to the next one" (20), "i am sorry" (7), "the end" (5). Clean-channel phantoms are longer (8.8 words on average) than phone-channel ones (2.5).
- **wav2vec2:** letter strings ("a a", "aaaaaata", "ahaha ahaahahahahahaa 00 got").
- **Parakeet:** "okay so", "yeah", "battem battem battem", "i guess i do not think it is like a terrible thing".

**Invented emergency keywords:** none for any model in 600 clips. The exact one-sided 95 % upper bound is 0.5 % of clips per model, or 1.0 % if the clean and phone versions of a clip are treated as one observation (300).

**Level:** Whisper's phantom rate barely depends on the noise level within -46 to -26 dB (82 to 90 % per 5 dB bin, pooled over the ESC kinds and channels; `phantom_by_level.csv`).

**Whisper's rate depends on the exact clip length.** The main run reported Whisper text on 33 of 100 clips of pure noise. Those clips were all exactly 4.00 s long. I re-created that check exactly, at several lengths (`phantom_by_exact_length.csv`):

| Length | 4.00 s | 4.01 s | 4.02 s | 4.10 s | 4.50 s | 5.00 s | 5.37 s |
|---|---|---|---|---|---|---|---|
| Whisper, clips with text (of 100) | 34 | 59 | 58 | 63 | 61 | 20 | 67 |

- **Whole-second clips give about half the phantoms.** 4.01 s vs 4.00 s: 27 clips gain text, 2 lose it, exact McNemar p = 1.6e-6.
- **The other two models do not change:** wav2vec2 stays at 19 to 22 and Parakeet at 0 to 3 at every length.
- **What this means for the main run:** its "33 of 100" figure is a low estimate for Whisper. For clips of arbitrary length the rate is about 60 % at that noise level, and 85 % over the kinds above.
- **Cause not investigated:** I did not find why whole-second lengths matter.

**Whisper's no-speech gate does nothing.** large-v3-turbo returns a no_speech_prob of about 1e-10 on every clip, speech or silence: the median is 8.6e-11 on non-speech and 6.0e-12 on speech. This holds through `mlx_whisper.transcribe` as well: a 5 s clip of zeros comes back as "Thank you." with no_speech_prob 1.1e-10. The default rule (skip when no_speech_prob > 0.6 and avg_logprob < -1) never fires: 85.0 % of non-speech clips still have text. The probability still ranks clips, so it is used below on a log scale.

## 2. Guards: phantoms removed vs real words dropped

**Metrics:**
- **Phantoms left:** the share of non-speech clips that still have text after the guard.
- **Words dropped:** the correctly recognized reference words in speech clips whose transcript the guard throws away, as a share of all reference words. Human and caller clips are pooled; both are also reported separately.
- **Keywords and calls:** emergency keyword hits lost, and caller calls whose detection is lost.

**Guards tested:**
- py-webrtcvad modes 0 to 3, and an energy VAD (300 to 3400 Hz frame energy above the floor estimate + 9 dB and above -55 dB). For both, the gate is the seconds of detected speech.
- Whisper avg_logprob, Whisper log10 no_speech_prob, and Whisper's default rule with re-tuned thresholds.
- Parakeet token confidence (parakeet-mlx's entropy-based confidence, geometric mean over tokens).
- wav2vec2 CTC confidence (mean max posterior over the non-blank frames).
- Each VAD OR each model score.

**How a point is chosen.** The rule was fixed before looking at any result: keep the fewest phantoms with words dropped at or below **0.25 %**, and on a tie take the fewer words dropped. It is applied by 2-fold cross-fitting over clusters (choose on one half, apply to the other), so the reported numbers are out of sample. CIs come from a cluster bootstrap (2,000 draws). Non-speech clusters are the clip (clean and phone together); speech clusters are the speaker (LibriSpeech), the sentence (FLEURS) or the script (callers).

**Recommended operating point per model** (`recommended_points.csv`, `fig2_operating_curves.png`). Both folds chose the same rule, with the same threshold except wav2vec2 (0.667 and 0.699).

| Model | Guard (full-data threshold) | Phantoms before | Phantoms left | Removed | Words dropped (all) | Words dropped (human) | Keyword hits lost | Calls lost |
|---|---|---|---|---|---|---|---|---|
| Whisper | energy VAD < 0.39 s **or** avg_logprob < -0.80 | 85.0 % | **20.8 % [17.5, 24.3]** | 75.5 % [71.5, 79.5] | 0.11 % [0.00, 0.26] | 0.10 % [0.00, 0.27] | 0 | 0 of 1,157 |
| wav2vec2 | CTC confidence < 0.667 | 16.7 % | **1.8 % [0.7, 3.3]** | 89.0 % [80.7, 95.9] | 0.20 % [0.12, 0.30] | 0.37 % [0.21, 0.56] | 0 | 0 of 742 |
| Parakeet | token confidence < 0.805 | 0.7 % (4 clips) | **0 of 600** | 4 of 4 | 0.00 % | 0.00 % | 0 | 0 of 1,154 |

- **Whisper's human WER** under the guard goes from 6.36 % to 6.46 %; wav2vec2's goes from 47.64 % to 47.78 %.
- **No calls lost.** With 0 calls lost out of about 1,150 detected calls, the rule of three gives an upper bound of about 0.26 % of calls. The calls come from 122 scripts read 7 times, though, so the real bound is wider.
- **Parakeet:** the guard removes all 4 phantoms, but 4 events cannot validate a threshold. The exact upper bound on Parakeet's remaining phantom rate is 1.0 % (0 of 300 clips).

**Each signal alone, Whisper, cross-fitted** (`guards_crossfit.csv`):

| Guard | Phantoms left | Words dropped |
|---|---|---|
| Default rule (no_speech_prob > 0.6 and avg_logprob < -1) | 85.0 % | 0 |
| log no_speech_prob | 64.0 % [59.8, 68.2] | 0.29 % |
| avg_logprob | 50.8 % [46.5, 55.2] | 0.21 % |
| webrtcvad mode 3 | 43.3 % [38.2, 48.7] | 0.12 % |
| energy VAD | 39.8 % [34.7, 45.0] | 0.24 % |
| webrtcvad mode 3 or avg_logprob | 21.3 % [18.0, 24.8] | 0.05 % [0.00, 0.13] |
| energy VAD or avg_logprob | 20.8 % [17.5, 24.3] | 0.11 % [0.00, 0.26] |

- **The two combinations are about equal**, and the webrtcvad one drops fewer words. Either is a reasonable default.
- **A VAD alone helps little for wav2vec2 and Parakeet** (at most 5 % of their phantoms removed), because their phantoms come from loud transient sounds that any VAD passes.

**What the guard cannot remove.** For Whisper, almost all remaining phantoms are transient sounds after the phone line: keyboard (28 of 30 clips still with text), door knock (23), crowd events (23) and engine (18). On the clean channel the same kinds keep 3 to 13 of 30. Silence, near-silence, hiss and comfort noise go to 0 of 30 on both channels, and rain and wind to 0 to 5 (`residual_phantoms_by_kind.csv`). The energy-and-confidence approach removes the "thank you on silence" failure, not text invented on loud non-speech events.

## Caveats

- **Thresholds are tied to this setup:** the -26 dBov level normalization, 250 ms padding, these channels and these models. They are not general recommendations. A system without level normalization needs its own curve.
- **Short utterances are not tested.** The speech clips are 1.6 to 12 s long. The Whisper rule drops clips with less than 0.39 s of energy-VAD speech, so a one-word "help!" could be dropped. In this set, 1 % of caller clips have less than 0.57 s of detected speech.
- **Clean speech is not tested.** The speech side is phone+noise10 and realistic only. Non-speech clips are clean and phone only, not the noise-plus-loss "realistic" chain.
- **Noise is not real call audio.** ESC-50 events and synthetic noise stand in for real call background. ESC-50 is CC BY-NC.
- **Phantom rates depend on the mix of kinds** (30 per kind), which is an arbitrary choice, and, for Whisper, on clip length (see above).
- **Keyword checks come from TTS callers only.** No human recordings of the caller scripts exist.
- **CIs leave out part of the selection effect.** Cross-fitting removes the in-sample bias of the chosen threshold, but the bootstrap holds the chosen rule fixed, so the CIs do not include the variability of choosing it.

## Files

- **Tables:**
  - `phantom_rates.csv`, `phantom_phrases.csv`, `phantom_emergency_keywords.csv` (empty)
  - `phantom_by_level.csv`, `phantom_by_exact_length.csv`
  - `operating_curves.csv`: in-sample curves; for the 2-D guards, only the Pareto front
  - `guards_crossfit.csv`, `recommended_points.csv`, `residual_phantoms_by_kind.csv`
  - `cache_agreement.csv`
  - `clips.csv.gz`: one row per clip, with VAD features, every model's text and scores, hits and reference words
- **Summary:** `summary.json`
- **Figures:** `fig1_phantom_rate_by_kind.png`, `fig2_operating_curves.png`
- **Raw shards:** `results/raw/followups/phantom/` (gitignored)
