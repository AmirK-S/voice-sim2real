# Phantom speech and guards, v2

Code: `scripts/09b_phantom_v2.py` (steps `segments`, `features`, `run`, `randlen`, `nsp`, `analyze`) and `scripts/09c_whisper_gate_silero.py` (steps `silero`, `defaults --set main|randlen`, `analyze`). Runs of 13/10/2026 on the same Mac, one ASR model on the GPU at a time. Extra packages, installed in the venv and not added to `pyproject.toml`: `webrtcvad-wheels` 2.0.14.post1 and `silero-vad` 6.2.3 (installed with `--no-deps`; it uses the torch already in the venv and its bundled jit weights, nothing was downloaded).

**Run status.** An overnight run was cut by a power loss. Before the cut, `segments`, `features`, the Parakeet and wav2vec2 shards, all three `randlen` checks and `nsp` had finished; the Whisper shards of the main v2 design had not been run, so the first `analyze` covered only Parakeet and wav2vec2. The Whisper shards (5,252 clips) were run on 13/10, then `analyze` was re-run for the three models. Every v2 output now covers all three models.

A **phantom** is any non-empty transcript (after the scoring normalizer) on a clip with no speech. A **guard** throws away the whole transcript of a clip. Everything is offline and per clip.

## 0. What the "33 of 100" in REPORT.md is, and the default-settings rate

**The 33 of 100 in REPORT.md** (also README and FINDINGS section 6) is Whisper large-v3-turbo on the main run's noise-only check: 100 clips of street, rain, crowd and home noise, each **exactly 4.00 s**, at -36 dB after the G.711 phone line, decoded with the **raw decoder** of `asr._whisper_batch`: `mlx_whisper.decoding.decode`, greedy at temperature 0, language "en", **no temperature fallback and no no-speech gate**, and a 30 s input window made of the clip's log-mel followed by the log-mel of 30 s of digital silence. Re-decoding the same clips today gives 34 of 100 (99 of 100 transcripts identical; one crowd clip gives "laughter" instead of "."), which I attribute to fp16 batch effects.

**With Whisper's standard settings the rate is much higher** (`gate_noise_only_arms.csv`; Clopper-Pearson 95 % CIs):

| Whisper large-v3-turbo on the main run's noise clips | 4.00 s clips (main run) | 3 to 6 s clips (random lengths) |
|---|---|---|
| Raw decoder, no gate, no fallback (as in the main run) | 34 / 100 [24.8, 44.2] | 56 / 100 [45.7, 65.9] |
| Raw decoder + default no-speech gate applied post hoc | 34 / 100 (the gate fires on 0 clips) | 56 / 100 (0 clips) |
| Raw decoder with the window built as `transcribe()` builds it | 94 / 100 [87.4, 97.8] | 91 / 100 [83.6, 95.8] |
| `mlx_whisper.transcribe`, all defaults (language detected) | **95 / 100 [88.7, 98.4]** | **100 / 100 [96.4, 100]** |
| `mlx_whisper.transcribe`, defaults + language "en" | 95 / 100 | 100 / 100 |
| `transcribe`, language "en", word timestamps, `hallucination_silence_threshold` = 2 s | 95 / 100 | 91 / 100 [83.6, 95.8] |
| Raw decoder + Silero VAD default as a guard | 0 / 100 [0, 3.6] | 0 / 100 [0, 3.6] |
| `transcribe` defaults + Silero VAD default as a guard | 2 / 100 [0.2, 7.0] | 3 / 100 [0.6, 8.5] |

- **The realistic default-settings rate on this noise is 95 to 100 %, not 33 %.** An off-the-shelf call to `mlx_whisper.transcribe` writes text on almost every clip, mostly "thank you" (80 of the 95 phantoms at 4.00 s). The 33 % understated the problem in two independent ways: the whole-second clip length (see `../NOTES.md`, and 56 % at random lengths) and the input window.
- **Why the default pipeline is worse: the input window, not the fallback.** `transcribe()` slices the clip's own log-mel and pads it with zeros in the normalized log-mel domain, which is not what silence looks like after normalization. `asr._whisper_batch` pads with the log-mel of real digital silence. Changing only this in the raw decoder takes the rate from 34 to 94 at 4.00 s. Temperature fallback was used on 21 (4.00 s) and 27 (random lengths) clips and did not remove text.
- **The default gate never fires.** large-v3-turbo under mlx-whisper 0.4.3 returns a no_speech_prob of about 1e-10 on every clip: median 5.2e-11 and maximum 1.4e-10 on the 4.00 s noise clips; over all 5,252 v2 clips the maximum is 5.2e-10 and the gate (no_speech_prob > 0.6 and avg_logprob < -1) fires on 0 clips, speech or non-speech. `nsp_check.json` confirms the no-speech token id (50363, `<|nospeech|>`), so this is the model's output, not a wrong token. The gate's speech cost is therefore zero, and so is its benefit.
- **`hallucination_silence_threshold`** at 2 s (my choice; the published 20 s cannot trigger on clips this short) changed little: 95 to 95 at 4.00 s, 100 to 91 at random lengths.
- **No emergency keyword** was invented in any of these 200 clips, under any arm (0.24 h of noise in all).
- **Consequence for the main run, not measured here:** every Whisper WER in the main run also went through `asr._whisper_batch`, whose docstring says the mel is "computed exactly as mlx_whisper.transcribe does". For clips of 30 s or less that is not true for the padding. The effect of this on speech WER is not measured.

## 1. v2 non-speech set: phantom rates

1,320 non-speech rows per model: 17 kinds x 40 clips (20 long 3 to 10 s, 20 short 0.5 to 2 s), each through two channels (comfort noise one), matched to the speech side's chain and random level. Raw decoder for Whisper. CIs: cluster bootstrap over clips (`summary.json`) and Clopper-Pearson per subset (`phantom_rates_v2.csv`).

| Model | All 1,320 rows | Long clips | Short clips | Clips with an invented emergency keyword |
|---|---|---|---|---|
| Whisper large-v3-turbo | **87.5 % [85.1, 89.9]** | 85.2 % | 89.8 % | 21, all bot echo |
| wav2vec2-base-960h | 33.6 % [30.0, 37.2] | 42.0 % | 25.3 % | 15, all bot echo |
| Parakeet TDT 0.6B v2 | 21.4 % [18.6, 24.4] | 8.3 % | 34.5 % | 20, all bot echo |

- **Whisper by kind:** 100 % on digital silence, comfort noise, line clicks and bot echo; 96 to 99 % on near-silence, hiss, rain, keyboard and sneezing; lowest on wind (62.5 %), breathing (63.8 %) and DTMF (67.5 %). Matched noise (the speech chain with the speech removed): 77.5 %.
- **Every invented emergency keyword comes from bot echo**, where the "phantom" is the model correctly transcribing the bot's own prompt ("help is on the way", "smoke", "ambulance"). That is a self-speech problem for echo cancellation, not a hallucination. Outside bot echo, no model invented a keyword: 0 in 1,240 rows (640 distinct clips, 1.36 h of audio), a rule-of-three bound of about 0.5 % per clip, or about 2.2 false alarms per hour of non-speech.
- **Parakeet's rate rose from v1 (0.7 %) to 21.4 %**, driven by short clips (34.5 %). v1 had no clips under 3 s. Parakeet is robust on long non-speech, not on short bursts.
- **Random lengths** (main-run recipe, 3 to 6 s): Whisper 56 / 100, wav2vec2 23, Parakeet 1 (`noise_only_random_lengths.csv`).

## 2. Guards on the v2 design

Speech side: 1,000 human clips (LibriSpeech and FLEURS, phone+noise10 and realistic, random level), 1,000 human 1 to 3 word segments, 1,708 TTS caller readings, 224 one-word TTS clips. Cost metrics: share of correctly recognized words thrown away ("words lost"), keyword hits lost on callers, one-word clips recognized and then thrown away. 95 % CIs from the cluster bootstrap (1,000 draws); for tuned rules the bootstrap re-runs the whole cross-fitted selection (nested).

**Whisper, operating points** (`silero_and_gate_v2.csv`, `recommended_points_v2.csv`):

| Guard | Phantoms left | Human words lost | Segment words lost | One-word clips dropped (of 162 recognized) | Keyword hits lost | Calls lost (of 1,158) |
|---|---|---|---|---|---|---|
| None | 87.5 % [85.0, 89.7] | 0 | 0 | 0 | 0 | 0 |
| Whisper default gate | 87.5 % (fires on 0 clips) | 0 | 0 | 0 | 0 | 0 |
| **Silero VAD default** (drop if no speech segment; threshold 0.5, min speech 250 ms) | **7.2 % [5.2, 9.2]** | **0.00 % [0, 0]** | 1.37 % [0.82, 2.04] | 7 (4.3 % [0.6, 10.2]) | 0 % [0, 0] | 0 |
| Tuned v2 rule, chosen on human clips + segments (energy VAD OR avg_logprob) | 43.9 % [39.1, 48.9] | 0.00 % [0, 0.24] | 1.93 % [1.54, 4.82] | 6 (3.7 % [0, 18.4]) | 0 % [0, 1.9] | 0 |
| Tuned v2 rule, chosen on human full clips only | 18.5 % [8.6, 23.0] | 0.52 % [0.24, 2.24] | 50.5 % [21.5, 54.4] | 156 (96 %) | 16.3 % [6.0, 36.4] | 200, in 56 of 84 scripts |

- **Silero's default operating point is the best Whisper guard measured here.** It removes 92 % of phantom rows (1,155 to 95) and throws away none of the 1,000 human clips. It costs 1.4 % of the words in 1 to 3 word segments and 7 of 162 recognized one-word clips. It is a fixed published setting, not tuned on this data, so there is no selection effect to cross-fit.
- **What Silero leaves:** 76 of the 95 remaining Whisper phantom rows are bot echo, which is real speech and which a VAD should not remove. Excluding bot echo, 19 of 1,240 rows keep text (1.5 %): sneezing 10, crowd events 4, crying baby 2, engine 2, matched noise 1. Silero triggers on 24 of 1,240 non-echo rows (1.9 %).
- **This weakens the v2 tuned-guard result.** The tuned energy-VAD-plus-score rule leaves 43.9 % of phantoms at a similar speech cost; Silero alone leaves 7.2 %. A threshold on Silero's maximum speech probability or speech seconds, tuned with the same cross-fitted rule, gives 6.4 to 7.3 % (point estimates, no nested CI) when chosen on human clips plus segments, so tuning adds little over the default.
- **The rule chosen on full human clips only is not usable.** Out of sample it loses 0.52 % of human words, above the 0.25 % budget it was chosen under, and it drops almost every short utterance (96 % of recognized one-word clips, 200 detected calls). Short utterances must be in the selection set.
- **Other models with Silero default:** Parakeet 21.4 % to 5.9 % [4.1, 7.7], wav2vec2 33.6 % to 7.2 % [5.2, 9.1], both with 0 human words lost, 0 keyword hits lost, and 5 of 135 (Parakeet) and 2 of 51 (wav2vec2) recognized one-word clips dropped.

## 3. Comparison with published work

- **Barański et al., ICASSP 2025 [71]:** Whisper large-v3 produced text on 40.3 % of 301,317 non-speech files; "thank you" was 24.8 % of hallucinations; Silero VAD was the best simple guard. Our default-settings rate on phone-line noise (95 to 100 %) is higher and our raw-decoder rate at 4.00 s (34 %) is lower, which shows how much the number depends on the decoding path, the clip length and the audio. The comparison is not like for like: a different model (large-v3 vs large-v3-turbo), wideband vs narrowband audio, their corpus vs ESC-50 and street noise at fixed levels. Their 40.3 % was probably gated, and for large-v3 the gate may work; for large-v3-turbo under mlx-whisper it never fires, so "gated" and "ungated" are the same here. "Thank you" dominates in both (80 of 95 phantoms here). Our Silero result agrees in direction with theirs.
- **Calm-Whisper [72]** scores any non-empty output, as we do, and cites Silero VAD v5 at 61 % utterance-level accuracy on ESC-50. On our phone-line ESC-50 and synthetic noise, Silero 6.2.3 at its default flagged speech on only 1.9 % of non-echo rows. Different version, task and metric; I do not reconcile the two.
- **Careless Whisper, Koenecke et al., FAccT 2024 [70]:** about 1 % of Whisper API transcriptions of real speech contained invented content, 38 % of it harmful, more often for speakers with long non-vocal stretches. We test the extreme of that condition (no speech at all) and do not measure hallucinations inside speech. Their harm finding is the reason the keyword count matters: here no keyword was invented outside bot echo (bound about 0.5 % per clip), but the fluent "thank you" that dominates would pass any downstream intent layer.

## Limits

- **Default-pipeline arms cover only the 200 main-recipe noise clips** (four noise types, -36 dB, phone line). `transcribe()` was not run on the 1,320 v2 non-speech rows or on speech, so the v2 rates in sections 1 and 2 are raw-decoder rates, and the speech-side cost of the default pipeline is not measured. Given section 0, the v2 raw-decoder rates are probably a low estimate of what `transcribe()` would give.
- **Silero is applied offline on the whole clip.** A streaming system applies it per frame with its own hysteresis; per-segment behaviour is not tested.
- **TTS for short utterances and callers.** One-word clips and callers are TTS; human short utterances are cut from read speech with Parakeet timestamps (+-0.1 s), which favours clean word boundaries.
- **Noise is not real call audio** (ESC-50, CC BY-NC, and synthetic line noise), and the mix of kinds is an arbitrary choice that sets the overall rates.
- **Fixed-rule CIs** (gate, Silero default) hold the rule fixed, which is right for a fixed rule; the tuned Silero thresholds have point estimates only.
- **Whisper decoding is not bit-reproducible across batches** (1 of 100 noise clips changed between two runs).

## Files

- `gate_noise_only_arms.csv`: every arm of section 0, with no_speech_prob median and max, gate firings, Silero triggers, fallback use, detected languages
- `gate_top_phrases.csv`: most frequent phantom texts per arm
- `silero_and_gate_v2.csv`, `silero_by_kind_v2.csv`, `summary_gate_silero.json`: section 2 for the gate and Silero
- `phantom_rates_v2.csv`, `phantom_emergency_keywords_v2.csv`, `level_by_kind_v2.csv`, `noise_only_random_lengths.csv`, `nsp_check.json`, `summary.json`: section 1
- `recommended_points_v2.csv`, `guards_crossfit_v2.csv`, `residual_phantoms_by_kind_v2.csv`, `short_utterances_v2.csv`: tuned guards
- Figures: `fig1_v2_phantom_rate_by_kind.png`, `fig2_v2_guard_tradeoffs.png`
- Raw shards (gitignored): `results/raw/followups/phantom_v2/` (`whisper/`, `parakeet/`, `wav2vec2/`, `whisper_defaults/`, `silero*.csv`, `features.csv`, `segments.csv`)
