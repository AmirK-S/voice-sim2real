# How much do synthetic callers lie? Findings (v2)

Run of 12-13/10/2026, local, on an Apple Silicon Mac. 73,746 scored transcriptions, plus 300 noise-only transcriptions. This version replaces the v1 findings (commit 54793f5) after a statistics review and a method review; every number below comes from the v2 rerun. The list of fixes is at the end.

## What this measures, and what it does not

**It measures the acoustic half of the sim-to-real gap on read speech.** The same scripts are read by real people and by 7 TTS voices, sent through simulated phone channels and transcribed by 3 public ASR models standing in for the voice agent's ear.

**It does not measure:**
- **Caller behaviour.** Real callers panic, hesitate, repeat, answer off-script and talk over the agent. Here every voice reads the identical script, so this half of the gap is absent.
- **A real voice-agent pipeline.** ASR is offline and non-streaming, on pre-segmented single utterances: no VAD, endpointing, barge-in or turn-taking.
- **Real emergency calls.** "Human" means LibriSpeech: audiobook volunteers reading in a quiet room. The production gap is certainly larger.

## Setup

- **LibriSpeech:** 250 test-clean sentences from 40 speakers. Each sentence has the human recording and 7 TTS renderings: Kokoro af_heart and am_michael, Piper lessac and ryan, Meta MMS-TTS, macOS Samantha and Daniel.
- **Rate-matched voices:** the same 7 TTS voices re-synthesized so that each sentence lasts as long as the human recording (median duration ratio 0.99 to 1.00; the macOS voices only reach 0.90 and 0.92 because `say -r` does not scale its pauses).
- **Caller set:** 122 written caller scripts (64 emergency, 16 fall-help, 16 fall-ok, 26 non-emergency), read by the 7 TTS voices. No human audio.
- **ASR models:**
  - Whisper large-v3-turbo (MLX, greedy)
  - Parakeet TDT 0.6B v2 (MLX, greedy)
  - wav2vec2-base-960h, used as an **in-domain control**. It was trained on LibriSpeech 960 h, so LibriSpeech test-clean is its home domain; its clean WER here, 3.3 %, matches its published test-clean result. Parakeet's training mix also includes LibriSpeech (from its model card, not re-checked here), and Whisper's training data is undocumented and likely includes audiobooks. The human baseline is therefore favoured for every model, most strongly for wav2vec2.
- **Preparation of every clip, clean included:**
  - edge silence trimmed, then a 250 ms lead and trail;
  - active speech level normalized to -26 dBov (P.56-style, like a handset AGC);
  - a -65 dBFS noise floor, so no voice has digital silence.
- **Channel conditions:**
  - clean
  - phone: 8 kHz, IRS-like handset filter (causal), G.711 mu-law
  - g722-wb: wideband handset filter, G.722 64 kb/s
  - opus-nb: Opus through libopus, 8 kHz VOIP mode, about 10 kb/s (SILK narrowband)
  - phone+noise10 and phone+noise5: ESC-50 noise (street, rain, crowd events) at 10 or 5 dB **measured after the phone line** (active speech level vs noise power; achieved 10.00 ± 0.004 dB per clip)
  - phone+muffle: mild pendant simulation (low-pass at 1.8 kHz, -8 dB, light reverb)
  - phone+loss5: 5 % lost 20 ms packets, concealment modelled on G.711 Appendix I
  - opus-nb+loss5: 5 % lost packets, concealed by the real libopus decoder
  - realistic: street noise at 10 dB, phone line, 3 % bursty loss
  - phone+noise5 and the two loss conditions are labelled stress conditions.
- **Grid actually run:**
  - wav2vec2 and Parakeet ran all 10 conditions.
  - Whisper ran 4 (clean, phone, phone+noise10, realistic), because it decodes at about 1.3 clips/s on this GPU.
  - The rate-matched voices ran on clean audio only, for all 3 models.
- **Scoring:** corpus WER after Whisper's English normalizer, now with its British-to-American spelling map. "Points" means WER percentage points.
- **Statistics:**
  - **CIs:** LibriSpeech CIs are 95 % **speaker-cluster** bootstrap intervals (40 speakers resampled, 10,000 replicates). In a simulation on this data, the old sentence-level bootstrap covered the human realistic WER only 87.5 % of the time; the speaker bootstrap covered it 94.3 %.
  - **p-values:** bootstrap CI-inversion p-values; the smallest reportable value is about 2e-4. Families of tests carry Holm (family-wise) or Benjamini-Hochberg (BH, false discovery rate) adjustments, named where used.
  - **Gap:** synthetic WER minus human WER on the same sentences, same channel, same model.
  - **DiD:** a lab-vs-reality comparison is tested by the **difference in differences**. That is the gap between two ASR models on a synthetic voice minus the same gap on human audio, bootstrapped per sentence.

## Findings

### 1. For two close ASR models, no single voice reliably distorts the comparison except MMS; the lab overstates certainty mainly when voices are averaged

The comparison is Whisper vs Parakeet: 7 voices x 4 conditions = 28 voice-level decisions.

**Reality.** On human audio, Parakeet is better on clean speech by 0.37 points [0.12, 0.67]. The other conditions show no clear winner:
- phone: +0.16 [-0.23, +0.57]
- phone+noise10: +0.49 [-0.15, +1.22]
- realistic: +0.60 [-0.04, +1.29]

**Single voices.**
- The DiD excludes 0 nominally in 5 of 28 decisions:
  - MMS in all 4 conditions, for example +1.60 [0.52, 2.62] points on phone+noise10;
  - Piper ryan on phone: +0.67 [0.16, 1.20].
- **None survives Holm correction**; the smallest adjusted p is 0.16.
- MMS consistently exaggerates Parakeet's advantage, because its rendering errors hurt Whisper more.

**Flips.** A flip now requires a real winner, so only clean audio qualifies. One voice flips the point estimate: macOS Daniel at -0.14 [-0.73, +0.49], which picks no one. Its DiD is -0.51 [-1.16, +0.14]. It is not a confident flip and its DiD is not significant.

**The 7-voice mean.** This is the "lab manufactures certainty" effect in the form the data supports.
- Averaging all voices gives the right winner on clean audio: +0.48 [0.17, 0.82].
- It also declares Parakeet significantly better on all three phone or noisy conditions, where real speech shows no winner.
- Its DiD is never significant, for example +0.15 [-0.48, +0.74] on realistic. Averaging 7 renderings of each sentence shrinks the lab's CI, not its bias.

**Large-gap pairs** (Whisper or Parakeet vs wav2vec2, 98 decisions). No voice flips the winner, but the size of the gap is distorted in both directions:
- Lab gap divided by real gap ranges from 0.37 to 3.72.
- On realistic, the real Whisper vs wav2vec2 gap is 21.7 points. The lab puts it at 9.6 points with Kokoro af_heart and 35.8 with macOS Daniel.
- The DiD is Holm-significant in 18 of 28 decisions for Whisper and 42 of 70 for Parakeet.

These pairs involve the in-domain control, whose human advantage is partly a training-domain advantage. Treat the direction as robust and the magnitudes as an upper bound.

### 2. Kokoro makes the lab easier than reality; most other voices make it harder

Across all conditions and models:
- The two Kokoro voices are significantly easier than human speech in 31 of 48 cells (30 after BH correction), and never harder.
- The five other voices are harder in 73 of 120 cells and easier in 7.

On clean audio, with 95 % CIs (points):

| Voice | Whisper gap | Parakeet gap | wav2vec2 gap (in-domain control) |
|---|---|---|---|
| Kokoro af_heart | -0.30 [-0.77, +0.14] | -0.28 [-0.60, +0.05] | -0.95 [-1.57, -0.31] |
| Kokoro am_michael | -0.37 [-0.79, +0.02] | -0.16 [-0.47, +0.14] | -0.60 [-1.27, +0.05] |
| Piper lessac | +0.16 [-0.36, +0.67] | +0.16 [-0.29, +0.59] | +1.25 [0.49, 1.98] |
| Piper ryan | +1.39 [0.49, 2.40] | +0.76 [0.05, 1.58] | +2.10 [1.15, 3.21] |
| macOS Samantha | +0.30 [-0.35, +0.94] | +0.42 [-0.12, +1.02] | +2.94 [2.23, 3.66] |
| macOS Daniel | +0.18 [-0.55, +0.98] | +0.69 [0.04, 1.46] | +3.68 [2.70, 4.70] |
| Meta MMS | +4.60 [3.45, 5.76] | +3.61 [2.72, 4.54] | +7.21 [6.08, 8.37] |

- **Strong models on clean audio:** only Piper ryan, MMS and (for Parakeet) Daniel are clearly harder; the rest are within noise of human speech.
- **Kokoro on clean audio:** its advantage is small and individually significant only for wav2vec2. It grows under noise (section 3).

**Spread.** For one model on the same 250 sentences, the worst voice has 3.9 to 4.5 times the WER of the best voice on clean audio:
- Whisper: 4.07 [3.20, 5.86]
- Parakeet: 3.90 [3.09, 5.28]

The worst voice is MMS in every clean cell, so the spread mostly measures one TTS model's failures (section 5).

**Speaking rate does not explain the clean-audio gaps.**
- Piper and macOS speak 25 to 40 % faster than the human readers: 3.6 to 3.9 words/s against 2.8.
- Re-synthesizing every voice at the human duration changed no clean gap significantly. All 21 voice x model CIs include 0; the largest change is +0.90 [0.00, 1.81] points, MMS on wav2vec2.
- The regression of the per-sentence gap on the rate difference is flat on clean audio:
  - Whisper: 0.00 [-1.68, +1.10] points per word/s
  - Parakeet: +0.50 [-0.43, +1.28]
- Under realistic noise the slope is positive for every model, so faster voices suffer more from noise:
  - Whisper: +2.1 [0.9, 3.3]
  - Parakeet: +2.2 [1.2, 3.3]
- The rate-matched voices were not run under noise, so that last effect is correlational.

### 3. Under noise, the lab's estimate of how much a channel costs depends on the voice

**Channel damage** is a voice's WER under a channel minus its clean WER. The error is synthetic damage minus human damage. 53 of 147 voice x channel x model cells differ significantly from reality after BH correction.

**Strong models.** Kokoro understates the cost of noise; the other voices are within noise of reality. A team testing on Kokoro would see about half the WER increase that the same noise causes on human speech.

| Model, condition | Real damage | Kokoro af_heart error | Kokoro am_michael error |
|---|---|---|---|
| Whisper, phone+noise10 | +2.13 | -1.18 [-2.11, -0.31] | -1.46 [-2.40, -0.57] |
| Whisper, realistic | +2.34 | -0.99 [-1.81, -0.21] | -1.32 [-2.20, -0.48] |
| Parakeet, phone+noise10 | +2.01 | -1.06 [-1.62, -0.50] | -1.50 [-2.17, -0.87] |

**The in-domain control** shows the same pattern, much larger and in both directions.
- **Under realistic** (real damage +22.7 points):
  - Kokoro af_heart understates the damage by 12.4 points [9.8, 15.1];
  - macOS Daniel overstates it by 11.5 points [7.8, 14.9];
  - the 7-voice mean still understates it by 4.0 [1.4, 6.8].
- **On the plain phone line:** real damage is +0.9 points; Kokoro af_heart shows +0.2 and the macOS voices +3.7 to +4.1.
- **Under muffling:** real damage is +4.3; Kokoro shows +1.6 to +2.0 and Daniel +16.0.

The likely reason is acoustic, not linguistic:
- Kokoro has a near-constant level, no room reverb and, before preparation, digital silence (-92 dB floor against about -56 dB of room tone in the human recordings).
- The macOS voices concentrate their energy where the phone filter cuts.

**Codecs.** For the strong models the codecs are nearly free on human speech:
- G.722 wideband equals clean for Parakeet (1.62 %).
- Opus at about 10 kb/s costs +0.37 points.
- 5 % packet loss costs nothing measurable with either concealment: Parakeet phone+loss5 1.83 % vs phone 1.80 %, and opus-nb+loss5 1.94 % vs opus-nb 1.99 %.

### 4. Emergency keyword recall: the voice matters a lot for the weak ear, and only marginally for the strong ones

Recall is scored on the 105 keyword instances in the 78 emergency and fall-help scripts that contain a keyword. The 6 instances in benign scripts, such as "help me with a question about my bill", are no longer counted. There is no human audio for this set, so every comparison is between voices or between channels.

**Pooled over the 7 voices** (sentence-cluster bootstrap CIs):

| Condition | Whisper | Parakeet | wav2vec2 (control) |
|---|---|---|---|
| clean | 97.4 % [96.1, 98.6] | 97.3 % [95.9, 98.6] | 85.8 % [82.1, 89.5] |
| realistic | 95.5 % [93.7, 97.3] | 94.6 % [92.6, 96.5] | 55.2 % [49.5, 60.9] |
| phone+noise5 | not run | 93.9 % [91.6, 96.1] | 31.8 % [25.5, 38.2] |

**Voice effects** (exact McNemar, Holm across the 42 voice pairs of each model):
- **wav2vec2:** 16 of 42 pairs differ. Under realistic, Kokoro af_heart reaches 79.0 % and Daniel 27.6 %: 56 discordant pairs against 2, Holm p = 1.5e-12.
- **Parakeet:** 2 of 42 pairs differ.
- **Whisper:** 0 of 42 pairs differ. Under realistic, Kokoro am_michael and Piper ryan get 105/105, and MMS and Daniel get 94/105 (89.5 % [82.0, 94.7]), but no pair survives correction.

**Zero-miss claims.**
- 36 of 168 voice x condition x model cells have zero misses. All are Kokoro or Piper voices; none is MMS or a macOS voice.
- With 105 instances, zero misses bounds the miss rate below 2.9 % (rule of three). The exact one-sided bound is 2.8 %, and the two-sided Clopper-Pearson bound is 3.5 %.
- Caveats:
  - the instances are not independent: 78 scripts, up to 3 keywords each, and "help" alone accounts for 13;
  - the bound applies only to this fixed set of written scripts;
  - the zero-miss cells were selected after the fact.
- For the strong models, the honest reading is "consistent with a few percent of misses on some voices", not "the voice creates the zero".

**Call-level detection** means any keyword appears in the transcript of a script that contains one (546 renderings).
- **Misses on clean / realistic:** Whisper 5 and 7, Parakeet 4 and 9, wav2vec2 54 and 202.
- **False alarms:** no strong model raised one on the 252 renderings of benign scripts without keywords; wav2vec2 raised 1 in each of 4 conditions.
- **Scripts with no keyword:** two fall-help scripts contain none ("I slipped in the shower and I think my leg is broken", "I tripped on the rug and hit my head"), so a keyword list misses them by design.

### 5. Two voice artifacts that a lab would blame on the ASR

**Meta MMS-TTS mis-renders words.**
- 80 reference words (1.8 % of all words, in 58 of 250 sentences) are wrong for all three models on MMS clean audio and right for all three on human audio.
- Removing these words halves the MMS gap: Whisper goes from +4.60 to +2.80 points, Parakeet from +3.61 to +1.79. MMS is a TTS fidelity failure, not a voice-bias measurement.
- The same filter removes most of Piper ryan's gap: Whisper +1.39 to +0.68, Parakeet +0.76 to +0.05.
- The filter is conservative (all three models must fail), so these counts are lower bounds on rendering errors.
- Nobody listened to the clips: that check is still to do.

**macOS Daniel says "hell" for "help".**
- On clean audio, Daniel's "help" is recognized in 7 of 13 scripts by Whisper, 5 by Parakeet and 3 by wav2vec2. Kokoro and Piper lessac get 38 to 39 of 39 across the three models.
- 30 of the 45 clean-audio "help" misses contain "hell": 21 on Daniel, 8 on Samantha.
- "Help" is the lowest-recall keyword for Whisper (0.89 [0.82, 0.96], clean) and Parakeet (0.86 [0.79, 0.92]).
- For wav2vec2, "overdose" (0.49) and "chest pain" (0.57) are lower. Under realistic, wav2vec2 recognizes "chest pain" in 1 of 35 renderings and "overdose" in 2.

### 6. Whisper hallucinates on noise alone

100 clips of pure noise (street, rain, crowd, home) were sent through the phone line, at the noise level of phone+noise10:
- **Whisper** returned text for 33 of them: "thank you" 26 times, and one runaway of 223 words ("ha ha ha ...") on a laughing clip.
- **Parakeet:** 2 of 100.
- **wav2vec2:** 21 of 100, short fragments such as "aca" and "baa".

On real speech, none of the models produced a runaway: no hypothesis was longer than 1.5 times its reference, except one clip each for Whisper and wav2vec2 on clean audio.

The v1 claim "Whisper did not hallucinate" rested on a test that could not detect hallucination, and is withdrawn. For an emergency line, text produced on silence or noise matters as much as recall.

### 7. Same-family bias: no evidence on clean audio; an ambiguous signal under noise

This is one TTS model and one ASR model from the same organization. MMS-TTS and wav2vec2-base share Meta, not a model family; the true same-family ASR would be mms-1b-all, which was not run. wav2vec2 also has 95 M parameters against 600 M or more for the others, and it is in-domain on the human side.

**Clean and phone audio:**
- In absolute points, wav2vec2 is penalized more by MMS than the cross-family models are: DiD +2.6 [1.6, 3.7] against Whisper and +3.6 [2.8, 4.5] against Parakeet.
- The other voices show the same direction, with medians of +0.9 and +1.2.
- In relative terms the MMS effect vanishes: the log-ratio DiD is -0.04 [-0.34, +0.24] against Whisper, while the other voices give +0.12.
- Conclusion: no family advantage, and no specific penalty.

**Noisy audio (phone+noise10, realistic):**
- wav2vec2 degrades relatively less on MMS than on the other voices: log-ratio DiD -0.86 [-1.08, -0.65] against Whisper on phone+noise10, against -0.28 for the other voices.
- This is the direction family bias would predict, but three confounds remain:
  - MMS audio is already degraded on clean audio;
  - wav2vec2's noisy WER is close to saturation (26 % on human speech);
  - the comparison carries the caveats above.
- It is not evidence of family bias.

### 8. PPI: a few human recordings plus the synthetic run give a valid interval, with a modest width gain

The aim is to estimate the human-speech WER over the 250 sentences from two inputs: all 250 synthetic results (the mean of the 7 voices per sentence) and n human-labeled sentences. The truth is a finite-population mean, so every method now uses the finite-population correction and a t interval. Each setting uses 1,000 random draws.

**Human-only intervals:**
- They under-cover at n = 10 even with t: 80 % for Whisper and 89 % for Parakeet on clean audio, because 8 to 11 % of draws contain only perfect transcripts.
- At n = 25 they reach 90 to 92 %.

**Classic PPI:**
- It restores coverage to 94 to 98 % at n = 10.
- It is narrower on every condition for Whisper and Parakeet: width ratio 0.73 to 0.94 at n = 10, and 0.77 to 0.91 at n = 25.
- The per-sentence correlation between human and synthetic results is only 0.53 to 0.69, which caps the gain.

**Largest gain:** noisy audio with the in-domain control. For wav2vec2 under phone+noise5 the correlation is 0.83, the width ratio is 0.56 from n = 10 to 50, and coverage is 95 to 97 %.

**PPI++:**
- The plug-in power-tuned version, no longer clipped at 1, is the narrowest but under-covers at n = 10 (76 to 90 %).
- The new cross-fit variant recovers most of the coverage (85 to 94 %) but loses the width gain at n = 10.
- Below about 25 labels, use classic PPI.

**In practice, for a scripted test set:** record about 25 of the scripts with real people and run every script through TTS. That gives a valid interval on the real-speech metric, 9 to 23 % narrower than the 25 human recordings alone. This relies on each human recording having a synthetic twin of the identical script, so it does not transfer to unscripted real calls.

## What did not work, was cut, or is still a limitation

- **Training-domain leakage (not fixable here).**
  - The human baseline is in-domain for wav2vec2 and probably for the other models.
  - The fix is an out-of-domain human reference, for example:
    - FLEURS en_us (CC BY 4.0), which was not authorized for download in this run;
    - or 30 to 60 caller scripts recorded by Amir over a real phone call, which would also give the caller set its first human baseline.
  - Until then, wav2vec2 results are an in-domain control, and gap magnitudes against human speech are upper bounds.
- **Whisper ran 4 of 10 conditions** (no g722-wb, opus-nb, muffle or stress conditions) because of GPU time. Its keyword and channel claims cover those 4 conditions only. To complete it, run `scripts/04_run_experiment.py --model whisper --conditions g722-wb,opus-nb,phone+muffle,phone+noise5,phone+loss5,opus-nb+loss5`. It is cached and takes about 40 minutes per condition.
- **Rate-matched voices were run on clean audio only**, so the noise x rate interaction is correlational. The macOS voices only reached 90 to 92 % of the human duration.
- **Nobody listened to the clips.** The MMS and Daniel verdicts rest on the shared-error filter and on the transcripts.
- **A noise-bed bug was found and fixed during the run.**
  - The problem: ESC-50 zero padding left some noise beds partly or fully silent; one sentence had no noise at all.
  - The fix: beds with more than 2 % silent frames are now rebuilt without the padding.
  - The noise conditions were re-run; only the changed clips were re-transcribed.
- **Simulation limits.**
  - The concealment is modelled on G.711 Appendix I, not a bit-exact implementation.
  - The IRS-like filter approximates P.48.
  - The ESC-50 categories approximate street and room noise. The crowd group consists of events (laughter, clapping, a crying baby) rather than steady babble.
- **TTS input text.**
  - LibriSpeech references have no punctuation, so TTS reads long run-on sentences.
  - Archaic names (Uncas, Hester, Dinah) produce pronunciation errors only on the TTS side; part of the rendering-error count comes from this.
- **Multiplicity.** The WER, damage and decision tables now carry BH or Holm columns. Extreme values quoted across many cells, such as the 0.37 to 3.72 ratio range, are descriptive.
- **Licence.** ESC-50 is CC BY-NC: mention it if results are shared.

## Fixes applied in v2 (summary)

**Statistics:**
- DiD test with Holm for lab-vs-reality decisions; a flip only counts against a real winner.
- `p_flip` renamed `share_sign_disagree` and dropped from the findings.
- Gap ratios reported only when the real gap is significant; the 7-voice mean reported separately; decision counts corrected.
- Speaker-cluster bootstrap for every LibriSpeech CI, plus a coverage simulation (`tables/bootstrap_coverage.csv`).
- Sentence-cluster CIs for pooled keyword recall.
- Empty McNemar cells skipped before Holm.
- BH columns on the WER, damage and decision tables, and a CI on the max/min voice ratio.
- p-values never printed as 0.
- Finite-population PPI with t intervals, unclipped PPI++ and a cross-fit variant.
- Log-ratio DiD for the same-family check.
- False alarms and call-level detection added.

**Method:**
- AGC to -26 dBov and a room-tone floor for every clip.
- SNR defined and calibrated after the phone line, and logged per clip.
- Wind removed from the noise mix; chained noise beds without loop seams.
- Causal IRS-like filter; Appendix I-style concealment on the 8 kHz stream.
- Opus (real decoder concealment) and G.722 conditions; stress conditions labelled.
- Speaking-rate column, regression and rate-matched rerun.
- Shared-error filter for TTS rendering errors.
- Spelling map in the normalizer.
- Keyword recall restricted to emergency and fall-help scripts.
- Noise-only hallucination test.
- Scope, title and caveats narrowed.
