# Follow-up: simulator-caused (TTS rendering) errors

Script: `scripts/07_tts_errors.py` (CPU only, reads `results/utterances.csv`, runs no ASR). All CIs are 95 % speaker-cluster bootstrap intervals (40 speakers, 10,000 replicates) unless stated otherwise. "Points" means WER percentage points. Nobody listened to any audio for this follow-up.

## Method

**Flag rules** (LibriSpeech subset, 250 sentences, 4,326 reference words, clean audio, per voice):
- **R3** (strict, the v2 rule): a reference word is wrong (substituted or deleted) for all 3 ASR models on the voice's clip, and right for all 3 on the human clip of the same sentence.
- **R2** (lenient): wrong for both strong models (Whisper, Parakeet) on the voice, right for both on human. wav2vec2 is ignored. R2 contains R3.
- A word flagged on clean audio is removed from that voice and from human in every channel condition, because every condition is made from the same TTS clip.
- **Mirror check for selection bias.** Removing only "synthetic wrong, human right" words favours the synthetic voice by construction. The mirror set is "wrong for all models of the rule on human, right on the voice" (reader deviations from the text, or reference errors). **R3sym / R2sym** remove both sets. I treat the symmetric versions as the fair cleaned gaps.

**Caller set (keywords).** There is no human audio, so other voices stand in for the human reference. A keyword instance (voice, script, keyword) is flagged when every model of the rule misses it on the voice's clean clip and at least 4 of the 6 other voices have it recognized by every model of the rule.

**Precision.** 30 flagged words were drawn from R2 (stratified by voice, seed 0; 23 of them are also R3). I labelled each from the reference text, the TTS input text and the three transcripts. Labels: `tts` means the evidence is consistent with a rendering error, `unclear` means it could be ASR or scoring, `not_tts` means clearly not a rendering error. The person who wrote the flag also did the labelling, from text only, without audio.

## Results

### 1. How many words are flagged

| Voice | R3 words | R2 words | Mirror (human-side) words, R3 / R2 |
|---|---|---|---|
| Kokoro af_heart | 8 | 10 | 11 / 13 |
| Kokoro am_michael | 7 | 11 | 11 / 14 |
| Piper lessac | 12 | 14 | 8 / 10 |
| Piper ryan | 31 | 38 | 7 / 10 |
| Meta MMS | 80 | 111 | 8 / 11 |
| macOS Samantha | 16 | 32 | 10 / 15 |
| macOS Daniel | 14 | 25 | 8 / 11 |

Only MMS (80 vs 8) and Piper ryan (31 vs 7) clearly exceed the mirror baseline. For the other five voices, the human recordings have about as many "every model wrong on this side only" words as the voice does. One-sided cleaning on those voices therefore mostly moves the gap in the synthetic voice's favour.

### 2. Share of each voice's clean WER on flagged words

Percent of all errors (insertions included) that fall on flagged words, R3 then R2:

| Voice | Whisper R3 | Whisper R2 | Parakeet R3 | Parakeet R2 |
|---|---|---|---|---|
| Kokoro af_heart | 11.0 [2.8, 21.1] | 13.7 [4.0, 26.1] | 13.8 [3.7, 24.5] | 17.2 [5.3, 30.2] |
| Kokoro am_michael | 10.0 [3.3, 17.5] | 15.7 [8.5, 24.2] | 11.1 [3.6, 19.4] | 17.5 [9.5, 25.8] |
| Piper lessac | 12.9 [6.4, 21.1] | 15.1 [7.2, 24.4] | 15.6 [7.8, 24.5] | 18.2 [8.9, 28.4] |
| Piper ryan | 21.2 [9.7, 34.2] | 26.0 [14.5, 38.4] | 30.1 [14.8, 45.9] | 36.9 [22.1, 51.7] |
| Meta MMS | 28.1 [22.3, 34.3] | 39.0 [32.3, 46.1] | 35.4 [28.7, 42.4] | 49.1 [41.9, 56.9] |
| macOS Samantha | 16.2 [6.9, 26.0] | 32.3 [20.5, 43.9] | 18.2 [8.0, 29.0] | 36.4 [23.6, 48.0] |
| macOS Daniel | 14.9 [8.1, 22.6] | 26.6 [16.4, 36.6] | 14.0 [7.8, 19.8] | 25.0 [15.1, 34.1] |

**Share of the gap**, computed only where the raw gap is positive and its CI excludes 0:
- **MMS, Whisper:** 40 % [32, 50] (R3), 56 % [46, 68] (R2).
- **MMS, Parakeet:** 51 % [42, 63] (R3), 71 % [61, 84] (R2).
- **MMS, wav2vec2:** 26 % [20, 32].
- **Piper ryan, Whisper:** 52 % [28, 96] (R3).
- **Piper ryan, Parakeet:** 94 % [51, 300] (R3). The ratio is unstable because the gap is small.
- **macOS Daniel, Parakeet:** 47 % [22, 183] (R3).

### 3. Precision of the flag (hand check, 30 items, text only)

- **Labels:** 23 `tts`, 7 `unclear`, 0 `not_tts`.
- **Strict precision** (unclear counted as wrong): 77 % [58, 90] (Clopper-Pearson).
- **R3 subset:** 20 of 23, 87 % [66, 97]. **R2-only items:** 3 of 7.
- **Causes among the 23 `tts` items:**
  - 9 garbled audio (mostly MMS);
  - 4 words read wrongly;
  - 2 names read wrongly;
  - 1 archaic word ("'tis" heard as "teas");
  - 1 coarticulation;
  - 5 run-on readings: the TTS input has no punctuation, so the voice reads without the pauses the human reader made ("playthings. See?" becomes "plaything sea");
  - 1 abbreviation: the TTS input says "hon" where the reference is scored as "honorable", and all models hear "hun".
- At least 6 of the 23 (the 5 run-on readings and the abbreviation) come from this pipeline's TTS input text, not from the TTS acoustic model, and could be fixed upstream.
- **Unclear items:** rare names (Uncas, Chelford), reductions that also occur in natural speech ("and" as "'n", dropped /h/), one possible accent effect (Daniel's "old" heard as "all"), and one possible language-model bias ("ruth" as "root" for the strong models, while wav2vec2 got it right).
- **Limits:** one labeller, who also designed the flag; no audio; 30 items. The interval is wide, and the true precision could be well below 77 % if the unclear items are not rendering errors.

### 4. Gaps on clean audio after cleaning (synthetic minus human, points)

| Voice, model | Raw (v2) | R3 one-sided | R3sym | R2sym |
|---|---|---|---|---|
| Kokoro af_heart, Whisper | -0.30 [-0.77, +0.14] | -0.49 [-1.00, +0.00] | -0.23 [-0.75, +0.27] | -0.23 [-0.75, +0.27] |
| Kokoro af_heart, Parakeet | -0.28 [-0.60, +0.05] | -0.46 [-0.80, -0.14] | -0.21 [-0.52, +0.07] | -0.21 [-0.51, +0.07] |
| Kokoro am_michael, Whisper | -0.37 [-0.79, +0.02] | -0.53 [-0.96, -0.14] | -0.28 [-0.70, +0.11] | -0.30 [-0.72, +0.07] |
| Kokoro am_michael, Parakeet | -0.16 [-0.47, +0.14] | -0.32 [-0.65, -0.02] | -0.07 [-0.40, +0.26] | -0.09 [-0.38, +0.17] |
| Piper lessac, Whisper | +0.16 [-0.36, +0.67] | -0.12 [-0.60, +0.37] | +0.07 [-0.40, +0.54] | +0.07 [-0.39, +0.53] |
| Piper lessac, Parakeet | +0.16 [-0.29, +0.59] | -0.12 [-0.54, +0.28] | +0.07 [-0.33, +0.44] | +0.07 [-0.33, +0.44] |
| Piper ryan, Whisper | +1.39 [0.49, 2.40] | +0.68 [0.02, 1.34] | +0.84 [0.20, 1.49] | +0.75 [0.16, 1.35] |
| Piper ryan, Parakeet | +0.76 [0.05, 1.58] | +0.05 [-0.40, +0.50] | +0.21 [-0.23, +0.65] | +0.12 [-0.31, +0.52] |
| Meta MMS, Whisper | +4.60 [3.45, 5.76] | +2.80 [1.87, 3.77] | +3.00 [2.10, 3.95] | +2.35 [1.55, 3.21] |
| Meta MMS, Parakeet | +3.61 [2.72, 4.54] | +1.79 [1.12, 2.47] | +1.98 [1.31, 2.68] | +1.33 [0.78, 1.91] |
| macOS Samantha, Whisper | +0.30 [-0.35, +0.94] | -0.07 [-0.65, +0.50] | +0.16 [-0.39, +0.70] | -0.09 [-0.60, +0.36] |
| macOS Samantha, Parakeet | +0.42 [-0.12, +1.02] | +0.05 [-0.40, +0.56] | +0.28 [-0.12, +0.74] | +0.02 [-0.24, +0.29] |
| macOS Daniel, Whisper | +0.18 [-0.55, +0.98] | -0.14 [-0.80, +0.56] | +0.05 [-0.60, +0.72] | -0.14 [-0.70, +0.43] |
| macOS Daniel, Parakeet | +0.69 [0.04, 1.46] | +0.37 [-0.19, +1.00] | +0.56 [0.04, 1.15] | +0.37 [-0.05, +0.83] |

wav2vec2 and all conditions are in `gap_cleaned.csv`. Reading the table:
- **One-sided cleaning makes Kokoro look significantly easier than human speech.** After the symmetric cleaning, no Kokoro clean cell is significant for the strong models. That apparent advantage is a selection artifact, not a finding.
- **MMS stays clearly harder** under every rule: +2.35 to +3.00 points for Whisper, +1.33 to +1.98 for Parakeet. Rendering errors explain roughly a third to a half of its gap, not all of it.
- **Piper ryan, Whisper:** the gap stays significant after symmetric cleaning (+0.84 [0.20, 1.49]). For Parakeet it does not.
- **Over all 7 voices x 24 cells**, with BH within each rule:
  - Kokoro is easier than human in 30 of 48 cells raw, 39 after R3, 44 after R2, and 28 or 27 after symmetric cleaning;
  - the other voices are harder in 67 of 120 cells raw, 52 after R3, and 61 after R3sym.
  - This BH family differs from the v2 table, so the raw counts differ slightly from FINDINGS.

### 5. v2 headline results after cleaning

**Whisper vs Parakeet, 28 voice-level decisions.**
- The DiD is nominally significant in 5 of 28 decisions under every rule, and Holm-significant in 0. The smallest Holm p goes from 0.16 (raw) to 0.11 (R3), 0.11 (R3sym) and 0.06 (R2).
- On clean audio the filter cannot change this comparison: a flagged word is an error for both models, so it cancels in the Whisper minus Parakeet difference. The MMS DiD on clean is +0.99 [0.24, 1.84] raw and +1.01 [0.25, 1.87] after R3.
- On noisy audio the MMS DiD does not shrink: phone+noise10 goes from +1.59 [0.52, 2.62] to +1.67 [0.62, 2.70] (R3) and +1.71 [0.67, 2.73] (R2).
- So the MMS distortion of this comparison comes from errors that only one model makes. The v2 explanation ("its rendering errors hurt Whisper more") is not tested by a shared-error filter, which by design removes only errors common to both models.
- The Daniel clean flip is unchanged: -0.14 [-0.73, +0.50] after R3.

**The 7-voice mean** (union of all voices' flagged words removed). It still declares Parakeet significantly better where real speech shows no winner:
- phone: +0.40 [0.14, 0.66] (R3);
- phone+noise10: +0.78 [0.50, 1.07];
- realistic: +0.73 [0.37, 1.09].

The "averaging manufactures certainty" finding survives cleaning.

**Large-gap pairs (vs wav2vec2).**
- The lab/real gap ratio ranges from 0.378 to 3.72 raw, 0.377 to 3.80 after R3, 0.377 to 3.77 after R3sym, and 0.379 to 4.60 after R2. Under R2 the maximum moves to MMS, Whisper vs wav2vec2 on clean audio.
- Holm-significant DiDs: Whisper 18/28 raw and 18 after R3; Parakeet 42/70 raw and 43 after R3.
- Realistic, Whisper vs wav2vec2: real gap 21.7 points. The lab gap is 9.6 with Kokoro af_heart and 36.0 with Daniel after R3, against 9.6 and 35.8 raw.

**Spread of voices on clean audio** (max/min WER):
- Whisper: 4.07 [3.20, 5.86] raw, 3.31 [2.64, 4.83] after R3, 3.02 [2.36, 4.51] after R2.
- Parakeet: 3.90 [3.09, 5.28] raw, 2.97 [2.41, 3.90] after R3, 2.45 [2.00, 3.25] after R2.
- MMS remains the worst voice in every case.

**Kokoro understating channel damage is robust to cleaning.** The error vs human damage is:
- Whisper, phone+noise10: af_heart -1.18 [-2.11, -0.31] raw, -1.11 [-2.06, -0.24] after R3; am_michael -1.46 raw, -1.44 [-2.38, -0.56] after R3.
- Parakeet, phone+noise10: af_heart -1.06 [-1.62, -0.50] raw, -1.00 [-1.57, -0.41] after R3; am_michael -1.50 raw, -1.48 [-2.14, -0.85] after R3.
- Damage cells significant after BH: 53 of 147 raw, 56 after R3, 58 after R2.

### 6. Keyword recall (caller set, 105 keyword instances per voice)

**Flagged instances.**
- R3 flags 11:
  - all 6 of Daniel's "help" instances that every model transcribed as "hell";
  - 5 on MMS: "help" heard as kelp / elp / tulp three times, "stabbed" as "stouted", "can't get up" as "can't give up".
- R2 adds 2 on MMS: "can't get up", and "fell" heard as feel / felt.

**macOS Daniel, clean recall:**
- Whisper: 94.3 % [88.0, 97.9] becomes 100 % [96.3, 100] (99/99).
- Parakeet: 92.4 % [85.5, 96.7] becomes 98.0 % [92.9, 99.8].
- **"help":**
  - Whisper: 7/13 becomes 7/7.
  - Parakeet: 5/13 becomes 5/7.
  - wav2vec2: 3/13 becomes 3/7.

**MMS, Whisper, clean recall:** 89.5 % [82.0, 94.7] becomes 94.0 % [87.4, 97.8] (R3) and 95.9 % [89.9, 98.9] (R2).

**Pooled over voices** (sentence-cluster CIs):
- Whisper, clean: 97.4 % [96.1, 98.6] becomes 98.9 % [98.2, 99.6].
- Parakeet, clean: 97.3 % [95.9, 98.6] becomes 98.8 % [97.9, 99.6].
- Under realistic: Whisper goes from 95.5 % to 96.5 % [95.0, 97.9], and Parakeet from 94.6 % to 96.0 % [94.5, 97.4].

**Voice effects** (exact McNemar between voice pairs):
- With Holm over the 126 voice-pair tests (the family that `05_analyze.py` actually uses), the counts reproduce v2 raw: Whisper 0, Parakeet 2, wav2vec2 16. After R3 they are 0, 0 and 15.
- With Holm per model over 42 tests (what FINDINGS section 4 says it does), the raw counts are Whisper 7, Parakeet 6 and wav2vec2 19. After R3 they are 0, 0 and 18.
- So for the strong models, every significant voice effect on keyword recall is carried by the flagged instances.
- **Side finding:** the FINDINGS wording ("Holm across the 42 voice pairs of each model") does not match the code (one Holm family across all 3 models). Under the stated per-model family, the strong models would show 7 and 6 significant voice pairs, not 0 and 2.

**Caveats:**
- The removed instances are, by construction, ones every model missed, so the cleaned recall is biased upward. It is not the recall a human saying "help" would get, which is unmeasured: the caller set has no human audio.
- Model-specific misses are not caught: Samantha's "help", missed by Parakeet twice and by wav2vec2 eight times while Whisper recognized it, stays in.
- The cross-voice reference assumes the other voices render the word correctly.

## Limits (summary)

- **Lower bound on recall.** The flag needs every model of the rule to fail, so rendering errors that only one model trips on are missed, and the counts are lower bounds on rendering errors.
- **Selection bias.** The one-sided cleaning is biased in the synthetic voice's favour; use the symmetric (R3sym / R2sym) gaps.
- **In-domain models.** The human-correct condition leans on models trained on LibriSpeech (wav2vec2, probably Parakeet), so "human right" is easier to satisfy than it would be on out-of-domain human speech.
- **Precision check.** The precision rests on 30 text-only judgements by one labeller; nobody listened to audio.
- **Flags from clean audio only.** Flags are derived from clean audio and applied to every condition; under noise, other words may become rendering-sensitive.
- **Rate-matched voices.** They are flagged too (`flag_summary.csv`, `gap_cleaned.csv`), but they are not discussed here.

## Files

- `flag_summary.csv`: flagged words per rule x voice x model; share of WER and of the gap, with CIs.
- `gap_cleaned.csv`: gaps for every voice x condition x model under raw, R3, R2, R3sym and R2sym, with BH.
- `decisions_cleaned.csv`: model-pair DiD, flips and lab/real ratios under each rule, with Holm.
- `voice_spread_cleaned.csv`, `channel_damage_cleaned.csv`.
- `keyword_flags.csv`, `keyword_recall_cleaned.csv`, `keyword_mcnemar_cleaned.csv`.
- `hand_check_sample.csv` (generated by the script), `hand_check_labels.csv` (hand labels), `hand_check.csv` (merged).
- `summary.json`.
- `fig1_flagged_share.png`, `fig2_gap_cleaned.png`, `fig3_keyword_recall_cleaned.png`.
