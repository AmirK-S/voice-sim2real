# Follow-up: FLEURS as an out-of-domain human baseline

**Question.** The v2 findings use LibriSpeech as "human speech". LibriSpeech is wav2vec2's training domain and likely part of Whisper's and Parakeet's. Do the v2 conclusions hold when the human side is out-of-domain read speech?

**Short answer.** The direction of the main lesson holds and gets stronger: the lab's picture depends on the voice and does not match real speech. Two v2 details do not hold. (1) On FLEURS, every voice except Meta MMS is easier than real speech, including the voices that were harder than LibriSpeech readers; this survives reading the FLEURS sentences with LibriSpeech-style text. (2) The "averaging declares a winner where the human CI includes 0" pattern does not repeat: on FLEURS the realistic channel shows Parakeet better at sentence level, and the voices understate it. That human result is not robust to plausible reader clustering (it needs a CI widening below 1.42x to hold).

**Revised after review (13/10/2026).** New analyses, no change to the transcripts of the first run:
- *Reader clustering* (`robustness/`): break-even widening for every decision, gap and damage cell; proxy speaker clusters from a voice and room fingerprint (validated on LibriSpeech); runs of consecutive sentence ids. See "Robustness to reader clustering".
- *TTS input text* (`textprep/`): the 7 voices re-read the 250 sentences with LibriSpeech-style text (lowercase, no punctuation, numbers spelled out), Whisper and Parakeet on clean audio (3,500 new decodes). See "TTS input text".
- *Same bootstrap unit*: the LibriSpeech counts are now also given with a sentence bootstrap.
- *Own noise of the recordings* (`robustness/`): channel damage recomputed by tercile of each recording's own SNR.
- Wording: "Parakeet wins on realistic" is now "Parakeet better at sentence level, not robust to plausible reader clustering"; the contrasts "harder on LibriSpeech, easier on FLEURS" and the MMS spread now carry the text-preparation result.

FLEURS is read speech (Wikipedia-style sentences read by volunteers with varied accents, microphones and rooms). It is not phone-call speech and not emergency speech.

## Setup

- **Data:** FLEURS en_us test split (google/fleurs, CC BY 4.0). Only `test.tsv` and `audio/test.tar.gz` were downloaded; only the 250 selected wavs were extracted.
- **Selection** (`sim2real.fleurs.select_fleurs`): one recording per FLoRes sentence id, raw duration 3 to 12 s (same window as the LibriSpeech subset), random recording per sentence (seed 0), then a round robin over gender. 298 sentences qualified; 250 were taken: 177 female and 73 male recordings, 5,050 reference words, 35.1 min.
- **No speaker ids.** FLEURS does not release speaker ids, so the speaker-cluster bootstrap of the main run is not possible. Each sentence is its own cluster (the provided grouping is the sentence id, with one recording kept per id). On LibriSpeech, the speaker-cluster CI of a gap was a median 1.06 times as wide as the sentence CI (10th to 90th percentile 0.90 to 1.28). FLEURS CIs below are therefore probably somewhat too narrow; results that would not survive a 1.28x widening are flagged.
- **TTS:** the 7 voices of the main run read the FLEURS raw transcription (case and punctuation kept), through `tts.synthesize`. No failures.
- **Pipeline:** identical clip preparation (`channel.prepare`) and channel code, seed from the sentence id so voices stay paired.
- **Grid:** Parakeet and wav2vec2 on clean, phone, phone+noise10, realistic; Whisper on clean and realistic. 20,000 scored transcriptions.
- **Scoring fix specific to FLEURS:** Whisper's English normalizer deletes any text between parentheses. 25 of the 250 FLEURS sentences contain parentheses that are part of the read sentence, so the first scoring counted those words as insertions (Parakeet clean human WER 5.48 % instead of 3.74 %). Parentheses and brackets are now replaced by spaces in references and hypotheses before normalization (`fleurs.scoring_ref`). The main run's LibriSpeech references have no parentheses, so it is unaffected.
- **Statistics:** 95 % percentile bootstrap, 10,000 replicates over sentences. DiD as in v2, Holm within each model pair over the voice-level decisions; the 7-voice mean is its own Holm family. LibriSpeech numbers are read from `results/tables/` (speaker-cluster bootstrap). "Points" means WER percentage points.

## Results

### 1. Human WER roughly doubles for the strong models, and wav2vec2 collapses

| Model, condition | FLEURS human WER | LibriSpeech human WER |
|---|---|---|
| Whisper, clean | 3.88 % [3.20, 4.59] | 1.99 % [1.42, 2.59] |
| Whisper, realistic | 8.38 % [7.27, 9.51] | 4.32 % [3.45, 5.29] |
| Parakeet, clean | 3.74 % [3.08, 4.44] | 1.62 % [1.17, 2.11] |
| Parakeet, phone | 4.08 % [3.38, 4.80] | 1.80 % [1.31, 2.34] |
| Parakeet, phone+noise10 | 7.33 % [6.27, 8.48] | 3.63 % [2.92, 4.39] |
| Parakeet, realistic | 7.15 % [6.20, 8.12] | 3.72 % [2.99, 4.56] |
| wav2vec2, clean | 16.81 % [15.16, 18.53] | 3.28 % [2.66, 3.90] |
| wav2vec2, realistic | 65.01 % [62.37, 67.68] | 26.01 % [23.20, 29.07] |

This confirms the v2 caveat that the LibriSpeech baseline is favourable, most of all for wav2vec2. Part of the FLEURS human WER is not acoustic: reference text with names, numbers and readers who deviate from the text (limits below).

### 2. Synthetic vs human: every voice except MMS is easier than FLEURS humans

- All 70 voice x model x condition cells differ from human speech (all BH-adjusted p < 0.05). 63 are easier, 7 harder; the 7 harder cells are all MMS (MMS is easier than humans only for wav2vec2 on the three phone conditions, where human speech collapses).
- All 60 non-MMS cells are easier. On LibriSpeech, the same 70 cells split into 16 easier, 26 harder and 28 not significant with the speaker-cluster CI, and 18 easier, 27 harder and 25 not significant with a sentence bootstrap (the unit used on FLEURS), so the contrast is not an artifact of the bootstrap unit.
- **Clean, Parakeet:** Kokoro af_heart -1.41 [-2.00, -0.81], macOS Daniel -1.94 [-2.49, -1.43], Piper ryan -1.41 [-1.97, -0.85], MMS +5.48 [4.49, 6.52]. On LibriSpeech, Daniel and Piper ryan were harder than humans (+0.69 and +0.76). The LibriSpeech voices read LibriSpeech-style text and the FLEURS voices read raw text, so part of such a contrast could be the input text; with LibriSpeech-style text on FLEURS, Daniel is still -1.88 [-2.44, -1.35] and Piper ryan -0.89 [-1.47, -0.34] (section "TTS input text").
- **Realistic, Whisper:** Kokoro af_heart -5.72 [-6.84, -4.65], the 7-voice mean -3.54 [-4.57, -2.56]. On LibriSpeech the mean was +0.97 [0.36, 1.57], harder than humans.
- The two rankings still agree in part: Spearman 0.46 across the 70 cells (p = 7e-5). Kokoro stays the easiest voice family and MMS the hardest. 96 % of the FLEURS gaps sit below their LibriSpeech counterpart (median shift -4.5 points over all cells, driven by wav2vec2; for the strong models the median shift is -2.6 points, range -6.5 to +1.9, the only upward shifts being MMS).
- **Spread:** worst voice / best voice is 6.11 [5.19, 8.51] for Whisper clean and 5.12 [4.18, 6.86] for Parakeet clean (LibriSpeech 4.07 and 3.90). MMS is still the worst voice in every strong-model cell. Part of the larger spread is a text-input artifact: the MMS tokenizer lowercases everything and knows only the digits 0 to 6, so "In 1990" becomes "in 10". With LibriSpeech-style text (numbers in words) the spread is 5.23 [4.23, 6.99] for Whisper and 4.40 [3.62, 5.77] for Parakeet, and the CIs overlap the LibriSpeech values.

**Robustness checks.**
- *Reference mismatches.* 26 words in 23 sentences are wrong for all 3 models on human clean audio but right in at least 19 of 21 synthetic decodes (mostly readers saying "street" for "streets", "edge" for "edges", or skipping a word). Removing them moves every gap up by 0.14 to 0.55 points; no conclusion changes (Parakeet clean, Kokoro af_heart: -0.90 instead of -1.41).
- *Names and digits.* TTS reads rare names from their spelling, which could make them easy to transcribe back. On the 95 sentences with no digits and no mid-sentence capitals, Kokoro af_heart is still easier than humans: Whisper clean -1.44 [-2.38, -0.55], Parakeet realistic -4.25 [-6.27, -2.56]. Names and digits do not explain the gap.
- *Reader clustering.* The smallest break-even widening of the 60 non-MMS gaps is 1.94x; 59 of 60 stay significant at 2x (`robustness/break_even_widening.csv`).

### 3. Channel damage: on FLEURS, every voice understates the cost of noise

Damage is a voice's WER under a channel minus its clean WER; the error is synthetic damage minus human damage.
- 43 of 49 voice x channel x model cells differ from reality after BH; all 43 understate the damage. For the strong models, 22 of 28. This count is less robust to reader clustering than the gaps: 42 of 49 stay significant at a 1.5x CI widening and 36 at 2x.
- **Realistic, Parakeet:** real damage +3.41 points; Kokoro af_heart shows +0.36 (error -3.05 [-3.98, -2.17]), the 7-voice mean +1.29 (error -2.12 [-2.99, -1.31]).
- **Realistic, Whisper:** real damage +4.50; Kokoro af_heart +0.53 (error -3.96 [-5.04, -2.96]); macOS Samantha, the closest voice, +2.57 (error -1.92 [-2.97, -0.93]).
- v2 found that only Kokoro understated noise damage on LibriSpeech (for example Parakeet phone+noise10 error -1.06) and the other voices were within noise. On FLEURS the Kokoro error is 3 to 6 times larger and the other voices understate too.
- **Caveat and check:** FLEURS recordings already carry room noise and reverb. The noise is added at 10 dB relative to the clip's active level, and for a noisy recording that level includes its own background, so the effective speech-to-noise ratio of FLEURS humans is probably below 10 dB. To test this, each human clip got a simple own-SNR proxy (95th minus 10th percentile of its 30 ms phone-band frame energies; `robustness/fleurs_own_snr.csv`) and the damage analysis was redone per tercile (about 83 sentences each, `robustness/channel_damage_by_own_snr.csv`). The real damage barely depends on it (Parakeet realistic: +3.45 points in the cleanest tercile, 42 to 60 dB, against +3.34 in the noisiest, 23 to 39 dB), and the understatement remains in the cleanest tercile: 35 of 49 cells understate after BH (42 in the noisiest), none overstate; Parakeet realistic, Kokoro af_heart -2.75 [-4.22, -1.33], 7-voice mean -2.11 [-3.52, -0.76]. The proxy is crude (it mixes background level with dynamic range), so this reduces but does not remove the concern. A phone line alone (Parakeet phone) costs humans only +0.34 points; Kokoro af_heart error -0.55 [-0.90, -0.21].

### 4. Whisper vs Parakeet: no confident flip, but the lab misses a real winner

**Reality on FLEURS** (Whisper WER minus Parakeet WER; positive = Parakeet better):
- clean: +0.14 [-0.36, +0.63], no winner (LibriSpeech: Parakeet by +0.37 [0.12, 0.67]);
- realistic: +1.23 [0.36, 2.10], **Parakeet better at sentence level** (LibriSpeech: +0.60 [-0.04, +1.29], CI includes 0). This is not robust to plausible reader clustering: the CI includes 0 once widened by 1.42x or more, and FLEURS has far fewer readers than recordings, so the 1.28x LibriSpeech factor (40 speakers, about 6 sentences each) is not a bound here. See "Robustness to reader clustering".

**Single voices (14 decisions: 7 voices x 2 conditions).**
- Realistic: 6 of 7 voices see no winner. Kokoro af_heart (-0.04 [-0.51, +0.44]) and Piper lessac (-0.02 [-0.66, +0.60]) point the wrong way; by the v2 definition these are flips, not confident flips. Only MMS finds the real winner (+1.48 [0.57, 2.39]).
- Clean: Kokoro am_michael (-0.36 [-0.71, -0.04]) and Piper lessac (-0.50 [-0.99, -0.04]) declare Whisper significantly better where FLEURS humans show no winner; MMS declares Parakeet better (+0.93 [0.18, 1.67]).
- The DiD is nominally significant in 5 of 14 decisions (realistic: Kokoro af_heart -1.27 [-2.30, -0.26], Piper lessac -1.25 [-2.35, -0.20], Piper ryan -1.09 [-2.10, -0.12], macOS Daniel -1.07 [-2.09, -0.10]; clean: Piper lessac -0.63 [-1.26, -0.02]). **None survives Holm**; the smallest adjusted p is 0.14. Same verdict as v2 (0 of 28). On realistic, 6 of the 7 voice-level DiDs are negative (the voices understate Parakeet's advantage): the direction is consistent even though no single one survives Holm, and the 7-voice-mean DiD below summarizes it.

**7-voice mean.**
- Clean: -0.05 [-0.31, +0.20], no winner, which matches reality.
- Realistic: +0.34 [0.04, 0.64], the same direction as the sentence-level human result at 0.28 times its size. DiD -0.88 [-1.79, -0.01], nominal p = 0.047, Holm p = 0.094 over the 2 mean decisions. Its break-even widening is 1.01x: any clustering at all makes its CI include 0.
- So the v2 pattern "averaging makes the lab confident where the human CI includes 0" does not repeat. On FLEURS, averaging gives the same sign as the human result with a narrow CI, and the voices understate the human gap (consistently in direction, not significantly after Holm).

**Large-gap pairs (vs wav2vec2, 42 voice decisions).**
- The winner never flips (0 of 42), as in v2.
- The lab gap is always smaller than the real gap: lab / real ratio 0.26 to 0.92 (LibriSpeech 0.37 to 3.72, both directions). Realistic, Whisper vs wav2vec2: real gap 56.6 points; Kokoro af_heart says 17.9 and macOS Daniel 47.1.
- The DiD is Holm-significant in 41 of 42 (Parakeet vs wav2vec2 27 of 28, Whisper vs wav2vec2 14 of 14). This mostly measures wav2vec2's collapse on out-of-domain human speech, which no TTS voice reproduces.

### 5. Robustness to reader clustering (review fix)

FLEURS has no speaker ids, and the selection (177 female, 73 male recordings, from a split with far fewer readers than recordings) suggests few readers with many recordings each. Three checks (`robustness/`):

**Break-even widening** (`break_even_widening.csv`): the factor by which a sentence-bootstrap CI must widen, around its point estimate, before it includes 0.
- Human Whisper minus Parakeet, realistic: **1.42x**. 7-voice-mean DiD, realistic: **1.01x**.
- Non-MMS gaps: smallest 1.94x; 59 of 60 still significant at 2x.
- Channel damage: 43 of 49 cells significant at 1x, 42 at 1.5x, 36 at 2x.

**Proxy speaker clusters** (`headline_by_grouping.csv`). Each human clip gets a fingerprint: long-term average log-mel spectrum of its quietest 15 % of frames (microphone and room), mean and SD of 19 MFCCs over its louder frames, and median F0; Ward clustering within gender, k = 10, 20 or 40 clusters in total. Validated on the LibriSpeech human clips, where speakers are known: at k = 40 the clusters match the 40 true speakers with adjusted Rand index 0.74, and give CIs with the same median width as true-speaker CIs (ratio 1.00, 10th to 90th percentile 0.92 to 1.08; `proxy_cluster_validation_librispeech.csv`). On LibriSpeech, though, true-speaker CIs are barely wider than sentence CIs (ratio 1.03), so this validation shows the proxy does not distort CIs; it cannot show that it captures strong reader effects.

| Grouping (clusters) | Human W - P, realistic | 7-voice-mean DiD, realistic | non-MMS gaps easier | damage cells understating (CI) |
|---|---|---|---|---|
| sentence, as reported (250) | +1.23 [0.36, 2.10] | -0.88 [-1.79, -0.01] | 60 of 60 | 43 of 49 |
| consecutive sentence ids (76) | +1.23 [0.48, 2.00] | -0.88 [-1.73, -0.03] | 60 of 60 | 44 of 49 |
| proxy readers, k = 10 | +1.23 [0.50, 1.93] | -0.88 [-1.60, -0.02] | 60 of 60 | 44 of 49 |
| proxy readers, k = 20 | +1.23 [0.45, 1.94] | -0.88 [-1.59, -0.07] | 60 of 60 | 43 of 49 |
| proxy readers, k = 40 | +1.23 [0.57, 1.97] | -0.88 [-1.64, -0.17] | 60 of 60 | 43 of 49 |

- None of these groupings widens the CIs; the proxy-cluster CIs are slightly narrower than the sentence CIs. Either readers share little error variance on these quantities, or the proxy does not recover FLEURS readers (it cannot be checked without speaker ids).
- **Reading:** the "every voice except MMS is easier" result is robust (it survives 2x). The human Parakeet advantage on realistic and the 7-voice-mean DiD are not shown to be robust: they hold under every grouping tried, but a modest true reader effect (1.42x and 1.01x) would remove them. Consecutive sentence ids may come from the same source article, which is why that grouping is also shown.

### 6. TTS input text (review fix)

The LibriSpeech voices read LibriSpeech-style text (sentence case, no inner punctuation, final period, numbers already in words). The FLEURS voices read the raw FLEURS text (case, commas, quotes, digits, abbreviations). The 250 FLEURS sentences were re-synthesized with LibriSpeech-style text built from the FLEURS normalized transcription with numbers spelled out (`scripts/08b_fleurs_textprep.py`, examples in `textprep/tts_text_variants.csv`), and Whisper and Parakeet transcribed the clean audio (`textprep/gaps_by_text_prep.csv`, `textprep/voice_spread_by_text_prep.csv`, `fig4_tts_text_prep.png`). 38 of the 250 sentences contain digits.

**Gaps on clean audio, synthetic minus FLEURS human (points), raw text then LibriSpeech-style text, 250 sentences:**

| Voice | Whisper raw | Whisper LS-style | Parakeet raw | Parakeet LS-style |
|---|---|---|---|---|
| Kokoro af_heart | -1.76 | -1.66 [-2.27, -1.06] | -1.41 | -1.29 [-1.89, -0.69] |
| Kokoro am_michael | -2.20 | -1.80 [-2.38, -1.21] | -1.70 | -1.58 [-2.16, -1.01] |
| Piper lessac | -1.68 | -0.79 [-1.38, -0.20] | -1.05 | -0.55 [-1.14, +0.02] |
| Piper ryan | -1.39 | -1.33 [-1.94, -0.75] | -1.41 | -0.89 [-1.47, -0.34] |
| Meta MMS | +6.28 | +5.23 [4.11, 6.34] | +5.49 | +4.46 [3.46, 5.43] |
| macOS Samantha | -2.18 | -2.04 [-2.63, -1.45] | -1.78 | -1.78 [-2.30, -1.28] |
| macOS Daniel | -2.22 | -2.14 [-2.71, -1.58] | -1.94 | -1.88 [-2.44, -1.35] |
| mean of 7 | -0.74 | -0.65 [-1.18, -0.14] | -0.54 | -0.50 [-1.01, -0.02] |

- **The input text moves gaps by up to about 1 point, and it does not explain the main contrast.** With LibriSpeech-style text, 11 of the 12 non-MMS cells are still significantly easier than FLEURS humans (Piper lessac on Parakeet now -0.55 [-1.14, +0.02]).
- **MMS:** its gap shrinks by 1.05 [0.34, 1.77] points (Whisper) and 1.03 [0.50, 1.63] (Parakeet). On the 212 sentences without digits the change is near zero (-0.05 and -0.16), so the MMS change comes from digits, as expected from its tokenizer.
- **Piper voices get harder without punctuation** (Whisper, Piper lessac +0.89 [0.38, 1.43]; Parakeet, Piper ryan +0.52 [0.12, 0.94]), consistent with the run-on readings found in the TTS-error follow-up.
- **Spread** (worst / best, clean): Whisper 6.11 to 5.23 [4.23, 6.99], Parakeet 5.12 to 4.40 [3.62, 5.77]; on the 212 sentences without digits, 5.17 and 4.57. MMS stays the worst voice. The "spread grows to 5 to 6x on FLEURS" contrast is partly a text artifact: with matched text preparation the spread is 4.4 to 5.2x against about 4x on LibriSpeech, with overlapping CIs.
- **Limits:** clean audio and the two strong models only; the noisy channels and wav2vec2 were not re-run with the new text. The LibriSpeech-style text drops case, so acronyms (FBI, METI) are read as words, as LibriSpeech transcripts would be.

## Do the v2 conclusions hold?

| v2 conclusion (LibriSpeech) | On FLEURS |
|---|---|
| No single voice confidently flips Whisper vs Parakeet; nothing survives Holm | Holds (0 of 14 after Holm). Two voices give the wrong point estimate on realistic. |
| Averaging 7 voices declares a winner where the human CI includes 0 | Does not repeat. On realistic, FLEURS humans show Parakeet better at sentence level (not robust to reader clustering beyond 1.42x); the mean agrees in sign at 0.28x the size. |
| The voice changes how big a large gap looks, 0.37x to 3.72x | Holds in spirit, one direction only: 0.26x to 0.92x, the lab always understates. |
| Kokoro is easier than humans; most other voices are harder | Half holds. Kokoro is easier, by more. Every other voice except MMS is also easier, also with LibriSpeech-style TTS text, and at a 2x CI widening. |
| MMS is the worst voice; worst / best about 4x on clean | Holds. 5.1x to 6.1x with raw text, 4.4x to 5.2x with LibriSpeech-style text (overlapping the LibriSpeech values). |
| Kokoro hides about half of the noise damage; other voices are close | Stronger, with caveats: Kokoro shows 3 to 12 % of the real damage of the noisy channels and all voices understate it. Caveats: FLEURS recordings carry their own noise (the understatement remains in the cleanest third, 35 of 49 cells), and the count drops from 43 to 36 of 49 at a 2x CI widening. |

The practical reading is the same as v2, and stronger: a synthetic run cannot stand in for the real WER level or the real cost of noise, and a close model comparison needs real recordings. What changes is the direction of the error. Against a harder, more varied human set, the lab is optimistic almost everywhere.

## Limits

- **No speaker ids:** sentence-level bootstrap only. Section 5 gives break-even widenings and proxy-cluster CIs; neither can replace real speaker ids.
- **Gender imbalance (not fixed):** 177 female and 73 male recordings. 123 of the 298 eligible sentences have a male recording, but one recording per sentence was drawn at random before balancing, so only 73 were picked. Preferring the scarcer gender within each sentence would fix it, but needs a new selection and a full re-run (20,000 decodes). The impact looks small: Parakeet clean human WER is 3.80 % on female and 3.58 % on male recordings.
- **Reference quality:** FLEURS transcripts contain names, numbers and readings that deviate from the text. A conservative filter finds 26 such words (0.5 % of words); the true number is larger, which inflates human WER and makes every voice look easier. The filter is one-sided (it only removes words humans fail), which is acceptable because it can only shrink the "voices are easier" effect. Numbers are scored after Whisper's normalizer, which does not fix every format ("11:35 pm" vs "eleven thirty five p.m.").
- **SNR on noisy recordings:** see section 3. The damage comparison is the least clean result here.
- **Partial grid:** Whisper ran clean and realistic only, so the Whisper vs Parakeet comparison has 2 conditions instead of 4. No caller scripts, no keyword recall, no rate-matched voices on FLEURS.
- **Training overlap unknown:** FLEURS is out of domain for wav2vec2 by construction. Whisper's and Parakeet's training data are not fully documented, so FLEURS is "less in-domain", not certainly unseen.
- **Still read speech:** no behaviour, no phone recordings, no emergency calls. FLEURS is a second read-speech domain, not a real-caller baseline.

## Files

- `fleurs_subset.csv`, `manifest.csv`: the 250 sentences and the audio paths (audio in `data/`, gitignored)
- `utterances.csv.gz`: 20,000 scored transcriptions
- `human_wer.csv`, `gaps.csv`, `gap_by_sentence_type.csv`, `voice_spread.csv`, `channel_damage.csv`, `decisions.csv`, `reference_mismatch.csv`, `summary.json`
- `fig1_gap_fleurs_vs_librispeech.png`, `fig2_decision_whisper_vs_parakeet.png`, `fig3_human_wer.png`
- `fig4_tts_text_prep.png`
- `robustness/`: `break_even_widening.csv`, `headline_by_grouping.csv`, `proxy_cluster_validation_librispeech.csv`, `fleurs_own_snr.csv`, `channel_damage_by_own_snr.csv`, `summary.json`
- `textprep/`: `tts_text_variants.csv`, `gaps_by_text_prep.csv`, `voice_spread_by_text_prep.csv`, `summary.json`
- Code: `src/sim2real/fleurs.py`, `scripts/08_fleurs.py` (steps `corpus`, `run`, `merge`, `analyze`), `scripts/08b_fleurs_textprep.py` (steps `synth`, `run`, `analyze`), `scripts/08c_fleurs_robustness.py`
