# Follow-up: FLEURS as an out-of-domain human baseline

**Question.** The v2 findings use LibriSpeech as "human speech". LibriSpeech is wav2vec2's training domain and likely part of Whisper's and Parakeet's. Do the v2 conclusions hold when the human side is out-of-domain read speech?

**Short answer.** The direction of the main lesson holds and gets stronger: the lab's picture depends on the voice and does not match real speech. Two v2 details do not hold. (1) On FLEURS, every voice except Meta MMS is easier than real speech, including the voices that were harder than LibriSpeech readers. (2) The "averaging manufactures a winner where real speech has none" pattern does not repeat: on FLEURS the realistic channel has a real winner (Parakeet) and the voices understate it.

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
- All 60 non-MMS cells are easier. On LibriSpeech, the same 70 cells split into 16 easier, 26 harder and 28 not significant.
- **Clean, Parakeet:** Kokoro af_heart -1.41 [-2.00, -0.81], macOS Daniel -1.94 [-2.49, -1.43], Piper ryan -1.41 [-1.97, -0.85], MMS +5.48 [4.49, 6.52]. On LibriSpeech, Daniel and Piper ryan were harder than humans (+0.69 and +0.76).
- **Realistic, Whisper:** Kokoro af_heart -5.72 [-6.84, -4.65], the 7-voice mean -3.54 [-4.57, -2.56]. On LibriSpeech the mean was +0.97 [0.36, 1.57], harder than humans.
- The two rankings still agree in part: Spearman 0.46 across the 70 cells (p = 7e-5). Kokoro stays the easiest voice family and MMS the hardest. 96 % of the FLEURS gaps sit below their LibriSpeech counterpart (median shift -4.5 points over all cells, driven by wav2vec2; for the strong models the median shift is -2.6 points, range -6.5 to +1.9, the only upward shifts being MMS).
- **Spread:** worst voice / best voice is 6.11 [5.19, 8.51] for Whisper clean and 5.12 [4.18, 6.86] for Parakeet clean (LibriSpeech 4.07 and 3.90). MMS is still the worst voice in every strong-model cell.

**Robustness checks.**
- *Reference mismatches.* 26 words in 23 sentences are wrong for all 3 models on human clean audio but right in at least 19 of 21 synthetic decodes (mostly readers saying "street" for "streets", "edge" for "edges", or skipping a word). Removing them moves every gap up by 0.14 to 0.55 points; no conclusion changes (Parakeet clean, Kokoro af_heart: -0.90 instead of -1.41).
- *Names and digits.* TTS reads rare names from their spelling, which could make them easy to transcribe back. On the 95 sentences with no digits and no mid-sentence capitals, Kokoro af_heart is still easier than humans: Whisper clean -1.44 [-2.38, -0.55], Parakeet realistic -4.25 [-6.27, -2.56]. Names and digits do not explain the gap.
- *Speaker clustering.* The gaps sit 1 to 6 points from zero with half-widths of about 0.3 to 1.1 points; a 1.28x widening changes none of them.

### 3. Channel damage: on FLEURS, every voice understates the cost of noise

Damage is a voice's WER under a channel minus its clean WER; the error is synthetic damage minus human damage.
- 43 of 49 voice x channel x model cells differ from reality after BH; all 43 understate the damage. For the strong models, 22 of 28.
- **Realistic, Parakeet:** real damage +3.41 points; Kokoro af_heart shows +0.36 (error -3.05 [-3.98, -2.17]), the 7-voice mean +1.29 (error -2.12 [-2.99, -1.31]).
- **Realistic, Whisper:** real damage +4.50; Kokoro af_heart +0.53 (error -3.96 [-5.04, -2.96]); macOS Samantha, the closest voice, +2.57 (error -1.92 [-2.97, -0.93]).
- v2 found that only Kokoro understated noise damage on LibriSpeech (for example Parakeet phone+noise10 error -1.06) and the other voices were within noise. On FLEURS the Kokoro error is 3 to 6 times larger and the other voices understate too.
- **Caveat:** FLEURS recordings already carry room noise and reverb. The noise is added at 10 dB relative to the clip's active level, and for a noisy recording that level includes its own background, so the effective speech-to-noise ratio of FLEURS humans is probably below 10 dB. Part of this effect may be "noisy recording plus added noise", not a pure voice effect. A phone line alone (Parakeet phone) costs humans only +0.34 points; Kokoro af_heart error -0.55 [-0.90, -0.21].

### 4. Whisper vs Parakeet: no confident flip, but the lab misses a real winner

**Reality on FLEURS** (Whisper WER minus Parakeet WER; positive = Parakeet better):
- clean: +0.14 [-0.36, +0.63], no winner (LibriSpeech: Parakeet by +0.37 [0.12, 0.67]);
- realistic: +1.23 [0.36, 2.10], **Parakeet wins** (LibriSpeech: +0.60 [-0.04, +1.29], no winner). With a 1.28x widening the CI would be about [0.12, 2.34], still excluding 0.

**Single voices (14 decisions: 7 voices x 2 conditions).**
- Realistic: 6 of 7 voices see no winner. Kokoro af_heart (-0.04 [-0.51, +0.44]) and Piper lessac (-0.02 [-0.66, +0.60]) point the wrong way; by the v2 definition these are flips, not confident flips. Only MMS finds the real winner (+1.48 [0.57, 2.39]).
- Clean: Kokoro am_michael (-0.36 [-0.71, -0.04]) and Piper lessac (-0.50 [-0.99, -0.04]) declare Whisper significantly better where FLEURS humans show no winner; MMS declares Parakeet better (+0.93 [0.18, 1.67]).
- The DiD is nominally significant in 5 of 14 decisions (realistic: Kokoro af_heart -1.27 [-2.30, -0.26], Piper lessac -1.25 [-2.35, -0.20], Piper ryan -1.09 [-2.10, -0.12], macOS Daniel -1.07 [-2.09, -0.10]; clean: Piper lessac -0.63 [-1.26, -0.02]). **None survives Holm**; the smallest adjusted p is 0.14. Same verdict as v2 (0 of 28).

**7-voice mean.**
- Clean: -0.05 [-0.31, +0.20], no winner, which matches reality.
- Realistic: +0.34 [0.04, 0.64], the right winner at 0.28 times the real size. DiD -0.88 [-1.79, -0.01], nominal p = 0.047, Holm p = 0.094 over the 2 mean decisions. With a 1.28x widening the DiD CI would include 0.
- So the v2 pattern "averaging makes the lab confident where real speech shows no winner" does not repeat. On FLEURS, averaging gives the right sign with a narrow CI, and the remaining bias is understatement of the real gap.

**Large-gap pairs (vs wav2vec2, 42 voice decisions).**
- The winner never flips (0 of 42), as in v2.
- The lab gap is always smaller than the real gap: lab / real ratio 0.26 to 0.92 (LibriSpeech 0.37 to 3.72, both directions). Realistic, Whisper vs wav2vec2: real gap 56.6 points; Kokoro af_heart says 17.9 and macOS Daniel 47.1.
- The DiD is Holm-significant in 41 of 42 (Parakeet vs wav2vec2 27 of 28, Whisper vs wav2vec2 14 of 14). This mostly measures wav2vec2's collapse on out-of-domain human speech, which no TTS voice reproduces.

## Do the v2 conclusions hold?

| v2 conclusion (LibriSpeech) | On FLEURS |
|---|---|
| No single voice confidently flips Whisper vs Parakeet; nothing survives Holm | Holds (0 of 14 after Holm). Two voices give the wrong point estimate on realistic. |
| Averaging 7 voices declares a winner where real speech has none | Does not repeat. Realistic now has a real winner; the mean finds it but at 0.28x its size. |
| The voice changes how big a large gap looks, 0.37x to 3.72x | Holds in spirit, one direction only: 0.26x to 0.92x, the lab always understates. |
| Kokoro is easier than humans; most other voices are harder | Half holds. Kokoro is easier, by more. Every other voice except MMS is also easier. |
| MMS is the worst voice; worst / best about 4x on clean | Holds; 5.1x to 6.1x on clean. |
| Kokoro hides about half of the noise damage; other voices are close | Stronger: Kokoro shows 3 to 12 % of the real damage of the noisy channels, and all voices understate it (with the SNR caveat above). |

The practical reading is the same as v2, and stronger: a synthetic run cannot stand in for the real WER level or the real cost of noise, and a close model comparison needs real recordings. What changes is the direction of the error. Against a harder, more varied human set, the lab is optimistic almost everywhere.

## Limits

- **No speaker ids:** sentence-level bootstrap only (see Setup); the LibriSpeech widening factor is used as a rough guide.
- **Gender imbalance:** 177 female and 73 male recordings, because one recording per sentence was drawn at random before balancing.
- **Reference quality:** FLEURS transcripts contain names, numbers and readings that deviate from the text. A conservative filter finds 26 such words (0.5 % of words); the true number is larger, which inflates human WER and makes every voice look easier. Numbers are scored after Whisper's normalizer, which does not fix every format ("11:35 pm" vs "eleven thirty five p.m.").
- **SNR on noisy recordings:** see section 3. The damage comparison is the least clean result here.
- **Partial grid:** Whisper ran clean and realistic only, so the Whisper vs Parakeet comparison has 2 conditions instead of 4. No caller scripts, no keyword recall, no rate-matched voices on FLEURS.
- **Training overlap unknown:** FLEURS is out of domain for wav2vec2 by construction. Whisper's and Parakeet's training data are not fully documented, so FLEURS is "less in-domain", not certainly unseen.
- **Still read speech:** no behaviour, no phone recordings, no emergency calls. FLEURS is a second read-speech domain, not a real-caller baseline.

## Files

- `fleurs_subset.csv`, `manifest.csv`: the 250 sentences and the audio paths (audio in `data/`, gitignored)
- `utterances.csv.gz`: 20,000 scored transcriptions
- `human_wer.csv`, `gaps.csv`, `gap_by_sentence_type.csv`, `voice_spread.csv`, `channel_damage.csv`, `decisions.csv`, `reference_mismatch.csv`, `summary.json`
- `fig1_gap_fleurs_vs_librispeech.png`, `fig2_decision_whisper_vs_parakeet.png`, `fig3_human_wer.png`
- Code: `src/sim2real/fleurs.py`, `scripts/08_fleurs.py` (steps `corpus`, `run`, `merge`, `analyze`)
