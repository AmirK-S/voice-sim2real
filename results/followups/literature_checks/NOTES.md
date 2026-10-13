# Follow-up: literature checks on existing results

**Question.** `results/followups/LITERATURE-ACTIONS.md` lists published methods that apply to this repo. Which of the cheap ones change our claims when applied to the transcripts we already have?

**Short answer.**
1. **The v2 interval for the 7-voice mean was too narrow, but the lab's winner survives.** The v2 bootstrap resampled speakers and held the 7 voices fixed. Resampling the voices as well (BOOT-BOTH, Deutsch et al. [28]) widens the Whisper vs Parakeet interval of the mean by 1.6x to 1.9x (voices drawn flat) or 1.8x to 2.3x (engines, then voices). Even so, the mean still declares Parakeet better on phone+noise10 and realistic, and on phone under the flat scheme. The human point estimate points the same way on every condition, the lab-minus-real offset is never significant, and the human sample is too small to confirm or rule out a gap of this size. **So "averaging manufactures certainty where real speech shows no winner" is not supported as worded.** What the data support is narrower: the v2 CI ignored voice-to-voice variance, and the lab declares a winner that 250 human sentences from 40 speakers cannot check.
2. **Other claims about the 7-voice mean do not survive voice resampling on LibriSpeech:** the mean's understatement of wav2vec2's realistic damage (4.0 points [1.4, 6.8]), 10 (flat) to 13 (nested) of its 14 nominally significant large-gap DiDs, and 15 to 18 of its 20 significant gaps against human WER. On FLEURS, the mean's understatement of noise damage survives (6 of 7 cells under every scheme). The mean's FLEURS realistic agreement with the human winner no longer excludes 0.
3. **No voice and no average ever made a confident wrong-sign decision**, on LibriSpeech or FLEURS. The lab's decision errors are all about significance (Type I and Type II in the sense of Mathur et al. [27]), not about order. Pairwise decision accuracy (Kocmi et al. [26]) is 18 of 18 for 6 of 7 voices on LibriSpeech (macOS Daniel 17 of 18) and 6 to 8 of 8 on FLEURS. Every swap is in the close pair, Whisper vs Parakeet.
4. **Our PPI code matches the reference implementation** `ppi_py` (ppi-python 0.2.3): identical point estimates, identical variance formula and power-tuning lambda once the documented choices are aligned. Run off the shelf on our draws, `ppi_py` gives the same picture: PPI covers, and power-tuned PPI++ under-covers at 10 labels (73 to 84 %). The "classic PPI below about 25 labels" rule stands.

Everything here reuses the scored transcripts of the main run and of `followups/fleurs`. There is no new transcription and no GPU work. Code: `scripts/10_literature_checks.py`. Bracketed numbers refer to `RELATED-WORK.md`.

## 1. Resampling voices as well as speakers (BOOT-BOTH)

**Method.** Each bootstrap replicate draws the speakers (LibriSpeech, 40 clusters) or sentences (FLEURS, no speaker ids) exactly as in v2, with the same seed. Each replicate also draws a set of voices, independently. The 7-voice mean of that replicate is a weighted mean of the per-voice statistics. Four schemes:
- `speaker_only`: voices fixed. This is the v2 method, and it reproduces `results/tables/decision_flips.csv` and `followups/fleurs/decisions.csv` (largest difference 5e-6, which is rounding in the CSVs).
- `voices_only`: sentences fixed, 7 voices drawn with replacement. It shows how much the voice draw alone contributes.
- `both_flat`: speakers and 7 voices drawn with replacement.
- `both_nested`: speakers, then 4 engines drawn with replacement, then within each drawn engine as many voices as it has, drawn with replacement from its own voices. The engines are Kokoro (af_heart and am_michael share one model), Piper (lessac and ryan, two VITS voice models), MMS-TTS (one VITS model) and macOS (Samantha and Daniel).

Clark [106] calls treating a sample of stimuli as fixed the language-as-fixed-effect fallacy. The voices are our stimuli. With 7 voices and 4 engines, the voice variance is itself poorly estimated. The two schemes bracket it; neither one is exact. 10,000 replicates, percentile intervals.

**Whisper minus Parakeet, 7-voice mean** (`mean_boot_both.csv`; points, positive = Parakeet better):

| Corpus, condition | Human gap | Mean, voices fixed (v2) | Mean, flat | Mean, nested | Width vs v2 | DiD, flat | DiD, nested |
|---|---|---|---|---|---|---|---|
| LibriSpeech, clean | +0.37 [0.12, 0.67] | +0.48 [0.17, 0.82] | [0.02, 1.02] | [0.00, 1.18] | 1.55x / 1.82x | +0.11 [-0.42, +0.67] | [-0.44, +0.80] |
| LibriSpeech, phone | +0.16 [-0.23, +0.57] | +0.44 [0.16, 0.73] | [0.02, 0.97] | [-0.01, 1.14] | 1.64x / 1.99x | +0.28 [-0.27, +0.85] | [-0.30, +0.98] |
| LibriSpeech, phone+noise10 | +0.49 [-0.15, +1.22] | +0.86 [0.58, 1.16] | [0.38, 1.47] | [0.34, 1.64] | 1.91x / 2.26x | +0.38 [-0.52, +1.23] | [-0.52, +1.38] |
| LibriSpeech, realistic | +0.60 [-0.04, +1.29] | +0.75 [0.40, 1.11] | [0.20, 1.38] | [0.17, 1.54] | 1.64x / 1.92x | +0.15 [-0.63, +0.94] | [-0.64, +1.08] |
| FLEURS, clean | +0.14 [-0.36, +0.63] | -0.05 [-0.31, +0.20] | [-0.46, +0.42] | [-0.45, +0.54] | 1.72x / 1.93x | -0.19 [-0.78, +0.43] | [-0.79, +0.53] |
| FLEURS, realistic | +1.23 [0.36, 2.10] | +0.34 [0.04, 0.64] | [-0.12, +0.89] | [-0.13, +1.04] | 1.68x / 1.95x | -0.88 [-1.88, +0.07] | [-1.86, +0.17] |

- **LibriSpeech.** Resampling voices roughly doubles the width of the interval, but on phone+noise10 and realistic the mean still excludes 0 under both schemes. On phone it excludes 0 under the flat scheme only (lower bound +0.02 flat, -0.01 nested). Most of the extra width comes from the voices and the speakers together: neither source alone gives it (`voices_only` widens by 1.08x to 1.42x).
- **The Type I count of the mean stays at 3 under the flat scheme and drops to 2 under the nested one.** A Type I here means the lab CI excludes 0 and the human CI does not. The v2 pattern is therefore not an artifact of holding the voices fixed. Under the nested scheme the clean-audio winner becomes borderline (lower bound 0.00), which counts as a Type II.
- **The lab offset (DiD) is not significant under any scheme**, as already noted in `followups/calibration`.
- **FLEURS, realistic.** With voices resampled, the mean's interval includes 0. So the lab now misses the sentence-level human winner, rather than agreeing with it at 0.28 times its size. That human winner is itself not robust to reader clustering (`followups/fleurs`, break-even widening 1.42x). The DiD of -0.88 loses its nominal significance (v2 p = 0.047; flat [-1.88, +0.07]).

**Can the human data say there is no winner?** (`human_gap_tost_mde.csv`; E7 in LITERATURE-ACTIONS.) Two checks:
- *Equivalence (TOST, Lakens [118]).* With a smallest effect of interest of 0.5 points (my illustrative choice, not pre-registered; Amir should set it from what would change a deployment decision), no condition is equivalent to 0. The 90 % CI of the human gap is [-0.17, +0.51] on phone, [-0.07, +1.09] on phone+noise10 and [+0.05, +1.18] on realistic.
- *Minimum detectable effect.* At 80 % power, before any multiplicity correction, the effect the human sample can detect is 2.8 bootstrap SEs: 0.58 points on phone, 0.98 on phone+noise10 and 0.96 on realistic. The lab's mean gaps there are 0.44, 0.86 and 0.75, all below what the human sample could reliably detect.

So "real speech shows no winner" means only that the human CI includes 0. The human sample is underpowered for gaps of the size the lab reports, and its point estimates agree with the lab in direction.

**Other claims about the mean** (`mean_vs_human_boot_both.csv`: mean minus human WER, and channel-damage error, per model x condition):

| Claim (source) | Voices fixed (v2) | Flat | Nested |
|---|---|---|---|
| wav2vec2 realistic damage understated by the mean, LibriSpeech (FINDINGS 3) | 3.99 [1.43, 6.77] | [-2.32, +9.19] | [-3.60, +10.21] |
| Whisper realistic, mean harder than humans, LibriSpeech (fleurs NOTES) | +0.97 [0.36, 1.57] | [-0.68, +2.82] | [-0.85, +3.56] |
| Whisper realistic, mean easier than humans, FLEURS | -3.54 [-4.57, -2.56] | [-5.66, -0.67] | [-5.68, +0.64] |
| Parakeet realistic damage understated by the mean, FLEURS | 2.12 [1.31, 2.99] | [1.18, 3.10] | [1.05, 3.18] |
| Significant mean-vs-human WER gaps, LibriSpeech (24 cells) | 20 | 5 | 2 |
| Significant mean damage errors, LibriSpeech (21 cells) | 8 | 2 | 1 |
| Significant mean-vs-human WER gaps, FLEURS (10 cells) | 10 | 7 | 3 |
| Significant mean damage errors, FLEURS (7 cells) | 6 | 6 | 6 |
| Large-gap DiDs of the mean, nominal p < 0.05, LibriSpeech (14) | 14 | 4 | 1 |

- On LibriSpeech, a statement about the 7-voice mean rarely survives voice resampling, because the voices disagree in both directions (for example, Kokoro understates wav2vec2's damage and macOS Daniel overstates it). This restates the main finding, that the voice choice drives the lab result, in the language of [28].
- On FLEURS, the understatement of noise damage is shared by every voice, so it survives.
- Single-voice results (for example the MMS DiDs, or Kokoro's understatement of noise damage) do not involve the voice draw and are unchanged.

## 2. Pairwise decision accuracy and Type I / Type II table

**Method.** One decision is a model pair x condition. LibriSpeech has 18 per voice: Whisper vs Parakeet on 4 conditions, Whisper vs wav2vec2 on 4, Parakeet vs wav2vec2 on 10. FLEURS has 8 per voice: 2, 2 and 4. For each voice, `decisions_by_voice.csv` and `pairwise_accuracy_type_I_II.csv` give:
- **pairwise accuracy** [26]: the share of decisions in which the lab's point estimate orders the pair as the human point estimate does. One minus it is the swap rate [116].
- **Type I** [27]: the lab CI excludes 0 and the human CI does not.
- **Type II** [27]: the human CI excludes 0 and the lab CI does not.
- **confident wrong sign**: both CIs exclude 0, with opposite signs.

Accuracy CIs come from the same speaker (or sentence) bootstrap. They are rough, because decisions share sentences.

| Corpus | Voice | Accuracy | Type I | Type II | Confident wrong sign |
|---|---|---|---|---|---|
| LibriSpeech (18 each) | Kokoro af_heart, am_michael, Piper lessac | 18/18 | 0, 0, 1 | 2, 1, 1 | 0 |
| | Piper ryan, MMS, Samantha | 18/18 | 3, 3, 2 | 0, 0, 1 | 0 |
| | Daniel | 17/18 | 0 | 1 | 0 |
| | mean: fixed / flat / nested | 18/18 | 3 / 3 / 2 | 0 / 0 / 1 | 0 |
| FLEURS (8 each) | Kokoro af_heart, Piper lessac | 6/8 | 0, 1 | 1, 1 | 0 |
| | Kokoro am_michael, Samantha, Daniel | 7/8 | 1, 0, 0 | 1, 1, 1 | 0 |
| | Piper ryan, MMS | 8/8 | 0, 1 | 1, 0 | 0 |
| | mean: fixed / flat / nested | 7/8 | 0 / 0 / 0 | 0 / 1 / 1 | 0 |

- **Restricted to decisions where the human CI excludes 0** (15 on LibriSpeech, 7 on FLEURS), accuracy is 15/15 for every voice except Daniel (14/15) on LibriSpeech, and 7/7 except Kokoro af_heart and Piper lessac (6/7) on FLEURS.
- **Accuracy is high mostly because 14 of 18 LibriSpeech decisions (6 of 8 on FLEURS) involve the wav2vec2 control**, whose gaps are large. For Whisper vs Parakeet alone, accuracy is 4/4 for 6 voices and 2/4 for Daniel on LibriSpeech, and 0/2 to 2/2 on FLEURS. With 1 human-significant close decision per corpus, this cannot be estimated with any precision.
- **The lab never confidently picks the wrong winner.** Its errors are about significance: it calls a winner the human sample does not show (Type I, mostly Piper ryan, MMS, Samantha and the mean on the noisy channels), or misses one the human sample shows (Type II, mostly Whisper vs Parakeet on clean LibriSpeech and on realistic FLEURS). Both depend on the human sample size as much as on the voice (section 1).
- Kendall tau was not added: with 3 models it takes only 4 values [29].

## 3. PPI and PPI++ against the reference implementation

`ppi_py` 0.2.3 was installed into the project venv (PyPI `ppi-python`, with its declared dependencies, statsmodels among them). It is not added to `pyproject.toml`. `tests/test_ppi_reference.py` runs the exactness checks below and skips when `ppi_py` is absent, so CI is unaffected.

**Exactness** (on every draw of the `ppi.csv` simulation; `summary.json`, `ppi_reference`):
- PPI point estimate (lambda = 1, f known for all 250 sentences): identical (largest difference 0).
- Standard error: identical to 4e-17 once our formula uses ppi_py's choices. Those are ddof 0, an added var(f)/N term for the unlabeled set, and no finite-population correction.
- Power-tuning lambda before clipping: ppi_py's equals ours x (n - 1)/n to 3e-15. The factor is ddof 0 against ddof 1. ppi_py clips lambda to [0, 1]; our PPI++ does not clip.
- Our recomputed coverages reproduce `results/tables/ppi.csv` exactly.

**Off-the-shelf behaviour on our draws** (`ppi_vs_ppi_py.csv`; Whisper and Parakeet, 7-voice-mean proxy, 1,000 draws per cell, ranges over models and conditions):

| n | Ours: PPI coverage | ppi_py: PPI coverage | Ours: PPI++ coverage | ppi_py: PPI++ (clipped) coverage | ppi_py: classical coverage |
|---|---|---|---|---|---|
| 10 | 94.2 to 97.8 % | 91.0 to 96.8 % | 76.2 to 86.2 % | 73.3 to 83.8 % | 74.6 to 89.3 % |
| 25 | 94.0 to 96.5 % | 95.3 to 98.0 % | 86.6 to 91.6 % | 87.5 to 91.5 % | 88.3 to 92.6 % |
| 50 | 94.2 to 95.7 % | 97.4 to 99.0 % | 89.8 to 94.7 % | 91.4 to 95.8 % | 92.5 to 95.6 % |

- ppi_py uses a normal quantile, so at n = 10 it is narrower and covers a little less than our t interval.
- At n = 50 it over-covers, because it ignores the finite-population correction (50 of 250 sentences).
- The conclusions of FINDINGS section 8 hold with the reference code: PPI restores coverage at small n, and power-tuned PPI++ under-covers at 10 labels whether lambda is clipped or not.

## Wording fixes applied to FINDINGS (section I of LITERATURE-ACTIONS)

Each one is marked in place as a correction:
- **Setup.** The DiD is bootstrapped by speaker cluster on LibriSpeech, not per sentence. The 7 voices come from 4 engines.
- **Section 1.** The BOOT-BOTH result, the TOST and the minimum detectable effect, and the restated averaging claim.
- **Section 3.** The 7-voice-mean understatement of wav2vec2 damage is not robust to voice resampling.
- **Section 6.** The 33 of 100 comes from the raw decoder, with no no-speech gate and no temperature fallback. `followups/phantom` shows that the gate never fires for large-v3-turbo. Fallback was not tested.
- **Section 8.** The "record about 25 scripts" advice needs about 20 distinct speakers drawn from the target callers: in the simulation, 25 random sentences came from 19.4 readers on average (5th to 95th percentile 17 to 22). If one person records all 25 scripts, the interval covers that person's WER, not callers'.
- **Section 8.** The ppi_py cross-check.
- **Limitations.** FLEURS has since been run.

Two items of section I were already handled in the working tree by the interrupted review-fix pass: the Holm family in section 4, and "rendering errors hurt Whisper more" in section 1. They are not part of this commit. For that second claim, the MMS penalty on Whisper has outside support from 2M-BELEBELE [1], which found that Whisper loses more on MMS-TTS audio than its competitor. That supports the direction, not our untested mechanism.

**Note on `RELATED-WORK.md`** (untracked, from the separate literature review; not edited here). Its lines 54 and 142 say that resampling only speakers is "precisely why" the mean's CI is too narrow and that BOOT-BOTH is "the fix". This check shows that BOOT-BOTH widens the interval by about 2x but does not remove the lab's winner on 2 or 3 of the 3 conditions. The remaining gap is mostly a power problem on the human side, not voice variance alone.

## Limits

- **Few voices and engines.** With 7 voices and 4 engines, any estimate of voice variance is rough. The flat and nested schemes bracket it but do not settle it. A team that picks voices on purpose, rather than at random, is not described by either scheme.
- **The equivalence margin of 0.5 points is illustrative.** The minimum detectable effect ignores multiplicity.
- **Small decision tables.** Accuracy and Type I/II counts are descriptive. Decisions share sentences, and the close pair has 4 (LibriSpeech) or 2 (FLEURS) decisions per voice.
- **FLEURS still uses a sentence bootstrap** (no speaker ids). Its CIs are probably too narrow (`followups/fleurs`, section 5).
- **Not done tonight.** The other LITERATURE-ACTIONS items (phantom gate arms, fidelity-weighted PPI, active-selection mixture, MMRV, glimpse proportion) are not addressed here.

## Files

- `mean_boot_both.csv`: 7-voice-mean lab gap and DiD for every model pair x condition, under the 4 schemes.
- `mean_vs_human_boot_both.csv`: mean minus human WER, and channel-damage error, per model x condition, under the 4 schemes.
- `human_gap_tost_mde.csv`: Whisper minus Parakeet human gap, 90 % and 95 % CIs, TOST at 0.5 points, minimum detectable effect.
- `decisions_by_voice.csv`: per voice (and mean scheme) x pair x condition: significance, sign agreement, Type I/II.
- `pairwise_accuracy_type_I_II.csv`: accuracy, swap rate and Type I/II counts per voice, by scope.
- `ppi_vs_ppi_py.csv`: coverage and width, ours against ppi_py, per model x condition x n.
- `summary.json`: reproduction checks, exactness checks, headline rows.
