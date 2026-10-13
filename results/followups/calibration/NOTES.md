# Calibrating synthetic voices with a few human recordings

Follow-up to the v2 findings. CPU only, on the existing LibriSpeech results (250 sentences, 40 speakers); no new transcription. Code: `src/sim2real/calibrate.py`, `scripts/06_calibration.py` (about 65 s). "Points" are WER percentage points. Per-sentence scores are errors divided by a constant mean word count, so their mean over a set is that set's corpus WER.

**Revised after review (13/10/2026).** Wording only for the main claims; two additions to the code (cross-fit PPI++, stratified design with at least 4 labels per stratum), which leave every earlier number unchanged (they use their own random streams). Changes: "false winner" is replaced by "winner where the 250-sentence human CI includes 0", winner rates are split by direction, every "bias" figure now sits next to the lab offset's CI, the fidelity sentence on model differences no longer says "cannot be predicted", the active-selection headlines use RMSE savings only, and the cells dropped from the medians are counted.

## Short answer

1. **No voice is a faithful per-sentence stand-in.** Single voices correlate with human per-sentence errors at r = 0.26 to 0.82 (median 0.49); the 7-voice mean at 0.49 to 0.83. For the close pair Whisper vs Parakeet, no voice shows a detectable correlation with the human per-sentence *difference* between models: r between -0.15 and +0.27, and every 95 % CI includes 0. The data cannot rule out correlations up to about 0.5 (largest upper bound 0.51).
2. **A few human recordings fix the v2 over-precision, but only by giving up the synthetic shortcut.** The plain 7-voice average reports an interval about 0.7 to 0.9 points wide, far narrower than its real uncertainty against human speech: on held-out speakers it covers the human difference in only 47 %, 65 % and 86 % of splits (phone+noise10, phone, realistic). Its mean offset toward Parakeet (+0.14 to +0.38 points) is a point estimate whose CI on the 250 sentences includes 0, so a bias is not demonstrated; the problem shown here is precision. With 25 calibration sentences, PPI covers the held-out human value in 94 to 96 % of splits and declares a winner in 4.6 to 6.6 % of splits on the three conditions where the 250-sentence human CI includes 0 (plain average: 77 to 100 %). Almost all of these winners point to Parakeet, the same direction as the human point estimates. PPI intervals are as wide as human-only intervals, so it also rarely detects the one condition where the human CI excludes 0 (Parakeet on clean audio: 8.4 % of splits).
3. **Active selection mostly shrinks the reported CI, not the actual error.** At 25 recordings the actual estimation error (RMSE) of the uncertainty-driven designs is 1 to 7 % lower than random PPI (median over 16 targets), worth about 0.5 to 4 recordings. Their reported intervals look 15 to 19 % narrower, but that is not a saving, because their coverage falls to 83 to 90 % in some cells. The two Poisson designs stayed closest to nominal at n = 25 (89 to 96 %). Requiring at least 4 labels per stratum lifts the stratified designs' minimum coverage at n = 25 from 83 to 90 %, but not at n = 75 to 100 (down to 72 %).

## 1. Voice fidelity

Tables: `fidelity.csv` (per voice x condition x model, 10 conditions where run, rate-matched voices on clean), `fidelity_diff.csv` (per model pair). CIs: 95 % speaker-cluster bootstrap, 2,000 replicates. Figure: `fig_fidelity.png`.

**Per-sentence correlation with human errors** (Pearson r, median over the condition x model cells):

| Voice | median r | median calibration slope (human on synthetic) |
|---|---|---|
| Kokoro am_michael | 0.58 | 0.75 |
| Kokoro af_heart | 0.53 | 0.71 |
| Piper lessac | 0.51 | 0.48 |
| macOS Daniel | 0.50 | 0.42 |
| macOS Samantha | 0.48 | 0.39 |
| Meta MMS | 0.41 | 0.24 |
| Piper ryan | 0.33 | 0.21 |
| mean of 7 voices | 0.49 to 0.83 (range) | 0.52 to 1.05 (range) |

- Examples with CIs: Kokoro am_michael on Parakeet clean r = 0.67 [0.48, 0.79], slope 0.76 [0.54, 0.97]; Piper ryan on Parakeet clean r = 0.30 [0.12, 0.58], slope 0.19 [0.05, 0.57]; MMS on Whisper clean slope 0.19 [0.06, 0.33].
- A slope well below 1 means the voice's errors are mostly its own: sentences where MMS or Piper ryan fail are often easy for humans. This matches the v2 rendering-error finding.
- The Kokoro voices, easiest on average (v2), are also the most faithful per sentence. Being easy and being faithful are different properties.
- Rate-matched voices change fidelity little (for example MMS on Whisper clean 0.34 to 0.39 [0.15, 0.62]).

**Agreement on model differences.**

- Whisper vs Parakeet, per-sentence difference: r between -0.15 and +0.27 across the 7 voices and 4 conditions (32 cells, rate-matched voices excluded); no CI excludes 0. The 7-voice mean: +0.15 [-0.11, +0.44] clean, +0.24 [-0.05, +0.49] phone, -0.06 [-0.21, +0.12] phone+noise10, +0.05 [-0.09, +0.19] realistic. So: no detectable correlation, but not proof of none; the upper bounds reach 0.51 (macOS Samantha, clean: +0.24 [-0.03, +0.51]).
- On sentences where humans and the voice both show a Whisper vs Parakeet difference (10 to 64 sentences per cell), the sign agrees 42 to 80 % of the time, about a coin flip at these sample sizes.
- Large-gap pairs (Whisper or Parakeet vs wav2vec2) are better: the 7-voice mean reaches r = 0.39 to 0.83, and its sign agrees with humans on 84 to 100 % of the sentences where humans differ.

**Limits.** Correlations are computed over 250 sentences of read audiobook speech, which is in-domain for wav2vec2 and probably seen by the other models. Many sentences have zero errors for both human and synthetic audio, which inflates correlations through the shared "easy sentence" mass. The sign-agreement metric counts a zero synthetic difference as disagreement (column `sign_agree`), which penalizes single voices against the 7-voice mean; `sign_agree_both_nonzero` drops those sentences.

## 2. Calibrated weighting (Whisper vs Parakeet)

Tables: `weighting.csv` (summary), `weighting_reps.csv.gz` (every split). Figure: `fig_weighting.png`.

**Design.** 1,000 random splits per n. Each split puts 16 speakers in the calibration pool and 24 in evaluation (speakers disjoint); n = 10, 25 or 50 calibration sentences are drawn from the pool; evaluation uses all sentences of the 24 evaluation speakers (about 150). The target is the human corpus-WER difference (Whisper minus Parakeet) on the evaluation sentences. The reference decision is the v2 one on all 250 sentences: Parakeet better on clean, +0.37 [0.12, 0.67]; no winner on phone, phone+noise10 and realistic.

**Methods.**
- plain average: 7-voice mean on the evaluation set, speaker-cluster bootstrap CI (the v2 approach, no labels).
- fidelity-weighted: voices weighted by the inverse mean squared error of their per-sentence difference on the calibration sentences (floored at 5 % of the mean MSE); same CI.
- linear-corrected: per voice and per model, OLS of human score on synthetic score on the calibration sentences, applied to the evaluation set, averaged over voices; CI by a joint bootstrap of calibration sentences (by speaker) and evaluation speakers. Note that this is close to a per-voice PPI++ (regression estimator).
- human only: the calibration sentences alone.
- PPI (lambda = 1) and PPI++ (plug-in lambda, not clipped) on the per-sentence model difference, with the 7-voice mean as the prediction. Two-sample variance S_r^2 (1/n + 1/N_eval), t interval with n - 1 df.
- PPI++ cross-fit (added after review): the labels are split in two halves, the lambda of one half is applied to the other, and the two estimates are averaged (`calibrate.two_sample_rectified_cf`); t with n - 2 df.

**What "winner" means here.** The reference is the human result on all 250 sentences. On phone, phone+noise10 and realistic its CI includes 0, but the point estimate favours Parakeet each time: +0.16 [-0.23, +0.57], +0.49 [-0.16, +1.22], +0.60 [-0.04, +1.29]. A winner declared on these conditions is therefore "a winner where the 250-sentence human CI includes 0", not a proven false winner. The lab offset of the plain average, its 7-voice-mean difference minus the human difference on the 250 sentences (difference in differences), is +0.11 [-0.28, +0.49] clean, +0.28 [-0.13, +0.68] phone, +0.38 [-0.39, +1.05] phone+noise10 and +0.15 [-0.48, +0.74] realistic (speaker-cluster bootstrap, `summary.json`). No offset CI excludes 0.

**Results at n = 25** (rates over 1,000 splits; Monte Carlo SE at most 1.6 percentage points). "Winner, P / W": share of splits declaring Parakeet / Whisper better. "Offset" is the mean error against the held-out human value; for the plain average it estimates the lab offset above, whose 250-sentence CI includes 0.

| Condition | Method | declares a winner | winner, P / W | coverage | offset (points) | median CI width (points) |
|---|---|---|---|---|---|---|
| phone+noise10 (human CI includes 0) | plain average | 100.0 % | 100.0 / 0.0 | 46.8 % | +0.38 | 0.71 |
| | fidelity-weighted | 99.8 % | 99.8 / 0.0 | 56.5 % | +0.27 | 0.69 |
| | linear-corrected | 9.9 % | 7.2 / 2.7 | 86.2 % | -0.08 | 3.90 |
| | human only | 3.5 % | 2.9 / 0.6 | 93.2 % | -0.04 | 4.39 |
| | PPI | 4.6 % | 4.1 / 0.5 | 93.9 % | -0.02 | 4.94 |
| | PPI++ | 4.5 % | 3.9 / 0.6 | 91.7 % | -0.01 | 4.25 |
| | PPI++ cross-fit | 3.9 % | 3.4 / 0.5 | 92.4 % | -0.07 | 4.63 |
| realistic (human CI includes 0) | plain average | 98.3 % | 98.3 / 0.0 | 86.1 % | +0.14 | 0.90 |
| | fidelity-weighted | 92.7 % | 92.7 / 0.0 | 84.7 % | +0.02 | 0.82 |
| | linear-corrected | 10.5 % | 9.5 / 1.0 | 86.5 % | -0.10 | 3.71 |
| | human only | 4.9 % | 4.8 / 0.1 | 93.5 % | -0.04 | 4.42 |
| | PPI | 6.6 % | 6.4 / 0.2 | 96.2 % | -0.04 | 4.81 |
| | PPI++ cross-fit | 5.1 % | 4.9 / 0.2 | 93.7 % | -0.04 | 4.68 |
| phone (human CI includes 0) | plain average | 76.9 % | 76.9 / 0.0 | 64.5 % | +0.28 | 0.73 |
| | fidelity-weighted | 43.8 % | 43.8 / 0.0 | 80.1 % | +0.15 | 0.67 |
| | PPI | 5.2 % | 4.6 / 0.6 | 94.9 % | +0.01 | 2.63 |
| | PPI++ cross-fit | 2.8 % | 2.4 / 0.4 | 92.3 % | -0.01 | 2.69 |
| clean (human: Parakeet) | plain average | 75.3 % | 75.3 / 0.0 | 96.4 % | +0.11 | 0.84 |
| | human only | 0.4 % | 0.4 / 0.0 | 90.5 % | +0.03 | 2.33 |
| | PPI | 8.7 % | 8.4 / 0.3 | 95.7 % | +0.04 | 2.45 |
| | PPI++ cross-fit | 2.6 % | 2.3 / 0.3 | 89.5 % | +0.04 | 2.46 |

- **Over-precision, not a demonstrated bias.** The plain average's interval is 0.7 to 0.9 points wide and covers the held-out human value in only 47 to 86 % of splits on the three conditions where the human CI includes 0. Its mean offset toward Parakeet (+0.14 to +0.38) matches the lab offsets above, whose CIs include 0. So it declares Parakeet better in 77 to 100 % of splits whatever the condition, with far more confidence than its real uncertainty allows. Its results do not depend on n (it uses no labels).
- **Direction of the winners.** Almost every winner declared by any method points to Parakeet, as do the human point estimates (realistic, n = 25: 64 of PPI's 66 winners; n = 50: 47 of 47). These winners may be right; the 250-sentence human data cannot say.
- **Fidelity weighting** reduces the offset (for example +0.38 to +0.27 on phone+noise10) but keeps the narrow CI, so it still declares a winner in 37 to 100 % of splits (all n, the three conditions where the human CI includes 0).
- **Linear correction** declares a winner in 4 to 11 % of splits but under-covers: 74 to 90 % across n and conditions.
- **PPI** is the only method near nominal coverage at every n: 93 to 98 %. PPI++ under-covers at n = 10 (63 to 91 %), as in v2. The cross-fit PPI++ does barely better at n = 10 (65 to 94 %) and reaches 90 to 95 % at n = 25 and 50, close to PPI but not better: with this weak correlation the fitted lambda gains almost nothing.
- **The price.** PPI intervals are as wide as human-only, or wider (2.4 to 4.9 points at n = 25 against 2.3 to 4.4), because the synthetic per-sentence difference shows no detectable correlation with the human one (section 1). On clean audio, where the 250-sentence human CI excludes 0 (+0.37 [0.12, 0.67]), no valid method detects the Parakeet advantage in more than 8.4 % of splits with 50 or fewer labels (PPI 4.9 to 8.4 %, human only 0 to 2.3 %). The plain average "detects" it 75 % of the time only because its offset points the same way. Note that this +0.37 reference uses all 250 sentences, including the calibration sentences of each split.
- n = 10 and n = 50 tell the same story (`weighting.csv`); human-only intervals under-cover at n = 10 (66 to 95 %) because many draws contain only zero differences.

**Limits.** One fixed dataset: the rates are Monte Carlo over splits of the same 250 sentences and 40 speakers and do not include the uncertainty of having drawn these speakers. The coverage figures are conditional on these 40 speakers: splitting them into calibration and evaluation groups does not test new speakers. The two-sample variance ignores speaker clustering inside the evaluation set; coverage was checked by simulation instead. Agreement with the evaluation-set human decision (column `decision_acc_vs_eval`) is lower for every method, because 24 evaluation speakers sometimes show a winner the full set does not.

## 3. Active selection

Tables: `active.csv` (coverage, median and mean CI width, bias, RMSE per target x condition x design x n), `active_savings.csv`. Figure: `fig_active.png`.

**Design.** Finite population of the 250 sentences; human audio is the oracle "recording". Targets: human corpus WER for each of the 3 models and the Whisper minus Parakeet difference, on clean, phone, phone+noise10 and realistic (16 targets). Budgets n = 10 to 100; 2,000 simulated samples per design and n. The prediction is the 7-voice mean.

Designs (all keep PPI design-unbiased, because selection depends only on synthetic results, fixed before any human label is seen):
- random, human only and random, PPI: simple random sample without replacement, finite-population correction, t interval.
- stratified: 5 equal-size strata by rank of an uncertainty score, Neyman allocation with the score as the proxy for the stratum SD (at least 2 per stratum), PPI rectifier per stratum, stratified variance, t with n - 5 df.
- Poisson IPW: each sentence recorded independently with probability proportional to the score (mixed 80/20 with uniform, capped at 1, summing to n), Horvitz-Thompson rectifier and variance, normal interval (Zrnic and Candes style active inference). The number of recordings is random with mean n.
- Scores: "disagreement" = SD of the 7 voices' per-sentence scores; "magnitude" = their root mean square.

**Results at n = 25, medians over the 16 targets** (at 50 in brackets). Only the RMSE-based saving is a saving in actual precision; the width-based figures are shown next to coverage and are not savings for designs that under-cover.

| Design | coverage, min | actual RMSE vs random PPI | recordings saved at equal RMSE | reported CI width vs random PPI | (width-based "saving") |
|---|---|---|---|---|---|
| random, human only | 88.8 % (90.8 %) | 1.17 (1.20) | -6.8 (-13.0) | 1.15 (1.17) | -6.4 (-12.0) |
| random, PPI | 93.6 % (93.8 %) | 1 | 0 | 1 | 0 |
| stratified, disagreement | 89.6 % (87.2 %) | 0.94 (1.01) | 2.5 (-1.0) | 0.85 (0.83) | 7.7 (12.5) |
| stratified, disagreement, min 4 per stratum | 91.2 % (88.5 %) | 0.93 (0.97) | 2.9 (3.0) | 0.90 (0.87) | 4.6 (11.5) |
| stratified, magnitude | 83.3 % (83.0 %) | 0.93 (0.93) | 3.6 (5.7) | 0.84 (0.81) | 8.4 (15.7) |
| stratified, magnitude, min 4 per stratum | 90.3 % (84.2 %) | 0.91 (0.94) | 4.2 (5.5) | 0.91 (0.85) | 4.0 (14.0) |
| Poisson IPW, disagreement | 89.6 % (90.0 %) | 0.99 (0.99) | 0.5 (0.4) | 0.81 (0.86) | 11.1 (12.5) |
| Poisson IPW, magnitude | 89.4 % (88.4 %) | 0.95 (0.97) | 3.1 (2.6) | 0.81 (0.84) | 11.6 (13.0) |

- **Cells left out of the medians.** "Recordings saved" is nan when the matching n falls outside the 10 to 100 grid. Over n = 25 and 50 and all designs: 20 rows for width against human only, 6 for width against random PPI, 3 for RMSE against human only, 0 for RMSE against random PPI. The medians above exclude them (at most 4 of 16 targets per cell, all at n = 50).
- **The reported gain is mostly not real.** All active designs are unbiased (mean error within 0.2 points in every cell), but their variance estimates are too small at these n: residuals are heavy-tailed (most sentences have zero error), and the rare large residual in a lightly sampled low-uncertainty region is usually missed. The CI looks 15 to 19 % narrower; the actual error is 1 to 7 % smaller (median).
- **Why the stratified designs under-cover.** In the low-uncertainty strata most residuals are exactly 0. With 2 labels per stratum (the original minimum) both are often 0, so the stratum's variance estimate is 0 while its true variance is not. Requiring 4 labels per stratum raises the minimum coverage at n = 25 from 83.3 to 90.3 % (magnitude) and from 89.6 to 91.2 % (disagreement). It does not fix larger n: at n = 75 to 100 the magnitude design still falls to 72 to 79 % on Whisper clean, because the low-uncertainty stratum then gets a small share of a larger budget and its rare non-zero residuals are still usually missed. All n in `active.csv`.
- **Where it may help (exploratory: the best of 128 target x condition x design cells at n = 25, not a pre-registered comparison):** strong models on clean and phone audio. Poisson IPW by magnitude: Whisper phone RMSE ratio 0.77 (about 11 recordings saved, coverage 95.1 %); Parakeet phone 0.79 (about 12.5 saved, coverage 94.4 %); Whisper minus Parakeet on clean 0.83 (about 8 saved, coverage 93.8 %). Under noise it saves little or nothing (RMSE ratio 0.93 to 1.04).
- **Random PPI is the safe default**: coverage 92.4 to 97.5 % in every cell (all n), and against random human-only it saves about 9.5 recordings at n = 25 and 17 at n = 50 (median over the 12 WER targets, equal RMSE, no cell left out). For the Whisper minus Parakeet difference it saves nothing: the human-only interval is 5 to 11 % narrower, because the synthetic difference does not track the human one.
- If one active design is used, use **Poisson IPW by synthetic error magnitude** and report it as "unbiased by design, coverage 89 to 96 % in simulation at n = 25" (it falls to 81 % at n = 100). Stratified Neyman allocation needs at least 4 labels per stratum, and even then under-covers at large n.

**Limits.** Monte Carlo SE: coverage about 0.5 percentage points, RMSE ratios about 2 %, so ratios between 0.96 and 1.04 are noise. "Recordings saved" interpolates on a 9-point n grid (log-log, curve made monotone). The target is the mean over these 250 sentences; generalizing to new speakers or to real calls adds variance that sentence-level sampling does not see. The uncertainty scores are simple; a score trained to predict human residuals could do better but would need labels of its own.

## Files

- `fidelity.csv`, `fidelity_diff.csv`: section 1
- `weighting.csv`, `weighting_reps.csv.gz`: section 2 (`weighting.csv` now has `share_winner_parakeet` and `share_winner_whisper`)
- `active.csv`, `active_savings.csv`: section 3
- `fig_fidelity.png`, `fig_weighting.png`, `fig_active.png`
- `summary.json`: settings, the full-set reference decisions, and the lab offset (7-voice-mean DiD) per condition with its CI
