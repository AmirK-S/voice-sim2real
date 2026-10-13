# Literature actions for tonight's follow-ups

*Status (13/10/2026): this is the planning document written before the follow-ups ran. Several items have since been done (BOOT-BOTH, TOST and minimum detectable effect, decision accuracy, ppi_py cross-check, Whisper gate and default-pipeline arms, Silero and WebRTC VAD baselines). What they found, and which claims changed, is in `../../CLAIMS-STATUS.md`; statements below about current results describe the state before those runs.*

The goal is not to reinvent anything. For each follow-up, this file names the published method to use, says exactly how to apply it in this repo, and whether our current approach conflicts with it. Bracketed numbers refer to the reference list in `RELATED-WORK.md`. I read the code paths before writing each item. Two quick checks were run on existing outputs (sections A2 and A5); nothing was re-run or committed. A second pass, written as a completeness review from the speech side, added the items marked *(2nd pass)*, which cite references 106 to 149, and resolved F3.

**Conflict** means: *yes* (our current approach contradicts the published method, or a reviewer would flag it), *partial* (same idea, different choice that matters), *no* (we already do it; only the citation or the name changes).

## Top changes, in priority order

1. **Phantom speech: say that the 33 % was measured with Whisper's gate off, and measure it with the gate on.** `asr._whisper_batch` calls `decode()` directly. It throws away `no_speech_prob`, `avg_logprob` and `compression_ratio`, and applies neither the no-speech gate nor temperature fallback of [69]. The default `mlx_whisper.transcribe` applies both. Barański et al. [71] probably measured their 40.3 % with the gate on, so only the gated number compares with theirs. (A2)
2. **Report each guard as a pair**, following [78]: the hallucination rate on noise, and the cost on speech (false rejection, keyword recall loss). Use the published baselines: Whisper's default gate [69], Silero VAD and WebRTC VAD [71, 81, 134, 135], the released BoH list plus delooping [71], and openai-whisper's `hallucination_silence_threshold`, which [71] tested and which the installed `mlx_whisper.transcribe` also exposes. Report phantom keywords as false alarms per hour of non-speech [133]. (A3 to A6, A4b)
3. **Treat voices as a sampled factor** when averaging them: resample voices and speakers together (BOOT-BOTH [28]). This is the direct statistical fix for "averaging manufactures certainty", which is Clark's language-as-fixed-effect fallacy [106] applied to voices. Resample engines, then voices within engines: the seven voices come from four engines. Reword the claim as "overconfident" until an equivalence test on the human gap [118] says real speech has no winner. (E5, E7)
4. **Replace inverse-MSE fidelity weights and the per-voice linear correction** with variance-optimal voice weights inside PPI [35, 44, 45], cross-fitted [36] and ridge-shrunk [38]. Test whether the weights transfer from LibriSpeech to FLEURS. (C1 to C4)
5. **Active selection: follow Zrnic and Candes [52] and Li et al. [53] on the uniform mixture.** Use fewer and larger strata [42, 43], and keep random PPI as the default. (B2 to B5)
6. **Decisions: add pairwise decision accuracy [26] and a Type I / Type II table [27].** Report Kendall tau only descriptively, because with 3 models it takes four values [29]. (E1 to E4)
7. **TTS rendering errors: add the utterance-level false-alarm rate of Lau et al. [58] and CrossASR's three-way verdict [59],** plus a listening audit that has a full denominator, because ASR language models mask misreadings [60]. (G1 to G4)
8. **Use the published names for our estimators** (difference estimator, regression/GREG estimator, cross-PPI, active inference), and report rho^2 and the effective-sample-size factor [39, 41]. (D1 to D3)
9. *(2nd pass)* **Back every null decision with a minimum detectable effect** [117, 110], call decision flips a swap rate [116], and add MMRV [120] next to SRCC. (E3, E4, E7)
10. *(2nd pass)* **Test the mechanism of the TTS x channel interaction** with glimpse proportion per voice and sentence [138] before calling it acoustic. No ASR re-run is needed. (J1)

---

## A. Phantom speech rate and guards

Code today: `scripts/04_run_experiment.py::noise_only` (100 clips of 4 s noise at -36 dB after the phone line, four noise groups), `src/sim2real/asr.py::_whisper_batch`, `results/noise_only.csv`.

### A1. Hallucination metric on non-speech
- **Paper:** Calm-Whisper [72]; Barański et al. [71].
- **Method:** a non-speech clip is hallucinated if its normalized transcript is non-empty. Report it per model and per noise type with a binomial CI.
- **Apply:** already `bool(asr.normalize(h))` in `noise_only`. Add CIs with the existing `stats.clopper_pearson` (or Wilson). Wilson 95 % gives Whisper 33/100 [24.6, 42.7], wav2vec2 21/100 [14.2, 30.0], Parakeet 2/100 [0.6, 7.0]. With 100 clips the noise-type breakdown (25 each) is descriptive only.
- **Conflict:** no.

### A2. Whisper's own gate as the first baseline
- **Paper:** Radford et al. [69], section 4.5.
- **Method:** a segment is skipped when no_speech_prob > 0.6 and avg_logprob <= -1.0. When compression ratio > 2.4 or avg_logprob < -1.0, the segment is re-decoded at a higher temperature (fallback). In the installed `mlx_whisper/transcribe.py` the defaults are `no_speech_threshold=0.6`, `logprob_threshold=-1.0`, `compression_ratio_threshold=2.4`, and the skip rule is at lines 301 to 306.
- **Apply:**
  1. In `asr._whisper_batch`, keep `r.no_speech_prob`, `r.avg_logprob` and `r.compression_ratio` from each `DecodingResult` (fields exist in `mlx_whisper/decoding.py`), for example behind a `return_meta=True` flag. The text cache stores text only, so meta needs its own cache file or a re-decode. The noise-only run is 100 clips, about 80 s at 1.3 clips/s.
  2. Write the three columns into `noise_only.csv` and apply the gate post hoc.
  3. Add a second arm that runs the same 100 clips through `mlx_whisper.transcribe` with default settings and `temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)`. That is what an off-the-shelf user gets, fallback included. Fallback cannot be simulated post hoc.
  4. For the cost side, log the same three values on 250 human LibriSpeech clips under phone+noise10 (about 3 min) and on the caller set under realistic.
- **Conflict: yes.** The headline "Whisper produced text on 33 of 100 noise clips" was measured with the gate and fallback off, and FINDINGS does not say so. Label the 33 % "raw decoder, no gate" and report the default-pipeline number next to it. Also, `_whisper_long` (clips over 30 s) does use the default gate, so long and short clips are decoded differently. This is minor, since few clips exceed 30 s.
- *(2nd pass)* **Comparability with [71]:** Barański et al. decode greedily at temperature 0 and leave other parameters at their defaults. The paper does not name its code path, but openai-whisper's `transcribe()` defaults include the no-speech gate, so their 40.3 % was probably gated. Compare our gated arm with theirs, and our raw-decoder 33 % with Calm-Whisper [72], which scores any non-empty output.

### A3. Threshold sweep instead of a single gate
- **Papers:** Jasiński et al. [76] (logistic regression on avg_logprob, compression ratio and no_speech_prob: F1 23.6 % on hallucinations inside real speech); Aparin et al. [77] (argue that confident phantom text passes the gate).
- **Method:** treat noise-only clips as positives and speech clips as negatives. Sweep no_speech_prob and avg_logprob, and report the ROC and precision-recall curves.
- **Apply:** a new analysis function (for example `scripts/09_phantom.py`) that reads the meta columns from A2. If a threshold is chosen, choose it on one part of the noise sources and report on the rest, so no source clip is in both.
- **Conflict:** no. This is new work, and [76] tested only hallucinations within speech, so the noise-only result fills a stated gap.

### A4. VAD guards
- **Papers:** Barański et al. [71] (Silero VAD best: padded-speech WER 8.0 against 104.8 unprocessed; WebRTC 68.3); McKinnon et al. [81] (Silero better than WebRTC; window size and hysteresis matter); Calm-Whisper [72] (cites Silero v5 at only 61 % utterance-level accuracy on ESC-50, which is our noise corpus).
- **Method:** run the VAD before ASR. If no speech region is found, return an empty transcript.
- **Apply:** a function in a new `src/sim2real/guards.py` taking the post-channel 16 kHz array. Silero comes from the `silero-vad` package; sweep its threshold and minimum speech duration. WebRTC VAD comes from `webrtcvad`; run it at 8 kHz natively at aggressiveness 0 to 3. Both are new dependencies, so add them with `uv add` only after Amir agrees. Report VAD false triggers on the 100 noise clips, and speech clips rejected or truncated on human LibriSpeech and on the caller set.
- **Conflict:** no (no VAD today). Measure VAD reliability on our narrowband, post-codec noise rather than assuming it, given the ESC-50 figure in [72].
- *(2nd pass)* **Citations and packages:** Silero VAD [134] (PyPI `silero-vad`; 8 and 16 kHz; requires torch, which is already in `.venv`, or onnxruntime, also present) and py-webrtcvad [135] (PyPI `webrtcvad`). The Silero model weights ship with the package or are downloaded on first use; check which before running, since a download needs Amir's agreement like the packages.

### A4b. *(2nd pass)* Whisper's silence-skipping heuristic
- **Paper:** Barański et al. [71] test openai-whisper's `hallucination_silence_threshold` (20 s) as a mitigation.
- **Method:** with word timestamps on, when a segment looks like a hallucination, silent stretches longer than the threshold are skipped.
- **Apply:** the installed `mlx_whisper/transcribe.py` exposes `hallucination_silence_threshold` (line 77, used at line 432). Our noise clips last 4 s, so the published 20 s never triggers; sweep 1 to 3 s (my choice, not theirs) with `word_timestamps=True`, as one more arm of A6.
- **Conflict:** no (new arm).

### A5. Bag of hallucinations and delooping
- **Paper:** Barański et al. [71]. The list is released under MIT at https://github.com/DSP-AGH/ICASSP2025_Whisper_Hallucination (`BoH.csv`, `hallucination_list.csv`).
- **Method:** remove known hallucinated phrases after decoding, and collapse repeated n-grams (delooping).
- **Check I ran:** `BoH.csv` has 294 entries. Its top entries are "thanks for watching" and "thank you for watching", and it includes "thank you very much", but **not a bare "thank you"**. Probably their n-gram logprob < -10 criterion excludes common phrases. Matched exactly against our 33 Whisper noise outputs (normalized), it catches **1**. Delooping (collapsing repeated n-grams) would catch the 223-word "ha ha ha ..." runaway, but not the single very long token "weeee...", which needs a character-repetition rule.
- **Apply:** `guards.boh_filter(hyp_norm, boh)` and `guards.deloop(hyp)`. Normalize BoH entries the same way as our hypotheses: their entries replace apostrophes with spaces ("i m not sure"). Report the released list as-is (1/33) and delooping separately.
- **Conflict: partial.** Adopting BoH "as published" does not remove our dominant phantom phrase. Do not add "thank you" by hand without measuring the cost on speech: two benign caller scripts contain it ("I'm fine, thank you.", "No thank you, I'm not hurt.", in `data.py`). If a frequency list is built from our own noise outputs, split it by noise source clip (build on one part, test on the other).

### A6. Report the trade-off, not one side
- **Paper:** Abbasihafshejani and Jadliwala [78]. They report, for every guard, the hallucination rate on non-speech, the false rejection rate and the WER cost on real speech.
- **Apply:** one table per guard (none, gate, gate plus fallback, Silero, WebRTC, BoH plus delooping, Silero plus BoH). Columns:
  - phantom rate on the 100 noise clips;
  - phantom clips that contain an emergency keyword (use `stats._contains` with `data.KEYWORDS` on the noise outputs; for an emergency line, this is the false alarm that matters), also expressed as false alarms per hour of non-speech, the keyword-spotting unit [133] *(2nd pass)*. With 100 x 4 s = 0.11 h of noise, one false alarm is about 9 per hour, so state the exposure next to the rate;
  - keyword recall on the caller set under realistic;
  - WER on human LibriSpeech under phone+noise10.
- **Conflict:** no. On an emergency line, a dropped "help" costs more than a phantom "thank you", so recall loss is the primary cost.

### A7. Fluent phantom text vs garble
- **Paper:** Sayed et al. [82]. CTC and transducer models fail as garble; Whisper fails as fluent invented text.
- **Apply:** for each phantom output, compute the share of tokens found in a word list (the reference vocabulary of LibriSpeech plus the caller scripts). wav2vec2's 21 outputs ("aca", "baa", ...) are mostly garble; Whisper's are fluent. Only fluent outputs would pass a downstream intent or keyword layer.
- **Conflict:** no.

### A8. Hallucination metric on speech
- **Papers:** Atwany et al. [73] (HER, labelled by an LLM judge); SHALLOW [79]; HALAS [75].
- **Apply:** on noise-only clips every output is a hallucination, so HER is not needed there. HER needs an LLM judge, which costs API calls; do not use it without Amir's approval. To validate any hallucination detector on real speech, use HALAS (`MatBar99/HALAS` on Hugging Face), which has labels for Whisper large-v3-turbo and Parakeet-TDT v2.
- **Conflict:** no.

### A8b. *(2nd pass)* Phantom rate by noise type
- **Paper:** Whisper-AT [131]: Whisper's representation is correlated with the noise type, and it recognizes speech conditioned on it.
- **Apply:** keep the per-group breakdown (25 clips per group) descriptive, and cite [131] as the reason to expect the rate to depend on the noise, so a 4-group, 100-clip result is not generalized to other noise.
- **Conflict:** no.

### A9. Wording of the Parakeet result
- **Paper:** HALAS [75]: on real speech, Parakeet-TDT v2 hallucinates in 33.7 % of transcripts, against 29.4 % for large-v3-turbo.
- **Apply:** in FINDINGS section 6 and the README, write "Parakeet is robust to non-speech (2/100)", not "hallucinates less".
- **Conflict:** partial (wording).

---

## B. Active selection of which sentences to record

Code today: `src/sim2real/calibrate.py` (`poisson_probs`, `poisson_ipw_ppi`, `make_strata`, `neyman_allocation`, `stratified_ppi`, `srs_ppi`) and `scripts/06_calibration.py::q3`. Results: `results/followups/calibration/NOTES.md` section 3.

### B1. The estimator
- **Paper:** Zrnic and Candes, Active Statistical Inference [52].
- **Method:** theta = (1/N) sum_i [f_i + (Y_i - f_i) xi_i / pi_i], with xi_i ~ Bernoulli(pi_i) and known pi_i, and the Horvitz-Thompson variance.
- **Apply:** this is exactly `poisson_ipw_ppi`. Rename it in the docstring and the notes as "active inference estimator [52]" and cite [57] (Active Testing) as the evaluation precursor.
- **Conflict:** no.

### B2. The sampling rule and the uniform mixture
- **Papers:** [52] (oracle pi proportional to sqrt(E[(Y - f)^2 | x]), mixed with uniform, pi_tau = (1 - tau) pi + tau n/N; sequential default tau = 0.5, batch tau tuned on historical data); Li, Zrnic and Candes [53] (robust interpolation between uniform and active, never worse than uniform).
- **Apply:**
  - Our scores are already on the RMS scale: "disagreement" is the SD across voices, "magnitude" is the RMS across voices. So the functional form matches the oracle's square-root scale.
  - The difference is the mixture. `poisson_probs(mix=0.2)` gives a minimum inclusion probability of 0.2 x 25/250 = 0.02, a weight of 50, on sentences where all voices are perfect. A rare human error there produces the heavy-tailed estimates and the 83 to 90 % coverage we observed.
  - Add `mix` to the grid in `q3` (0.2, 0.5, 0.8) and report coverage and RMSE for each. 0.5 is the published default.
  - Optionally implement the robust choice of [53].
  - A label-free way to tune tau (my suggestion, not from the papers): run the design on synthetic data alone, treating one voice as "human" and the other six as the proxy, leave one voice out.
- **Conflict: partial** (mixture weight 0.2 against a default of 0.5).

### B3. Intervals at n of about 25
- **Papers:** [52] (asymptotic normal intervals); Park et al., R-AutoEval+ [48] (finite-sample guarantee, falls back to real data only when the proxy is poor).
- **Apply:** keep RMSE-based "recordings saved" as the primary measure of benefit (already in `active_savings.csv`). The reported-width gain is not real at these n. If an active design is ever recommended, attach its simulated coverage.
- **Conflict:** no. Our "reported CI shrinks, actual error does not" is the known failure mode [53] in a cold start. Cite it as a replication.

### B4. Stratified design
- **Papers:** StratPPI, Fisch et al. [42] (per-stratum lambda; few strata at small n: K = 2 in their simulations; Neyman allocation proportional to w_k times the stratum residual SD); Mozer, Pashley and Miratrix [43] (design-based exact variance; oversample strata with larger residual variance).
- **Current:** `make_strata(u, H=5)`, `neyman_allocation(min_per=2)` with S_h proxied by the *mean* of u in the stratum, lambda = 1 per stratum, t with n - H df. This gave the worst coverage (down to 83 %).
- **Apply:**
  1. Use H = 2 or 3 and `min_per = 5`, so that n = 25 never leaves 2 labels in a stratum.
  2. Proxy S_h by the RMS of u within the stratum (sqrt(mean(u^2))), which tracks a residual SD better than the mean.
  3. Keep lambda = 1 per stratum below about 10 labels per stratum (see D5).
  4. The variance formula in `stratified_ppi` already matches the design-based variance of [43].
- **Conflict: yes** (stratum count and minimum per stratum), partial (S_h proxy).

### B5. Baseline and expectations
- **Papers:** Sfyraki and Wang [54] (the optimal query probability drifts towards uniform at the budget); Angelopoulos et al. [56] (fixed-rate policies are robust; active policies need accurate uncertainty, which is hard in a cold start); Gligoric et al. [55] (gains with hundreds of labels).
- **Apply:** keep "random, PPI" as the reference and the recommended default, as NOTES already concludes. Present the negative result as agreeing with [54, 56].
- **Conflict:** no.

### B6. Who makes the recordings
- **Papers:** Kluger et al. [49] (PPI under cluster designs, bootstrap); Liu and Peng [32]; Miller [30].
- **Apply:** the simulation samples sentences from 40 readers. The practical advice in FINDINGS section 8 ("record about 25 of the scripts with real people") is valid only if the recorders are a random sample of the target callers. If one person records all 25 scripts, the interval covers that person's WER, not callers'. State the number of speakers needed. If a speaker-level target is wanted, sample speakers first and use a cluster bootstrap [49].
- **Conflict: partial** (scope of the claim).

---

## C. Fidelity-weighted and linearly corrected voice ensembles

Code today: `calibrate.fidelity_weights`, `calibrate.linear_corrected`, `calibrate.two_sample_rectified`, `calibrate.ppipp_lambda`; `scripts/06_calibration.py::q2`.

### C1. Weight voices by the variance of the rectified estimator
- **Papers:** PPI by mixture of experts [45] (beta minimizes Var(F_beta - Y), with a best-expert guarantee for oracle weights); vector-lambda PPI++ [35]; MultiPPI [44].
- **Method:** with F the (7 x n) matrix of per-voice scores on the labelled sentences, choose w to minimize the sample variance of y - F^T w. This is the OLS (or ridge) regression of y on the 7 voice columns; the intercept cancels in the difference estimator. Then estimate mean_eval(F^T w) + mean_cal(y - F^T w), with `two_sample_rectified` using f = F^T w.
- **Current:** `fidelity_weights` uses the inverse MSE between each voice's difference and the human difference. That rewards closeness to the human value, not low rectifier variance. The weighted average is then used *without* a rectifier, so its bias stays and its CI stays narrow (37 to 100 % false winners in NOTES).
- **Conflict: yes.** Replace it with "fidelity-weighted PPI" as defined above, and compare it with the 7-voice mean PPI and the best single voice PPI. Test whether estimated weights keep the best-expert property at n = 25.
- *(2nd pass)* **Nesting:** the two Kokoro voices come from one model, and Piper and MMS are all VITS models [145, 146, 147]. Their columns in F are strongly correlated, which is one more reason for the shrinkage of C3; report the weights per engine as well as per voice.

### C2. Do not fit and evaluate on the same labels
- **Papers:** cross-prediction-powered inference [36]; Eyre and Madras [38] (PPI++ is OLS and has high variance with few labels; use ridge or other robust regressors).
- **Current:** `linear_corrected` fits OLS per voice on the calibration sentences and evaluates on the same fit. Its coverage of 74 to 90 % is the expected optimism of a regression estimator that reuses its labels.
- **Apply:** fit the C1 weights with K-fold cross-fitting over the calibration sentences (by speaker if possible): fit on K - 1 folds, rectify on the held-out fold, average. Use a ridge penalty. Choosing it by leave-one-out on the calibration set is my suggestion, not from [38]. C1 with cross-fitting supersedes `linear_corrected`. Keep the old method only as a documented negative result.
- **Conflict: yes.**

### C3. Shrinkage at n = 10 to 25
- **Paper:** Eyre and Madras [38] (Ridge-PPI, Sigmoid-PPI).
- **Apply:** with 7 weights and 25 labels, unregularized OLS is ill-posed. Shrink towards the equal-weight vector (the 7-voice mean), not towards zero, so that lambda tends to the plain-mean PPI as the penalty grows.
- **Conflict:** partial (no shrinkage today).

### C4. Weights learned on independent data
- **Papers:** [34] (PPI is valid for any predictor fixed independently of the labels); FAB-PPI [47] (a prior on predictor quality).
- **Apply:** fit the voice weights on LibriSpeech using all 250 human labels, then apply them unchanged to FLEURS with n FLEURS labels. Plain fixed-lambda PPI stays valid because the weights do not depend on FLEURS labels. Report whether the weights transfer across corpora and conditions (clean vs noise). I found no published test of this, so it is the most original item in this follow-up.
- **Conflict:** no (new).

### C5. An independent fidelity score, before fitting
- **Papers:** TTSDS2 [10] (distribution distance to real speech, validated against human ratings); Rossenbach et al. [11] (MOS-style scores did not predict ASR usefulness).
- **Apply:** compute TTSDS2 per voice against our human recordings, then test whether it predicts each voice's gap scale (0.37x to 3.72x) and its calibration slope (`fidelity.csv`). Do this before fitting weights on the same human data, to avoid circularity. Do not use UTMOS or NISQA as fidelity weights. The TTSDS2 code release was not checked here; confirm it before planning on it.
- **Conflict:** no.

### C6. Speaker clustering in the evaluation set
- **Papers:** Miller [30]; Kluger et al. [49].
- **Apply:** `two_sample_rectified` treats evaluation sentences as independent (NOTES says so and checks coverage by simulation). Add a speaker-level variance: compute residual means per speaker, then their variance. Report coverage with both variances.
- **Conflict:** partial (documented).

---

## D. PPI core (`src/sim2real/stats.py`)

### D1. Names
- **Papers:** Angelopoulos et al. [34]; Mozer [41]; PPI++ [35]; cross-PPI [36].
- **Apply:** in the `ppi_mean_finite` docstring, FINDINGS section 8 and the README:
  - "ppi" is the difference estimator (PPI with lambda = 1) under simple random sampling without replacement;
  - "ppi++" is the regression (GREG) estimator. Using lambda = cov/var without the (1 + n/N) factor is correct here, because f is known for all N sentences, so its mean carries no sampling error;
  - "ppi++cf" is a two-fold cross-PPI [36] (the paper uses K folds).
- **Conflict:** no (math is right; names change).

### D2. Ratio estimand
- **Paper:** the GREG / ratio-estimator view in [41], and the M-estimation form of PPI++ [35].
- **Check:** corpus WER is sum(errors) / sum(words). Here each sentence's score is errors_i / nbar, with nbar the mean reference word count over the target set. Word counts come from the script and are known for every sentence without recording, so the mean score equals the corpus WER exactly. In `q2`, `k = 1 / D.n[ev].mean()` uses the evaluation set's mean, which is correct for that target.
- **Apply:** state this in FINDINGS section 8, because reviewers who know [35] will ask.
- **Conflict:** no.

### D3. Report what limits the gain
- **Papers:** Dorner et al. [39]; Chaganty et al. [40].
- **Apply:** next to every PPI result (`06_calibration.q1` output, the section 8 table), add:
  - rho;
  - the effective-sample-size factor 1/(1 - rho^2): 1.39 at rho = 0.53, 1.91 at 0.69, 3.21 at 0.83;
  - the optimal width ratio sqrt(1 - rho^2): 0.85, 0.72, 0.56;
  - the lambda = 1 ratio sqrt(1 + (s_f/s_y)^2 - 2 rho s_f/s_y), which equals the optimum only when s_f/s_y = rho.
  The wav2vec2 phone+noise5 case (rho = 0.83, ratio 0.56) sits exactly on the optimum. Do not quote the "2x cap" of [39] as a theorem for WER: it is proved for binary outcomes where the judge is no better than the evaluated model.
- **Conflict:** no.

### D4. Cross-check against the reference implementation
- **Paper:** [34] (`ppi_py`, PyPI package `ppi-python`).
- **Apply:** a test in `tests/` comparing `stats.ppi_mean` (infinite-population form, normal quantile) with `ppi_py.ppi_mean_ci`. Our finite-population version differs only by the factor (1 - n/N) on the labelled-term variance. Adding the dependency needs Amir's agreement.
- **Conflict:** no.

### D5. Small n
- **Papers:** Eyre and Madras [38] (PPI++ can be worse than classical with scarce labels; their evidence is MAE, not coverage); R-AutoEval+ [48]; Bayesian PPI [46] (credible intervals only; no general small-n advantage in its own results).
- **Apply:** keep the rule "classic PPI below about 25 labels" and cite [38] as the reason. Our coverage simulation at n = 10/25/50 adds coverage evidence that [38] lacks.
- **Conflict:** no.

### D6. Paired differences for decisions
- **Papers:** AutoEval [37] (PPI on comparisons, Bonferroni-corrected intervals); Chatzi et al. [33] (prediction-powered rank-sets).
- **Apply:** `q2` already applies PPI to the per-sentence Whisper minus Parakeet difference with the synthetic difference as predictor; this is the published approach. For a final "which of the 3 models wins" statement, Holm intervals over the 3 pairs are enough; rank-sets [33] are optional.
- **Conflict:** no.

---

## E. Rank agreement and decision statistics

Code today: `scripts/05_analyze.py::flips` (writes `results/tables/decision_flips.csv`), `scripts/08_fleurs.py::analyze`.

### E1. Kendall tau, descriptively only
- **Papers:** Perlitz et al. [29]; VoiceBench [3] (tau 0.87, W 0.83); 2M-BELEBELE [1] (adjacent swaps).
- **Apply:** report tau per voice x condition and the number of adjacent swaps, so the numbers can be compared with [1] and [3]. Say that with 3 models tau takes only the values -1, -1/3, 1/3 and 1, so it cannot carry a test. The pairwise DiD with Holm stays the primary analysis.
- **Conflict:** no.

### E2. Pairwise decision accuracy
- **Paper:** Kocmi et al. [26].
- **Method:** the share of system-pair decisions in which the proxy orders the pair as humans do.
- **Apply:** from `decision_flips.csv`, per voice, take all (model pair x condition) rows. Accuracy is the share where the sign of `synth_gap` equals the sign of `human_gap`, computed (a) on all rows and (b) on rows where `human_gap_ci_excludes_0`. Get the CI from the speaker bootstrap already in `flips` (`sd["boot"]` and `hd["boot"]`).
- **Conflict:** no (new column).

### E3. Type I / Type II decision table
- **Paper:** Mathur et al. [27].
- **Method:**
  - Type I: the lab declares a winner (`synth_gap_ci_excludes_0`) where humans do not;
  - Type II: humans have a winner and the lab does not;
  - plus "confident wrong sign".
- **Apply:** a groupby on `decision_flips.csv` by voice (and for `synthetic_mean`). The 7-voice mean on phone, phone+noise10 and realistic gives three Type I errors. This is how the "manufactured certainty" result should be tabulated.
- *(2nd pass)* **Name:** in information retrieval, the share of system pairs ordered differently by two test sets is the swap rate [116]; use that name for our flip counts so readers can place them.
- **Conflict:** no.

### E4. Sim-vs-real correlation across cells
- **Paper:** Kadian et al. [22] (SRCC).
- **Apply:** per voice, take the Pearson or Spearman correlation between lab WER and real WER across (condition x model) cells (up to 3 x 10). Do the same for channel damage, which is the quantity the lab gets wrong. CI by speaker bootstrap. Do not pool the in-domain wav2vec2 control into a cross-system correlation without a version that excludes it (outlier warning of [27]).
- *(2nd pass)* **MMRV:** SIMPLER [120] adds the Mean Maximum Rank Violation, a rank-violation score weighted by the real margin between the systems, because Pearson correlation can miss ranking errors. With 3 models it is cheap: for each model, the largest real margin over the pairs the lab orders wrongly, averaged over models. Report it per voice x condition next to SRCC.
- **Conflict:** no.

### E5. Voices as a sampled factor
- **Paper:** Deutsch et al. [28] (BOOT-BOTH: resample systems and inputs; PERM-BOTH for tests).
- **Current:** `add_synth_mean` averages the 7 voices per sentence, and `flips` bootstraps that mean over speakers only. The voice set is treated as fixed, so voice-to-voice variance never enters the CI. That is the mechanism behind "averaging shrinks the CI, not the bias".
- **Apply:** in `flips`, for `synthetic_mean`, draw in each replicate a multiset of 7 voices with replacement and a speaker resample. Compute the voice-mean gap and the DiD under both. Report the resulting CI and the Type I count (E3). With only 7 voices, the voice variance is itself uncertain; say so. Use PERM-BOTH for "voice A is more faithful than voice B" in `fidelity.csv`.
- *(2nd pass)* **Why, in one citation:** treating a sample of stimuli as fixed is Clark's language-as-fixed-effect fallacy [106]; the standard remedies are random effects for stimuli [107, 108, 109] or resampling them [28].
- *(2nd pass)* **Nesting:** resample at two levels, engines (Kokoro, Piper, MMS, macOS) and then voices within engines, because the two Kokoro voices share one model and Piper and MMS are VITS models. With four engines the voice-level variance is poorly estimated; report the CI from both the flat (7 voices) and the nested resampling.
- *(2nd pass)* **Model-based check (optional):** a Poisson mixed model of errors per sentence with offset log(words), fixed effects for ASR model, source (human or TTS) and their interaction, and crossed random intercepts for sentence and reader, plus a random model effect by voice, as in Liu, Veliche and Peng [115] and Barr et al. [108]. It needs `statsmodels` (not installed) or another GLMM package, so only with Amir's agreement; the bootstrap stays primary.
- **Conflict: yes.** This is the most direct published fix for our headline finding. It turns the finding into a recommendation: when averaging voices, resample them.

### E6. Calibration and distribution fidelity per voice
- **Papers:** Lost in Simulation [19] (ECE-style: difficulty-bin-weighted mean absolute gap between human and simulated outcomes); Williams [23] (Cramer-von Mises divergence with critical values).
- **Apply:** in `06_calibration.q1`, per voice and cell, bin sentences by human per-sentence WER (or by synthetic mean, to avoid conditioning on the outcome), then report the weighted mean of |synthetic - human| and whether TTS is optimistic on hard bins. Compute the CvM divergence between per-utterance WER distributions. Williams' critical values assume independent samples, but ours are paired by sentence, so use a permutation test that swaps the voice and human labels within each sentence.
- **Conflict:** no.

### E7. *(2nd pass)* Null results: equivalence and minimum detectable effects
- **Papers:** Lakens [118] (TOST equivalence tests); Card et al. [117] (power in NLP comparisons); Westfall, Kenny and Judd [110] (power when participants respond to samples of stimuli, here voices).
- **Current:** "real speech shows no winner" rests on human CIs that include 0, for example +0.60 [-0.04, +1.29] points on realistic, in the same direction as the 7-voice mean. "No single voice flips" rests on Holm-adjusted p above 0.16.
- **Apply:**
  1. For each condition, run TOST on the human Whisper minus Parakeet gap with a smallest effect of interest stated in advance (for example 0.5 WER points, my suggestion; Amir should pick it from what would change a deployment decision). Use the speaker-bootstrap 90 % CI: equivalence holds if it lies inside (-delta, +delta).
  2. Report the minimum detectable DiD at 80 % power for the speaker bootstrap, from the bootstrap SE: about 2.8 x SE at alpha 0.05, and larger after Holm.
  3. Reword until then: the 7-voice mean is "more confident than the human data can confirm", not "declares a winner where none exists".
- **Conflict: yes** (wording of the headline result).

---

## F. FLEURS out-of-domain human baseline (`scripts/08_fleurs.py`)

### F1. Report the 2M-BELEBELE statistic
- **Paper:** Costa-jussà et al. [1]. It used FLEURS with MMS-TTS from the FLEURS transcripts and counted adjacent swaps between synthetic and human rankings.
- **Apply:** in `analyze`, add per voice x condition the adjacent-swap count and whether the top system changes. Compare the MMS penalty on Whisper with theirs: they found that Whisper loses more on MMS-TTS audio than its competitor. Our CIs and DiD then show whether their kind of flip survives inference on the same corpus.
- **Conflict:** no.

### F2. Normalized Intelligibility as a descriptive anchor
- **Paper:** Wang et al. [7]: Norm_I = exp((WER_r - WER_s) / WER_r), bounded in (0, e).
- **Apply:** one column per voice x model x condition, on LibriSpeech and on FLEURS, so the synthetic-vs-real ratio is comparable across corpora. It is undefined or unstable when WER_r is near 0 (strong models on clean audio); report it only where WER_r is at least about 2 %, which is my cut-off, not theirs.
- **Conflict:** no.

### F3. Check that FLEURS really is out of domain for each model
- **Apply:** Whisper's paper evaluates on FLEURS, and its training data is undocumented, so FLEURS text or audio leaking into it cannot be ruled out. wav2vec2-base-960h was trained on LibriSpeech only. *(2nd pass)* The Parakeet TDT 0.6B v2 model card [143] lists Granary (about 120,000 h): LibriSpeech 960 h, Fisher, NSC Part 1, VCTK, VoxPopuli, Europarl-ASR, MLS, Common Voice 7.0 and AMI, plus pseudo-labelled YTC, YODAS and Librilight. FLEURS is not listed, and it is not among the card's evaluation sets. Write "out of domain for wav2vec2 and, per its model card, for Parakeet; unknown for Whisper".
- **Conflict:** resolved for Parakeet; open for Whisper.

### F4. No speaker ids
- **Papers:** Liu and Peng [32]; Miller [30].
- **Apply:** FLEURS TSVs have no speaker id, so `analyze` uses a sentence bootstrap and reports the LibriSpeech ratio of speaker CI width to sentence CI width (`ls_speaker_over_sentence_ci_width`). Keep that and say the FLEURS CIs are likely too narrow by about that factor.
- **Conflict:** no (documented).

### F5. Framing
- **Papers:** WildASR [5] ("real source, controlled perturbation"); Kubis et al. [4].
- **Apply:** the human-plus-channel arm on FLEURS follows WildASR's principle; the TTS-plus-channel arm is the one under test. Use [4] to scope claims: TTS can be an adequate proxy for coarse aggregates, and breaks for close comparisons, channel cost and keywords.
- **Conflict:** no.

### F6. *(2nd pass)* A real-acoustics counterpart of LibriSpeech
- **Paper:** VOiCES [123]: LibriSpeech speech re-recorded in real furnished rooms with played-back noise, twelve microphones, about 120 h per microphone, CC BY 4.0.
- **Apply (later, not tonight):** it would let the human-plus-channel arm be checked against real room acoustics on LibriSpeech material, in the spirit of REVERB's SimData vs RealData [121]. Check which LibriSpeech subset it draws from before assuming any overlap with our 250 test-clean sentences; I did not check this. A download needs Amir's agreement.
- **Conflict:** no.

---

## G. Automatic detection of TTS rendering errors (`scripts/07_tts_errors.py`)

### G1. Utterance-level false-alarm rate, comparable to Lau et al.
- **Paper:** Lau et al. [58]: 21 to 34 % of TTS-found failures are false alarms (by TTS: 17 % to 32 %); wav2vec2 has the most.
- **Method:** a failure on TTS audio is a false alarm if the human recording of the same text is transcribed correctly. One reading of the paper adds that at least one other ASR must also fail on the TTS clip. Check the paper's exact definition, and compute both variants.
- **Apply:** a new function in `07_tts_errors.py`. For each voice x ASR on clean LibriSpeech, take the utterances with WER > 0 on TTS, and compute the share whose human twin has WER = 0. Report it next to the word-level R3/R2 counts.
- **Conflict:** no. R3/R2 is the stricter, word-level version.

### G2. CrossASR's three-way verdict for the caller set
- **Paper:** CrossASR / CrossASR++ [59].
- **Method:** success if the system under test is right; failed test case if it is wrong while another ASR is right on the same clip; indeterminable if all ASRs are wrong, because the TTS may be at fault.
- **Apply:** in `keyword_flags` (caller set, no human audio), report per voice the share of indeterminable keyword instances. Then report how many of them our cross-voice rule confirms (at least 4 of 6 other voices recognized).
- **Conflict:** no. Our rule is CrossASR's rule extended with agreement across voices; cite it as such.

### G3. Expect misses; measure them by listening
- **Paper:** Luo and Wan [60]: the ASR round trip exposed only 9 of 55 real reading errors, and recognizers differ greatly in how much they repair.
- **Apply:**
  1. Run a listening audit with a full denominator: every flagged item plus a random, voice-stratified sample of unflagged words and keyword instances. Labels come from listening, ideally by a second person who did not design the flag. This gives precision and an estimate of the miss rate. The current check is 30 items, text only, one labeller.
  2. Priority items: every "help" on macOS Daniel and Samantha (13 scripts each), and every MMS keyword instance.
  3. Span isolation for emergency keywords: synthesize each keyword alone in a short carrier phrase per voice and transcribe it with the three models. A miss in isolation confirms a rendering error.
  4. *(2nd pass)* For the listening protocol, follow listener transcription as in the SUS test [136]: listeners write down what they hear, so that context cannot repair a misreading the way an ASR language model does [60]. A semantically neutral carrier phrase (step 3) serves the same purpose.
- **Conflict: partial.** Present R3/R2 counts as lower bounds for two reasons (the conservative rule and LM masking), not one.

### G4. The voting rule
- **Papers:** ROVER [63] (word-level voting); Yu and Kang [61] (vote across families; rank averaging and conjunctive max-rank).
- **Apply:** our three ASRs are three families (encoder-decoder, TDT transducer, CTC), as [61] recommends. Add a 2-of-3 rule next to R3 (all 3) and R2 (both strong models), and report precision against the G3 labels for each rule.
- **Conflict:** no.

### G5. Optional
- Lau et al.'s text-only false-alarm estimator (https://github.com/julianyonghao/FAinASRtest; its training data is hosted externally) could serve as a baseline on the caller set. It is not needed tonight.
- The Rogan-Gladen correction of keyword miss rates [51] assumes that the proxy's sensitivity does not change between LibriSpeech and caller scripts, which is doubtful.
- Plain Dawid-Skene [66] is not appropriate, because our raters are correlated. If a latent-label model is wanted, use a correlation-aware one [67].

---

## H. Later, not tonight

- **Channel reference tools [84]:** unit-test `channel.py` against ITU-T STL. Use sv56 for active level and SNR, the IRS and G.712 filters, G.711 Appendix I concealment and G.722 concealment, in `scripts/02_channel_selftest.py`.
- **Real loss patterns [92]:** replace i.i.d. 5 % and our Gilbert-Elliott parameters with PLC Challenge traces, and report results binned by maximum burst length (up to 120, 120 to 320, 320 to 1000 ms).
- **Concealment vs zero-fill arms [90, 91]:** frozen ASRs were not trained on concealed speech.
- **Cellular codecs [89, 94]:** add AMR-NB and GSM-FR.
- **Held-out noise and filter, Aurora style [83].**
- **A real transmission:** a PBX route with real codecs, as in ASVspoof 2021 [94], or paired corpora [85, 86, 87].
- **Keyword metrics:** report keyword error rate next to WER [98], and publish the keyword list and normalizer as a pinned, versioned artifact [100]. Matching is already phrase-aware (`stats._contains`).

## I. Documentation fixes found while reading the code

- **FINDINGS, setup:** it says the DiD is "bootstrapped per sentence", but `05_analyze.flips` uses the speaker-cluster weights `L.W`. The code is right; fix the text.
- **FINDINGS section 4:** the Holm family wording does not match the code; already noted in `tts_errors/NOTES.md`.
- **FINDINGS section 5:** "its rendering errors hurt Whisper more" is not tested by a shared-error filter (`tts_errors/NOTES.md` section 5). It is consistent with 2M-BELEBELE [1], which can be cited as outside support for the direction.
- **FINDINGS section 6 and README:** state the Whisper decoding settings (no gate, no fallback) next to the 33 %.
- **FINDINGS section 8 and README:** "record about 25 of the scripts with real people" needs the number of speakers (B6).
- *(2nd pass)* **README TL;DR and FINDINGS section 1:** "averaging voices made the lab confident where real speech shows no winner" should read "more confident than the human data can confirm" until E7 is run.
- *(2nd pass)* **README TL;DR:** "from 0.37x to 3.72x the real gap" needs FINDINGS' own caveat: these pairs all involve the in-domain wav2vec2 control, and the range is a descriptive extreme over many cells.
- *(2nd pass)* **FINDINGS setup:** say that the 7 voices come from 4 engines (one Kokoro model, three VITS models, macOS).
- *(2nd pass)* **FINDINGS limitations:** "FLEURS ... not authorized for download in this run" is now outdated if the FLEURS run completes; F3 gives the per-model out-of-domain status.

## J. *(2nd pass)* Mechanism checks for the TTS x channel interaction (FINDINGS section 3)

### J1. Glimpse proportion
- **Paper:** Cooke [138]: intelligibility in noise follows the proportion of spectro-temporal regions where speech exceeds the noise by a local margin (3 dB in the paper's main model).
- **Apply:** with `channel.add_noise(..., return_noise=True)`, compute per voice and sentence the glimpse proportion on the post-line signal (a gammatone or mel filterbank, 10 ms frames). Then regress the per-sentence channel damage on the glimpse-proportion difference between voice and human. If Kokoro's higher glimpse proportion at the same active-level SNR explains its smaller damage, the interaction is acoustic, as FINDINGS proposes. If not, the explanation stays open. No ASR re-run is needed.
- **Conflict:** no (new analysis). It turns the "likely reason" paragraph of section 3 into a test.

### J2. Context for the direction
- **Papers:** Yang et al. [12] (some modern TTS more intelligible in noise than human speech, for human listeners); Pisoni, Nusbaum and Greene [137] (rule-based synthetic speech less intelligible than natural speech).
- **Apply:** cite both when presenting the Kokoro result, so the claim reads "the direction depends on the voice and the TTS generation", which is what our five other voices show.
- **Conflict:** no.
