"""Tests for sim2real.stats with synthetic data (no audio, no models).

Run: uv run --with pytest pytest -q tests/test_stats.py
"""

import numpy as np
import pytest

from sim2real import stats as S


# --- text scoring -----------------------------------------------------------


def test_wer_basic():
    assert S.utterance_wer("Help, he is not breathing.", "help he is not breathing") == 0
    assert S.utterance_wer("help he is not breathing", "hell he is not breathing") == pytest.approx(0.2)
    assert S.wer_counts("a b c", "") == (3, 3)
    assert S.corpus_wer(["a b", "c d e f"], ["a x", "c d e f"]) == pytest.approx(1 / 6)


def test_keyword_recall_whole_words_and_phrases():
    refs = ["Help, he is not breathing", "There is a fire and a gun", "fine thanks"]
    hyps = ["Hell he is not breathing", "there is a fire and a gum", "fine thanks"]
    kr = S.keyword_recall(refs, hyps)
    assert kr["per_keyword"]["help"] == (0, 1)
    assert kr["per_keyword"]["not breathing"] == (1, 1)
    assert kr["per_keyword"]["fire"] == (1, 1)
    assert kr["per_keyword"]["gun"] == (0, 1)
    assert kr["total"] == 4 and kr["hits"] == 2
    # "helpful" must not count as "help"
    assert "help" not in S.keyword_hits("that was helpful", "that was helpful")


# --- paired bootstrap -------------------------------------------------------


def test_paired_bootstrap_coverage():
    """95% CI should contain the true difference roughly 95% of the time."""
    true_diff, hits, reps = 0.03, 0, 300
    for seed in range(reps):
        rng = np.random.default_rng(1000 + seed)
        difficulty = rng.gamma(2, 0.08, size=150)  # shared per sentence
        a = difficulty + rng.normal(true_diff, 0.05, 150)
        b = difficulty + rng.normal(0, 0.05, 150)
        r = S.paired_bootstrap_ci(a, b, n=2000, seed=seed)
        hits += r["lo"] <= true_diff <= r["hi"]
    assert 0.90 <= hits / reps <= 0.99


def test_paired_bootstrap_detects_clear_gap_and_is_repeatable():
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 0.3, 200)
    a, b = base + 0.05, base
    r1 = S.paired_bootstrap_ci(a, b, seed=7)
    r2 = S.paired_bootstrap_ci(a, b, seed=7)
    assert r1 == r2
    assert r1["lo"] > 0 and r1["p_a_better"] == 0


# --- McNemar, rule of three, Clopper-Pearson --------------------------------


def test_mcnemar():
    assert S.mcnemar(0, 0)["p_value"] == 1.0
    assert S.mcnemar(5, 5)["p_value"] == pytest.approx(1.0)
    # 0 vs 10 discordant: p = 2 * 0.5**10
    assert S.mcnemar(0, 10)["p_value"] == pytest.approx(2 * 0.5**10)
    assert S.mcnemar(3, 12)["p_value"] < 0.05


def test_rule_of_three_and_exact_bound():
    assert S.rule_of_three(300) == pytest.approx(0.01)
    ub = S.clopper_pearson_upper_zero(300)
    assert (1 - ub) ** 300 == pytest.approx(0.05)
    assert abs(ub - S.rule_of_three(300)) < 0.0005
    lo, hi = S.clopper_pearson(0, 300, alpha=0.10)  # two-sided 90% = one-sided 95%
    assert lo == 0 and hi == pytest.approx(ub)


def test_clopper_pearson_coverage_conservative():
    rng = np.random.default_rng(3)
    p, n, reps = 0.1, 60, 1000
    ks = rng.binomial(n, p, reps)
    cover = np.mean([lo <= p <= hi for lo, hi in (S.clopper_pearson(int(k), n) for k in ks)])
    assert cover >= 0.94  # exact interval is conservative


# --- decision flip ----------------------------------------------------------


def _scores(rng, n, mean_a, mean_b, difficulty):
    a = np.clip(difficulty + mean_a + rng.normal(0, 0.03, n), 0, None)
    b = np.clip(difficulty + mean_b + rng.normal(0, 0.03, n), 0, None)
    return {"A": a, "B": b}


def test_decision_flip_detects_flipped_voice():
    rng = np.random.default_rng(0)
    n = 200
    diff = rng.gamma(2, 0.05, n)
    data = {
        "human": _scores(rng, n, 0.10, 0.15, diff),  # A wins in reality
        "voice_same": _scores(rng, n, 0.05, 0.09, diff),  # A wins in the lab too
        "voice_flip": _scores(rng, n, 0.12, 0.06, diff),  # lab picks B
    }
    r = S.decision_flip(data, n=2000, seed=1)
    assert r["human"]["winner"] == "A"
    assert r["human"]["p_winner_stable"] > 0.99
    assert not r["conditions"]["voice_same"]["flip"]
    assert r["conditions"]["voice_same"]["p_flip"] < 0.01
    assert r["conditions"]["voice_flip"]["flip"]
    assert r["conditions"]["voice_flip"]["p_flip"] > 0.99
    assert r["conditions"]["voice_flip"]["paired_with_human"]


def test_decision_flip_near_tie_is_uncertain():
    rng = np.random.default_rng(5)
    n = 100
    diff = rng.gamma(2, 0.05, n)
    data = {"human": _scores(rng, n, 0.10, 0.10, diff), "tts": _scores(rng, n, 0.10, 0.10, diff)}
    r = S.decision_flip(data, n=2000, seed=0)
    assert 0.1 < r["conditions"]["tts"]["p_flip"] < 0.9


# --- PPI --------------------------------------------------------------------


def _ppi_world(rng, n_lab, n_unl, bias=0.04, noise=0.03):
    """Per-sentence human WER y and a biased, correlated synthetic WER f."""
    total = n_lab + n_unl
    difficulty = rng.gamma(2, 0.06, total)
    y = difficulty + rng.normal(0, noise, total)
    f = 0.8 * difficulty - bias + rng.normal(0, noise, total)  # synth is "easier"
    return y[:n_lab], f[:n_lab], f[n_lab:]


def test_ppi_coverage_and_narrower_than_naive():
    true_mean = 2 * 0.06  # E[difficulty] + 0
    reps, hits, hits_naive, ratios = 400, 0, 0, []
    for seed in range(reps):
        rng = np.random.default_rng(seed)
        y, f, fu = _ppi_world(rng, 40, 600)
        r = S.ppi_mean(y, f, fu)
        hits += r["lo"] <= true_mean <= r["hi"]
        hits_naive += r["naive"]["lo"] <= true_mean <= r["naive"]["hi"]
        ratios.append(r["width_ratio"])
    assert 0.91 <= hits / reps <= 0.98
    assert 0.90 <= hits_naive / reps <= 0.98
    assert np.median(ratios) < 0.8  # PPI is clearly tighter here


def test_ppi_synth_only_is_biased_but_ppi_is_not():
    rng = np.random.default_rng(42)
    y, f, fu = _ppi_world(rng, 50, 5000, bias=0.05)
    r = S.ppi_mean(y, f, fu)
    true_mean = 0.12
    assert not (r["synth_only"]["lo"] <= true_mean <= r["synth_only"]["hi"])
    assert r["lo"] <= true_mean <= r["hi"]


def test_ppi_power_tuning_with_useless_proxy():
    """If synthetic scores are unrelated noise, PPI++ falls back near naive."""
    reps, hits, ratios = 300, 0, []
    for seed in range(reps):
        rng = np.random.default_rng(10_000 + seed)
        y = rng.normal(0.2, 0.05, 30)
        f = rng.normal(0.1, 0.2, 30)
        fu = rng.normal(0.1, 0.2, 500)
        r = S.ppi_mean(y, f, fu, power_tuning=True)
        hits += r["lo"] <= 0.2 <= r["hi"]
        ratios.append(r["width_ratio"])
        assert 0 <= r["lam"] <= 1
    assert hits / reps >= 0.90
    assert np.median(ratios) < 1.1
