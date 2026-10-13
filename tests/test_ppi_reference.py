"""Cross-check sim2real.stats PPI formulas against the reference implementation ppi_py
(Angelopoulos et al., PyPI ppi-python). Skipped when ppi_py is not installed (it is not a
project dependency).

Run: uv run --with pytest pytest -q tests/test_ppi_reference.py
"""

import numpy as np
import pytest

from sim2real import stats as S

ppi_py = pytest.importorskip("ppi_py")
from ppi_py.ppi import _calc_lam_glm  # noqa: E402


def _data(n=25, N=250, seed=0):
    rng = np.random.default_rng(seed)
    f_all = rng.gamma(1.0, 0.03, size=N)
    y_all = np.clip(f_all * 0.8 + rng.normal(0, 0.02, size=N), 0, None)
    lab = rng.choice(N, size=n, replace=False)
    return y_all, f_all, lab


def test_point_estimate_matches_reference():
    y_all, f_all, lab = _data()
    ours = S.ppi_mean_finite(y_all[lab], f_all[lab], f_all.mean(), len(f_all), method="ppi")
    ref = float(np.squeeze(ppi_py.ppi_mean_pointestimate(y_all[lab], f_all[lab], f_all, lam=1)))
    assert float(ours["estimate"][0]) == pytest.approx(ref, abs=1e-12)


def test_variance_matches_reference_up_to_documented_choices():
    # ppi_py: normal quantile, ddof 0, adds var(f_unlabeled)/N, no finite-population correction.
    y_all, f_all, lab = _data()
    y, f, n, N = y_all[lab], f_all[lab], len(lab), len(f_all)
    lo, hi = ppi_py.ppi_mean_ci(y, f, f_all, alpha=0.05, lam=1)
    from scipy.stats import norm
    se_ref = (float(np.squeeze(hi)) - float(np.squeeze(lo))) / (2 * norm.ppf(0.975))
    se_same_choices = np.sqrt(f_all.var() / N + (y - f).var() / n)
    assert se_ref == pytest.approx(se_same_choices, rel=1e-10)


def test_power_tuning_lambda_matches_reference_before_clipping():
    y_all, f_all, lab = _data()
    y, f, n, N = y_all[lab], f_all[lab], len(lab), len(f_all)
    pe = f_all.mean() + (y - f).mean()
    lam_ref = float(_calc_lam_glm(y - pe, f - pe, f_all - pe, np.eye(1), clip=False))
    r = S.ppi_mean(y, f, f_all, power_tuning=True)
    # ppi_py's covariance uses ddof 0, ours ddof 1
    assert lam_ref == pytest.approx(r["lam"] * (n - 1) / n, rel=1e-10)
