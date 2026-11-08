from datetime import timedelta

import numpy as np
import pytest

from ai_trader.core.registry import registered
from ai_trader.core.schemas import ToolResult
from ai_trader.core.time import LookAheadError
from ai_trader.features.transforms import adf_kpss, ewma_sigma_ahead, fracdiff, min_fracdiff_d, psi
from ai_trader.tools.vol import fit_garch_t

from .conftest import T0


def rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_ewma_alignment_has_no_lookahead():
    r = np.full(50, 0.01)
    r[40] = 0.5
    s = ewma_sigma_ahead(r, 0.94, 20)
    assert abs(s[40] - 0.01) < 1e-9
    assert s[41] > 0.1


def test_fracdiff_limits():
    x = np.cumsum(rng().normal(size=500))
    np.testing.assert_allclose(fracdiff(x, 0.0), x)
    np.testing.assert_allclose(fracdiff(x, 1.0)[1:], np.diff(x))


def test_stationarity_verdicts():
    g = rng()
    walk = np.cumsum(g.normal(size=1500))
    noise = g.standard_t(5, 1500)
    assert not adf_kpss(walk).stationary
    assert adf_kpss(noise).stationary
    d = min_fracdiff_d(walk)
    assert d is not None and 0 < d <= 1


def test_psi_detects_drift():
    g = rng()
    a = g.normal(size=5000)
    assert psi(a, g.normal(size=5000)) < 0.05
    assert psi(a, g.normal(1.0, 2.0, size=5000)) > 0.25


def test_toolresult_refuses_lookahead():
    with pytest.raises(LookAheadError):
        ToolResult(tool="x", version="0", asset="A", as_of=T0, data_max_ts=T0 + timedelta(seconds=1), value={})


def test_garch_t_recovers_simulated_parameters():
    omega, alpha, beta, nu, n = 2e-6, 0.08, 0.90, 5.0, 6000
    z = rng(7).standard_t(nu, n) / np.sqrt(nu / (nu - 2))
    h = np.empty(n)
    r = np.empty(n)
    h[0] = omega / (1 - alpha - beta)
    for t in range(n):
        if t:
            h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
        r[t] = np.sqrt(h[t]) * z[t]
    p = fit_garch_t(r)
    assert p["converged"]
    assert abs((p["alpha"] + p["beta"]) - (alpha + beta)) < 0.03
    assert 3 < p["nu"] < 9


def test_registered_tools_are_in_catalogue():
    names = set(registered())
    assert {"vol.ewma", "vol.garch_t"} <= names
