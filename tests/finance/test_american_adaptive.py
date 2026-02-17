"""Tests for adaptive-grid American option pricing."""

from __future__ import annotations

import numpy as np
import pytest

from chernoffpy import CrankNicolson
from chernoffpy.finance.adaptive_grid import estimate_free_boundary, make_sinh_grid
from chernoffpy.finance.american import AmericanPricer
from chernoffpy.finance.american_analytical import american_binomial
from chernoffpy.finance.validation import DividendSchedule, GridConfig, MarketParams


@pytest.fixture
def adaptive_grid_cfg():
    return GridConfig(N=1024, L=8.0, taper_width=2.0)


class TestSinhGrid:

    def test_grid_contains_center(self):
        center = -0.25
        x = make_sinh_grid(N=1024, L=8.0, center=center, alpha=0.2)
        idx = int(np.argmin(np.abs(x - center)))
        assert abs(x[idx] - center) < 0.03

    def test_grid_symmetric_around_center0(self):
        x = make_sinh_grid(N=1024, L=8.0, center=0.0, alpha=0.2)
        assert np.max(np.abs(x + x[::-1])) < 2e-12

    def test_grid_denser_at_center(self):
        x = make_sinh_grid(N=1024, L=8.0, center=-0.2, alpha=0.2)
        dx = np.diff(x)
        i_mid = len(dx) // 2
        assert dx[i_mid] < dx[10]
        assert dx[i_mid] < dx[-10]

    def test_grid_density_ratio(self):
        x = make_sinh_grid(N=2048, L=8.0, center=-0.2, alpha=0.2)
        dx = np.diff(x)
        ratio = float(dx[5] / dx[len(dx) // 2])
        assert ratio > 1.015

    def test_grid_spans_domain(self):
        L = 8.0
        x = make_sinh_grid(N=512, L=L, center=-0.3, alpha=0.2)
        assert pytest.approx(-L, rel=0.0, abs=1e-12) == x[0]
        assert pytest.approx(L, rel=0.0, abs=1e-12) == x[-1]


class TestFreeBoundaryEstimate:

    def test_put_fb_negative(self):
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        assert estimate_free_boundary(m, "put") < 0.0

    def test_put_fb_moves_with_rate(self):
        low_r = MarketParams(S=100, K=100, T=1.0, r=0.02, sigma=0.2)
        high_r = MarketParams(S=100, K=100, T=1.0, r=0.10, sigma=0.2)
        assert estimate_free_boundary(high_r, "put") > estimate_free_boundary(low_r, "put")

    def test_put_fb_moves_with_vol(self):
        low_v = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.1)
        high_v = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.4)
        assert estimate_free_boundary(high_v, "put") < estimate_free_boundary(low_v, "put")

    def test_call_no_dividend_no_exercise(self):
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0)
        assert estimate_free_boundary(m, "call") > 0.5


class TestAdaptiveAmericanPricing:

    def test_adaptive_more_accurate_than_uniform(self, adaptive_grid_cfg):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        ref = american_binomial(100, 100, 0.05, 0.2, 1.0, "put", n_steps=50000)

        uniform = AmericanPricer(CrankNicolson(), adaptive_grid_cfg, adaptive=False)
        adaptive = AmericanPricer(CrankNicolson(), adaptive_grid_cfg, adaptive=True)

        p_u = uniform.price(market, n_steps=50, option_type="put").price
        p_a = adaptive.price(market, n_steps=50, option_type="put").price

        err_u = abs(p_u - ref) / ref
        err_a = abs(p_a - ref) / ref
        assert err_a <= err_u

    def test_adaptive_atm_put_vs_crr(self, adaptive_grid_cfg):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        ref = american_binomial(100, 100, 0.05, 0.2, 1.0, "put", n_steps=50000)

        pricer = AmericanPricer(CrankNicolson(), adaptive_grid_cfg, adaptive=True)
        p = pricer.price(market, n_steps=50, option_type="put").price
        assert abs(p - ref) / ref < 5e-4

    def test_adaptive_itm_put(self, adaptive_grid_cfg):
        market = MarketParams(S=90, K=100, T=1.0, r=0.05, sigma=0.2)
        ref = american_binomial(90, 100, 0.05, 0.2, 1.0, "put", n_steps=40000)
        pricer = AmericanPricer(CrankNicolson(), adaptive_grid_cfg, adaptive=True)
        p = pricer.price(market, n_steps=50, option_type="put").price
        assert abs(p - ref) / ref < 0.001

    def test_adaptive_otm_put(self, adaptive_grid_cfg):
        market = MarketParams(S=110, K=100, T=1.0, r=0.05, sigma=0.2)
        ref = american_binomial(110, 100, 0.05, 0.2, 1.0, "put", n_steps=40000)
        pricer = AmericanPricer(CrankNicolson(), adaptive_grid_cfg, adaptive=True)
        p = pricer.price(market, n_steps=50, option_type="put").price
        assert abs(p - ref) / max(ref, 1e-12) < 0.005

    def test_adaptive_false_equals_default_mode(self, adaptive_grid_cfg):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        p_default = AmericanPricer(CrankNicolson(), adaptive_grid_cfg).price(
            market, n_steps=80, option_type="put"
        ).price
        p_explicit = AmericanPricer(CrankNicolson(), adaptive_grid_cfg, adaptive=False).price(
            market, n_steps=80, option_type="put"
        ).price
        assert abs(p_default - p_explicit) < 1e-12

    def test_adaptive_with_dividends(self, adaptive_grid_cfg):
        market = MarketParams(S=100, K=100, T=1.0, r=0.04, sigma=0.2)
        divs = DividendSchedule(times=(0.5,), amounts=(2.0,), proportional=False)
        pricer = AmericanPricer(CrankNicolson(), adaptive_grid_cfg, adaptive=True)
        r = pricer.price(market, n_steps=80, option_type="put", dividends=divs)
        assert np.isfinite(r.price)
        assert r.price > 0.0

    def test_adaptive_with_continuous_yield(self, adaptive_grid_cfg):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.03)
        p_uniform = AmericanPricer(CrankNicolson(), adaptive_grid_cfg, adaptive=False).price(
            market, n_steps=80, option_type="put"
        ).price
        p_adapt = AmericanPricer(CrankNicolson(), adaptive_grid_cfg, adaptive=True).price(
            market, n_steps=80, option_type="put"
        ).price
        assert abs(p_adapt - p_uniform) / max(p_uniform, 1e-12) < 0.05


class TestAdaptiveConvergence:

    def test_stable_over_n_steps(self, adaptive_grid_cfg):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        ref = american_binomial(100, 100, 0.05, 0.2, 1.0, "put", n_steps=50000)

        pricer = AmericanPricer(CrankNicolson(), adaptive_grid_cfg, adaptive=True)
        e50 = abs(pricer.price(market, n_steps=50, option_type="put").price - ref)
        e100 = abs(pricer.price(market, n_steps=100, option_type="put").price - ref)

        assert e50 / ref < 0.01
        assert e100 / ref < 0.02

    def test_adaptive_cn_better_than_uniform_cn_same_n(self, adaptive_grid_cfg):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        ref = american_binomial(100, 100, 0.05, 0.2, 1.0, "put", n_steps=50000)

        uniform = AmericanPricer(CrankNicolson(), adaptive_grid_cfg, adaptive=False)
        adaptive = AmericanPricer(CrankNicolson(), adaptive_grid_cfg, adaptive=True)

        err_uniform_50 = abs(uniform.price(market, n_steps=50, option_type="put").price - ref)
        err_adaptive_50 = abs(adaptive.price(market, n_steps=50, option_type="put").price - ref)
        assert err_adaptive_50 <= err_uniform_50


class TestAdaptiveEdgeCases:

    def test_zero_rate(self, adaptive_grid_cfg):
        market = MarketParams(S=100, K=100, T=1.0, r=1e-8, sigma=0.2)
        pricer = AmericanPricer(CrankNicolson(), adaptive_grid_cfg, adaptive=True)
        r = pricer.price(market, n_steps=80, option_type="put")
        assert np.isfinite(r.price)
        assert r.price >= r.european_price

    def test_high_vol(self, adaptive_grid_cfg):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.8)
        pricer = AmericanPricer(CrankNicolson(), adaptive_grid_cfg, adaptive=True)
        assert np.isfinite(pricer.price(market, n_steps=80, option_type="put").price)

    def test_short_expiry(self, adaptive_grid_cfg):
        market = MarketParams(S=100, K=100, T=1 / 365, r=0.05, sigma=0.2)
        pricer = AmericanPricer(CrankNicolson(), adaptive_grid_cfg, adaptive=True)
        assert np.isfinite(pricer.price(market, n_steps=40, option_type="put").price)

