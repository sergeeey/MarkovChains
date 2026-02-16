"""Tests for AmericanPricer via Chernoff projection."""

import numpy as np
import pytest

from chernoffpy import BackwardEuler, CrankNicolson
from chernoffpy.finance.american import AmericanPricer
from chernoffpy.finance.american_analytical import american_binomial
from chernoffpy.finance.transforms import bs_exact_price
from chernoffpy.finance.validation import GridConfig, MarketParams


@pytest.fixture
def amer_grid():
    return GridConfig(N=1024, L=8.0, taper_width=2.0)


@pytest.fixture
def amer_cn(amer_grid):
    return AmericanPricer(CrankNicolson(), amer_grid)


@pytest.fixture
def amer_be(amer_grid):
    return AmericanPricer(BackwardEuler(), amer_grid)


class TestAmericanValidation:

    def test_valid_params_accepted(self, amer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        result = amer_cn.price(market, n_steps=80, option_type="put")
        assert result.price >= 0.0

    def test_invalid_option_type_rejected(self, amer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        with pytest.raises(ValueError, match="option_type"):
            amer_cn.price(market, option_type="straddle")

    def test_invalid_n_steps_rejected(self, amer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        with pytest.raises(ValueError, match="n_steps"):
            amer_cn.price(market, n_steps=0)


class TestAmericanPutPricing:

    def test_amer_put_atm_vs_binomial(self, amer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        num = amer_cn.price(market, n_steps=140, option_type="put").price
        ref = american_binomial(100, 100, 0.05, 0.2, 1.0, "put", n_steps=5000)
        assert abs(num - ref) / ref < 0.03

    def test_amer_put_itm_vs_binomial(self, amer_cn):
        market = MarketParams(S=90, K=100, T=1.0, r=0.05, sigma=0.2)
        num = amer_cn.price(market, n_steps=140, option_type="put").price
        ref = american_binomial(90, 100, 0.05, 0.2, 1.0, "put", n_steps=5000)
        assert abs(num - ref) / ref < 0.03

    def test_amer_put_otm_vs_binomial(self, amer_cn):
        market = MarketParams(S=110, K=100, T=1.0, r=0.05, sigma=0.2)
        num = amer_cn.price(market, n_steps=140, option_type="put").price
        ref = american_binomial(110, 100, 0.05, 0.2, 1.0, "put", n_steps=5000)
        assert abs(num - ref) / max(ref, 1e-12) < 0.05

    def test_amer_put_deep_itm(self, amer_cn):
        market = MarketParams(S=50, K=100, T=1.0, r=0.05, sigma=0.2)
        p = amer_cn.price(market, n_steps=120, option_type="put").price
        assert p >= 50.0

    def test_amer_put_deep_otm(self, amer_cn):
        market = MarketParams(S=150, K=100, T=1.0, r=0.05, sigma=0.2)
        result = amer_cn.price(market, n_steps=120, option_type="put")
        assert result.price < 0.3
        assert result.price >= result.european_price

    def test_amer_put_high_vol(self, amer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.6)
        result = amer_cn.price(market, n_steps=120, option_type="put")
        assert np.isfinite(result.price)
        assert result.price >= result.european_price

    def test_amer_put_low_vol(self, amer_cn):
        market = MarketParams(S=90, K=100, T=1.0, r=0.05, sigma=0.05)
        p = amer_cn.price(market, n_steps=120, option_type="put").price
        assert p >= 10.0

    def test_amer_put_short_expiry(self, amer_cn):
        market = MarketParams(S=100, K=100, T=7 / 365, r=0.05, sigma=0.2)
        result = amer_cn.price(market, n_steps=80, option_type="put")
        assert np.isfinite(result.price)
        assert result.price >= max(market.K - market.S, 0.0)

    def test_amer_put_long_expiry(self, amer_cn):
        market = MarketParams(S=100, K=100, T=5.0, r=0.05, sigma=0.2)
        result = amer_cn.price(market, n_steps=180, option_type="put")
        assert np.isfinite(result.price)
        assert result.price >= result.european_price

    def test_amer_put_high_rate(self, amer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.15, sigma=0.2)
        result = amer_cn.price(market, n_steps=120, option_type="put")
        assert result.early_exercise_premium > 0.01

    def test_amer_put_zero_rate(self, amer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=1e-8, sigma=0.2)
        result = amer_cn.price(market, n_steps=120, option_type="put")
        assert result.early_exercise_premium < 0.05


class TestAmericanCallPricing:

    def test_amer_call_equals_european(self, amer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        result = amer_cn.price(market, n_steps=120, option_type="call")
        assert abs(result.price - result.european_price) / result.european_price < 2e-3

    def test_amer_call_eep_zero(self, amer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        result = amer_cn.price(market, n_steps=120, option_type="call")
        assert result.early_exercise_premium < 1e-3

    def test_amer_call_atm(self, amer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        num = amer_cn.price(market, n_steps=120, option_type="call").price
        bs = bs_exact_price(market, "call")
        assert abs(num - bs) / bs < 0.01


class TestAmericanProperties:

    def test_amer_ge_european(self, amer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        for opt in ("call", "put"):
            result = amer_cn.price(market, n_steps=120, option_type=opt)
            assert result.price >= result.european_price

    def test_amer_ge_intrinsic(self, amer_cn):
        market = MarketParams(S=90, K=100, T=1.0, r=0.05, sigma=0.2)
        result = amer_cn.price(market, n_steps=120, option_type="put")
        assert result.price >= max(market.K - market.S, 0.0)

    def test_nonnegative(self, amer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        for opt in ("call", "put"):
            assert amer_cn.price(market, n_steps=120, option_type=opt).price >= 0.0

    def test_put_le_strike(self, amer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        p = amer_cn.price(market, n_steps=120, option_type="put").price
        assert p <= market.K + 1e-8

    def test_call_le_spot(self, amer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        c = amer_cn.price(market, n_steps=120, option_type="call").price
        assert c <= market.S + 1e-8

    def test_eep_nonnegative(self, amer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        for opt in ("call", "put"):
            eep = amer_cn.price(market, n_steps=120, option_type=opt).early_exercise_premium
            assert eep >= -1e-10

    def test_eep_increases_with_rate(self, amer_cn):
        low = MarketParams(S=100, K=100, T=1.0, r=0.01, sigma=0.2)
        high = MarketParams(S=100, K=100, T=1.0, r=0.10, sigma=0.2)
        e_low = amer_cn.price(low, n_steps=120, option_type="put").early_exercise_premium
        e_high = amer_cn.price(high, n_steps=120, option_type="put").early_exercise_premium
        assert e_high >= e_low

    def test_monotonicity_in_vol(self, amer_cn):
        low = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.1)
        high = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.4)
        p_low = amer_cn.price(low, n_steps=120, option_type="put").price
        p_high = amer_cn.price(high, n_steps=120, option_type="put").price
        assert p_high >= p_low


class TestAmericanConvergence:

    def test_convergence_rate(self, amer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        ref = american_binomial(100, 100, 0.05, 0.2, 1.0, "put", n_steps=5000)

        e50 = abs(amer_cn.price(market, n_steps=50, option_type="put").price - ref)
        e100 = abs(amer_cn.price(market, n_steps=100, option_type="put").price - ref)
        e150 = abs(amer_cn.price(market, n_steps=150, option_type="put").price - ref)

        assert e150 <= e100 + 1e-8
        assert e100 <= e50 + 1e-8

    def test_cn_better_than_be(self, amer_cn, amer_be):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        ref = american_binomial(100, 100, 0.05, 0.2, 1.0, "put", n_steps=5000)
        err_cn = abs(amer_cn.price(market, n_steps=120, option_type="put").price - ref)
        err_be = abs(amer_be.price(market, n_steps=120, option_type="put").price - ref)
        assert err_cn <= err_be + 1e-8


class TestExerciseBoundary:

    def test_boundary_exists_for_put(self, amer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        b = amer_cn.price(market, n_steps=120, option_type="put", return_boundary=True).exercise_boundary
        assert b is not None
        assert len(b) == 120

    def test_boundary_below_strike_for_put(self, amer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        b = amer_cn.price(market, n_steps=120, option_type="put", return_boundary=True).exercise_boundary
        assert np.max(b) <= market.K + 1e-8

    def test_boundary_has_finite_values(self, amer_cn):
        market = MarketParams(S=95, K=100, T=1.0, r=0.05, sigma=0.25)
        b = amer_cn.price(market, n_steps=120, option_type="put", return_boundary=True).exercise_boundary
        assert np.all(np.isfinite(b))


class TestAmericanEdgeCases:

    def test_very_short_expiry(self, amer_cn):
        market = MarketParams(S=100, K=100, T=1 / 365, r=0.05, sigma=0.2)
        p = amer_cn.price(market, n_steps=40, option_type="put").price
        assert np.isfinite(p)

    def test_very_high_vol(self, amer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=1.0)
        p = amer_cn.price(market, n_steps=120, option_type="put").price
        assert np.isfinite(p)

    def test_result_structure(self, amer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        r = amer_cn.price(market, n_steps=80, option_type="put")
        assert "Crank" in r.method_name
        assert r.n_steps == 80
        assert r.market == market

