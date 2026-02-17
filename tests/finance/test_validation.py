"""Tests for validation dataclasses and edge cases (21 tests).

Covers:
- MarketParams: 5 ValueError branches
- GridConfig: 4 ValueError branches
- EuropeanPricer: n_steps validation
- Overflow protection for low volatility
- Non-negative price guarantee
- Adaptive theta step for short expiry
"""

import numpy as np
import pytest

from chernoffpy import CrankNicolson, BackwardEuler
from chernoffpy.finance.validation import MarketParams, GridConfig
from chernoffpy.finance.transforms import (
    bs_to_heat_initial,
    compute_transform_params,
    extract_price_at_spot,
    make_grid,
    bs_exact_price,
)
from chernoffpy.finance.european import EuropeanPricer
from chernoffpy.finance.greeks import compute_greeks


# ---------------------------------------------------------------------------
# MarketParams validation (5 tests)
# ---------------------------------------------------------------------------

class TestMarketParamsValidation:

    def test_negative_spot(self):
        with pytest.raises(ValueError, match="Spot price S must be positive"):
            MarketParams(S=-1.0, K=100, T=1.0, r=0.05, sigma=0.20)

    def test_zero_spot(self):
        with pytest.raises(ValueError, match="Spot price S must be positive"):
            MarketParams(S=0.0, K=100, T=1.0, r=0.05, sigma=0.20)

    def test_negative_strike(self):
        with pytest.raises(ValueError, match="Strike price K must be positive"):
            MarketParams(S=100, K=-50.0, T=1.0, r=0.05, sigma=0.20)

    def test_zero_expiry(self):
        with pytest.raises(ValueError, match="Time to expiry T must be positive"):
            MarketParams(S=100, K=100, T=0.0, r=0.05, sigma=0.20)

    def test_negative_rate(self):
        with pytest.raises(ValueError, match="Risk-free rate r must be non-negative"):
            MarketParams(S=100, K=100, T=1.0, r=-0.01, sigma=0.20)

    def test_zero_volatility(self):
        with pytest.raises(ValueError, match="Volatility sigma must be positive"):
            MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.0)

    def test_negative_volatility(self):
        with pytest.raises(ValueError, match="Volatility sigma must be positive"):
            MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=-0.10)


# ---------------------------------------------------------------------------
# GridConfig validation (4 tests)
# ---------------------------------------------------------------------------

class TestGridConfigValidation:

    def test_small_N(self):
        with pytest.raises(ValueError, match="Grid size N must be >= 64"):
            GridConfig(N=32)

    def test_zero_L(self):
        with pytest.raises(ValueError, match="Domain half-width L must be positive"):
            GridConfig(L=0.0)

    def test_negative_L(self):
        with pytest.raises(ValueError, match="Domain half-width L must be positive"):
            GridConfig(L=-5.0)

    def test_taper_width_too_large(self):
        with pytest.raises(ValueError, match="Taper width must be in"):
            GridConfig(L=5.0, taper_width=5.0)

    def test_taper_width_zero(self):
        with pytest.raises(ValueError, match="Taper width must be in"):
            GridConfig(taper_width=0.0)


# ---------------------------------------------------------------------------
# n_steps validation (1 test)
# ---------------------------------------------------------------------------

class TestNStepsValidation:

    def test_zero_n_steps(self):
        """n_steps=0 should raise ValueError."""
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        pricer = EuropeanPricer(CrankNicolson())
        with pytest.raises(ValueError, match="n_steps must be >= 1"):
            pricer.price(market, n_steps=0)

    def test_negative_n_steps(self):
        """n_steps=-1 should raise ValueError."""
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        pricer = EuropeanPricer(CrankNicolson())
        with pytest.raises(ValueError, match="n_steps must be >= 1"):
            pricer.price(market, n_steps=-1)


# ---------------------------------------------------------------------------
# Invalid option_type (2 tests)
# ---------------------------------------------------------------------------

class TestInvalidOptionType:

    def test_invalid_option_type_initial(self):
        """Invalid option_type in bs_to_heat_initial."""
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        config = GridConfig()
        x = make_grid(config)
        with pytest.raises(ValueError, match="option_type must be"):
            bs_to_heat_initial(x, market, config, "forward")

    def test_invalid_option_type_bs(self):
        """Invalid option_type in bs_exact_price."""
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        with pytest.raises(ValueError, match="option_type must be"):
            bs_exact_price(market, "straddle")


# ---------------------------------------------------------------------------
# Overflow protection for low volatility (2 tests)
# ---------------------------------------------------------------------------

class TestOverflowProtection:

    def test_low_vol_no_nan(self):
        """Low volatility sigma=0.03 should not produce NaN."""
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.03)
        config = GridConfig()
        x = make_grid(config)
        u0 = bs_to_heat_initial(x, market, config, "call")
        assert not np.any(np.isnan(u0)), "NaN in initial condition for low vol"
        assert not np.any(np.isinf(u0)), "Inf in initial condition for low vol"

    def test_low_vol_pricing(self):
        """Low vol pricing should produce finite, reasonable price."""
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.05)
        pricer = EuropeanPricer(CrankNicolson())
        result = pricer.price(market, n_steps=50, option_type="call")
        assert np.isfinite(result.price), f"Non-finite price: {result.price}"
        assert result.price >= 0, f"Negative price: {result.price}"
        bs = bs_exact_price(market, "call")
        # Low-vol edge case: wider tolerance but still catches regressions.
        assert result.price == pytest.approx(bs, rel=0.05, abs=0.1)


# ---------------------------------------------------------------------------
# Non-negative price guarantee (2 tests)
# ---------------------------------------------------------------------------

class TestNonNegativePrice:

    def test_deep_otm_call_nonnegative(self):
        """Deep OTM call (S=20, K=100) price must be >= 0."""
        market = MarketParams(S=20.0, K=100.0, T=1.0, r=0.05, sigma=0.20)
        pricer = EuropeanPricer(CrankNicolson())
        result = pricer.price(market, n_steps=50, option_type="call")
        assert result.price >= 0, f"Negative price: {result.price}"

    def test_deep_otm_put_nonnegative(self):
        """Deep OTM put (S=500, K=100) price must be >= 0."""
        market = MarketParams(S=500.0, K=100.0, T=1.0, r=0.05, sigma=0.20)
        pricer = EuropeanPricer(CrankNicolson())
        result = pricer.price(market, n_steps=50, option_type="put")
        assert result.price >= 0, f"Negative price: {result.price}"


# ---------------------------------------------------------------------------
# Adaptive theta step (1 test)
# ---------------------------------------------------------------------------

class TestAdaptiveThetaStep:

    def test_short_expiry_greeks_finite(self):
        """Greeks for very short expiry T=1 hour should be finite."""
        market = MarketParams(S=100, K=100, T=1 / (365 * 24), r=0.05, sigma=0.20)
        pricer = EuropeanPricer(CrankNicolson())
        greeks = compute_greeks(pricer, market, n_steps=50, option_type="call")
        assert np.isfinite(greeks.delta), f"Non-finite delta: {greeks.delta}"
        assert np.isfinite(greeks.theta), f"Non-finite theta: {greeks.theta}"
        assert np.isfinite(greeks.vega), f"Non-finite vega: {greeks.vega}"
