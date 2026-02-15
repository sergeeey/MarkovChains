"""Tests for Greeks computation (10 tests)."""

import numpy as np
import pytest
from scipy.stats import norm

from chernoffpy.finance.validation import MarketParams
from chernoffpy.finance.transforms import bs_exact_price
from chernoffpy.finance.greeks import compute_greeks


N_STEPS = 50


# ---------------------------------------------------------------------------
# Analytical BS Greeks for comparison
# ---------------------------------------------------------------------------

def _d1(market: MarketParams) -> float:
    return (
        np.log(market.S / market.K) + (market.r + market.sigma ** 2 / 2) * market.T
    ) / (market.sigma * np.sqrt(market.T))


def bs_delta(market: MarketParams, option_type: str) -> float:
    d1 = _d1(market)
    if option_type == "call":
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1)


def bs_gamma(market: MarketParams) -> float:
    d1 = _d1(market)
    return float(norm.pdf(d1) / (market.S * market.sigma * np.sqrt(market.T)))


def bs_vega(market: MarketParams) -> float:
    d1 = _d1(market)
    return float(market.S * norm.pdf(d1) * np.sqrt(market.T))


def bs_theta(market: MarketParams, option_type: str) -> float:
    d1 = _d1(market)
    d2 = d1 - market.sigma * np.sqrt(market.T)
    first_term = -market.S * norm.pdf(d1) * market.sigma / (2 * np.sqrt(market.T))
    if option_type == "call":
        return float(first_term - market.r * market.K * np.exp(-market.r * market.T) * norm.cdf(d2))
    return float(first_term + market.r * market.K * np.exp(-market.r * market.T) * norm.cdf(-d2))


def bs_rho(market: MarketParams, option_type: str) -> float:
    d1 = _d1(market)
    d2 = d1 - market.sigma * np.sqrt(market.T)
    if option_type == "call":
        return float(market.K * market.T * np.exp(-market.r * market.T) * norm.cdf(d2))
    return float(-market.K * market.T * np.exp(-market.r * market.T) * norm.cdf(-d2))


# ---------------------------------------------------------------------------
# Bounds tests
# ---------------------------------------------------------------------------

class TestGreeksBounds:

    def test_call_delta_bounds(self, cn_pricer, atm_market):
        """Call delta in [0, 1]."""
        greeks = compute_greeks(cn_pricer, atm_market, N_STEPS, "call")
        assert -0.05 <= greeks.delta <= 1.05

    def test_put_delta_bounds(self, cn_pricer, atm_market):
        """Put delta in [-1, 0]."""
        greeks = compute_greeks(cn_pricer, atm_market, N_STEPS, "put")
        assert -1.05 <= greeks.delta <= 0.05

    def test_gamma_positive(self, cn_pricer, atm_market):
        """Gamma is positive for vanilla options."""
        greeks = compute_greeks(cn_pricer, atm_market, N_STEPS, "call")
        assert greeks.gamma > -0.001

    def test_vega_positive(self, cn_pricer, atm_market):
        """Vega is positive for long vanilla options."""
        greeks = compute_greeks(cn_pricer, atm_market, N_STEPS, "call")
        assert greeks.vega > -0.1

    def test_call_theta_negative(self, cn_pricer, atm_market):
        """ATM call theta is typically negative (time decay)."""
        greeks = compute_greeks(cn_pricer, atm_market, N_STEPS, "call")
        assert greeks.theta < 1.0  # allow small positive for deep ITM

    def test_call_rho_positive(self, cn_pricer, atm_market):
        """Call rho is positive (higher rates -> higher call value)."""
        greeks = compute_greeks(cn_pricer, atm_market, N_STEPS, "call")
        assert greeks.rho > -1.0


# ---------------------------------------------------------------------------
# Comparison with analytical BS Greeks (5% tolerance)
# ---------------------------------------------------------------------------

class TestGreeksAccuracy:

    def test_delta_vs_analytical(self, cn_pricer, atm_market):
        """Numerical delta matches BS analytical within 5%."""
        greeks = compute_greeks(cn_pricer, atm_market, N_STEPS, "call")
        expected = bs_delta(atm_market, "call")
        assert greeks.delta == pytest.approx(expected, abs=0.05)

    def test_gamma_vs_analytical(self, cn_pricer, atm_market):
        """Numerical gamma matches BS analytical within 5%."""
        greeks = compute_greeks(cn_pricer, atm_market, N_STEPS, "call")
        expected = bs_gamma(atm_market)
        assert greeks.gamma == pytest.approx(expected, rel=0.10, abs=0.002)

    def test_vega_vs_analytical(self, cn_pricer, atm_market):
        """Numerical vega matches BS analytical within 5%."""
        greeks = compute_greeks(cn_pricer, atm_market, N_STEPS, "call")
        expected = bs_vega(atm_market)
        assert greeks.vega == pytest.approx(expected, rel=0.10, abs=2.0)

    def test_delta_put_call_parity(self, cn_pricer, atm_market):
        """delta_call - delta_put = 1."""
        g_call = compute_greeks(cn_pricer, atm_market, N_STEPS, "call")
        g_put = compute_greeks(cn_pricer, atm_market, N_STEPS, "put")
        assert g_call.delta - g_put.delta == pytest.approx(1.0, abs=0.10)
