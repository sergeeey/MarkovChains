"""Tests for EuropeanPricer (20 tests)."""

import numpy as np
import pytest

from chernoffpy import BackwardEuler, CrankNicolson
from chernoffpy.finance.validation import MarketParams, GridConfig
from chernoffpy.finance.transforms import bs_exact_price
from chernoffpy.finance.european import EuropeanPricer


N_STEPS = 50


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _check_price(pricer, market, option_type, rel_tol=0.01, abs_tol=0.15):
    """Assert Chernoff price is close to BS exact price."""
    result = pricer.price(market, n_steps=N_STEPS, option_type=option_type)
    bs = bs_exact_price(market, option_type)
    assert result.price == pytest.approx(bs, rel=rel_tol, abs=abs_tol), (
        f"{option_type} price={result.price:.4f}, BS={bs:.4f}, "
        f"rel_err={result.certificate.rel_error:.4f}"
    )
    return result


# ---------------------------------------------------------------------------
# Call prices (5 moneyness levels)
# ---------------------------------------------------------------------------

class TestCallPrices:

    def test_atm_call(self, cn_pricer, atm_market):
        _check_price(cn_pricer, atm_market, "call")

    def test_itm_call(self, cn_pricer, itm_call_market):
        _check_price(cn_pricer, itm_call_market, "call")

    def test_otm_call(self, cn_pricer, otm_call_market):
        _check_price(cn_pricer, otm_call_market, "call")

    def test_deep_itm_call(self, cn_pricer, deep_itm_call_market):
        _check_price(cn_pricer, deep_itm_call_market, "call", rel_tol=0.02)

    def test_deep_otm_call(self, cn_pricer, deep_otm_call_market):
        _check_price(cn_pricer, deep_otm_call_market, "call", rel_tol=0.05, abs_tol=0.5)


# ---------------------------------------------------------------------------
# Put prices (5 moneyness levels)
# ---------------------------------------------------------------------------

class TestPutPrices:

    def test_atm_put(self, cn_pricer, atm_market):
        _check_price(cn_pricer, atm_market, "put")

    def test_itm_put(self, cn_pricer, otm_call_market):
        """S=90, K=100 is ITM for put."""
        _check_price(cn_pricer, otm_call_market, "put")

    def test_otm_put(self, cn_pricer, itm_call_market):
        """S=110, K=100 is OTM for put."""
        _check_price(cn_pricer, itm_call_market, "put")

    def test_deep_itm_put(self, cn_pricer, deep_otm_call_market):
        """S=60, K=100 is deep ITM for put."""
        _check_price(cn_pricer, deep_otm_call_market, "put", rel_tol=0.02)

    def test_deep_otm_put(self, cn_pricer, deep_itm_call_market):
        """S=150, K=100 is deep OTM for put."""
        _check_price(cn_pricer, deep_itm_call_market, "put", rel_tol=0.05, abs_tol=0.5)


# ---------------------------------------------------------------------------
# Arbitrage constraints
# ---------------------------------------------------------------------------

class TestArbitrageConstraints:

    def test_put_call_parity(self, cn_pricer, atm_market):
        """Call - Put = S - K*exp(-rT) (within tolerance)."""
        call = cn_pricer.price(atm_market, N_STEPS, "call").price
        put = cn_pricer.price(atm_market, N_STEPS, "put").price
        parity = atm_market.S - atm_market.K * np.exp(-atm_market.r * atm_market.T)
        assert call - put == pytest.approx(parity, abs=0.15)

    def test_nonnegative_price(self, cn_pricer, atm_market):
        """Option prices must be non-negative."""
        call = cn_pricer.price(atm_market, N_STEPS, "call").price
        put = cn_pricer.price(atm_market, N_STEPS, "put").price
        assert call >= -0.01
        assert put >= -0.01

    def test_call_upper_bound(self, cn_pricer, atm_market):
        """Call price <= spot price S."""
        call = cn_pricer.price(atm_market, N_STEPS, "call").price
        assert call <= atm_market.S + 0.01

    def test_put_upper_bound(self, cn_pricer, atm_market):
        """Put price <= K * exp(-rT)."""
        put = cn_pricer.price(atm_market, N_STEPS, "put").price
        assert put <= atm_market.K * np.exp(-atm_market.r * atm_market.T) + 0.01


# ---------------------------------------------------------------------------
# Certificate and convergence
# ---------------------------------------------------------------------------

class TestCertificate:

    def test_error_decomposition(self, cn_pricer, atm_market):
        """Total error ~ chernoff_error + domain_error (triangle inequality)."""
        result = cn_pricer.price(atm_market, N_STEPS, "call")
        cert = result.certificate
        # By triangle inequality: abs_error <= chernoff_error + domain_error
        assert cert.abs_error <= cert.chernoff_error + cert.domain_error + 1e-10


class TestConvergence:

    def test_backward_euler_order1(self, be_pricer, atm_market):
        """BackwardEuler convergence rate alpha ~ 1."""
        errors = []
        ns = [5, 10, 20, 40, 80]
        for n in ns:
            result = be_pricer.price(atm_market, n, "call")
            errors.append(result.certificate.chernoff_error)
        # Fit alpha from log-log regression
        alpha = _fit_rate(ns, errors)
        assert 0.5 < alpha < 1.8, f"Expected alpha ~ 1, got {alpha:.3f}"

    def test_crank_nicolson_order2(self, cn_pricer, atm_market):
        """CrankNicolson convergence rate alpha ~ 2."""
        errors = []
        ns = [5, 10, 20, 40, 80]
        for n in ns:
            result = cn_pricer.price(atm_market, n, "call")
            errors.append(result.certificate.chernoff_error)
        alpha = _fit_rate(ns, errors)
        assert 1.5 < alpha < 4.5, f"Expected alpha >= 2, got {alpha:.3f}"


class TestSpecialCases:

    def test_high_volatility(self, high_vol_market, default_grid):
        """Pricing with sigma=80% still converges."""
        pricer = EuropeanPricer(CrankNicolson(), default_grid)
        result = pricer.price(high_vol_market, N_STEPS, "call")
        bs = bs_exact_price(high_vol_market, "call")
        assert result.price == pytest.approx(bs, rel=0.03, abs=0.5)

    def test_short_expiry(self, short_expiry_market, default_grid):
        """Pricing with T=1 week works."""
        pricer = EuropeanPricer(CrankNicolson(), default_grid)
        result = pricer.price(short_expiry_market, N_STEPS, "call")
        bs = bs_exact_price(short_expiry_market, "call")
        assert result.price == pytest.approx(bs, rel=0.02, abs=0.15)

    def test_fine_grid_reduces_domain_error(self, atm_market, default_grid, fine_grid):
        """Larger domain L=15 should reduce domain_error (truncation + taper)."""
        pricer_default = EuropeanPricer(CrankNicolson(), default_grid)
        pricer_fine = EuropeanPricer(CrankNicolson(), fine_grid)
        r_default = pricer_default.price(atm_market, N_STEPS, "call")
        r_fine = pricer_fine.price(atm_market, N_STEPS, "call")
        # Domain error (FFT exact vs BS exact) should be smaller with larger L
        assert r_fine.certificate.domain_error <= r_default.certificate.domain_error + 1e-6


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _fit_rate(ns, errors):
    """Estimate convergence rate alpha from error_n ~ C / n^alpha."""
    # Filter out near-zero errors to avoid log(0)
    valid = [(n, e) for n, e in zip(ns, errors) if e > 1e-15]
    if len(valid) < 2:
        return 0.0
    log_n = np.log([v[0] for v in valid])
    log_e = np.log([v[1] for v in valid])
    coeffs = np.polyfit(log_n, log_e, 1)
    return -coeffs[0]
