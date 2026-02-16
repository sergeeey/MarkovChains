"""Tests for semi-closed-form Heston analytical pricing."""

import numpy as np

from chernoffpy.finance.heston_analytical import heston_call, heston_price, heston_put
from chernoffpy.finance.heston_params import HestonParams
from chernoffpy.finance.transforms import bs_exact_price
from chernoffpy.finance.validation import MarketParams


class TestHestonAnalytical:

    def test_heston_reduces_to_bs(self):
        h = heston_call(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 1e-12, 0.0)
        bs = bs_exact_price(MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2), "call")
        assert abs(h - bs) / bs < 1e-8

    def test_heston_call_nonnegative(self):
        p = heston_call(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7)
        assert p >= 0.0

    def test_heston_put_call_parity(self):
        c = heston_call(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7)
        p = heston_put(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7)
        rhs = 100 - 100 * np.exp(-0.05)
        assert abs((c - p) - rhs) / max(abs(rhs), 1e-12) < 1e-4

    def test_heston_atm_benchmark(self):
        p = heston_call(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7)
        assert 8.0 < p < 14.0

    def test_heston_smile_proxy(self):
        p_otm_put = heston_put(100, 120, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7)
        p_atm_put = heston_put(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7)
        assert p_otm_put > p_atm_put

    def test_negative_rho_skew(self):
        neg = heston_call(100, 90, 1.0, 0.05, 0.04, 2.0, 0.04, 0.4, -0.8)
        zero = heston_call(100, 90, 1.0, 0.05, 0.04, 2.0, 0.04, 0.4, 0.0)
        assert abs(neg - zero) > 1e-3

    def test_feller_condition_check(self):
        ok = HestonParams(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7)
        bad = HestonParams(100, 100, 1.0, 0.05, 0.04, 0.5, 0.04, 0.5, -0.7)
        assert ok.feller_condition is True
        assert bad.feller_condition is False

    def test_high_vol_of_vol(self):
        p = heston_call(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 1.0, -0.7)
        assert np.isfinite(p)
        assert p > 0.0

    def test_heston_price_dispatch(self):
        c = heston_price(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, "call")
        p = heston_price(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, "put")
        assert c >= 0.0
        assert p >= 0.0
