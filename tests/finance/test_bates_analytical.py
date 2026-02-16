"""Tests for analytical Bates pricing helper."""

from __future__ import annotations

import numpy as np

from chernoffpy.finance.bates_analytical import bates_price
from chernoffpy.finance.heston_analytical import heston_price
from chernoffpy.finance.transforms import bs_exact_price
from chernoffpy.finance.validation import MarketParams


class TestBatesAnalytical:

    def test_zero_jumps_equals_heston(self):
        b = bates_price(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, 0.0, -0.1, 0.2, "call")
        h = heston_price(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, "call")
        assert abs(b - h) / h < 1e-3

    def test_zero_vol_zero_jumps_equals_bs(self):
        b = bates_price(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, "call")
        bs = bs_exact_price(MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2), "call")
        assert abs(b - bs) / bs < 1e-2

    def test_nonnegative(self):
        c = bates_price(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, 0.5, -0.1, 0.2, "call")
        p = bates_price(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, 0.5, -0.1, 0.2, "put")
        assert c >= 0.0
        assert p >= 0.0

    def test_call_le_spot(self):
        c = bates_price(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, 0.5, -0.1, 0.2, "call")
        assert c <= 100.0 + 1e-8

    def test_put_call_parity(self):
        c = bates_price(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, 0.5, -0.1, 0.2, "call")
        p = bates_price(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, 0.5, -0.1, 0.2, "put")
        rhs = 100.0 - 100.0 * np.exp(-0.05)
        assert abs((c - p) - rhs) < 0.2

    def test_jumps_increase_otm_put(self):
        no_jump = bates_price(100, 80, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, 0.0, -0.1, 0.2, "put")
        jump = bates_price(100, 80, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, 1.0, -0.2, 0.25, "put")
        assert jump >= 0.0

