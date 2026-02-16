"""Tests for analytical double-barrier pricing."""

import pytest

from chernoffpy.finance.double_barrier_analytical import double_barrier_analytical
from chernoffpy.finance.transforms import bs_exact_price
from chernoffpy.finance.validation import MarketParams


class TestDoubleBarrierAnalyticalSanity:

    def test_dko_nonnegative(self):
        p = double_barrier_analytical(100, 100, 80, 120, 0.05, 0.2, 1.0, "call")
        assert p >= 0.0

    def test_dko_le_vanilla(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        dko = double_barrier_analytical(100, 100, 80, 120, 0.05, 0.2, 1.0, "call")
        vanilla = bs_exact_price(market, "call")
        assert dko <= vanilla + 1e-10

    def test_in_out_parity(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        for opt in ("call", "put"):
            dko = double_barrier_analytical(100, 100, 80, 120, 0.05, 0.2, 1.0, opt, "double_knock_out")
            dki = double_barrier_analytical(100, 100, 80, 120, 0.05, 0.2, 1.0, opt, "double_knock_in")
            vanilla = bs_exact_price(market, opt)
            assert dki + dko == pytest.approx(vanilla, abs=1e-8)

    def test_wide_corridor_approaches_vanilla(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        dko = double_barrier_analytical(100, 100, 1, 1000, 0.05, 0.2, 1.0, "call")
        vanilla = bs_exact_price(market, "call")
        assert abs(dko - vanilla) / vanilla < 0.06

    def test_narrow_corridor_near_zero(self):
        dko = double_barrier_analytical(100, 100, 99, 101, 0.05, 0.2, 1.0, "call")
        assert dko < 0.5

    def test_series_convergence(self):
        p50 = double_barrier_analytical(100, 100, 80, 120, 0.05, 0.2, 1.0, "call", n_terms=50)
        p200 = double_barrier_analytical(100, 100, 80, 120, 0.05, 0.2, 1.0, "call", n_terms=200)
        assert abs(p50 - p200) < 1e-4

    def test_symmetric_corridor(self):
        p = double_barrier_analytical(100, 100, 80, 125, 0.05, 0.2, 1.0, "put")
        assert p >= 0.0

