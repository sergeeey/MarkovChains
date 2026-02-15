"""Tests for analytical barrier formulas (Reiner-Rubinstein)."""

import pytest

from chernoffpy.finance.barrier_analytical import barrier_analytical
from chernoffpy.finance.transforms import bs_exact_price
from chernoffpy.finance.validation import BarrierParams, MarketParams


class TestBarrierAnalyticalSanity:
    """Reiner-Rubinstein formulas: baseline checks."""

    def test_doc_nonnegative(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=90, barrier_type="down_and_out")
        price = barrier_analytical(market, params, "call")
        assert price >= 0.0

    def test_doc_le_vanilla(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=90, barrier_type="down_and_out")
        doc = barrier_analytical(market, params, "call")
        vanilla = bs_exact_price(market, "call")
        assert doc <= vanilla + 1e-12

    def test_in_out_parity(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)

        for option_type in ("call", "put"):
            for in_type, out_type, barrier in (
                ("down_and_in", "down_and_out", 90.0),
                ("up_and_in", "up_and_out", 120.0),
            ):
                p_in = barrier_analytical(market, BarrierParams(barrier, in_type), option_type)
                p_out = barrier_analytical(market, BarrierParams(barrier, out_type), option_type)
                vanilla = bs_exact_price(market, option_type)
                assert p_in + p_out == pytest.approx(vanilla, abs=1e-10)

    def test_barrier_at_zero_equals_vanilla(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=1e-6, barrier_type="down_and_out")
        doc = barrier_analytical(market, params, "call")
        vanilla = bs_exact_price(market, "call")
        assert doc == pytest.approx(vanilla, rel=1e-10, abs=1e-10)

    def test_barrier_at_spot_equals_zero(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=100.0, barrier_type="down_and_out")
        doc = barrier_analytical(market, params, "call")
        assert doc == pytest.approx(0.0, abs=1e-12)

    def test_doc_k_greater_b_numeric(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=90.0, barrier_type="down_and_out")
        price = barrier_analytical(market, params, "call")
        assert price == pytest.approx(8.6654716582, rel=1e-10)

    def test_uop_k_less_b_numeric(self):
        market = MarketParams(S=100, K=90, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=120.0, barrier_type="up_and_out")
        price = barrier_analytical(market, params, "put")
        assert price == pytest.approx(2.2675461345, rel=1e-10)

    def test_symmetry_call_put_parity_separate(self):
        market = MarketParams(S=105, K=100, T=0.75, r=0.03, sigma=0.25)
        down_b = 85.0
        up_b = 130.0

        for option_type in ("call", "put"):
            d_in = barrier_analytical(market, BarrierParams(down_b, "down_and_in"), option_type)
            d_out = barrier_analytical(market, BarrierParams(down_b, "down_and_out"), option_type)
            u_in = barrier_analytical(market, BarrierParams(up_b, "up_and_in"), option_type)
            u_out = barrier_analytical(market, BarrierParams(up_b, "up_and_out"), option_type)
            vanilla = bs_exact_price(market, option_type)

            assert d_in + d_out == pytest.approx(vanilla, abs=1e-10)
            assert u_in + u_out == pytest.approx(vanilla, abs=1e-10)
