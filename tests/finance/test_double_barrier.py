"""Tests for double barrier pricer."""

import numpy as np
import pytest

from chernoffpy import BackwardEuler, CrankNicolson
from chernoffpy.finance.barrier import BarrierPricer
from chernoffpy.finance.double_barrier import DoubleBarrierPricer
from chernoffpy.finance.double_barrier_analytical import double_barrier_analytical
from chernoffpy.finance.transforms import bs_exact_price
from chernoffpy.finance.validation import DoubleBarrierParams, MarketParams, BarrierParams


class TestDoubleBarrierValidation:

    def test_spot_below_lower_barrier_raises(self):
        p = DoubleBarrierPricer(CrankNicolson())
        m = MarketParams(S=80, K=100, T=1.0, r=0.05, sigma=0.2)
        bp = DoubleBarrierParams(lower_barrier=80, upper_barrier=120, barrier_type="double_knock_out")
        with pytest.raises(ValueError, match="Spot"):
            p.price(m, bp)

    def test_spot_above_upper_barrier_raises(self):
        p = DoubleBarrierPricer(CrankNicolson())
        m = MarketParams(S=120, K=100, T=1.0, r=0.05, sigma=0.2)
        bp = DoubleBarrierParams(lower_barrier=80, upper_barrier=120, barrier_type="double_knock_out")
        with pytest.raises(ValueError, match="Spot"):
            p.price(m, bp)

    def test_lower_ge_upper_raises(self):
        with pytest.raises(ValueError, match="lower_barrier"):
            DoubleBarrierParams(lower_barrier=120, upper_barrier=120, barrier_type="double_knock_out")

    def test_negative_barrier_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            DoubleBarrierParams(lower_barrier=-1, upper_barrier=120, barrier_type="double_knock_out")

    def test_negative_rebate_raises(self):
        with pytest.raises(ValueError, match="rebate"):
            DoubleBarrierParams(lower_barrier=80, upper_barrier=120, barrier_type="double_knock_out", rebate=-1)


class TestDoubleKnockoutPricing:

    def test_dko_call_atm_vs_analytical(self):
        p = DoubleBarrierPricer(CrankNicolson())
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        bp = DoubleBarrierParams(80, 120, "double_knock_out")
        num = p.price(m, bp, n_steps=80, option_type="call").price
        ref = double_barrier_analytical(100, 100, 80, 120, 0.05, 0.2, 1.0, "call")
        assert abs(num - ref) / ref < 0.08

    def test_dko_call_wide_corridor(self):
        p = DoubleBarrierPricer(CrankNicolson())
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        bp = DoubleBarrierParams(50, 200, "double_knock_out")
        num = p.price(m, bp, n_steps=80, option_type="call").price
        vanilla = bs_exact_price(m, "call")
        assert abs(num - vanilla) / vanilla < 0.01

    def test_dko_call_narrow_corridor(self):
        p = DoubleBarrierPricer(CrankNicolson())
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        bp = DoubleBarrierParams(95, 105, "double_knock_out")
        num = p.price(m, bp, n_steps=80, option_type="call").price
        assert num < 0.5

    def test_dko_put_atm_vs_analytical(self):
        p = DoubleBarrierPricer(CrankNicolson())
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        bp = DoubleBarrierParams(80, 120, "double_knock_out")
        num = p.price(m, bp, n_steps=80, option_type="put").price
        ref = double_barrier_analytical(100, 100, 80, 120, 0.05, 0.2, 1.0, "put")
        assert abs(num - ref) / (ref if ref > 1e-8 else 1) < 0.12

    def test_dko_call_asymmetric(self):
        p = DoubleBarrierPricer(CrankNicolson())
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        bp = DoubleBarrierParams(70, 110, "double_knock_out")
        num = p.price(m, bp, n_steps=80, option_type="call").price
        assert np.isfinite(num) and num >= 0

    def test_dko_low_vol(self):
        p = DoubleBarrierPricer(CrankNicolson())
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.1)
        bp = DoubleBarrierParams(80, 120, "double_knock_out")
        num = p.price(m, bp, n_steps=80, option_type="call").price
        assert np.isfinite(num) and num >= 0

    def test_dko_high_vol(self):
        p = DoubleBarrierPricer(CrankNicolson())
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.6)
        bp = DoubleBarrierParams(80, 120, "double_knock_out")
        r = p.price(m, bp, n_steps=80, option_type="call")
        assert np.isfinite(r.price) and r.price < r.vanilla_price


class TestDoubleKnockinParity:

    def test_dki_call_parity(self):
        p = DoubleBarrierPricer(CrankNicolson())
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        dko = p.price(m, DoubleBarrierParams(80, 120, "double_knock_out"), n_steps=80, option_type="call")
        dki = p.price(m, DoubleBarrierParams(80, 120, "double_knock_in"), n_steps=80, option_type="call")
        assert abs(dki.price + dko.price - dko.vanilla_price) < 0.08

    def test_dki_put_parity(self):
        p = DoubleBarrierPricer(CrankNicolson())
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        dko = p.price(m, DoubleBarrierParams(80, 120, "double_knock_out"), n_steps=80, option_type="put")
        dki = p.price(m, DoubleBarrierParams(80, 120, "double_knock_in"), n_steps=80, option_type="put")
        assert abs(dki.price + dko.price - dko.vanilla_price) < 0.08

    def test_dki_call_vs_analytical(self):
        p = DoubleBarrierPricer(CrankNicolson())
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        num = p.price(m, DoubleBarrierParams(80, 120, "double_knock_in"), n_steps=80, option_type="call").price
        ref = double_barrier_analytical(100, 100, 80, 120, 0.05, 0.2, 1.0, "call", "double_knock_in")
        assert abs(num - ref) / (ref if ref > 1e-8 else 1) < 0.08

    def test_dki_wide_corridor_near_zero(self):
        p = DoubleBarrierPricer(CrankNicolson())
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        num = p.price(m, DoubleBarrierParams(1, 1000, "double_knock_in"), n_steps=80, option_type="call").price
        assert num < 1.0

    def test_dki_narrow_corridor_near_vanilla(self):
        p = DoubleBarrierPricer(CrankNicolson())
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        r = p.price(m, DoubleBarrierParams(95, 105, "double_knock_in"), n_steps=80, option_type="call")
        assert abs(r.price - r.vanilla_price) < 1.0


class TestDoubleBarrierProperties:

    def test_dko_le_vanilla(self):
        p = DoubleBarrierPricer(CrankNicolson())
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        r = p.price(m, DoubleBarrierParams(80, 120, "double_knock_out"), n_steps=80, option_type="call")
        assert r.price <= r.vanilla_price + 1e-8

    def test_dki_le_vanilla(self):
        p = DoubleBarrierPricer(CrankNicolson())
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        r = p.price(m, DoubleBarrierParams(80, 120, "double_knock_in"), n_steps=80, option_type="call")
        assert r.price <= r.vanilla_price + 1e-8

    def test_dko_le_single_ko(self):
        dp = DoubleBarrierPricer(CrankNicolson())
        sp = BarrierPricer(CrankNicolson())
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)

        dko = dp.price(m, DoubleBarrierParams(80, 120, "double_knock_out"), n_steps=80, option_type="call").price
        doc = sp.price(m, BarrierParams(80, "down_and_out"), n_steps=80, option_type="call").price
        uoc = sp.price(m, BarrierParams(120, "up_and_out"), n_steps=80, option_type="call").price
        assert dko <= min(doc, uoc) + 1e-8

    def test_nonnegative(self):
        p = DoubleBarrierPricer(CrankNicolson())
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        for bt in ("double_knock_out", "double_knock_in"):
            for ot in ("call", "put"):
                r = p.price(m, DoubleBarrierParams(80, 120, bt), n_steps=80, option_type=ot)
                assert r.price >= 0.0

    def test_wider_corridor_higher_price(self):
        p = DoubleBarrierPricer(CrankNicolson())
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        narrow = p.price(m, DoubleBarrierParams(90, 110, "double_knock_out"), n_steps=80, option_type="call").price
        wide = p.price(m, DoubleBarrierParams(80, 120, "double_knock_out"), n_steps=80, option_type="call").price
        assert wide >= narrow


class TestDoubleBarrierConvergence:

    def test_convergence_with_n(self):
        p = DoubleBarrierPricer(CrankNicolson())
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        ref = double_barrier_analytical(100, 100, 80, 120, 0.05, 0.2, 1.0, "call")
        e20 = abs(p.price(m, DoubleBarrierParams(80, 120, "double_knock_out"), n_steps=20, option_type="call").price - ref)
        e50 = abs(p.price(m, DoubleBarrierParams(80, 120, "double_knock_out"), n_steps=50, option_type="call").price - ref)
        e80 = abs(p.price(m, DoubleBarrierParams(80, 120, "double_knock_out"), n_steps=80, option_type="call").price - ref)
        assert e80 <= e50 + 1e-8
        assert e50 <= e20 + 1e-8

    def test_cn_better_than_be(self):
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        ref = double_barrier_analytical(100, 100, 80, 120, 0.05, 0.2, 1.0, "call")
        cn = DoubleBarrierPricer(CrankNicolson()).price(m, DoubleBarrierParams(80, 120, "double_knock_out"), n_steps=80, option_type="call").price
        be = DoubleBarrierPricer(BackwardEuler()).price(m, DoubleBarrierParams(80, 120, "double_knock_out"), n_steps=80, option_type="call").price
        assert abs(cn - ref) <= abs(be - ref) + 1e-8


class TestDoubleBarrierEdgeCases:

    def test_short_expiry(self):
        p = DoubleBarrierPricer(CrankNicolson())
        m = MarketParams(S=100, K=100, T=1/365, r=0.05, sigma=0.2)
        r = p.price(m, DoubleBarrierParams(80, 120, "double_knock_out"), n_steps=50, option_type="call")
        assert np.isfinite(r.price)

    def test_high_vol_wide_corridor(self):
        p = DoubleBarrierPricer(CrankNicolson())
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.8)
        r = p.price(m, DoubleBarrierParams(50, 200, "double_knock_out"), n_steps=80, option_type="call")
        assert np.isfinite(r.price)

