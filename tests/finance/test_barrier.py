"""Tests for BarrierPricer using Chernoff + Dirichlet projection."""

import numpy as np
import pytest

from chernoffpy import BackwardEuler, CrankNicolson
from chernoffpy.finance.barrier import BarrierPricer
from chernoffpy.finance.barrier_analytical import barrier_analytical
from chernoffpy.finance.transforms import bs_exact_price
from chernoffpy.finance.validation import BarrierParams, GridConfig, MarketParams


@pytest.fixture
def barrier_grid():
    return GridConfig(N=1024, L=8.0, taper_width=2.0)


@pytest.fixture
def barrier_pricer_cn(barrier_grid):
    return BarrierPricer(CrankNicolson(), barrier_grid)


@pytest.fixture
def barrier_pricer_be(barrier_grid):
    return BarrierPricer(BackwardEuler(), barrier_grid)


def _rel_error(a: float, b: float) -> float:
    if abs(b) < 1e-12:
        return abs(a - b)
    return abs(a - b) / abs(b)


class TestBarrierValidation:

    def test_down_barrier_must_be_below_spot(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=100.0, barrier_type="down_and_out")
        with pytest.raises(ValueError, match="Down barrier"):
            barrier_pricer_cn.price(market, params)

    def test_up_barrier_must_be_above_spot(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=100.0, barrier_type="up_and_out")
        with pytest.raises(ValueError, match="Up barrier"):
            barrier_pricer_cn.price(market, params)

    def test_negative_barrier_rejected(self):
        with pytest.raises(ValueError, match="barrier must be > 0"):
            BarrierParams(barrier=-1.0, barrier_type="down_and_out")

    def test_negative_rebate_rejected(self):
        with pytest.raises(ValueError, match="rebate must be >= 0"):
            BarrierParams(barrier=90.0, barrier_type="down_and_out", rebate=-0.1)

    def test_invalid_option_type(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=90.0, barrier_type="down_and_out")
        with pytest.raises(ValueError, match="option_type"):
            barrier_pricer_cn.price(market, params, option_type="straddle")


class TestKnockoutPricing:

    def test_doc_atm_vs_analytical(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=90.0, barrier_type="down_and_out")
        num = barrier_pricer_cn.price(market, params, n_steps=80, option_type="call").price
        ref = barrier_analytical(market, params, "call")
        assert _rel_error(num, ref) < 0.01

    def test_doc_itm_vs_analytical(self, barrier_pricer_cn):
        market = MarketParams(S=110, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=90.0, barrier_type="down_and_out")
        num = barrier_pricer_cn.price(market, params, n_steps=80, option_type="call").price
        ref = barrier_analytical(market, params, "call")
        assert _rel_error(num, ref) < 0.01

    def test_doc_otm_vs_analytical(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=110, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=90.0, barrier_type="down_and_out")
        num = barrier_pricer_cn.price(market, params, n_steps=80, option_type="call").price
        ref = barrier_analytical(market, params, "call")
        assert _rel_error(num, ref) < 0.02

    def test_doc_barrier_near_spot(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=99.0, barrier_type="down_and_out")
        num = barrier_pricer_cn.price(market, params, n_steps=80, option_type="call").price
        vanilla = bs_exact_price(market, "call")
        assert np.isfinite(num)
        assert 0.0 <= num <= vanilla + 1e-6

    def test_doc_barrier_far_from_spot(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=50.0, barrier_type="down_and_out")
        num = barrier_pricer_cn.price(market, params, n_steps=80, option_type="call").price
        vanilla = bs_exact_price(market, "call")
        assert abs(num - vanilla) / vanilla < 0.005

    def test_uoc_atm_vs_analytical(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=120.0, barrier_type="up_and_out")
        num = barrier_pricer_cn.price(market, params, n_steps=80, option_type="call").price
        ref = barrier_analytical(market, params, "call")
        assert _rel_error(num, ref) < 0.01

    def test_uoc_barrier_near_spot(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=101.0, barrier_type="up_and_out")
        num = barrier_pricer_cn.price(market, params, n_steps=80, option_type="call").price
        vanilla = bs_exact_price(market, "call")
        assert np.isfinite(num)
        assert 0.0 <= num <= vanilla + 1e-6

    def test_uoc_barrier_far(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=200.0, barrier_type="up_and_out")
        num = barrier_pricer_cn.price(market, params, n_steps=80, option_type="call").price
        vanilla = bs_exact_price(market, "call")
        assert abs(num - vanilla) / vanilla < 0.01

    def test_dop_atm_vs_analytical(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=90.0, barrier_type="down_and_out")
        num = barrier_pricer_cn.price(market, params, n_steps=80, option_type="put").price
        ref = barrier_analytical(market, params, "put")
        assert _rel_error(num, ref) < 0.01

    def test_uop_atm_vs_analytical(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=120.0, barrier_type="up_and_out")
        num = barrier_pricer_cn.price(market, params, n_steps=80, option_type="put").price
        ref = barrier_analytical(market, params, "put")
        assert _rel_error(num, ref) < 0.01


class TestKnockinParity:

    def test_dic_equals_vanilla_minus_doc(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        out_params = BarrierParams(barrier=90.0, barrier_type="down_and_out")
        in_params = BarrierParams(barrier=90.0, barrier_type="down_and_in")

        out_r = barrier_pricer_cn.price(market, out_params, n_steps=80, option_type="call")
        in_r = barrier_pricer_cn.price(market, in_params, n_steps=80, option_type="call")
        assert in_r.price == pytest.approx(out_r.vanilla_price - out_r.knockout_price, abs=1e-10)

    def test_uic_equals_vanilla_minus_uoc(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        out_params = BarrierParams(barrier=120.0, barrier_type="up_and_out")
        in_params = BarrierParams(barrier=120.0, barrier_type="up_and_in")

        out_r = barrier_pricer_cn.price(market, out_params, n_steps=80, option_type="call")
        in_r = barrier_pricer_cn.price(market, in_params, n_steps=80, option_type="call")
        assert in_r.price == pytest.approx(out_r.vanilla_price - out_r.knockout_price, abs=1e-10)

    def test_dip_parity(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        out_params = BarrierParams(barrier=90.0, barrier_type="down_and_out")
        in_params = BarrierParams(barrier=90.0, barrier_type="down_and_in")

        out_r = barrier_pricer_cn.price(market, out_params, n_steps=80, option_type="put")
        in_r = barrier_pricer_cn.price(market, in_params, n_steps=80, option_type="put")
        vanilla = out_r.vanilla_price
        assert in_r.price + out_r.price == pytest.approx(vanilla, abs=1e-10)

    def test_uip_parity(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        out_params = BarrierParams(barrier=120.0, barrier_type="up_and_out")
        in_params = BarrierParams(barrier=120.0, barrier_type="up_and_in")

        out_r = barrier_pricer_cn.price(market, out_params, n_steps=80, option_type="put")
        in_r = barrier_pricer_cn.price(market, in_params, n_steps=80, option_type="put")
        vanilla = out_r.vanilla_price
        assert in_r.price + out_r.price == pytest.approx(vanilla, abs=1e-10)

    def test_knockin_vs_analytical(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        in_params = BarrierParams(barrier=90.0, barrier_type="down_and_in")
        num = barrier_pricer_cn.price(market, in_params, n_steps=80, option_type="call").price
        ref = barrier_analytical(market, in_params, "call")
        assert _rel_error(num, ref) < 0.05


class TestBarrierProperties:

    def test_knockout_le_vanilla(self, barrier_pricer_cn):
        market = MarketParams(S=105, K=100, T=1.0, r=0.05, sigma=0.20)
        for bt, b in (("down_and_out", 90.0), ("up_and_out", 130.0)):
            for opt in ("call", "put"):
                p = barrier_pricer_cn.price(market, BarrierParams(b, bt), n_steps=80, option_type=opt).price
                vanilla = bs_exact_price(market, opt)
                assert p <= vanilla + 1e-6

    def test_knockin_le_vanilla(self, barrier_pricer_cn):
        market = MarketParams(S=105, K=100, T=1.0, r=0.05, sigma=0.20)
        for bt, b in (("down_and_in", 90.0), ("up_and_in", 130.0)):
            for opt in ("call", "put"):
                p = barrier_pricer_cn.price(market, BarrierParams(b, bt), n_steps=80, option_type=opt).price
                vanilla = bs_exact_price(market, opt)
                assert p <= vanilla + 1e-6

    def test_nonnegative_prices(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        for bt, b in (
            ("down_and_out", 90.0),
            ("down_and_in", 90.0),
            ("up_and_out", 120.0),
            ("up_and_in", 120.0),
        ):
            for opt in ("call", "put"):
                p = barrier_pricer_cn.price(market, BarrierParams(b, bt), n_steps=80, option_type=opt).price
                assert p >= 0.0
                assert np.isfinite(p)

    def test_barrier_far_below_equals_vanilla(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        doc = barrier_pricer_cn.price(
            market,
            BarrierParams(barrier=50.0, barrier_type="down_and_out"),
            n_steps=80,
            option_type="call",
        ).price
        vanilla = bs_exact_price(market, "call")
        assert abs(doc - vanilla) / vanilla < 0.005

    def test_up_barrier_far_above_equals_vanilla(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        uoc = barrier_pricer_cn.price(
            market,
            BarrierParams(barrier=200.0, barrier_type="up_and_out"),
            n_steps=80,
            option_type="call",
        ).price
        vanilla = bs_exact_price(market, "call")
        assert abs(uoc - vanilla) / vanilla < 0.01

    def test_barrier_near_spot_small_knockout(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        uoc = barrier_pricer_cn.price(
            market,
            BarrierParams(barrier=101.0, barrier_type="up_and_out"),
            n_steps=80,
            option_type="call",
        ).price
        assert uoc < 0.1


class TestBarrierConvergence:

    def test_doc_convergence_rate(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=90.0, barrier_type="down_and_out")
        ref = barrier_analytical(market, params, "call")

        e_300 = abs(barrier_pricer_cn.price(market, params, n_steps=300, option_type="call").price - ref)
        e_400 = abs(barrier_pricer_cn.price(market, params, n_steps=400, option_type="call").price - ref)
        e_500 = abs(barrier_pricer_cn.price(market, params, n_steps=500, option_type="call").price - ref)

        assert e_500 <= e_400 + 1e-12
        assert e_400 <= e_300 + 1e-12

    def test_doc_cn_better_than_be(self, barrier_pricer_cn, barrier_pricer_be):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=90.0, barrier_type="down_and_out")
        ref = barrier_analytical(market, params, "call")

        err_cn = abs(barrier_pricer_cn.price(market, params, n_steps=300, option_type="call").price - ref)
        err_be = abs(barrier_pricer_be.price(market, params, n_steps=300, option_type="call").price - ref)

        assert err_cn <= err_be + 1e-8


class TestBarrierEdgeCases:

    def test_short_expiry(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1 / 365, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=90.0, barrier_type="down_and_out")
        price = barrier_pricer_cn.price(market, params, n_steps=80, option_type="call").price
        assert np.isfinite(price)
        assert price >= 0.0

    def test_high_vol(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.80)
        params = BarrierParams(barrier=90.0, barrier_type="down_and_out")
        price = barrier_pricer_cn.price(market, params, n_steps=80, option_type="call").price
        assert np.isfinite(price)
        assert price >= 0.0

    def test_low_vol(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.05)
        params = BarrierParams(barrier=90.0, barrier_type="down_and_out")
        price = barrier_pricer_cn.price(market, params, n_steps=80, option_type="call").price
        assert np.isfinite(price)
        assert price >= 0.0

    def test_result_structure(self, barrier_pricer_cn):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=90.0, barrier_type="down_and_in")
        result = barrier_pricer_cn.price(market, params, n_steps=80, option_type="call")

        assert result.barrier_type == "down_and_in"
        assert result.n_steps == 80
        assert result.market == market
        assert result.barrier_params == params
        assert result.knockout_price >= 0.0
        assert result.vanilla_price >= 0.0
