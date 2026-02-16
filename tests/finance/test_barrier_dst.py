"""Tests for DST-based barrier pricers."""

from __future__ import annotations

import numpy as np
import pytest

from chernoffpy import BackwardEuler, CrankNicolson
from chernoffpy.finance.barrier import BarrierPricer
from chernoffpy.finance.barrier_analytical import barrier_analytical
from chernoffpy.finance.barrier_dst import BarrierDSTPricer, DoubleBarrierDSTPricer
from chernoffpy.finance.double_barrier_analytical import double_barrier_analytical
from chernoffpy.finance.transforms import bs_exact_price
from chernoffpy.finance.validation import (
    BarrierParams,
    DoubleBarrierParams,
    GridConfig,
    MarketParams,
)


def _rel_error(a: float, b: float) -> float:
    if abs(b) < 1e-12:
        return abs(a - b)
    return abs(a - b) / abs(b)


@pytest.fixture
def grid_dst() -> GridConfig:
    return GridConfig(N=2048, L=8.0, taper_width=2.0)


@pytest.fixture
def pricer_dst_cn(grid_dst: GridConfig) -> BarrierDSTPricer:
    return BarrierDSTPricer(CrankNicolson(), grid_dst)


@pytest.fixture
def pricer_dst_be(grid_dst: GridConfig) -> BarrierDSTPricer:
    return BarrierDSTPricer(BackwardEuler(), grid_dst)


class TestBarrierDSTValidation:

    def test_down_barrier_must_be_below_spot(self, pricer_dst_cn: BarrierDSTPricer):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=100.0, barrier_type="down_and_out")
        with pytest.raises(ValueError, match="Down barrier"):
            pricer_dst_cn.price(market, params)

    def test_up_barrier_must_be_above_spot(self, pricer_dst_cn: BarrierDSTPricer):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=100.0, barrier_type="up_and_out")
        with pytest.raises(ValueError, match="Up barrier"):
            pricer_dst_cn.price(market, params)

    def test_invalid_option_type(self, pricer_dst_cn: BarrierDSTPricer):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=90.0, barrier_type="down_and_out")
        with pytest.raises(ValueError, match="option_type"):
            pricer_dst_cn.price(market, params, option_type="straddle")

    def test_invalid_steps(self, pricer_dst_cn: BarrierDSTPricer):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=90.0, barrier_type="down_and_out")
        with pytest.raises(ValueError, match="n_steps"):
            pricer_dst_cn.price(market, params, n_steps=0)


class TestDSTvsAnalytical:

    def test_doc_atm_vs_analytical(self, pricer_dst_cn: BarrierDSTPricer):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=90.0, barrier_type="down_and_out")
        num = pricer_dst_cn.price(market, params, n_steps=140, option_type="call").price
        ref = barrier_analytical(market, params, "call")
        assert _rel_error(num, ref) < 0.015

    def test_doc_near_spot_vs_analytical(self, pricer_dst_cn: BarrierDSTPricer):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=99.0, barrier_type="down_and_out")
        num = pricer_dst_cn.price(market, params, n_steps=180, option_type="call").price
        ref = barrier_analytical(market, params, "call")
        assert _rel_error(num, ref) < 0.035

    def test_doc_far_barrier_near_vanilla(self, pricer_dst_cn: BarrierDSTPricer):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=50.0, barrier_type="down_and_out")
        num = pricer_dst_cn.price(market, params, n_steps=100, option_type="call").price
        vanilla = bs_exact_price(market, "call")
        assert abs(num - vanilla) / vanilla < 0.008

    def test_uoc_atm_vs_analytical(self, pricer_dst_cn: BarrierDSTPricer):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=120.0, barrier_type="up_and_out")
        num = pricer_dst_cn.price(market, params, n_steps=140, option_type="call").price
        ref = barrier_analytical(market, params, "call")
        assert _rel_error(num, ref) < 0.015

    def test_dop_atm_vs_analytical(self, pricer_dst_cn: BarrierDSTPricer):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=90.0, barrier_type="down_and_out")
        num = pricer_dst_cn.price(market, params, n_steps=140, option_type="put").price
        ref = barrier_analytical(market, params, "put")
        assert _rel_error(num, ref) < 0.015

    def test_uop_atm_vs_analytical(self, pricer_dst_cn: BarrierDSTPricer):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=120.0, barrier_type="up_and_out")
        num = pricer_dst_cn.price(market, params, n_steps=140, option_type="put").price
        ref = barrier_analytical(market, params, "put")
        assert _rel_error(num, ref) < 0.015


class TestDSTDoubleBarrier:

    def test_dko_corridor_vs_analytical(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = DoubleBarrierParams(80.0, 120.0, "double_knock_out")
        pricer = DoubleBarrierDSTPricer(CrankNicolson(), GridConfig(N=2048, L=8.0, taper_width=2.0))
        num = pricer.price(market, params, n_steps=140, option_type="call").price
        ref = double_barrier_analytical(100, 100, 80, 120, 0.05, 0.2, 1.0, "call")
        assert _rel_error(num, ref) < 0.06

    def test_dko_narrow_corridor(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = DoubleBarrierParams(95.0, 105.0, "double_knock_out")
        pricer = DoubleBarrierDSTPricer(CrankNicolson(), GridConfig(N=2048, L=8.0, taper_width=2.0))
        num = pricer.price(market, params, n_steps=180, option_type="call").price
        ref = double_barrier_analytical(100, 100, 95, 105, 0.05, 0.2, 1.0, "call")
        assert abs(num - ref) < 1e-3

    def test_dko_wide_corridor_near_vanilla(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = DoubleBarrierParams(50.0, 200.0, "double_knock_out")
        pricer = DoubleBarrierDSTPricer(CrankNicolson(), GridConfig(N=1024, L=8.0, taper_width=2.0))
        num = pricer.price(market, params, n_steps=100, option_type="call").price
        vanilla = bs_exact_price(market, "call")
        assert abs(num - vanilla) / vanilla < 0.01

    def test_double_knockin_parity(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        out_params = DoubleBarrierParams(80.0, 120.0, "double_knock_out")
        in_params = DoubleBarrierParams(80.0, 120.0, "double_knock_in")
        pricer = DoubleBarrierDSTPricer(CrankNicolson(), GridConfig(N=1024, L=8.0, taper_width=2.0))
        out_r = pricer.price(market, out_params, n_steps=100, option_type="call")
        in_r = pricer.price(market, in_params, n_steps=100, option_type="call")
        assert in_r.price + out_r.price == pytest.approx(out_r.vanilla_price, abs=5e-3)


class TestDSTConvergence:

    def test_convergence_with_n_steps(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=99.0, barrier_type="down_and_out")
        ref = barrier_analytical(market, params, "call")

        pricer = BarrierDSTPricer(CrankNicolson(), GridConfig(N=2048, L=8.0, taper_width=2.0))
        e20 = abs(pricer.price(market, params, n_steps=20, option_type="call").price - ref)
        e60 = abs(pricer.price(market, params, n_steps=60, option_type="call").price - ref)
        e120 = abs(pricer.price(market, params, n_steps=120, option_type="call").price - ref)

        assert e120 <= e60 + 1e-10
        assert e60 <= e20 + 1e-10

    def test_convergence_with_n_grid(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=99.0, barrier_type="down_and_out")
        ref = barrier_analytical(market, params, "call")

        e512 = abs(
            BarrierDSTPricer(CrankNicolson(), GridConfig(N=512, L=8.0, taper_width=2.0))
            .price(market, params, n_steps=120, option_type="call")
            .price
            - ref
        )
        e1024 = abs(
            BarrierDSTPricer(CrankNicolson(), GridConfig(N=1024, L=8.0, taper_width=2.0))
            .price(market, params, n_steps=120, option_type="call")
            .price
            - ref
        )
        e2048 = abs(
            BarrierDSTPricer(CrankNicolson(), GridConfig(N=2048, L=8.0, taper_width=2.0))
            .price(market, params, n_steps=120, option_type="call")
            .price
            - ref
        )

        assert e2048 <= e1024 + 1e-10
        assert e1024 <= e512 + 1e-10

    def test_cn_not_worse_than_be(self, pricer_dst_cn: BarrierDSTPricer, pricer_dst_be: BarrierDSTPricer):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=90.0, barrier_type="down_and_out")
        ref = barrier_analytical(market, params, "call")
        err_cn = abs(pricer_dst_cn.price(market, params, n_steps=120, option_type="call").price - ref)
        err_be = abs(pricer_dst_be.price(market, params, n_steps=120, option_type="call").price - ref)
        assert err_cn <= err_be + 1e-10

    def test_dst_more_accurate_than_fft_for_near_barrier(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        params = BarrierParams(barrier=99.0, barrier_type="down_and_out")
        ref = barrier_analytical(market, params, "call")

        dst_pricer = BarrierDSTPricer(CrankNicolson(), GridConfig(N=2048, L=8.0, taper_width=2.0))
        fft_pricer = BarrierPricer(CrankNicolson(), GridConfig(N=2048, L=8.0, taper_width=2.0))

        err_dst = abs(dst_pricer.price(market, params, n_steps=120, option_type="call").price - ref)
        err_fft = abs(fft_pricer.price(market, params, n_steps=120, option_type="call").price - ref)

        assert err_dst < err_fft


class TestDSTKnockinParity:

    def test_dic_parity(self, pricer_dst_cn: BarrierDSTPricer):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        out_params = BarrierParams(barrier=90.0, barrier_type="down_and_out")
        in_params = BarrierParams(barrier=90.0, barrier_type="down_and_in")

        out_r = pricer_dst_cn.price(market, out_params, n_steps=120, option_type="call")
        in_r = pricer_dst_cn.price(market, in_params, n_steps=120, option_type="call")
        assert in_r.price + out_r.price == pytest.approx(out_r.vanilla_price, abs=5e-3)

    def test_uic_parity(self, pricer_dst_cn: BarrierDSTPricer):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        out_params = BarrierParams(barrier=120.0, barrier_type="up_and_out")
        in_params = BarrierParams(barrier=120.0, barrier_type="up_and_in")

        out_r = pricer_dst_cn.price(market, out_params, n_steps=120, option_type="call")
        in_r = pricer_dst_cn.price(market, in_params, n_steps=120, option_type="call")
        assert in_r.price + out_r.price == pytest.approx(out_r.vanilla_price, abs=5e-3)

    def test_dip_parity(self, pricer_dst_cn: BarrierDSTPricer):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        out_params = BarrierParams(barrier=90.0, barrier_type="down_and_out")
        in_params = BarrierParams(barrier=90.0, barrier_type="down_and_in")

        out_r = pricer_dst_cn.price(market, out_params, n_steps=120, option_type="put")
        in_r = pricer_dst_cn.price(market, in_params, n_steps=120, option_type="put")
        assert in_r.price + out_r.price == pytest.approx(out_r.vanilla_price, abs=5e-3)

    def test_uip_parity(self, pricer_dst_cn: BarrierDSTPricer):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        out_params = BarrierParams(barrier=120.0, barrier_type="up_and_out")
        in_params = BarrierParams(barrier=120.0, barrier_type="up_and_in")

        out_r = pricer_dst_cn.price(market, out_params, n_steps=120, option_type="put")
        in_r = pricer_dst_cn.price(market, in_params, n_steps=120, option_type="put")
        assert in_r.price + out_r.price == pytest.approx(out_r.vanilla_price, abs=5e-3)


class TestDSTProperties:

    def test_knockout_le_vanilla(self, pricer_dst_cn: BarrierDSTPricer):
        market = MarketParams(S=105, K=100, T=1.0, r=0.05, sigma=0.20)
        for bt, b in (("down_and_out", 90.0), ("up_and_out", 130.0)):
            for opt in ("call", "put"):
                p = pricer_dst_cn.price(market, BarrierParams(b, bt), n_steps=100, option_type=opt).price
                assert p <= bs_exact_price(market, opt) + 1e-8

    def test_knockin_le_vanilla(self, pricer_dst_cn: BarrierDSTPricer):
        market = MarketParams(S=105, K=100, T=1.0, r=0.05, sigma=0.20)
        for bt, b in (("down_and_in", 90.0), ("up_and_in", 130.0)):
            for opt in ("call", "put"):
                p = pricer_dst_cn.price(market, BarrierParams(b, bt), n_steps=100, option_type=opt).price
                assert p <= bs_exact_price(market, opt) + 1e-8

    def test_nonnegative_and_finite(self, pricer_dst_cn: BarrierDSTPricer):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        for bt, b in (
            ("down_and_out", 90.0),
            ("down_and_in", 90.0),
            ("up_and_out", 120.0),
            ("up_and_in", 120.0),
        ):
            for opt in ("call", "put"):
                p = pricer_dst_cn.price(market, BarrierParams(b, bt), n_steps=100, option_type=opt).price
                assert np.isfinite(p)
                assert p >= 0.0

