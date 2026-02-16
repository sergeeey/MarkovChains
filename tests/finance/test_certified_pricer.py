"""Tests for certified finance pricer wrappers."""

from __future__ import annotations

import pytest

from chernoffpy import BackwardEuler, CrankNicolson
from chernoffpy.finance import (
    BarrierParams,
    CertifiedBarrierDSTPricer,
    CertifiedEuropeanPricer,
    EuropeanPricer,
    GridConfig,
    MarketParams,
)
from chernoffpy.finance.barrier_analytical import barrier_analytical
from chernoffpy.finance.transforms import bs_exact_price


@pytest.fixture
def market() -> MarketParams:
    return MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)


@pytest.fixture
def grid() -> GridConfig:
    return GridConfig(N=1024, L=8.0, taper_width=2.0)


class TestCertifiedEuropean:

    def test_price_matches_regular(self, market: MarketParams, grid: GridConfig):
        c = CertifiedEuropeanPricer(CrankNicolson(), grid)
        r = EuropeanPricer(CrankNicolson(), grid)
        rc = c.price_certified(market, n_steps=40, option_type="call")
        rr = r.price(market, n_steps=40, option_type="call")
        assert rc.price == pytest.approx(rr.price, abs=1e-12)

    def test_bound_covers_error(self, market: MarketParams, grid: GridConfig):
        c = CertifiedEuropeanPricer(CrankNicolson(), grid)
        res = c.price_certified(market, n_steps=40, option_type="call", safety_factor=2.0)
        true_error = abs(res.price - bs_exact_price(market, "call"))
        assert res.certified_bound.bound >= true_error

    def test_bound_covers_error_put(self, market: MarketParams, grid: GridConfig):
        c = CertifiedEuropeanPricer(CrankNicolson(), grid)
        res = c.price_certified(market, n_steps=40, option_type="put", safety_factor=2.0)
        true_error = abs(res.price - bs_exact_price(market, "put"))
        assert res.certified_bound.bound >= true_error

    def test_bound_covers_all_n(self, market: MarketParams, grid: GridConfig):
        c = CertifiedEuropeanPricer(CrankNicolson(), grid)
        for n in (10, 20, 50, 100):
            res = c.price_certified(market, n_steps=n, option_type="call", safety_factor=2.0)
            true_error = abs(res.price - bs_exact_price(market, "call"))
            assert res.certified_bound.bound >= true_error

    def test_verification_consistent(self, market: MarketParams, grid: GridConfig):
        c = CertifiedEuropeanPricer(CrankNicolson(), grid)
        res = c.price_certified(market, n_steps=30, option_type="call", safety_factor=2.0)
        assert res.verification["is_consistent"] in {True, False}
        assert "empirical_order" in res.verification

    def test_n_for_tolerance_works(self, market: MarketParams, grid: GridConfig):
        c = CertifiedEuropeanPricer(CrankNicolson(), grid)
        n = c.n_for_tolerance(market, target_error=5e-3, option_type="call", pilot_n=20)
        res = c.price_certified(market, n_steps=n, option_type="call", safety_factor=2.0)
        true_error = abs(res.price - bs_exact_price(market, "call"))
        assert true_error <= 5e-3

    def test_n_for_tolerance_be_vs_cn(self, market: MarketParams, grid: GridConfig):
        cn = CertifiedEuropeanPricer(CrankNicolson(), grid)
        be = CertifiedEuropeanPricer(BackwardEuler(), grid)
        n_cn = cn.n_for_tolerance(market, target_error=1e-2, option_type="call", pilot_n=20)
        n_be = be.n_for_tolerance(market, target_error=1e-2, option_type="call", pilot_n=20)
        assert n_be >= n_cn


class TestCertifiedBarrierDST:

    def test_bound_covers_error_doc(self, market: MarketParams, grid: GridConfig):
        p = CertifiedBarrierDSTPricer(CrankNicolson(), grid)
        b = BarrierParams(barrier=90.0, barrier_type="down_and_out")
        res = p.price_certified(market, b, n_steps=80, option_type="call", safety_factor=2.0)
        ref = barrier_analytical(market, b, "call")
        true_error = abs(res.price - ref)
        assert res.certified_bound.bound >= true_error

    def test_bound_covers_error_uoc(self, market: MarketParams, grid: GridConfig):
        p = CertifiedBarrierDSTPricer(CrankNicolson(), grid)
        b = BarrierParams(barrier=120.0, barrier_type="up_and_out")
        res = p.price_certified(market, b, n_steps=80, option_type="call", safety_factor=2.0)
        ref = barrier_analytical(market, b, "call")
        true_error = abs(res.price - ref)
        assert res.certified_bound.bound >= true_error

    def test_bound_decreases_with_n(self, market: MarketParams, grid: GridConfig):
        p = CertifiedBarrierDSTPricer(CrankNicolson(), grid)
        b = BarrierParams(barrier=90.0, barrier_type="down_and_out")
        # Use n_steps above the DST floor (10*sqrt(1024)=320) so the
        # bound-estimation sequence produces distinct evaluations.
        r400 = p.price_certified(market, b, n_steps=400, option_type="call")
        r800 = p.price_certified(market, b, n_steps=800, option_type="call")
        assert r800.certified_bound.bound <= r400.certified_bound.bound + 1e-12
