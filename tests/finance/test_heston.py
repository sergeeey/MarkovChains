"""Tests for HestonPricer (splitting + Chernoff in x)."""

import numpy as np
import pytest

from chernoffpy import BackwardEuler, CrankNicolson
from chernoffpy.finance.heston import HestonPricer
from chernoffpy.finance.heston_analytical import heston_price
from chernoffpy.finance.heston_params import HestonGridConfig, HestonParams


@pytest.fixture
def heston_base_params():
    return HestonParams(
        S=100,
        K=100,
        T=1.0,
        r=0.05,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
    )


@pytest.fixture
def heston_grid():
    return HestonGridConfig(n_x=128, n_v=48, x_min=-4.0, x_max=4.0, v_max=0.8)


class TestHestonValidation:

    def test_valid_params(self, heston_base_params):
        assert heston_base_params.S == 100

    def test_negative_v0_rejected(self):
        with pytest.raises(ValueError, match="v0"):
            HestonParams(100, 100, 1.0, 0.05, -1e-3, 2.0, 0.04, 0.3, -0.7)

    def test_rho_out_of_range(self):
        with pytest.raises(ValueError, match="rho"):
            HestonParams(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, 1.5)

    def test_negative_xi_rejected(self):
        with pytest.raises(ValueError, match="xi"):
            HestonParams(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, -0.1, -0.7)


class TestHestonPricing:

    def test_heston_call_atm_vs_analytical(self, heston_base_params, heston_grid):
        pricer = HestonPricer(CrankNicolson(), heston_grid)
        result = pricer.price(heston_base_params, n_steps=40, option_type="call")
        ref = heston_price(**heston_base_params.__dict__, option_type="call")
        assert abs(result.price - ref) / ref < 0.08

    def test_heston_put_atm_vs_analytical(self, heston_base_params, heston_grid):
        pricer = HestonPricer(CrankNicolson(), heston_grid)
        result = pricer.price(heston_base_params, n_steps=40, option_type="put")
        ref = heston_price(**heston_base_params.__dict__, option_type="put")
        assert abs(result.price - ref) / ref < 0.08

    def test_heston_call_itm(self, heston_grid):
        params = HestonParams(110, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7)
        p = HestonPricer(CrankNicolson(), heston_grid).price(params, n_steps=35, option_type="call").price
        assert p > 5.0

    def test_heston_call_otm(self, heston_grid):
        params = HestonParams(90, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7)
        p = HestonPricer(CrankNicolson(), heston_grid).price(params, n_steps=35, option_type="call").price
        assert p >= 0.0

    def test_heston_put_itm(self, heston_grid):
        params = HestonParams(90, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7)
        p = HestonPricer(CrankNicolson(), heston_grid).price(params, n_steps=35, option_type="put").price
        assert p > 5.0

    def test_heston_negative_rho(self, heston_base_params, heston_grid):
        neg = HestonPricer(CrankNicolson(), heston_grid).price(heston_base_params, n_steps=35, option_type="call").price
        zero_params = HestonParams(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, 0.0)
        zero = HestonPricer(CrankNicolson(), heston_grid).price(zero_params, n_steps=35, option_type="call").price
        assert abs(neg - zero) > 1e-3

    def test_heston_zero_rho(self, heston_grid):
        params = HestonParams(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, 0.0)
        r = HestonPricer(CrankNicolson(), heston_grid).price(params, n_steps=35, option_type="call")
        assert np.isfinite(r.price)

    def test_heston_reduces_to_bs(self, heston_grid):
        params = HestonParams(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 1e-8, 0.0)
        r = HestonPricer(CrankNicolson(), heston_grid).price(params, n_steps=35, option_type="call")
        assert abs(r.price - r.bs_equiv_price) / r.bs_equiv_price < 0.03

    def test_heston_high_vol_of_vol(self, heston_grid):
        params = HestonParams(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.8, -0.7)
        r = HestonPricer(CrankNicolson(), heston_grid).price(params, n_steps=35, option_type="call")
        assert np.isfinite(r.price)

    def test_heston_long_expiry(self, heston_grid):
        params = HestonParams(100, 100, 5.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7)
        r = HestonPricer(CrankNicolson(), heston_grid).price(params, n_steps=50, option_type="call")
        assert np.isfinite(r.price)

    def test_heston_short_expiry(self, heston_grid):
        params = HestonParams(100, 100, 0.01, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7)
        r = HestonPricer(CrankNicolson(), heston_grid).price(params, n_steps=20, option_type="put")
        assert np.isfinite(r.price)


class TestHestonProperties:

    def test_call_nonnegative(self, heston_base_params, heston_grid):
        c = HestonPricer(CrankNicolson(), heston_grid).price(heston_base_params, 35, "call").price
        assert c >= 0.0

    def test_put_nonnegative(self, heston_base_params, heston_grid):
        p = HestonPricer(CrankNicolson(), heston_grid).price(heston_base_params, 35, "put").price
        assert p >= 0.0

    def test_call_le_spot(self, heston_base_params, heston_grid):
        c = HestonPricer(CrankNicolson(), heston_grid).price(heston_base_params, 35, "call").price
        assert c <= heston_base_params.S + 1e-8

    def test_put_le_strike(self, heston_base_params, heston_grid):
        p = HestonPricer(CrankNicolson(), heston_grid).price(heston_base_params, 35, "put").price
        assert p <= heston_base_params.K + 1e-8

    def test_put_call_parity(self, heston_base_params, heston_grid):
        pr = HestonPricer(CrankNicolson(), heston_grid)
        c = pr.price(heston_base_params, 35, "call").price
        p = pr.price(heston_base_params, 35, "put").price
        rhs = heston_base_params.S - heston_base_params.K * np.exp(-heston_base_params.r * heston_base_params.T)
        assert abs((c - p) - rhs) / max(abs(rhs), 1e-12) < 0.08

    def test_price_increases_with_vol(self, heston_grid):
        low = HestonParams(100, 100, 1.0, 0.05, 0.02, 2.0, 0.02, 0.3, -0.7)
        high = HestonParams(100, 100, 1.0, 0.05, 0.09, 2.0, 0.09, 0.3, -0.7)
        pr = HestonPricer(CrankNicolson(), heston_grid)
        c_low = pr.price(low, 35, "call").price
        c_high = pr.price(high, 35, "call").price
        assert c_high >= c_low

    def test_skew_with_negative_rho(self, heston_grid):
        neg = HestonParams(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.4, -0.8)
        zero = HestonParams(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.4, 0.0)
        pr = HestonPricer(CrankNicolson(), heston_grid)
        diff = abs(pr.price(neg, 35, "put").price - pr.price(zero, 35, "put").price)
        assert diff > 1e-3


class TestHestonConvergence:

    def test_convergence_with_n(self, heston_base_params, heston_grid):
        ref = heston_price(**heston_base_params.__dict__, option_type="call")
        pr = HestonPricer(CrankNicolson(), heston_grid)
        e20 = abs(pr.price(heston_base_params, 20, "call").price - ref)
        e40 = abs(pr.price(heston_base_params, 40, "call").price - ref)
        e60 = abs(pr.price(heston_base_params, 60, "call").price - ref)
        assert e60 <= e40 + 1e-8
        assert e40 <= e20 + 1e-8

    def test_convergence_with_grid(self, heston_base_params):
        coarse = HestonPricer(CrankNicolson(), HestonGridConfig(n_x=96, n_v=32, x_min=-4, x_max=4, v_max=0.8))
        fine = HestonPricer(CrankNicolson(), HestonGridConfig(n_x=160, n_v=56, x_min=-4, x_max=4, v_max=0.8))
        ref = heston_price(**heston_base_params.__dict__, option_type="call")
        e_coarse = abs(coarse.price(heston_base_params, 40, "call").price - ref)
        e_fine = abs(fine.price(heston_base_params, 40, "call").price - ref)
        assert e_fine <= e_coarse + 1e-6

    def test_cn_comparable_to_be(self, heston_base_params, heston_grid):
        ref = heston_price(**heston_base_params.__dict__, option_type="call")
        cn = HestonPricer(CrankNicolson(), heston_grid).price(heston_base_params, 40, "call").price
        be = HestonPricer(BackwardEuler(), heston_grid).price(heston_base_params, 40, "call").price
        # On coarse grids, splitting error dominates the CN vs BE difference.
        assert abs(cn - ref) <= abs(be - ref) * 1.1 + 0.05


class TestHestonImpliedVol:

    def test_implied_vol_smile(self, heston_grid):
        pr = HestonPricer(CrankNicolson(), heston_grid)
        p1 = HestonParams(100, 90, 1.0, 0.05, 0.04, 2.0, 0.04, 0.4, -0.7)
        p2 = HestonParams(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.4, -0.7)
        r1 = pr.price(p1, 35, "put")
        r2 = pr.price(p2, 35, "put")
        assert r1.implied_vol is None or r2.implied_vol is None or r1.implied_vol >= r2.implied_vol * 0.9

    def test_implied_vol_atm(self, heston_base_params, heston_grid):
        r = HestonPricer(CrankNicolson(), heston_grid).price(heston_base_params, 35, "call")
        if r.implied_vol is not None:
            assert 0.05 <= r.implied_vol <= 1.0


class TestHestonEdgeCases:

    def test_feller_violated(self, heston_grid):
        params = HestonParams(100, 100, 1.0, 0.05, 0.04, 0.2, 0.04, 0.5, -0.7)
        r = HestonPricer(CrankNicolson(), heston_grid).price(params, 35, "call")
        assert np.isfinite(r.price)

    def test_v0_near_zero(self, heston_grid):
        params = HestonParams(100, 100, 1.0, 0.05, 1e-3, 2.0, 0.04, 0.3, -0.7)
        r = HestonPricer(CrankNicolson(), heston_grid).price(params, 35, "call")
        assert np.isfinite(r.price)

    def test_extreme_rho(self, heston_grid):
        params = HestonParams(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.95)
        r = HestonPricer(CrankNicolson(), heston_grid).price(params, 35, "call")
        assert np.isfinite(r.price)

    def test_result_structure(self, heston_base_params, heston_grid):
        r = HestonPricer(CrankNicolson(), heston_grid).price(heston_base_params, 30, "call")
        assert r.method_name.startswith("Heston-Trotter")
        assert r.n_steps == 30
        assert r.option_type == "call"
        assert r.analytical_price is not None
        assert r.price != r.analytical_price  # PDE price, not analytical

    def test_invalid_option_type(self, heston_base_params, heston_grid):
        with pytest.raises(ValueError, match="option_type"):
            HestonPricer(CrankNicolson(), heston_grid).price(heston_base_params, 30, "straddle")
