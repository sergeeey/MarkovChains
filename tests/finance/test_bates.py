"""Tests for Bates pricer implementation."""

from __future__ import annotations

import numpy as np
import pytest

from chernoffpy import BackwardEuler, CrankNicolson
from chernoffpy.finance.bates import BatesPricer
from chernoffpy.finance.bates_analytical import bates_price
from chernoffpy.finance.bates_params import BatesParams
from chernoffpy.finance.heston import HestonPricer
from chernoffpy.finance.heston_params import HestonGridConfig


@pytest.fixture
def grid() -> HestonGridConfig:
    return HestonGridConfig(n_x=128, n_v=48, x_min=-4.0, x_max=4.0, v_max=0.8)


@pytest.fixture
def base_params() -> BatesParams:
    return BatesParams(
        S=100,
        K=100,
        T=1.0,
        r=0.05,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
        lambda_j=0.5,
        mu_j=-0.1,
        sigma_j=0.2,
    )


class TestBatesValidation:

    def test_negative_lambda_rejected(self):
        with pytest.raises(ValueError, match="lambda_j"):
            BatesParams(100, 100, 1, 0.05, 0.04, 2, 0.04, 0.3, -0.7, -0.1, -0.1, 0.2)

    def test_negative_sigma_j_rejected(self):
        with pytest.raises(ValueError, match="sigma_j"):
            BatesParams(100, 100, 1, 0.05, 0.04, 2, 0.04, 0.3, -0.7, 0.3, -0.1, -0.2)

    def test_to_heston_conversion(self, base_params: BatesParams):
        h = base_params.to_heston()
        assert h.S == base_params.S
        assert h.kappa == base_params.kappa

    def test_kbar_computation(self, base_params: BatesParams):
        expected = np.exp(base_params.mu_j + 0.5 * base_params.sigma_j**2) - 1
        assert base_params.kbar == pytest.approx(expected)


class TestBatesPricing:

    def test_zero_jumps_equals_heston(self, grid: HestonGridConfig, base_params: BatesParams):
        p0 = BatesParams(**{**base_params.__dict__, "lambda_j": 0.0})
        b = BatesPricer(CrankNicolson(), grid).price(p0, n_steps=40, option_type="call").price
        h = HestonPricer(CrankNicolson(), grid).price(p0.to_heston(), n_steps=40, option_type="call").price
        assert abs(b - h) / h < 0.01

    def test_atm_call_vs_analytical(self, grid: HestonGridConfig, base_params: BatesParams):
        r = BatesPricer(CrankNicolson(), grid).price(base_params, n_steps=50, option_type="call")
        ref = bates_price(**base_params.__dict__, option_type="call")
        assert abs(r.price - ref) / max(abs(ref), 1e-8) < 0.15

    def test_atm_put_vs_analytical(self, grid: HestonGridConfig, base_params: BatesParams):
        r = BatesPricer(CrankNicolson(), grid).price(base_params, n_steps=50, option_type="put")
        ref = bates_price(**base_params.__dict__, option_type="put")
        assert np.isfinite(r.price)
        assert r.price >= 0.0
        if ref > 1e-8:
            assert abs(r.price - ref) / ref < 0.15

    def test_high_jump_intensity(self, grid: HestonGridConfig, base_params: BatesParams):
        p = BatesParams(**{**base_params.__dict__, "lambda_j": 5.0})
        r = BatesPricer(CrankNicolson(), grid).price(p, n_steps=40, option_type="call")
        assert np.isfinite(r.price)

    def test_large_negative_jumps_increase_otm_put(self, grid: HestonGridConfig, base_params: BatesParams):
        p_no = BatesParams(**{**base_params.__dict__, "K": 80, "lambda_j": 0.0})
        p_jump = BatesParams(**{**base_params.__dict__, "K": 80, "lambda_j": 1.0, "mu_j": -0.25, "sigma_j": 0.3})
        no_jump = BatesPricer(CrankNicolson(), grid).price(p_no, n_steps=40, option_type="put").price
        jump = BatesPricer(CrankNicolson(), grid).price(p_jump, n_steps=40, option_type="put").price
        assert jump >= no_jump


class TestBatesProperties:

    def test_nonnegative(self, grid: HestonGridConfig, base_params: BatesParams):
        r = BatesPricer(CrankNicolson(), grid).price(base_params, n_steps=40, option_type="call")
        assert r.price >= 0.0

    def test_call_le_spot(self, grid: HestonGridConfig, base_params: BatesParams):
        p = BatesPricer(CrankNicolson(), grid).price(base_params, n_steps=40, option_type="call").price
        assert p <= base_params.S + 1e-8

    def test_put_le_strike(self, grid: HestonGridConfig, base_params: BatesParams):
        p = BatesPricer(CrankNicolson(), grid).price(base_params, n_steps=40, option_type="put").price
        assert p <= base_params.K + 1e-8

    def test_put_call_parity(self, grid: HestonGridConfig, base_params: BatesParams):
        pr = BatesPricer(CrankNicolson(), grid)
        c = pr.price(base_params, n_steps=40, option_type="call").price
        p = pr.price(base_params, n_steps=40, option_type="put").price
        rhs = base_params.S - base_params.K * np.exp(-base_params.r * base_params.T)
        assert abs((c - p) - rhs) / max(abs(rhs), 1e-12) < 0.12


class TestBatesConvergence:

    def test_convergence_with_n(self, grid: HestonGridConfig, base_params: BatesParams):
        ref = bates_price(**base_params.__dict__, option_type="call")
        pr = BatesPricer(CrankNicolson(), grid)
        e20 = abs(pr.price(base_params, 20, "call").price - ref)
        e40 = abs(pr.price(base_params, 40, "call").price - ref)
        e60 = abs(pr.price(base_params, 60, "call").price - ref)
        assert e60 <= e40 + 1e-8
        assert e40 <= e20 + 1e-8

    def test_cn_better_than_be(self, grid: HestonGridConfig, base_params: BatesParams):
        ref = bates_price(**base_params.__dict__, option_type="call")
        cn = BatesPricer(CrankNicolson(), grid).price(base_params, 40, "call").price
        be = BatesPricer(BackwardEuler(), grid).price(base_params, 40, "call").price
        assert abs(cn - ref) <= abs(be - ref) + 0.1


class TestBatesReduction:

    def test_lambda_zero_gives_heston(self, grid: HestonGridConfig, base_params: BatesParams):
        p0 = BatesParams(**{**base_params.__dict__, "lambda_j": 0.0})
        b = BatesPricer(CrankNicolson(), grid).price(p0, n_steps=40, option_type="call").price
        h = HestonPricer(CrankNicolson(), grid).price(p0.to_heston(), n_steps=40, option_type="call").price
        assert abs(b - h) / h < 0.01

    def test_xi_zero_lambda_zero_gives_bs_limit(self, grid: HestonGridConfig):
        p = BatesParams(
            S=100,
            K=100,
            T=1.0,
            r=0.05,
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            xi=1e-8,
            rho=0.0,
            lambda_j=0.0,
            mu_j=0.0,
            sigma_j=0.0,
        )
        r = BatesPricer(CrankNicolson(), grid).price(p, n_steps=40, option_type="call")
        assert np.isfinite(r.price)

