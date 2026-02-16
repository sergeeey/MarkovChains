"""Tests for HestonFastPricer."""

from __future__ import annotations

import time

import numpy as np
import pytest

from chernoffpy import CrankNicolson
from chernoffpy.accel import HAS_NUMBA
from chernoffpy.finance.heston import HestonPricer
from chernoffpy.finance.heston_analytical import heston_price
from chernoffpy.finance.heston_fast import HestonFastPricer
from chernoffpy.finance.heston_params import HestonGridConfig, HestonParams


@pytest.fixture
def base_params() -> HestonParams:
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
def grid() -> HestonGridConfig:
    return HestonGridConfig(n_x=128, n_v=48, x_min=-4.0, x_max=4.0, v_max=0.8)


class TestHestonFastPricing:

    def test_call_matches_original(self, base_params: HestonParams, grid: HestonGridConfig):
        slow = HestonPricer(CrankNicolson(), grid).price(base_params, n_steps=40, option_type="call").price
        fast = HestonFastPricer(CrankNicolson(), grid).price(base_params, n_steps=40, option_type="call").price
        assert abs(fast - slow) / max(abs(slow), 1e-12) < 0.01

    def test_put_matches_original(self, base_params: HestonParams, grid: HestonGridConfig):
        slow = HestonPricer(CrankNicolson(), grid).price(base_params, n_steps=40, option_type="put").price
        fast = HestonFastPricer(CrankNicolson(), grid).price(base_params, n_steps=40, option_type="put").price
        assert abs(fast - slow) / max(abs(slow), 1e-12) < 0.01

    def test_bs_reduction(self, grid: HestonGridConfig):
        params = HestonParams(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 1e-8, 0.0)
        result = HestonFastPricer(CrankNicolson(), grid).price(params, n_steps=35, option_type="call")
        assert abs(result.price - result.bs_equiv_price) / result.bs_equiv_price < 0.03

    def test_put_call_parity(self, base_params: HestonParams, grid: HestonGridConfig):
        pr = HestonFastPricer(CrankNicolson(), grid)
        call = pr.price(base_params, 35, "call").price
        put = pr.price(base_params, 35, "put").price
        rhs = base_params.S - base_params.K * np.exp(-base_params.r * base_params.T)
        assert abs((call - put) - rhs) / max(abs(rhs), 1e-12) < 0.08

    def test_vs_analytical(self, base_params: HestonParams, grid: HestonGridConfig):
        fast = HestonFastPricer(CrankNicolson(), grid).price(base_params, n_steps=40, option_type="call").price
        ref = heston_price(**base_params.__dict__, option_type="call")
        assert abs(fast - ref) / ref < 0.08


class TestHestonFastProperties:

    def test_nonnegative(self, base_params: HestonParams, grid: HestonGridConfig):
        r = HestonFastPricer(CrankNicolson(), grid).price(base_params, n_steps=35, option_type="call")
        assert r.price >= 0.0

    def test_call_le_spot(self, base_params: HestonParams, grid: HestonGridConfig):
        p = HestonFastPricer(CrankNicolson(), grid).price(base_params, n_steps=35, option_type="call").price
        assert p <= base_params.S + 1e-8

    def test_put_le_strike(self, base_params: HestonParams, grid: HestonGridConfig):
        p = HestonFastPricer(CrankNicolson(), grid).price(base_params, n_steps=35, option_type="put").price
        assert p <= base_params.K + 1e-8

    def test_convergence_with_n_steps(self, base_params: HestonParams, grid: HestonGridConfig):
        ref = heston_price(**base_params.__dict__, option_type="call")
        pr = HestonFastPricer(CrankNicolson(), grid)
        e20 = abs(pr.price(base_params, 20, "call").price - ref)
        e40 = abs(pr.price(base_params, 40, "call").price - ref)
        e60 = abs(pr.price(base_params, 60, "call").price - ref)
        assert e60 <= e40 + 1e-8
        assert e40 <= e20 + 1e-8


class TestHestonFastPerformance:

    def test_faster_than_original_after_warmup(self, base_params: HestonParams):
        grid = HestonGridConfig(n_x=192, n_v=64, x_min=-4.0, x_max=4.0, v_max=0.8)
        slow = HestonPricer(CrankNicolson(), grid)
        fast = HestonFastPricer(CrankNicolson(), grid)

        slow.price(base_params, n_steps=25, option_type="call")
        fast.price(base_params, n_steps=25, option_type="call")

        t0 = time.perf_counter()
        slow.price(base_params, n_steps=25, option_type="call")
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        fast.price(base_params, n_steps=25, option_type="call")
        t3 = time.perf_counter()

        slow_time = t1 - t0
        fast_time = t3 - t2

        target = 0.9 if HAS_NUMBA else 1.1
        assert fast_time <= slow_time * target

    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba is not installed")
    def test_numba_warmup_effect(self, base_params: HestonParams):
        grid = HestonGridConfig(n_x=160, n_v=56, x_min=-4.0, x_max=4.0, v_max=0.8)
        fast = HestonFastPricer(CrankNicolson(), grid)

        t0 = time.perf_counter()
        fast.price(base_params, n_steps=20, option_type="call")
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        fast.price(base_params, n_steps=20, option_type="call")
        t3 = time.perf_counter()

        assert (t3 - t2) <= (t1 - t0) * 1.2
