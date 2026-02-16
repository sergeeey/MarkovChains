"""Tests for certified bound core utilities."""

from __future__ import annotations

import numpy as np
import pytest

from chernoffpy import BackwardEuler, CrankNicolson, PadeChernoff
from chernoffpy.certified import (
    CertifiedBound,
    ChernoffOrder,
    PayoffRegularity,
    compute_certified_bound,
    effective_order,
    n_steps_for_tolerance,
    verify_convergence_order,
)


class _UnknownChernoff:
    pass


class TestChernoffOrder:

    def test_backward_euler_order_1(self):
        o = ChernoffOrder.from_chernoff(BackwardEuler())
        assert o.k == 1

    def test_crank_nicolson_order_2(self):
        o = ChernoffOrder.from_chernoff(CrankNicolson())
        assert o.k == 2

    def test_pade22_order_4(self):
        o = ChernoffOrder.from_chernoff(PadeChernoff(2, 2))
        assert o.k == 4

    def test_unknown_defaults_to_1(self):
        o = ChernoffOrder.from_chernoff(_UnknownChernoff())
        assert o.k == 1
        assert o.is_exact is False


class TestPayoffRegularity:

    def test_vanilla_call_kf_2(self):
        assert PayoffRegularity.vanilla_call().k_f == 2

    def test_digital_kf_0(self):
        assert PayoffRegularity.digital().k_f == 0

    def test_smooth_kf_large(self):
        assert PayoffRegularity.smooth(50).k_f == 50

    def test_barrier_dst_kf_2(self):
        assert PayoffRegularity.barrier().k_f == 2

    def test_barrier_fft_kf_0(self):
        assert PayoffRegularity.barrier_fft().k_f == 0


class TestEffectiveOrder:

    def test_cn_vanilla_gives_2(self):
        assert effective_order(ChernoffOrder(2, "CN"), PayoffRegularity.vanilla_call()) == 2

    def test_pade_vanilla_gives_2(self):
        assert effective_order(ChernoffOrder(4, "Pade"), PayoffRegularity.vanilla_call()) == 2

    def test_be_smooth_gives_1(self):
        assert effective_order(ChernoffOrder(1, "BE"), PayoffRegularity.smooth()) == 1

    def test_cn_digital_gives_0(self):
        assert effective_order(ChernoffOrder(2, "CN"), PayoffRegularity.digital()) == 0


class TestCertifiedBound:

    def test_bound_with_exact(self):
        exact = 10.0
        prices = {20: 10.005, 40: 10.00125, 80: 10.0003125}
        b = compute_certified_bound(
            prices=prices,
            chernoff_order=ChernoffOrder(2, "CN"),
            payoff_reg=PayoffRegularity.vanilla_call(),
            n_target=20,
            safety_factor=2.0,
            exact_price=exact,
        )
        assert b.bound >= abs(prices[20] - exact)

    def test_bound_without_exact(self):
        prices = {20: 10.005, 40: 10.00125, 80: 10.0003125}
        b = compute_certified_bound(
            prices=prices,
            chernoff_order=ChernoffOrder(2, "CN"),
            payoff_reg=PayoffRegularity.vanilla_call(),
            n_target=20,
            safety_factor=2.0,
            exact_price=None,
        )
        assert b.bound > 0
        assert b.is_certified is False

    def test_bound_decreases_with_n(self):
        exact = 10.0
        prices = {20: 10.004, 40: 10.001, 80: 10.00025}
        b20 = compute_certified_bound(prices, ChernoffOrder(2, "CN"), PayoffRegularity.vanilla_call(), 20, 2.0, exact)
        b40 = compute_certified_bound(prices, ChernoffOrder(2, "CN"), PayoffRegularity.vanilla_call(), 40, 2.0, exact)
        assert b40.bound < b20.bound

    def test_safety_factor(self):
        exact = 10.0
        prices = {20: 10.004, 40: 10.001, 80: 10.00025}
        b1 = compute_certified_bound(prices, ChernoffOrder(2, "CN"), PayoffRegularity.vanilla_call(), 20, 1.0, exact)
        b3 = compute_certified_bound(prices, ChernoffOrder(2, "CN"), PayoffRegularity.vanilla_call(), 20, 3.0, exact)
        assert b3.bound > b1.bound

    def test_zero_order_not_certified(self):
        prices = {20: 1.0, 40: 1.0, 80: 1.0}
        b = compute_certified_bound(prices, ChernoffOrder(2, "CN"), PayoffRegularity.digital(), 20)
        assert b.is_certified is False
        assert b.effective_order == 0


class TestVerifyOrder:

    def test_cn_order_2(self):
        exact = 1.0
        prices = {20: exact + 2 / 20**2, 40: exact + 2 / 40**2, 80: exact + 2 / 80**2}
        v = verify_convergence_order(prices, expected_order=2, exact_price=exact, tolerance=0.2)
        assert v["is_consistent"] is True
        assert v["empirical_order"] == pytest.approx(2.0, rel=0.1)

    def test_be_order_1(self):
        exact = 1.0
        prices = {20: exact + 1 / 20, 40: exact + 1 / 40, 80: exact + 1 / 80}
        v = verify_convergence_order(prices, expected_order=1, exact_price=exact, tolerance=0.2)
        assert v["is_consistent"] is True
        assert v["empirical_order"] == pytest.approx(1.0, rel=0.1)

    def test_mismatch_detected(self):
        exact = 1.0
        prices = {20: exact + 1 / 20, 40: exact + 1 / 40, 80: exact + 1 / 80}
        v = verify_convergence_order(prices, expected_order=2, exact_price=exact, tolerance=0.2)
        assert v["is_consistent"] is False


class TestNStepsForTolerance:

    def test_cn_0001_gives_reasonable_n(self):
        n = n_steps_for_tolerance(target_error=1e-3, constant_B=1.0, effective_order=2, safety_factor=2.0)
        assert 10 <= n <= 100

    def test_be_needs_more_steps(self):
        n_cn = n_steps_for_tolerance(target_error=1e-3, constant_B=1.0, effective_order=2, safety_factor=2.0)
        n_be = n_steps_for_tolerance(target_error=1e-3, constant_B=1.0, effective_order=1, safety_factor=2.0)
        assert n_be > n_cn

    def test_zero_order_raises(self):
        with pytest.raises(ValueError):
            n_steps_for_tolerance(target_error=1e-3, constant_B=1.0, effective_order=0)
