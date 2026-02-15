"""Tests for chernoffpy.functions — all 7 classes and helper utilities."""

import warnings

import numpy as np
import pytest

from chernoffpy.functions import (
    BackwardEuler,
    ChernoffFunction,
    CrankNicolson,
    PadeChernoff,
    PhysicalG,
    PhysicalS,
    _compute_pade_coefficients,
)


# ===================================================================
# C(0) = Identity for every concrete class
# ===================================================================

class TestIdentityAtZero:
    """C(0)f = f for all Chernoff functions."""

    @pytest.mark.parametrize("cls", [PhysicalG, PhysicalS, BackwardEuler, CrankNicolson])
    def test_identity_simple_classes(self, cls, small_grid, sin_initial):
        cf = cls()
        result = cf.apply(sin_initial, small_grid, t=0.0)
        np.testing.assert_allclose(result, sin_initial, atol=1e-14)

    def test_identity_pade_01(self, small_grid, sin_initial):
        cf = PadeChernoff(0, 1)
        result = cf.apply(sin_initial, small_grid, t=0.0)
        np.testing.assert_allclose(result, sin_initial, atol=1e-14)

    def test_identity_pade_11(self, small_grid, sin_initial):
        cf = PadeChernoff(1, 1)
        result = cf.apply(sin_initial, small_grid, t=0.0)
        np.testing.assert_allclose(result, sin_initial, atol=1e-14)

    def test_identity_pade_12(self, small_grid, sin_initial):
        cf = PadeChernoff(1, 2)
        result = cf.apply(sin_initial, small_grid, t=0.0)
        np.testing.assert_allclose(result, sin_initial, atol=1e-14)

    def test_identity_gaussian(self, small_grid, gaussian_initial):
        cf = CrankNicolson()
        result = cf.apply(gaussian_initial, small_grid, t=0.0)
        np.testing.assert_allclose(result, gaussian_initial, atol=1e-14)


# ===================================================================
# PhysicalG / PhysicalS: weights and properties
# ===================================================================

class TestPhysicalG:
    def test_order(self):
        assert PhysicalG().order == 1

    def test_name_contains_order(self):
        assert "1" in PhysicalG().name

    def test_weights_sum_to_one(self, small_grid, sin_initial):
        """Applying to constant function should preserve the constant."""
        cf = PhysicalG()
        const = np.ones_like(small_grid) * 3.7
        result = cf.apply(const, small_grid, t=0.5)
        np.testing.assert_allclose(result, const, atol=1e-10)

    def test_smoothing_effect(self, small_grid):
        """After applying G(t) with t>0, oscillations should decrease."""
        cf = PhysicalG()
        high_freq = np.sin(10 * small_grid)
        result = cf.apply(high_freq, small_grid, t=0.1)
        assert np.max(np.abs(result)) < np.max(np.abs(high_freq))


class TestPhysicalS:
    def test_order(self):
        assert PhysicalS().order == 2

    def test_name_contains_order(self):
        assert "2" in PhysicalS().name

    def test_weights_sum_to_one(self, small_grid):
        """Applying to constant function should preserve the constant."""
        cf = PhysicalS()
        const = np.ones_like(small_grid) * 2.5
        result = cf.apply(const, small_grid, t=0.5)
        np.testing.assert_allclose(result, const, atol=1e-10)

    def test_smoothing_effect(self, small_grid):
        cf = PhysicalS()
        high_freq = np.sin(10 * small_grid)
        result = cf.apply(high_freq, small_grid, t=0.1)
        assert np.max(np.abs(result)) < np.max(np.abs(high_freq))


# ===================================================================
# BackwardEuler / CrankNicolson: Fourier multipliers
# ===================================================================

class TestBackwardEuler:
    def test_order(self):
        assert BackwardEuler().order == 1

    def test_multiplier_positive(self):
        """1/(1+t*xi^2) > 0 for all xi, t > 0."""
        be = BackwardEuler()
        xi_sq = np.array([0.0, 1.0, 10.0, 100.0])
        m = be.multiplier(xi_sq, t=1.0)
        assert np.all(m > 0)

    def test_multiplier_bounded_by_one(self):
        be = BackwardEuler()
        xi_sq = np.linspace(0, 1000, 500)
        m = be.multiplier(xi_sq, t=0.5)
        assert np.all(m <= 1.0 + 1e-15)

    def test_multiplier_at_zero_freq(self):
        """At xi=0, multiplier = 1 (constants preserved)."""
        be = BackwardEuler()
        m = be.multiplier(np.array([0.0]), t=1.0)
        np.testing.assert_allclose(m, [1.0], atol=1e-15)


class TestCrankNicolson:
    def test_order(self):
        assert CrankNicolson().order == 2

    def test_multiplier_abs_bounded_by_one(self):
        """|(1 - u/2)/(1 + u/2)| <= 1 for u >= 0 (A-stability)."""
        cn = CrankNicolson()
        xi_sq = np.linspace(0, 1000, 500)
        m = cn.multiplier(xi_sq, t=0.5)
        assert np.all(np.abs(m) <= 1.0 + 1e-14)

    def test_multiplier_at_zero_freq(self):
        cn = CrankNicolson()
        m = cn.multiplier(np.array([0.0]), t=1.0)
        np.testing.assert_allclose(m, [1.0], atol=1e-15)

    def test_multiplier_negative_for_large_freq(self):
        """For large xi: (1-u/2)/(1+u/2) < 0 — alternating sign."""
        cn = CrankNicolson()
        xi_sq = np.array([100.0])
        m = cn.multiplier(xi_sq, t=1.0)
        assert m[0] < 0


# ===================================================================
# PadeChernoff: coefficients, equivalences, stability
# ===================================================================

class TestPadeCoefficients:
    def test_pade_01_coefficients(self):
        """[0/1]: p=[1], q=[1, -1] => p(z)/q(z) = 1/(1-z)."""
        p, q = _compute_pade_coefficients(0, 1)
        np.testing.assert_allclose(p, [1.0], atol=1e-14)
        np.testing.assert_allclose(q, [1.0, -1.0], atol=1e-14)

    def test_pade_11_coefficients(self):
        """[1/1]: p=[1, 1/2], q=[1, -1/2] => (1+z/2)/(1-z/2)."""
        p, q = _compute_pade_coefficients(1, 1)
        np.testing.assert_allclose(p, [1.0, 0.5], atol=1e-14)
        np.testing.assert_allclose(q, [1.0, -0.5], atol=1e-14)

    def test_pade_coefficients_sum(self):
        """p(0)/q(0) = 1 for any [m/n] (since e^0 = 1)."""
        for m, n in [(0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (3, 1)]:
            p, q = _compute_pade_coefficients(m, n)
            assert abs(p[0] / q[0] - 1.0) < 1e-14, f"Failed for [{m}/{n}]"


class TestPadeChernoff:
    def test_order_mn(self):
        assert PadeChernoff(2, 1).order == 3
        assert PadeChernoff(1, 2).order == 3
        assert PadeChernoff(0, 1).order == 1

    def test_pade_01_equals_backward_euler(self, small_grid, sin_initial):
        """Padé [0/1] and BackwardEuler should produce identical multipliers."""
        be = BackwardEuler()
        pade01 = PadeChernoff(0, 1)
        xi_sq = np.linspace(0, 100, 50)
        t = 0.5
        np.testing.assert_allclose(
            pade01.multiplier(xi_sq, t),
            be.multiplier(xi_sq, t),
            atol=1e-12,
        )

    def test_pade_11_equals_crank_nicolson(self, small_grid, sin_initial):
        """Padé [1/1] and CrankNicolson should produce identical multipliers."""
        cn = CrankNicolson()
        pade11 = PadeChernoff(1, 1)
        xi_sq = np.linspace(0, 100, 50)
        t = 0.5
        np.testing.assert_allclose(
            pade11.multiplier(xi_sq, t),
            cn.multiplier(xi_sq, t),
            atol=1e-12,
        )

    def test_a_stable_when_m_leq_n(self):
        """No warning for m <= n."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PadeChernoff(1, 2)
            assert len(w) == 0

    def test_warning_when_m_gt_n(self):
        """Warning emitted for m > n (not A-stable)."""
        with pytest.warns(UserWarning, match="NOT A-stable"):
            PadeChernoff(2, 1)

    def test_a_stability_multiplier_bounded(self):
        """For A-stable [1/2]: |multiplier| <= 1 on negative real axis."""
        pade12 = PadeChernoff(1, 2)
        xi_sq = np.linspace(0, 1000, 500)
        m = pade12.multiplier(xi_sq, t=0.5)
        assert np.all(np.abs(m) <= 1.0 + 1e-12)


# ===================================================================
# compose / compose_all
# ===================================================================

class TestComposition:
    def test_compose_n1_equals_apply(self, small_grid, sin_initial):
        """compose(f, x, t, n=1) = apply(f, x, t)."""
        cf = BackwardEuler()
        t = 0.5
        composed = cf.compose(sin_initial, small_grid, t, n=1)
        applied = cf.apply(sin_initial, small_grid, t)
        np.testing.assert_allclose(composed, applied, atol=1e-15)

    def test_compose_all_length(self, small_grid, sin_initial):
        """compose_all returns exactly n_max arrays."""
        cf = CrankNicolson()
        results = cf.compose_all(sin_initial, small_grid, t=1.0, n_max=5)
        assert len(results) == 5

    def test_compose_all_consistent_with_compose(self, small_grid, sin_initial):
        """compose_all[n-1] should equal compose(..., n)."""
        cf = PhysicalG()
        t = 1.0
        n_max = 3
        all_results = cf.compose_all(sin_initial, small_grid, t, n_max)
        for n in range(1, n_max + 1):
            single = cf.compose(sin_initial, small_grid, t, n)
            np.testing.assert_allclose(all_results[n - 1], single, atol=1e-14)

    def test_compose_all_shapes(self, small_grid, sin_initial):
        """Each result should have the same shape as input."""
        cf = BackwardEuler()
        results = cf.compose_all(sin_initial, small_grid, t=1.0, n_max=3)
        for r in results:
            assert r.shape == sin_initial.shape
