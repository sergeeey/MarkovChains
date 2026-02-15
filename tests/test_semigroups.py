"""Tests for chernoffpy.semigroups — HeatSemigroup exact solutions."""

import numpy as np
import pytest

from chernoffpy.semigroups import HeatSemigroup


# ===================================================================
# solve_fourier
# ===================================================================

class TestSolveFourier:
    def test_identity_at_t0(self, small_grid, sin_initial):
        """e^{0*A} f = f."""
        result = HeatSemigroup.solve_fourier(sin_initial, small_grid, t=0.0)
        np.testing.assert_allclose(result, sin_initial, atol=1e-14)

    def test_sin_decays_exponentially(self, small_grid):
        """sin(x) is eigenfunction of d²/dx²: e^{tA} sin(x) = e^{-t} sin(x)."""
        f = np.sin(small_grid)
        t = 0.5
        result = HeatSemigroup.solve_fourier(f, small_grid, t)
        expected = np.exp(-t) * f
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_constant_preserved(self, small_grid):
        """Constants are in the kernel of d²/dx²: e^{tA} c = c."""
        c = np.ones_like(small_grid) * 5.0
        result = HeatSemigroup.solve_fourier(c, small_grid, t=2.0)
        np.testing.assert_allclose(result, c, atol=1e-12)

    def test_high_freq_damped(self, small_grid):
        """High-frequency modes decay faster: e^{-k²t} for mode k."""
        f = np.sin(5 * small_grid)
        t = 0.5
        result = HeatSemigroup.solve_fourier(f, small_grid, t)
        # Decay factor e^{-25*0.5} = e^{-12.5} ≈ 3.7e-6
        assert np.max(np.abs(result)) < 1e-4

    def test_non_negativity_gaussian(self, small_grid, gaussian_initial):
        """Heat equation preserves non-negativity (maximum principle)."""
        result = HeatSemigroup.solve_fourier(gaussian_initial, small_grid, t=0.5)
        # Allow small numerical noise
        assert np.min(result) > -1e-10


# ===================================================================
# solve_convolution
# ===================================================================

class TestSolveConvolution:
    def test_identity_at_t0(self, small_grid):
        """e^{0*A} f = f via convolution path."""
        f_func = np.sin
        result = HeatSemigroup.solve_convolution(f_func, small_grid, t=0.0)
        expected = np.sin(small_grid)
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_gaussian_closed_form(self):
        """Gaussian initial data has closed-form solution.

        e^{tA} exp(-x²/(2σ²)) = σ/√(σ²+2t) * exp(-x²/(2(σ²+2t)))
        """
        sigma = 1.0
        t = 0.5
        x = np.linspace(-5, 5, 50)

        def f(x_val):
            return np.exp(-x_val**2 / (2 * sigma**2))

        result = HeatSemigroup.solve_convolution(f, x, t, integration_limit=15.0)
        sigma_new = np.sqrt(sigma**2 + 2 * t)
        expected = (sigma / sigma_new) * np.exp(-x**2 / (2 * sigma_new**2))
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_matches_fourier(self, small_grid):
        """Convolution and Fourier methods should agree for periodic functions."""
        f_func = np.sin
        f_values = np.sin(small_grid)
        t = 0.3

        result_fourier = HeatSemigroup.solve_fourier(f_values, small_grid, t)
        result_conv = HeatSemigroup.solve_convolution(f_func, small_grid, t,
                                                       integration_limit=15.0)
        # Convolution on periodic domain is approximate, so larger tolerance
        np.testing.assert_allclose(result_conv, result_fourier, atol=0.05)


# ===================================================================
# eigenfunction_solution
# ===================================================================

class TestEigenfunctionSolution:
    def test_single_mode_exact(self):
        """c_1 = 1, rest zero: u(t,x) = e^{-pi^2 t} sqrt(2) sin(pi x)."""
        L = 1.0
        x = np.linspace(0, L, 100, endpoint=False)
        t = 0.1
        coeffs = np.array([1.0])
        result = HeatSemigroup.eigenfunction_solution(x, t, coeffs, L=L)
        expected = np.exp(-(np.pi / L)**2 * t) * np.sqrt(2 / L) * np.sin(np.pi * x / L)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_dirichlet_bcs(self):
        """Solution should vanish at x=0 and x=L (Dirichlet)."""
        L = 1.0
        x = np.array([0.0, L])
        coeffs = np.array([1.0, 0.5, 0.3])
        result = HeatSemigroup.eigenfunction_solution(x, t=0.5, coeffs=coeffs, L=L)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-12)

    def test_decay_ordering(self):
        """Higher modes decay faster: c_1 component decays slowest."""
        L = 1.0
        x = np.linspace(0, L, 200, endpoint=False)
        t = 0.5

        u1 = HeatSemigroup.eigenfunction_solution(x, t, np.array([1.0, 0.0, 0.0]), L=L)
        u2 = HeatSemigroup.eigenfunction_solution(x, t, np.array([0.0, 1.0, 0.0]), L=L)
        u3 = HeatSemigroup.eigenfunction_solution(x, t, np.array([0.0, 0.0, 1.0]), L=L)

        assert np.max(np.abs(u1)) > np.max(np.abs(u2))
        assert np.max(np.abs(u2)) > np.max(np.abs(u3))
