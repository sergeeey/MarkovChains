"""Tests for chernoffpy.analysis — error computation, rate fitting, tables."""

import numpy as np
import pytest

from chernoffpy.functions import BackwardEuler, CrankNicolson
from chernoffpy.semigroups import HeatSemigroup
from chernoffpy.analysis import compute_errors, convergence_rate, convergence_table


# ===================================================================
# compute_errors
# ===================================================================

class TestComputeErrors:
    def test_correct_length(self, small_grid, sin_initial):
        exact = HeatSemigroup.solve_fourier(sin_initial, small_grid, t=1.0)
        errors = compute_errors(BackwardEuler(), sin_initial, small_grid, exact, t=1.0, n_max=5)
        assert len(errors) == 5

    def test_errors_positive(self, small_grid, sin_initial):
        exact = HeatSemigroup.solve_fourier(sin_initial, small_grid, t=1.0)
        errors = compute_errors(BackwardEuler(), sin_initial, small_grid, exact, t=1.0, n_max=5)
        assert np.all(errors > 0)

    def test_errors_decreasing_smooth(self, small_grid, sin_initial):
        """For smooth data, errors should generally decrease with n."""
        exact = HeatSemigroup.solve_fourier(sin_initial, small_grid, t=1.0)
        errors = compute_errors(BackwardEuler(), sin_initial, small_grid, exact, t=1.0, n_max=5)
        # Check overall trend: last error < first error
        assert errors[-1] < errors[0]

    def test_sup_norm_manual(self, small_grid, sin_initial):
        """Verify sup-norm formula against manual computation."""
        cf = BackwardEuler()
        exact = HeatSemigroup.solve_fourier(sin_initial, small_grid, t=1.0)
        approx = cf.compose(sin_initial, small_grid, t=1.0, n=2)
        expected_err = np.max(np.abs(approx - exact))
        errors = compute_errors(cf, sin_initial, small_grid, exact, t=1.0, n_max=2)
        np.testing.assert_allclose(errors[1], expected_err, atol=1e-15)

    def test_l2_norm_manual(self, small_grid, sin_initial):
        """Verify L2-norm formula against manual computation."""
        cf = BackwardEuler()
        exact = HeatSemigroup.solve_fourier(sin_initial, small_grid, t=1.0)
        approx = cf.compose(sin_initial, small_grid, t=1.0, n=2)
        dx = small_grid[1] - small_grid[0]
        diff = approx - exact
        expected_err = np.sqrt(np.sum(diff**2) * dx)
        errors = compute_errors(cf, sin_initial, small_grid, exact, t=1.0, n_max=2, norm="L2")
        np.testing.assert_allclose(errors[1], expected_err, atol=1e-15)

    def test_unknown_norm_raises(self, small_grid, sin_initial):
        exact = HeatSemigroup.solve_fourier(sin_initial, small_grid, t=1.0)
        with pytest.raises(ValueError, match="Unknown norm"):
            compute_errors(BackwardEuler(), sin_initial, small_grid, exact,
                           t=1.0, n_max=2, norm="H1")


# ===================================================================
# convergence_rate
# ===================================================================

class TestConvergenceRate:
    def test_synthetic_rate_k1(self):
        """d_n = 5 * n^{-1} should give alpha ≈ 1."""
        n = np.arange(1, 21)
        errors = 5.0 * n.astype(float)**(-1)
        alpha, C = convergence_rate(errors)
        assert abs(alpha - 1.0) < 0.1

    def test_synthetic_rate_k2(self):
        """d_n = 3 * n^{-2} should give alpha ≈ 2."""
        n = np.arange(1, 21)
        errors = 3.0 * n.astype(float)**(-2)
        alpha, C = convergence_rate(errors)
        assert abs(alpha - 2.0) < 0.1

    def test_synthetic_rate_k3(self):
        """d_n = 1.5 * n^{-3} should give alpha ≈ 3."""
        n = np.arange(1, 21)
        errors = 1.5 * n.astype(float)**(-3)
        alpha, C = convergence_rate(errors)
        assert abs(alpha - 3.0) < 0.1

    def test_all_zeros(self):
        """Edge case: all-zero errors should return (0, 0)."""
        errors = np.zeros(10)
        alpha, C = convergence_rate(errors)
        assert alpha == 0.0
        assert C == 0.0


# ===================================================================
# convergence_table
# ===================================================================

class TestConvergenceTable:
    def test_returns_string(self, small_grid, sin_initial):
        exact = HeatSemigroup.solve_fourier(sin_initial, small_grid, t=1.0)
        table = convergence_table([BackwardEuler()], sin_initial, small_grid, exact,
                                  t=1.0, n_max=5)
        assert isinstance(table, str)

    def test_contains_headers(self, small_grid, sin_initial):
        exact = HeatSemigroup.solve_fourier(sin_initial, small_grid, t=1.0)
        table = convergence_table([BackwardEuler()], sin_initial, small_grid, exact,
                                  t=1.0, n_max=5)
        assert "Theoretical" in table
        assert "Empirical" in table

    def test_correct_row_count(self, small_grid, sin_initial):
        """Header + separator + one row per function."""
        exact = HeatSemigroup.solve_fourier(sin_initial, small_grid, t=1.0)
        funcs = [BackwardEuler(), CrankNicolson()]
        table = convergence_table(funcs, sin_initial, small_grid, exact,
                                  t=1.0, n_max=5)
        lines = table.strip().split("\n")
        assert len(lines) == 2 + len(funcs)  # header + separator + data rows
