"""Integration tests — Galkin-Remizov convergence rate verification.

These are THE KEY TESTS: they verify that the library reproduces the
theoretical convergence rates predicted by the Galkin-Remizov theorem.
"""

import warnings

import numpy as np
import pytest

from chernoffpy.functions import (
    BackwardEuler,
    CrankNicolson,
    PadeChernoff,
    PhysicalG,
    PhysicalS,
)
from chernoffpy.semigroups import HeatSemigroup
from chernoffpy.analysis import compute_errors, convergence_rate


# ===================================================================
# Galkin-Remizov: order-k method => alpha ≈ k for smooth data
# ===================================================================

class TestGalkinRemizov:
    """Verify convergence rates match theoretical predictions.

    For f = sin(x) (infinitely smooth), order-k methods should achieve alpha ≈ k.
    Using medium grid (N=128), n_max=15 for reliable rate estimation.
    """

    @pytest.fixture
    def setup(self, medium_grid, sin_medium):
        exact = HeatSemigroup.solve_fourier(sin_medium, medium_grid, t=1.0)
        return medium_grid, sin_medium, exact

    def test_order1_backward_euler(self, setup):
        """BackwardEuler (order 1): alpha ≈ 1."""
        grid, f, exact = setup
        errors = compute_errors(BackwardEuler(), f, grid, exact, t=1.0, n_max=15)
        alpha, _ = convergence_rate(errors)
        assert 0.7 < alpha < 1.5, f"Expected alpha ≈ 1, got {alpha:.3f}"

    def test_order1_physical_g(self, setup):
        """PhysicalG (order 1): alpha ≈ 1."""
        grid, f, exact = setup
        errors = compute_errors(PhysicalG(), f, grid, exact, t=1.0, n_max=15)
        alpha, _ = convergence_rate(errors)
        assert 0.7 < alpha < 1.5, f"Expected alpha ≈ 1, got {alpha:.3f}"

    def test_order2_crank_nicolson(self, setup):
        """CrankNicolson (order 2): alpha ≈ 2."""
        grid, f, exact = setup
        errors = compute_errors(CrankNicolson(), f, grid, exact, t=1.0, n_max=15)
        alpha, _ = convergence_rate(errors)
        assert 1.5 < alpha < 2.8, f"Expected alpha ≈ 2, got {alpha:.3f}"

    def test_order2_physical_s(self, setup):
        """PhysicalS (order 2): alpha ≈ 2."""
        grid, f, exact = setup
        errors = compute_errors(PhysicalS(), f, grid, exact, t=1.0, n_max=15)
        alpha, _ = convergence_rate(errors)
        assert 1.5 < alpha < 2.8, f"Expected alpha ≈ 2, got {alpha:.3f}"

    def test_order3_pade_12(self, setup):
        """Padé [1/2] (order 3, A-stable): alpha ≈ 3."""
        grid, f, exact = setup
        pade12 = PadeChernoff(1, 2)
        errors = compute_errors(pade12, f, grid, exact, t=1.0, n_max=15)
        alpha, _ = convergence_rate(errors)
        assert 2.3 < alpha < 3.8, f"Expected alpha ≈ 3, got {alpha:.3f}"


# ===================================================================
# Regularity limitation: |sin(x)| prevents high-order convergence
# ===================================================================

class TestRegularityLimitation:
    """Non-smooth data limits convergence regardless of method order."""

    def test_abs_sin_limits_order2(self, medium_grid, abs_sin_medium):
        """|sin(x)| (only C^0) should prevent order-2 method from achieving alpha=2."""
        exact = HeatSemigroup.solve_fourier(abs_sin_medium, medium_grid, t=1.0)
        errors = compute_errors(CrankNicolson(), abs_sin_medium, medium_grid, exact,
                                t=1.0, n_max=15)
        alpha, _ = convergence_rate(errors)
        # Should be noticeably below 2 due to limited regularity
        assert alpha < 2.2, f"Unexpectedly high rate {alpha:.3f} for non-smooth data"


# ===================================================================
# Padé-classical equivalence: full pipeline
# ===================================================================

class TestPadeClassicalEquivalence:
    """Verify that Padé [0/1] and [1/1] produce the same final results
    as BackwardEuler and CrankNicolson through the full computation pipeline."""

    def test_pade_01_equals_backward_euler_full(self, medium_grid, sin_medium):
        """Full pipeline: Padé [0/1] ≡ BackwardEuler."""
        exact = HeatSemigroup.solve_fourier(sin_medium, medium_grid, t=1.0)

        be_errors = compute_errors(BackwardEuler(), sin_medium, medium_grid, exact,
                                   t=1.0, n_max=10)
        pade_errors = compute_errors(PadeChernoff(0, 1), sin_medium, medium_grid, exact,
                                     t=1.0, n_max=10)
        np.testing.assert_allclose(pade_errors, be_errors, rtol=1e-10)

    def test_pade_11_equals_crank_nicolson_full(self, medium_grid, sin_medium):
        """Full pipeline: Padé [1/1] ≡ CrankNicolson."""
        exact = HeatSemigroup.solve_fourier(sin_medium, medium_grid, t=1.0)

        cn_errors = compute_errors(CrankNicolson(), sin_medium, medium_grid, exact,
                                   t=1.0, n_max=10)
        pade_errors = compute_errors(PadeChernoff(1, 1), sin_medium, medium_grid, exact,
                                     t=1.0, n_max=10)
        np.testing.assert_allclose(pade_errors, cn_errors, rtol=1e-10)


# ===================================================================
# Unstable Padé: m > n causes divergence
# ===================================================================

class TestUnstablePade:
    def test_unstable_pade_growing_errors(self, medium_grid, sin_medium):
        """Padé [2/1] (m > n, NOT A-stable) should show growing errors at large n."""
        exact = HeatSemigroup.solve_fourier(sin_medium, medium_grid, t=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pade21 = PadeChernoff(2, 1)
        errors = compute_errors(pade21, sin_medium, medium_grid, exact, t=1.0, n_max=10)
        # Errors should grow (or at least not converge) for large n
        # For unstable methods, later errors are typically larger
        assert errors[-1] > errors[2], (
            f"Expected divergence: err[9]={errors[-1]:.3e} should exceed err[2]={errors[2]:.3e}"
        )
