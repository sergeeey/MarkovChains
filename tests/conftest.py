"""Shared fixtures for ChernoffPy tests."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

@pytest.fixture
def small_grid():
    """N=64 uniform grid on [-pi, pi) — fast unit tests."""
    return np.linspace(-np.pi, np.pi, 64, endpoint=False)


@pytest.fixture
def medium_grid():
    """N=128 uniform grid on [-pi, pi) — convergence tests."""
    return np.linspace(-np.pi, np.pi, 128, endpoint=False)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

@pytest.fixture
def sin_initial(small_grid):
    """sin(x) on small_grid — smooth, eigenfunction of Laplacian."""
    return np.sin(small_grid)


@pytest.fixture
def gaussian_initial(small_grid):
    """Gaussian exp(-x^2) on small_grid — smooth, decays fast."""
    return np.exp(-small_grid**2)


@pytest.fixture
def abs_sin_initial(small_grid):
    """|sin(x)| on small_grid — only C^0, limited regularity."""
    return np.abs(np.sin(small_grid))


@pytest.fixture
def sin_medium(medium_grid):
    """sin(x) on medium_grid."""
    return np.sin(medium_grid)


@pytest.fixture
def abs_sin_medium(medium_grid):
    """|sin(x)| on medium_grid."""
    return np.abs(np.sin(medium_grid))
