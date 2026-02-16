"""Tests for optional acceleration kernels and backend helpers."""

from __future__ import annotations

import numpy as np
import pytest

from chernoffpy.accel import (
    HAS_NUMBA,
    _mixed_deriv_numpy,
    _thomas_batch_numpy,
    mixed_deriv_step,
    thomas_solve_batch,
)
from chernoffpy.backends import HAS_CUPY, get_backend, to_backend, to_numpy


@pytest.fixture
def sample_u() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(size=(32, 24))


@pytest.fixture
def sample_v_grid() -> np.ndarray:
    return np.linspace(0.0, 0.8, 24)


class TestThomasBatch:

    def test_output_matches_numpy(self, sample_u: np.ndarray, sample_v_grid: np.ndarray):
        expected = _thomas_batch_numpy(sample_u, sample_v_grid, 2.0, 0.04, 0.3, 0.01)
        got = thomas_solve_batch(sample_u, sample_v_grid, 2.0, 0.04, 0.3, 0.01)
        assert np.allclose(got, expected, rtol=1e-11, atol=1e-11)

    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba is not installed")
    def test_parallel_matches_numpy(self, sample_u: np.ndarray, sample_v_grid: np.ndarray):
        expected = _thomas_batch_numpy(sample_u, sample_v_grid, 1.7, 0.06, 0.45, 0.02)
        got = thomas_solve_batch(sample_u, sample_v_grid, 1.7, 0.06, 0.45, 0.02)
        assert np.allclose(got, expected, rtol=1e-11, atol=1e-11)

    def test_identity_for_zero_dt(self, sample_u: np.ndarray, sample_v_grid: np.ndarray):
        got = thomas_solve_batch(sample_u, sample_v_grid, 2.0, 0.04, 0.3, 0.0)
        assert np.allclose(got, sample_u)

    def test_handles_v_zero(self, sample_u: np.ndarray):
        v_grid = np.zeros(sample_u.shape[1])
        got = thomas_solve_batch(sample_u, v_grid, 2.0, 0.04, 0.3, 0.01)
        assert np.isfinite(got).all()


class TestMixedDeriv:

    def test_output_matches_numpy(self, sample_u: np.ndarray, sample_v_grid: np.ndarray):
        dx = 0.05
        dv = sample_v_grid[1] - sample_v_grid[0]
        expected = _mixed_deriv_numpy(sample_u, sample_v_grid, -0.7, 0.3, dx, dv, 0.01)
        got = mixed_deriv_step(sample_u, sample_v_grid, -0.7, 0.3, dx, dv, 0.01)
        assert np.allclose(got, expected, rtol=1e-11, atol=1e-11)

    def test_zero_rho_is_identity(self, sample_u: np.ndarray, sample_v_grid: np.ndarray):
        dx = 0.05
        dv = sample_v_grid[1] - sample_v_grid[0]
        got = mixed_deriv_step(sample_u, sample_v_grid, 0.0, 0.3, dx, dv, 0.01)
        assert np.allclose(got, sample_u)

    def test_sign_changes_with_rho(self, sample_u: np.ndarray, sample_v_grid: np.ndarray):
        dx = 0.05
        dv = sample_v_grid[1] - sample_v_grid[0]
        pos = mixed_deriv_step(sample_u, sample_v_grid, 0.5, 0.3, dx, dv, 0.01)
        neg = mixed_deriv_step(sample_u, sample_v_grid, -0.5, 0.3, dx, dv, 0.01)
        assert np.linalg.norm(pos - sample_u) > 0
        assert np.linalg.norm(neg - sample_u) > 0
        assert np.linalg.norm((pos - sample_u) + (neg - sample_u)) < np.linalg.norm(pos - sample_u)


class TestBackend:

    def test_numpy_default(self):
        xp = get_backend()
        assert xp is np

    def test_cupy_optional(self):
        if HAS_CUPY:
            assert get_backend("cupy").__name__ == "cupy"
        else:
            with pytest.raises(ImportError):
                get_backend("cupy")

    def test_to_numpy_idempotent(self):
        arr = np.array([1.0, 2.0, 3.0])
        out = to_numpy(arr)
        assert out is arr

    def test_to_backend_numpy(self):
        arr = [1, 2, 3]
        out = to_backend(arr, np)
        assert isinstance(out, np.ndarray)
