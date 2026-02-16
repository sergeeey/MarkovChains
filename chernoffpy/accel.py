"""Optional acceleration kernels (NumPy fallback + Numba JIT path)."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:  # pragma: no cover - depends on environment
    HAS_NUMBA = False
    njit = None
    prange = range


def thomas_solve_batch(
    u: np.ndarray,
    v_grid: np.ndarray,
    kappa: float,
    theta: float,
    xi_param: float,
    dt: float,
) -> np.ndarray:
    """Solve all x-slices of tri-diagonal v-step systems."""
    if dt <= 0.0:
        return np.copy(u)
    if HAS_NUMBA:
        return _thomas_batch_numba(
            u,
            v_grid,
            float(kappa),
            float(theta),
            float(xi_param),
            float(dt),
            u.shape[0],
            u.shape[1],
        )
    return _thomas_batch_numpy(u, v_grid, kappa, theta, xi_param, dt)


def mixed_deriv_step(
    u: np.ndarray,
    v_grid: np.ndarray,
    rho: float,
    xi_param: float,
    dx: float,
    dv: float,
    dt: float,
) -> np.ndarray:
    """Apply explicit mixed derivative correction term."""
    if dt <= 0.0 or abs(rho) < 1e-12 or abs(xi_param) < 1e-12:
        return np.copy(u)

    if HAS_NUMBA:
        return _mixed_deriv_numba(
            u,
            v_grid,
            float(rho),
            float(xi_param),
            float(dx),
            float(dv),
            float(dt),
            u.shape[0],
            u.shape[1],
        )
    return _mixed_deriv_numpy(u, v_grid, rho, xi_param, dx, dv, dt)


def _thomas_batch_numpy(
    u: np.ndarray,
    v_grid: np.ndarray,
    kappa: float,
    theta: float,
    xi_param: float,
    dt: float,
) -> np.ndarray:
    """NumPy reference implementation for batched Thomas solves."""
    n_x, n_v = u.shape
    dv = max(v_grid[1] - v_grid[0], 1e-12) if n_v > 1 else 1.0
    u_new = np.empty_like(u)

    for i in range(n_x):
        rhs = np.copy(u[i, :])
        a = np.zeros(n_v, dtype=u.dtype)
        b = np.ones(n_v, dtype=u.dtype)
        c = np.zeros(n_v, dtype=u.dtype)

        for j in range(1, n_v - 1):
            v = max(float(v_grid[j]), 1e-8)
            diff = 0.5 * xi_param * xi_param * v / (dv * dv)
            conv = kappa * (theta - v) / (2.0 * dv)
            a[j] = -dt * (diff - conv)
            b[j] = 1.0 + dt * (2.0 * diff)
            c[j] = -dt * (diff + conv)

        b[0] = 1.0
        c[0] = -1.0
        rhs[0] = 0.0

        a[-1] = -1.0
        b[-1] = 1.0
        rhs[-1] = 0.0

        cp = np.zeros(n_v, dtype=u.dtype)
        dp = np.zeros(n_v, dtype=u.dtype)

        denom0 = b[0] if abs(b[0]) > 1e-14 else 1e-14
        cp[0] = c[0] / denom0
        dp[0] = rhs[0] / denom0

        for j in range(1, n_v):
            denom = b[j] - a[j] * cp[j - 1]
            if abs(denom) < 1e-14:
                denom = 1e-14
            cp[j] = c[j] / denom if j < n_v - 1 else 0.0
            dp[j] = (rhs[j] - a[j] * dp[j - 1]) / denom

        x = np.zeros(n_v, dtype=u.dtype)
        x[-1] = dp[-1]
        for j in range(n_v - 2, -1, -1):
            x[j] = dp[j] - cp[j] * x[j + 1]

        u_new[i, :] = x

    return u_new


def _mixed_deriv_numpy(
    u: np.ndarray,
    v_grid: np.ndarray,
    rho: float,
    xi_param: float,
    dx: float,
    dv: float,
    dt: float,
) -> np.ndarray:
    """Vectorized NumPy implementation for mixed derivative step."""
    u_new = np.copy(u)
    v_safe = np.maximum(v_grid[1:-1], 1e-8)
    mixed = (u[2:, 2:] - u[2:, :-2] - u[:-2, 2:] + u[:-2, :-2]) / (4.0 * dx * dv)
    u_new[1:-1, 1:-1] += dt * rho * xi_param * v_safe[np.newaxis, :] * mixed
    return u_new


if HAS_NUMBA:

    @njit(parallel=True, cache=True)
    def _thomas_batch_numba(
        u: np.ndarray,
        v_grid: np.ndarray,
        kappa: float,
        theta: float,
        xi_param: float,
        dt: float,
        n_x: int,
        n_v: int,
    ) -> np.ndarray:
        """Numba-parallel implementation of batched Thomas solves."""
        u_new = np.empty((n_x, n_v), dtype=np.float64)
        dv = max(v_grid[1] - v_grid[0], 1e-12) if n_v > 1 else 1.0

        for i in prange(n_x):
            rhs = np.empty(n_v, dtype=np.float64)
            a = np.zeros(n_v, dtype=np.float64)
            b = np.ones(n_v, dtype=np.float64)
            c = np.zeros(n_v, dtype=np.float64)

            for j in range(n_v):
                rhs[j] = u[i, j]

            for j in range(1, n_v - 1):
                v = v_grid[j]
                if v < 1e-8:
                    v = 1e-8
                diff = 0.5 * xi_param * xi_param * v / (dv * dv)
                conv = kappa * (theta - v) / (2.0 * dv)
                a[j] = -dt * (diff - conv)
                b[j] = 1.0 + dt * (2.0 * diff)
                c[j] = -dt * (diff + conv)

            b[0] = 1.0
            c[0] = -1.0
            rhs[0] = 0.0

            a[n_v - 1] = -1.0
            b[n_v - 1] = 1.0
            rhs[n_v - 1] = 0.0

            cp = np.zeros(n_v, dtype=np.float64)
            dp = np.zeros(n_v, dtype=np.float64)

            denom0 = b[0]
            if abs(denom0) < 1e-14:
                denom0 = 1e-14
            cp[0] = c[0] / denom0
            dp[0] = rhs[0] / denom0

            for j in range(1, n_v):
                denom = b[j] - a[j] * cp[j - 1]
                if abs(denom) < 1e-14:
                    denom = 1e-14
                if j < n_v - 1:
                    cp[j] = c[j] / denom
                else:
                    cp[j] = 0.0
                dp[j] = (rhs[j] - a[j] * dp[j - 1]) / denom

            u_new[i, n_v - 1] = dp[n_v - 1]
            for j in range(n_v - 2, -1, -1):
                u_new[i, j] = dp[j] - cp[j] * u_new[i, j + 1]

        return u_new


    @njit(cache=True)
    def _mixed_deriv_numba(
        u: np.ndarray,
        v_grid: np.ndarray,
        rho: float,
        xi_param: float,
        dx: float,
        dv: float,
        dt: float,
        n_x: int,
        n_v: int,
    ) -> np.ndarray:
        """Numba implementation of explicit mixed derivative correction."""
        u_new = np.copy(u)
        coeff = rho * xi_param * dt / (4.0 * dx * dv)

        for i in range(1, n_x - 1):
            for j in range(1, n_v - 1):
                v = v_grid[j]
                if v < 1e-8:
                    v = 1e-8
                mixed = u[i + 1, j + 1] - u[i + 1, j - 1] - u[i - 1, j + 1] + u[i - 1, j - 1]
                u_new[i, j] += coeff * v * mixed

        return u_new

