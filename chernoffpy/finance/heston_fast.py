"""Fast Heston pricer with batch FFT and optional Numba acceleration."""

from __future__ import annotations

import numpy as np

from ..accel import mixed_deriv_step, thomas_solve_batch
from .heston import HestonPricer
from .heston_params import HestonParams


class HestonFastPricer(HestonPricer):
    """Drop-in faster variant of :class:`HestonPricer`."""

    def _step_lx(
        self,
        u: np.ndarray,
        x_grid: np.ndarray,
        v_grid: np.ndarray,
        params: HestonParams,
        dt: float,
    ) -> np.ndarray:
        """Batch FFT over all v-slices at once (axis=0)."""
        n_x = len(x_grid)
        dx = x_grid[1] - x_grid[0]
        xi_grid = 2.0 * np.pi * np.fft.fftfreq(n_x, d=dx)
        xi_sq = xi_grid * xi_grid

        v_safe = np.maximum(v_grid, 1e-8)
        dt_heat = 0.5 * v_safe * dt

        u_hat = np.fft.fft(u, axis=0)

        if hasattr(self.chernoff, "multiplier"):
            multiplier = self.chernoff.multiplier(xi_sq[:, None], dt_heat[None, :])
        else:
            z = xi_sq[:, None] * dt_heat[None, :]
            name = self.chernoff.__class__.__name__.lower()
            if "crank" in name or "nicolson" in name:
                multiplier = (1.0 - 0.5 * z) / (1.0 + 0.5 * z)
            elif "pade" in name:
                multiplier = (1.0 - 0.5 * z + z * z / 12.0) / (1.0 + 0.5 * z + z * z / 12.0)
            else:
                multiplier = 1.0 / (1.0 + z)

        drift = params.r - 0.5 * v_safe
        discount = np.exp(dt * (1j * xi_grid[:, None] * drift[None, :] - params.r))

        u_hat *= multiplier * discount
        return np.real(np.fft.ifft(u_hat, axis=0))

    def _step_lv(
        self,
        u: np.ndarray,
        x_grid: np.ndarray,
        v_grid: np.ndarray,
        params: HestonParams,
        dt: float,
        option_type: str,
    ) -> np.ndarray:
        """Batched Thomas solve for v-operator (Numba if available)."""
        if params.xi < 1e-8:
            return u
        return thomas_solve_batch(
            u,
            v_grid,
            params.kappa,
            params.theta,
            params.xi,
            dt,
        )

    def _step_lmix(
        self,
        u: np.ndarray,
        x_grid: np.ndarray,
        v_grid: np.ndarray,
        params: HestonParams,
        dt: float,
    ) -> np.ndarray:
        """Mixed derivative step via Numba kernel or vectorized NumPy fallback."""
        if abs(params.rho) < 1e-12 or params.xi < 1e-12:
            return u

        dx = x_grid[1] - x_grid[0]
        dv = v_grid[1] - v_grid[0] if len(v_grid) > 1 else 1.0

        return mixed_deriv_step(
            u,
            v_grid,
            params.rho,
            params.xi,
            dx,
            dv,
            dt,
        )
