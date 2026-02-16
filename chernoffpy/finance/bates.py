"""Bates PDE pricer via Strang splitting: jumps + Heston operators."""

from __future__ import annotations

import numpy as np

from .bates_analytical import bates_price
from .bates_params import BatesParams, BatesPricingResult
from .heston import HestonPricer
from .heston_params import HestonGridConfig


class BatesPricer:
    """Price European options under Bates model (Heston + Merton jumps)."""

    def __init__(self, chernoff, heston_grid: HestonGridConfig | None = None):
        self.chernoff = chernoff
        self.heston_grid = heston_grid if heston_grid is not None else HestonGridConfig()
        self._heston = HestonPricer(chernoff, self.heston_grid)

    def _jump_multiplier(
        self,
        xi_grid: np.ndarray,
        dt: float,
        lambda_j: float,
        mu_j: float,
        sigma_j: float,
        kbar: float = 0.0,
    ) -> np.ndarray:
        """Fourier multiplier for jump semigroup exp(dt * L_J).

        Includes the risk-neutral drift compensator -λk̄ so that the
        Heston sub-solver can use the true rate r for discounting.
        """
        if lambda_j <= 1e-14 or dt <= 0.0:
            return np.ones_like(xi_grid, dtype=complex)
        p_hat = np.exp(1j * mu_j * xi_grid - 0.5 * sigma_j * sigma_j * xi_grid * xi_grid)
        return np.exp(dt * lambda_j * (p_hat - 1.0 - 1j * xi_grid * kbar))

    @staticmethod
    def _apply_jumps(u: np.ndarray, jump_mult: np.ndarray) -> np.ndarray:
        """Apply jump step to all variance slices (FFT along x-axis)."""
        if np.allclose(jump_mult, 1.0):
            return u
        u_hat = np.fft.fft(u, axis=0)
        u_hat *= jump_mult[:, None]
        return np.real(np.fft.ifft(u_hat, axis=0))

    def price(
        self,
        params: BatesParams,
        n_steps: int = 50,
        option_type: str = "call",
    ) -> BatesPricingResult:
        """Compute option price in Bates model by Strang splitting."""
        if option_type not in {"call", "put"}:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        gc = self.heston_grid
        x_grid = np.linspace(gc.x_min, gc.x_max, gc.n_x)
        v_grid = np.linspace(0.0, gc.v_max, gc.n_v)
        dt = params.T / n_steps

        s_grid = params.K * np.exp(x_grid)
        if option_type == "call":
            payoff = np.maximum(s_grid - params.K, 0.0)
        else:
            payoff = np.maximum(params.K - s_grid, 0.0)

        u = np.tile(payoff[:, None], (1, gc.n_v))

        h_params = params.to_heston()

        dx = x_grid[1] - x_grid[0]
        xi_grid = 2.0 * np.pi * np.fft.fftfreq(len(x_grid), d=dx)
        jump_half = self._jump_multiplier(
            xi_grid,
            0.5 * dt,
            params.lambda_j,
            params.mu_j,
            params.sigma_j,
            kbar=params.kbar,
        )

        for _ in range(n_steps):
            u = self._apply_jumps(u, jump_half)

            u = self._heston._step_lx(u, x_grid, v_grid, h_params, 0.5 * dt)
            u = self._heston._step_lv(u, x_grid, v_grid, h_params, 0.5 * dt, option_type)
            u = self._heston._step_lmix(u, x_grid, v_grid, h_params, dt)
            u = self._heston._step_lv(u, x_grid, v_grid, h_params, 0.5 * dt, option_type)
            u = self._heston._step_lx(u, x_grid, v_grid, h_params, 0.5 * dt)

            u = self._apply_jumps(u, jump_half)

        x0 = float(np.log(params.S / params.K))
        num_price = max(0.0, self._heston._interpolate_2d(u, x_grid, v_grid, x0, params.v0))

        analytical = None
        try:
            analytical = bates_price(
                S=params.S,
                K=params.K,
                T=params.T,
                r=params.r,
                v0=params.v0,
                kappa=params.kappa,
                theta=params.theta,
                xi=params.xi,
                rho=params.rho,
                lambda_j=params.lambda_j,
                mu_j=params.mu_j,
                sigma_j=params.sigma_j,
                option_type=option_type,
            )
        except Exception:
            analytical = None

        return BatesPricingResult(
            price=num_price,
            analytical_price=analytical,
            option_type=option_type,
            method_name=f"Bates-Trotter-{self.chernoff.name}",
            n_steps=n_steps,
            params=params,
        )
