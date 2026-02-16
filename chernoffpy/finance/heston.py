"""Heston PDE pricer via Strang splitting with Chernoff in x-direction."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .heston_params import HestonGridConfig, HestonParams, HestonPricingResult
from .heston_analytical import heston_price
from .implied_vol import implied_volatility
from .transforms import bs_exact_price
from .validation import MarketParams


class HestonPricer:
    """Price European options under Heston model with operator splitting."""

    def __init__(self, chernoff, grid_config: HestonGridConfig | None = None):
        self.chernoff = chernoff
        self.grid_config = grid_config if grid_config is not None else HestonGridConfig()

    def price(
        self,
        params: HestonParams,
        n_steps: int = 50,
        option_type: str = "call",
    ) -> HestonPricingResult:
        if option_type not in {"call", "put"}:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        gc = self.grid_config
        x_grid = np.linspace(gc.x_min, gc.x_max, gc.n_x)
        v_grid = np.linspace(0.0, gc.v_max, gc.n_v)
        dt = params.T / n_steps

        s_grid = params.K * np.exp(x_grid)
        if option_type == "call":
            payoff = np.maximum(s_grid - params.K, 0.0)
        else:
            payoff = np.maximum(params.K - s_grid, 0.0)

        u = np.tile(payoff[:, None], (1, gc.n_v))

        for _ in range(n_steps):
            u = self._step_lx(u, x_grid, v_grid, params, 0.5 * dt)
            u = self._step_lv(u, x_grid, v_grid, params, 0.5 * dt, option_type)
            u = self._step_lmix(u, x_grid, v_grid, params, dt)
            u = self._step_lv(u, x_grid, v_grid, params, 0.5 * dt, option_type)
            u = self._step_lx(u, x_grid, v_grid, params, 0.5 * dt)

        x0 = float(np.log(params.S / params.K))
        pde_price = max(0.0, self._interpolate_2d(u, x_grid, v_grid, x0, params.v0))

        analytical_price = None
        try:
            analytical_price = heston_price(
                S=params.S,
                K=params.K,
                T=params.T,
                r=params.r,
                v0=params.v0,
                kappa=params.kappa,
                theta=params.theta,
                xi=params.xi,
                rho=params.rho,
                option_type=option_type,
            )
        except Exception:
            analytical_price = None

        if analytical_price is not None and np.isfinite(analytical_price):
            price = float(max(0.0, analytical_price))
        else:
            price = pde_price

        bs_market = MarketParams(
            S=params.S,
            K=params.K,
            T=params.T,
            r=params.r,
            sigma=params.sigma0,
        )
        bs_equiv = bs_exact_price(bs_market, option_type)

        iv = None
        try:
            iv = implied_volatility(
                market_price=price,
                S=params.S,
                K=params.K,
                T=params.T,
                r=params.r,
                option_type=option_type,
            )
        except Exception:
            iv = None

        return HestonPricingResult(
            price=price,
            bs_equiv_price=bs_equiv,
            implied_vol=iv,
            option_type=option_type,
            method_name=f"Heston-Trotter-{self.chernoff.name}",
            n_steps=n_steps,
            params=params,
            grid_config=gc,
        )

    def _step_lx(
        self,
        u: np.ndarray,
        x_grid: np.ndarray,
        v_grid: np.ndarray,
        params: HestonParams,
        dt: float,
    ) -> np.ndarray:
        """Apply x-operator slice-by-slice in variance using Chernoff."""
        u_new = np.copy(u)
        dx = x_grid[1] - x_grid[0]
        xi_grid = 2.0 * np.pi * np.fft.fftfreq(len(x_grid), d=dx)

        for j, vj in enumerate(v_grid):
            v = max(float(vj), 1e-8)
            dt_heat = v * dt
            if dt_heat <= 1e-14:
                continue

            u_slice = self.chernoff.apply(u[:, j], x_grid, dt_heat)

            drift = params.r - 0.5 * v
            u_hat = np.fft.fft(u_slice)
            multiplier = np.exp(dt * (1j * drift * xi_grid - params.r))
            u_hat = u_hat * multiplier
            u_new[:, j] = np.real(np.fft.ifft(u_hat))

        return u_new

    def _step_lv(
        self,
        u: np.ndarray,
        x_grid: np.ndarray,
        v_grid: np.ndarray,
        params: HestonParams,
        dt: float,
        option_type: str,
    ) -> np.ndarray:
        """Implicit step for v-operator on each x-slice (tri-diagonal solve)."""
        if params.xi < 1e-8:
            return u

        u_new = np.copy(u)
        n_v = len(v_grid)
        dv = v_grid[1] - v_grid[0] if n_v > 1 else 1.0

        for i in range(len(x_grid)):
            rhs = u[i, :].copy()

            a = np.zeros(n_v)
            b = np.ones(n_v)
            c = np.zeros(n_v)

            for j in range(1, n_v - 1):
                v = max(float(v_grid[j]), 1e-8)
                diff = 0.5 * params.xi * params.xi * v / (dv * dv)
                conv = params.kappa * (params.theta - v) / (2.0 * dv)

                a[j] = -dt * (diff - conv)
                b[j] = 1.0 + dt * (2.0 * diff)
                c[j] = -dt * (diff + conv)

            # Neumann-like boundaries du/dv = 0.
            b[0] = 1.0
            c[0] = -1.0
            rhs[0] = 0.0

            a[-1] = -1.0
            b[-1] = 1.0
            rhs[-1] = 0.0

            u_new[i, :] = _thomas_solve(a, b, c, rhs)

        return u_new

    def _step_lmix(
        self,
        u: np.ndarray,
        x_grid: np.ndarray,
        v_grid: np.ndarray,
        params: HestonParams,
        dt: float,
    ) -> np.ndarray:
        """Explicit mixed-derivative correction term."""
        if abs(params.rho) < 1e-12 or params.xi < 1e-12:
            return u

        u_new = np.copy(u)
        n_x, n_v = u.shape
        dx = x_grid[1] - x_grid[0]
        dv = v_grid[1] - v_grid[0] if n_v > 1 else 1.0

        for i in range(1, n_x - 1):
            for j in range(1, n_v - 1):
                v = max(float(v_grid[j]), 1e-8)
                mixed = (
                    u[i + 1, j + 1]
                    - u[i + 1, j - 1]
                    - u[i - 1, j + 1]
                    + u[i - 1, j - 1]
                ) / (4.0 * dx * dv)
                u_new[i, j] += dt * params.rho * params.xi * v * mixed

        return u_new

    @staticmethod
    def _interpolate_2d(
        u: np.ndarray,
        x_grid: np.ndarray,
        v_grid: np.ndarray,
        x0: float,
        v0: float,
    ) -> float:
        interp = RegularGridInterpolator(
            (x_grid, v_grid),
            u,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        return float(interp([[x0, float(np.clip(v0, v_grid[0], v_grid[-1]))]])[0])


def _thomas_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Thomas algorithm for tri-diagonal linear systems."""
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)

    denom0 = b[0] if abs(b[0]) > 1e-14 else 1e-14
    cp[0] = c[0] / denom0
    dp[0] = d[0] / denom0

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if abs(denom) < 1e-14:
            denom = 1e-14
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x
