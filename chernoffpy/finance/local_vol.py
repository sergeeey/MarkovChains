"""Local volatility pricing via frozen-coefficient Chernoff steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .transforms import bs_to_heat_initial, extract_price_at_spot, make_grid
from .validation import GridConfig, MarketParams


class VolSurface(Protocol):
    """Volatility surface sigma(S, t)."""

    def __call__(self, S: float | np.ndarray, t: float) -> float | np.ndarray:
        ...


@dataclass(frozen=True)
class LocalVolParams:
    """Model parameters for local-vol pricing."""

    S: float
    K: float
    T: float
    r: float
    vol_surface: VolSurface

    def __post_init__(self):
        if self.S <= 0:
            raise ValueError(f"S must be > 0, got {self.S}")
        if self.K <= 0:
            raise ValueError(f"K must be > 0, got {self.K}")
        if self.T <= 0:
            raise ValueError(f"T must be > 0, got {self.T}")
        if self.r < 0:
            raise ValueError(f"r must be >= 0, got {self.r}")
        if not callable(self.vol_surface):
            raise ValueError("vol_surface must be callable")


@dataclass
class LocalVolResult:
    """Pricing result for local-vol solver."""

    price: float
    method_name: str
    n_steps: int
    option_type: str
    sigma_effective_mean: float


class LocalVolPricer:
    """Price options under local volatility with frozen coefficients.

    The scheme uses a spot-centered effective volatility and a predictor-corrector
    step to reduce single-sigma reduction bias.
    """

    def __init__(self, chernoff, grid_config: GridConfig | None = None):
        self.chernoff = chernoff
        self.grid_config = grid_config if grid_config is not None else GridConfig()

    def _effective_sigma(
        self,
        x_grid: np.ndarray,
        s_grid: np.ndarray,
        u: np.ndarray,
        x0: float,
        t: float,
        vol_surface: VolSurface,
    ) -> float:
        """Compute robust effective sigma around spot on current step."""
        sigma_local = vol_surface(s_grid, t)
        sigma_arr = np.asarray(sigma_local, dtype=float)
        if sigma_arr.ndim == 0:
            sigma_arr = np.full_like(s_grid, float(sigma_arr))

        # Guard against pathological surfaces: replace NaN with ATM-like vol (20%),
        # clip to [1e-6, 3.0] (300% vol upper bound covers extreme skew scenarios).
        sigma_arr = np.nan_to_num(sigma_arr, nan=0.2, posinf=5.0, neginf=1e-6)
        sigma_arr = np.clip(sigma_arr, 1e-6, 3.0)

        # Spot-centered Gaussian weights suppress unstable far-tail influence.
        # Width 0.75 in log-moneyness units ≈ ±75% from spot — captures the
        # local vol smile near ATM without contamination from deep OTM nodes.
        width = 0.75
        w_spot = np.exp(-0.5 * ((x_grid - x0) / width) ** 2)

        # Blend locality with current solution magnitude.
        w_sol = np.abs(u) + 1e-12
        weights = w_spot * w_sol
        denom = np.sum(weights)
        if denom <= 0:
            return float(np.clip(np.interp(x0, x_grid, sigma_arr), 1e-6, 3.0))

        sigma_eff = float(np.sum(weights * sigma_arr) / denom)
        return float(np.clip(sigma_eff, 1e-6, 3.0))

    def price(
        self,
        params: LocalVolParams,
        n_steps: int = 100,
        option_type: str = "call",
    ) -> LocalVolResult:
        """Price an option under local volatility using predictor-corrector Chernoff steps.

        Each step computes an effective sigma via Gaussian-weighted averaging
        around the spot, then applies the Chernoff operator with dt_heat = 0.5*sigma^2*dt.
        """
        if option_type not in {"call", "put"}:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        cfg = self.grid_config
        x_grid = make_grid(cfg)
        s_grid = params.K * np.exp(x_grid)
        x0 = np.log(params.S / params.K)

        sigma0 = float(np.asarray(params.vol_surface(params.S, params.T)))
        sigma0 = float(np.clip(np.nan_to_num(sigma0, nan=0.2, posinf=3.0, neginf=1e-6), 1e-6, 3.0))

        market_ref = MarketParams(
            S=params.S,
            K=params.K,
            T=params.T,
            r=params.r,
            sigma=sigma0,
        )

        u = bs_to_heat_initial(x_grid, market_ref, cfg, option_type)

        dt = params.T / n_steps
        sigma_history: list[float] = []

        for step in range(n_steps):
            t_left = params.T - step * dt

            # Predictor sigma at left endpoint.
            sigma_pred = self._effective_sigma(
                x_grid, s_grid, u, x0, t_left, params.vol_surface
            )
            dt_heat_pred = 0.5 * sigma_pred ** 2 * dt
            u_pred = self.chernoff.apply(u, x_grid, dt_heat_pred)

            # Corrector sigma at midpoint using predicted state.
            sigma_corr = self._effective_sigma(
                x_grid,
                s_grid,
                0.5 * (u + u_pred),
                x0,
                max(0.0, t_left - 0.5 * dt),
                params.vol_surface,
            )

            sigma_step = 0.5 * (sigma_pred + sigma_corr)
            sigma_history.append(sigma_step)

            dt_heat = 0.5 * sigma_step ** 2 * dt
            u = self.chernoff.apply(u, x_grid, dt_heat)

        price = max(0.0, extract_price_at_spot(u, x_grid, market_ref))
        sigma_mean = float(np.mean(sigma_history)) if sigma_history else sigma0

        return LocalVolResult(
            price=price,
            method_name=self.chernoff.name,
            n_steps=n_steps,
            option_type=option_type,
            sigma_effective_mean=sigma_mean,
        )


def flat_vol(sigma: float) -> VolSurface:
    """Constant volatility surface sigma(S,t)=const."""

    def _vol(S: float | np.ndarray, t: float) -> float | np.ndarray:
        return sigma

    return _vol


def linear_skew(sigma_atm: float, skew: float, S_ref: float) -> VolSurface:
    """Linear log-skew surface: sigma = sigma_atm + skew*ln(S/S_ref)."""

    def _vol(S: float | np.ndarray, t: float) -> float | np.ndarray:
        return sigma_atm + skew * np.log(np.asarray(S) / S_ref)

    return _vol


def time_dependent_vol(sigmas: list[float], times: list[float]) -> VolSurface:
    """Piecewise-constant time curve sigma(t)."""
    if len(sigmas) == 0:
        raise ValueError("sigmas must be non-empty")
    if len(times) < 2:
        raise ValueError("times must contain at least two points")
    if len(sigmas) != len(times) - 1:
        raise ValueError("len(sigmas) must equal len(times) - 1")

    def _vol(S: float | np.ndarray, t: float) -> float | np.ndarray:
        for i in range(len(times) - 1):
            if t <= times[i + 1]:
                return sigmas[i]
        return sigmas[-1]

    return _vol
