"""Data classes for Heston stochastic-volatility pricing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HestonParams:
    """Input parameters of the Heston model."""

    S: float
    K: float
    T: float
    r: float
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def __post_init__(self):
        if self.S <= 0:
            raise ValueError(f"S must be > 0, got {self.S}")
        if self.K <= 0:
            raise ValueError(f"K must be > 0, got {self.K}")
        if self.T <= 0:
            raise ValueError(f"T must be > 0, got {self.T}")
        if self.r < 0:
            raise ValueError(f"r must be >= 0, got {self.r}")
        if self.v0 < 0:
            raise ValueError(f"v0 must be >= 0, got {self.v0}")
        if self.kappa < 0:
            raise ValueError(f"kappa must be >= 0, got {self.kappa}")
        if self.theta < 0:
            raise ValueError(f"theta must be >= 0, got {self.theta}")
        if self.xi < 0:
            raise ValueError(f"xi must be >= 0, got {self.xi}")
        if not (-1.0 < self.rho < 1.0):
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}")

    @property
    def feller_condition(self) -> bool:
        """Return True if the Feller condition is satisfied."""
        return 2.0 * self.kappa * self.theta > self.xi * self.xi

    @property
    def sigma0(self) -> float:
        """Initial volatility sqrt(v0)."""
        return float(np.sqrt(max(self.v0, 0.0)))


@dataclass(frozen=True)
class HestonGridConfig:
    """2D finite-difference/splitting grid for Heston PDE."""

    n_x: int = 256
    n_v: int = 64
    x_min: float = -5.0
    x_max: float = 5.0
    v_max: float = 1.0

    def __post_init__(self):
        if self.n_x < 32:
            raise ValueError(f"n_x must be >= 32, got {self.n_x}")
        if self.n_v < 16:
            raise ValueError(f"n_v must be >= 16, got {self.n_v}")
        if self.x_min >= self.x_max:
            raise ValueError(
                f"x_min must be < x_max, got x_min={self.x_min}, x_max={self.x_max}"
            )
        if self.v_max <= 0:
            raise ValueError(f"v_max must be > 0, got {self.v_max}")


@dataclass
class HestonPricingResult:
    """Output bundle for Heston pricing."""

    price: float
    bs_equiv_price: float
    implied_vol: float | None
    option_type: str
    method_name: str
    n_steps: int
    params: HestonParams
    grid_config: HestonGridConfig
