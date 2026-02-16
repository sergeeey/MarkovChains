"""Data classes for Bates (Heston + jumps) option pricing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .heston_params import HestonParams


@dataclass(frozen=True)
class BatesParams:
    """Input parameters of the Bates model (Heston + Merton jumps)."""

    S: float
    K: float
    T: float
    r: float

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    lambda_j: float
    mu_j: float
    sigma_j: float

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
        if self.lambda_j < 0:
            raise ValueError(f"lambda_j must be >= 0, got {self.lambda_j}")
        if self.sigma_j < 0:
            raise ValueError(f"sigma_j must be >= 0, got {self.sigma_j}")

    @property
    def kbar(self) -> float:
        """Jump compensator E[e^J - 1] for J ~ N(mu_j, sigma_j^2)."""
        return float(np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1.0)

    def to_heston(self, r_override: float | None = None) -> HestonParams:
        """Convert to equivalent Heston parameters (no jumps)."""
        return HestonParams(
            S=self.S,
            K=self.K,
            T=self.T,
            r=self.r if r_override is None else r_override,
            v0=self.v0,
            kappa=self.kappa,
            theta=self.theta,
            xi=self.xi,
            rho=self.rho,
        )


@dataclass
class BatesPricingResult:
    """Output result bundle for Bates pricing."""

    price: float
    analytical_price: float | None
    option_type: str
    method_name: str
    n_steps: int
    params: BatesParams
