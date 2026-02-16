"""Data classes for option pricing and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


BarrierType = Literal[
    "down_and_out",
    "down_and_in",
    "up_and_out",
    "up_and_in",
]

DoubleBarrierType = Literal["double_knock_out", "double_knock_in"]
ExerciseType = Literal["european", "american"]


@dataclass(frozen=True)
class MarketParams:
    """Market parameters for option pricing. All values validated on creation."""

    S: float      # Spot price
    K: float      # Strike price
    T: float      # Time to expiry (years)
    r: float      # Risk-free rate (annualized)
    sigma: float  # Volatility (annualized)
    q: float = 0.0  # Continuous dividend yield

    def __post_init__(self):
        if self.S <= 0:
            raise ValueError(f"Spot price S must be positive, got {self.S}")
        if self.K <= 0:
            raise ValueError(f"Strike price K must be positive, got {self.K}")
        if self.T <= 0:
            raise ValueError(f"Time to expiry T must be positive, got {self.T}")
        if self.r < 0:
            raise ValueError(f"Risk-free rate r must be non-negative, got {self.r}")
        if self.sigma <= 0:
            raise ValueError(f"Volatility sigma must be positive, got {self.sigma}")
        if self.q < 0:
            raise ValueError(f"Dividend yield q must be non-negative, got {self.q}")


@dataclass(frozen=True)
class DividendSchedule:
    """Schedule of discrete dividends for pricing with jumps in spot level."""

    times: tuple[float, ...]
    amounts: tuple[float, ...]
    proportional: bool = False

    def __post_init__(self):
        if len(self.times) != len(self.amounts):
            raise ValueError("times and amounts must have same length")
        if len(self.times) == 0:
            return
        if tuple(sorted(self.times)) != self.times:
            raise ValueError("Dividend times must be sorted")
        for t in self.times:
            if t <= 0:
                raise ValueError(f"Dividend time must be > 0, got {t}")
        for a in self.amounts:
            if a <= 0:
                raise ValueError(f"Dividend amount must be > 0, got {a}")
        if self.proportional and any(a >= 1.0 for a in self.amounts):
            raise ValueError("Proportional dividend amounts must be < 1.0")


@dataclass
class GridConfig:
    """Spatial grid configuration for the heat equation solver."""

    N: int = 2048            # Number of grid points (power of 2 recommended for FFT)
    L: float = 10.0          # Domain half-width: grid spans [-L, L)
    taper_width: float = 2.0 # Width of cosine taper zone at each boundary
    # Phase 8 (optional, backward-compatible)
    snap_points: tuple[float, ...] = ()
    center: float = 0.0
    stretch: float = 0.0

    def __post_init__(self):
        if self.N < 64:
            raise ValueError(f"Grid size N must be >= 64, got {self.N}")
        if self.L <= 0:
            raise ValueError(f"Domain half-width L must be positive, got {self.L}")
        if self.taper_width <= 0 or self.taper_width >= self.L:
            raise ValueError(
                f"Taper width must be in (0, L={self.L}), got {self.taper_width}"
            )
        if not isinstance(self.snap_points, tuple):
            raise ValueError("snap_points must be a tuple")
        if len(self.snap_points) > 2:
            raise ValueError("snap_points supports up to 2 points")
        if self.stretch < 0:
            raise ValueError(f"stretch must be >= 0, got {self.stretch}")


@dataclass(frozen=True)
class BarrierParams:
    """Barrier option parameters."""

    barrier: float
    barrier_type: BarrierType
    rebate: float = 0.0

    def __post_init__(self):
        if self.barrier <= 0:
            raise ValueError(f"barrier must be > 0, got {self.barrier}")
        if self.rebate < 0:
            raise ValueError(f"rebate must be >= 0, got {self.rebate}")


@dataclass(frozen=True)
class DoubleBarrierParams:
    """Parameters for double barrier options."""

    lower_barrier: float
    upper_barrier: float
    barrier_type: DoubleBarrierType
    rebate: float = 0.0

    def __post_init__(self):
        if self.lower_barrier <= 0:
            raise ValueError(
                f"lower_barrier must be > 0, got {self.lower_barrier}"
            )
        if self.upper_barrier <= 0:
            raise ValueError(
                f"upper_barrier must be > 0, got {self.upper_barrier}"
            )
        if self.lower_barrier >= self.upper_barrier:
            raise ValueError(
                f"lower_barrier ({self.lower_barrier}) must be < "
                f"upper_barrier ({self.upper_barrier})"
            )
        if self.rebate < 0:
            raise ValueError(f"rebate must be >= 0, got {self.rebate}")


@dataclass
class ValidationCertificate:
    """Accuracy certificate decomposing total error into components.

    Total error = chernoff_error + domain_error (approximately).
    - chernoff_error: |Chernoff price - FFT exact price| (approximation quality)
    - domain_error:   |FFT exact price - BS exact price|  (truncation + taper)
    """

    bs_price: float        # Exact Black-Scholes analytical price
    computed_price: float  # Price from Chernoff approximation
    abs_error: float       # |computed - bs_exact|
    rel_error: float       # |computed - bs_exact| / bs_exact
    chernoff_error: float  # |chernoff - fft_exact|
    domain_error: float    # |fft_exact - bs_exact|
    certified_bound: float | None = None  # a priori certified upper bound
    certified_order: int | None = None    # theoretical effective order
    is_certified: bool = False            # whether certified bound is available


@dataclass
class PricingResult:
    """Complete result of vanilla option pricing."""

    price: float
    method_name: str
    n_steps: int
    market: MarketParams
    certificate: ValidationCertificate


@dataclass
class BarrierPricingResult:
    """Complete result of barrier option pricing."""

    price: float
    vanilla_price: float
    knockout_price: float
    barrier_type: BarrierType
    method_name: str
    n_steps: int
    market: MarketParams
    barrier_params: BarrierParams
    certificate: ValidationCertificate | None = None


@dataclass
class DoubleBarrierPricingResult:
    """Complete result of double-barrier option pricing."""

    price: float
    vanilla_price: float
    knockout_price: float
    barrier_type: DoubleBarrierType
    method_name: str
    n_steps: int
    market: MarketParams
    barrier_params: DoubleBarrierParams
    certificate: ValidationCertificate | None = None


@dataclass
class AmericanPricingResult:
    """Complete result of American option pricing."""

    price: float
    european_price: float
    early_exercise_premium: float
    exercise_boundary: np.ndarray | None
    option_type: str
    method_name: str
    n_steps: int
    market: MarketParams
    certificate: ValidationCertificate | None = None


@dataclass
class GreeksResult:
    """Option Greeks (sensitivities)."""

    delta: float  # dV/dS
    gamma: float  # d2V/dS2
    vega: float   # dV/dsigma
    theta: float  # dV/dt (= -dV/dT, typically negative for long options)
    rho: float    # dV/dr
