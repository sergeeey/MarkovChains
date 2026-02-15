"""Data classes for European option pricing with accuracy certificates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketParams:
    """Market parameters for option pricing. All values validated on creation."""

    S: float      # Spot price
    K: float      # Strike price
    T: float      # Time to expiry (years)
    r: float      # Risk-free rate (annualized)
    sigma: float  # Volatility (annualized)

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


@dataclass
class GridConfig:
    """Spatial grid configuration for the heat equation solver."""

    N: int = 2048            # Number of grid points (power of 2 recommended for FFT)
    L: float = 10.0          # Domain half-width: grid spans [-L, L)
    taper_width: float = 2.0 # Width of cosine taper zone at each boundary

    def __post_init__(self):
        if self.N < 64:
            raise ValueError(f"Grid size N must be >= 64, got {self.N}")
        if self.L <= 0:
            raise ValueError(f"Domain half-width L must be positive, got {self.L}")
        if self.taper_width <= 0 or self.taper_width >= self.L:
            raise ValueError(
                f"Taper width must be in (0, L={self.L}), got {self.taper_width}"
            )


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


@dataclass
class PricingResult:
    """Complete result of option pricing."""

    price: float
    method_name: str
    n_steps: int
    market: MarketParams
    certificate: ValidationCertificate


@dataclass
class GreeksResult:
    """Option Greeks (sensitivities)."""

    delta: float  # dV/dS
    gamma: float  # d²V/dS²
    vega: float   # dV/dσ
    theta: float  # dV/dt (= -dV/dT, typically negative for long options)
    rho: float    # dV/dr
