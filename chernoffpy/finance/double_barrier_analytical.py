"""Analytical double-barrier pricing (Fourier series reference implementation).

Used as a ground-truth approximation in tests.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _bs_price(S: float, K: float, r: float, sigma: float, T: float, option_type: str) -> float:
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    if option_type == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    if option_type == "put":
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def _u0_on_interval(
    x: np.ndarray,
    K: float,
    r: float,
    sigma: float,
    option_type: str,
) -> np.ndarray:
    """Initial condition for heat equation after Wilmott substitution."""
    alpha = -(r / sigma ** 2 - 0.5)

    if option_type == "call":
        payoff = np.maximum(np.exp(x) - 1.0, 0.0)
    elif option_type == "put":
        payoff = np.maximum(1.0 - np.exp(x), 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    return np.exp(-alpha * x) * payoff


def double_barrier_analytical(
    S: float,
    K: float,
    B_lower: float,
    B_upper: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str,
    barrier_type: str = "double_knock_out",
    n_terms: int = 100,
) -> float:
    """Approximate analytical price via Fourier sine series on [xL, xU]."""
    if B_lower <= 0 or B_upper <= 0:
        raise ValueError("barriers must be positive")
    if B_lower >= B_upper:
        raise ValueError("B_lower must be < B_upper")
    if not (B_lower < S < B_upper):
        if "knock_out" in barrier_type:
            return 0.0

    if barrier_type not in {"double_knock_out", "double_knock_in"}:
        raise ValueError(
            f"barrier_type must be 'double_knock_out' or 'double_knock_in', got '{barrier_type}'"
        )

    if barrier_type == "double_knock_in":
        dko = double_barrier_analytical(
            S, K, B_lower, B_upper, r, sigma, T, option_type,
            barrier_type="double_knock_out",
            n_terms=n_terms,
        )
        vanilla = _bs_price(S, K, r, sigma, T, option_type)
        return max(0.0, vanilla - dko)

    x0 = np.log(S / K)
    xL = np.log(B_lower / K)
    xU = np.log(B_upper / K)
    L = xU - xL

    alpha = -(r / sigma ** 2 - 0.5)
    beta = -(r / sigma ** 2 - 0.5) ** 2 - 2 * r / sigma ** 2
    tau = 0.5 * sigma ** 2 * T

    # Dense quadrature grid for Fourier coefficients.
    x_quad = np.linspace(xL, xU, 4000)
    u0 = _u0_on_interval(x_quad, K, r, sigma, option_type)

    w_val = 0.0
    for n in range(1, n_terms + 1):
        kn = n * np.pi / L
        basis = np.sin(kn * (x_quad - xL))
        A_n = 2.0 / L * np.trapz(u0 * basis, x_quad)
        w_val += A_n * np.sin(kn * (x0 - xL)) * np.exp(-kn ** 2 * tau)

    price = K * np.exp(alpha * x0 + beta * tau - r * T) * w_val
    return float(max(0.0, price))

