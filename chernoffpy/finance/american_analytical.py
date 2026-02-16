"""Reference American-option approximations used for testing."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _bs_price(S: float, K: float, r: float, sigma: float, T: float, option_type: str) -> float:
    if T <= 0:
        if option_type == "call":
            return max(0.0, S - K)
        return max(0.0, K - S)

    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    if option_type == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    if option_type == "put":
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def american_binomial(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "put",
    n_steps: int = 10000,
    american: bool = True,
) -> float:
    """CRR binomial pricer for European/American options."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if option_type not in {"call", "put"}:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    if T <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)

    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    p = float(np.clip(p, 0.0, 1.0))
    disc = np.exp(-r * dt)

    j = np.arange(n_steps, -1, -1)
    S_T = S * (u ** j) * (d ** (n_steps - j))

    if option_type == "call":
        V = np.maximum(S_T - K, 0.0)
    else:
        V = np.maximum(K - S_T, 0.0)

    for i in range(n_steps - 1, -1, -1):
        V = disc * (p * V[:-1] + (1.0 - p) * V[1:])

        if american:
            j_i = np.arange(i, -1, -1)
            S_i = S * (u ** j_i) * (d ** (i - j_i))
            if option_type == "call":
                exercise = np.maximum(S_i - K, 0.0)
            else:
                exercise = np.maximum(K - S_i, 0.0)
            V = np.maximum(V, exercise)

    return float(V[0])


def american_baw(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "put",
) -> float:
    """BAW-style fast approximation.

    For calls without dividends, early exercise is suboptimal, so price=BS.
    For puts, use a high-quality Richardson extrapolation of CRR prices.
    """
    if option_type not in {"call", "put"}:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    if option_type == "call":
        return _bs_price(S, K, r, sigma, T, "call")

    p1 = american_binomial(S, K, r, sigma, T, option_type="put", n_steps=800, american=True)
    p2 = american_binomial(S, K, r, sigma, T, option_type="put", n_steps=1600, american=True)
    richardson = 2.0 * p2 - p1
    return float(max(richardson, 0.0))
