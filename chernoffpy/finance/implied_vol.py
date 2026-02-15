"""Implied volatility solver for Black-Scholes prices."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, newton

from .transforms import bs_exact_price
from .validation import MarketParams


def _intrinsic_value(S: float, K: float, r: float, T: float, option_type: str) -> float:
    if option_type == "call":
        return max(0.0, S - K * np.exp(-r * T))
    if option_type == "put":
        return max(0.0, K * np.exp(-r * T) - S)
    raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def _upper_bound(S: float, K: float, r: float, T: float, option_type: str) -> float:
    if option_type == "call":
        return S
    if option_type == "put":
        return K * np.exp(-r * T)
    raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    method: str = "brentq",
) -> float:
    """Recover implied volatility from option market price."""
    if market_price < 0:
        raise ValueError(f"market_price must be >= 0, got {market_price}")
    if option_type not in {"call", "put"}:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    intrinsic = _intrinsic_value(S, K, r, T, option_type)
    upper = _upper_bound(S, K, r, T, option_type)
    if market_price < intrinsic - 1e-12:
        raise ValueError(
            f"market_price={market_price} is below intrinsic value={intrinsic}"
        )
    if market_price > upper + 1e-12:
        raise ValueError(
            f"market_price={market_price} is above theoretical upper bound={upper}"
        )
    if market_price <= intrinsic + 1e-12:
        return 1e-6

    def objective(sigma: float) -> float:
        market = MarketParams(S=S, K=K, T=T, r=r, sigma=sigma)
        return bs_exact_price(market, option_type) - market_price

    sigma_low = 1e-6
    sigma_high = 5.0

    if method == "brentq":
        return float(brentq(objective, sigma_low, sigma_high, xtol=1e-12))
    if method == "newton":
        def deriv(sigma: float) -> float:
            h = max(1e-5, sigma * 1e-4)
            return (objective(sigma + h) - objective(sigma - h)) / (2 * h)

        return float(newton(func=objective, x0=0.2, fprime=deriv, tol=1e-12, maxiter=100))

    raise ValueError(f"method must be 'brentq' or 'newton', got '{method}'")
