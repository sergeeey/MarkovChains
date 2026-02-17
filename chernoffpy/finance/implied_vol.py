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
    # Tolerance for comparison with intrinsic/upper bounds (avoids
    # false rejection from floating-point rounding in BS formula).
    _PRICE_TOL = 1e-12
    if market_price < intrinsic - _PRICE_TOL:
        raise ValueError(
            f"market_price={market_price} is below intrinsic value={intrinsic}"
        )
    if market_price > upper + _PRICE_TOL:
        raise ValueError(
            f"market_price={market_price} is above theoretical upper bound={upper}"
        )
    # At-the-money or deep-in-the-money: return minimum sigma floor.
    _SIGMA_FLOOR = 1e-6  # Minimum representable vol (avoids division by zero in BS)
    if market_price <= intrinsic + _PRICE_TOL:
        return _SIGMA_FLOOR

    def objective(sigma: float) -> float:
        market = MarketParams(S=S, K=K, T=T, r=r, sigma=sigma)
        return bs_exact_price(market, option_type) - market_price

    # Search bracket: sigma in [1e-6, 5.0] covers all practical implied vols.
    # sigma_low > 0 avoids BS singularity at sigma=0.
    # sigma_high = 5.0 (500% vol) is well above any observed market IV.
    sigma_low = _SIGMA_FLOOR
    sigma_high = 5.0

    # Root-finding tolerance: 1e-12 gives ~10 decimal places of IV precision.
    _ROOT_TOL = 1e-12

    if method == "brentq":
        return float(brentq(objective, sigma_low, sigma_high, xtol=_ROOT_TOL))
    if method == "newton":
        def deriv(sigma: float) -> float:
            # Central difference with adaptive step size for numerical vega.
            h = max(1e-5, sigma * 1e-4)
            return (objective(sigma + h) - objective(sigma - h)) / (2 * h)

        return float(newton(func=objective, x0=0.2, fprime=deriv, tol=_ROOT_TOL, maxiter=100))

    raise ValueError(f"method must be 'brentq' or 'newton', got '{method}'")
