"""Analytical barrier option formulas (Reiner-Rubinstein, q=0).

Implements 4 knock-out cases explicitly and derives knock-in prices via
in-out parity: V_in = V_vanilla - V_out.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .validation import BarrierParams, MarketParams


def _bs_vanilla_price(market: MarketParams, option_type: str) -> float:
    sqrt_t = np.sqrt(market.T)
    d1 = (
        np.log(market.S / market.K)
        + (market.r + 0.5 * market.sigma ** 2) * market.T
    ) / (market.sigma * sqrt_t)
    d2 = d1 - market.sigma * sqrt_t

    if option_type == "call":
        return float(
            market.S * norm.cdf(d1)
            - market.K * np.exp(-market.r * market.T) * norm.cdf(d2)
        )
    if option_type == "put":
        return float(
            market.K * np.exp(-market.r * market.T) * norm.cdf(-d2)
            - market.S * norm.cdf(-d1)
        )
    raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def _rr_blocks(market: MarketParams, barrier: float):
    s = market.S
    k = market.K
    t = market.T
    r = market.r
    sigma = market.sigma

    sig_sqrt_t = sigma * np.sqrt(t)
    lam = (r + 0.5 * sigma ** 2) / sigma ** 2

    x1 = np.log(s / k) / sig_sqrt_t + lam * sig_sqrt_t
    x2 = np.log(s / barrier) / sig_sqrt_t + lam * sig_sqrt_t
    y1 = np.log((barrier ** 2) / (s * k)) / sig_sqrt_t + lam * sig_sqrt_t
    y2 = np.log(barrier / s) / sig_sqrt_t + lam * sig_sqrt_t

    def A(phi: int) -> float:
        return float(
            phi * s * norm.cdf(phi * x1)
            - phi * k * np.exp(-r * t) * norm.cdf(phi * x1 - phi * sig_sqrt_t)
        )

    def B(phi: int) -> float:
        return float(
            phi * s * norm.cdf(phi * x2)
            - phi * k * np.exp(-r * t) * norm.cdf(phi * x2 - phi * sig_sqrt_t)
        )

    def C(phi: int, eta: int) -> float:
        return float(
            phi
            * s
            * (barrier / s) ** (2 * lam)
            * norm.cdf(eta * y1)
            - phi
            * k
            * np.exp(-r * t)
            * (barrier / s) ** (2 * lam - 2)
            * norm.cdf(eta * y1 - eta * sig_sqrt_t)
        )

    def D(phi: int, eta: int) -> float:
        return float(
            phi
            * s
            * (barrier / s) ** (2 * lam)
            * norm.cdf(eta * y2)
            - phi
            * k
            * np.exp(-r * t)
            * (barrier / s) ** (2 * lam - 2)
            * norm.cdf(eta * y2 - eta * sig_sqrt_t)
        )

    return A, B, C, D


def barrier_analytical(
    market: MarketParams,
    barrier_params: BarrierParams,
    option_type: str = "call",
) -> float:
    """Analytical barrier price under Black-Scholes without dividends."""
    bt = barrier_params.barrier_type
    b = barrier_params.barrier

    if option_type not in {"call", "put"}:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    if "down" in bt and market.S <= b and "out" in bt:
        return 0.0
    if "up" in bt and market.S >= b and "out" in bt:
        return 0.0

    if "out" in bt:
        A, B, C, D = _rr_blocks(market, b)

        if option_type == "call" and bt == "down_and_out":
            if market.K > b:
                return max(0.0, A(1) - C(1, 1))
            return max(0.0, B(1) - D(1, 1))

        if option_type == "put" and bt == "down_and_out":
            if market.K > b:
                return max(0.0, A(-1) - B(-1) + C(-1, 1) - D(-1, 1))
            return 0.0

        if option_type == "call" and bt == "up_and_out":
            if market.K > b:
                return 0.0
            return max(0.0, A(1) - B(1) + C(1, -1) - D(1, -1))

        if option_type == "put" and bt == "up_and_out":
            if market.K > b:
                return max(0.0, B(-1) - D(-1, -1))
            return max(0.0, A(-1) - C(-1, -1))

        raise ValueError(f"Unsupported barrier_type '{bt}'")

    out_type = bt.replace("_in", "_out")
    out_params = BarrierParams(
        barrier=b,
        barrier_type=out_type,
        rebate=barrier_params.rebate,
    )
    vanilla = _bs_vanilla_price(market, option_type)
    out_price = barrier_analytical(market, out_params, option_type)
    return max(0.0, vanilla - out_price)
