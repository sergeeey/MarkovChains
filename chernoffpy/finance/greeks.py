"""Greeks computation for European options.

Delta, Gamma: extracted from the spatial solution u(x, tau) without repricing.
Vega, Theta, Rho: central finite differences with 2 repricings each.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from .validation import GreeksResult, MarketParams
from .transforms import compute_transform_params, _MAX_EXP

if TYPE_CHECKING:
    from .european import EuropeanPricer


def compute_greeks(
    pricer: EuropeanPricer,
    market: MarketParams,
    n_steps: int = 50,
    option_type: str = "call",
    h_sigma: float = 0.001,
    h_T: float = 1 / 365,
    h_r: float = 0.0001,
) -> GreeksResult:
    """Compute option Greeks.

    Parameters:
        pricer: EuropeanPricer instance
        market: Market parameters
        n_steps: Chernoff composition steps
        option_type: "call" or "put"
        h_sigma: bump size for vega
        h_T: bump size for theta (default 1 day)
        h_r: bump size for rho
    """
    sol = pricer._solve(market, n_steps, option_type)

    # --- Delta and Gamma from spatial derivatives (no repricing) ---
    x_grid = sol["x_grid"]
    u_final = sol["u_final"]
    alpha = sol["alpha"]
    beta = sol["beta"]
    t_eff = sol["t_eff"]
    dx = x_grid[1] - x_grid[0]
    x0 = np.log(market.S / market.K)

    # V(x) = K * exp(alpha*x + beta*t_eff) * u(x)
    # Clip exponent to prevent overflow for low volatility (large |alpha|)
    exp_arg = np.clip(alpha * x_grid + beta * t_eff, -_MAX_EXP, _MAX_EXP)
    V_grid = market.K * np.exp(exp_arg) * u_final

    # dV/dx and d²V/dx² via central differences
    dV_dx = np.gradient(V_grid, dx)
    d2V_dx2 = np.gradient(dV_dx, dx)

    dV_dx_at_x0 = float(np.interp(x0, x_grid, dV_dx))
    d2V_dx2_at_x0 = float(np.interp(x0, x_grid, d2V_dx2))

    # Delta = dV/dS = (1/S) * dV/dx   (since x = ln(S/K), dx/dS = 1/S)
    delta = dV_dx_at_x0 / market.S

    # Gamma = d²V/dS² = (d²V/dx² - dV/dx) / S²
    gamma = (d2V_dx2_at_x0 - dV_dx_at_x0) / market.S ** 2

    # --- Vega: central FD by sigma (2 repricings) ---
    market_up = dataclasses.replace(market, sigma=market.sigma + h_sigma)
    market_dn = dataclasses.replace(market, sigma=market.sigma - h_sigma)
    price_up = pricer._solve(market_up, n_steps, option_type)["price"]
    price_dn = pricer._solve(market_dn, n_steps, option_type)["price"]
    vega = (price_up - price_dn) / (2 * h_sigma)

    # --- Theta: central FD by T, reported as dV/dt = -dV/dT ---
    # Adaptive step: ensure h_T doesn't exceed T/4 for short-expiry options
    h_T = min(h_T, market.T / 4)
    T_up = market.T + h_T
    T_dn = max(1e-6, market.T - h_T)
    market_up = dataclasses.replace(market, T=T_up)
    market_dn = dataclasses.replace(market, T=T_dn)
    price_up = pricer._solve(market_up, n_steps, option_type)["price"]
    price_dn = pricer._solve(market_dn, n_steps, option_type)["price"]
    actual_h_T = T_up - T_dn
    theta = -(price_up - price_dn) / actual_h_T

    # --- Rho: central FD by r ---
    r_up = market.r + h_r
    r_dn = max(0.0, market.r - h_r)
    market_up = dataclasses.replace(market, r=r_up)
    market_dn = dataclasses.replace(market, r=r_dn)
    price_up = pricer._solve(market_up, n_steps, option_type)["price"]
    price_dn = pricer._solve(market_dn, n_steps, option_type)["price"]
    actual_h_r = r_up - r_dn
    rho = (price_up - price_dn) / actual_h_r

    return GreeksResult(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
    )
