"""Semi-closed-form Bates option pricing via characteristic functions.

References:
- Bates (1996), RFS
- Lewis (2001)
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from .heston_analytical import heston_price
from .transforms import bs_exact_price
from .validation import MarketParams


def _heston_cf(
    u: complex,
    S: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> complex:
    """Characteristic function of log(S_T) in Heston model.

    Uses the 'Little Heston Trap' formulation (Albrecher et al.) for
    numerical stability when evaluating at complex arguments.
    """
    i = 1j
    x0 = np.log(S)

    if xi < 1e-10:
        # Degenerate variance -> GBM with vol sqrt(v0)
        sigma = np.sqrt(max(v0, 0.0))
        return np.exp(i * u * (x0 + (r - 0.5 * sigma * sigma) * T) - 0.5 * sigma * sigma * u * u * T)

    a = kappa - rho * xi * i * u

    d = np.sqrt(a * a + xi * xi * (i * u + u * u))
    # Enforce Re(d) > 0 for Little Trap stability.
    if np.real(d) < 0:
        d = -d

    g = (a + d) / (a - d)

    exp_dT = np.exp(d * T)
    one_minus_gexp = 1.0 - g * exp_dT
    one_minus_g = 1.0 - g

    if abs(one_minus_gexp) < 1e-14:
        one_minus_gexp = 1e-14 + 0j
    if abs(one_minus_g) < 1e-14:
        one_minus_g = 1e-14 + 0j

    C = (
        i * u * r * T
        + (kappa * theta / (xi * xi))
        * ((a + d) * T - 2.0 * np.log(one_minus_gexp / one_minus_g))
    )
    D = ((a + d) / (xi * xi)) * ((1.0 - exp_dT) / one_minus_gexp)

    return np.exp(C + D * v0 + i * u * x0)


def bates_cf(
    u: complex,
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
) -> complex:
    """Characteristic function of log(S_T) in Bates model."""
    h_cf = _heston_cf(u, S, T, r, v0, kappa, theta, xi, rho)
    kbar = np.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0
    jump_cf = np.exp(
        lambda_j
        * T
        * (np.exp(1j * u * mu_j - 0.5 * sigma_j * sigma_j * u * u) - 1.0 - 1j * u * kbar)
    )
    return h_cf * jump_cf


def bates_price(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
    option_type: str = "call",
) -> float:
    """Bates price via Lewis integral representation."""
    if option_type not in {"call", "put"}:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    if lambda_j < 1e-12:
        return heston_price(S, K, T, r, v0, kappa, theta, xi, rho, option_type)

    if xi < 1e-10 and lambda_j < 1e-10:
        return bs_exact_price(MarketParams(S=S, K=K, T=T, r=r, sigma=np.sqrt(max(v0, 0.0))), option_type)

    x0 = np.log(S / K)

    def integrand(u: float) -> float:
        z = u - 0.5j
        phi = bates_cf(
            z,
            S,
            K,
            T,
            r,
            v0,
            kappa,
            theta,
            xi,
            rho,
            lambda_j,
            mu_j,
            sigma_j,
        )
        # bates_cf returns CF of log(S_T); Lewis formula needs CF of
        # log-return log(S_T/S), so divide out the spot factor.
        phi /= np.exp(1j * z * np.log(S))
        return float(np.real(np.exp(1j * u * x0) * phi / (u * u + 0.25)))

    integral, _ = quad(integrand, 0.0, 200.0, limit=500, epsabs=1e-8, epsrel=1e-8)
    call = S - np.sqrt(S * K) * np.exp(-r * T) * integral / np.pi

    if option_type == "call":
        return float(max(0.0, call))
    return float(max(0.0, call - S + K * np.exp(-r * T)))
