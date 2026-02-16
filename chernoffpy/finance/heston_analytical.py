"""Semi-closed-form Heston option pricing (Little Heston Trap)."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from .transforms import bs_exact_price
from .validation import MarketParams


def heston_call(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> float:
    """European call price in the Heston model."""
    if xi < 1e-6:
        return bs_exact_price(
            MarketParams(S=S, K=K, T=T, r=r, sigma=float(np.sqrt(max(v0, 0.0)))),
            "call",
        )

    p1 = _heston_p(S, K, T, r, v0, kappa, theta, xi, rho, j=1)
    p2 = _heston_p(S, K, T, r, v0, kappa, theta, xi, rho, j=2)
    price = S * p1 - K * np.exp(-r * T) * p2
    return float(max(0.0, price))


def heston_put(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> float:
    """European put via put-call parity."""
    call = heston_call(S, K, T, r, v0, kappa, theta, xi, rho)
    return float(max(0.0, call - S + K * np.exp(-r * T)))


def heston_price(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    option_type: str = "call",
) -> float:
    """European option price in Heston model."""
    if option_type == "call":
        return heston_call(S, K, T, r, v0, kappa, theta, xi, rho)
    if option_type == "put":
        return heston_put(S, K, T, r, v0, kappa, theta, xi, rho)
    raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def _heston_p(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    j: int,
) -> float:
    """Risk-neutral probability term Pj from Heston's formula."""

    def integrand(phi: float) -> float:
        cf = _char_func(phi, S, T, r, v0, kappa, theta, xi, rho, j)
        kernel = np.exp(-1j * phi * np.log(K)) * cf / (1j * phi)
        return float(np.real(kernel))

    # Finite upper bound is standard in practical implementations.
    integral, _ = quad(integrand, 1e-9, 200.0, limit=500, epsabs=1e-8, epsrel=1e-8)
    return float(0.5 + integral / np.pi)


def _char_func(
    phi: float,
    S: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    j: int,
) -> complex:
    """Heston characteristic function using the Little Heston Trap."""
    i = 1j

    if j == 1:
        u = 0.5
        b = kappa - rho * xi
    else:
        u = -0.5
        b = kappa

    a = kappa * theta
    x = np.log(S)

    d = np.sqrt((rho * xi * i * phi - b) ** 2 - xi * xi * (2.0 * u * i * phi - phi * phi))

    # Little Heston Trap form.
    g = (b - rho * xi * i * phi + d) / (b - rho * xi * i * phi - d)
    exp_dT = np.exp(d * T)

    one_minus_g_exp = 1.0 - g * exp_dT
    one_minus_g = 1.0 - g

    # Avoid accidental singularity from floating round-off.
    if abs(one_minus_g_exp) < 1e-14:
        one_minus_g_exp = 1e-14 + 0j
    if abs(one_minus_g) < 1e-14:
        one_minus_g = 1e-14 + 0j

    C = (
        r * i * phi * T
        + (a / (xi * xi))
        * ((b - rho * xi * i * phi + d) * T - 2.0 * np.log(one_minus_g_exp / one_minus_g))
    )
    D = ((b - rho * xi * i * phi + d) / (xi * xi)) * ((1.0 - exp_dT) / one_minus_g_exp)

    return np.exp(C + D * v0 + i * phi * x)

