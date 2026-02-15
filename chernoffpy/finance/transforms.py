"""Black-Scholes <-> Heat equation transforms via Wilmott substitution.

The substitution:
    x = ln(S/K),  tau = sigma^2 * T / 2
    k = 2r / sigma^2
    alpha = -(k-1)/2,  beta = -(k+1)^2 / 4
    V(S, t) = K * exp(alpha*x + beta*tau) * u(x, tau)

transforms the Black-Scholes PDE into the heat equation u_tau = u_xx.
"""

import numpy as np
from scipy.stats import norm

from .validation import MarketParams, GridConfig

# Maximum safe exponent to prevent overflow (exp(709) ~ float64 max)
_MAX_EXP = 700.0


def compute_transform_params(market: MarketParams) -> tuple[float, float, float, float]:
    """Compute Wilmott transform parameters from market data.

    Returns:
        (k, alpha, beta, t_eff) where t_eff = sigma^2 * T / 2 is the
        effective time for the heat equation.
    """
    k = 2 * market.r / market.sigma ** 2
    alpha = -(k - 1) / 2
    beta = -(k + 1) ** 2 / 4
    t_eff = market.sigma ** 2 * market.T / 2
    return k, alpha, beta, t_eff


def make_grid(config: GridConfig) -> np.ndarray:
    """Create uniform spatial grid on [-L, L) with N points (endpoint excluded for FFT)."""
    return np.linspace(-config.L, config.L, config.N, endpoint=False)


def make_taper(x_grid: np.ndarray, config: GridConfig) -> np.ndarray:
    """Create cosine taper that smoothly zeros the function near domain boundaries.

    taper(x) = 1                                        if |x| <= L - taper_width
    taper(x) = 0.5*(1 + cos(pi*(|x| - edge)/width))    if edge < |x| < L
    """
    taper = np.ones_like(x_grid)
    edge = config.L - config.taper_width
    abs_x = np.abs(x_grid)
    mask = abs_x > edge
    t = (abs_x[mask] - edge) / config.taper_width
    taper[mask] = 0.5 * (1 + np.cos(np.pi * t))
    return taper


def bs_to_heat_initial(
    x_grid: np.ndarray,
    market: MarketParams,
    config: GridConfig,
    option_type: str = "call",
) -> np.ndarray:
    """Convert BS payoff to heat equation initial condition u(x, 0) with cosine taper.

    u(x, 0) = exp(-alpha * x) * payoff_normalized(x) * taper(x)

    where payoff_normalized = max(e^x - 1, 0) for call, max(1 - e^x, 0) for put.
    """
    k, alpha, beta, t_eff = compute_transform_params(market)

    if option_type == "call":
        payoff = np.maximum(np.exp(x_grid) - 1, 0)
    elif option_type == "put":
        payoff = np.maximum(1 - np.exp(x_grid), 0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    # u(x, 0) = exp(-alpha*x) * payoff = exp((k-1)x/2) * payoff
    # Clip exponent to prevent overflow for low volatility (large |alpha|)
    raw_exponent = -alpha * x_grid
    exponent = np.clip(raw_exponent, -_MAX_EXP, _MAX_EXP)
    u0 = np.exp(exponent) * payoff

    # Overflow taper: smoothly zero IC where exp(-alpha*x) would be too large.
    # Without this, huge IC values contaminate FFT for low volatility (large |alpha|).
    # exp(10) ~ 2.2e4 is safe for FFT; fully tapered at 2x onset.
    _OVERFLOW_ONSET = 10.0
    _OVERFLOW_FULL = 2.0 * _OVERFLOW_ONSET  # fully tapered at 2x onset
    abs_exp = np.abs(raw_exponent)
    overflow_mask = abs_exp > _OVERFLOW_ONSET
    if np.any(overflow_mask):
        overflow_taper = np.ones_like(x_grid)
        t = np.minimum(
            (abs_exp[overflow_mask] - _OVERFLOW_ONSET)
            / (_OVERFLOW_FULL - _OVERFLOW_ONSET),
            1.0,
        )
        overflow_taper[overflow_mask] = 0.5 * (1 + np.cos(np.pi * t))
        u0 = np.where(overflow_taper > 0, u0 * overflow_taper, 0.0)

    taper = make_taper(x_grid, config)
    return np.where(taper > 0, u0 * taper, 0.0)


def heat_to_bs_price(
    u: np.ndarray, x_grid: np.ndarray, market: MarketParams
) -> np.ndarray:
    """Convert heat solution u(x, t_eff) back to BS prices V(S) on the full grid.

    V(S_i) = K * exp(alpha * x_i + beta * t_eff) * u(x_i, t_eff)
    """
    k, alpha, beta, t_eff = compute_transform_params(market)
    exponent = np.clip(alpha * x_grid + beta * t_eff, -_MAX_EXP, _MAX_EXP)
    return market.K * np.exp(exponent) * u


def extract_price_at_spot(
    u: np.ndarray, x_grid: np.ndarray, market: MarketParams
) -> float:
    """Extract option price at the spot S0 by interpolating the heat solution.

    x0 = ln(S0/K), then V(S0) = K * exp(alpha*x0 + beta*t_eff) * u(x0, t_eff).
    """
    k, alpha, beta, t_eff = compute_transform_params(market)
    x0 = np.log(market.S / market.K)
    u_at_x0 = float(np.interp(x0, x_grid, u))
    exponent = np.clip(alpha * x0 + beta * t_eff, -_MAX_EXP, _MAX_EXP)
    return market.K * np.exp(exponent) * u_at_x0


def bs_exact_price(market: MarketParams, option_type: str = "call") -> float:
    """Exact Black-Scholes formula.

    Call = S*N(d1) - K*exp(-rT)*N(d2)
    Put  = K*exp(-rT)*N(-d2) - S*N(-d1)
    """
    sqrt_T = np.sqrt(market.T)
    d1 = (
        np.log(market.S / market.K) + (market.r + market.sigma ** 2 / 2) * market.T
    ) / (market.sigma * sqrt_T)
    d2 = d1 - market.sigma * sqrt_T

    if option_type == "call":
        return float(
            market.S * norm.cdf(d1)
            - market.K * np.exp(-market.r * market.T) * norm.cdf(d2)
        )
    elif option_type == "put":
        return float(
            market.K * np.exp(-market.r * market.T) * norm.cdf(-d2)
            - market.S * norm.cdf(-d1)
        )
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
