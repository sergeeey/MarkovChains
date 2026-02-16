"""Discrete dividend projection helpers for heat-equation solvers."""

from __future__ import annotations

import numpy as np

from .validation import DividendSchedule


def apply_discrete_dividend(
    u: np.ndarray,
    x_grid: np.ndarray,
    amount: float,
    strike: float,
    proportional: bool = False,
    alpha: float = 0.0,
) -> np.ndarray:
    """Apply one discrete dividend jump by interpolation in log-space.

    When working in heat variables, the transform V = K*exp(αx+βτ)*u
    requires a correction factor exp(α(x'-x)) where x' is the post-dividend
    log-moneyness.  Pass the Wilmott α parameter to enable this correction.
    """
    if amount <= 0.0:
        return np.copy(u)

    if proportional:
        if amount >= 1.0:
            raise ValueError("Proportional dividend amount must be < 1.0")
        shift = np.log(1.0 - amount)
        x_shifted = x_grid + shift
        u_interp = np.interp(x_shifted, x_grid, u, left=0.0, right=u[-1])
        if abs(alpha) > 1e-14:
            u_interp *= np.exp(alpha * shift)
        return u_interp

    s_grid = strike * np.exp(x_grid)
    s_after = np.maximum(s_grid - amount, 1e-10)
    x_after = np.log(s_after / strike)
    u_interp = np.interp(x_after, x_grid, u, left=0.0, right=u[-1])
    if abs(alpha) > 1e-14:
        u_interp *= np.exp(alpha * (x_after - x_grid))
    return u_interp


def find_dividend_steps(
    schedule: DividendSchedule,
    maturity: float,
    n_steps: int,
) -> dict[int, list[tuple[float, bool]]]:
    """Map dividend times to backward-time solver steps.

    Solver loop evolves from expiry backward to valuation time.
    For a dividend at time t_div in [0, T], step index is mapped by
    step ~ round((T - t_div)/dt), clamped to [0, n_steps-1].
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    dt = maturity / n_steps
    mapping: dict[int, list[tuple[float, bool]]] = {}

    for t_div, amount in zip(schedule.times, schedule.amounts):
        if t_div >= maturity:
            continue
        step = int(round((maturity - t_div) / dt))
        step = max(0, min(step, n_steps - 1))
        mapping.setdefault(step, []).append((float(amount), schedule.proportional))

    return mapping

