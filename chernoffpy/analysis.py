"""
Convergence rate analysis for Chernoff approximations.

Key tool: given errors d_1, d_2, ..., d_N for n = 1, 2, ..., N,
fit d_n ~ C * n^{-alpha} to determine the empirical convergence order alpha.

By Galkin-Remizov (2025): alpha should equal k (the order of the Chernoff function)
for sufficiently smooth initial conditions f ∈ D(A^k).
"""

import numpy as np
from chernoffpy.functions import ChernoffFunction
from chernoffpy.semigroups import HeatSemigroup


def compute_errors(chernoff: ChernoffFunction,
                   f_values: np.ndarray,
                   x_grid: np.ndarray,
                   exact: np.ndarray,
                   t: float,
                   n_max: int,
                   norm: str = "sup") -> np.ndarray:
    """Compute errors ||C(t/n)^n f - e^{tA} f|| for n = 1, ..., n_max.

    Args:
        chernoff: Chernoff function to test
        f_values: initial condition on x_grid
        x_grid: spatial discretization
        exact: exact solution e^{tA} f on x_grid
        t: time
        n_max: maximum number of steps
        norm: "sup" for sup-norm, "L2" for L²-norm

    Returns:
        Array of errors [d_1, d_2, ..., d_{n_max}]
    """
    errors = np.zeros(n_max)
    for n in range(1, n_max + 1):
        approx = chernoff.compose(f_values, x_grid, t, n)
        diff = approx - exact
        if norm == "sup":
            errors[n - 1] = np.max(np.abs(diff))
        elif norm == "L2":
            dx = x_grid[1] - x_grid[0]
            errors[n - 1] = np.sqrt(np.sum(diff**2) * dx)
        else:
            raise ValueError(f"Unknown norm: {norm}")
    return errors


def convergence_rate(errors: np.ndarray, skip_first: int = 1) -> tuple[float, float]:
    """Estimate convergence order by fitting d_n ~ C * n^{-alpha}.

    Uses linear regression on log(d_n) = log(C) - alpha * log(n).

    Args:
        errors: array of errors for n = 1, 2, ..., N
        skip_first: skip first few values (n=1 may be outlier)

    Returns:
        (alpha, C) where d_n ≈ C * n^{-alpha}
    """
    n_values = np.arange(1, len(errors) + 1)

    # Filter out zeros and very small values
    mask = (errors > 1e-15) & (n_values > skip_first)
    if np.sum(mask) < 2:
        return 0.0, 0.0

    log_n = np.log(n_values[mask])
    log_d = np.log(errors[mask])

    # Linear regression: log_d = log_C - alpha * log_n
    coeffs = np.polyfit(log_n, log_d, 1)
    alpha = -coeffs[0]
    C = np.exp(coeffs[1])

    return alpha, C


def convergence_table(chernoff_functions: list[ChernoffFunction],
                      f_values: np.ndarray,
                      x_grid: np.ndarray,
                      exact: np.ndarray,
                      t: float,
                      n_max: int = 20,
                      norm: str = "sup") -> str:
    """Generate a formatted convergence table for multiple Chernoff functions.

    Returns a markdown-formatted table comparing theoretical and empirical orders.
    """
    lines = []
    lines.append(f"| Chernoff Function | Theoretical k | Empirical α | Ratio α/k | Max Error (n={n_max}) |")
    lines.append("|-------------------|---------------|-------------|-----------|----------------------|")

    for cf in chernoff_functions:
        errors = compute_errors(cf, f_values, x_grid, exact, t, n_max, norm)
        alpha, C = convergence_rate(errors)
        ratio = alpha / cf.order if cf.order > 0 else float('inf')
        final_err = errors[-1]
        lines.append(f"| {cf.name:40s} | {cf.order:13d} | {alpha:11.3f} | {ratio:9.3f} | {final_err:20.6e} |")

    return "\n".join(lines)
