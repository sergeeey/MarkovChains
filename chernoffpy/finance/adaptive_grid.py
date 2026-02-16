"""Adaptive grid utilities for snapping and stretching in finance solvers."""

from __future__ import annotations

import numpy as np

from .validation import GridConfig


def snap_grid_to_barrier(
    barrier: float,
    K: float,
    N: int = 2048,
    L: float = 6.0,
    taper_width: float = 2.0,
) -> GridConfig:
    """Return GridConfig with one snapped log-barrier point."""
    x_barrier = float(np.log(barrier / K))
    L_eff = max(float(L), abs(x_barrier) + 1.0)
    taper_eff = min(taper_width, 0.9 * L_eff)
    return GridConfig(
        N=N,
        L=L_eff,
        taper_width=taper_eff,
        snap_points=(x_barrier,),
    )


def snap_grid_to_double_barrier(
    lower_barrier: float,
    upper_barrier: float,
    K: float,
    N: int = 2048,
    L: float = 6.0,
    taper_width: float = 2.0,
) -> GridConfig:
    """Return GridConfig with both log-barriers snapped."""
    x_lower = float(np.log(lower_barrier / K))
    x_upper = float(np.log(upper_barrier / K))
    low, up = sorted((x_lower, x_upper))
    L_eff = max(float(L), abs(low) + 1.0, abs(up) + 1.0)
    taper_eff = min(taper_width, 0.9 * L_eff)
    return GridConfig(
        N=N,
        L=L_eff,
        taper_width=taper_eff,
        snap_points=(low, up),
    )


def make_stretched_config(
    N: int = 2048,
    L: float = 4.0,
    center: float = 0.0,
    intensity: float = 3.0,
    taper_width: float = 2.0,
) -> GridConfig:
    """Return GridConfig for sinh-stretched mesh around center."""
    taper_eff = min(taper_width, 0.9 * L)
    return GridConfig(
        N=N,
        L=L,
        taper_width=taper_eff,
        center=center,
        stretch=max(0.0, intensity),
    )


def make_snapped_grid(config: GridConfig) -> np.ndarray:
    """Build a uniform grid where one or two points are exact nodes."""
    N = config.N
    L = config.L
    points = tuple(float(x) for x in config.snap_points)

    if len(points) == 0:
        return np.linspace(-L, L, N, endpoint=False)

    dx0 = 2.0 * L / N

    if len(points) == 1:
        x_b = points[0]
        m = int(round((x_b + L) / dx0))
        x_start = x_b - m * dx0
        return x_start + np.arange(N) * dx0

    x_l, x_u = sorted(points)
    span = x_u - x_l
    if span <= 0:
        return np.linspace(-L, L, N, endpoint=False)

    m_between = max(1, int(round(span / dx0)))
    dx = span / m_between

    i_lower = int(round((x_l + L) / dx))
    i_lower = max(0, min(i_lower, N - 1 - m_between))

    x_start = x_l - i_lower * dx
    return x_start + np.arange(N) * dx


def make_stretched_grid(config: GridConfig) -> np.ndarray:
    """Build a sinh-stretched, generally non-uniform grid."""
    N = config.N
    L = config.L
    c = config.center
    s = config.stretch

    if s <= 0:
        return np.linspace(c - L, c + L, N, endpoint=False)

    eta = np.linspace(0.0, 1.0, N)
    d = 1.0 / max(s, 1e-8)
    sinh_half = np.sinh(0.5 / d)
    scale = L / sinh_half
    x = c + scale * np.sinh((eta - 0.5) / d)

    # Preserve monotonicity and exact node count.
    return np.asarray(x, dtype=float)


def compute_grid_quality(x_grid: np.ndarray, points_of_interest: list[float]) -> dict:
    """Return basic mesh diagnostics and snapping errors."""
    x = np.asarray(x_grid, dtype=float)
    dx = np.diff(x)
    dx_min = float(np.min(dx)) if dx.size else 0.0
    dx_max = float(np.max(dx)) if dx.size else 0.0
    ratio = float(dx_max / dx_min) if dx_min > 0 else float("inf")

    snap_errors: dict[float, float] = {}
    for p in points_of_interest:
        idx = int(np.argmin(np.abs(x - p)))
        snap_errors[float(p)] = float(abs(x[idx] - p))

    return {
        "N": int(len(x)),
        "dx_min": dx_min,
        "dx_max": dx_max,
        "dx_ratio": ratio,
        "snap_errors": snap_errors,
    }
