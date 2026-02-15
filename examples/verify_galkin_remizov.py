#!/usr/bin/env python3
"""
Verify the Galkin-Remizov theorem (Israel J. Math. 2025) numerically.

Tests: For the heat equation du/dt = d²u/dx² with various initial conditions,
the Chernoff approximation C(t/n)^n converges with rate O(1/n^k)
where k = order of the Chernoff function.

Chernoff functions tested:
- PhysicalG (k=1): O(1/n)   — from arXiv:2301.05284
- PhysicalS (k=2): O(1/n²)  — from arXiv:2301.05284
- BackwardEuler (k=1): O(1/n)   — Padé [0/1]
- CrankNicolson (k=2): O(1/n²)  — Padé [1/1]
- Padé [2/1] (k=3): O(1/n³)
- Padé [1/2] (k=3): O(1/n³)

Initial conditions with varying smoothness:
- sin(x)         — C∞, in all H^k spaces
- |sin(x)|       — C⁰, in H^{1/2-ε} only → slow convergence for high-order methods
- |sin(x)|^{5/2} — C², in H^{5/2-ε} → intermediate
- exp(-x²)       — C∞, Schwartz class

This script reproduces and extends the results of arXiv:2301.05284
using the ChernoffPy library.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from chernoffpy.functions import (
    PhysicalG, PhysicalS, BackwardEuler, CrankNicolson, PadeChernoff
)
from chernoffpy.semigroups import HeatSemigroup
from chernoffpy.analysis import compute_errors, convergence_rate


def main():
    # ===== SETUP =====
    N_points = 2000          # spatial grid resolution
    x = np.linspace(-np.pi, np.pi, N_points, endpoint=False)
    dx = x[1] - x[0]
    t = 0.5                  # time
    n_max = 30               # max Chernoff steps (paper used n=10, we use more)

    # ===== INITIAL CONDITIONS =====
    initial_conditions = {
        "sin(x)": np.sin(x),
        "|sin(x)|": np.abs(np.sin(x)),
        "|sin(x)|^{5/2}": np.abs(np.sin(x)) ** 2.5,
        "exp(-x²)": np.exp(-x**2),
    }

    # ===== CHERNOFF FUNCTIONS =====
    # Only A-stable methods (m <= n for Padé [m/n]) — unstable ones blow up!
    chernoff_functions = [
        PhysicalG(),
        PhysicalS(),
        BackwardEuler(),
        CrankNicolson(),
        PadeChernoff(1, 2),   # Padé [1/2], order 3, A-stable
        PadeChernoff(2, 2),   # Padé [2/2], order 4, A-stable
    ]

    # ===== MAIN COMPUTATION =====
    print("=" * 80)
    print("VERIFICATION OF GALKIN-REMIZOV THEOREM")
    print(f"Heat equation du/dt = d²u/dx², t = {t}, grid = {N_points} points on [-π, π]")
    print("=" * 80)

    for ic_name, f_values in initial_conditions.items():
        # Exact solution via Fourier
        exact = HeatSemigroup.solve_fourier(f_values, x, t)

        print(f"\n{'─' * 70}")
        print(f"Initial condition: u₀(x) = {ic_name}")
        print(f"{'─' * 70}")
        print(f"{'Chernoff Function':45s} | k (theor.) | α (empir.) | α/k  | err(n={n_max})")
        print(f"{'-'*45}-+-{'-'*10}-+-{'-'*10}-+-{'-'*4}-+-{'-'*12}")

        for cf in chernoff_functions:
            errors = compute_errors(cf, f_values, x, exact, t, n_max)
            alpha, C = convergence_rate(errors, skip_first=2)
            ratio = alpha / cf.order if cf.order > 0 else 0
            print(f"{cf.name:45s} | {cf.order:10d} | {alpha:10.3f} | {ratio:4.2f} | {errors[-1]:.6e}")

    # ===== CONVERGENCE PLOT =====
    print("\n\nGenerating convergence plot...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    n_values = np.arange(1, n_max + 1)

    for ax, (ic_name, f_values) in zip(axes.flat, initial_conditions.items()):
        exact = HeatSemigroup.solve_fourier(f_values, x, t)

        for cf in chernoff_functions:
            errors = compute_errors(cf, f_values, x, exact, t, n_max)
            mask = errors > 1e-15
            ax.loglog(n_values[mask], errors[mask], 'o-', markersize=3, label=cf.name)

        # Reference slopes
        n_ref = np.array([3, n_max])
        for k, style in [(1, ':'), (2, '--'), (3, '-.')]:
            ax.loglog(n_ref, 0.5 * n_ref.astype(float)**(-k), style, color='gray',
                      alpha=0.5, label=f"O(1/n^{k})" if ax == axes[0, 0] else None)

        ax.set_xlabel("n (number of Chernoff steps)")
        ax.set_ylabel("||C(t/n)^n f - e^{tA} f||_∞")
        ax.set_title(f"u₀(x) = {ic_name}")
        ax.grid(True, alpha=0.3)

    axes[0, 0].legend(fontsize=7, loc='lower left')
    fig.suptitle("Galkin-Remizov Verification: Convergence rates of Chernoff approximations",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "..", "convergence_rates.png"), dpi=150)
    # plt.show()  # non-interactive mode
    print("Plot saved to convergence_rates.png")

    # ===== PADÉ COMPARISON =====
    print("\n" + "=" * 80)
    print("PADÉ APPROXIMANT COMPARISON (smooth initial data: sin(x))")
    print("=" * 80)
    f_smooth = np.sin(x)
    exact_smooth = HeatSemigroup.solve_fourier(f_smooth, x, t)

    print(f"\n{'Padé [m/n]':15s} | k=m+n | α (empir.) | α/k  | A-stable? | err(n={n_max})")
    print(f"{'-'*15}-+-{'-'*5}-+-{'-'*10}-+-{'-'*4}-+-{'-'*9}-+-{'-'*12}")

    for m, n_pade in [(0, 1), (1, 1), (2, 1), (1, 2), (3, 1), (2, 2), (1, 3)]:
        cf = PadeChernoff(m, n_pade)
        errors = compute_errors(cf, f_smooth, x, exact_smooth, t, n_max)
        alpha, C = convergence_rate(errors, skip_first=2)
        ratio = alpha / cf.order if cf.order > 0 else 0
        a_stable = "Yes" if m <= n_pade else "No"
        print(f"[{m}/{n_pade}]{' '*(10-len(f'[{m}/{n_pade}]'))} | {cf.order:5d} | {alpha:10.3f} | {ratio:4.2f} | {a_stable:9s} | {errors[-1]:.6e}")


if __name__ == "__main__":
    main()
