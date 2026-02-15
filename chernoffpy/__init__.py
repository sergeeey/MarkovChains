"""
ChernoffPy — first open-source library for Chernoff approximations of operator semigroups.

Based on:
- Chernoff's theorem (1968)
- Galkin-Remizov convergence rate theory (Israel J. Math. 2025)
- Numerical code from arXiv:2301.05284 (Dragunova, Nikbakht, Remizov 2023)

Core concepts:
- ChernoffFunction: operator-valued function C(t) satisfying C(0) = I, ||C(t)|| <= 1
- C(t/n)^n -> e^{tA} strongly (Chernoff's theorem)
- If C matches k derivatives of e^{tA} at t=0, convergence rate is O(1/n^k) (Galkin-Remizov)
"""

from chernoffpy.functions import (
    ChernoffFunction,
    BackwardEuler,
    CrankNicolson,
    PadeChernoff,
    PhysicalG,
    PhysicalS,
)
from chernoffpy.semigroups import HeatSemigroup
from chernoffpy.analysis import compute_errors, convergence_rate, convergence_table

__version__ = "0.1.0"
__author__ = "Built on the theory of I.D. Remizov, O.E. Galkin et al."
