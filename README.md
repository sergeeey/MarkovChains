# ChernoffPy

First open-source Python library for **Chernoff approximations of operator semigroups** with automatic convergence rate analysis. Implements classical and Padé-based Chernoff functions for the heat equation, with tools to verify the Galkin–Remizov convergence rate theorem numerically.

## Installation

```bash
# From source (editable)
pip install -e ".[dev]"

# Minimal (no test dependencies)
pip install -e .
```

## Quick Start

```python
import numpy as np
from chernoffpy import CrankNicolson, HeatSemigroup, convergence_rate, compute_errors

x = np.linspace(-np.pi, np.pi, 128, endpoint=False)
f = np.sin(x)
exact = HeatSemigroup.solve_fourier(f, x, t=1.0)
errors = compute_errors(CrankNicolson(), f, x, exact, t=1.0, n_max=15)
alpha, C = convergence_rate(errors)
print(f"Empirical rate: O(1/n^{alpha:.2f})")  # ≈ O(1/n^2)
```

## Mathematical Background

### Chernoff's Theorem (1968)

Let `C(t)` be an operator-valued function satisfying `C(0) = I` and `‖C(t)‖ ≤ 1`. If the derivative `C'(0)f = Af` exists on a core for the generator `A`, then `C(t/n)^n f → e^{tA} f` strongly as `n → ∞`.

### Galkin–Remizov Convergence Rate (Israel J. Math. 2025)

If `C(t)` matches the first `k` derivatives of `e^{tA}` at `t = 0`, then the convergence rate is `O(1/n^k)` for sufficiently smooth initial data `f ∈ D(A^k)`. This connects the **order of tangency** to the **speed of convergence**.

### Implemented Chernoff Functions

| Function | Order | Method |
|----------|-------|--------|
| `PhysicalG` | 1 | Weighted average with shifts ±2√t |
| `PhysicalS` | 2 | Weighted average with shifts ±√(6t) |
| `BackwardEuler` | 1 | Fourier multiplier 1/(1+tξ²) |
| `CrankNicolson` | 2 | Fourier multiplier (1−tξ²/2)/(1+tξ²/2) |
| `PadeChernoff(m,n)` | m+n | General Padé [m/n] approximant of e^z |

## Running Tests

```bash
pytest -v --tb=short
pytest --cov=chernoffpy --cov-report=term-missing
```

## References

- Dragunova, Nikbakht, Remizov. *Chernoff approximations of Feller semigroups in Riemannian manifolds.* [arXiv:2301.05284](https://arxiv.org/abs/2301.05284) (2023)
- Galkin, Remizov. *Convergence rates for Chernoff-type approximations of operator semigroups.* Israel J. Math. (2025)
- Chernoff, P.R. *Note on product formulas for operator semigroups.* J. Funct. Anal. 2(2), 238–242 (1968)

## License

MIT
