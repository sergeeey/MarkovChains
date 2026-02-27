# ChernoffPy

**Financial derivatives pricing via Chernoff operator splitting with certified error bounds.**

[![CI](https://github.com/sergeeey/MarkovChains/actions/workflows/ci.yml/badge.svg)](https://github.com/sergeeey/MarkovChains/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/chernoffpy)](https://pypi.org/project/chernoffpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ChernoffPy prices options by applying Chernoff product formulas to transformed pricing PDEs.
The library includes European, barrier, double-barrier, American, Heston, and Bates pricing,
plus calibration tools and **certified error-bound utilities** — a feature unique to this library.

## What makes it different?

Most option pricing libraries give you a number. ChernoffPy gives you a number **and a proof** that
the true price lies within a certified interval. This is made possible by convergence-rate estimates
from [Galkin & Remizov (2025)](https://doi.org/10.1007/s11856-024-2699-7).

## Quick example

```python
from chernoffpy import CrankNicolson
from chernoffpy.finance import CertifiedEuropeanPricer, MarketParams

market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
res = CertifiedEuropeanPricer(CrankNicolson()).price_certified(market, n_steps=50)

print(f"Price: {res.price:.4f}")
print(f"Error bound: ±{res.certified_bound.bound:.6f}")
# Price: 10.4506
# Error bound: ±0.000312
```

## Supported option types

| Type | Pricer | Notes |
|------|--------|-------|
| European call/put | `EuropeanPricer` | Black-Scholes PDE |
| Barrier (single) | `BarrierDSTPricer` | DST, no Gibbs artifacts |
| Barrier (double) | `DoubleBarrierDSTPricer` | DST method |
| American | `AmericanPricer` | Early exercise + dividends |
| Heston | `HestonFastPricer` | Stochastic volatility |
| Bates | `BatesPricer` | Heston + Merton jumps |
| Local volatility | `LocalVolPricer` | Dupire surface |

→ [Full decision guide](user-guide/choosing-pricer.md)

## Installation

```bash
pip install chernoffpy
pip install "chernoffpy[fast]"   # + Numba JIT
pip install "chernoffpy[gpu]"    # + CuPy GPU
pip install "chernoffpy[all]"    # Everything
```

→ [Getting Started](getting-started.md) for a step-by-step walkthrough.
