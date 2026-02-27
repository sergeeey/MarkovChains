# ChernoffPy

**Financial derivatives pricing via Chernoff operator splitting with certified error bounds.**

[![CI](https://github.com/sergeeey/MarkovChains/actions/workflows/ci.yml/badge.svg)](https://github.com/sergeeey/MarkovChains/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/chernoffpy)](https://pypi.org/project/chernoffpy/)
[![Python](https://img.shields.io/pypi/pyversions/chernoffpy)](https://pypi.org/project/chernoffpy/)
[![Coverage](https://img.shields.io/badge/coverage-80%25+-brightgreen)](https://github.com/sergeeey/MarkovChains)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ChernoffPy prices options by applying Chernoff product formulas to transformed pricing PDEs.
The library includes European, barrier, double-barrier, American, Heston, and Bates pricing,
plus calibration tools and certified error-bound utilities.

## Installation

```bash
pip install chernoffpy
pip install "chernoffpy[fast]"
pip install "chernoffpy[gpu]"
pip install "chernoffpy[all]"
```

## Quick Start

### European Option

```python
from chernoffpy import CrankNicolson
from chernoffpy.finance import EuropeanPricer, MarketParams

market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
pricer = EuropeanPricer(CrankNicolson())
result = pricer.price(market, n_steps=50, option_type="call")
print(result.price)
```

### Barrier Option (DST)

```python
from chernoffpy import CrankNicolson
from chernoffpy.finance import BarrierDSTPricer, BarrierParams, MarketParams

market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
params = BarrierParams(barrier=90, barrier_type="down_and_out")
result = BarrierDSTPricer(CrankNicolson()).price(market, params, n_steps=80, option_type="call")
print(result.price)
```

### Certified Error Bound

```python
from chernoffpy import CrankNicolson
from chernoffpy.finance import CertifiedEuropeanPricer, MarketParams

market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
res = CertifiedEuropeanPricer(CrankNicolson()).price_certified(market, n_steps=50, option_type="call")
print(res.price, res.certified_bound.bound)
```

### Heston / Bates

```python
from chernoffpy import CrankNicolson
from chernoffpy.finance import HestonFastPricer, BatesPricer, BatesParams
from chernoffpy.finance.heston_params import HestonParams

heston = HestonParams(S=100, K=100, T=1.0, r=0.05, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
print(HestonFastPricer(CrankNicolson()).price(heston, n_steps=50, option_type="call").price)

bates = BatesParams(S=100, K=100, T=1.0, r=0.05, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
                    lambda_j=0.5, mu_j=-0.1, sigma_j=0.2)
print(BatesPricer(CrankNicolson()).price(bates, n_steps=50, option_type="call").price)
```

## Choosing the Right Pricer

### Which pricer for which option type?

| Option type | Recommended pricer | Notes |
|-------------|-------------------|-------|
| European call/put | `EuropeanPricer` | Fastest; use `CertifiedEuropeanPricer` for guaranteed error bounds |
| Single barrier (down/up, in/out) | `BarrierDSTPricer` | **Recommended** — DST avoids Gibbs artifacts near barrier |
| Single barrier (high performance) | `BarrierPricer` | FFT-based, slightly faster but possible Gibbs at barrier |
| Double barrier | `DoubleBarrierDSTPricer` | DST, no artifacts; `DoubleBarrierPricer` for FFT variant |
| American (no dividends) | `AmericanPricer` | Payoff projection method; use `CrankNicolson()` or `PadeChernoff()` |
| American (discrete dividends) | `AmericanPricer` + `DividendSchedule` | Pass `dividends=` argument to `.price()` |
| Stochastic volatility (Heston) | `HestonFastPricer` | Faster than `HestonPricer`; use `HestonPricer` only for custom grids |
| Jump-diffusion (Heston + jumps) | `BatesPricer` | Heston model with Merton jump component |
| Local volatility | `LocalVolPricer` | Dupire surface; use `flat_vol`, `linear_skew`, or custom `LocalVolParams` |
| With guaranteed error bound | `CertifiedEuropeanPricer` / `CertifiedBarrierDSTPricer` | Returns `certified_bound` alongside price |

### Which Chernoff function (scheme) to use?

| Scheme | Order | Stability | When to use |
|--------|-------|-----------|-------------|
| `BackwardEuler()` | O(1/n) | A-stable ✓ | Debugging, large time steps |
| `CrankNicolson()` | O(1/n²) | A-stable ✓ | **Default choice** — best speed/accuracy trade-off |
| `PadeChernoff(1, 2)` | O(1/n³) | A-stable ✓ | High accuracy, smooth payoffs |
| `PadeChernoff(2, 2)` | O(1/n⁴) | A-stable ✓ | Maximum accuracy |
| `PhysicalG()` | O(1/n) | A-stable ✓ | Explicit scheme, large grids |
| `PhysicalS()` | O(1/n²) | A-stable ✓ | Explicit, 2nd order |
| `PadeChernoff(2, 1)` | O(1/n²) | ⚠ NOT A-stable | Avoid — high-frequency divergence |

**Rule of thumb:** start with `CrankNicolson()`. Switch to `PadeChernoff(1, 2)` if you need tighter certified bounds.

### Quick decision flowchart

```
Vanilla option?       → EuropeanPricer
Needs error bound?    → CertifiedEuropeanPricer
Barrier present?      → BarrierDSTPricer  (single)
                      → DoubleBarrierDSTPricer  (double)
Early exercise?       → AmericanPricer
Discrete dividends?   → AmericanPricer + DividendSchedule
Stoch. volatility?    → HestonFastPricer
Stoch. vol + jumps?   → BatesPricer
Custom vol surface?   → LocalVolPricer
```

## Features

- European pricing and Greeks
- Barrier and double-barrier options (FFT and DST)
- American options with early exercise and discrete dividends
- Local volatility and implied volatility
- Heston stochastic volatility and Bates jump-diffusion
- Calibration helpers (flat/skew/SVI)
- Certified bounds based on Chernoff convergence rates
- Optional acceleration via Numba (`[fast]`)

## Mathematical Basis

ChernoffPy uses the Chernoff product formula
`exp(tL) f = lim_{n->inf} F(t/n)^n f`
with practical schemes (Backward Euler, Crank-Nicolson, Padé).
Certified-bound utilities are motivated by convergence-rate estimates in
Galkin & Remizov (2025, Israel Journal of Mathematics).

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

## References

1. Chernoff, P. (1968), *J. Functional Analysis*.
2. Galkin, O. & Remizov, I. (2025), *Israel Journal of Mathematics*.
3. Butko, Ya. (2020), *Lecture Notes in Mathematics*.

## License

MIT. See `LICENSE`.
