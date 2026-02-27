# Getting Started

## Installation

```bash
pip install chernoffpy
```

For optional acceleration:

```bash
pip install "chernoffpy[fast]"   # Numba JIT (recommended for production)
pip install "chernoffpy[gpu]"    # CuPy GPU support (requires CUDA 12)
pip install "chernoffpy[all]"    # Everything above
```

**Requirements:** Python ≥ 3.11, NumPy ≥ 1.24, SciPy ≥ 1.10.

---

## Your first option price

### European call

```python
from chernoffpy import CrankNicolson
from chernoffpy.finance import EuropeanPricer, MarketParams

market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
pricer = EuropeanPricer(CrankNicolson())
result = pricer.price(market, n_steps=50, option_type="call")

print(result.price)        # ~10.4506
print(result.certificate)  # validation metadata
```

### Barrier option

```python
from chernoffpy import CrankNicolson
from chernoffpy.finance import BarrierDSTPricer, BarrierParams, MarketParams

market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
params = BarrierParams(barrier=90, barrier_type="down_and_out")
result = BarrierDSTPricer(CrankNicolson()).price(market, params, n_steps=80)

print(result.price)   # cheaper than European because of knock-out risk
```

### With a certified error bound

```python
from chernoffpy import CrankNicolson
from chernoffpy.finance import CertifiedEuropeanPricer, MarketParams

market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
res = CertifiedEuropeanPricer(CrankNicolson()).price_certified(market, n_steps=50)

print(f"Price:      {res.price:.6f}")
print(f"Bound:    ±{res.certified_bound.bound:.6f}")
print(f"True price is in [{res.price - res.certified_bound.bound:.6f}, "
      f"{res.price + res.certified_bound.bound:.6f}]")
```

### American put

```python
from chernoffpy import CrankNicolson
from chernoffpy.finance import AmericanPricer, MarketParams

market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
result = AmericanPricer(CrankNicolson()).price(market, n_steps=100, option_type="put")
print(result.price)   # > European put because of early exercise premium
```

### Heston stochastic volatility

```python
from chernoffpy import CrankNicolson
from chernoffpy.finance import HestonFastPricer
from chernoffpy.finance.heston_params import HestonParams

heston = HestonParams(
    S=100, K=100, T=1.0, r=0.05,
    v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7
)
result = HestonFastPricer(CrankNicolson()).price(heston, n_steps=50)
print(result.price)
```

---

## Choosing `n_steps`

`n_steps` controls the number of Chernoff composition steps. Higher = more accurate, slower.

| Use case | Recommended `n_steps` |
|----------|-----------------------|
| Quick estimate | 20–30 |
| Standard pricing | 50–80 |
| High accuracy / certified bounds | 100–200 |

For a rigorous approach, use `n_steps_for_tolerance()`:

```python
from chernoffpy import CrankNicolson
from chernoffpy.certified import n_steps_for_tolerance

n = n_steps_for_tolerance(scheme=CrankNicolson(), tolerance=1e-4, T=1.0)
print(n)   # minimum steps to guarantee error < 1e-4
```

---

## Next steps

- [Choosing the right pricer](user-guide/choosing-pricer.md) — full decision guide
- [Certified error bounds](user-guide/certified-bounds.md) — how the mathematical guarantees work
- [API Reference](api/functions.md) — full documentation of all classes and functions
