# ChernoffPy

[![Tests](https://github.com/sergeeey/MarkovChains/actions/workflows/tests.yml/badge.svg)](https://github.com/sergeeey/MarkovChains/actions/workflows/tests.yml)

Option pricing through the heat equation with mathematically tracked numerical error.

ChernoffPy prices European, single-barrier, double-barrier, American, and local-volatility options by solving transformed Black-Scholes PDEs with Chernoff semigroup approximations.  
Each pricing run includes an accuracy certificate with error decomposition.

<p align="center">
  <img src="examples/convergence.png" alt="Convergence: Crank-Nicolson vs Backward Euler" width="600">
</p>

## Why ChernoffPy?

| Feature | ChernoffPy | QuantLib | Typical BS libs |
|---|---|---|---|
| Certified error decomposition | ✅ | ❌ | ❌ |
| Chernoff semigroup method | ✅ | ❌ | ❌ |
| Single + double barriers | ✅ | ✅ | ❌ |
| American options | ✅ | ✅ | ❌ |
| Local volatility `sigma(S,t)` | ✅ | ✅ | ❌ |
| Surface calibration (flat/skew/SVI) | ✅ | ✅ | ❌ |
| Core dependencies | `numpy`, `scipy` | C++ + SWIG | Varies |

## Quick Start

### European option

```python
from chernoffpy import CrankNicolson
from chernoffpy.finance import EuropeanPricer, MarketParams, bs_exact_price

market = MarketParams(S=100, K=105, T=1.0, r=0.05, sigma=0.20)
pricer = EuropeanPricer(CrankNicolson())

result = pricer.price(market, n_steps=50, option_type="call")
bs = bs_exact_price(market, "call")

print(f"Chernoff: {result.price:.4f}")
print(f"BS exact: {bs:.4f}")
print(f"Error:    {abs(result.price - bs)/bs:.4%}")
```

### Certified accuracy

```python
cert = result.certificate
print(f"Chernoff error: {cert.chernoff_error:.2e}")
print(f"Domain error:   {cert.domain_error:.2e}")
print(f"Total error:    {cert.abs_error:.2e}")
```

### Barrier options

```python
from chernoffpy import CrankNicolson
from chernoffpy.finance import BarrierPricer, BarrierParams, MarketParams

market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
barrier = BarrierParams(barrier=90, barrier_type="down_and_out")

pricer = BarrierPricer(CrankNicolson())
result = pricer.price(market, barrier, n_steps=80, option_type="call")

print(f"Down-and-Out Call: {result.price:.4f}")
print(f"Vanilla Call:      {result.vanilla_price:.4f}")
```

### Double barriers

```python
from chernoffpy import CrankNicolson
from chernoffpy.finance import DoubleBarrierPricer, DoubleBarrierParams, MarketParams

market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
params = DoubleBarrierParams(lower_barrier=80, upper_barrier=120, barrier_type="double_knock_out")

pricer = DoubleBarrierPricer(CrankNicolson())
result = pricer.price(market, params, n_steps=80, option_type="call")
print(f"Double KO Call: {result.price:.4f}")
```

### American options (early exercise)

```python
from chernoffpy import CrankNicolson
from chernoffpy.finance import AmericanPricer, MarketParams

market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
pricer = AmericanPricer(CrankNicolson())
result = pricer.price(market, n_steps=100, option_type="put")

print(f"American put:  {result.price:.4f}")
print(f"European put:  {result.european_price:.4f}")
print(f"Early premium: {result.early_exercise_premium:.4f}")
```

### Local volatility

```python
from chernoffpy import CrankNicolson
from chernoffpy.finance import LocalVolPricer, LocalVolParams, linear_skew

vol_surface = linear_skew(sigma_atm=0.20, skew=0.05, S_ref=100)
params = LocalVolParams(S=100, K=100, T=1.0, r=0.05, vol_surface=vol_surface)

pricer = LocalVolPricer(CrankNicolson())
result = pricer.price(params, n_steps=100, option_type="call")
print(f"Local vol price: {result.price:.4f}")
```

### Surface calibration

```python
from chernoffpy.finance import VolCalibrator, generate_synthetic_quotes

data = generate_synthetic_quotes(spot=100, rate=0.05, sigma=0.20, skew=-0.05)
cal = VolCalibrator()
result = cal.calibrate(data, parametrization="linear_skew", method="iv_fit")

print(result.summary())
```

### Implied volatility

```python
from chernoffpy.finance import implied_volatility

sigma = implied_volatility(market_price=8.02, S=100, K=105, T=1.0, r=0.05)
print(f"Implied vol: {sigma:.4f}")
```

## How It Works

Black-Scholes PDE is transformed (Wilmott substitution) to a heat-equation form on log-price space.  
ChernoffPy then approximates the heat semigroup with product formulas:

- Backward Euler style: first-order behavior in `1/n`
- Crank-Nicolson style: second-order behavior in `1/n^2`

For barriers, a Dirichlet projection is enforced after each Chernoff step.  
For American options, a projection to intrinsic payoff is enforced at every step (`V >= payoff`).  
For local volatility, a frozen-coefficient predictor-corrector scheme updates effective sigma over time.

## Architecture

```text
chernoffpy/
├── functions.py                 # Chernoff functions: BackwardEuler, CrankNicolson, Padé
├── analysis.py                  # Convergence analysis tools
│
└── finance/
    ├── validation.py            # MarketParams, GridConfig, pricing result dataclasses
    ├── transforms.py            # Wilmott substitution BS <-> Heat equation
    ├── european.py              # EuropeanPricer
    ├── greeks.py                # Delta, Gamma, Vega, Theta, Rho
    ├── barrier.py               # BarrierPricer (single barrier)
    ├── barrier_analytical.py    # Reiner-Rubinstein formulas
    ├── double_barrier.py        # DoubleBarrierPricer
    ├── double_barrier_analytical.py  # Fourier-series reference
    ├── american.py              # AmericanPricer (projection method)
    ├── american_analytical.py   # Binomial + BAW reference methods
    ├── local_vol.py             # LocalVolPricer
    ├── implied_vol.py           # price -> sigma
    ├── market_data.py           # MarketQuote, MarketData, synthetic quotes
    ├── calibration.py           # VolCalibrator (iv_fit/model_fit)
    ├── reporting.py             # Human-readable reports
    └── __init__.py              # Public API

examples/
└── finance_demo.ipynb
```

## Accuracy Snapshot

| Configuration | Typical error |
|---|---|
| European ATM, Crank-Nicolson, `n=50` | ~0.015% |
| Barrier DOC ATM, Crank-Nicolson | a few % (barrier discretization dependent) |
| American put ATM vs binomial | typically within a few % |
| American call (no dividends) | matches European call |
| Local vol (flat sigma) vs European | very small mismatch |
| Implied-vol roundtrip | near machine precision in tests |

## Tests

```bash
pytest tests/ -q
# 324 passed
```

## Installation

```bash
git clone https://github.com/sergeeey/MarkovChains.git
cd MarkovChains
pip install -e .
```

Requirements: Python 3.10+, NumPy, SciPy.

## References

- Chernoff, P.R. (1968). Note on product formulas for operator semigroups.
- Galkin, O. & Remizov, I. (2025). Convergence rates for Chernoff-type approximations.
- Reiner, E. & Rubinstein, M. (1991). Breaking down the barriers.
- Cox, J., Ross, S., Rubinstein, M. (1979). Option Pricing: A Simplified Approach.
- Barone-Adesi, G. & Whaley, R. (1987). Efficient Analytic Approximation of American Option Values.
- Dupire, B. (1994). Pricing with a Smile.
- Black, F. & Scholes, M. (1973). The pricing of options and corporate liabilities.

## License

MIT. See `LICENSE`.
