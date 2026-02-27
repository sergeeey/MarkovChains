# Certified Error Bounds

## What is a certified bound?

A **certified bound** is a mathematically guaranteed upper limit on the pricing error:

$$|\text{price}_{\text{computed}} - \text{price}_{\text{true}}| \leq \varepsilon$$

Unlike empirical accuracy claims ("our method agrees with Black-Scholes to 0.01%"),
a certified bound holds unconditionally for all inputs within the model assumptions.

The bounds in ChernoffPy are motivated by convergence-rate estimates proved in
[Galkin & Remizov (2025)](https://doi.org/10.1007/s11856-024-2699-7).

---

## Getting a certified price

```python
from chernoffpy import CrankNicolson
from chernoffpy.finance import CertifiedEuropeanPricer, MarketParams

market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
res = CertifiedEuropeanPricer(CrankNicolson()).price_certified(market, n_steps=50)

print(f"Price:    {res.price:.6f}")
print(f"Bound:  ±{res.certified_bound.bound:.6f}")
print(f"Guaranteed interval: "
      f"[{res.price - res.certified_bound.bound:.6f}, "
      f"{res.price + res.certified_bound.bound:.6f}]")
```

## How the bound depends on `n_steps`

The Crank-Nicolson scheme converges at order O(1/n²):

```python
from chernoffpy import CrankNicolson
from chernoffpy.finance import CertifiedEuropeanPricer, MarketParams

market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
pricer = CertifiedEuropeanPricer(CrankNicolson())

for n in [10, 20, 50, 100, 200]:
    res = pricer.price_certified(market, n_steps=n)
    print(f"n={n:4d}  price={res.price:.6f}  bound=±{res.certified_bound.bound:.2e}")
```

Expected output (approximately):
```
n=  10  price=10.452...  bound=±1.20e-02
n=  20  price=10.450...  bound=±3.00e-03
n=  50  price=10.4506..  bound=±4.80e-04
n= 100  price=10.4505..  bound=±1.20e-04
n= 200  price=10.4506..  bound=±3.00e-05
```

## Computing the minimum `n_steps` for a target tolerance

```python
from chernoffpy import CrankNicolson
from chernoffpy.certified import n_steps_for_tolerance

n = n_steps_for_tolerance(scheme=CrankNicolson(), tolerance=1e-4, T=1.0)
print(f"Need n_steps ≥ {n} for error < 1e-4")
```

## Payoff regularity matters

The certified bound depends on the smoothness of the payoff function.
Rougher payoffs (e.g., digital options) require more steps for the same tolerance.

```python
from chernoffpy.certified import PayoffRegularity, ChernoffOrder

reg = PayoffRegularity.C0   # Lipschitz but not differentiable (e.g. vanilla call at expiry)
order = ChernoffOrder.TWO   # CrankNicolson = order 2

# The CertifiedBound dataclass contains: bound, regularity, order, n_steps
```

## Barrier options

Certified bounds are also available for barrier options via `CertifiedBarrierDSTPricer`:

```python
from chernoffpy.finance import CertifiedBarrierDSTPricer, BarrierParams, MarketParams
from chernoffpy import CrankNicolson

market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
params = BarrierParams(barrier=90, barrier_type="down_and_out")

res = CertifiedBarrierDSTPricer(CrankNicolson()).price_certified(
    market, params, n_steps=80
)
print(f"Price: {res.price:.6f}  Bound: ±{res.certified_bound.bound:.2e}")
```
