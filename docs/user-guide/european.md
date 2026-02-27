# European Options

## Basic pricing

```python
from chernoffpy import CrankNicolson
from chernoffpy.finance import EuropeanPricer, MarketParams

market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
pricer = EuropeanPricer(CrankNicolson())

call = pricer.price(market, n_steps=50, option_type="call")
put  = pricer.price(market, n_steps=50, option_type="put")

print(f"Call: {call.price:.4f}")   # ~10.4506
print(f"Put:  {put.price:.4f}")    # ~5.5735  (put-call parity holds)
```

## Greeks

```python
from chernoffpy.finance import compute_greeks

greeks = compute_greeks(pricer, market, n_steps=50, option_type="call")

print(f"Delta: {greeks.delta:.4f}")   # ~0.6368
print(f"Gamma: {greeks.gamma:.4f}")   # ~0.0188
print(f"Vega:  {greeks.vega:.4f}")    # ~37.52
print(f"Theta: {greeks.theta:.4f}")   # ~-6.41 (per year)
print(f"Rho:   {greeks.rho:.4f}")     # ~53.23
```

## Certified error bound

```python
from chernoffpy.finance import CertifiedEuropeanPricer

cert_pricer = CertifiedEuropeanPricer(CrankNicolson())
res = cert_pricer.price_certified(market, n_steps=50, option_type="call")

print(f"Price:   {res.price:.6f}")
print(f"Bound: ±{res.certified_bound.bound:.6f}")
```

The `bound` is a **mathematically guaranteed** upper limit on `|price_true - price_computed|`.

## Comparing schemes

```python
from chernoffpy import BackwardEuler, CrankNicolson, PadeChernoff
from chernoffpy.finance import EuropeanPricer, MarketParams, bs_exact_price

market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
exact = bs_exact_price(market, "call")

for scheme, label in [
    (BackwardEuler(), "BackwardEuler O(1/n)"),
    (CrankNicolson(), "CrankNicolson O(1/n²)"),
    (PadeChernoff(1, 2), "Padé[1/2]   O(1/n³)"),
]:
    result = EuropeanPricer(scheme).price(market, n_steps=50)
    err = abs(result.price - exact)
    print(f"{label:30s}  price={result.price:.6f}  err={err:.2e}")
```
