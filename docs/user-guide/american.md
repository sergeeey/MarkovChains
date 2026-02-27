# American Options

American options can be exercised at any time before expiry. ChernoffPy handles them
via a **payoff projection method**: at each time step the solution is compared against
the intrinsic value and set to `max(continuation, intrinsic)`.

## Basic American put

```python
from chernoffpy import CrankNicolson
from chernoffpy.finance import AmericanPricer, MarketParams

market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
result = AmericanPricer(CrankNicolson()).price(market, n_steps=100, option_type="put")

print(result.price)   # > European put (early exercise premium)
```

!!! note
    American calls on non-dividend-paying assets are never early-exercised, so their price
    equals the European call. Early exercise is only meaningful for **puts** or **dividend-paying calls**.

## With discrete dividends

Discrete dividends cause a jump in the stock price at the ex-dividend date.
ChernoffPy handles this with a jump-adjust step at each dividend.

```python
from chernoffpy import CrankNicolson
from chernoffpy.finance import AmericanPricer, DividendSchedule, MarketParams

market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)

# Two dividends: 2.0 at t=0.25, 2.0 at t=0.75
dividends = DividendSchedule(times=[0.25, 0.75], amounts=[2.0, 2.0])

result = AmericanPricer(CrankNicolson()).price(
    market, n_steps=100, option_type="call", dividends=dividends
)
print(result.price)   # early exercise may be optimal before ex-dividend date
```

## Comparing with analytical benchmarks

```python
from chernoffpy.finance import AmericanPricer, MarketParams, american_baw, american_binomial

market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)

chernoff = AmericanPricer(CrankNicolson()).price(market, n_steps=200, option_type="put").price
baw      = american_baw(market, option_type="put")           # Barone-Adesi-Whaley approx
binomial = american_binomial(market, n=1000, option_type="put")  # CRR tree

print(f"Chernoff: {chernoff:.4f}")
print(f"BAW:      {baw:.4f}")
print(f"Binomial: {binomial:.4f}")
```

## Notes on `n_steps`

American options require more steps than European due to the free boundary:

- `n_steps=50` — rough estimate
- `n_steps=100` — standard accuracy
- `n_steps=200+` — benchmarking / tight comparison with binomial trees
