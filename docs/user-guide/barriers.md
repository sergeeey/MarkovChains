# Barrier Options

## Barrier types

ChernoffPy supports all 8 standard single-barrier variants:

| `barrier_type` | Description |
|----------------|-------------|
| `"down_and_out"` | Option knocked out if S falls below barrier |
| `"down_and_in"` | Option activated only if S falls below barrier |
| `"up_and_out"` | Option knocked out if S rises above barrier |
| `"up_and_in"` | Option activated only if S rises above barrier |

Each can be combined with `option_type="call"` or `"put"`.

## Single barrier (DST — recommended)

```python
from chernoffpy import CrankNicolson
from chernoffpy.finance import BarrierDSTPricer, BarrierParams, MarketParams

market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
params = BarrierParams(barrier=90, barrier_type="down_and_out")

result = BarrierDSTPricer(CrankNicolson()).price(
    market, params, n_steps=80, option_type="call"
)
print(result.price)   # cheaper than vanilla (knock-out risk)
```

The DST method enforces zero boundary conditions exactly via Discrete Sine Transform —
no Gibbs oscillations near the barrier.

## Double barrier

```python
from chernoffpy.finance import DoubleBarrierDSTPricer, DoubleBarrierParams

params = DoubleBarrierParams(lower_barrier=85, upper_barrier=115)
result = DoubleBarrierDSTPricer(CrankNicolson()).price(
    market, params, n_steps=100, option_type="call"
)
print(result.price)   # very cheap — knocked out by either barrier
```

## All 8 variants at once

```python
from chernoffpy.finance import BarrierDSTPricer, BarrierParams, MarketParams
from chernoffpy import CrankNicolson

market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
pricer = BarrierDSTPricer(CrankNicolson())

barrier_types = ["down_and_out", "down_and_in", "up_and_out", "up_and_in"]
barrier_level = 90  # adjust to 110 for up-barriers

for bt in barrier_types:
    for opt in ["call", "put"]:
        b = barrier_level if "down" in bt else 110
        res = pricer.price(market, BarrierParams(b, bt), n_steps=80, option_type=opt)
        print(f"{bt:15s} {opt}: {res.price:.4f}")
```

## Certified barrier bound

```python
from chernoffpy.finance import CertifiedBarrierDSTPricer

res = CertifiedBarrierDSTPricer(CrankNicolson()).price_certified(
    market, BarrierParams(90, "down_and_out"), n_steps=80
)
print(f"Price: {res.price:.6f}  Bound: ±{res.certified_bound.bound:.6f}")
```

## Adaptive grid (barrier snapping)

When the barrier does not fall on a grid point, interpolation error can dominate.
Use barrier snapping to align the grid exactly:

```python
from chernoffpy.finance import BarrierDSTPricer, BarrierParams, MarketParams, GridConfig
from chernoffpy.finance.adaptive_grid import make_stretched_config

market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
params = BarrierParams(barrier=87.3, barrier_type="down_and_out")

grid_cfg = make_stretched_config(market, barrier=87.3, n_points=256)
result = BarrierDSTPricer(CrankNicolson()).price(
    market, params, n_steps=80, grid_config=grid_cfg
)
```
