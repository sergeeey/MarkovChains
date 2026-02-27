# Heston & Bates Models

## Heston model (stochastic volatility)

The Heston model captures the **volatility smile** by making variance stochastic:

$$dS = r S\,dt + \sqrt{v}\,S\,dW_1$$
$$dv = \kappa(\theta - v)\,dt + \xi\sqrt{v}\,dW_2, \quad dW_1 dW_2 = \rho\,dt$$

**Parameters:**

| Parameter | Meaning | Typical range |
|-----------|---------|---------------|
| `v0` | Initial variance | 0.01 – 0.25 |
| `kappa` | Mean-reversion speed | 0.5 – 5.0 |
| `theta` | Long-run variance | 0.01 – 0.25 |
| `xi` | Vol of vol | 0.1 – 1.0 |
| `rho` | Spot-vol correlation | -0.9 – 0.0 (typically negative) |

## HestonFastPricer (recommended)

```python
from chernoffpy import CrankNicolson
from chernoffpy.finance import HestonFastPricer
from chernoffpy.finance.heston_params import HestonParams

params = HestonParams(
    S=100, K=100, T=1.0, r=0.05,
    v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7
)
result = HestonFastPricer(CrankNicolson()).price(params, n_steps=50, option_type="call")
print(f"Heston call: {result.price:.4f}")
```

`HestonFastPricer` uses a pre-optimised grid (512 × 96 points in S × v space).
Use `HestonPricer` with a custom `HestonGridConfig` only if you need non-default grid resolution.

## Comparing with analytical Heston price

```python
from chernoffpy.finance import heston_price

analytical = heston_price(params, option_type="call")
chernoff   = HestonFastPricer(CrankNicolson()).price(params, n_steps=50).price

print(f"Analytical: {analytical:.6f}")
print(f"Chernoff:   {chernoff:.6f}")
print(f"Error:      {abs(chernoff - analytical):.2e}")
```

---

## Bates model (Heston + Merton jumps)

Bates extends Heston with discrete jumps in the spot price:

$$dS/S = r\,dt + \sqrt{v}\,dW + (e^J - 1)\,dN_t$$

where $J \sim \mathcal{N}(\mu_j, \sigma_j^2)$ and $N_t$ is a Poisson process with intensity $\lambda_j$.

**Additional parameters:**

| Parameter | Meaning |
|-----------|---------|
| `lambda_j` | Jump intensity (jumps per year) |
| `mu_j` | Mean log-jump size |
| `sigma_j` | Std of log-jump size |

```python
from chernoffpy.finance import BatesPricer
from chernoffpy.finance.bates_params import BatesParams

params = BatesParams(
    S=100, K=100, T=1.0, r=0.05,
    v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
    lambda_j=0.5, mu_j=-0.1, sigma_j=0.2
)
result = BatesPricer(CrankNicolson()).price(params, n_steps=50, option_type="call")
print(f"Bates call: {result.price:.4f}")
```

## Comparing Bates with analytical

```python
from chernoffpy.finance import bates_price

analytical = bates_price(params, option_type="call")
chernoff   = BatesPricer(CrankNicolson()).price(params, n_steps=50).price

print(f"Analytical: {analytical:.6f}")
print(f"Chernoff:   {chernoff:.6f}")
```

---

## Calibration

Use `VolCalibrator` to fit model parameters to market quotes:

```python
from chernoffpy.finance import VolCalibrator, generate_synthetic_quotes

# Generate synthetic market data (replace with real quotes)
quotes = generate_synthetic_quotes(S=100, r=0.05, sigma_atm=0.20)

calibrator = VolCalibrator()
result = calibrator.fit(quotes)
print(result.params)   # calibrated flat/skew/SVI parameters
```
