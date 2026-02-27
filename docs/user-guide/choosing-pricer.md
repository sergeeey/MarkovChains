# Choosing the Right Pricer

## Option type → Pricer

| Option type | Pricer | Import |
|-------------|--------|--------|
| European call/put | `EuropeanPricer` | `chernoffpy.finance` |
| European + certified bound | `CertifiedEuropeanPricer` | `chernoffpy.finance` |
| Single barrier (no Gibbs) | `BarrierDSTPricer` | `chernoffpy.finance` |
| Single barrier (fast, FFT) | `BarrierPricer` | `chernoffpy.finance` |
| Single barrier + certified | `CertifiedBarrierDSTPricer` | `chernoffpy.finance` |
| Double barrier | `DoubleBarrierDSTPricer` | `chernoffpy.finance` |
| Double barrier (FFT) | `DoubleBarrierPricer` | `chernoffpy.finance` |
| American (no dividends) | `AmericanPricer` | `chernoffpy.finance` |
| American + discrete dividends | `AmericanPricer` + `DividendSchedule` | `chernoffpy.finance` |
| Heston (stoch. vol.) | `HestonFastPricer` | `chernoffpy.finance` |
| Heston (custom grid) | `HestonPricer` | `chernoffpy.finance` |
| Bates (Heston + jumps) | `BatesPricer` | `chernoffpy.finance` |
| Local volatility | `LocalVolPricer` | `chernoffpy.finance` |

---

## DST vs FFT for barrier options

Both `BarrierDSTPricer` and `BarrierPricer` price the same product. The difference is numerical:

| Method | Algorithm | Gibbs artifacts? | Speed |
|--------|-----------|-----------------|-------|
| `BarrierDSTPricer` | Discrete Sine Transform | **No** — exact Dirichlet BCs | Slightly slower |
| `BarrierPricer` | Fast Fourier Transform | Possible near barrier | Faster |

**Use `BarrierDSTPricer` by default.** Switch to `BarrierPricer` only if you are pricing many
options in a tight loop and have verified accuracy is acceptable for your parameters.

---

## Chernoff scheme → accuracy trade-off

| Scheme | Convergence order | A-stable | Best for |
|--------|------------------|----------|---------|
| `BackwardEuler()` | O(1/n) | ✓ | Debugging, coarse checks |
| `CrankNicolson()` | O(1/n²) | ✓ | **Default** — best balance |
| `PadeChernoff(1, 2)` | O(1/n³) | ✓ | Tight certified bounds |
| `PadeChernoff(2, 2)` | O(1/n⁴) | ✓ | Maximum accuracy |
| `PhysicalG()` | O(1/n) | ✓ | Explicit scheme |
| `PhysicalS()` | O(1/n²) | ✓ | Explicit, 2nd order |

!!! warning "Avoid `PadeChernoff(2, 1)`"
    This configuration is **not A-stable**. High-frequency components are amplified and the
    approximation diverges. ChernoffPy will emit a `UserWarning` if you use it.

---

## Decision flowchart

```
Do you need a guaranteed error bound?
  YES → CertifiedEuropeanPricer / CertifiedBarrierDSTPricer
  NO  →
        Is there a barrier?
          YES →  Single barrier → BarrierDSTPricer
                 Double barrier → DoubleBarrierDSTPricer
          NO  →
                Is there early exercise?
                  YES → AmericanPricer  (+DividendSchedule if dividends)
                  NO  →
                        Is volatility stochastic?
                          YES, Heston       → HestonFastPricer
                          YES, Heston+jumps → BatesPricer
                          YES, local vol    → LocalVolPricer
                          NO  → EuropeanPricer
```

---

## Scheme selection rule of thumb

Start with `CrankNicolson()`. It is second-order accurate, A-stable, and works well for all
option types. Move to `PadeChernoff(1, 2)` when:

- You are computing certified bounds and want fewer steps for the same tolerance
- The payoff is smooth (European, barrier away from the money)

Stick with `CrankNicolson()` for:

- American options (discontinuous free boundary)
- Barrier options close to the strike
- Heston / Bates (2D problem; higher-order gains are smaller)
