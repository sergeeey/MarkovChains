# Benchmarks: ChernoffPy vs QuantLib

Comprehensive numerical comparison of ChernoffPy against QuantLib (industry-standard C++ library) for option pricing.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all benchmarks (generates CSV tables)
python benchmarks/vs_quantlib.py

# Generate plots from results
python benchmarks/plot_results.py
```

## Structure

```
benchmarks/
├── helpers.py              # QuantLib wrapper functions
├── vs_quantlib.py          # Main benchmark script
├── plot_results.py         # Visualization script
├── requirements.txt        # Dependencies (QuantLib-Python)
├── README.md              # This file
└── results/
    ├── REPORT.md          # Human-readable summary
    ├── tables/            # CSV data files
    │   ├── european_comparison.csv
    │   ├── barrier_comparison.csv
    │   ├── american_comparison.csv
    │   ├── heston_comparison.csv
    │   ├── certified_bounds.csv
    │   └── double_barrier_comparison.csv
    └── plots/             # PNG figures for paper
        ├── convergence_european.png
        ├── efficiency_scatter.png
        ├── certified_bounds.png
        ├── barrier_comparison.png
        ├── convergence_heston.png
        └── convergence_american.png
```

## Benchmarks Overview

### 1. European Options
- **ChernoffPy:** BackwardEuler, CrankNicolson, Padé schemes
- **QuantLib:** AnalyticEuropeanEngine, FdBlackScholesVanillaEngine
- **Exact:** Black-Scholes closed-form

### 2. Barrier Options
- **ChernoffPy:** BarrierDSTPricer (DST-based, Gibbs-free)
- **QuantLib:** AnalyticBarrierEngine, FdBlackScholesBarrierEngine
- **Exact:** Reiner-Rubinstein formulas

### 3. American Options
- **ChernoffPy:** AmericanPricer with payoff projection
- **QuantLib:** FdBlackScholesVanillaEngine, BinomialVanillaEngine
- **Exact:** CRR Binomial n=50000 (quasi-exact)

### 4. Heston Model
- **ChernoffPy:** HestonFastPricer (splitting + FFT)
- **QuantLib:** AnalyticHestonEngine, FdHestonVanillaEngine
- **Exact:** Lewis characteristic function quadrature

### 5. Certified Bounds
- **ChernoffPy only:** Provable upper bounds on discretization error
- **QuantLib:** Not available

## Key Metrics

For each test, we measure:
1. **Price** - Numerical value
2. **Error %** - |price - exact| / exact × 100
3. **Time (ms)** - Median of 10 runs
4. **Efficiency** - Error / Time (accuracy per ms)
5. **Certified Bound** - Only ChernoffPy (provable upper bound)

## Results Summary

See `results/REPORT.md` for detailed analysis and conclusions for paper.

### Quick Highlights

| Feature | ChernoffPy | QuantLib |
|---------|-----------|----------|
| European Accuracy | 0.015% @ n=50 | 0.017% @ n=50 |
| Certified Bounds | ✓ Yes (2× ratio) | ✗ No |
| Barrier Stability | ✓ Gibbs-free | ⚠ Near-barrier issues |
| American Accuracy | 0.19% | 0.02% |
| Heston Speed | ~11s | ~0.25s |

## Customization

Edit `vs_quantlib.py` to add:
- New parameter configurations
- Additional option types
- Different grid sizes
- More n_steps values

## Troubleshooting

### QuantLib Installation Issues
```bash
# If pip install fails on Windows, try:
conda install -c conda-forge quantlib
```

### Slow Heston Benchmarks
The Heston benchmark is intentionally thorough (large grids). To speed up:
- Reduce `n_runs` from 3 to 1 in `benchmark_heston()`
- Use smaller grids: `(64, 32)` instead of `(512, 96)`
- Skip large n_steps values

## Citation

If using these benchmarks in academic work:

```bibtex
@software{chernoffpy_benchmarks,
  title={ChernoffPy: Chernoff Approximations for Option Pricing},
  author={Based on Galkin-Remizov convergence theory},
  year={2026},
  note={Benchmark comparison with QuantLib}
}
```

## References

- QuantLib: https://www.quantlib.org/
- Galkin-Remizov (2025): Convergence rate theory for Chernoff approximations
- Dragunova-Nikbakht-Remizov (2023): Original numerical code (arXiv:2301.05284)
