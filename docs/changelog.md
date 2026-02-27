# Changelog

## [0.1.0] - 2026-02-16

### Added

- European option pricing with Chernoff schemes (BE/CN/Padé)
- Barrier and double-barrier pricing (including DST-based barrier solver)
- American pricing with early exercise and discrete-dividend support
- Local-volatility pricing, implied volatility, and calibration helpers
- Heston stochastic-volatility pricing and accelerated HestonFastPricer
- Bates model pricing (Heston + Merton jumps)
- Certified error-bound core utilities and certified pricer wrappers
- Optional acceleration modules (`numba`/`cupy`) and benchmark script

### Validation

- Comprehensive automated tests across core and finance modules
- Analytical references: Black-Scholes, barrier formulas, Heston/Bates CF methods
- CI-ready packaging metadata and workflow templates
