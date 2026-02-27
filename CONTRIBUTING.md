# Contributing to ChernoffPy

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Setting Up the Development Environment

```bash
git clone https://github.com/sergeeey/MarkovChains.git
cd MarkovChains/ChernoffPy
pip install -e ".[dev]"
```

For optional acceleration:
```bash
pip install -e ".[fast]"   # Numba JIT
pip install -e ".[gpu]"    # CuPy GPU support
pip install -e ".[all]"    # Everything
```

## Running Tests

```bash
# All tests
pytest tests/ -q

# With coverage report
pytest tests/ --cov=chernoffpy --cov-report=term-missing

# Specific module
pytest tests/finance/ -q
pytest tests/test_functions.py -q
```

Coverage must stay ≥ 80% — the CI will fail otherwise.

## Code Style

- **Formatter:** [Black](https://black.readthedocs.io/) with 100-char line length
- **Type hints:** required on all public functions and methods
- **Docstrings:** NumPy-style for public API, with `Parameters`, `Returns`, and `Examples` sections
- **No `print()`:** use `structlog` or `logging` for any diagnostic output

Format before committing:
```bash
black chernoffpy/ tests/ --line-length 100
```

## Commit Message Convention

```
feat: add double-barrier DST pricer
fix: overflow in bs_to_heat_initial for large alpha
docs: add decision guide to README
test: cover AmericanPricer with non-zero dividends
refactor: extract grid snapping to adaptive_grid module
```

## Submitting a Pull Request

1. Fork the repository and create a feature branch from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. Write your code with type hints and docstrings.
3. Add tests covering the new behaviour. Aim for edge cases, not just happy paths.
4. Run the full test suite locally to confirm nothing is broken.
5. Open a PR with a clear description of:
   - **What** the change does
   - **Why** it is needed
   - **How** it was tested

## Mathematical Contributions

ChernoffPy is grounded in the theory of Chernoff operator semigroups (Galkin & Remizov, 2025).
If you are adding a new pricing model or Chernoff function:

- Include a reference to the relevant paper or textbook
- Add a convergence test that verifies the expected order `O(1/n^k)`
- Compare against an analytical benchmark where one exists

## Project Structure

```
chernoffpy/
├── functions.py      # Chernoff function implementations (abstract base + 5 schemes)
├── semigroups.py     # Exact heat semigroup solutions (for validation)
├── analysis.py       # Convergence analysis utilities
├── backends.py       # NumPy / CuPy abstraction
├── accel.py          # Numba JIT kernels
├── certified.py      # Certified upper error bounds
└── finance/          # Option pricing layer (builds on core)
    ├── validation.py # Dataclasses (MarketParams, GridConfig, ...)
    ├── transforms.py # Black-Scholes ↔ heat equation transforms
    ├── european.py   # EuropeanPricer
    ├── barrier*.py   # Barrier pricers (FFT and DST)
    ├── american*.py  # American pricer
    ├── heston*.py    # Heston and fast Heston
    ├── bates*.py     # Bates jump-diffusion
    └── ...
```

**Dependency rule:** `finance/` may depend on core (`functions.py`, `semigroups.py`). Core must never import from `finance/`.

## Reporting Issues

Please open a GitHub issue with:
- ChernoffPy version (`python -c "import chernoffpy; print(chernoffpy.__version__)"`)
- Python version and OS
- Minimal reproducible example
- Expected vs actual result (with numbers)
