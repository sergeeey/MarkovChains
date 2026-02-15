"""Fixtures for European option pricing tests."""

import pytest

from chernoffpy import CrankNicolson, BackwardEuler
from chernoffpy.finance.validation import MarketParams, GridConfig
from chernoffpy.finance.european import EuropeanPricer


# ---------------------------------------------------------------------------
# Market parameters
# ---------------------------------------------------------------------------

@pytest.fixture
def atm_market():
    """ATM: S=K=100, T=1y, r=5%, sigma=20%."""
    return MarketParams(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)


@pytest.fixture
def itm_call_market():
    """ITM call / OTM put: S=110, K=100."""
    return MarketParams(S=110.0, K=100.0, T=1.0, r=0.05, sigma=0.20)


@pytest.fixture
def otm_call_market():
    """OTM call / ITM put: S=90, K=100."""
    return MarketParams(S=90.0, K=100.0, T=1.0, r=0.05, sigma=0.20)


@pytest.fixture
def deep_itm_call_market():
    """Deep ITM call / Deep OTM put: S=150, K=100."""
    return MarketParams(S=150.0, K=100.0, T=1.0, r=0.05, sigma=0.20)


@pytest.fixture
def deep_otm_call_market():
    """Deep OTM call / Deep ITM put: S=60, K=100."""
    return MarketParams(S=60.0, K=100.0, T=1.0, r=0.05, sigma=0.20)


@pytest.fixture
def high_vol_market():
    """High volatility: sigma=80%."""
    return MarketParams(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.80)


@pytest.fixture
def short_expiry_market():
    """Short expiry: T=1 week."""
    return MarketParams(S=100.0, K=100.0, T=1 / 52, r=0.05, sigma=0.20)


# ---------------------------------------------------------------------------
# Grid configs
# ---------------------------------------------------------------------------

@pytest.fixture
def default_grid():
    """Default: N=2048, L=10, taper=2."""
    return GridConfig()


@pytest.fixture
def fine_grid():
    """Fine: N=4096, L=15, taper=2."""
    return GridConfig(N=4096, L=15.0, taper_width=2.0)


# ---------------------------------------------------------------------------
# Pricers
# ---------------------------------------------------------------------------

@pytest.fixture
def cn_pricer(default_grid):
    """Crank-Nicolson pricer with default grid."""
    return EuropeanPricer(CrankNicolson(), default_grid)


@pytest.fixture
def be_pricer(default_grid):
    """Backward Euler pricer with default grid."""
    return EuropeanPricer(BackwardEuler(), default_grid)
