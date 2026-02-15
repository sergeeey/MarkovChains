"""European option pricing using Chernoff approximation of the heat semigroup.

Converts the Black-Scholes PDE to a heat equation via Wilmott substitution,
then solves it using ChernoffPy's operator approximation machinery.
Each price comes with a ValidationCertificate decomposing the error.
"""

from .validation import (
    GridConfig,
    GreeksResult,
    MarketParams,
    PricingResult,
    ValidationCertificate,
)
from .transforms import (
    bs_exact_price,
    bs_to_heat_initial,
    compute_transform_params,
    extract_price_at_spot,
    heat_to_bs_price,
    make_grid,
    make_taper,
)
from .european import EuropeanPricer
from .greeks import compute_greeks

__all__ = [
    "MarketParams",
    "GridConfig",
    "ValidationCertificate",
    "PricingResult",
    "GreeksResult",
    "EuropeanPricer",
    "compute_greeks",
    "compute_transform_params",
    "make_grid",
    "make_taper",
    "bs_to_heat_initial",
    "heat_to_bs_price",
    "extract_price_at_spot",
    "bs_exact_price",
]
