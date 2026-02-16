"""European and exotic option pricing via Chernoff approximations."""

from .validation import (
    BarrierParams,
    BarrierPricingResult,
    DoubleBarrierParams,
    DoubleBarrierPricingResult,
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
from .barrier import BarrierPricer
from .double_barrier import DoubleBarrierPricer
from .local_vol import LocalVolParams, LocalVolPricer, flat_vol, linear_skew, time_dependent_vol
from .implied_vol import implied_volatility
from .reporting import (
    barrier_result_to_report,
    certificate_to_report,
    pricing_result_to_report,
)

__all__ = [
    # Existing public API
    "MarketParams",
    "GridConfig",
    "EuropeanPricer",
    "BarrierPricer",
    "DoubleBarrierPricer",
    "compute_greeks",
    "bs_exact_price",
    "PricingResult",
    "ValidationCertificate",
    "GreeksResult",
    "BarrierParams",
    "BarrierPricingResult",
    "DoubleBarrierParams",
    "DoubleBarrierPricingResult",
    # Phase 3+
    "LocalVolPricer",
    "LocalVolParams",
    "flat_vol",
    "linear_skew",
    "time_dependent_vol",
    "implied_volatility",
    "certificate_to_report",
    "pricing_result_to_report",
    "barrier_result_to_report",
]
