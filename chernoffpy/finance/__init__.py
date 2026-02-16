"""European and exotic option pricing via Chernoff approximations."""

from .validation import (
    AmericanPricingResult,
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
from .barrier_dst import BarrierDSTPricer, DoubleBarrierDSTPricer
from .american import AmericanPricer
from .heston import HestonPricer
from .heston_fast import HestonFastPricer
from .local_vol import LocalVolParams, LocalVolPricer, flat_vol, linear_skew, time_dependent_vol
from .implied_vol import implied_volatility
from .market_data import CalibrationResult, MarketData, MarketQuote, generate_synthetic_quotes
from .calibration import VolCalibrator
from .american_analytical import american_baw, american_binomial
from .heston_params import HestonGridConfig, HestonParams, HestonPricingResult
from .heston_analytical import heston_price
from .adaptive_grid import (
    compute_grid_quality,
    make_stretched_config,
    snap_grid_to_barrier,
    snap_grid_to_double_barrier,
)
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
    "BarrierDSTPricer",
    "DoubleBarrierDSTPricer",
    "AmericanPricer",
    "HestonPricer",
    "HestonFastPricer",
    "compute_greeks",
    "bs_exact_price",
    "PricingResult",
    "ValidationCertificate",
    "GreeksResult",
    "BarrierParams",
    "BarrierPricingResult",
    "DoubleBarrierParams",
    "DoubleBarrierPricingResult",
    "AmericanPricingResult",
    "HestonParams",
    "HestonGridConfig",
    "HestonPricingResult",
    "heston_price",
    "snap_grid_to_barrier",
    "snap_grid_to_double_barrier",
    "make_stretched_config",
    "compute_grid_quality",
    # Phase 3+
    "LocalVolPricer",
    "LocalVolParams",
    "flat_vol",
    "linear_skew",
    "time_dependent_vol",
    "implied_volatility",
    "MarketQuote",
    "MarketData",
    "CalibrationResult",
    "generate_synthetic_quotes",
    "VolCalibrator",
    # Analytical references
    "american_baw",
    "american_binomial",
    # Reporting
    "certificate_to_report",
    "pricing_result_to_report",
    "barrier_result_to_report",
]
