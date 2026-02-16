"""Market data containers and synthetic quote generation for calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .transforms import bs_exact_price
from .validation import MarketParams


@dataclass(frozen=True)
class MarketQuote:
    """One market option quote used in calibration."""

    strike: float
    expiry: float
    price: float
    option_type: Literal["call", "put"] = "call"
    weight: float = 1.0

    def __post_init__(self):
        if self.strike <= 0:
            raise ValueError(f"strike must be > 0, got {self.strike}")
        if self.expiry <= 0:
            raise ValueError(f"expiry must be > 0, got {self.expiry}")
        if self.price < 0:
            raise ValueError(f"price must be >= 0, got {self.price}")
        if self.weight < 0:
            raise ValueError(f"weight must be >= 0, got {self.weight}")
        if self.option_type not in {"call", "put"}:
            raise ValueError(
                f"option_type must be 'call' or 'put', got '{self.option_type}'"
            )


@dataclass
class MarketData:
    """Collection of market quotes for volatility calibration."""

    spot: float
    rate: float
    quotes: list[MarketQuote] = field(default_factory=list)

    def __post_init__(self):
        if self.spot <= 0:
            raise ValueError(f"spot must be > 0, got {self.spot}")

    def add_quote(
        self,
        strike: float,
        expiry: float,
        price: float,
        option_type: Literal["call", "put"] = "call",
        weight: float = 1.0,
    ) -> "MarketData":
        """Add one quote and return self for chaining."""
        self.quotes.append(
            MarketQuote(
                strike=strike,
                expiry=expiry,
                price=price,
                option_type=option_type,
                weight=weight,
            )
        )
        return self

    @property
    def strikes(self) -> np.ndarray:
        return np.array([q.strike for q in self.quotes], dtype=float)

    @property
    def expiries(self) -> np.ndarray:
        return np.array([q.expiry for q in self.quotes], dtype=float)

    @property
    def prices(self) -> np.ndarray:
        return np.array([q.price for q in self.quotes], dtype=float)

    def implied_vols(self) -> np.ndarray:
        """Compute implied volatility for each quote."""
        from .implied_vol import implied_volatility

        ivs = []
        for q in self.quotes:
            ivs.append(
                implied_volatility(
                    market_price=q.price,
                    S=self.spot,
                    K=q.strike,
                    T=q.expiry,
                    r=self.rate,
                    option_type=q.option_type,
                )
            )
        return np.array(ivs, dtype=float)


@dataclass
class CalibrationResult:
    """Result bundle returned by the calibration routines."""

    params: np.ndarray
    param_names: list[str]
    rmse: float
    max_error: float
    n_quotes: int
    n_iterations: int
    success: bool
    model_prices: np.ndarray
    market_prices: np.ndarray
    vol_surface: object

    def summary(self) -> str:
        """Human-readable summary of calibration quality and parameters."""
        lines = [
            "=" * 50,
            "  ChernoffPy Calibration Result",
            "=" * 50,
            f"  Quotes:     {self.n_quotes}",
            f"  Parameters: {len(self.params)}",
            f"  RMSE:       {self.rmse:.6f}",
            f"  Max error:  {self.max_error:.6f}",
            f"  Converged:  {'Yes' if self.success else 'No'}",
            f"  Iterations: {self.n_iterations}",
            "",
            "  Parameters:",
        ]
        for name, value in zip(self.param_names, self.params):
            lines.append(f"    {name:<12} = {value:.6f}")
        lines.append("=" * 50)
        return "\n".join(lines)


def generate_synthetic_quotes(
    spot: float = 100.0,
    rate: float = 0.05,
    sigma: float = 0.20,
    strikes: tuple[float, ...] = (90.0, 95.0, 100.0, 105.0, 110.0),
    expiries: tuple[float, ...] = (0.25, 0.5, 1.0),
    skew: float = 0.0,
    option_type: Literal["call", "put"] = "call",
) -> MarketData:
    """Generate synthetic Black-Scholes quotes for calibration tests."""
    data = MarketData(spot=spot, rate=rate)

    for expiry in expiries:
        for strike in strikes:
            sigma_k = sigma + skew * np.log(strike / spot)
            sigma_k = max(0.01, float(sigma_k))
            market = MarketParams(S=spot, K=strike, T=expiry, r=rate, sigma=sigma_k)
            price = bs_exact_price(market, option_type)
            data.add_quote(
                strike=strike,
                expiry=expiry,
                price=price,
                option_type=option_type,
            )

    return data
