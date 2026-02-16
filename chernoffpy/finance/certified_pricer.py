"""Certified wrappers for finance pricers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chernoffpy.certified import (
    ChernoffOrder,
    PayoffRegularity,
    compute_certified_bound,
    n_steps_for_tolerance,
    verify_convergence_order,
)

from .barrier_analytical import barrier_analytical
from .barrier_dst import BarrierDSTPricer
from .european import EuropeanPricer
from .transforms import bs_exact_price
from .validation import BarrierParams, GridConfig, MarketParams


@dataclass
class CertifiedPricingResult:
    """Bundle: price + certified bound + wrapped pricer result + diagnostics."""

    price: float
    certified_bound: Any
    pricing_result: Any
    verification: dict[str, Any]


class CertifiedEuropeanPricer:
    """European pricer with certified upper bounds for discretization error."""

    def __init__(self, chernoff, grid_config: GridConfig | None = None):
        self.chernoff = chernoff
        self.grid_config = grid_config if grid_config is not None else GridConfig()
        self._pricer = EuropeanPricer(chernoff, self.grid_config)
        self._order = ChernoffOrder.from_chernoff(chernoff)

    def price_certified(
        self,
        market: MarketParams,
        n_steps: int = 50,
        option_type: str = "call",
        safety_factor: float = 2.0,
    ) -> CertifiedPricingResult:
        """Price option and return certified bound based on multi-n evaluation."""
        ns = [n_steps, 2 * n_steps, 4 * n_steps]
        results = {}
        prices = {}
        for n in ns:
            r = self._pricer.price(market, n_steps=n, option_type=option_type)
            results[n] = r
            prices[n] = r.price

        exact = bs_exact_price(market, option_type)
        bound = compute_certified_bound(
            prices=prices,
            chernoff_order=self._order,
            payoff_reg=PayoffRegularity.vanilla_call(),
            n_target=n_steps,
            safety_factor=safety_factor,
            exact_price=exact,
        )
        verification = verify_convergence_order(
            prices=prices,
            expected_order=bound.effective_order,
            exact_price=exact,
        )

        result = results[n_steps]
        result.certificate.certified_bound = bound.bound
        result.certificate.certified_order = bound.effective_order
        result.certificate.is_certified = bound.is_certified

        return CertifiedPricingResult(
            price=result.price,
            certified_bound=bound,
            pricing_result=result,
            verification=verification,
        )

    def n_for_tolerance(
        self,
        market: MarketParams,
        target_error: float,
        option_type: str = "call",
        pilot_n: int = 20,
        safety_factor: float = 2.0,
    ) -> int:
        """Recommend n_steps for target certified error tolerance."""
        pilot = self.price_certified(
            market=market,
            n_steps=pilot_n,
            option_type=option_type,
            safety_factor=safety_factor,
        )
        return n_steps_for_tolerance(
            target_error=target_error,
            constant_B=pilot.certified_bound.constant_B,
            effective_order=pilot.certified_bound.effective_order,
            safety_factor=safety_factor,
        )


class CertifiedBarrierDSTPricer:
    """DST barrier pricer with certified bounds."""

    def __init__(self, chernoff, grid_config: GridConfig | None = None):
        self.chernoff = chernoff
        self.grid_config = grid_config if grid_config is not None else GridConfig()
        self._pricer = BarrierDSTPricer(chernoff, self.grid_config)
        self._order = ChernoffOrder.from_chernoff(chernoff)

    def price_certified(
        self,
        market: MarketParams,
        barrier_params: BarrierParams,
        n_steps: int = 50,
        option_type: str = "call",
        safety_factor: float = 2.0,
    ) -> CertifiedPricingResult:
        """Price barrier option and return certified bound.

        The DST pricer uses n_internal = max(n_steps, floor) where
        floor = int(10 * sqrt(N)).  When n_steps < floor all evaluations
        collapse to the same n_internal, making Richardson bounds degenerate.
        We ensure the bound-estimation sequence starts above the floor.
        """
        import numpy as np

        n_floor = int(10 * np.sqrt(self.grid_config.N))
        n_base = max(n_steps, n_floor + 1)
        ns = [n_base, 2 * n_base, 4 * n_base]

        results = {}
        prices = {}
        for n in ns:
            r = self._pricer.price(
                market=market,
                barrier_params=barrier_params,
                n_steps=n,
                option_type=option_type,
            )
            results[n] = r
            prices[n] = r.price

        # Also evaluate at the user's n_steps for the returned price
        if n_steps not in results:
            r_user = self._pricer.price(
                market=market,
                barrier_params=barrier_params,
                n_steps=n_steps,
                option_type=option_type,
            )
            results[n_steps] = r_user

        try:
            exact = barrier_analytical(market, barrier_params, option_type)
        except Exception:
            exact = None

        bound = compute_certified_bound(
            prices=prices,
            chernoff_order=self._order,
            payoff_reg=PayoffRegularity.barrier(),
            n_target=n_base,
            safety_factor=safety_factor,
            exact_price=exact,
        )
        verification = verify_convergence_order(
            prices=prices,
            expected_order=bound.effective_order,
            exact_price=exact,
        )

        return CertifiedPricingResult(
            price=results[n_steps].price,
            certified_bound=bound,
            pricing_result=results[n_steps],
            verification=verification,
        )
