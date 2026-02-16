"""Double barrier option pricer via Chernoff approximation."""

from __future__ import annotations

import numpy as np

from .european import EuropeanPricer
from .transforms import (
    bs_to_heat_initial,
    compute_transform_params,
    extract_price_at_spot,
    make_grid,
)
from .validation import (
    DoubleBarrierParams,
    DoubleBarrierPricingResult,
    GridConfig,
    MarketParams,
)


class DoubleBarrierPricer:
    """Price double barrier options with two Dirichlet projections."""

    def __init__(self, chernoff, grid_config: GridConfig | None = None):
        self.chernoff = chernoff
        self.grid_config = grid_config if grid_config is not None else GridConfig()
        self._european = EuropeanPricer(chernoff, self.grid_config)

    def price(
        self,
        market: MarketParams,
        barrier_params: DoubleBarrierParams,
        n_steps: int = 50,
        option_type: str = "call",
    ) -> DoubleBarrierPricingResult:
        if option_type not in {"call", "put"}:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        self._validate(market, barrier_params)

        if "out" in barrier_params.barrier_type:
            return self._price_knockout(market, barrier_params, n_steps, option_type)
        return self._price_knockin(market, barrier_params, n_steps, option_type)

    def _validate(self, market: MarketParams, barrier_params: DoubleBarrierParams) -> None:
        s = market.S
        b_l = barrier_params.lower_barrier
        b_u = barrier_params.upper_barrier

        if s <= b_l:
            raise ValueError(f"Spot ({s}) must be > lower barrier ({b_l})")
        if s >= b_u:
            raise ValueError(f"Spot ({s}) must be < upper barrier ({b_u})")

    def _build_aligned_grid(self, x_lower: float, x_upper: float, option_type: str) -> np.ndarray:
        """Shift base grid to align a barrier node with dominant payoff side."""
        x_grid = make_grid(self.grid_config)
        target = x_upper if option_type == "call" else x_lower
        idx = int(np.argmin(np.abs(x_grid - target)))
        shift = target - x_grid[idx]
        return x_grid + shift

    def _price_knockout(
        self,
        market: MarketParams,
        barrier_params: DoubleBarrierParams,
        n_steps: int,
        option_type: str,
    ) -> DoubleBarrierPricingResult:
        x_lower = np.log(barrier_params.lower_barrier / market.K)
        x_upper = np.log(barrier_params.upper_barrier / market.K)
        x_grid = self._build_aligned_grid(x_lower, x_upper, option_type)

        u = bs_to_heat_initial(x_grid, market, self.grid_config, option_type)
        barrier_mask = (x_grid <= x_lower + 1e-14) | (x_grid >= x_upper - 1e-14)
        u = np.where(barrier_mask, 0.0, u)

        _, _, _, t_eff = compute_transform_params(market)
        n_internal = max(n_steps, 500)
        dt = t_eff / n_internal

        for _ in range(n_internal):
            u = self.chernoff.apply(u, x_grid, dt)
            u = np.where(barrier_mask, 0.0, u)

        dko_price = max(0.0, extract_price_at_spot(u, x_grid, market))
        vanilla_result = self._european.price(market, n_steps=n_steps, option_type=option_type)

        return DoubleBarrierPricingResult(
            price=dko_price,
            vanilla_price=vanilla_result.price,
            knockout_price=dko_price,
            barrier_type=barrier_params.barrier_type,
            method_name=self.chernoff.name,
            n_steps=n_steps,
            market=market,
            barrier_params=barrier_params,
            certificate=None,
        )

    def _price_knockin(
        self,
        market: MarketParams,
        barrier_params: DoubleBarrierParams,
        n_steps: int,
        option_type: str,
    ) -> DoubleBarrierPricingResult:
        out_params = DoubleBarrierParams(
            lower_barrier=barrier_params.lower_barrier,
            upper_barrier=barrier_params.upper_barrier,
            barrier_type="double_knock_out",
            rebate=barrier_params.rebate,
        )

        ko_result = self._price_knockout(market, out_params, n_steps, option_type)
        dki_price = max(0.0, ko_result.vanilla_price - ko_result.knockout_price)

        return DoubleBarrierPricingResult(
            price=dki_price,
            vanilla_price=ko_result.vanilla_price,
            knockout_price=ko_result.knockout_price,
            barrier_type=barrier_params.barrier_type,
            method_name=self.chernoff.name,
            n_steps=n_steps,
            market=market,
            barrier_params=barrier_params,
            certificate=None,
        )
