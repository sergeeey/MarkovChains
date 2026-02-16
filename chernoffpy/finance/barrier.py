"""Barrier option pricer via Chernoff approximation with Dirichlet projection."""

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
    BarrierParams,
    BarrierPricingResult,
    GridConfig,
    MarketParams,
)


class BarrierPricer:
    """Price barrier options using Chernoff steps and barrier projection.

    Knock-out is computed by applying Chernoff on the heat equation and forcing
    zero values on/behind the barrier after each time step.
    Knock-in is computed with in-out parity.
    """

    def __init__(self, chernoff, grid_config: GridConfig | None = None):
        self.chernoff = chernoff
        self.grid_config = grid_config if grid_config is not None else GridConfig()
        self._european = EuropeanPricer(chernoff, self.grid_config)

    def price(
        self,
        market: MarketParams,
        barrier_params: BarrierParams,
        n_steps: int = 50,
        option_type: str = "call",
    ) -> BarrierPricingResult:
        """Compute barrier option price for call/put and 4 barrier types."""
        if option_type not in {"call", "put"}:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        self._validate(market, barrier_params)

        if barrier_params.barrier_type.endswith("_out"):
            return self._price_knockout(market, barrier_params, n_steps, option_type)
        return self._price_knockin(market, barrier_params, n_steps, option_type)

    def _validate(self, market: MarketParams, barrier_params: BarrierParams) -> None:
        b = barrier_params.barrier
        s = market.S
        bt = barrier_params.barrier_type

        if bt.startswith("down") and b >= s:
            raise ValueError(f"Down barrier ({b}) must be < spot ({s})")
        if bt.startswith("up") and b <= s:
            raise ValueError(f"Up barrier ({b}) must be > spot ({s})")

    def _build_aligned_grid(self, config: GridConfig, x_barrier: float) -> np.ndarray:
        """Shift uniform grid so barrier lies exactly on one node.

        This reduces barrier-location bias from hard projection on nearest nodes.
        """
        x_grid = make_grid(config)
        idx = int(np.argmin(np.abs(x_grid - x_barrier)))
        shift = x_barrier - x_grid[idx]
        return x_grid + shift

    def _price_knockout(
        self,
        market: MarketParams,
        barrier_params: BarrierParams,
        n_steps: int,
        option_type: str,
    ) -> BarrierPricingResult:
        config = self.grid_config
        x_barrier = np.log(barrier_params.barrier / market.K)
        x_grid = self._build_aligned_grid(config, x_barrier)
        _, _, _, t_eff = compute_transform_params(market)

        u = bs_to_heat_initial(x_grid, market, config, option_type)

        if barrier_params.barrier_type.startswith("down"):
            barrier_mask = x_grid <= x_barrier + 1e-14
        else:
            barrier_mask = x_grid >= x_barrier - 1e-14

        u = np.where(barrier_mask, 0.0, u)

        # Balance temporal accuracy vs Gibbs artifacts from barrier projection.
        # sqrt(N) scaling: enough steps for convergence without excess ringing.
        n_internal = max(n_steps, int(10 * np.sqrt(config.N)))
        dt = t_eff / n_internal

        for _ in range(n_internal):
            u = self.chernoff.apply(u, x_grid, dt)
            u = np.where(barrier_mask, 0.0, u)

        ko_price = max(0.0, extract_price_at_spot(u, x_grid, market))
        vanilla_result = self._european.price(market, n_steps=n_steps, option_type=option_type)

        return BarrierPricingResult(
            price=ko_price,
            vanilla_price=vanilla_result.price,
            knockout_price=ko_price,
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
        barrier_params: BarrierParams,
        n_steps: int,
        option_type: str,
    ) -> BarrierPricingResult:
        out_type = barrier_params.barrier_type.replace("_in", "_out")
        out_params = BarrierParams(
            barrier=barrier_params.barrier,
            barrier_type=out_type,
            rebate=barrier_params.rebate,
        )

        ko_result = self._price_knockout(market, out_params, n_steps, option_type)
        ki_price = max(0.0, ko_result.vanilla_price - ko_result.knockout_price)

        return BarrierPricingResult(
            price=ki_price,
            vanilla_price=ko_result.vanilla_price,
            knockout_price=ko_result.knockout_price,
            barrier_type=barrier_params.barrier_type,
            method_name=self.chernoff.name,
            n_steps=n_steps,
            market=market,
            barrier_params=barrier_params,
            certificate=None,
        )
