"""American option pricer via Chernoff projection onto intrinsic payoff."""

from __future__ import annotations

import numpy as np

from .european import EuropeanPricer
from .transforms import (
    bs_to_heat_initial,
    compute_transform_params,
    extract_price_at_spot,
    make_grid,
)
from .validation import AmericanPricingResult, GridConfig, MarketParams


class AmericanPricer:
    """Price American options with Chernoff steps and payoff projection."""

    def __init__(self, chernoff, grid_config: GridConfig | None = None):
        self.chernoff = chernoff
        self.grid_config = grid_config if grid_config is not None else GridConfig()
        self._european = EuropeanPricer(chernoff, self.grid_config)

    def price(
        self,
        market: MarketParams,
        n_steps: int = 100,
        option_type: str = "put",
        return_boundary: bool = False,
    ) -> AmericanPricingResult:
        """Compute American option price in Black-Scholes setting."""
        if option_type not in {"call", "put"}:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        euro = self._european.price(market, n_steps=n_steps, option_type=option_type)

        # Without dividends, early exercise is suboptimal for calls.
        if option_type == "call":
            return AmericanPricingResult(
                price=euro.price,
                european_price=euro.price,
                early_exercise_premium=0.0,
                exercise_boundary=None,
                option_type=option_type,
                method_name=self.chernoff.name,
                n_steps=n_steps,
                market=market,
                certificate=euro.certificate,
            )

        cfg = self.grid_config
        x_grid = make_grid(cfg)
        s_grid = market.K * np.exp(x_grid)
        _, alpha, beta, t_eff = compute_transform_params(market)

        # Heat-equation initial condition (tau=0) from payoff.
        u = bs_to_heat_initial(x_grid, market, cfg, option_type)

        payoff_physical = np.maximum(market.K - s_grid, 0.0)

        dt = t_eff / n_steps
        boundaries = [] if return_boundary else None

        for step in range(n_steps):
            u = self.chernoff.apply(u, x_grid, dt)

            tau = (step + 1) * dt
            exponent = np.clip(-alpha * x_grid - beta * tau, -700.0, 700.0)
            payoff_heat_tau = (payoff_physical / market.K) * np.exp(exponent)

            # Projection step: American value cannot be below intrinsic payoff.
            u = np.maximum(u, payoff_heat_tau)

            if boundaries is not None:
                close = np.isclose(u, payoff_heat_tau, rtol=1e-6, atol=1e-10)
                put_itm = s_grid <= market.K
                mask = close & put_itm
                if np.any(mask):
                    boundaries.append(float(np.max(s_grid[mask])))
                else:
                    boundaries.append(0.0)

        amer_price = max(0.0, extract_price_at_spot(u, x_grid, market))

        intrinsic = max(market.K - market.S, 0.0)
        amer_price = max(amer_price, euro.price, intrinsic)
        eep = amer_price - euro.price

        boundary_arr = np.array(boundaries, dtype=float) if boundaries is not None else None

        return AmericanPricingResult(
            price=amer_price,
            european_price=euro.price,
            early_exercise_premium=eep,
            exercise_boundary=boundary_arr,
            option_type=option_type,
            method_name=self.chernoff.name,
            n_steps=n_steps,
            market=market,
            certificate=euro.certificate,
        )
