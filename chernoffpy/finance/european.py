"""European option pricer using Chernoff approximation of the heat semigroup."""

import numpy as np

from chernoffpy.functions import ChernoffFunction
from chernoffpy.semigroups import HeatSemigroup

from .validation import (
    GridConfig,
    MarketParams,
    PricingResult,
    ValidationCertificate,
)
from .transforms import (
    bs_exact_price,
    bs_to_heat_initial,
    compute_transform_params,
    extract_price_at_spot,
    make_grid,
)


class EuropeanPricer:
    """Price European options via Chernoff approximation.

    Workflow:
        1. Transform BS payoff to heat equation initial condition (Wilmott substitution)
        2. Apply cosine taper for FFT compatibility
        3. Solve heat equation via chernoff.compose(u0, x_grid, t_eff, n_steps)
        4. Transform back to BS price
        5. Validate against FFT exact + BS exact -> ValidationCertificate
    """

    def __init__(
        self,
        chernoff: ChernoffFunction,
        grid_config: GridConfig | None = None,
    ):
        self.chernoff = chernoff
        self.grid_config = grid_config if grid_config is not None else GridConfig()

    def _solve(
        self, market: MarketParams, n_steps: int, option_type: str
    ) -> dict:
        """Internal solver returning all intermediate values for reuse by greeks."""
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        config = self.grid_config
        x_grid = make_grid(config)
        k, alpha, beta, t_eff = compute_transform_params(market)

        u0 = bs_to_heat_initial(x_grid, market, config, option_type)
        u_final = self.chernoff.compose(u0, x_grid, t_eff, n_steps)
        price = extract_price_at_spot(u_final, x_grid, market)

        return {
            "x_grid": x_grid,
            "u0": u0,
            "u_final": u_final,
            "price": price,
            "alpha": alpha,
            "beta": beta,
            "t_eff": t_eff,
            "config": config,
        }

    def price(
        self,
        market: MarketParams,
        n_steps: int = 50,
        option_type: str = "call",
    ) -> PricingResult:
        """Price a European option with accuracy certificate.

        Parameters:
            market: Market parameters (S, K, T, r, sigma)
            n_steps: Number of Chernoff composition steps
            option_type: "call" or "put"

        Returns:
            PricingResult containing price, method info, and ValidationCertificate
        """
        sol = self._solve(market, n_steps, option_type)

        # FFT exact solution on same grid with same initial condition
        u_exact_fft = HeatSemigroup.solve_fourier(
            sol["u0"], sol["x_grid"], sol["t_eff"]
        )
        fft_price = extract_price_at_spot(u_exact_fft, sol["x_grid"], market)

        # Exact Black-Scholes analytical price
        bs_price = bs_exact_price(market, option_type)

        # Floor at zero: FFT artifacts can produce tiny negative values for deep OTM
        computed_price = max(0.0, sol["price"])
        fft_price = max(0.0, fft_price)
        certificate = ValidationCertificate(
            bs_price=bs_price,
            computed_price=computed_price,
            abs_error=abs(computed_price - bs_price),
            rel_error=(
                abs(computed_price - bs_price) / bs_price
                if bs_price > 1e-10
                else float("inf")
            ),
            chernoff_error=abs(computed_price - fft_price),
            domain_error=abs(fft_price - bs_price),
        )

        return PricingResult(
            price=computed_price,
            method_name=self.chernoff.name,
            n_steps=n_steps,
            market=market,
            certificate=certificate,
        )
