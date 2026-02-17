"""Barrier option pricing via Chernoff multipliers in DST-I basis.

This module provides Dirichlet-consistent alternatives to FFT+projection
barrier pricers. The solution is evolved on interior nodes of a finite domain
using DST-I, where boundary values are implicitly zero.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dst, idst

from .european import EuropeanPricer
from .transforms import compute_transform_params
from .validation import (
    BarrierParams,
    BarrierPricingResult,
    DoubleBarrierParams,
    DoubleBarrierPricingResult,
    GridConfig,
    MarketParams,
)

_MAX_EXP = 700.0


class _DSTStepper:
    """Shared utilities for DST-based Chernoff stepping."""

    def __init__(self, chernoff) -> None:
        self.chernoff = chernoff

    def _chernoff_multiplier(self, eigenvalues: np.ndarray, dt: float) -> np.ndarray:
        """Return Chernoff multiplier for arbitrary Laplacian eigenvalues."""
        if hasattr(self.chernoff, "multiplier"):
            return self.chernoff.multiplier(eigenvalues, dt)

        z = eigenvalues * dt
        name = self.chernoff.__class__.__name__.lower()
        if "crank" in name or "nicolson" in name:
            return (1.0 - 0.5 * z) / (1.0 + 0.5 * z)
        if "pade" in name:
            return (1.0 - 0.5 * z + z * z / 12.0) / (1.0 + 0.5 * z + z * z / 12.0)
        return 1.0 / (1.0 + z)

    @staticmethod
    def _apply_dst_step(u: np.ndarray, multiplier: np.ndarray, n_nodes: int) -> np.ndarray:
        """Single DST-I step on interior nodes with inverse scaling."""
        u_hat = dst(u, type=1)
        u_hat *= multiplier
        return idst(u_hat, type=1)


class BarrierDSTPricer(_DSTStepper):
    """Single-barrier pricing via DST-I on Dirichlet domains."""

    def __init__(self, chernoff, grid_config: GridConfig | None = None):
        super().__init__(chernoff)
        self.grid_config = grid_config if grid_config is not None else GridConfig()
        self._european = EuropeanPricer(chernoff, self.grid_config)

    def price(
        self,
        market: MarketParams,
        barrier_params: BarrierParams,
        n_steps: int = 50,
        option_type: str = "call",
    ) -> BarrierPricingResult:
        """Price a single-barrier option using DST-I Chernoff stepping.

        Uses n_internal = max(n_steps, 10*sqrt(N)) time steps to balance
        temporal accuracy with the spatial resolution of the DST grid.
        """
        if option_type not in {"call", "put"}:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        self._validate(market, barrier_params)

        if barrier_params.barrier_type.endswith("_out"):
            return self._price_knockout(market, barrier_params, n_steps, option_type)
        return self._price_knockin(market, barrier_params, n_steps, option_type)

    @staticmethod
    def _validate(market: MarketParams, barrier_params: BarrierParams) -> None:
        b = barrier_params.barrier
        s = market.S
        bt = barrier_params.barrier_type

        if bt.startswith("down") and b >= s:
            raise ValueError(f"Down barrier ({b}) must be < spot ({s})")
        if bt.startswith("up") and b <= s:
            raise ValueError(f"Up barrier ({b}) must be > spot ({s})")

    def _build_domain(self, market: MarketParams, barrier_params: BarrierParams) -> tuple[float, float]:
        x_barrier = np.log(barrier_params.barrier / market.K)
        if barrier_params.barrier_type.startswith("down"):
            x_left = x_barrier
            x_right = self.grid_config.L
        else:
            x_left = -self.grid_config.L
            x_right = x_barrier
        if x_right <= x_left:
            raise ValueError(
                "Invalid DST domain. Increase grid_config.L so barrier lies strictly inside [-L, L]."
            )
        return float(x_left), float(x_right)

    def _initial_condition(
        self,
        x_internal: np.ndarray,
        market: MarketParams,
        alpha: float,
        option_type: str,
    ) -> np.ndarray:
        s_grid = market.K * np.exp(x_internal)
        if option_type == "call":
            payoff = np.maximum(s_grid - market.K, 0.0)
        else:
            payoff = np.maximum(market.K - s_grid, 0.0)

        exponent = np.clip(-alpha * x_internal, -_MAX_EXP, _MAX_EXP)
        with np.errstate(over="ignore", invalid="ignore"):
            u0 = np.exp(exponent) * (payoff / market.K)
        return np.nan_to_num(u0, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_far_boundary_taper(
        self,
        u: np.ndarray,
        x_internal: np.ndarray,
        x_left: float,
        x_right: float,
        barrier_type: str,
    ) -> np.ndarray:
        tw = min(self.grid_config.taper_width, 0.49 * (x_right - x_left))
        taper = np.ones_like(u)
        if barrier_type.startswith("down"):
            d = x_right - x_internal
            mask = d < tw
            t = d[mask] / tw
            taper[mask] = 0.5 * (1.0 - np.cos(np.pi * t))
        else:
            d = x_internal - x_left
            mask = d < tw
            t = d[mask] / tw
            taper[mask] = 0.5 * (1.0 - np.cos(np.pi * t))
        return u * taper

    def _price_knockout(
        self,
        market: MarketParams,
        barrier_params: BarrierParams,
        n_steps: int,
        option_type: str,
    ) -> BarrierPricingResult:
        x_left, x_right = self._build_domain(market, barrier_params)
        _, alpha, beta, t_eff = compute_transform_params(market)

        n_nodes = self.grid_config.N
        dx = (x_right - x_left) / n_nodes
        x_internal = x_left + np.arange(1, n_nodes) * dx

        u = self._initial_condition(x_internal, market, alpha, option_type)
        u = self._apply_far_boundary_taper(
            u,
            x_internal,
            x_left,
            x_right,
            barrier_params.barrier_type,
        )

        k = np.arange(1, n_nodes)
        eigenvalues = (k * np.pi / (x_right - x_left)) ** 2
        n_internal = max(n_steps, int(10 * np.sqrt(n_nodes)))
        dt = t_eff / n_internal
        multiplier = self._chernoff_multiplier(eigenvalues, dt)

        for _ in range(n_internal):
            u = self._apply_dst_step(u, multiplier, n_nodes)

        exponent = np.clip(alpha * x_internal + beta * t_eff, -_MAX_EXP, _MAX_EXP)
        with np.errstate(over="ignore", invalid="ignore"):
            v_internal = market.K * np.exp(exponent) * u
        v_internal = np.nan_to_num(v_internal, nan=0.0, posinf=0.0, neginf=0.0)

        x0 = float(np.log(market.S / market.K))
        if not (x_left <= x0 <= x_right):
            raise ValueError("Spot is outside DST pricing domain")

        ko_price = float(np.interp(x0, x_internal, v_internal))
        ko_price = max(0.0, ko_price)

        vanilla_result = self._european.price(market, n_steps=n_steps, option_type=option_type)

        return BarrierPricingResult(
            price=ko_price,
            vanilla_price=vanilla_result.price,
            knockout_price=ko_price,
            barrier_type=barrier_params.barrier_type,
            method_name=f"DST-{self.chernoff.name}",
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
        out_params = BarrierParams(
            barrier=barrier_params.barrier,
            barrier_type=barrier_params.barrier_type.replace("_in", "_out"),
            rebate=barrier_params.rebate,
        )
        ko_result = self._price_knockout(market, out_params, n_steps, option_type)
        ki_price = max(0.0, ko_result.vanilla_price - ko_result.knockout_price)

        return BarrierPricingResult(
            price=ki_price,
            vanilla_price=ko_result.vanilla_price,
            knockout_price=ko_result.knockout_price,
            barrier_type=barrier_params.barrier_type,
            method_name=f"DST-{self.chernoff.name}",
            n_steps=n_steps,
            market=market,
            barrier_params=barrier_params,
            certificate=None,
        )


class DoubleBarrierDSTPricer(_DSTStepper):
    """Double-barrier pricing via DST-I on the corridor [xL, xU]."""

    def __init__(self, chernoff, grid_config: GridConfig | None = None):
        super().__init__(chernoff)
        self.grid_config = grid_config if grid_config is not None else GridConfig()
        self._european = EuropeanPricer(chernoff, self.grid_config)

    @staticmethod
    def _validate(market: MarketParams, params: DoubleBarrierParams) -> None:
        s = market.S
        if s <= params.lower_barrier:
            raise ValueError(f"Spot ({s}) must be > lower barrier ({params.lower_barrier})")
        if s >= params.upper_barrier:
            raise ValueError(f"Spot ({s}) must be < upper barrier ({params.upper_barrier})")

    def price(
        self,
        market: MarketParams,
        barrier_params: DoubleBarrierParams,
        n_steps: int = 50,
        option_type: str = "call",
    ) -> DoubleBarrierPricingResult:
        """Price a double-barrier option using DST-I on the corridor [xL, xU].

        Both barriers are enforced via Dirichlet BCs (sine transform).
        """
        if option_type not in {"call", "put"}:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        self._validate(market, barrier_params)

        if barrier_params.barrier_type == "double_knock_out":
            return self._price_knockout(market, barrier_params, n_steps, option_type)
        return self._price_knockin(market, barrier_params, n_steps, option_type)

    def _price_knockout(
        self,
        market: MarketParams,
        barrier_params: DoubleBarrierParams,
        n_steps: int,
        option_type: str,
    ) -> DoubleBarrierPricingResult:
        x_left = float(np.log(barrier_params.lower_barrier / market.K))
        x_right = float(np.log(barrier_params.upper_barrier / market.K))
        if x_right <= x_left:
            raise ValueError("Invalid double-barrier domain")

        _, alpha, beta, t_eff = compute_transform_params(market)

        n_nodes = self.grid_config.N
        dx = (x_right - x_left) / n_nodes
        x_internal = x_left + np.arange(1, n_nodes) * dx

        s_grid = market.K * np.exp(x_internal)
        if option_type == "call":
            payoff = np.maximum(s_grid - market.K, 0.0)
        else:
            payoff = np.maximum(market.K - s_grid, 0.0)

        exponent0 = np.clip(-alpha * x_internal, -_MAX_EXP, _MAX_EXP)
        with np.errstate(over="ignore", invalid="ignore"):
            u = np.exp(exponent0) * (payoff / market.K)
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)

        k = np.arange(1, n_nodes)
        eigenvalues = (k * np.pi / (x_right - x_left)) ** 2
        n_internal = max(n_steps, int(10 * np.sqrt(n_nodes)))
        dt = t_eff / n_internal
        multiplier = self._chernoff_multiplier(eigenvalues, dt)

        for _ in range(n_internal):
            u = self._apply_dst_step(u, multiplier, n_nodes)

        exponent = np.clip(alpha * x_internal + beta * t_eff, -_MAX_EXP, _MAX_EXP)
        with np.errstate(over="ignore", invalid="ignore"):
            v_internal = market.K * np.exp(exponent) * u
        v_internal = np.nan_to_num(v_internal, nan=0.0, posinf=0.0, neginf=0.0)

        x0 = float(np.log(market.S / market.K))
        if not (x_left <= x0 <= x_right):
            raise ValueError("Spot is outside double-barrier DST domain")

        dko_price = float(np.interp(x0, x_internal, v_internal))
        dko_price = max(0.0, dko_price)

        vanilla_result = self._european.price(market, n_steps=n_steps, option_type=option_type)

        return DoubleBarrierPricingResult(
            price=dko_price,
            vanilla_price=vanilla_result.price,
            knockout_price=dko_price,
            barrier_type=barrier_params.barrier_type,
            method_name=f"DST-{self.chernoff.name}",
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
        ki_price = max(0.0, ko_result.vanilla_price - ko_result.knockout_price)

        return DoubleBarrierPricingResult(
            price=ki_price,
            vanilla_price=ko_result.vanilla_price,
            knockout_price=ko_result.knockout_price,
            barrier_type=barrier_params.barrier_type,
            method_name=f"DST-{self.chernoff.name}",
            n_steps=n_steps,
            market=market,
            barrier_params=barrier_params,
            certificate=None,
        )

