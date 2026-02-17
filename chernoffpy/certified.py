"""Certified error bounds for Chernoff approximations.

Implements practical upper bounds inspired by convergence-rate estimates
for Chernoff product formulas (Galkin-Remizov, 2025, Israel J. Math.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ChernoffOrder:
    """Theoretical convergence order of a Chernoff method."""

    k: int
    name: str
    is_exact: bool = True

    @staticmethod
    def from_chernoff(chernoff: Any) -> "ChernoffOrder":
        """Infer formal order from a Chernoff implementation."""
        if hasattr(chernoff, "order"):
            try:
                k = int(chernoff.order)
                return ChernoffOrder(k=max(k, 1), name=chernoff.__class__.__name__)
            except Exception:
                pass

        name = chernoff.__class__.__name__.lower()
        if "crank" in name or "nicolson" in name or "cn" in name:
            return ChernoffOrder(k=2, name="Crank-Nicolson")
        if "strang" in name:
            return ChernoffOrder(k=2, name="Strang splitting")
        if "pade" in name:
            if "33" in name:
                return ChernoffOrder(k=6, name="Pad\u00e9 [3/3]")
            if "22" in name:
                return ChernoffOrder(k=4, name="Pad\u00e9 [2/2]")
            return ChernoffOrder(k=4, name="Pad\u00e9")
        if "backward" in name or "euler" in name:
            return ChernoffOrder(k=1, name="Backward Euler")

        return ChernoffOrder(k=1, name=chernoff.__class__.__name__, is_exact=False)


@dataclass(frozen=True)
class PayoffRegularity:
    """Effective payoff smoothness class used in rate estimates."""

    k_f: int
    name: str

    @staticmethod
    def vanilla_call() -> "PayoffRegularity":
        """Vanilla call/put: payoff in H^2 (kink at strike limits regularity)."""
        return PayoffRegularity(k_f=2, name="vanilla call/put")

    @staticmethod
    def digital() -> "PayoffRegularity":
        """Digital option: discontinuous payoff, k_f=0 (no Sobolev regularity)."""
        return PayoffRegularity(k_f=0, name="digital")

    @staticmethod
    def smooth(k: int = 100) -> "PayoffRegularity":
        """Smooth payoff (e.g., power options): arbitrarily high regularity."""
        return PayoffRegularity(k_f=max(k, 1), name="smooth payoff")

    @staticmethod
    def barrier() -> "PayoffRegularity":
        """Barrier option via DST: Dirichlet BCs restore H^2 regularity."""
        return PayoffRegularity(k_f=2, name="barrier (DST)")

    @staticmethod
    def barrier_fft() -> "PayoffRegularity":
        """Barrier option via FFT+projection: Gibbs artifacts destroy regularity."""
        return PayoffRegularity(k_f=0, name="barrier (FFT+projection)")


@dataclass(frozen=True)
class CertifiedBound:
    """Certified (or conservative) upper bound for pricing error."""

    bound: float
    effective_order: int
    n_steps: int
    method: str
    constant_B: float
    safety_factor: float
    is_certified: bool
    reference: str = "Galkin-Remizov (2025), Israel J. Math."


def effective_order(chernoff_order: ChernoffOrder, payoff_reg: PayoffRegularity) -> int:
    """Effective algebraic order: min(method order, payoff regularity)."""
    return min(chernoff_order.k, payoff_reg.k_f)


def compute_certified_bound(
    prices: dict[int, float],
    chernoff_order: ChernoffOrder,
    payoff_reg: PayoffRegularity,
    n_target: int,
    safety_factor: float = 2.0,
    exact_price: float | None = None,
) -> CertifiedBound:
    """Compute conservative upper bound for |V_n - V_exact|.

    Uses exact-comparison mode when exact price is known, otherwise
    Richardson self-convergence mode.
    """
    if len(prices) < 2:
        raise ValueError("Need at least 2 price computations")
    if n_target < 1:
        raise ValueError("n_target must be >= 1")
    if safety_factor < 1.0:
        raise ValueError("safety_factor must be >= 1.0")

    p_eff = effective_order(chernoff_order, payoff_reg)

    if p_eff == 0:
        ns = sorted(prices.keys())
        n1, n2 = ns[-2], ns[-1]
        diff = abs(prices[n1] - prices[n2])
        return CertifiedBound(
            bound=float(safety_factor * diff),
            effective_order=0,
            n_steps=n_target,
            method="difference (no algebraic convergence)",
            constant_B=float(diff),
            safety_factor=float(safety_factor),
            is_certified=False,
        )

    if exact_price is not None:
        b_values = [abs(v - exact_price) * (n ** p_eff) for n, v in prices.items()]
        b_values = [b for b in b_values if np.isfinite(b)]
        if not b_values:
            return CertifiedBound(
                bound=0.0,
                effective_order=p_eff,
                n_steps=n_target,
                method="exact match",
                constant_B=0.0,
                safety_factor=float(safety_factor),
                is_certified=True,
            )
        b_const = float(max(b_values))
        method = "exact comparison"

        # Tight bound at n_target: use its own observed error directly
        # instead of inflated max(B) / n_target^p which over-estimates
        # when domain truncation error dominates at larger n.
        if n_target in prices:
            bound = float(safety_factor * abs(prices[n_target] - exact_price))
        else:
            bound = float(safety_factor * b_const / (n_target ** p_eff))

        return CertifiedBound(
            bound=bound,
            effective_order=p_eff,
            n_steps=n_target,
            method=method,
            constant_B=b_const,
            safety_factor=float(safety_factor),
            is_certified=True,
        )
    else:
        sorted_n = sorted(prices.keys())
        b_values: list[float] = []
        for i in range(len(sorted_n) - 1):
            n1 = sorted_n[i]
            n2 = sorted_n[i + 1]
            diff = abs(prices[n1] - prices[n2])
            ratio = (n1 / n2) ** p_eff
            denom = abs(1.0 - ratio)
            if denom < 1e-14:
                continue
            err_est = diff / denom
            b_values.append(float(err_est * (n1 ** p_eff)))

        if not b_values:
            return CertifiedBound(
                bound=0.0,
                effective_order=p_eff,
                n_steps=n_target,
                method="self-convergence (zero diff)",
                constant_B=0.0,
                safety_factor=float(safety_factor),
                is_certified=False,
            )

        b_const = float(max(b_values))
        method = "Richardson self-convergence"

    # Richardson mode cannot detect domain truncation error,
    # so the bound is not certified.
    bound = float(safety_factor * b_const / (n_target ** p_eff))

    return CertifiedBound(
        bound=bound,
        effective_order=p_eff,
        n_steps=n_target,
        method=method,
        constant_B=b_const,
        safety_factor=float(safety_factor),
        is_certified=False,
    )


def verify_convergence_order(
    prices: dict[int, float],
    expected_order: int,
    exact_price: float | None = None,
    tolerance: float = 0.3,
) -> dict[str, Any]:
    """Estimate empirical order and compare with expected theory order."""
    if len(prices) < 2:
        return {
            "empirical_order": None,
            "expected_order": expected_order,
            "is_consistent": False,
            "details": "Need at least 2 price points",
        }

    sorted_n = sorted(prices.keys())

    if exact_price is not None:
        errs = [(n, abs(prices[n] - exact_price)) for n in sorted_n]
    else:
        finest_n = sorted_n[-1]
        errs = [(n, abs(prices[n] - prices[finest_n])) for n in sorted_n[:-1]]

    errs = [(n, e) for n, e in errs if e > 1e-14]
    if len(errs) < 2:
        return {
            "empirical_order": None,
            "expected_order": expected_order,
            "is_consistent": False,
            "details": "Insufficient non-zero errors",
        }

    # Filter plateau: when errors stop decreasing (domain error floor),
    # exclude plateaued points so the regression reflects true convergence.
    if len(errs) >= 3:
        filtered = [errs[0]]
        for i in range(1, len(errs)):
            prev_err = filtered[-1][1]
            curr_err = errs[i][1]
            # Keep point only if error decreased by at least 20%
            if curr_err < 0.8 * prev_err:
                filtered.append(errs[i])
        if len(filtered) >= 2:
            errs = filtered

    if len(errs) < 2:
        return {
            "empirical_order": None,
            "expected_order": expected_order,
            "is_consistent": False,
            "details": "All errors plateaued (domain error floor)",
        }

    log_n = np.log([n for n, _ in errs])
    log_e = np.log([e for _, e in errs])
    slope, _ = np.polyfit(log_n, log_e, 1)
    empirical = float(-slope)
    # One-sided: the theorem gives a lower bound on convergence rate,
    # so empirical order higher than expected is correct behaviour.
    consistent = empirical >= expected_order - tolerance

    return {
        "empirical_order": empirical,
        "expected_order": expected_order,
        "is_consistent": consistent,
        "details": (
            f"Empirical order {empirical:.3f} vs expected {expected_order}; "
            f"tolerance {tolerance:.3f}"
        ),
    }


def n_steps_for_tolerance(
    target_error: float,
    constant_B: float,
    effective_order: int,
    safety_factor: float = 2.0,
) -> int:
    """Recommend n so that safety_factor * B / n^p <= target_error."""
    if target_error <= 0:
        raise ValueError("target_error must be > 0")
    if effective_order <= 0:
        raise ValueError("effective_order must be >= 1")
    if safety_factor < 1.0:
        raise ValueError("safety_factor must be >= 1.0")
    if constant_B <= 0:
        return 1

    n = (safety_factor * constant_B / target_error) ** (1.0 / effective_order)
    return max(1, int(np.ceil(n)))
