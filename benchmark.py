"""Simple benchmark runner for ChernoffPy finance pricers."""

from __future__ import annotations

import time

import numpy as np


def bench(func, *args, n_runs: int = 8, warmup: int = 2, **kwargs) -> dict[str, float]:
    """Measure average runtime in milliseconds."""
    for _ in range(warmup):
        func(*args, **kwargs)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
    }


def run_benchmarks() -> None:
    from chernoffpy import CrankNicolson
    from chernoffpy.accel import HAS_NUMBA
    from chernoffpy.finance import AmericanPricer, BarrierParams, BarrierPricer, EuropeanPricer, MarketParams
    from chernoffpy.finance.heston import HestonPricer
    from chernoffpy.finance.heston_fast import HestonFastPricer
    from chernoffpy.finance.heston_params import HestonGridConfig, HestonParams

    cn = CrankNicolson()
    market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    barrier = BarrierParams(barrier=90.0, barrier_type="down_and_out")
    heston = HestonParams(
        S=100,
        K=100,
        T=1.0,
        r=0.05,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
    )
    heston_grid = HestonGridConfig(n_x=192, n_v=64, x_min=-4.0, x_max=4.0, v_max=0.8)

    results: dict[str, dict[str, float]] = {}

    ep = EuropeanPricer(cn)
    bp = BarrierPricer(cn)
    ap = AmericanPricer(cn)
    hp = HestonPricer(cn, heston_grid)
    hfp = HestonFastPricer(cn, heston_grid)

    results["European CN n=50"] = bench(ep.price, market, n_steps=50, option_type="call")
    results["Barrier DOC n=50"] = bench(bp.price, market, barrier, n_steps=50, option_type="call")
    results["American put n=80"] = bench(ap.price, market, n_steps=80, option_type="put")
    results["Heston n=35"] = bench(hp.price, heston, n_steps=35, option_type="call", n_runs=4)
    results["Heston Fast n=35"] = bench(hfp.price, heston, n_steps=35, option_type="call", n_runs=4)

    print("=" * 64)
    print("ChernoffPy Benchmarks")
    print("=" * 64)
    for name, stats in results.items():
        print(f"{name:24s}  {stats['mean_ms']:9.2f} ms  (+/- {stats['std_ms']:.2f})")

    speedup = results["Heston n=35"]["mean_ms"] / max(results["Heston Fast n=35"]["mean_ms"], 1e-12)
    print(f"\nHeston speedup: {speedup:.2f}x")
    print(f"Numba available: {HAS_NUMBA}")


if __name__ == "__main__":
    run_benchmarks()
