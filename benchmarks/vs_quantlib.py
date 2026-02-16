"""
Benchmark: ChernoffPy vs QuantLib.

Измеряет точность и скорость для одинаковых конфигураций.
Генерирует таблицы (CSV) и графики (PNG) для статьи.

Запуск:
    cd E:\MarkovChains\ChernoffPy
    pip install QuantLib-Python
    python benchmarks/vs_quantlib.py
"""

from __future__ import annotations

import sys
import time
import csv
import os
from pathlib import Path

import numpy as np

# Add parent directory to path for importing chernoffpy
sys.path.insert(0, str(Path(__file__).parent.parent))

from chernoffpy import CrankNicolson, BackwardEuler, PadeChernoff
from chernoffpy.finance import (
    EuropeanPricer,
    AmericanPricer,
    BarrierDSTPricer,
    DoubleBarrierDSTPricer,
    CertifiedEuropeanPricer,
    MarketParams,
    BarrierParams,
    DoubleBarrierParams,
    GridConfig,
)
from chernoffpy.finance.european import bs_exact_price
from chernoffpy.finance.barrier_analytical import barrier_analytical
from chernoffpy.finance.double_barrier_analytical import double_barrier_analytical
from chernoffpy.finance.american_analytical import american_binomial
from chernoffpy.finance.heston_fast import HestonFastPricer
from chernoffpy.finance.heston_params import HestonParams, HestonGridConfig
from chernoffpy.finance.heston_analytical import heston_price

from helpers import (
    ql_european_fdm,
    ql_european_analytical,
    ql_american_fdm,
    ql_american_crr,
    ql_barrier_analytical,
    ql_barrier_fdm,
    ql_double_barrier_fdm,
    ql_heston_analytical,
    ql_heston_fdm,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Timing utility
# ═══════════════════════════════════════════════════════════════════════════════

def measure(func, *args, n_runs: int = 10, warmup: int = 2, **kwargs):
    """Замерить время (медиана из n_runs)."""
    # Warmup runs
    for _ in range(warmup):
        result = func(*args, **kwargs)
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    return {
        "price": result,
        "time_ms": np.median(times) * 1000,
        "time_std_ms": np.std(times) * 1000,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Table 1: European Options
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_european():
    """
    ChernoffPy vs QuantLib для European call/put.
    
    Exact = BS analytical (обе библиотеки должны совпадать).
    Сравниваем FDM-engine QuantLib vs Chernoff iterations.
    """
    print("  Setting up European benchmark...")
    
    market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    exact = bs_exact_price(market, "call")
    
    results = []
    
    # ChernoffPy: BE, CN, Padé × n_steps
    for scheme_name, scheme in [
        ("ChernoffPy BE", BackwardEuler()),
        ("ChernoffPy CN", CrankNicolson()),
        ("ChernoffPy Padé", PadeChernoff(1, 2)),  # A-stable [1/2]
    ]:
        pricer = EuropeanPricer(scheme)
        for n in [10, 20, 50, 100, 200]:
            m = measure(pricer.price, market, n, "call", n_runs=10)
            error_pct = abs(m["price"].price - exact) / exact * 100
            results.append({
                "method": scheme_name,
                "n_steps": n,
                "price": m["price"].price,
                "error_pct": error_pct,
                "time_ms": m["time_ms"],
                "efficiency": error_pct / max(m["time_ms"], 0.01),
            })
    
    # QuantLib FDM: n_time × n_spot
    for n_time in [10, 20, 50, 100, 200, 500]:
        for n_spot in [100, 200]:
            m = measure(
                ql_european_fdm, 100, 100, 1.0, 0.05, 0.20,
                "call", n_time, n_spot,
                n_runs=10,
            )
            error_pct = abs(m["price"] - exact) / exact * 100
            results.append({
                "method": f"QuantLib FDM ({n_spot}pts)",
                "n_steps": n_time,
                "price": m["price"],
                "error_pct": error_pct,
                "time_ms": m["time_ms"],
                "efficiency": error_pct / max(m["time_ms"], 0.01),
            })
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Table 2: Barrier Options
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_barrier():
    """
    ChernoffPy DST vs QuantLib FDM для barrier options.
    
    Exact = Reiner-Rubinstein analytical.
    Ключевой результат: DST eliminates Gibbs ringing.
    """
    print("  Setting up Barrier benchmark...")
    
    market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    
    configs = [
        ("DOC B=90", BarrierParams(barrier=90, barrier_type="down_and_out"), "call", "DownOut"),
        ("DOC B=99", BarrierParams(barrier=99, barrier_type="down_and_out"), "call", "DownOut"),
        ("UOC B=120", BarrierParams(barrier=120, barrier_type="up_and_out"), "call", "UpOut"),
    ]
    
    results = []
    
    for name, bp, opt_type, ql_bt in configs:
        exact = barrier_analytical(market, bp, opt_type)
        
        # ChernoffPy DST
        pricer = BarrierDSTPricer(CrankNicolson())
        for n in [20, 50, 100, 200]:
            m = measure(pricer.price, market, bp, n, opt_type, n_runs=10)
            error_pct = abs(m["price"].price - exact) / max(exact, 1e-10) * 100
            results.append({
                "config": name,
                "method": "ChernoffPy DST",
                "n_steps": n,
                "price": m["price"].price,
                "exact": exact,
                "error_pct": error_pct,
                "time_ms": m["time_ms"],
            })
        
        # QuantLib FDM
        for n_time in [50, 100, 200, 500, 1000]:
            for n_spot in [200, 500]:
                m = measure(
                    ql_barrier_fdm, 100, 100, bp.barrier,
                    1.0, 0.05, 0.20, opt_type, ql_bt, n_time, n_spot,
                    n_runs=10,
                )
                error_pct = abs(m["price"] - exact) / max(exact, 1e-10) * 100
                results.append({
                    "config": name,
                    "method": f"QuantLib FDM ({n_spot}pts)",
                    "n_steps": n_time,
                    "price": m["price"],
                    "exact": exact,
                    "error_pct": error_pct,
                    "time_ms": m["time_ms"],
                })
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Table 3: American Options
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_american():
    """
    ChernoffPy vs QuantLib FDM для American put.
    
    Exact = CRR binomial n=50000 (quasi-exact).
    """
    print("  Setting up American benchmark...")
    
    market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    
    # Use QuantLib CRR as quasi-exact reference
    print("  Computing quasi-exact American price (CRR n=50000)...")
    exact = ql_american_crr(100, 100, 1.0, 0.05, 0.20, "put", n=50000)
    print(f"  Quasi-exact price: {exact:.6f}")
    
    results = []
    
    # ChernoffPy
    pricer = AmericanPricer(CrankNicolson())
    for n in [20, 50, 100, 200]:
        m = measure(pricer.price, market, n, "put", n_runs=10)
        error_pct = abs(m["price"].price - exact) / exact * 100
        results.append({
            "method": "ChernoffPy CN",
            "n_steps": n,
            "price": m["price"].price,
            "error_pct": error_pct,
            "time_ms": m["time_ms"],
        })
    
    # QuantLib FDM
    for n_time in [50, 100, 200, 500]:
        for n_spot in [200, 500]:
            m = measure(
                ql_american_fdm, 100, 100, 1.0, 0.05, 0.20,
                "put", n_time, n_spot,
                n_runs=10,
            )
            error_pct = abs(m["price"] - exact) / exact * 100
            results.append({
                "method": f"QuantLib FDM ({n_spot}pts)",
                "n_steps": n_time,
                "price": m["price"],
                "error_pct": error_pct,
                "time_ms": m["time_ms"],
            })
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Table 4: Heston
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_heston():
    """
    ChernoffPy HestonFastPricer vs QuantLib FdHeston.
    
    Exact = Lewis CF quadrature (оба: ChernoffPy и QuantLib
    имеют аналитику — сравниваем PDE engines).
    """
    print("  Setting up Heston benchmark...")
    
    params = HestonParams(
        S=100, K=100, T=1.0, r=0.05,
        v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
    )
    
    exact = heston_price(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, "call")
    
    # Проверка: QuantLib analytical должен ≈ нашему
    ql_exact = ql_heston_analytical(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, "call")
    print(f"  Heston exact: ChernoffPy={exact:.6f}, QuantLib={ql_exact:.6f}, diff={abs(exact-ql_exact):.2e}")
    
    results = []
    
    # ChernoffPy
    for nx, nv in [(128, 48), (256, 64), (512, 96)]:
        grid = HestonGridConfig(n_x=nx, n_v=nv)
        pricer = HestonFastPricer(CrankNicolson(), grid)
        for n in [20, 50, 100]:
            m = measure(pricer.price, params, n, "call", n_runs=3)
            error_pct = abs(m["price"].price - exact) / exact * 100
            results.append({
                "method": f"ChernoffPy ({nx}×{nv})",
                "n_steps": n,
                "price": m["price"].price,
                "error_pct": error_pct,
                "time_ms": m["time_ms"],
            })
    
    # QuantLib FDM Heston
    for n_time in [50, 100, 200]:
        for n_spot, n_vol in [(100, 50), (200, 100)]:
            m = measure(
                ql_heston_fdm, 100, 100, 1.0, 0.05,
                0.04, 2.0, 0.04, 0.3, -0.7,
                "call", n_time, n_spot, n_vol,
                n_runs=3,
            )
            error_pct = abs(m["price"] - exact) / exact * 100
            results.append({
                "method": f"QuantLib FDM ({n_spot}×{n_vol})",
                "n_steps": n_time,
                "price": m["price"],
                "error_pct": error_pct,
                "time_ms": m["time_ms"],
            })
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Table 5: Certified Bounds (только ChernoffPy)
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_certified():
    """
    Certified bounds — уникальная фича ChernoffPy.
    
    Показываем: для каждого n bound ≥ error,
    и bound tight (ratio < 5).
    """
    print("  Setting up Certified bounds benchmark...")
    
    market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    exact = bs_exact_price(market, "call")
    
    pricer = CertifiedEuropeanPricer(CrankNicolson())
    
    results = []
    for n in [10, 20, 50, 100, 200]:
        r = pricer.price_certified(market, n_steps=n, option_type="call")
        true_error = abs(r.price - exact)
        bound = r.certified_bound.bound
        results.append({
            "n_steps": n,
            "price": r.price,
            "true_error": true_error,
            "certified_bound": bound,
            "ratio": bound / max(true_error, 1e-15),
            "is_valid": bound >= true_error,
        })
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Double Barrier Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_double_barrier():
    """Double barrier options comparison."""
    print("  Setting up Double Barrier benchmark...")
    
    market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    bp = DoubleBarrierParams(
        lower_barrier=90,
        upper_barrier=120,
        barrier_type="double_knock_out",
    )
    
    # Exact price using analytical formula
    try:
        exact = double_barrier_analytical(
            market.S, market.K, bp.lower_barrier, bp.upper_barrier,
            market.r, market.sigma, market.T, "call"
        )
        print(f"  Double barrier exact price: {exact:.6f}")
    except Exception as e:
        print(f"  Warning: Could not compute exact price: {e}")
        exact = None
    
    results = []
    
    # ChernoffPy DST
    pricer = DoubleBarrierDSTPricer(CrankNicolson())
    for n in [20, 50, 100, 200]:
        m = measure(pricer.price, market, bp, n, "call", n_runs=10)
        if exact:
            error_pct = abs(m["price"].price - exact) / max(exact, 1e-10) * 100
        else:
            error_pct = None
        results.append({
            "config": "DKO B=90/120",
            "method": "ChernoffPy DST",
            "n_steps": n,
            "price": m["price"].price,
            "exact": exact,
            "error_pct": error_pct,
            "time_ms": m["time_ms"],
        })
    
    # QuantLib FDM - skip for now due to API issues
    print("  QuantLib double barrier: skipped (API compatibility)")
    # for n_time in [50, 100, 200, 500]:
    #     for n_spot in [200, 500]:
    #         m = measure(
    #             ql_double_barrier_fdm, 100, 100, 90, 120,
    #             1.0, 0.05, 0.20, "call", n_time, n_spot,
    #             n_runs=10,
    #         )
    #         if exact:
    #             error_pct = abs(m["price"] - exact) / max(exact, 1e-10) * 100
    #         else:
    #             error_pct = None
    #         results.append({
    #             "config": "DKO B=90/120",
    #             "method": f"QuantLib FDM ({n_spot}pts)",
    #             "n_steps": n_time,
    #             "price": m["price"],
    #             "exact": exact,
    #             "error_pct": error_pct,
    #             "time_ms": m["time_ms"],
    #         })
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Output: CSV + Console
# ═══════════════════════════════════════════════════════════════════════════════

def save_csv(data, filename):
    """Сохранить список dict в CSV."""
    os.makedirs("benchmarks/results/tables", exist_ok=True)
    path = f"benchmarks/results/tables/{filename}"
    if not data:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=data[0].keys())
        w.writeheader()
        w.writerows(data)
    print(f"  Saved: {path}")


def print_table(data, title, key_cols=None):
    """Красивый вывод таблицы в консоль."""
    if not data:
        return
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    
    cols = key_cols or list(data[0].keys())
    # Header
    header = " | ".join(f"{c:>15}" for c in cols)
    print(f"  {header}")
    print(f"  {'-' * len(header)}")
    # Rows
    for row in data:
        vals = []
        for c in cols:
            v = row.get(c, "")
            if isinstance(v, float):
                if "pct" in c or "error" in c.lower():
                    vals.append(f"{v:>15.4f}")
                elif "time" in c:
                    vals.append(f"{v:>15.2f}")
                else:
                    vals.append(f"{v:>15.6f}")
            elif v is None:
                vals.append(f"{'N/A':>15}")
            else:
                vals.append(f"{v:>15}")
        print(f"  {' | '.join(vals)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  ChernoffPy vs QuantLib — Benchmark Suite")
    print("=" * 80)
    
    try:
        import QuantLib as ql
        print(f"  QuantLib version: {ql.__version__}")
    except ImportError:
        print("  ERROR: QuantLib not installed!")
        print("  Run: pip install QuantLib-Python")
        sys.exit(1)
    
    try:
        from chernoffpy import __version__
        print(f"  ChernoffPy version: {__version__}")
    except:
        print(f"  ChernoffPy version: unknown")
    
    # Table 1: European
    print("\n[1/6] European options...")
    european = benchmark_european()
    print_table(european, "European Call ATM", 
                ["method", "n_steps", "price", "error_pct", "time_ms"])
    save_csv(european, "european_comparison.csv")
    
    # Table 2: Barrier
    print("\n[2/6] Barrier options...")
    barrier = benchmark_barrier()
    print_table(barrier, "Barrier Options",
                ["config", "method", "n_steps", "error_pct", "time_ms"])
    save_csv(barrier, "barrier_comparison.csv")
    
    # Table 3: American
    print("\n[3/6] American options...")
    american = benchmark_american()
    print_table(american, "American Put ATM",
                ["method", "n_steps", "price", "error_pct", "time_ms"])
    save_csv(american, "american_comparison.csv")
    
    # Table 4: Heston
    print("\n[4/6] Heston model...")
    heston = benchmark_heston()
    print_table(heston, "Heston Call ATM",
                ["method", "n_steps", "price", "error_pct", "time_ms"])
    save_csv(heston, "heston_comparison.csv")
    
    # Table 5: Certified bounds
    print("\n[5/6] Certified bounds (ChernoffPy only)...")
    certified = benchmark_certified()
    print_table(certified, "Certified Error Bounds",
                ["n_steps", "price", "true_error", "certified_bound", "ratio", "is_valid"])
    save_csv(certified, "certified_bounds.csv")
    
    # Table 6: Double Barrier
    print("\n[6/6] Double Barrier options...")
    double_barrier = benchmark_double_barrier()
    print_table(double_barrier, "Double Barrier Options",
                ["config", "method", "n_steps", "price", "error_pct", "time_ms"])
    save_csv(double_barrier, "double_barrier_comparison.csv")
    
    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print("  Tables saved to benchmarks/results/tables/")
    print("  Run 'python benchmarks/plot_results.py' for convergence plots")


if __name__ == "__main__":
    main()
