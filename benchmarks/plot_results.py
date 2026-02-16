"""
Графики для статьи.

Запуск: python benchmarks/plot_results.py

Генерирует:
1. Convergence plot: error vs n_steps (log-log)
2. Efficiency scatter: error vs time
3. Certified bounds: bound vs true error
4. Barrier comparison: DST vs QuantLib FDM
"""

from __future__ import annotations

import sys
import csv
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_csv(filename):
    """Load CSV file from results/tables."""
    path = f"benchmarks/results/tables/{filename}"
    if not os.path.exists(path):
        print(f"  Warning: {path} not found. Run vs_quantlib.py first.")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_convergence_european():
    """Log-log: ошибка vs n_steps для European call."""
    data = load_csv("european_comparison.csv")
    if not data:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    methods = {}
    for row in data:
        method = row["method"]
        if method not in methods:
            methods[method] = {"n": [], "err": []}
        methods[method]["n"].append(int(row["n_steps"]))
        methods[method]["err"].append(float(row["error_pct"]))
    
    markers = {
        "ChernoffPy BE": "s",
        "ChernoffPy CN": "o",
        "ChernoffPy Padé": "^",
    }
    colors = {
        "ChernoffPy BE": "C0",
        "ChernoffPy CN": "C1",
        "ChernoffPy Padé": "C2",
    }
    
    for method, d in methods.items():
        if "QuantLib" in method:
            ax.loglog(d["n"], d["err"], "--", label=method, alpha=0.6, linewidth=1.5)
        else:
            marker = markers.get(method, "x")
            color = colors.get(method, "gray")
            ax.loglog(d["n"], d["err"], f"-{marker}",
                      label=method, color=color, markersize=8, linewidth=2)
    
    # Reference slopes
    ns = np.array([10, 200])
    ax.loglog(ns, 10 / ns, ":", color="gray", alpha=0.5, label="O(1/n)")
    ax.loglog(ns, 100 / ns**2, ":", color="gray", alpha=0.5, label="O(1/n²)")
    
    ax.set_xlabel("Number of Chernoff steps (n)", fontsize=12)
    ax.set_ylabel("Relative error (%)", fontsize=12)
    ax.set_title("European Call: Convergence Comparison", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    
    os.makedirs("benchmarks/results/plots", exist_ok=True)
    fig.savefig("benchmarks/results/plots/convergence_european.png",
                dpi=150, bbox_inches="tight")
    print("  Saved: convergence_european.png")
    plt.close(fig)


def plot_efficiency_scatter():
    """Scatter: ошибка vs время (Pareto frontier)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # European data
    data = load_csv("european_comparison.csv")
    if data:
        chernoff_times = []
        chernoff_errors = []
        quantlib_times = []
        quantlib_errors = []
        
        for row in data:
            if "ChernoffPy CN" in row["method"]:
                chernoff_times.append(float(row["time_ms"]))
                chernoff_errors.append(float(row["error_pct"]))
            elif "QuantLib" in row["method"]:
                quantlib_times.append(float(row["time_ms"]))
                quantlib_errors.append(float(row["error_pct"]))
        
        ax.scatter(chernoff_times, chernoff_errors,
                   c="C1", marker="o", s=80, zorder=5, label="ChernoffPy CN", edgecolors='black', linewidth=0.5)
        ax.scatter(quantlib_times, quantlib_errors,
                   c="C3", marker="s", s=60, alpha=0.6, label="QuantLib FDM", edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Relative error (%)", fontsize=12)
    ax.set_title("Efficiency: Accuracy vs Speed (European Call)", fontsize=14)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    
    fig.savefig("benchmarks/results/plots/efficiency_scatter.png",
                dpi=150, bbox_inches="tight")
    print("  Saved: efficiency_scatter.png")
    plt.close(fig)


def plot_certified_bounds():
    """Bound vs true error — shows guarantee holds."""
    data = load_csv("certified_bounds.csv")
    if not data:
        return
    
    ns = [int(r["n_steps"]) for r in data]
    errors = [float(r["true_error"]) for r in data]
    bounds = [float(r["certified_bound"]) for r in data]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.semilogy(ns, errors, "-o", label="True error", color="C0",
                markersize=8, linewidth=2)
    ax.semilogy(ns, bounds, "-s", label="Certified bound", color="C3",
                markersize=8, linewidth=2)
    ax.fill_between(ns, errors, bounds, alpha=0.2, color="C3",
                    label="Safety margin")
    
    ax.set_xlabel("Number of steps (n)", fontsize=12)
    ax.set_ylabel("Absolute error", fontsize=12)
    ax.set_title("Certified Error Bounds (Galkin-Remizov 2025)", fontsize=14)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    
    fig.savefig("benchmarks/results/plots/certified_bounds.png",
                dpi=150, bbox_inches="tight")
    print("  Saved: certified_bounds.png")
    plt.close(fig)


def plot_barrier_comparison():
    """Barrier: DST vs QuantLib for near and far barriers."""
    data = load_csv("barrier_comparison.csv")
    if not data:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    configs = [
        ("DOC B=90", "Down-and-Out Call (B=90, far barrier)", axes[0]),
        ("DOC B=99", "Down-and-Out Call (B=99, near barrier)", axes[1]),
    ]
    
    for config_name, title, ax in configs:
        # Filter data for this config
        config_data = [r for r in data if r["config"] == config_name]
        
        methods = {}
        for row in config_data:
            method = row["method"]
            if method not in methods:
                methods[method] = {"n": [], "err": []}
            methods[method]["n"].append(int(row["n_steps"]))
            methods[method]["err"].append(float(row["error_pct"]))
        
        # Plot ChernoffPy DST
        if "ChernoffPy DST" in methods:
            d = methods["ChernoffPy DST"]
            ax.loglog(d["n"], d["err"], "-o", label="ChernoffPy DST",
                     color="C1", markersize=8, linewidth=2)
            
            # Add floor line annotation for DST
            # DST floor = 10*sqrt(N) ≈ 452 for N=2048
            if config_name == "DOC B=99":
                ax.axvline(x=452, color="C1", linestyle=":", alpha=0.5)
                ax.text(470, ax.get_ylim()[1]*0.5, "floor=452",
                       fontsize=8, color="C1", alpha=0.7)
        
        # Plot QuantLib FDM variants
        for method, d in methods.items():
            if "QuantLib" in method:
                label = method.replace("QuantLib FDM ", "QL ")
                ax.loglog(d["n"], d["err"], "--", label=label,
                         alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel("Number of steps (n)", fontsize=11)
        ax.set_ylabel("Relative error (%)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    fig.savefig("benchmarks/results/plots/barrier_comparison.png",
                dpi=150, bbox_inches="tight")
    print("  Saved: barrier_comparison.png")
    plt.close(fig)


def plot_convergence_heston():
    """Heston convergence comparison."""
    data = load_csv("heston_comparison.csv")
    if not data:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    methods = {}
    for row in data:
        method = row["method"]
        if method not in methods:
            methods[method] = {"n": [], "err": []}
        methods[method]["n"].append(int(row["n_steps"]))
        methods[method]["err"].append(float(row["error_pct"]))
    
    # Plot ChernoffPy variants
    colors = plt.cm.tab10(np.linspace(0, 0.3, 3))
    for i, (method, d) in enumerate(methods.items()):
        if "ChernoffPy" in method:
            ax.loglog(d["n"], d["err"], "-o", label=method,
                     color=colors[i], markersize=8, linewidth=2)
    
    # Plot QuantLib variants
    for method, d in methods.items():
        if "QuantLib" in method:
            label = method.replace("QuantLib FDM ", "QL ")
            ax.loglog(d["n"], d["err"], "--", label=label,
                     alpha=0.7, linewidth=1.5)
    
    # Reference slopes
    ns = np.array([20, 100])
    ax.loglog(ns, 1 / ns, ":", color="gray", alpha=0.5, label="O(1/n)")
    ax.loglog(ns, 10 / ns**2, ":", color="gray", alpha=0.5, label="O(1/n²)")
    
    ax.set_xlabel("Number of steps (n)", fontsize=12)
    ax.set_ylabel("Relative error (%)", fontsize=12)
    ax.set_title("Heston Model: Convergence Comparison", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    
    fig.savefig("benchmarks/results/plots/convergence_heston.png",
                dpi=150, bbox_inches="tight")
    print("  Saved: convergence_heston.png")
    plt.close(fig)


def plot_american_comparison():
    """American option convergence."""
    data = load_csv("american_comparison.csv")
    if not data:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    methods = {}
    for row in data:
        method = row["method"]
        if method not in methods:
            methods[method] = {"n": [], "err": []}
        methods[method]["n"].append(int(row["n_steps"]))
        methods[method]["err"].append(float(row["error_pct"]))
    
    # Plot ChernoffPy
    if "ChernoffPy CN" in methods:
        d = methods["ChernoffPy CN"]
        ax.loglog(d["n"], d["err"], "-o", label="ChernoffPy CN",
                 color="C1", markersize=8, linewidth=2)
    
    # Plot QuantLib
    for method, d in methods.items():
        if "QuantLib" in method:
            label = method.replace("QuantLib FDM ", "QL ")
            ax.loglog(d["n"], d["err"], "--", label=label,
                     alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel("Number of steps (n)", fontsize=12)
    ax.set_ylabel("Relative error (%)", fontsize=12)
    ax.set_title("American Put: Convergence Comparison", fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    
    fig.savefig("benchmarks/results/plots/convergence_american.png",
                dpi=150, bbox_inches="tight")
    print("  Saved: convergence_american.png")
    plt.close(fig)


def create_summary_figure():
    """Create a comprehensive summary figure for the paper."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # This will be a comprehensive summary plot combining key results
    # For now, we'll create individual plots above
    
    print("  Summary figure creation: using individual plots above")


def main():
    print("=" * 80)
    print("  Generating plots for ChernoffPy vs QuantLib benchmarks")
    print("=" * 80)
    
    os.makedirs("benchmarks/results/plots", exist_ok=True)
    
    print("\n[1/6] European convergence...")
    plot_convergence_european()
    
    print("\n[2/6] Efficiency scatter...")
    plot_efficiency_scatter()
    
    print("\n[3/6] Certified bounds...")
    plot_certified_bounds()
    
    print("\n[4/6] Barrier comparison...")
    plot_barrier_comparison()
    
    print("\n[5/6] Heston convergence...")
    plot_convergence_heston()
    
    print("\n[6/6] American convergence...")
    plot_american_comparison()
    
    print("\n" + "=" * 80)
    print("  All plots saved to benchmarks/results/plots/")
    print("=" * 80)


if __name__ == "__main__":
    main()
