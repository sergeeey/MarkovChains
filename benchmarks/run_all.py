"""
Convenience script to run all benchmarks and generate all plots.

Usage:
    python benchmarks/run_all.py

This will:
1. Run vs_quantlib.py (generate CSV tables)
2. Run plot_results.py (generate PNG plots)
3. Print summary of results
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and report success/failure."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=False,
            text=True,
            check=True,
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ {description} failed: {e}")
        return False


def main():
    print("=" * 60)
    print("  ChernoffPy vs QuantLib - Full Benchmark Suite")
    print("=" * 60)
    
    # Check QuantLib is installed
    try:
        import QuantLib as ql
        print(f"\n✓ QuantLib {ql.__version__} is installed")
    except ImportError:
        print("\n✗ QuantLib is not installed!")
        print("  Run: pip install QuantLib-Python")
        sys.exit(1)
    
    # Run benchmarks
    success = True
    
    success &= run_command(
        [sys.executable, "benchmarks/vs_quantlib.py"],
        "Running benchmarks (vs_quantlib.py)"
    )
    
    success &= run_command(
        [sys.executable, "benchmarks/plot_results.py"],
        "Generating plots (plot_results.py)"
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    
    if success:
        print("\n✓ All benchmarks completed successfully!")
        print("\nGenerated files:")
        print("  - benchmarks/results/tables/*.csv (6 files)")
        print("  - benchmarks/results/plots/*.png (6 files)")
        print("  - benchmarks/results/REPORT.md")
        print("\nNext steps:")
        print("  - View REPORT.md for detailed analysis")
        print("  - Check plots/ for visualizations")
        print("  - Import CSV files into your analysis tool")
    else:
        print("\n✗ Some benchmarks failed. Check output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
