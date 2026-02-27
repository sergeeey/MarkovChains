"""American option pricing with discrete dividends.

Demonstrates:
- AmericanPricer vs European pricer (early exercise premium)
- Effect of discrete dividends on American call (can make early exercise optimal)
- Comparison with Barone-Adesi-Whaley approximation and CRR binomial tree
- Early exercise boundary visualization
"""

import numpy as np

from chernoffpy import CrankNicolson
from chernoffpy.finance import (
    AmericanPricer,
    DividendSchedule,
    EuropeanPricer,
    MarketParams,
    american_baw,
    american_binomial,
    bs_exact_price,
)


def main() -> None:
    market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    scheme = CrankNicolson()

    # ------------------------------------------------------------------
    # 1. American put vs European put: early exercise premium
    # ------------------------------------------------------------------
    print("=" * 60)
    print("1. EARLY EXERCISE PREMIUM (PUT)")
    print("=" * 60)

    eur_put = EuropeanPricer(scheme).price(market, n_steps=100, option_type="put").price
    amer_put = AmericanPricer(scheme).price(market, n_steps=100, option_type="put").price
    baw_put = american_baw(market.S, market.K, market.r, market.sigma, market.T, option_type="put")
    crr_put = american_binomial(market.S, market.K, market.r, market.sigma, market.T, option_type="put", n_steps=2000)
    bs_put = bs_exact_price(market, "put")

    print(f"  Black-Scholes (European):  {bs_put:.4f}")
    print(f"  Chernoff European:         {eur_put:.4f}  (error vs BS: {abs(eur_put-bs_put):.2e})")
    print(f"  Chernoff American:         {amer_put:.4f}")
    print(f"  BAW approximation:         {baw_put:.4f}")
    print(f"  CRR binomial (n=2000):     {crr_put:.4f}")
    print(f"  Early exercise premium:    {amer_put - eur_put:.4f}")

    # ------------------------------------------------------------------
    # 2. American call: no dividends → equals European
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("2. AMERICAN CALL WITHOUT DIVIDENDS")
    print("=" * 60)

    eur_call = bs_exact_price(market, "call")
    amer_call = AmericanPricer(scheme).price(market, n_steps=100, option_type="call").price

    print(f"  Black-Scholes (European):  {eur_call:.4f}")
    print(f"  Chernoff American call:    {amer_call:.4f}")
    print(f"  Difference:                {abs(amer_call - eur_call):.2e}  (should be ~0)")

    # ------------------------------------------------------------------
    # 3. American PUT with proportional dividend (vs absolute)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("3. AMERICAN PUT: PROPORTIONAL vs ABSOLUTE DIVIDEND")
    print("=" * 60)

    # Absolute: $2.0 at t=0.5  |  Proportional: 2% of spot at t=0.5
    # Both approximate the same economic event (2% div on $100 stock)
    abs_sched  = DividendSchedule(times=(0.5,), amounts=(2.0,),  proportional=False)
    prop_sched = DividendSchedule(times=(0.5,), amounts=(0.02,), proportional=True)

    put_no_div  = AmericanPricer(scheme).price(market, n_steps=100, option_type="put").price
    put_abs     = AmericanPricer(scheme).price(market, n_steps=100, option_type="put", dividends=abs_sched).price
    put_prop    = AmericanPricer(scheme).price(market, n_steps=100, option_type="put", dividends=prop_sched).price

    print(f"  American put (no dividend):        {put_no_div:.4f}")
    print(f"  American put (abs $2.0 at t=0.5):  {put_abs:.4f}  (+{put_abs-put_no_div:.4f})")
    print(f"  American put (prop 2% at t=0.5):   {put_prop:.4f}  (+{put_prop-put_no_div:.4f})")
    print(f"  Absolute vs proportional diff:     {abs(put_abs - put_prop):.4f}  (should be small)")
    print("  (Both model ~same dividend; proportional is numerically cleaner)")

    # ------------------------------------------------------------------
    # 4. American PUT sensitivity to dividend size (put benefits from dividends)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("4. AMERICAN PUT SENSITIVITY TO DIVIDEND SIZE")
    print("=" * 60)
    print(f"  {'Dividend':>10}  {'Amer put':>10}  {'No-div put':>10}  {'Effect':>10}")
    print(f"  {'-'*44}")

    base_put = AmericanPricer(scheme).price(market, n_steps=100, option_type="put").price
    for d in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        if d == 0.0:
            amer_p = base_put
        else:
            divs = DividendSchedule(times=(0.5,), amounts=(d,))
            amer_p = AmericanPricer(scheme).price(
                market, n_steps=100, option_type="put", dividends=divs
            ).price
        print(f"  {d:>10.1f}  {amer_p:>10.4f}  {base_put:>10.4f}  {amer_p-base_put:>+10.4f}")

    # ------------------------------------------------------------------
    # 5. Early exercise boundary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("5. EARLY EXERCISE BOUNDARY (AMERICAN PUT)")
    print("=" * 60)

    result_with_boundary = AmericanPricer(scheme).price(
        market, n_steps=50, option_type="put", return_boundary=True
    )

    if result_with_boundary.exercise_boundary is not None:
        boundary = result_with_boundary.exercise_boundary
        times = np.linspace(0, market.T, len(boundary))
        print("  Time     Critical S (exercise if S < S*)")
        print("  " + "-" * 36)
        for i in range(0, len(times), max(1, len(times) // 8)):
            print(f"  t={times[i]:.2f}    S* = {boundary[i]:.2f}")
    else:
        print("  (boundary not available for this configuration)")


if __name__ == "__main__":
    main()
