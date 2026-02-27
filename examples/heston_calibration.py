"""Heston model calibration to a synthetic volatility surface.

Demonstrates:
- Generating a synthetic option surface with vol skew
- Calibrating flat vol and linear-skew parametrizations to market quotes
- Computing implied vs model volatility comparison
- Pricing with calibrated parameters
"""

import numpy as np

from chernoffpy import CrankNicolson
from chernoffpy.finance import (
    EuropeanPricer,
    HestonFastPricer,
    MarketParams,
    VolCalibrator,
    bs_exact_price,
    generate_synthetic_quotes,
    implied_volatility,
)
from chernoffpy.finance.heston_params import HestonParams


def print_vol_surface(market_data, label: str) -> None:
    """Print implied vol surface as a table."""
    strikes = sorted(set(q.strike for q in market_data.quotes))
    expiries = sorted(set(q.expiry for q in market_data.quotes))

    print(f"\n{label}")
    header = f"  {'K\\T':>6}" + "".join(f"  T={t:.2f}" for t in expiries)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for K in strikes:
        row = f"  {K:>6.0f}"
        for T in expiries:
            iv = next(
                (
                    implied_volatility(q.price, market_data.spot, K, T, market_data.rate, q.option_type)
                    for q in market_data.quotes
                    if q.strike == K and q.expiry == T
                ),
                None,
            )
            row += f"  {iv*100:6.2f}%" if iv is not None else "      N/A"
        print(row)


def main() -> None:
    S, r = 100.0, 0.05
    strikes = (85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0)
    expiries = (0.25, 0.5, 1.0, 2.0)

    # ------------------------------------------------------------------
    # 1. Generate synthetic market data with vol skew
    # ------------------------------------------------------------------
    print("=" * 60)
    print("1. SYNTHETIC MARKET DATA (vol skew = -0.05 / log-moneyness)")
    print("=" * 60)

    market_data = generate_synthetic_quotes(
        spot=S,
        rate=r,
        sigma=0.20,
        strikes=strikes,
        expiries=expiries,
        skew=-0.05,
        option_type="call",
    )
    print(f"  Generated {len(market_data.quotes)} quotes")
    print_vol_surface(market_data, "  Implied volatility surface (target)")

    # ------------------------------------------------------------------
    # 2. Calibrate flat vol (single parameter)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("2. CALIBRATION: FLAT VOL")
    print("=" * 60)

    calibrator = VolCalibrator()
    result_flat = calibrator.calibrate(market_data, parametrization="flat")
    print(result_flat.summary())

    # ------------------------------------------------------------------
    # 3. Calibrate linear skew (two parameters: atm vol + skew slope)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("3. CALIBRATION: LINEAR SKEW")
    print("=" * 60)

    result_skew = calibrator.calibrate(market_data, parametrization="linear_skew")
    print(result_skew.summary())

    # ------------------------------------------------------------------
    # 4. Pricing with calibrated parameters
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("4. PRICING WITH CALIBRATED VOL SURFACE")
    print("=" * 60)

    test_params = [
        MarketParams(S=S, K=90, T=1.0, r=r, sigma=result_skew.vol_surface(90.0, 1.0)),
        MarketParams(S=S, K=100, T=1.0, r=r, sigma=result_skew.vol_surface(100.0, 1.0)),
        MarketParams(S=S, K=110, T=1.0, r=r, sigma=result_skew.vol_surface(110.0, 1.0)),
    ]
    scheme = CrankNicolson()
    pricer = EuropeanPricer(scheme)

    print(f"  {'Strike':>8}  {'Calib σ':>8}  {'Chernoff':>10}  {'BS exact':>10}")
    print(f"  {'-'*44}")
    for mp in test_params:
        chernoff_price = pricer.price(mp, n_steps=50, option_type="call").price
        bs_price = bs_exact_price(mp, "call")
        print(
            f"  {mp.K:>8.0f}  {mp.sigma:>8.4f}  {chernoff_price:>10.4f}  {bs_price:>10.4f}"
        )

    # ------------------------------------------------------------------
    # 5. Heston model reference prices
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("5. HESTON MODEL REFERENCE PRICES")
    print("=" * 60)
    print("   (Heston generates skew intrinsically via stochastic vol)")

    heston_params = HestonParams(
        S=S, K=100, T=1.0, r=r,
        v0=0.04, kappa=2.0, theta=0.04, xi=0.4, rho=-0.6,
    )
    heston_pricer = HestonFastPricer(scheme)

    strikes_test = [85, 90, 95, 100, 105, 110, 115]
    print(f"\n  {'Strike':>8}  {'Heston':>10}  {'BS flat':>10}  {'Heston IV':>10}")
    print(f"  {'-'*46}")
    for K in strikes_test:
        h = HestonParams(
            S=heston_params.S, K=K, T=heston_params.T, r=heston_params.r,
            v0=heston_params.v0, kappa=heston_params.kappa,
            theta=heston_params.theta, xi=heston_params.xi, rho=heston_params.rho,
        )
        heston_price = heston_pricer.price(h, n_steps=50, option_type="call").price
        bs_atm = bs_exact_price(MarketParams(S=S, K=K, T=1.0, r=r, sigma=0.20), "call")
        iv = implied_volatility(heston_price, S, K, 1.0, r, "call")
        print(f"  {K:>8d}  {heston_price:>10.4f}  {bs_atm:>10.4f}  {iv*100:>9.2f}%")

    print()
    print("  Note: Heston IV decreases for higher strikes (negative skew due to ρ < 0)")


if __name__ == "__main__":
    main()
