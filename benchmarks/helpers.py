"""
QuantLib wrapper functions for benchmarking.

This module provides clean interfaces to QuantLib pricing engines
for comparison with ChernoffPy implementations.
"""

from __future__ import annotations

import numpy as np


def _make_ql_process(S: float, r: float, sigma: float, q: float = 0.0):
    """Create QuantLib Black-Scholes-Merton process."""
    import QuantLib as ql
    
    today = ql.Date.todaysDate()
    
    spot = ql.SimpleQuote(S)
    rate = ql.SimpleQuote(r)
    vol = ql.SimpleQuote(sigma)
    div = ql.SimpleQuote(q)
    
    spot_handle = ql.QuoteHandle(spot)
    rate_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, ql.QuoteHandle(rate), ql.Actual365Fixed())
    )
    div_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, ql.QuoteHandle(div), ql.Actual365Fixed())
    )
    vol_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(),
                           ql.QuoteHandle(vol), ql.Actual365Fixed())
    )
    
    process = ql.BlackScholesMertonProcess(
        spot_handle, div_handle, rate_handle, vol_handle
    )
    return process, today


def ql_european_analytical(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """QuantLib analytical European option price."""
    import QuantLib as ql
    
    process, today = _make_ql_process(S, r, sigma, q)
    maturity = today + ql.Period(int(T * 365), ql.Days)
    
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type == "call" else ql.Option.Put, K
    )
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)
    
    engine = ql.AnalyticEuropeanEngine(process)
    option.setPricingEngine(engine)
    return option.NPV()


def ql_european_fdm(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_time: int = 50,
    n_spot: int = 200,
    q: float = 0.0,
) -> float:
    """QuantLib FDM European option price."""
    import QuantLib as ql
    
    process, today = _make_ql_process(S, r, sigma, q)
    maturity = today + ql.Period(int(T * 365), ql.Days)
    
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type == "call" else ql.Option.Put, K
    )
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)
    
    engine = ql.FdBlackScholesVanillaEngine(process, n_time, n_spot)
    option.setPricingEngine(engine)
    return option.NPV()


def ql_american_fdm(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "put",
    n_time: int = 100,
    n_spot: int = 200,
    q: float = 0.0,
) -> float:
    """QuantLib FDM American option price."""
    import QuantLib as ql
    
    process, today = _make_ql_process(S, r, sigma, q)
    maturity = today + ql.Period(int(T * 365), ql.Days)
    
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type == "call" else ql.Option.Put, K
    )
    exercise = ql.AmericanExercise(today, maturity)
    option = ql.VanillaOption(payoff, exercise)
    
    engine = ql.FdBlackScholesVanillaEngine(process, n_time, n_spot)
    option.setPricingEngine(engine)
    return option.NPV()


def ql_american_crr(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "put",
    n: int = 10000,
) -> float:
    """
    CRR Binomial tree for American options (quasi-exact reference).
    
    This is our 'ground truth' for American option benchmarking.
    """
    import QuantLib as ql
    
    process, today = _make_ql_process(S, r, sigma, 0.0)
    maturity = today + ql.Period(int(T * 365), ql.Days)
    
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type == "call" else ql.Option.Put, K
    )
    exercise = ql.AmericanExercise(today, maturity)
    option = ql.VanillaOption(payoff, exercise)
    
    # Use BinomialVanillaEngine with CRR
    timeSteps = n
    engine = ql.BinomialVanillaEngine(process, "crr", timeSteps)
    option.setPricingEngine(engine)
    return option.NPV()


def ql_barrier_analytical(
    S: float,
    K: float,
    B: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    barrier_type: str = "DownOut",
    q: float = 0.0,
) -> float:
    """QuantLib analytical barrier option price."""
    import QuantLib as ql
    
    process, today = _make_ql_process(S, r, sigma, q)
    maturity = today + ql.Period(int(T * 365), ql.Days)
    
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type == "call" else ql.Option.Put, K
    )
    exercise = ql.EuropeanExercise(maturity)
    
    bt = getattr(ql.Barrier, barrier_type)
    option = ql.BarrierOption(bt, B, 0.0, payoff, exercise)
    
    engine = ql.AnalyticBarrierEngine(process)
    option.setPricingEngine(engine)
    return option.NPV()


def ql_barrier_fdm(
    S: float,
    K: float,
    B: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    barrier_type: str = "DownOut",
    n_time: int = 100,
    n_spot: int = 200,
    q: float = 0.0,
) -> float:
    """QuantLib FDM barrier option price."""
    import QuantLib as ql
    
    process, today = _make_ql_process(S, r, sigma, q)
    maturity = today + ql.Period(int(T * 365), ql.Days)
    
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type == "call" else ql.Option.Put, K
    )
    exercise = ql.EuropeanExercise(maturity)
    
    bt = getattr(ql.Barrier, barrier_type)
    option = ql.BarrierOption(bt, B, 0.0, payoff, exercise)
    
    engine = ql.FdBlackScholesBarrierEngine(process, n_time, n_spot)
    option.setPricingEngine(engine)
    return option.NPV()


def ql_double_barrier_fdm(
    S: float,
    K: float,
    B_low: float,
    B_high: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_time: int = 100,
    n_spot: int = 200,
    q: float = 0.0,
) -> float:
    """QuantLib FDM double barrier option price."""
    import QuantLib as ql
    
    process, today = _make_ql_process(S, r, sigma, q)
    maturity = today + ql.Period(int(T * 365), ql.Days)
    
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type == "call" else ql.Option.Put, K
    )
    exercise = ql.EuropeanExercise(maturity)
    
    option = ql.DoubleBarrierOption(
        ql.DoubleBarrier.KnockOut, B_low, B_high, 0.0, payoff, exercise
    )
    
    engine = ql.FdBlackScholesBarrierEngine(process, n_time, n_spot)
    option.setPricingEngine(engine)
    return option.NPV()


def _make_heston_process(
    S: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
):
    """Create QuantLib Heston process."""
    import QuantLib as ql
    
    today = ql.Date.todaysDate()
    
    spot = ql.QuoteHandle(ql.SimpleQuote(S))
    rate = ql.YieldTermStructureHandle(
        ql.FlatForward(today, r, ql.Actual365Fixed())
    )
    div = ql.YieldTermStructureHandle(
        ql.FlatForward(today, 0.0, ql.Actual365Fixed())
    )
    
    heston_process = ql.HestonProcess(rate, div, spot, v0, kappa, theta, xi, rho)
    return heston_process, today


def ql_heston_analytical(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    option_type: str = "call",
) -> float:
    """QuantLib analytical Heston price (Lewis/Gatheral)."""
    import QuantLib as ql
    
    heston_process, today = _make_heston_process(S, r, v0, kappa, theta, xi, rho)
    heston_model = ql.HestonModel(heston_process)
    
    maturity = today + ql.Period(int(T * 365), ql.Days)
    
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type == "call" else ql.Option.Put, K
    )
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)
    
    engine = ql.AnalyticHestonEngine(heston_model)
    option.setPricingEngine(engine)
    return option.NPV()


def ql_heston_fdm(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    option_type: str = "call",
    n_time: int = 50,
    n_spot: int = 100,
    n_vol: int = 50,
) -> float:
    """QuantLib FDM Heston price (2D finite differences)."""
    import QuantLib as ql
    
    heston_process, today = _make_heston_process(S, r, v0, kappa, theta, xi, rho)
    heston_model = ql.HestonModel(heston_process)
    
    maturity = today + ql.Period(int(T * 365), ql.Days)
    
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type == "call" else ql.Option.Put, K
    )
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)
    
    engine = ql.FdHestonVanillaEngine(heston_model, n_time, n_spot, n_vol)
    option.setPricingEngine(engine)
    return option.NPV()
