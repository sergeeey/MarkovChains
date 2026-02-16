"""Tests for continuous/discrete dividend support."""

from __future__ import annotations

import numpy as np
import pytest

from chernoffpy import CrankNicolson
from chernoffpy.finance.american import AmericanPricer
from chernoffpy.finance.dividends import apply_discrete_dividend, find_dividend_steps
from chernoffpy.finance.european import EuropeanPricer
from chernoffpy.finance.transforms import bs_exact_price
from chernoffpy.finance.validation import DividendSchedule, GridConfig, MarketParams


def _crr_american_put_proportional_dividend(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    t_div: float,
    delta: float,
) -> float:
    """CRR benchmark with one proportional dividend (recombining tree)."""
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)

    div_step = int(round(t_div / dt))
    div_step = max(0, min(div_step, n_steps))

    stock = np.zeros((n_steps + 1, n_steps + 1))
    stock[0, 0] = S

    for i in range(1, n_steps + 1):
        stock[i, 0] = stock[i - 1, 0] * d
        for j in range(1, i + 1):
            stock[i, j] = stock[i - 1, j - 1] * u
        if i == div_step:
            stock[i, : i + 1] *= (1.0 - delta)

    value = np.maximum(K - stock[n_steps, : n_steps + 1], 0.0)

    disc = np.exp(-r * dt)
    for i in range(n_steps - 1, -1, -1):
        cont = disc * (p * value[1 : i + 2] + (1 - p) * value[0 : i + 1])
        ex = np.maximum(K - stock[i, : i + 1], 0.0)
        value = np.maximum(cont, ex)

    return float(value[0])


class TestDividendScheduleValidation:

    def test_valid_schedule(self):
        s = DividendSchedule(times=(0.25, 0.5), amounts=(1.0, 1.2))
        assert len(s.times) == 2

    def test_empty_schedule(self):
        s = DividendSchedule(times=(), amounts=())
        assert s.times == ()

    def test_negative_amount_rejected(self):
        with pytest.raises(ValueError, match="amount"):
            DividendSchedule(times=(0.25,), amounts=(-1.0,))

    def test_negative_time_rejected(self):
        with pytest.raises(ValueError, match="time"):
            DividendSchedule(times=(-0.1,), amounts=(1.0,))

    def test_unsorted_rejected(self):
        with pytest.raises(ValueError, match="sorted"):
            DividendSchedule(times=(0.5, 0.25), amounts=(1.0, 1.0))

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            DividendSchedule(times=(0.25,), amounts=(1.0, 1.0))


class TestApplyDividend:

    def test_absolute_dividend_shifts_grid(self):
        x = np.linspace(-1, 1, 128)
        u = np.exp(x)
        shifted = apply_discrete_dividend(u, x, amount=2.0, strike=100.0, proportional=False)
        assert np.isfinite(shifted).all()
        assert shifted.shape == u.shape

    def test_proportional_dividend_shifts_grid(self):
        x = np.linspace(-1, 1, 128)
        u = np.cos(x)
        shifted = apply_discrete_dividend(u, x, amount=0.02, strike=100.0, proportional=True)
        assert np.isfinite(shifted).all()

    def test_small_dividend_near_identity(self):
        x = np.linspace(-1, 1, 128)
        u = np.sin(x)
        shifted = apply_discrete_dividend(u, x, amount=1e-6, strike=100.0, proportional=False)
        assert np.max(np.abs(shifted[1:-1] - u[1:-1])) < 1e-3

    def test_large_dividend_clamps(self):
        x = np.linspace(-6, 2, 256)
        u = np.exp(-x**2)
        shifted = apply_discrete_dividend(u, x, amount=500.0, strike=100.0, proportional=False)
        assert np.isfinite(shifted).all()

    def test_find_dividend_steps(self):
        sched = DividendSchedule(times=(0.25, 0.5, 0.99), amounts=(1.0, 1.0, 1.0))
        mapping = find_dividend_steps(sched, maturity=1.0, n_steps=50)
        assert len(mapping) >= 1


class TestContinuousYield:

    def test_european_call_q_positive(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.03)
        pricer = EuropeanPricer(CrankNicolson(), GridConfig(N=2048, L=10.0, taper_width=2.0))
        price = pricer.price(market, n_steps=80, option_type="call").price
        bs = bs_exact_price(market, "call")
        assert abs(price - bs) / bs < 0.01

    def test_european_put_q_positive(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.03)
        pricer = EuropeanPricer(CrankNicolson(), GridConfig(N=2048, L=10.0, taper_width=2.0))
        price = pricer.price(market, n_steps=80, option_type="put").price
        bs = bs_exact_price(market, "put")
        assert abs(price - bs) / bs < 0.01

    def test_put_call_parity_with_q(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.03)
        pricer = EuropeanPricer(CrankNicolson())
        c = pricer.price(market, n_steps=80, option_type="call").price
        p = pricer.price(market, n_steps=80, option_type="put").price
        rhs = market.S * np.exp(-market.q * market.T) - market.K * np.exp(-market.r * market.T)
        assert abs((c - p) - rhs) < 0.05

    def test_q_zero_unchanged(self):
        market0 = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        marketq0 = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0)
        pricer = EuropeanPricer(CrankNicolson())
        p0 = pricer.price(market0, n_steps=80, option_type="call").price
        p1 = pricer.price(marketq0, n_steps=80, option_type="call").price
        assert p0 == pytest.approx(p1, abs=1e-12)

    def test_high_yield(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.10)
        pricer = EuropeanPricer(CrankNicolson())
        c = pricer.price(market, n_steps=80, option_type="call").price
        p = pricer.price(market, n_steps=80, option_type="put").price
        assert np.isfinite(c) and np.isfinite(p)
        assert c <= p + 5.0


class TestAmericanDiscreteDividend:

    def test_no_dividend_unchanged(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        pricer = AmericanPricer(CrankNicolson())
        a0 = pricer.price(market, n_steps=80, option_type="put").price
        a1 = pricer.price(market, n_steps=80, option_type="put", dividends=None).price
        assert a0 == pytest.approx(a1, abs=1e-12)

    def test_dividend_at_expiry_ignored(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        sched = DividendSchedule(times=(1.0,), amounts=(2.0,))
        pricer = AmericanPricer(CrankNicolson())
        p_no = pricer.price(market, n_steps=80, option_type="put").price
        p_div = pricer.price(market, n_steps=80, option_type="put", dividends=sched).price
        assert abs(p_no - p_div) < 0.05

    def test_proportional_vs_absolute(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        p = AmericanPricer(CrankNicolson())
        abs_sched = DividendSchedule(times=(0.5,), amounts=(2.0,), proportional=False)
        prop_sched = DividendSchedule(times=(0.5,), amounts=(0.02,), proportional=True)
        v_abs = p.price(market, n_steps=100, option_type="put", dividends=abs_sched).price
        v_prop = p.price(market, n_steps=100, option_type="put", dividends=prop_sched).price
        assert abs(v_abs - v_prop) < 2.0

    def test_american_call_with_dividend_ge_euro(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        sched = DividendSchedule(times=(0.5,), amounts=(1.5,))
        amer = AmericanPricer(CrankNicolson()).price(market, n_steps=100, option_type="call", dividends=sched)
        euro = EuropeanPricer(CrankNicolson()).price(market, n_steps=100, option_type="call")
        assert amer.price >= euro.price - 1e-10

    def test_convergence_with_n(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        sched = DividendSchedule(times=(0.5,), amounts=(0.02,), proportional=True)
        pr = AmericanPricer(CrankNicolson())
        p40 = pr.price(market, n_steps=40, option_type="put", dividends=sched).price
        p80 = pr.price(market, n_steps=80, option_type="put", dividends=sched).price
        p160 = pr.price(market, n_steps=160, option_type="put", dividends=sched).price
        assert abs(p160 - p80) <= abs(p80 - p40) + 0.1

    def test_proportional_dividend_vs_crr_benchmark(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.03, sigma=0.25)
        sched = DividendSchedule(times=(0.5,), amounts=(0.02,), proportional=True)

        pricer = AmericanPricer(CrankNicolson(), GridConfig(N=1024, L=8.0, taper_width=2.0))
        num = pricer.price(market, n_steps=180, option_type="put", dividends=sched).price

        ref = _crr_american_put_proportional_dividend(
            S=100,
            K=100,
            T=1.0,
            r=0.03,
            sigma=0.25,
            n_steps=1200,
            t_div=0.5,
            delta=0.02,
        )

        assert abs(num - ref) / ref < 0.15

