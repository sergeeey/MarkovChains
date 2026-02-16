"""Tests for American-option analytical/benchmark references."""

import numpy as np

from chernoffpy.finance.american_analytical import american_baw, american_binomial
from chernoffpy.finance.transforms import bs_exact_price
from chernoffpy.finance.validation import MarketParams


class TestBinomialTree:

    def test_european_call_matches_bs(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        binom = american_binomial(100, 100, 0.05, 0.2, 1.0, "call", n_steps=3000, american=False)
        bs = bs_exact_price(market, "call")
        assert abs(binom - bs) / bs < 0.01

    def test_american_put_ge_european(self):
        amer = american_binomial(100, 100, 0.05, 0.2, 1.0, "put", n_steps=3000, american=True)
        euro = american_binomial(100, 100, 0.05, 0.2, 1.0, "put", n_steps=3000, american=False)
        assert amer >= euro

    def test_american_call_equals_european(self):
        amer = american_binomial(100, 100, 0.05, 0.2, 1.0, "call", n_steps=3000, american=True)
        euro = american_binomial(100, 100, 0.05, 0.2, 1.0, "call", n_steps=3000, american=False)
        assert abs(amer - euro) / max(euro, 1e-12) < 2e-3

    def test_convergence_with_n(self):
        p200 = american_binomial(100, 100, 0.05, 0.2, 1.0, "put", n_steps=200)
        p4000 = american_binomial(100, 100, 0.05, 0.2, 1.0, "put", n_steps=4000)
        assert abs(p200 - p4000) < 0.08

    def test_deep_itm_put(self):
        p = american_binomial(50, 100, 0.05, 0.2, 1.0, "put", n_steps=3000)
        assert p > 45.0

    def test_deep_otm_put(self):
        p = american_binomial(180, 100, 0.05, 0.2, 1.0, "put", n_steps=3000)
        assert p < 0.15


class TestBAW:

    def test_baw_close_to_binomial(self):
        baw = american_baw(100, 100, 0.05, 0.2, 1.0, "put")
        ref = american_binomial(100, 100, 0.05, 0.2, 1.0, "put", n_steps=5000)
        assert abs(baw - ref) / max(ref, 1e-12) < 0.005

    def test_baw_call_no_dividends(self):
        baw = american_baw(100, 100, 0.05, 0.2, 1.0, "call")
        bs = bs_exact_price(MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2), "call")
        assert abs(baw - bs) / bs < 1e-8

    def test_baw_nonnegative(self):
        p = american_baw(100, 100, 0.05, 0.2, 1.0, "put")
        assert np.isfinite(p)
        assert p >= 0.0
