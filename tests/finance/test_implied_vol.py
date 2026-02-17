"""Tests for implied volatility solver."""

import pytest

from chernoffpy.finance.implied_vol import implied_volatility
from chernoffpy.finance.transforms import bs_exact_price
from chernoffpy.finance.validation import MarketParams


class TestImpliedVol:

    def test_roundtrip_atm_call(self):
        sigma = 0.20
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=sigma)
        price = bs_exact_price(market, "call")
        iv = implied_volatility(price, 100, 100, 1.0, 0.05, "call")
        assert abs(iv - sigma) < 1e-6

    def test_roundtrip_atm_put(self):
        sigma = 0.20
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=sigma)
        price = bs_exact_price(market, "put")
        iv = implied_volatility(price, 100, 100, 1.0, 0.05, "put")
        assert abs(iv - sigma) < 1e-6

    def test_roundtrip_otm(self):
        sigma = 0.32
        market = MarketParams(S=90, K=100, T=1.0, r=0.05, sigma=sigma)
        price = bs_exact_price(market, "call")
        iv = implied_volatility(price, 90, 100, 1.0, 0.05, "call")
        assert abs(iv - sigma) < 1e-6

    def test_roundtrip_itm(self):
        sigma = 0.28
        market = MarketParams(S=120, K=100, T=1.0, r=0.05, sigma=sigma)
        price = bs_exact_price(market, "put")
        iv = implied_volatility(price, 120, 100, 1.0, 0.05, "put")
        assert abs(iv - sigma) < 1e-6

    def test_high_vol(self):
        sigma = 1.0
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=sigma)
        price = bs_exact_price(market, "call")
        iv = implied_volatility(price, 100, 100, 1.0, 0.05, "call")
        assert abs(iv - sigma) < 1e-6

    def test_low_vol(self):
        sigma = 0.05
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=sigma)
        price = bs_exact_price(market, "put")
        iv = implied_volatility(price, 100, 100, 1.0, 0.05, "put")
        assert abs(iv - sigma) < 1e-6

    def test_invalid_price_raises(self):
        # Below intrinsic for call: intrinsic = max(S-K*e^{-rT},0) > 0
        with pytest.raises(ValueError, match="below intrinsic"):
            implied_volatility(0.01, 200, 100, 1.0, 0.05, "call")

    def test_zero_price(self):
        iv = implied_volatility(0.0, 100, 200, 1.0, 0.05, "call")
        assert iv == pytest.approx(1e-6)

    def test_newton_roundtrip(self):
        sigma = 0.25
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=sigma)
        price = bs_exact_price(market, "call")
        iv = implied_volatility(price, 100, 100, 1.0, 0.05, "call", method="newton")
        assert abs(iv - sigma) < 1e-6

    def test_negative_price_raises(self):
        with pytest.raises(ValueError, match="market_price must be >= 0"):
            implied_volatility(-1.0, 100, 100, 1.0, 0.05, "call")

    def test_price_above_upper_bound_raises(self):
        with pytest.raises(ValueError, match="above theoretical upper bound"):
            implied_volatility(150.0, 100, 100, 1.0, 0.05, "call")

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be"):
            implied_volatility(5.0, 100, 100, 1.0, 0.05, "call", method="bisect")
