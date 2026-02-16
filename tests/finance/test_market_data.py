"""Tests for market-data structures used by calibration."""

import numpy as np
import pytest

from chernoffpy.finance.market_data import (
    CalibrationResult,
    MarketData,
    MarketQuote,
    generate_synthetic_quotes,
)


class TestMarketQuote:

    def test_valid_quote(self):
        q = MarketQuote(strike=100, expiry=1.0, price=10.0, option_type="call", weight=2.0)
        assert q.strike == 100
        assert q.expiry == 1.0
        assert q.price == 10.0
        assert q.option_type == "call"
        assert q.weight == 2.0

    def test_negative_strike_raises(self):
        with pytest.raises(ValueError, match="strike must be > 0"):
            MarketQuote(strike=-1, expiry=1.0, price=1.0)

    def test_negative_price_raises(self):
        with pytest.raises(ValueError, match="price must be >= 0"):
            MarketQuote(strike=100, expiry=1.0, price=-1.0)

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            MarketQuote(strike=100, expiry=1.0, price=1.0, option_type="straddle")  # type: ignore[arg-type]


class TestMarketData:

    def test_add_quotes(self):
        data = MarketData(spot=100, rate=0.05)
        data.add_quote(strike=100, expiry=1.0, price=10.0)
        data.add_quote(strike=110, expiry=1.0, price=6.0, option_type="put")
        assert len(data.quotes) == 2

    def test_implied_vols_roundtrip(self):
        data = generate_synthetic_quotes(
            spot=100,
            rate=0.05,
            sigma=0.20,
            strikes=(100,),
            expiries=(0.5, 1.0),
            option_type="call",
        )
        ivs = data.implied_vols()
        assert np.all(np.isfinite(ivs))
        assert np.max(np.abs(ivs - 0.20)) < 1e-6

    def test_strikes_expiries_arrays(self):
        data = MarketData(spot=100, rate=0.05)
        data.add_quote(strike=95, expiry=0.5, price=8.0)
        data.add_quote(strike=105, expiry=1.0, price=7.0)
        assert np.array_equal(data.strikes, np.array([95.0, 105.0]))
        assert np.array_equal(data.expiries, np.array([0.5, 1.0]))
        assert np.array_equal(data.prices, np.array([8.0, 7.0]))

    def test_empty_quotes(self):
        data = MarketData(spot=100, rate=0.05)
        assert len(data.quotes) == 0
        assert data.strikes.size == 0
        assert data.expiries.size == 0
        assert data.prices.size == 0


class TestSyntheticData:

    def test_generate_flat(self):
        data = generate_synthetic_quotes(sigma=0.22, skew=0.0)
        ivs = data.implied_vols()
        assert np.max(np.abs(ivs - 0.22)) < 1e-6

    def test_generate_skew(self):
        data = generate_synthetic_quotes(sigma=0.20, skew=-0.08)
        ivs = data.implied_vols()
        assert np.std(ivs) > 1e-3


class TestCalibrationResult:

    def test_summary_contains_metrics(self):
        result = CalibrationResult(
            params=np.array([0.2]),
            param_names=["sigma"],
            rmse=0.01,
            max_error=0.02,
            n_quotes=5,
            n_iterations=10,
            success=True,
            model_prices=np.array([1.0, 2.0]),
            market_prices=np.array([1.0, 2.0]),
            vol_surface=lambda s, t: 0.2,
        )
        text = result.summary()
        assert "ChernoffPy Calibration Result" in text
        assert "sigma" in text
        assert "RMSE:" in text
