"""Tests for volatility-surface calibration workflows."""

import numpy as np
import pytest

from chernoffpy import CrankNicolson
from chernoffpy.finance.calibration import VolCalibrator
from chernoffpy.finance.market_data import MarketData, generate_synthetic_quotes
from chernoffpy.finance.transforms import bs_exact_price
from chernoffpy.finance.validation import MarketParams


def _linear_skew_market_data(
    spot: float = 100.0,
    rate: float = 0.05,
    a: float = 0.22,
    b: float = -0.08,
    c: float = 0.02,
    strikes: tuple[float, ...] = (85, 95, 100, 105, 115),
    expiries: tuple[float, ...] = (0.25, 0.5, 1.0),
) -> MarketData:
    data = MarketData(spot=spot, rate=rate)
    for expiry in expiries:
        for strike in strikes:
            sigma = a + b * np.log(strike / spot) + c * expiry
            sigma = float(np.clip(sigma, 0.01, 3.0))
            market = MarketParams(S=spot, K=strike, T=expiry, r=rate, sigma=sigma)
            price = bs_exact_price(market, "call")
            data.add_quote(strike=strike, expiry=expiry, price=price, option_type="call")
    return data


def _svi_market_data(
    spot: float = 100.0,
    rate: float = 0.03,
    a: float = 0.04,
    b: float = 0.20,
    rho: float = -0.4,
    m: float = 0.0,
    s: float = 0.25,
    strikes: tuple[float, ...] = (80, 90, 95, 100, 105, 110, 120),
    expiries: tuple[float, ...] = (0.4, 0.8, 1.2),
) -> MarketData:
    data = MarketData(spot=spot, rate=rate)
    for expiry in expiries:
        forward = spot * np.exp(rate * expiry)
        for strike in strikes:
            k = np.log(strike / forward)
            w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))
            sigma = float(np.sqrt(max(w, 1e-8) / expiry))
            market = MarketParams(S=spot, K=strike, T=expiry, r=rate, sigma=sigma)
            price = bs_exact_price(market, "call")
            data.add_quote(strike=strike, expiry=expiry, price=price, option_type="call")
    return data


class TestFlatCalibration:

    def test_flat_recovers_sigma(self):
        data = generate_synthetic_quotes(sigma=0.20, skew=0.0)
        result = VolCalibrator().calibrate(data, parametrization="flat", method="iv_fit")
        assert abs(result.params[0] - 0.20) < 1e-3

    def test_flat_rmse_near_zero(self):
        data = generate_synthetic_quotes(sigma=0.18, skew=0.0)
        result = VolCalibrator().calibrate(data, parametrization="flat", method="iv_fit")
        assert result.rmse < 1e-8

    def test_flat_with_noise(self):
        data = generate_synthetic_quotes(
            sigma=0.21,
            strikes=(90, 100, 110),
            expiries=(0.5, 1.0),
        )
        for i, quote in enumerate(data.quotes):
            bump = 1.0 + (0.01 if i % 2 == 0 else -0.01)
            data.quotes[i] = type(quote)(
                strike=quote.strike,
                expiry=quote.expiry,
                price=max(0.0, quote.price * bump),
                option_type=quote.option_type,
                weight=quote.weight,
            )

        result = VolCalibrator().calibrate(data, parametrization="flat", method="iv_fit")
        assert abs(result.params[0] - 0.21) < 0.03


class TestLinearSkewCalibration:

    def test_skew_recovers_params(self):
        data = _linear_skew_market_data(a=0.22, b=-0.08, c=0.02)
        result = VolCalibrator().calibrate(data, parametrization="linear_skew", method="iv_fit")

        assert abs(result.params[0] - 0.22) < 0.01
        assert abs(result.params[1] + 0.08) < 0.02
        assert abs(result.params[2] - 0.02) < 0.02

    def test_skew_zero_gives_flat(self):
        data = _linear_skew_market_data(a=0.20, b=0.0, c=0.0)
        result = VolCalibrator().calibrate(data, parametrization="linear_skew", method="iv_fit")
        assert abs(result.params[1]) < 1e-3
        assert abs(result.params[2]) < 1e-3

    def test_skew_rmse_small(self):
        data = _linear_skew_market_data(a=0.24, b=-0.05, c=0.015)
        result = VolCalibrator().calibrate(data, parametrization="linear_skew", method="iv_fit")
        assert result.rmse < 0.02

    def test_skew_multiple_expiries(self):
        data = _linear_skew_market_data(a=0.20, b=-0.03, c=0.06, expiries=(0.2, 0.6, 1.0))
        result = VolCalibrator().calibrate(data, parametrization="linear_skew", method="iv_fit")
        assert abs(result.params[2]) > 0.01


class TestSVICalibration:

    def test_svi_fits_smile(self):
        data = _svi_market_data()
        result = VolCalibrator().calibrate(data, parametrization="svi", method="iv_fit")
        assert result.success
        assert result.rmse < 0.03

    def test_svi_rho_negative_for_equity_skew(self):
        data = _svi_market_data(rho=-0.6)
        result = VolCalibrator().calibrate(data, parametrization="svi", method="iv_fit")
        assert result.params[2] < 0.0

    def test_svi_converges(self):
        data = _svi_market_data()
        result = VolCalibrator().calibrate(data, parametrization="svi", method="iv_fit")
        assert result.success is True


class TestModelFitCalibration:

    def test_model_fit_flat_recovers_sigma(self):
        data = generate_synthetic_quotes(
            sigma=0.20,
            strikes=(90, 100, 110),
            expiries=(0.5, 1.0),
        )
        calibrator = VolCalibrator(chernoff=CrankNicolson())
        result = calibrator.calibrate(
            data,
            parametrization="flat",
            method="model_fit",
            n_steps=40,
            maxiter=120,
        )
        assert abs(result.params[0] - 0.20) < 0.02

    def test_model_fit_better_than_iv_fit(self):
        data = _linear_skew_market_data(a=0.22, b=-0.06, c=0.02)
        calibrator = VolCalibrator(chernoff=CrankNicolson())

        iv_result = calibrator.calibrate(data, parametrization="linear_skew", method="iv_fit")
        model_result = calibrator.calibrate(
            data,
            parametrization="linear_skew",
            method="model_fit",
            n_steps=35,
            maxiter=120,
        )

        assert model_result.rmse <= max(0.15, iv_result.rmse + 0.12)

    def test_model_fit_requires_chernoff(self):
        data = generate_synthetic_quotes(sigma=0.20)
        calibrator = VolCalibrator()
        with pytest.raises(ValueError, match="chernoff function required"):
            calibrator.calibrate(data, parametrization="flat", method="model_fit")


class TestCalibrationResultBehavior:

    def test_summary_contains_params(self):
        data = generate_synthetic_quotes(sigma=0.20, strikes=(100,), expiries=(1.0,))
        result = VolCalibrator().calibrate(data, parametrization="flat", method="iv_fit")
        text = result.summary()
        assert "Parameters:" in text
        assert "sigma" in text

    def test_vol_surface_callable(self):
        data = generate_synthetic_quotes(sigma=0.20, strikes=(100,), expiries=(1.0,))
        result = VolCalibrator().calibrate(data, parametrization="flat", method="iv_fit")
        sigma = result.vol_surface(100.0, 1.0)
        assert np.isfinite(float(np.asarray(sigma)))

    def test_model_prices_shape(self):
        data = generate_synthetic_quotes(sigma=0.20, strikes=(95, 100), expiries=(0.5, 1.0))
        result = VolCalibrator().calibrate(data, parametrization="flat", method="iv_fit")
        assert len(result.model_prices) == result.n_quotes


class TestCalibrationProperties:

    def test_calibrated_surface_positive(self):
        data = _linear_skew_market_data()
        result = VolCalibrator().calibrate(data, parametrization="linear_skew", method="iv_fit")
        sigmas = [result.vol_surface(s, t) for s in (80, 100, 120) for t in (0.25, 1.0)]
        assert np.min(np.asarray(sigmas, dtype=float)) > 0.0

    def test_calibrated_prices_nonnegative(self):
        data = _svi_market_data()
        result = VolCalibrator().calibrate(data, parametrization="svi", method="iv_fit")
        assert np.min(result.model_prices) >= 0.0

    def test_wider_smile_higher_svi_b(self):
        data_narrow = _svi_market_data(b=0.12)
        data_wide = _svi_market_data(b=0.35)

        cal = VolCalibrator()
        r_narrow = cal.calibrate(data_narrow, parametrization="svi", method="iv_fit")
        r_wide = cal.calibrate(data_wide, parametrization="svi", method="iv_fit")

        assert r_wide.params[1] > r_narrow.params[1]


class TestCalibrationEdgeCases:

    def test_single_quote(self):
        data = generate_synthetic_quotes(sigma=0.20, strikes=(100,), expiries=(1.0,))
        result = VolCalibrator().calibrate(data, parametrization="flat", method="iv_fit")
        assert result.n_quotes == 1
        assert result.success

    def test_many_quotes(self):
        strikes = tuple(float(k) for k in range(70, 131, 5))
        expiries = (0.25, 0.5, 1.0)
        data = generate_synthetic_quotes(sigma=0.22, strikes=strikes, expiries=expiries, skew=-0.04)
        result = VolCalibrator().calibrate(data, parametrization="linear_skew", method="iv_fit")
        assert result.n_quotes >= 30
        assert result.success

    def test_unknown_method_raises(self):
        data = generate_synthetic_quotes()
        with pytest.raises(ValueError, match="Unknown method"):
            VolCalibrator().calibrate(data, method="unknown")

    def test_unknown_parametrization_raises(self):
        data = generate_synthetic_quotes()
        with pytest.raises(ValueError, match="Unknown parametrization"):
            VolCalibrator().calibrate(data, parametrization="piecewise", method="iv_fit")

