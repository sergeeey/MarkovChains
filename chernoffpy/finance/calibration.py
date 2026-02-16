"""Volatility-surface calibration routines for ChernoffPy finance module."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, minimize

from .local_vol import LocalVolParams, LocalVolPricer, flat_vol
from .market_data import CalibrationResult, MarketData
from .transforms import bs_exact_price
from .validation import MarketParams


class VolCalibrator:
    """Calibrate volatility surfaces to market option quotes."""

    def __init__(self, chernoff=None, grid_config=None):
        self.chernoff = chernoff
        self.grid_config = grid_config

    def calibrate(
        self,
        market_data: MarketData,
        parametrization: str = "flat",
        method: str = "iv_fit",
        **kwargs,
    ) -> CalibrationResult:
        """Calibrate a surface using implied-vol fit or full model fit."""
        if method == "iv_fit":
            return self._calibrate_iv_fit(market_data, parametrization, **kwargs)
        if method == "model_fit":
            return self._calibrate_model_fit(market_data, parametrization, **kwargs)
        raise ValueError(f"Unknown method: {method}")

    def _calibrate_iv_fit(
        self,
        market_data: MarketData,
        parametrization: str,
        **kwargs,
    ) -> CalibrationResult:
        if len(market_data.quotes) == 0:
            raise ValueError("market_data must contain at least one quote")

        ivs = market_data.implied_vols()
        strikes = market_data.strikes
        expiries = market_data.expiries

        if parametrization == "flat":
            return self._fit_flat(market_data, ivs)
        if parametrization == "linear_skew":
            return self._fit_linear_skew(market_data, ivs, strikes, expiries)
        if parametrization == "svi":
            return self._fit_svi(market_data, ivs)
        raise ValueError(f"Unknown parametrization: {parametrization}")

    def _fit_flat(self, market_data: MarketData, ivs: np.ndarray) -> CalibrationResult:
        sigma_opt = float(np.mean(ivs))
        vol_surface = flat_vol(sigma_opt)

        model_prices = self._compute_model_prices_bs(market_data, sigma_opt)
        errors = model_prices - market_data.prices

        return CalibrationResult(
            params=np.array([sigma_opt], dtype=float),
            param_names=["sigma"],
            rmse=float(np.sqrt(np.mean(errors**2))),
            max_error=float(np.max(np.abs(errors))),
            n_quotes=len(market_data.quotes),
            n_iterations=0,
            success=True,
            model_prices=model_prices,
            market_prices=market_data.prices,
            vol_surface=vol_surface,
        )

    def _fit_linear_skew(
        self,
        market_data: MarketData,
        ivs: np.ndarray,
        strikes: np.ndarray,
        expiries: np.ndarray,
    ) -> CalibrationResult:
        s_ref = market_data.spot
        weights = np.array([q.weight for q in market_data.quotes], dtype=float)

        def residuals(theta: np.ndarray) -> np.ndarray:
            a, b, c = theta
            model_ivs = a + b * np.log(strikes / s_ref) + c * expiries
            return weights * (model_ivs - ivs)

        x0 = np.array([float(np.mean(ivs)), 0.0, 0.0], dtype=float)
        fit = least_squares(
            residuals,
            x0,
            bounds=([0.01, -1.0, -1.0], [2.0, 1.0, 1.0]),
            max_nfev=2000,
        )

        a, b, c = [float(v) for v in fit.x]
        vol_surface = _linear_skew_with_time(a, b, c, s_ref)

        model_prices = self._compute_model_prices_bs_surface(market_data, vol_surface)
        errors = model_prices - market_data.prices

        return CalibrationResult(
            params=fit.x.astype(float),
            param_names=["sigma_atm", "skew", "time_slope"],
            rmse=float(np.sqrt(np.mean(errors**2))),
            max_error=float(np.max(np.abs(errors))),
            n_quotes=len(market_data.quotes),
            n_iterations=int(fit.nfev),
            success=bool(fit.success),
            model_prices=model_prices,
            market_prices=market_data.prices,
            vol_surface=vol_surface,
        )

    def _fit_svi(self, market_data: MarketData, ivs: np.ndarray) -> CalibrationResult:
        weights = np.array([q.weight for q in market_data.quotes], dtype=float)
        s0 = market_data.spot
        r = market_data.rate

        def residuals(theta: np.ndarray) -> np.ndarray:
            a, b, rho, m, s = theta
            out = []
            for q, iv, w in zip(market_data.quotes, ivs, weights):
                fwd = s0 * np.exp(r * q.expiry)
                k = np.log(q.strike / fwd)
                w_market = iv**2 * q.expiry
                w_model = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))
                out.append(w * (w_model - w_market))
            return np.array(out, dtype=float)

        avg_var = float(np.mean(ivs**2 * market_data.expiries))
        x0 = np.array([avg_var, 0.1, -0.5, 0.0, 0.1], dtype=float)
        fit = least_squares(
            residuals,
            x0,
            bounds=([-0.5, 0.001, -0.999, -2.0, 0.001], [1.0, 5.0, 0.999, 2.0, 5.0]),
            max_nfev=3000,
        )

        a, b, rho, m, s = [float(v) for v in fit.x]
        vol_surface = _svi_surface(a, b, rho, m, s, s0, r)

        model_prices = self._compute_model_prices_bs_surface(market_data, vol_surface)
        errors = model_prices - market_data.prices

        return CalibrationResult(
            params=fit.x.astype(float),
            param_names=["a", "b", "rho", "m", "s"],
            rmse=float(np.sqrt(np.mean(errors**2))),
            max_error=float(np.max(np.abs(errors))),
            n_quotes=len(market_data.quotes),
            n_iterations=int(fit.nfev),
            success=bool(fit.success),
            model_prices=model_prices,
            market_prices=market_data.prices,
            vol_surface=vol_surface,
        )

    def _calibrate_model_fit(
        self,
        market_data: MarketData,
        parametrization: str,
        **kwargs,
    ) -> CalibrationResult:
        if self.chernoff is None:
            raise ValueError(
                "chernoff function required for method='model_fit'. "
                "Pass CrankNicolson() to VolCalibrator constructor."
            )
        if len(market_data.quotes) == 0:
            raise ValueError("market_data must contain at least one quote")

        if parametrization not in {"flat", "linear_skew"}:
            raise ValueError(f"Unknown parametrization: {parametrization}")

        pricer = LocalVolPricer(self.chernoff, self.grid_config)
        n_steps = int(kwargs.get("n_steps", 60))

        def build_surface(theta: np.ndarray):
            if parametrization == "flat":
                return flat_vol(float(np.clip(theta[0], 0.01, 3.0)))
            return _linear_skew_with_time(
                float(theta[0]),
                float(theta[1]),
                float(theta[2]),
                market_data.spot,
            )

        def objective(theta: np.ndarray) -> float:
            vol_surface = build_surface(theta)
            total = 0.0
            for q in market_data.quotes:
                params = LocalVolParams(
                    S=market_data.spot,
                    K=q.strike,
                    T=q.expiry,
                    r=market_data.rate,
                    vol_surface=vol_surface,
                )
                price = pricer.price(params, n_steps=n_steps, option_type=q.option_type).price
                diff = price - q.price
                total += q.weight * diff * diff
            return float(total)

        iv_start = self._calibrate_iv_fit(market_data, parametrization)
        opt = minimize(
            objective,
            iv_start.params,
            method="Nelder-Mead",
            options={"maxiter": int(kwargs.get("maxiter", 250)), "xatol": 1e-6, "fatol": 1e-8},
        )

        vol_surface = build_surface(np.asarray(opt.x, dtype=float))
        model_prices = np.array(
            [
                pricer.price(
                    LocalVolParams(
                        S=market_data.spot,
                        K=q.strike,
                        T=q.expiry,
                        r=market_data.rate,
                        vol_surface=vol_surface,
                    ),
                    n_steps=n_steps,
                    option_type=q.option_type,
                ).price
                for q in market_data.quotes
            ],
            dtype=float,
        )
        errors = model_prices - market_data.prices

        names = {
            "flat": ["sigma"],
            "linear_skew": ["sigma_atm", "skew", "time_slope"],
        }[parametrization]

        return CalibrationResult(
            params=np.asarray(opt.x, dtype=float),
            param_names=names,
            rmse=float(np.sqrt(np.mean(errors**2))),
            max_error=float(np.max(np.abs(errors))),
            n_quotes=len(market_data.quotes),
            n_iterations=int(getattr(opt, "nfev", 0)),
            success=bool(opt.success),
            model_prices=model_prices,
            market_prices=market_data.prices,
            vol_surface=vol_surface,
        )

    def _compute_model_prices_bs(self, market_data: MarketData, sigma: float) -> np.ndarray:
        prices = []
        sigma = float(np.clip(sigma, 0.01, 3.0))
        for q in market_data.quotes:
            market = MarketParams(
                S=market_data.spot,
                K=q.strike,
                T=q.expiry,
                r=market_data.rate,
                sigma=sigma,
            )
            prices.append(bs_exact_price(market, q.option_type))
        return np.array(prices, dtype=float)

    def _compute_model_prices_bs_surface(self, market_data: MarketData, vol_surface) -> np.ndarray:
        prices = []
        for q in market_data.quotes:
            sigma = float(vol_surface(q.strike, q.expiry))
            sigma = float(np.clip(np.nan_to_num(sigma, nan=0.2, posinf=3.0, neginf=0.01), 0.01, 3.0))
            market = MarketParams(
                S=market_data.spot,
                K=q.strike,
                T=q.expiry,
                r=market_data.rate,
                sigma=sigma,
            )
            prices.append(bs_exact_price(market, q.option_type))
        return np.array(prices, dtype=float)


def _linear_skew_with_time(a: float, b: float, c: float, s_ref: float):
    """Surface sigma(S,t) = a + b*log(S/s_ref) + c*t with clipping."""

    def _vol(S: float | np.ndarray, t: float) -> float | np.ndarray:
        values = a + b * np.log(np.asarray(S) / s_ref) + c * t
        return np.clip(values, 0.01, 3.0)

    return _vol


def _svi_surface(a: float, b: float, rho: float, m: float, s: float, s0: float, r: float):
    """SVI surface converted from total variance to local sigma approximation."""

    def _vol(S: float | np.ndarray, t: float) -> float | np.ndarray:
        tau = max(float(t), 1e-6)
        fwd = s0 * np.exp(r * tau)
        k = np.log(np.asarray(S) / fwd)
        w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))
        sigma = np.sqrt(np.maximum(w, 1e-8) / tau)
        return np.clip(sigma, 0.01, 3.0)

    return _vol

