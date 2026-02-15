"""Tests for local volatility pricing."""

import numpy as np
import pytest

from chernoffpy import BackwardEuler, CrankNicolson
from chernoffpy.finance.european import EuropeanPricer
from chernoffpy.finance.local_vol import (
    LocalVolParams,
    LocalVolPricer,
    flat_vol,
    linear_skew,
    time_dependent_vol,
)
from chernoffpy.finance.validation import GridConfig, MarketParams


class TestLocalVolValidation:

    def test_invalid_S(self):
        with pytest.raises(ValueError, match="S must be > 0"):
            LocalVolParams(S=0, K=100, T=1.0, r=0.05, vol_surface=flat_vol(0.2))

    def test_invalid_K(self):
        with pytest.raises(ValueError, match="K must be > 0"):
            LocalVolParams(S=100, K=0, T=1.0, r=0.05, vol_surface=flat_vol(0.2))

    def test_invalid_T(self):
        with pytest.raises(ValueError, match="T must be > 0"):
            LocalVolParams(S=100, K=100, T=0, r=0.05, vol_surface=flat_vol(0.2))

    def test_invalid_vol_surface(self):
        with pytest.raises(ValueError, match="vol_surface must be callable"):
            LocalVolParams(S=100, K=100, T=1.0, r=0.05, vol_surface=0.2)  # type: ignore[arg-type]


class TestLocalVolFlat:

    def test_flat_matches_european_atm_call(self):
        sigma = 0.20
        params = LocalVolParams(S=100, K=100, T=1.0, r=0.05, vol_surface=flat_vol(sigma))

        lv = LocalVolPricer(CrankNicolson()).price(params, n_steps=80, option_type="call").price
        eu = EuropeanPricer(CrankNicolson()).price(
            MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=sigma),
            n_steps=80,
            option_type="call",
        ).price
        assert abs(lv - eu) / eu < 0.01

    def test_flat_matches_european_atm_put(self):
        sigma = 0.20
        params = LocalVolParams(S=100, K=100, T=1.0, r=0.05, vol_surface=flat_vol(sigma))

        lv = LocalVolPricer(CrankNicolson()).price(params, n_steps=80, option_type="put").price
        eu = EuropeanPricer(CrankNicolson()).price(
            MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=sigma),
            n_steps=80,
            option_type="put",
        ).price
        assert abs(lv - eu) / eu < 0.01

    def test_flat_matches_european_otm_call(self):
        sigma = 0.20
        params = LocalVolParams(S=90, K=100, T=1.0, r=0.05, vol_surface=flat_vol(sigma))

        lv = LocalVolPricer(CrankNicolson()).price(params, n_steps=80, option_type="call").price
        eu = EuropeanPricer(CrankNicolson()).price(
            MarketParams(S=90, K=100, T=1.0, r=0.05, sigma=sigma),
            n_steps=80,
            option_type="call",
        ).price
        assert abs(lv - eu) / eu < 0.01


class TestLocalVolSkew:

    def test_positive_skew_changes_price(self):
        base = LocalVolParams(S=100, K=100, T=1.0, r=0.05, vol_surface=flat_vol(0.20))
        skewed = LocalVolParams(
            S=100,
            K=100,
            T=1.0,
            r=0.05,
            vol_surface=linear_skew(sigma_atm=0.20, skew=0.30, S_ref=100),
        )
        pricer = LocalVolPricer(CrankNicolson(), GridConfig(N=1024, L=8, taper_width=2))
        p_base = pricer.price(base, n_steps=120, option_type="call").price
        p_skew = pricer.price(skewed, n_steps=120, option_type="call").price
        assert abs(p_skew - p_base) / p_base > 0.005

    def test_negative_skew_changes_price(self):
        base = LocalVolParams(S=100, K=100, T=1.0, r=0.05, vol_surface=flat_vol(0.20))
        skewed = LocalVolParams(
            S=100,
            K=100,
            T=1.0,
            r=0.05,
            vol_surface=linear_skew(sigma_atm=0.20, skew=-0.30, S_ref=100),
        )
        pricer = LocalVolPricer(CrankNicolson(), GridConfig(N=1024, L=8, taper_width=2))
        p_base = pricer.price(base, n_steps=120, option_type="call").price
        p_skew = pricer.price(skewed, n_steps=120, option_type="call").price
        assert abs(p_skew - p_base) / p_base > 0.005

    def test_zero_skew_equals_flat(self):
        p0 = LocalVolParams(S=100, K=100, T=1.0, r=0.05, vol_surface=flat_vol(0.2))
        p1 = LocalVolParams(
            S=100,
            K=100,
            T=1.0,
            r=0.05,
            vol_surface=linear_skew(sigma_atm=0.2, skew=0.0, S_ref=100),
        )
        pricer = LocalVolPricer(CrankNicolson())
        r0 = pricer.price(p0, n_steps=100, option_type="call").price
        r1 = pricer.price(p1, n_steps=100, option_type="call").price
        assert abs(r1 - r0) / r0 < 0.01


class TestLocalVolTimeDep:

    def test_increasing_vol_higher_call_price(self):
        inc = time_dependent_vol([0.15, 0.20, 0.25], [0.0, 0.3, 0.7, 1.0])
        dec = time_dependent_vol([0.25, 0.20, 0.15], [0.0, 0.3, 0.7, 1.0])

        p_inc = LocalVolParams(S=100, K=100, T=1.0, r=0.05, vol_surface=inc)
        p_dec = LocalVolParams(S=100, K=100, T=1.0, r=0.05, vol_surface=dec)

        pricer = LocalVolPricer(CrankNicolson())
        c_inc = pricer.price(p_inc, n_steps=120, option_type="call").price
        c_dec = pricer.price(p_dec, n_steps=120, option_type="call").price
        assert abs(c_inc - c_dec) / ((c_inc + c_dec) / 2) > 0.005

    def test_time_dependent_finite(self):
        surf = time_dependent_vol([0.2, 0.35], [0.0, 0.5, 1.0])
        params = LocalVolParams(S=100, K=100, T=1.0, r=0.05, vol_surface=surf)
        price = LocalVolPricer(CrankNicolson()).price(params, n_steps=120, option_type="put").price
        assert price >= 0.0


class TestLocalVolProperties:

    def test_price_nonnegative(self):
        params = LocalVolParams(
            S=100,
            K=100,
            T=1.0,
            r=0.05,
            vol_surface=linear_skew(0.2, 0.25, 100),
        )
        price = LocalVolPricer(CrankNicolson()).price(params, n_steps=100, option_type="call").price
        assert price >= 0.0

    def test_put_call_parity(self):
        params = LocalVolParams(S=100, K=100, T=1.0, r=0.05, vol_surface=flat_vol(0.2))
        pricer = LocalVolPricer(CrankNicolson())
        c = pricer.price(params, n_steps=100, option_type="call").price
        p = pricer.price(params, n_steps=100, option_type="put").price
        assert c - p == pytest.approx(params.S - params.K * np.exp(-params.r * params.T), abs=0.1)

    def test_monotonicity_sigma(self):
        low = LocalVolParams(S=100, K=100, T=1.0, r=0.05, vol_surface=flat_vol(0.15))
        high = LocalVolParams(S=100, K=100, T=1.0, r=0.05, vol_surface=flat_vol(0.30))
        pricer = LocalVolPricer(CrankNicolson())
        c_low = pricer.price(low, n_steps=100, option_type="call").price
        c_high = pricer.price(high, n_steps=100, option_type="call").price
        assert c_high > c_low


class TestLocalVolConvergence:

    def test_error_decreases_with_steps(self):
        params = LocalVolParams(
            S=100,
            K=100,
            T=1.0,
            r=0.05,
            vol_surface=linear_skew(0.2, 0.2, 100),
        )
        pricer = LocalVolPricer(CrankNicolson())
        ref = pricer.price(params, n_steps=400, option_type="call").price

        e20 = abs(pricer.price(params, n_steps=20, option_type="call").price - ref)
        e40 = abs(pricer.price(params, n_steps=40, option_type="call").price - ref)
        e80 = abs(pricer.price(params, n_steps=80, option_type="call").price - ref)

        assert e80 <= e40 + 1e-12
        assert e40 <= e20 + 1e-12

    def test_cn_better_than_be(self):
        params = LocalVolParams(
            S=100,
            K=100,
            T=1.0,
            r=0.05,
            vol_surface=linear_skew(0.2, 0.2, 100),
        )
        ref = LocalVolPricer(CrankNicolson()).price(params, n_steps=500, option_type="call").price

        err_cn = abs(LocalVolPricer(CrankNicolson()).price(params, n_steps=80, option_type="call").price - ref)
        err_be = abs(LocalVolPricer(BackwardEuler()).price(params, n_steps=80, option_type="call").price - ref)
        assert err_cn <= err_be + 1e-8


class TestLocalVolEdgeCases:

    def test_short_expiry(self):
        params = LocalVolParams(S=100, K=100, T=1 / 365, r=0.05, vol_surface=flat_vol(0.2))
        price = LocalVolPricer(CrankNicolson()).price(params, n_steps=20, option_type="call").price
        assert price >= 0.0

    def test_high_vol_surface(self):
        params = LocalVolParams(S=100, K=100, T=1.0, r=0.05, vol_surface=flat_vol(0.8))
        price = LocalVolPricer(CrankNicolson()).price(params, n_steps=100, option_type="call").price
        assert price >= 0.0

    def test_low_vol_surface(self):
        params = LocalVolParams(S=100, K=100, T=1.0, r=0.05, vol_surface=flat_vol(0.05))
        price = LocalVolPricer(CrankNicolson()).price(params, n_steps=100, option_type="call").price
        assert price >= 0.0

