"""Tests for BS <-> Heat equation transforms (21 tests)."""

import numpy as np
import pytest

from chernoffpy.finance.validation import MarketParams, GridConfig
from chernoffpy.finance.transforms import (
    compute_transform_params,
    make_grid,
    make_taper,
    bs_to_heat_initial,
    heat_to_bs_price,
    bs_exact_price,
)


class TestTransformParams:
    """Tests for compute_transform_params."""

    def test_known_values(self, atm_market):
        """Verify k, alpha, beta, t_eff for standard ATM params."""
        k, alpha, beta, t_eff = compute_transform_params(atm_market)
        # k = 2*0.05 / 0.04 = 2.5
        assert k == pytest.approx(2.5)
        # alpha = -(2.5 - 1)/2 = -0.75
        assert alpha == pytest.approx(-0.75)
        # beta = -(2.5 + 1)^2 / 4 = -3.0625
        assert beta == pytest.approx(-3.0625)
        # t_eff = 0.04 * 1 / 2 = 0.02
        assert t_eff == pytest.approx(0.02)

    def test_zero_rate(self):
        """k=0 when r=0, alpha=0.5, beta=-0.25."""
        market = MarketParams(S=100, K=100, T=1.0, r=0.0, sigma=0.20)
        k, alpha, beta, t_eff = compute_transform_params(market)
        assert k == pytest.approx(0.0)
        assert alpha == pytest.approx(0.5)
        assert beta == pytest.approx(-0.25)

    def test_t_eff_proportional(self):
        """t_eff scales linearly with sigma^2 * T."""
        m1 = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        m2 = MarketParams(S=100, K=100, T=2.0, r=0.05, sigma=0.40)
        _, _, _, t1 = compute_transform_params(m1)
        _, _, _, t2 = compute_transform_params(m2)
        # t2/t1 = (0.16 * 2) / (0.04 * 1) = 8
        assert t2 / t1 == pytest.approx(8.0)


class TestGrid:
    """Tests for make_grid."""

    def test_correct_size(self, default_grid):
        """Grid has exactly N points."""
        x = make_grid(default_grid)
        assert len(x) == default_grid.N

    def test_bounds(self, default_grid):
        """Grid spans [-L, L) — min near -L, max < L."""
        x = make_grid(default_grid)
        assert x[0] == pytest.approx(-default_grid.L)
        assert x[-1] < default_grid.L

    def test_uniform_spacing(self, default_grid):
        """All spacings are equal (2L/N)."""
        x = make_grid(default_grid)
        dx = np.diff(x)
        expected = 2 * default_grid.L / default_grid.N
        np.testing.assert_allclose(dx, expected, atol=1e-14)

    def test_endpoint_excluded(self, default_grid):
        """The point x=L is NOT in the grid (needed for FFT periodicity)."""
        x = make_grid(default_grid)
        assert not np.any(np.isclose(x, default_grid.L))


class TestTaper:
    """Tests for make_taper."""

    def test_center_is_one(self, default_grid):
        """Taper is exactly 1.0 in the interior |x| < L - taper_width."""
        x = make_grid(default_grid)
        taper = make_taper(x, default_grid)
        edge = default_grid.L - default_grid.taper_width
        interior = np.abs(x) < edge - 0.01  # small margin
        np.testing.assert_allclose(taper[interior], 1.0, atol=1e-14)

    def test_boundary_near_zero(self, default_grid):
        """Taper approaches 0 near domain boundaries."""
        x = make_grid(default_grid)
        taper = make_taper(x, default_grid)
        # Points very close to L should have small taper
        near_boundary = np.abs(x) > default_grid.L - 0.1
        assert np.all(taper[near_boundary] < 0.05)

    def test_symmetric(self, default_grid):
        """Taper is symmetric: taper(-x) = taper(x)."""
        x = make_grid(default_grid)
        taper = make_taper(x, default_grid)
        taper_flipped = make_taper(-x, default_grid)
        np.testing.assert_allclose(taper, taper_flipped, atol=1e-14)

    def test_smooth_transition(self, default_grid):
        """Taper is monotonically non-increasing from center to edge (right half)."""
        x = make_grid(default_grid)
        taper = make_taper(x, default_grid)
        right = x >= 0
        taper_right = taper[right]
        # Should be monotonically non-increasing
        assert np.all(np.diff(taper_right) <= 1e-14)


class TestInitialCondition:
    """Tests for bs_to_heat_initial."""

    def test_call_nonnegative(self, atm_market, default_grid):
        """Call initial condition u0 >= 0 everywhere."""
        x = make_grid(default_grid)
        u0 = bs_to_heat_initial(x, atm_market, default_grid, "call")
        assert np.all(u0 >= -1e-15)

    def test_put_nonnegative(self, atm_market, default_grid):
        """Put initial condition u0 >= 0 everywhere."""
        x = make_grid(default_grid)
        u0 = bs_to_heat_initial(x, atm_market, default_grid, "put")
        assert np.all(u0 >= -1e-15)

    def test_call_shape(self, atm_market, default_grid):
        """Call IC is positive for x > 0 (ITM) and ~zero for x << 0 (deep OTM)."""
        x = make_grid(default_grid)
        u0 = bs_to_heat_initial(x, atm_market, default_grid, "call")
        # Should be positive for moderate positive x (away from taper)
        mid_itm = (x > 0.5) & (x < 5.0)
        assert np.all(u0[mid_itm] > 0)
        # Should be ~zero for deep OTM
        deep_otm = x < -3.0
        np.testing.assert_allclose(u0[deep_otm], 0, atol=1e-10)

    def test_put_shape(self, atm_market, default_grid):
        """Put IC is positive for x < 0 (ITM) and ~zero for x >> 0 (deep OTM)."""
        x = make_grid(default_grid)
        u0 = bs_to_heat_initial(x, atm_market, default_grid, "put")
        # Should be positive for moderate negative x
        mid_itm = (x < -0.5) & (x > -5.0)
        assert np.all(u0[mid_itm] > 0)
        # Should be ~zero for deep OTM
        deep_otm = x > 3.0
        np.testing.assert_allclose(u0[deep_otm], 0, atol=1e-10)


class TestLowVolNoNaN:
    """Regression: low-vol + wide grid must not produce NaN (inf * taper=0 bug)."""

    def test_low_vol_wide_grid_no_nan(self):
        """sigma=0.03, L=15 — price must be finite (not NaN)."""
        market = MarketParams(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.03)
        config = GridConfig(N=2048, L=15.0, taper_width=2.0)
        x = make_grid(config)
        u0 = bs_to_heat_initial(x, market, config, "call")
        assert not np.any(np.isnan(u0)), "IC contains NaN for sigma=0.03, L=15"
        assert not np.any(np.isinf(u0)), "IC contains Inf for sigma=0.03, L=15"


class TestHeatToBsPrice:
    """Tests for heat_to_bs_price back-transform."""

    def test_nonnegative_output(self, atm_market, default_grid):
        """V(S) >= 0 when u >= 0 (call IC -> back-transform)."""
        x = make_grid(default_grid)
        u0 = bs_to_heat_initial(x, atm_market, default_grid, "call")
        V = heat_to_bs_price(u0, x, atm_market)
        assert np.all(V >= -1e-10), "Negative prices in back-transform"

    def test_zero_u_gives_zero_price(self, atm_market, default_grid):
        """If heat solution is identically zero, all prices are zero."""
        x = make_grid(default_grid)
        u_zero = np.zeros_like(x)
        V = heat_to_bs_price(u_zero, x, atm_market)
        np.testing.assert_allclose(V, 0.0, atol=1e-15)

    def test_low_vol_no_nan(self):
        """Back-transform for low vol must not produce NaN."""
        market = MarketParams(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.03)
        config = GridConfig(N=2048, L=15.0, taper_width=2.0)
        x = make_grid(config)
        u0 = bs_to_heat_initial(x, market, config, "call")
        V = heat_to_bs_price(u0, x, market)
        assert not np.any(np.isnan(V)), "NaN in back-transform for low vol"


class TestBSExact:
    """Tests for bs_exact_price."""

    def test_known_call_value(self, atm_market):
        """ATM call with known parameters has expected price ~10.45."""
        price = bs_exact_price(atm_market, "call")
        assert price == pytest.approx(10.4506, abs=0.01)

    def test_put_call_parity(self, atm_market):
        """C - P = S - K*exp(-rT)."""
        call = bs_exact_price(atm_market, "call")
        put = bs_exact_price(atm_market, "put")
        parity_rhs = atm_market.S - atm_market.K * np.exp(
            -atm_market.r * atm_market.T
        )
        assert call - put == pytest.approx(parity_rhs, abs=1e-10)
