"""Tests for adaptive grid utilities and integration with pricing modules."""

import numpy as np
import pytest

from chernoffpy import CrankNicolson
from chernoffpy.finance.adaptive_grid import (
    compute_grid_quality,
    make_snapped_grid,
    make_stretched_config,
    make_stretched_grid,
    snap_grid_to_barrier,
    snap_grid_to_double_barrier,
)
from chernoffpy.finance.barrier import BarrierPricer
from chernoffpy.finance.barrier_analytical import barrier_analytical
from chernoffpy.finance.double_barrier import DoubleBarrierPricer
from chernoffpy.finance.double_barrier_analytical import double_barrier_analytical
from chernoffpy.finance.transforms import make_grid
from chernoffpy.finance.validation import BarrierParams, DoubleBarrierParams, GridConfig, MarketParams


class TestBarrierSnapping:

    def test_single_barrier_snaps_exactly(self):
        cfg = snap_grid_to_barrier(barrier=90, K=100, N=1024, L=6.0)
        x = make_snapped_grid(cfg)
        xb = np.log(90 / 100)
        err = np.min(np.abs(x - xb))
        assert err < 1e-14

    def test_single_barrier_various_levels(self):
        for b in (80, 85, 90, 95):
            cfg = snap_grid_to_barrier(barrier=b, K=100, N=1024, L=6.0)
            x = make_snapped_grid(cfg)
            xb = np.log(b / 100)
            assert np.min(np.abs(x - xb)) < 1e-14

    def test_double_barrier_both_snap(self):
        cfg = snap_grid_to_double_barrier(80, 120, K=100, N=1200, L=6.0)
        x = make_snapped_grid(cfg)
        xl = np.log(80 / 100)
        xu = np.log(120 / 100)
        assert np.min(np.abs(x - xl)) < 1e-14
        assert np.min(np.abs(x - xu)) < 1e-14

    def test_snapped_grid_correct_size(self):
        cfg = snap_grid_to_barrier(90, K=100, N=777, L=6.0)
        x = make_snapped_grid(cfg)
        assert len(x) == 777

    def test_snapped_grid_covers_domain_shifted(self):
        cfg = snap_grid_to_barrier(90, K=100, N=1024, L=6.0)
        x = make_snapped_grid(cfg)
        assert np.isfinite(x[0]) and np.isfinite(x[-1])

    def test_snap_with_barrier_near_edge(self):
        cfg = snap_grid_to_barrier(5, K=100, N=1024, L=6.0)
        x = make_snapped_grid(cfg)
        xb = np.log(5 / 100)
        assert np.min(np.abs(x - xb)) < 1e-14

    def test_snap_preserves_monotonicity(self):
        cfg = snap_grid_to_double_barrier(80, 120, K=100, N=1024, L=6.0)
        x = make_snapped_grid(cfg)
        assert np.all(np.diff(x) > 0)

    def test_no_snap_equals_linspace(self):
        cfg = GridConfig(N=128, L=3.0)
        x = make_grid(cfg)
        y = np.linspace(-3.0, 3.0, 128, endpoint=False)
        np.testing.assert_allclose(x, y, atol=1e-14)


class TestSinhStretching:

    def test_zero_intensity_is_uniform(self):
        cfg = make_stretched_config(N=256, L=4.0, center=0.0, intensity=0.0)
        x = make_stretched_grid(cfg)
        y = np.linspace(-4.0, 4.0, 256, endpoint=False)
        np.testing.assert_allclose(x, y, atol=1e-14)

    def test_stretching_increases_density_at_center(self):
        cfg = make_stretched_config(N=512, L=4.0, center=0.0, intensity=3.0)
        x = make_stretched_grid(cfg)
        dx = np.diff(x)
        c = len(dx) // 2
        assert dx[c] < dx[0]
        assert dx[c] < dx[-1]

    def test_density_ratio_scales_with_intensity(self):
        cfg = make_stretched_config(N=512, L=4.0, center=0.0, intensity=10.0)
        x = make_stretched_grid(cfg)
        q = compute_grid_quality(x, [0.0])
        assert q["dx_ratio"] > 5.0

    def test_grid_symmetric(self):
        cfg = make_stretched_config(N=511, L=4.0, center=0.0, intensity=3.0)
        x = make_stretched_grid(cfg)
        np.testing.assert_allclose(x, -x[::-1], atol=1e-10)

    def test_custom_center(self):
        cfg = make_stretched_config(N=512, L=4.0, center=0.5, intensity=3.0)
        x = make_stretched_grid(cfg)
        dx = np.diff(x)
        idx = np.argmin(np.abs(x - 0.5))
        assert dx[max(0, idx - 1)] <= np.median(dx)

    def test_grid_covers_domain(self):
        cfg = make_stretched_config(N=256, L=4.0, center=0.0, intensity=3.0)
        x = make_stretched_grid(cfg)
        assert x[0] == pytest.approx(-4.0, abs=1e-12)
        assert x[-1] == pytest.approx(4.0, abs=1e-12)

    def test_grid_monotonic(self):
        cfg = make_stretched_config(N=256, L=4.0, center=0.1, intensity=4.0)
        x = make_stretched_grid(cfg)
        assert np.all(np.diff(x) > 0)


class TestGridQuality:

    def test_uniform_ratio_is_one(self):
        x = np.linspace(-5, 5, 1000, endpoint=False)
        q = compute_grid_quality(x, [0.0])
        assert q["dx_ratio"] == pytest.approx(1.0, abs=1e-12)

    def test_stretched_ratio_gt_one(self):
        x = make_stretched_grid(make_stretched_config(N=512, L=4.0, intensity=3.0))
        q = compute_grid_quality(x, [0.0])
        assert q["dx_ratio"] > 1.0

    def test_snap_error_is_zero(self):
        cfg = snap_grid_to_barrier(90, K=100, N=512, L=6.0)
        x = make_snapped_grid(cfg)
        xb = np.log(90 / 100)
        q = compute_grid_quality(x, [xb])
        assert q["snap_errors"][xb] < 1e-14


class TestBarrierPricingWithSnapping:

    def test_doc_snapped_not_worse(self):
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        bp = BarrierParams(barrier=90, barrier_type="down_and_out")

        base = BarrierPricer(CrankNicolson(), GridConfig(N=1024, L=8.0, taper_width=2.0))
        snap_cfg = snap_grid_to_barrier(90, K=100, N=1024, L=8.0, taper_width=2.0)
        snapped = BarrierPricer(CrankNicolson(), snap_cfg)

        p_base = base.price(m, bp, n_steps=80, option_type="call").price
        p_snap = snapped.price(m, bp, n_steps=80, option_type="call").price
        ref = barrier_analytical(m, bp, "call")

        e_base = abs(p_base - ref) / ref
        e_snap = abs(p_snap - ref) / ref
        assert e_snap <= e_base + 1e-8

    def test_doc_snapped_near_spot(self):
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        bp = BarrierParams(barrier=99, barrier_type="down_and_out")
        snap_cfg = snap_grid_to_barrier(99, K=100, N=1024, L=8.0, taper_width=2.0)
        pr = BarrierPricer(CrankNicolson(), snap_cfg)
        price = pr.price(m, bp, n_steps=80, option_type="call").price
        assert np.isfinite(price)
        assert price >= 0.0

    def test_double_barrier_snapped(self):
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        bp = DoubleBarrierParams(80, 120, "double_knock_out")

        base = DoubleBarrierPricer(CrankNicolson(), GridConfig(N=1024, L=8.0, taper_width=2.0))
        snap_cfg = snap_grid_to_double_barrier(80, 120, K=100, N=1024, L=8.0, taper_width=2.0)
        snapped = DoubleBarrierPricer(CrankNicolson(), snap_cfg)

        p_base = base.price(m, bp, n_steps=80, option_type="call").price
        p_snap = snapped.price(m, bp, n_steps=80, option_type="call").price
        ref = double_barrier_analytical(100, 100, 80, 120, 0.05, 0.2, 1.0, "call")

        e_base = abs(p_base - ref) / ref
        e_snap = abs(p_snap - ref) / ref
        assert e_snap <= e_base + 1e-8

    def test_snapped_does_not_break_european_grid_default(self):
        a = make_grid(GridConfig(N=256, L=4.0, taper_width=1.0))
        b = np.linspace(-4.0, 4.0, 256, endpoint=False)
        np.testing.assert_allclose(a, b, atol=1e-14)

    def test_snapped_preserves_inout_parity(self):
        m = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        cfg = snap_grid_to_barrier(90, K=100, N=1024, L=8.0, taper_width=2.0)
        pr = BarrierPricer(CrankNicolson(), cfg)

        out_r = pr.price(m, BarrierParams(90, "down_and_out"), n_steps=80, option_type="call")
        in_r = pr.price(m, BarrierParams(90, "down_and_in"), n_steps=80, option_type="call")
        assert abs(in_r.price + out_r.price - out_r.vanilla_price) < 0.01
