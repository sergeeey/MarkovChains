"""Tests for reporting helpers."""

from chernoffpy import CrankNicolson
from chernoffpy.finance.barrier import BarrierPricer
from chernoffpy.finance.reporting import (
    barrier_result_to_report,
    certificate_to_report,
    pricing_result_to_report,
)
from chernoffpy.finance.european import EuropeanPricer
from chernoffpy.finance.validation import BarrierParams, MarketParams, ValidationCertificate


class TestReporting:

    def test_certificate_report_contains_errors(self):
        cert = ValidationCertificate(
            bs_price=10.0,
            computed_price=9.99,
            abs_error=0.01,
            rel_error=0.001,
            chernoff_error=0.009,
            domain_error=0.001,
        )
        text = certificate_to_report(cert, method_name="CN", n_steps=50)
        assert "Chernoff error" in text
        assert "Domain error" in text
        assert "Total error" in text

    def test_certificate_report_status_certified(self):
        cert = ValidationCertificate(
            bs_price=10.0,
            computed_price=9.9995,
            abs_error=0.0005,
            rel_error=5e-5,
            chernoff_error=4e-4,
            domain_error=1e-4,
        )
        text = certificate_to_report(cert)
        assert "CERTIFIED" in text

    def test_certificate_report_status_warning(self):
        cert = ValidationCertificate(
            bs_price=10.0,
            computed_price=9.8,
            abs_error=0.2,
            rel_error=0.02,
            chernoff_error=0.19,
            domain_error=0.01,
        )
        text = certificate_to_report(cert)
        assert "WARNING" in text

    def test_pricing_result_report(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        result = EuropeanPricer(CrankNicolson()).price(market, n_steps=20, option_type="call")
        text = pricing_result_to_report(result)
        assert "Pricing Result" in text
        assert "Method:" in text
        assert "Status:" in text

    def test_barrier_result_report(self):
        market = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        params = BarrierParams(barrier=90, barrier_type="down_and_in")
        result = BarrierPricer(CrankNicolson()).price(market, params, n_steps=50, option_type="call")
        text = barrier_result_to_report(result)
        assert "Barrier Pricing Result" in text
        assert "Parity gap" in text
