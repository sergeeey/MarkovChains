"""Text reporting helpers for pricing and validation results."""

from __future__ import annotations

from .validation import BarrierPricingResult, PricingResult, ValidationCertificate


def _status_from_rel_error(rel_error: float) -> str:
    return "CERTIFIED" if rel_error < 1e-3 else "WARNING"


def certificate_to_report(
    cert: ValidationCertificate,
    method_name: str = "",
    n_steps: int = 0,
) -> str:
    """Build a human-readable report from ValidationCertificate."""
    status = _status_from_rel_error(cert.rel_error)
    method_line = method_name if method_name else "N/A"
    steps_line = n_steps if n_steps else "N/A"

    lines = [
        "ChernoffPy Validation Certificate",
        "=" * 46,
        f"Method:    {method_line}",
        f"n_steps:   {steps_line}",
        f"BS exact:  {cert.bs_price:.6f}",
        f"Computed:  {cert.computed_price:.6f}",
        "",
        "Error Decomposition:",
        f"  Chernoff error: {cert.chernoff_error:.3e}",
        f"  Domain error:   {cert.domain_error:.3e}",
        f"  Total error:    {cert.abs_error:.3e}",
        f"  Relative error: {cert.rel_error:.4%}",
        "",
        f"Status: {status}",
    ]
    return "\n".join(lines)


def pricing_result_to_report(result: PricingResult) -> str:
    """Build full report from PricingResult."""
    header = [
        "Pricing Result",
        "-" * 46,
        f"Price:     {result.price:.6f}",
        f"Method:    {result.method_name}",
        f"n_steps:   {result.n_steps}",
        (
            f"Market:    S={result.market.S}, K={result.market.K}, "
            f"T={result.market.T}, r={result.market.r}, sigma={result.market.sigma}"
        ),
        "",
    ]
    body = certificate_to_report(
        result.certificate,
        method_name=result.method_name,
        n_steps=result.n_steps,
    )
    return "\n".join(header) + body


def barrier_result_to_report(result: BarrierPricingResult) -> str:
    """Build report for barrier pricing result including parity diagnostic."""
    implied_knockin = result.vanilla_price - result.knockout_price
    parity_gap = abs(result.price - implied_knockin)

    lines = [
        "Barrier Pricing Result",
        "-" * 46,
        f"Barrier type: {result.barrier_type}",
        f"Price:        {result.price:.6f}",
        f"Vanilla:      {result.vanilla_price:.6f}",
        f"Knock-out:    {result.knockout_price:.6f}",
        f"Parity gap:   {parity_gap:.3e}",
        f"Method:       {result.method_name}",
        f"n_steps:      {result.n_steps}",
    ]
    return "\n".join(lines)
