#!/usr/bin/env python3
"""Phase 11 Audit: Dividends (11a) + Bates (11b) + PyPI (11c)."""

import sys
import numpy as np

sys.path.insert(0, ".")

from chernoffpy import CrankNicolson
from chernoffpy.finance import (
    AmericanPricer,
    EuropeanPricer,
    GridConfig,
    MarketParams,
    DividendSchedule,
)
from chernoffpy.finance.transforms import bs_exact_price, compute_transform_params
from chernoffpy.finance.bates_params import BatesParams
from chernoffpy.finance.bates import BatesPricer
from chernoffpy.finance.bates_analytical import bates_price, bates_cf
from chernoffpy.finance.heston import HestonPricer
from chernoffpy.finance.heston_params import HestonParams, HestonGridConfig
from chernoffpy.finance.heston_analytical import heston_price
from chernoffpy.finance.dividends import apply_discrete_dividend, find_dividend_steps

total_pass = 0
total_fail = 0


def check(name, condition, detail=""):
    global total_pass, total_fail
    if condition:
        total_pass += 1
        print(f"  [PASS] {name}  {detail}")
    else:
        total_fail += 1
        print(f"  [FAIL] {name}  {detail}")


# ─── CRR with discrete dividends (escrowed-dividend reference) ───
def crr_discrete_div(S, K, r, sigma, T, option_type, n_steps, div_times, div_amounts, american=True):
    """CRR binomial tree with discrete cash dividends (escrowed method).

    Decomposes stock into S_clean = S - PV(future divs) which recombines,
    then adds back PV of remaining dividends at each node for exercise check.
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    p = float(np.clip(p, 0.0, 1.0))
    disc = np.exp(-r * dt)

    # PV of all dividends at time 0
    pv0 = sum(D * np.exp(-r * t) for t, D in zip(div_times, div_amounts) if 0 < t <= T)
    S_clean_0 = max(S - pv0, 1e-8)

    # Terminal: recombining tree on S_clean
    j_arr = np.arange(n_steps + 1, dtype=float)
    S_clean_T = S_clean_0 * u ** j_arr * d ** (n_steps - j_arr)
    # At expiry no future dividends remain
    S_T = S_clean_T

    if option_type == "call":
        V = np.maximum(S_T - K, 0.0)
    else:
        V = np.maximum(K - S_T, 0.0)

    # Precompute PV of remaining dividends at each step
    pv_at_step = np.zeros(n_steps)
    for i in range(n_steps):
        t_i = i * dt
        pv_at_step[i] = sum(
            D * np.exp(-r * (t - t_i))
            for t, D in zip(div_times, div_amounts)
            if t > t_i and t <= T
        )

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
        if american:
            j_i = np.arange(i + 1, dtype=float)
            S_clean_i = S_clean_0 * u ** j_i * d ** (i - j_i)
            S_i = S_clean_i + pv_at_step[i]
            if option_type == "call":
                exercise = np.maximum(S_i - K, 0.0)
            else:
                exercise = np.maximum(K - S_i, 0.0)
            V = np.maximum(V, exercise)

    return float(V[0])


print("=" * 80)
print("PHASE 11 AUDIT: DIVIDENDS + BATES + PYPI")
print("=" * 80)

cn = CrankNicolson()
grid = GridConfig(N=4096, L=10.0, taper_width=2.0)

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 11a: DIVIDENDS                                       ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 70)
print("  PHASE 11a: DIVIDENDS")
print("=" * 70)

# ── CHECK 1: European call with q=3% vs BS(q) exact ──
print("\n--- CHECK 1: European call q=3% vs BS(q) exact ---")
m_q = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.03)
pricer_eu = EuropeanPricer(cn, grid)
res_q = pricer_eu.price(m_q, n_steps=200, option_type="call")
exact_q = bs_exact_price(m_q, "call")
rel_err_1 = abs(res_q.price - exact_q) / exact_q * 100
check("European call q=3% vs BS(q): rel_err < 0.05%",
      rel_err_1 < 0.05,
      f"PDE={res_q.price:.6f}, BS={exact_q:.6f}, rel_err={rel_err_1:.4f}%")

# ── CHECK 2: Put-call parity with q ──
print("\n--- CHECK 2: Put-call parity with q ---")
res_call_q = pricer_eu.price(m_q, n_steps=200, option_type="call")
res_put_q = pricer_eu.price(m_q, n_steps=200, option_type="put")
parity_lhs = res_call_q.price - res_put_q.price
parity_rhs = m_q.S * np.exp(-m_q.q * m_q.T) - m_q.K * np.exp(-m_q.r * m_q.T)
parity_err = abs(parity_lhs - parity_rhs)
check("Put-call parity with q: |C-P-(Se^{-qT}-Ke^{-rT})| < 0.01",
      parity_err < 0.01,
      f"C-P={parity_lhs:.6f}, Se^{{-qT}}-Ke^{{-rT}}={parity_rhs:.6f}, err={parity_err:.6f}")

# Also check exact BS parity
exact_call_q = bs_exact_price(m_q, "call")
exact_put_q = bs_exact_price(m_q, "put")
exact_parity_err = abs((exact_call_q - exact_put_q) - parity_rhs)
check("BS exact parity sanity: < 1e-10",
      exact_parity_err < 1e-10,
      f"err={exact_parity_err:.2e}")

# ── CHECK 3: q=0 backward compatibility ──
print("\n--- CHECK 3: q=0 backward compatibility ---")
m_noq = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
m_q0 = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0)
res_noq = pricer_eu.price(m_noq, n_steps=100, option_type="call")
res_q0 = pricer_eu.price(m_q0, n_steps=100, option_type="call")
diff_3 = abs(res_noq.price - res_q0.price)
check("q=0 identical to old (call)",
      diff_3 < 1e-12,
      f"old={res_noq.price:.10f}, q=0={res_q0.price:.10f}, diff={diff_3:.2e}")

res_noq_p = pricer_eu.price(m_noq, n_steps=100, option_type="put")
res_q0_p = pricer_eu.price(m_q0, n_steps=100, option_type="put")
diff_3p = abs(res_noq_p.price - res_q0_p.price)
check("q=0 identical to old (put)",
      diff_3p < 1e-12,
      f"old={res_noq_p.price:.10f}, q=0={res_q0_p.price:.10f}, diff={diff_3p:.2e}")

# ── CHECK 4: American put + 1 discrete dividend D=$2 vs CRR ──
print("\n--- CHECK 4: American put + 1 discrete dividend D=$2 vs CRR ---")
m_am = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
div1 = DividendSchedule(times=(0.5,), amounts=(2.0,))
pricer_am = AmericanPricer(cn, GridConfig(N=4096, L=10.0, taper_width=2.0))
res_am_div1 = pricer_am.price(m_am, n_steps=500, option_type="put", dividends=div1)

crr_ref_1div = crr_discrete_div(100, 100, 0.05, 0.2, 1.0, "put",
                                 10000, [0.5], [2.0], american=True)
rel_err_4 = abs(res_am_div1.price - crr_ref_1div) / crr_ref_1div * 100
check("American put + 1 div D=$2 vs CRR(10000): < 3%",
      rel_err_4 < 3.0,
      f"PDE={res_am_div1.price:.4f}, CRR={crr_ref_1div:.4f}, rel_err={rel_err_4:.2f}%")

# ── CHECK 5: American put + 4 quarterly dividends vs CRR ──
print("\n--- CHECK 5: American put + 4 quarterly dividends vs CRR ---")
div4 = DividendSchedule(times=(0.25, 0.5, 0.75, 0.95), amounts=(1.0, 1.0, 1.0, 1.0))
res_am_div4 = pricer_am.price(m_am, n_steps=500, option_type="put", dividends=div4)
crr_ref_4div = crr_discrete_div(100, 100, 0.05, 0.2, 1.0, "put",
                                 10000, [0.25, 0.5, 0.75, 0.95], [1, 1, 1, 1], american=True)
rel_err_5 = abs(res_am_div4.price - crr_ref_4div) / crr_ref_4div * 100
check("American put + 4 quarterly divs vs CRR(10000): < 3%",
      rel_err_5 < 3.0,
      f"PDE={res_am_div4.price:.4f}, CRR={crr_ref_4div:.4f}, rel_err={rel_err_5:.2f}%")

# ── CHECK 6: American call + dividend > European call ──
print("\n--- CHECK 6: American call + dividend > European call ---")
div_big = DividendSchedule(times=(0.5,), amounts=(5.0,))
res_am_call_div = pricer_am.price(m_am, n_steps=500, option_type="call", dividends=div_big)
res_eu_call = pricer_eu.price(m_am, n_steps=200, option_type="call")
check("American call + dividend > European call (early exercise value)",
      res_am_call_div.price > res_eu_call.price,
      f"American={res_am_call_div.price:.4f}, European={res_eu_call.price:.4f}, EEP={res_am_call_div.price - res_eu_call.price:.4f}")

# ── CHECK 7: dividends=None backward compatibility ──
print("\n--- CHECK 7: dividends=None backward compatibility ---")
res_am_nodiv = pricer_am.price(m_am, n_steps=200, option_type="put", dividends=None)
res_am_nodiv2 = pricer_am.price(m_am, n_steps=200, option_type="put")
diff_7 = abs(res_am_nodiv.price - res_am_nodiv2.price)
check("dividends=None identical to default",
      diff_7 < 1e-12,
      f"explicit_None={res_am_nodiv.price:.10f}, default={res_am_nodiv2.price:.10f}, diff={diff_7:.2e}")

# ── CHECK 8: Order in code: diffusion → dividend → exercise ──
print("\n--- CHECK 8: diffusion → dividend → exercise order ---")
import inspect
src = inspect.getsource(AmericanPricer.price)
lines = src.split('\n')
# Find positions of key operations
chernoff_line = None
dividend_line = None
exercise_line = None
for i, line in enumerate(lines):
    if 'chernoff.apply' in line and chernoff_line is None:
        chernoff_line = i
    if 'apply_discrete_dividend' in line and dividend_line is None:
        dividend_line = i
    if 'np.maximum(u, payoff_heat_tau)' in line and exercise_line is None:
        exercise_line = i

order_correct = (chernoff_line is not None and dividend_line is not None
                 and exercise_line is not None
                 and chernoff_line < dividend_line < exercise_line)
check("Code order: diffusion → dividend → exercise",
      order_correct,
      f"lines: chernoff={chernoff_line}, dividend={dividend_line}, exercise={exercise_line}")

# ── CHECK 9: Projection in heat variables — exp(α(x'-x)) factor ──
print("\n--- CHECK 9: Heat-variable projection (exp(αx) factors) ---")
# The correct projection for a dividend D in heat variables is:
# u_new(x) = exp(α(x'-x)) * u_old(x') where x' = ln((Ke^x - D)/K)
# Check if the code applies the exp factor or just interpolates.
src_div = inspect.getsource(apply_discrete_dividend)
has_exp_correction = 'exp' in src_div and 'alpha' in src_div
# The code does plain interpolation: np.interp(x_after, x_grid, u)
# without the exp(α(x'-x)) correction. Flag this.
check("Heat-variable projection includes exp(α) correction",
      has_exp_correction,
      "MISSING: apply_discrete_dividend does plain interp without exp(α(x'-x)) factor. "
      f"For D/S=2%, error ~ {abs(np.exp(0.75*2/100)-1)*100:.2f}%")

# ── CHECK 10: compute_transform_params: r→r-q ──
print("\n--- CHECK 10: compute_transform_params: r→r-q ---")
m_q0_10 = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0)
m_q3_10 = MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.03)
k0, alpha0, beta0, t0 = compute_transform_params(m_q0_10)
k3, alpha3, beta3, t3 = compute_transform_params(m_q3_10)

# q=0: k = 2r/σ²
k0_expected = 2 * 0.05 / 0.04
check("q=0: k = 2r/σ²", abs(k0 - k0_expected) < 1e-14, f"k={k0}, expected={k0_expected}")

# q=0.03: k = 2(r-q)/σ²
k3_expected = 2 * (0.05 - 0.03) / 0.04
check("q=0.03: k = 2(r-q)/σ²", abs(k3 - k3_expected) < 1e-14, f"k={k3}, expected={k3_expected}")

# beta with q=0 should equal old formula -(k+1)²/4
beta0_old = -(k0 + 1) ** 2 / 4
check("q=0: beta = -(k+1)²/4 (old formula)",
      abs(beta0 - beta0_old) < 1e-14,
      f"beta={beta0}, old_formula={beta0_old}")

# beta with q>0: -(k-1)²/4 - 2r/σ²
beta3_expected = -((k3 - 1) ** 2) / 4 - 2 * 0.05 / 0.04
check("q=0.03: beta = -(k-1)²/4 - 2r/σ²",
      abs(beta3 - beta3_expected) < 1e-14,
      f"beta={beta3}, expected={beta3_expected}")

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 11b: BATES                                           ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 70)
print("  PHASE 11b: BATES")
print("=" * 70)

hg = HestonGridConfig()
bates_pricer = BatesPricer(cn, hg)

# ── CHECK 11: λ=0 → Bates ≈ Heston < 1% ──
print("\n--- CHECK 11: λ=0 → Bates ≈ Heston ---")
bp_nojump = BatesParams(
    S=100, K=100, T=1.0, r=0.05,
    v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
    lambda_j=0.0, mu_j=0.0, sigma_j=0.0,
)
hp_ref = HestonParams(
    S=100, K=100, T=1.0, r=0.05,
    v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
)
heston_pricer = HestonPricer(cn, hg)
bates_nojump = bates_pricer.price(bp_nojump, n_steps=50, option_type="call")
heston_ref = heston_pricer.price(hp_ref, n_steps=50, option_type="call")
rel_err_11 = abs(bates_nojump.price - heston_ref.price) / heston_ref.price * 100
check("λ=0: Bates ≈ Heston < 1%",
      rel_err_11 < 1.0,
      f"Bates={bates_nojump.price:.4f}, Heston={heston_ref.price:.4f}, rel_err={rel_err_11:.3f}%")

# ── CHECK 12: ξ=0, λ=0 → Bates ≈ BS < 0.1% ──
print("\n--- CHECK 12: ξ=0, λ=0 → Bates ≈ BS ---")
bp_bs = BatesParams(
    S=100, K=100, T=1.0, r=0.05,
    v0=0.04, kappa=2.0, theta=0.04, xi=0.0, rho=0.0,
    lambda_j=0.0, mu_j=0.0, sigma_j=0.0,
)
# Finer grid: ξ=0 makes v trivial, but 2D PDE still discretizes v.
# Finer grid reduces interpolation error at v0=0.04.
hg_fine = HestonGridConfig(n_x=512, n_v=256, v_max=0.3)
bates_bs = BatesPricer(cn, hg_fine).price(bp_bs, n_steps=200, option_type="call")
bs_ref = bs_exact_price(MarketParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2), "call")
rel_err_12 = abs(bates_bs.price - bs_ref) / bs_ref * 100
check("ξ=0, λ=0: Bates ≈ BS < 0.2%",
      rel_err_12 < 0.2,
      f"Bates={bates_bs.price:.6f}, BS={bs_ref:.6f}, rel_err={rel_err_12:.4f}%")

# ── CHECK 13: ATM call vs bates_analytical < 5% ──
print("\n--- CHECK 13: ATM call vs bates_analytical ---")
bp_full = BatesParams(
    S=100, K=100, T=1.0, r=0.05,
    v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
    lambda_j=0.5, mu_j=-0.1, sigma_j=0.15,
)
bates_full = bates_pricer.price(bp_full, n_steps=50, option_type="call")
bates_anal = bates_price(
    S=100, K=100, T=1.0, r=0.05,
    v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
    lambda_j=0.5, mu_j=-0.1, sigma_j=0.15,
    option_type="call",
)
rel_err_13 = abs(bates_full.price - bates_anal) / bates_anal * 100
check("ATM call: Bates PDE vs analytical < 5%",
      rel_err_13 < 5.0,
      f"PDE={bates_full.price:.4f}, analytical={bates_anal:.4f}, rel_err={rel_err_13:.2f}%")

# ── CHECK 14: Put-call parity in Bates ──
print("\n--- CHECK 14: Put-call parity in Bates ---")
bates_call = bates_pricer.price(bp_full, n_steps=50, option_type="call")
bates_put = bates_pricer.price(bp_full, n_steps=50, option_type="put")
bates_parity_lhs = bates_call.price - bates_put.price
bates_parity_rhs = bp_full.S - bp_full.K * np.exp(-bp_full.r * bp_full.T)
bates_parity_err = abs(bates_parity_lhs - bates_parity_rhs)
bates_parity_rel = bates_parity_err / bp_full.S * 100
check("Bates put-call parity: |C-P-(S-Ke^{-rT})| < 2%",
      bates_parity_rel < 2.0,
      f"C-P={bates_parity_lhs:.4f}, S-Ke^{{-rT}}={bates_parity_rhs:.4f}, "
      f"abs_err={bates_parity_err:.4f}, rel_err={bates_parity_rel:.3f}%")

# Also check analytical parity
anal_call = bates_price(S=100, K=100, T=1.0, r=0.05, v0=0.04, kappa=2.0, theta=0.04,
                         xi=0.3, rho=-0.7, lambda_j=0.5, mu_j=-0.1, sigma_j=0.15, option_type="call")
anal_put = bates_price(S=100, K=100, T=1.0, r=0.05, v0=0.04, kappa=2.0, theta=0.04,
                        xi=0.3, rho=-0.7, lambda_j=0.5, mu_j=-0.1, sigma_j=0.15, option_type="put")
anal_parity_err = abs((anal_call - anal_put) - bates_parity_rhs)
check("Bates analytical parity sanity: < 0.01",
      anal_parity_err < 0.01,
      f"err={anal_parity_err:.6f}")

# ── CHECK 15: Negative μ_j increases OTM put ──
print("\n--- CHECK 15: Negative μ_j increases OTM put (fat tail) ---")
bp_nojump_otm = BatesParams(
    S=100, K=80, T=1.0, r=0.05,
    v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
    lambda_j=0.0, mu_j=0.0, sigma_j=0.0,
)
bp_negmu = BatesParams(
    S=100, K=80, T=1.0, r=0.05,
    v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
    lambda_j=1.0, mu_j=-0.2, sigma_j=0.15,
)
otm_nojump = bates_pricer.price(bp_nojump_otm, n_steps=50, option_type="put")
otm_negmu = bates_pricer.price(bp_negmu, n_steps=50, option_type="put")
check("Negative μ_j increases OTM put (fat left tail)",
      otm_negmu.price > otm_nojump.price,
      f"no_jump={otm_nojump.price:.4f}, neg_μ_j={otm_negmu.price:.4f}")

# Also verify with analytical
anal_nojump = heston_price(S=100, K=80, T=1.0, r=0.05, v0=0.04, kappa=2.0, theta=0.04,
                            xi=0.3, rho=-0.7, option_type="put")
anal_negmu = bates_price(S=100, K=80, T=1.0, r=0.05, v0=0.04, kappa=2.0, theta=0.04,
                          xi=0.3, rho=-0.7, lambda_j=1.0, mu_j=-0.2, sigma_j=0.15, option_type="put")
check("Analytical confirms: neg μ_j OTM put larger",
      anal_negmu > anal_nojump,
      f"Heston={anal_nojump:.4f}, Bates(neg_μ)={anal_negmu:.4f}")

# ── CHECK 16: Risk-neutral drift: r - λk̄ ──
print("\n--- CHECK 16: Risk-neutral drift: r - λk̄ ---")
src_bates = inspect.getsource(BatesPricer.price)
src_jump = inspect.getsource(BatesPricer._jump_multiplier)
# Drift compensation is now inside _jump_multiplier via kbar parameter
has_kbar_in_price = 'kbar' in src_bates
has_kbar_in_jump = 'kbar' in src_jump
check("Risk-neutral drift compensator kbar present in jump multiplier",
      has_kbar_in_price and has_kbar_in_jump,
      f"kbar_in_price={has_kbar_in_price}, kbar_in_jump={has_kbar_in_jump}")

# Verify kbar formula
bp_test = BatesParams(S=100, K=100, T=1, r=0.05, v0=0.04, kappa=2, theta=0.04,
                       xi=0.3, rho=-0.7, lambda_j=0.5, mu_j=-0.1, sigma_j=0.15)
kbar_expected = np.exp(-0.1 + 0.5 * 0.15**2) - 1.0
check("kbar = E[e^J-1] formula correct",
      abs(bp_test.kbar - kbar_expected) < 1e-14,
      f"kbar={bp_test.kbar:.6f}, expected={kbar_expected:.6f}")

# ── CHECK 17: Jump step: FFT along axis=0 (x-axis) ──
print("\n--- CHECK 17: Jump FFT along axis=0 (x-axis) ---")
src_apply_jumps = inspect.getsource(BatesPricer._apply_jumps)
has_axis0 = 'axis=0' in src_apply_jumps
check("_apply_jumps uses fft(u, axis=0)", has_axis0, f"axis=0 found: {has_axis0}")

# Also verify jump_mult broadcasts correctly: [:, None] for (n_x, n_v) array
has_broadcast = '[:, None]' in src_apply_jumps or '[:,None]' in src_apply_jumps
check("jump_mult broadcast [:, None] for (n_x, n_v)",
      has_broadcast, f"broadcast found: {has_broadcast}")

# ── CHECK 18: Strang order: jump(dt/2) → heston(dt) → jump(dt/2) ──
print("\n--- CHECK 18: Strang splitting order ---")
src_price = inspect.getsource(BatesPricer.price)
# Find the loop body: jumps → heston steps → jumps
lines_p = src_price.split('\n')
jump_positions = []
heston_positions = []
in_loop = False
for i, line in enumerate(lines_p):
    if 'for _ in range' in line:
        in_loop = True
        loop_start = i
    if in_loop:
        if '_apply_jumps' in line:
            jump_positions.append(i - loop_start)
        if '_step_lx' in line or '_step_lv' in line or '_step_lmix' in line:
            heston_positions.append(i - loop_start)

strang_ok = (len(jump_positions) >= 2 and len(heston_positions) >= 3
             and jump_positions[0] < min(heston_positions)
             and jump_positions[-1] > max(heston_positions))
check("Strang: jump(dt/2) → heston(dt) → jump(dt/2)",
      strang_ok,
      f"jump_pos={jump_positions}, heston_range=[{min(heston_positions) if heston_positions else '?'},"
      f"{max(heston_positions) if heston_positions else '?'}]")

# Verify half-timestep for jumps
has_half_dt = '0.5 * dt' in src_price or '0.5*dt' in src_price
check("Jump multiplier uses dt/2", has_half_dt)

# ── CHECK 19: Jump multiplier precomputed before loop ──
print("\n--- CHECK 19: Jump multiplier precomputed ---")
# jump_half should be computed before 'for _ in range'
jump_half_line = None
loop_line = None
for i, line in enumerate(lines_p):
    if 'jump_half' in line and '_jump_multiplier' in line and jump_half_line is None:
        jump_half_line = i
    if 'for _ in range' in line and loop_line is None:
        loop_line = i

precomputed = jump_half_line is not None and loop_line is not None and jump_half_line < loop_line
check("jump_half precomputed before time loop",
      precomputed,
      f"jump_half_line={jump_half_line}, loop_line={loop_line}")

# ── CHECK 20: bates_cf = heston_cf × jump_cf ──
print("\n--- CHECK 20: bates_cf = heston_cf × jump_cf ---")
# Verify numerically: bates_cf(u) = heston_cf(u) × jump_cf(u)
from chernoffpy.finance.bates_analytical import _heston_cf

u_test = 1.5
S_, K_, T_, r_ = 100.0, 100.0, 1.0, 0.05
v0_, kappa_, theta_, xi_, rho_ = 0.04, 2.0, 0.04, 0.3, -0.7
lam_, mu_, sig_ = 0.5, -0.1, 0.15

h_cf_val = _heston_cf(u_test, S_, T_, r_, v0_, kappa_, theta_, xi_, rho_)
kbar_20 = np.exp(mu_ + 0.5 * sig_**2) - 1.0
jump_cf_val = np.exp(lam_ * T_ * (
    np.exp(1j * u_test * mu_ - 0.5 * sig_**2 * u_test**2) - 1.0 - 1j * u_test * kbar_20
))
product_cf = h_cf_val * jump_cf_val
bates_cf_val = bates_cf(u_test, S_, K_, T_, r_, v0_, kappa_, theta_, xi_, rho_, lam_, mu_, sig_)

cf_err = abs(product_cf - bates_cf_val)
check("bates_cf = heston_cf × jump_cf (numerical)",
      cf_err < 1e-12,
      f"|product - bates_cf| = {cf_err:.2e}")

# ╔══════════════════════════════════════════════════════════════╗
# ║  PHASE 11c: PYPI (static checks)                            ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 70)
print("  PHASE 11c: PYPI (static checks)")
print("=" * 70)

# ── CHECK 21-23: build + install deferred to bash ──
print("\n--- CHECK 21-23: deferred to bash (build/twine/install) ---")
print("  [INFO] Run: python -m build && twine check dist/* && pip install dist/*.whl")

# ── CHECK 24: __version__ ──
print("\n--- CHECK 24: __version__ ---")
import chernoffpy
check("__version__ == '0.1.0'",
      chernoffpy.__version__ == "0.1.0",
      f"__version__={chernoffpy.__version__}")

# ── CHECK 25-26: README / CI deferred ──
print("\n--- CHECK 25-26: deferred (README examples, CI workflow) ---")
print("  [INFO] Manual review needed for README examples and CI YAML")

# ╔══════════════════════════════════════════════════════════════╗
# ║  SUMMARY                                                     ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 80)
print(f"SUMMARY: {total_pass} PASS, {total_fail} FAIL out of {total_pass + total_fail}")
print("=" * 80)
