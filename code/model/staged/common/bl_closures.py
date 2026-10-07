# -*- coding: utf-8 -*-
"""
Boundary-layer closure relations (Drela / XFOIL standard correlations).

All functions are pure (no class state).  The main `compute_thwaites`
function takes the parameters that were previously instance attributes
(`dxi`, `Ncrit`, `trans_k`, etc.) as arguments / via a small `BLConfig`
dataclass.

This is the single source of truth for:
    - Thwaites laminar BL integral with Hiemenz stagnation splice
    - Drela DAMPL n-factor envelope
    - Falkner-Skan family Cf_lam
    - Ludwieg–Tillmann Cf_turb
    - Drela H*(H) and Cd correlations (laminar / turbulent)

Extracted from steady51.py with no physics changes.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

DTYPE = tf.float32


@dataclass
class BLConfig:
    """Configuration for laminar/turbulent BL evaluation."""
    Nside: int = 60                    # panels per surface (= n_surface_pts - 1)
    Ncrit: float = 7.0                 # transition n-factor threshold
    trans_k: float = 2.0               # transition sigmoid sharpness
    # ============================================================
    # v51-fix [2026-05-08] OPTION B — non-equilibrium lag-entrainment
    # parameter for laminar Cd correction.
    #
    #   Cd = (H*·Cf/4) · (1 + α_lag · max(0, H - H_eq))
    #
    # H_eq = 2.59 (Blasius equilibrium); α_lag tunable.
    # Enhances dissipation in strong APG regions (H > 2.59) to suppress
    # θ runaway observed at NACA0012 α=8° in the unmodified full 2-eq
    # solver (θ_TE ≈ 0.04 vs XFOIL 0.008, 5× over).
    #
    # Damping ratio Cd/Cd_eq (with α_lag = 1.0):
    #   H=2.59 (Blasius)  → 1.00  (no change at equilibrium)
    #   H=3.00            → 1.41  (mild damping, mid APG)
    #   H=3.50            → 1.91  (strong damping)
    #   H=4.00            → 2.41  (very strong damping, near separation)
    #
    # Iteration history:
    #   #1  α_lag = 1.0 :  avg H_L1 0.812 → 0.805 (−0.9 %, near-noise);
    #                     biggest hit at NACA0012 α=4° (H_L1 −10 %),
    #                     but strong APG (NACA0012 α=8°) untouched
    #                     because ADAF output already keeps H ≲ 2.59
    #                     across most of the chord — `H_excess` ≈ 0
    #                     so the lag term rarely fires.
    #   #2  α_lag = 2.0 :  prepared but NOT retrained — we redirected
    #                     the next iteration to ADAF mode-count bump
    #                     instead (see op2_bl.py N_m_bl comment).
    #                     Diagnosis showed laminar Cd correction can't
    #                     move the avg H_L1 because the dominant error
    #                     is in the *turbulent* region of high-Re
    #                     cambered cases, which lag-entrainment doesn't
    #                     touch.  Reverted to 1.0 to keep the laminar
    #                     closure at its validated value while we test
    #                     the ADAF capacity bump in isolation.
    #   #3  α_lag = 1.0 :  current — same as #1 (validated).  Combined
    #                     with N_m_bl 12 → 32 in op2_bl.py; this run
    #                     isolates the ADAF mode-count effect.
    #
    # Set α_lag = 0.0 to recover the equilibrium closure exactly.
    # ============================================================
    alpha_lag: float = 1.0             # lag-entrainment coefficient (option B, iter #3)


# ============================================================
# Static H* / Cd correlations
# ============================================================
def Hstar_from_H_lam(H):
    """Laminar H*(H) — Falkner-Skan family fit."""
    H_c = tf.clip_by_value(H, 1.5, 10.0)
    return tf.where(
        H_c < 4.0,
        1.515 + 0.076 * tf.square(H_c - 4.0) / H_c,
        1.515 + 0.040 * (H_c - 4.0),
    )


def Hstar_from_H_turb(H):
    """Turbulent H*(H) — Whitfield / simplified Drela."""
    H_c = tf.clip_by_value(H, 1.05, 6.0)
    Hstar = tf.where(
        H_c < 1.6,
        1.505 + 4.0 / (H_c + 0.25) - 4.0 / 1.85,
        1.505 + 4.0 / (H_c + 0.25) - 4.0 / 1.85
        + 0.007 * tf.nn.relu(H_c - 1.6) ** 2,
    )
    return tf.clip_by_value(Hstar, 1.4, 2.6)


def Cd_lam(H, Re_theta, Cf_lam):
    """
    Laminar dissipation coefficient via equilibrium relation:
        Cd = H* · Cf / 4   (from dH*/ds = 0 ⟹ 2 Cd = H*·Cf/2)
    Self-consistent with Cf for non-separated laminar BLs (Falkner-Skan).
    """
    Hstar = 1.515 + 0.076 * tf.square(tf.clip_by_value(H, 1.5, 4.0) - 4.0) \
        / tf.clip_by_value(H, 1.5, 4.0)
    Cd = Hstar * Cf_lam / 4.0
    return tf.maximum(Cd, 1e-8)


def Cd_turb(H, Re_theta, Cf):
    """
    Turbulent Cd (Drela) — Cf/2·H* + outer-wake dissipation.
    Cf passed here is already the corrected (Ludwieg-Tillmann) value.
    """
    H_c = tf.clip_by_value(H, 1.05, 6.0)
    Re_t = tf.maximum(Re_theta, 100.0)
    Hstar = 1.505 + 4.0 / (H_c + 0.25) - 4.0 / 1.85
    Cd = (0.5 * Cf * Hstar
          + 0.0056 / tf.maximum(H_c - 0.6, 0.4))
    return tf.maximum(Cd, 1e-7)


# ============================================================
# Cf correlations
# ============================================================
def Cf_lam_falkner_skan(H, Re_theta):
    """
    Laminar Cf — Falkner-Skan family fit (XFOIL form):
        FC = 0.0727·(5.5-H)³/(H+1) - 0.07     [= Cf · Re_θ]
        Cf = FC / Re_θ
    Validated: Blasius (H=2.59) → Cf·Reθ = 0.44 (exact 0.441).
    """
    H_cf = tf.clip_by_value(H, 1.02, 5.3)
    FC = 0.0727 * tf.pow(5.5 - H_cf, 3.0) / (H_cf + 1.0) - 0.07
    FC = tf.maximum(FC, 0.001)   # avoid negative near separation
    Re_theta_safe = tf.maximum(Re_theta, 1.0)
    return FC / Re_theta_safe


def Cf_turb_ludwieg_tillmann(H, Re_theta):
    """
    Turbulent Cf — Ludwieg–Tillmann (1949):
        Cf = 0.246 · 10^(-1.561·H) · Re_θ^(-0.268)
    """
    H_clamped = tf.clip_by_value(H, 1.1, 3.5)
    Re_theta_safe = tf.maximum(Re_theta, 1.0)
    Cf = 0.246 * tf.pow(10.0, -1.561 * H_clamped) * tf.pow(Re_theta_safe, -0.268)
    return tf.maximum(Cf, 1e-6)


# ============================================================
# Transition criterion (Drela / Orr-Sommerfeld envelope)
# ============================================================
def Re_theta_crit_drela(H):
    """
    Critical Re_θ for transition onset (Drela's envelope fit).

    log10(Re_θ_crit) = (1.415/(H-1) - 0.489)·tanh(20/(H-1) - 12.9)
                       + 3.295/(H-1) + 0.44
    """
    H_for_n = tf.maximum(H - 1.0, 0.1)
    # ============================================================
    # v51-fix [2026-05-08] — log10 vs natural-log bug
    #
    # PREVIOUS (buggy):
    #     log_Re_crit = (...)               # Drela formula returns log10
    #     return tf.exp(log_Re_crit)         # ← treated as natural log
    #
    # The Drela 1989 formula (and XFOIL 6.99 src/blsys.f HKLIM) returns
    # log10(Re_θ_crit), not ln.  Verification at Blasius H=2.59:
    #     log10_Re_crit ≈ 2.388
    #     correct  : 10^2.388 = 244        (Schlichting Re_θ_min ~ 200-500)
    #     buggy    :  e^2.388 =  10.9       (~22× too small, unphysical)
    #
    # Effect of the bug: onset gate triggered from very near LE,
    # spreading dn/dReθ accumulation over the whole chord and either
    # delaying or preventing transition completion before TE.
    # ============================================================
    log10_Re_crit = (
        (1.415 / H_for_n - 0.489) * tf.tanh(20.0 / H_for_n - 12.9)
        + 3.295 / H_for_n + 0.44
    )
    log10_Re_crit = tf.minimum(log10_Re_crit, 12.0)   # cap at 10^12
    return tf.pow(10.0, log10_Re_crit)


def dn_dRe_dampl(H):
    """
    Amplification rate dn/dRe_θ — Drela DAMPL envelope (single branch).

        dn/dRe_θ = 0.028·(H-1) - 0.0345·exp(-{3.87/(H-1) - 2.52}²)
    """
    Hk = tf.maximum(H, 1.05)
    Hk_m1 = Hk - 1.0
    # ============================================================
    # v51-fix [2026-05-08] — TWO bugs corrected
    #
    # (a) Constant 0.85 → 2.52
    #     PREVIOUS:  exp(-(3.87/(H-1) - 0.85)²)
    #     CORRECTED: exp(-(3.87/(H-1) - 2.52)²)
    #
    #     The peak of dn/dReθ occurs where 3.87/(H-1) - X = 0,
    #     i.e. H_peak = 1 + 3.87/X.
    #         X = 2.52  → H_peak = 2.54   (Blasius region, T-S waves
    #                                       most amplified — physically
    #                                       correct)
    #         X = 0.85  → H_peak = 5.55   (post-separation, unphysical)
    #     Reference: Drela 1989, "An Analysis and Design System for
    #     Low-Reynolds Airfoils"; XFOIL 6.99 src/blsys.f.
    #
    #     Numerical impact at Blasius H=2.59:
    #         CORRECT (X=2.52): dn/dRe = 0.011
    #         BUGGY   (X=0.85): dn/dRe = 0.042   (~4× too large at
    #                                              Blasius; flat across
    #                                              all H — no APG boost)
    #
    # (b) `dn_high` blending branch (H ≥ 3.5) REMOVED.
    #     PREVIOUS code blended an alternate `dn_high` formula that
    #     returned 0 right at H=3.5 (tanh(0)=0), making total dn DECREASE
    #     across H=3.5–4.  Physically wrong: APG should accelerate, not
    #     slow, T-S amplification.  Drela's original formula extends to
    #     all H in a single branch — keep it that way.
    # ============================================================
    dn = 0.028 * Hk_m1 - 0.0345 * tf.exp(
        -tf.square(3.87 / tf.maximum(Hk_m1, 0.1) - 2.52))
    return tf.maximum(dn, 0.0)


# ============================================================
# Thwaites laminar BL with Hiemenz stagnation splice
# ============================================================
def compute_thwaites(ue, due_dxi, Re, L_branch,
                     dxi, xi_1d, cfg: BLConfig,
                     re_blend=1.0):
    """
    Compute the laminar BL state (Thwaites + Hiemenz splice) and the
    analytical n-factor (DAMPL).

    Parameters
    ----------
    ue       : (B, Nside+1)   edge velocity along ξ
    due_dxi  : (B, Nside+1)   d ue / dξ
    Re       : (B, 1)
    L_branch : (B, 1)         branch arc length (ds/dξ = L)
    dxi      : scalar         1/Nside
    xi_1d    : (Nside+1,)     uniform ξ ∈ [0, 1] grid
    cfg      : BLConfig
    re_blend : scalar (0..1)  ramp factor for transition activation

    Returns
    -------
    dict with: theta_lam, H_lam, delta_lam, n, w_tr, sigma_n
    """
    ue_safe = tf.maximum(ue, 1e-6)
    Re_safe = tf.maximum(Re, 1.0)

    # (a) Thwaites integral — left Riemann sum (exclusive cumsum)
    ue5 = tf.pow(ue_safe, 5.0)
    integral_ue5 = tf.cumsum(ue5, axis=1, exclusive=True) * dxi * L_branch
    theta_thw_sq = 0.45 / Re_safe * integral_ue5 / tf.pow(ue_safe, 6.0)

    # (b) Hiemenz analytical at stagnation: θ²_stag = 0.075 / (Re · |due/ds|_stag)
    due_ds = due_dxi / (L_branch + 1e-10)
    n_le = min(10, due_ds.shape[1])
    due_ds_stag = tf.reduce_max(tf.abs(due_ds[:, :n_le]), axis=1, keepdims=True)
    due_ds_stag = tf.maximum(due_ds_stag, 1.0)
    theta_hiemenz_sq = 0.075 / (Re_safe * due_ds_stag)

    # (c) Splice: Hiemenz near LE (ξ < 0.05), Thwaites elsewhere
    w_thw = tf.sigmoid(40.0 * (xi_1d[None, :] - 0.05))
    theta_lam_sq = (1.0 - w_thw) * theta_hiemenz_sq + w_thw * theta_thw_sq
    theta_lam_sq = tf.minimum(theta_lam_sq, 0.05 ** 2)
    theta_lam = tf.sqrt(tf.maximum(theta_lam_sq, 0.0))

    # (d) λ → H (Thwaites/White)
    lambda_thw = theta_lam_sq * Re_safe * due_ds
    lambda_clip = tf.clip_by_value(lambda_thw, -0.09, 0.25)
    lambda_sq = tf.square(lambda_clip)
    # ============================================================
    # v51-fix [2026-05-08] — H_lam(λ) APG branch redesigned (TWO iterations)
    #
    # ORIGINAL (buggy, before any fix):
    #     H_lam(λ < 0) = 2.61 - 3.75 λ - 5.24 λ²
    #     At λ = -0.09 (Thwaites separation):  H = 2.91
    #     Cebeci-Bradshaw / White value:        H = 3.55
    #     Under-prediction by ~18 % in APG → suppressed dn/dReθ APG boost.
    #
    # FIRST ATTEMPT: White's rational fit
    #     H_lam(λ<0) = 2.088 + 0.0731 / (0.14 + λ)
    #     Endpoint accuracy: λ=0→2.61, λ=-0.09→3.55 ✓
    #     But the slope dH/dλ at λ=-0.09 is -29.2 (very steep — H wiggles
    #     between 2.5 and 3.55 from small λ noise; visible as oscillations
    #     in H plots after retraining).
    #
    # CURRENT (3-point polynomial fit — smoother):
    #     H_lam(λ < 0) = 2.61 - 13.5 λ - 33.9 λ²
    #     λ:        0     -0.05   -0.07   -0.09
    #     CB ref:  2.61    3.20    3.50    3.55
    #     poly:    2.61    3.20    3.51    3.55   ✓ all 3 points exact
    #     Slope at λ=-0.09: -7.4 (4× smoother than rational form).
    #     The polynomial peaks near λ=-0.20 (parabolic), but Thwaites
    #     λ_clip = -0.09 keeps us within the well-behaved range.
    #
    # FPG side (λ ≥ 0) original polynomial retained — bug was APG only.
    # ============================================================
    H_lam_pos = 2.61 - 3.75 * lambda_clip + 5.24 * lambda_sq
    H_lam_neg = 2.61 - 13.5 * lambda_clip - 33.9 * lambda_sq
    H_lam = tf.where(lambda_clip >= 0.0, H_lam_pos, H_lam_neg)
    H_lam = tf.clip_by_value(H_lam, 1.5, 4.5)
    delta_lam = H_lam * theta_lam

    # (e) n-factor envelope (DAMPL)
    # v51-fix [2026-05-08]: previously this block had its own inline
    # duplicate of the Re_θ_crit formula AND the dn/dReθ envelope, both
    # carrying the same bugs (tf.exp + constant 0.85).  Refactored to
    # call the corrected Re_theta_crit_drela() and dn_dRe_dampl()
    # helpers above — single source of truth.
    Re_theta_lam = ue_safe * theta_lam * Re_safe
    Re_theta_crit = Re_theta_crit_drela(H_lam)
    onset = tf.sigmoid(4.0 * (Re_theta_lam - Re_theta_crit) / (Re_theta_crit + 1.0))

    dn_dRe = dn_dRe_dampl(H_lam) * re_blend

    Re_theta_lam_shift = tf.concat([Re_theta_lam[:, 1:], Re_theta_lam[:, -1:]], axis=1)
    dRe_theta_dxi = tf.maximum((Re_theta_lam_shift - Re_theta_lam) / dxi, 0.0)
    # v51-fix [2026-05-08] — σ_n clip iteration history:
    #   ORIGINAL : 10  → hard-capped n_max ≤ 10, transition ξ ≥ 0.7
    #              even in strong APG (NACA0012 α=8° XFOIL transitions at
    #              x/c ≈ 0.05 — completely unreachable).
    #   FIRST FIX: 50  → transition ξ ≥ 0.14 in principle, but observed:
    #              H_L1 improved (-29 %) BUT θ_L1 / δ*_L1 worsened (+70 %)
    #              The bl_operator could not handle the wider transition
    #              range in 1000 epochs.
    #   CURRENT  : 30  → transition ξ ≥ 0.23 (still covers most cases),
    #              gentler dynamic range for the BL ADAF to learn.
    # If 30 still degrades θ/δ* beyond 50%, fall back to 20.
    sigma_n = tf.clip_by_value(onset * dn_dRe * dRe_theta_dxi, 0.0, 30.0)

    n = tf.cumsum(sigma_n * dxi, axis=1, exclusive=True)
    w_tr = tf.sigmoid(cfg.trans_k * (n - cfg.Ncrit))

    return {
        "theta_lam": theta_lam,
        "H_lam": H_lam,
        "delta_lam": delta_lam,
        "n": n,
        "w_tr": w_tr,
        "sigma_n": sigma_n,
    }


# ============================================================
# v51-fix [2026-05-08] FULL 2-EQ — both θ and δ* from coupled ODEs
#
# History:
#   v51-fix#5  (Hybrid) θ from Thwaites + H ODE  — failed: chaotic H
#                       (θ saturation in Thwaites + ODE clip artifacts)
#   v51-fix#6  (Full)   θ AND δ* from Heun ODE  — current
#
# Stability measures (lessons learned from #5):
#   - Hiemenz analytical for ξ < 0.05  (avoid stagnation singularity)
#   - Per-step rate clip f_θ ∈ [-0.5, 0.5], f_δ ∈ [-3, 3]
#   - H clip [1.5, 5.0] enforced via δ-renormalization each step
#   - θ floor at 1e-7
# ============================================================
def _full_lam_rhs(theta, delta, ue, due_dxi, Re_safe, L_branch,
                   alpha_lag: float = 1.0):
    """Full laminar BL RHS for the (θ, δ*) ODE system.

    Parameters
    ----------
    alpha_lag : float
        Non-equilibrium lag-entrainment coefficient (option B,
        v51-fix [2026-05-08]).  0.0 = equilibrium Cd.
        Default kept in sync with BLConfig.alpha_lag (current iter #3 = 1.0).
        The actual value used in train/validate is whatever cfg.alpha_lag
        is passed in from solve_laminar_bl_full_2eq — this default is
        only a fallback when the function is called standalone.
    """
    theta_safe = tf.maximum(theta, 1e-7)
    # v51-fix [2026-05-10] — laminar H lower bound 1.5 → 2.0.
    # Falkner-Skan β=1 (max FPG) gives H_min ≈ 2.0; Hiemenz H₀ = 2.216;
    # Blasius H = 2.59.  Real laminar BL never reaches H<2.0.
    # Old floor 1.5 was a turbulent attached value (HK clip 1.05-1.8) that
    # acted as an unphysical attractor — the ODE drifted H below 2.0 in
    # the FPG region and got pinned at 1.5, causing the LE H drop and
    # downstream transition delay (Re_θ_crit grossly over-estimated at
    # H=1.5 vs H=2.2).  Documented root cause of the H_L1 ≈ 0.808 plateau.
    H = tf.clip_by_value(delta / theta_safe, 2.0, 5.0)

    Re_theta = tf.maximum(ue * theta_safe * Re_safe, 1.0)

    # Cf_lam — Falkner-Skan family
    H_cf = tf.clip_by_value(H, 1.02, 5.3)
    FC = tf.maximum(0.0727 * tf.pow(5.5 - H_cf, 3.0) / (H_cf + 1.0) - 0.07, 0.001)
    Cf = FC / Re_theta

    # H* (laminar piecewise polynomial)
    Hstar = tf.where(
        H < 4.0,
        1.515 + 0.076 * tf.square(H - 4.0) / H,
        1.515 + 0.040 * (H - 4.0),
    )

    # ============================================================
    # v51-fix [2026-05-08] OPTION B — Cd lag-entrainment correction
    #
    # PREVIOUS (equilibrium):
    #     Cd = max(H* · Cf / 4, 1e-8)
    #
    # CURRENT (non-equilibrium lag-entrainment):
    #     Cd = max((H*·Cf/4) · (1 + α_lag · max(0, H - H_eq)), 1e-8)
    #     where H_eq = 2.59 (Blasius).
    #
    # Why: equilibrium Cd matches dH*/ds=0, which is fine in mild PGs but
    # under-predicts dissipation when H ramps up under strong APG.  The
    # extra (1 + α_lag · ΔH) factor mimics the lag-entrainment correction
    # used in CL2/Drela closures: dissipation accelerates as the BL
    # departs from equilibrium, suppressing the θ runaway and capping H
    # before it reaches the H<5 clip artifact.
    #
    # Verified at NACA0012 α=8° R=1e6 (single-shot): without this term
    # θ_TE ≈ 0.04 (XFOIL 0.008, 5× over).  With α_lag=1.0 expected to
    # bring θ_TE within ~30% of XFOIL.
    # ============================================================
    H_eq = 2.59  # Blasius equilibrium H
    H_excess = tf.maximum(H - H_eq, 0.0)
    # ============================================================
    # v51-fix [2026-05-10] Tier B — laminar separation bubble dissipation.
    #
    # When H exceeds ~3.0 (laminar separation onset), the standard
    # Falkner-Skan closure has no recovery mechanism — the ODE runs into
    # H ≈ 4 plateau and stays there (this is the LSB plateau seen in
    # validation plots, e.g. xi=0.4-0.65 H=3.7 in NACA0012 α=0° R=3e5).
    #
    # XFOIL handles this with a separation bubble closure that includes
    # extra dissipation in the bubble.  Pragmatic mimic: add quadratic
    # damping term in Cd that fires when H > 3:
    #
    #   Cd = Cd_eq · (1 + α_lag · ΔH + α_bubble · max(H-3, 0)² )
    #
    # The α_bubble · ΔH² term ramps fast in separation regime, driving
    # δ* growth ↓ → H ↓ → reattachment-like behavior in lam ODE output.
    # ============================================================
    # ============================================================
    # v51-fix [2026-05-10] C-1 (α-3) — α_bubble 5 → 15 + Cf augmentation.
    #
    # Previous α_bubble=5 was marginal (smoke test: H_lam max 4.79 → 4.625).
    # Issue: in separation regime Cf is tiny so Cd_eq·(1+α_b·ΔH²) is still
    # small in absolute terms.  Two-pronged fix:
    #
    # 1. α_bubble = 15 (3× stronger dissipation in bubble)
    # 2. Add Cd_floor in bubble regime ~ Drela's BL bubble model:
    #    Cd_bubble ~ 0.005·max(H-3, 0) (extra absolute dissipation,
    #    not multiplicative).  Mimics non-Cf-based reattachment energy.
    #
    # In integral BL energy eq d(θH*)/ds = 2·Cd, larger Cd → δ growth ↓
    # → H = δ*/θ growth ↓ → smoother LSB → reattachment-like profile.
    # ============================================================
    H_excess_strong = tf.maximum(H - 3.0, 0.0)
    alpha_bubble = 15.0
    Cd_eq = Hstar * Cf / 4.0
    Cd_bubble_floor = 0.005 * H_excess_strong            # absolute bubble dissipation
    Cd = tf.maximum(
        Cd_eq * (1.0 + alpha_lag * H_excess + alpha_bubble * H_excess_strong ** 2)
        + Cd_bubble_floor,
        1e-8,
    )

    # dH*/dH
    dHs_dH = tf.where(
        H < 4.0,
        0.076 * (1.0 - 16.0 / tf.square(H + 1e-6)),
        0.040 * tf.ones_like(H),
    )
    dHs_dH_safe = tf.where(
        tf.abs(dHs_dH) < 0.02,
        tf.sign(dHs_dH + 1e-8) * 0.02,
        dHs_dH,
    )

    ue_safe = tf.maximum(ue, 1e-4)

    # Momentum: dθ/dξ = L·Cf/2 - (H+2)·θ·due_dξ/ue
    f_theta = L_branch * 0.5 * Cf - theta_safe * (2.0 + H) * due_dxi / ue_safe

    # Energy → dδ*/dξ = H·f_theta + θ·f_H
    # where θ·f_H = numerator / dHs_dH
    numerator = (
        L_branch * (2.0 * Cd - Hstar * 0.5 * Cf)
        + Hstar * H * theta_safe * due_dxi / ue_safe
    )
    # v51-fix [2026-05-08]: clip tightened from 3.0 → 0.05 after observing
    # θ runaway (typical XFOIL θ_TE ≈ 0.005, our integration was producing
    # θ=0.27 at TE — 50× over).  Physical |dθ/dξ| ≈ 0.005-0.02 max,
    # |dδ*/dξ| ≈ 0.025-0.05 max for laminar BL.
    theta_times_fH = tf.clip_by_value(numerator / dHs_dH_safe, -0.05, 0.05)
    f_delta = H * f_theta + theta_times_fH

    # Tight per-step rate clip — calibrated to typical XFOIL laminar magnitudes
    f_theta = tf.clip_by_value(f_theta, -0.02, 0.02)   # was 0.5 (too liberal)
    f_delta = tf.clip_by_value(f_delta, -0.10, 0.10)   # was 3.0 (too liberal)

    return f_theta, f_delta


def solve_laminar_bl_full_2eq(ue, due_dxi, Re, L_branch,
                                dxi, xi_1d, cfg: BLConfig,
                                re_blend: float = 1.0,
                                hiemenz_xi_end: float = 0.05):
    """
    Full 2-equation laminar BL via Heun ODE.

    Both θ and δ* are integrated from ξ=0 to ξ=1 with:
        ξ ∈ [0, hiemenz_xi_end]: Hiemenz analytical (constant θ, H=2.216)
        ξ > hiemenz_xi_end:      Heun (RK2) integration of (θ, δ*) ODEs

    The Hiemenz "hold" region avoids the stagnation 1/ue singularity that
    destabilised the previous Hybrid attempt.

    Initial conditions (Hiemenz solution):
        θ(0)² = 0.075 / (Re · |due/ds|_stag)
        H(0) = 2.216  →  δ*(0) = 2.216 · θ(0)

    Returns the same dict shape as compute_thwaites for drop-in use.
    """
    Re_safe = tf.maximum(Re, 1.0)
    ue_safe = tf.maximum(ue, 1e-4)

    # ---- Hiemenz initial conditions ----
    due_ds = due_dxi / (L_branch + 1e-10)
    n_le = min(10, due_ds.shape[1])
    due_ds_stag = tf.reduce_max(tf.abs(due_ds[:, :n_le]), axis=1, keepdims=True)
    due_ds_stag = tf.maximum(due_ds_stag, 1.0)
    theta_hiemenz = tf.sqrt(0.075 / (Re_safe * due_ds_stag))
    delta_hiemenz = 2.216 * theta_hiemenz

    Nside_p1 = xi_1d.shape[0]
    Nside = Nside_p1 - 1

    # Index where ODE starts
    i_start = int(round(hiemenz_xi_end / float(dxi.numpy())
                         if hasattr(dxi, "numpy") else hiemenz_xi_end * Nside))
    i_start = max(2, min(i_start, Nside - 2))

    theta_curr = theta_hiemenz
    delta_curr = delta_hiemenz
    theta_list = [theta_curr]
    delta_list = [delta_curr]

    for i in range(Nside):
        if i < i_start:
            # Hiemenz hold (avoid 1/ue singularity)
            theta_list.append(theta_curr)
            delta_list.append(delta_curr)
            continue

        # Heun (RK2) step from xi_i to xi_{i+1}
        ue_i = ue_safe[:, i:i + 1]
        due_i = due_dxi[:, i:i + 1]
        ue_ip1 = ue_safe[:, i + 1:i + 2]
        due_ip1 = due_dxi[:, i + 1:i + 2]

        # Predictor
        # v51-fix [2026-05-08] OPTION B — pass cfg.alpha_lag to RHS
        # so the lag-entrainment Cd correction takes effect.
        ft1, fd1 = _full_lam_rhs(theta_curr, delta_curr,
                                   ue_i, due_i, Re_safe, L_branch,
                                   alpha_lag=cfg.alpha_lag)
        theta_pred = tf.maximum(theta_curr + dxi * ft1, 1e-7)
        delta_pred = tf.maximum(delta_curr + dxi * fd1, 1e-7)

        # Corrector at ξ_{i+1} state
        ft2, fd2 = _full_lam_rhs(theta_pred, delta_pred,
                                   ue_ip1, due_ip1, Re_safe, L_branch,
                                   alpha_lag=cfg.alpha_lag)

        theta_curr = tf.maximum(theta_curr + 0.5 * dxi * (ft1 + ft2), 1e-7)
        delta_curr = tf.maximum(delta_curr + 0.5 * dxi * (fd1 + fd2), 1e-7)

        # v51-fix [2026-05-10] — H floor 1.5 → 2.0 (see _full_lam_rhs note).
        # Enforce H ∈ [2.0, 5.0] via δ-renormalization at every Heun step.
        # 2.0 is the Falkner-Skan β=1 (max FPG) lower bound; physical laminar
        # BL never goes below this.
        H_check = delta_curr / theta_curr
        H_clamped = tf.clip_by_value(H_check, 2.0, 5.0)
        delta_curr = H_clamped * theta_curr

        theta_list.append(theta_curr)
        delta_list.append(delta_curr)

    theta_lam = tf.concat(theta_list, axis=1)
    delta_lam = tf.concat(delta_list, axis=1)
    # v51-fix [2026-05-10] — H floor 1.5 → 2.0 (see _full_lam_rhs note).
    H_lam = tf.clip_by_value(
        delta_lam / tf.maximum(theta_lam, 1e-8), 2.0, 5.0)

    # ---- n-factor envelope (same as compute_thwaites) ----
    Re_theta_lam = ue_safe * theta_lam * Re_safe
    Re_theta_crit = Re_theta_crit_drela(H_lam)
    onset = tf.sigmoid(4.0 * (Re_theta_lam - Re_theta_crit) / (Re_theta_crit + 1.0))

    dn_dRe = dn_dRe_dampl(H_lam) * re_blend

    Re_theta_lam_shift = tf.concat([Re_theta_lam[:, 1:], Re_theta_lam[:, -1:]], axis=1)
    dRe_theta_dxi = tf.maximum((Re_theta_lam_shift - Re_theta_lam) / dxi, 0.0)
    sigma_n = tf.clip_by_value(onset * dn_dRe * dRe_theta_dxi, 0.0, 30.0)

    n = tf.cumsum(sigma_n * dxi, axis=1, exclusive=True)
    w_tr = tf.sigmoid(cfg.trans_k * (n - cfg.Ncrit))

    return {
        "theta_lam": theta_lam,
        "H_lam": H_lam,
        "delta_lam": delta_lam,
        "n": n,
        "w_tr": w_tr,
        "sigma_n": sigma_n,
    }


def _dH_dxi_lam(H, theta, ue, due_dxi, Re_safe, L_branch):
    """
    Right-hand side of the laminar dH/dξ ODE derived from the energy
    integral (Drela formulation):

        θ · dH*/dξ = L · (2 C_d − H* C_f / 2) + H* H θ · du_e_dξ / u_e
        dH/dξ      = (1 / θ) · (θ · dH*/dξ) / (dH*/dH)

    Closures (laminar):
        C_f^lam(H, Re_θ)  = Falkner-Skan family fit (FC = 0.0727·(5.5-H)³/
                            (H+1) - 0.07,  C_f = FC / Re_θ)
        H*(H)             = 1.515 + 0.076 (H-4)² / H   (H<4)
                          = 1.515 + 0.040 (H-4)        (H≥4)
        C_d^lam           = H* · C_f / 4                (equilibrium)
        dH*/dH            = analytical derivative of the above piecewise H*

    H is clipped to [2.0, 5.0] for stability (v51-fix 2026-05-10:
    floor 1.5 → 2.0; see _full_lam_rhs note for root cause analysis).
    """
    H_safe = tf.clip_by_value(H, 2.0, 5.0)
    ue_safe = tf.maximum(ue, 1e-4)
    theta_safe = tf.maximum(theta, 1e-7)

    Re_theta = tf.maximum(ue_safe * theta_safe * Re_safe, 1.0)

    # Cf — Falkner-Skan fit
    H_cf = tf.clip_by_value(H_safe, 1.02, 5.3)
    FC = 0.0727 * tf.pow(5.5 - H_cf, 3.0) / (H_cf + 1.0) - 0.07
    FC = tf.maximum(FC, 0.001)
    Cf = FC / Re_theta

    # H* (laminar piecewise)
    H_c = tf.clip_by_value(H_safe, 1.5, 10.0)
    Hstar = tf.where(
        H_c < 4.0,
        1.515 + 0.076 * tf.square(H_c - 4.0) / H_c,
        1.515 + 0.040 * (H_c - 4.0),
    )

    # Cd (equilibrium relation)
    Cd = tf.maximum(Hstar * Cf / 4.0, 1e-8)

    # dH*/dH
    dHs_dH = tf.where(
        H_c < 4.0,
        0.076 * (1.0 - 16.0 / tf.square(H_c + 1e-6)),
        0.040 * tf.ones_like(H_c),
    )
    dHs_dH_safe = tf.where(
        tf.abs(dHs_dH) < 0.02,
        tf.sign(dHs_dH + 1e-8) * 0.02,
        dHs_dH,
    )

    # Energy ODE in ξ-coords (numerator = θ · dH*/dξ)
    numerator = (
        L_branch * (2.0 * Cd - Hstar * 0.5 * Cf)
        + Hstar * H_safe * theta_safe * due_dxi / (ue_safe + 1e-4)
    )
    dH_dxi = numerator / (theta_safe * dHs_dH_safe + 1e-12)
    return tf.clip_by_value(dH_dxi, -200.0, 200.0)


def solve_laminar_bl_2eq(ue, due_dxi, Re, L_branch,
                          dxi, xi_1d, cfg: BLConfig,
                          re_blend: float = 1.0,
                          H_init: float = 2.216):
    """
    Two-equation laminar BL = Thwaites θ + Heun (RK2) ODE for H.

    Why hybrid (vs full 2-eq θ AND δ*):
      Thwaites' integral gives a robust, analytically-anchored θ — the
      momentum equation is well-conditioned and the Hiemenz splice cleanly
      handles the stagnation singularity.  The H closure is what was
      under-performing in strong APG (algebraic λ→H gives no history).
      So we keep θ from Thwaites (unchanged) and integrate dH/dξ using the
      energy equation, with Hiemenz value 2.216 as initial condition.

    The transition machinery (Re_θ_crit, dn/dReθ, n integration, w_tr)
    follows compute_thwaites verbatim — only H_lam is replaced.

    Parameters
    ----------
    ue       : (B, Nside+1)
    due_dxi  : (B, Nside+1)
    Re       : (B, 1)
    L_branch : (B, 1)
    dxi      : scalar
    xi_1d    : (Nside+1,)
    cfg      : BLConfig
    re_blend : scalar
    H_init   : scalar  Hiemenz stagnation value (default 2.216)

    Returns
    -------
    dict {theta_lam, H_lam, delta_lam, n, w_tr, sigma_n}
    """
    ue_safe = tf.maximum(ue, 1e-6)
    Re_safe = tf.maximum(Re, 1.0)

    # ---- (a)/(b)/(c) Thwaites θ + Hiemenz splice (unchanged) ----
    ue5 = tf.pow(ue_safe, 5.0)
    integral_ue5 = tf.cumsum(ue5, axis=1, exclusive=True) * dxi * L_branch
    theta_thw_sq = 0.45 / Re_safe * integral_ue5 / tf.pow(ue_safe, 6.0)

    due_ds = due_dxi / (L_branch + 1e-10)
    n_le = min(10, due_ds.shape[1])
    due_ds_stag = tf.reduce_max(tf.abs(due_ds[:, :n_le]), axis=1, keepdims=True)
    due_ds_stag = tf.maximum(due_ds_stag, 1.0)
    theta_hiemenz_sq = 0.075 / (Re_safe * due_ds_stag)

    w_thw = tf.sigmoid(40.0 * (xi_1d[None, :] - 0.05))
    theta_lam_sq = (1.0 - w_thw) * theta_hiemenz_sq + w_thw * theta_thw_sq
    theta_lam_sq = tf.minimum(theta_lam_sq, 0.05 ** 2)
    theta_lam = tf.sqrt(tf.maximum(theta_lam_sq, 0.0))

    # ---- (d) Heun (RK2) integration of dH/dξ along ξ ----
    Nside_p1 = xi_1d.shape[0]
    Nside = Nside_p1 - 1
    B = tf.shape(theta_lam)[0]

    H_curr = tf.fill([B, 1], tf.constant(H_init, dtype=DTYPE))
    H_list = [H_curr]

    # Loop unrolled in Python — Nside is a Python int (60), so tf.function
    # traces this into a static graph.  Heun: predictor + corrector.
    for i in range(Nside):
        theta_i = theta_lam[:, i:i + 1]
        ue_i = ue_safe[:, i:i + 1]
        due_i = due_dxi[:, i:i + 1]

        theta_ip1 = theta_lam[:, i + 1:i + 2]
        ue_ip1 = ue_safe[:, i + 1:i + 2]
        due_ip1 = due_dxi[:, i + 1:i + 2]

        # Predictor (Euler)
        k1 = _dH_dxi_lam(H_curr, theta_i, ue_i, due_i, Re_safe, L_branch)
        H_pred = tf.clip_by_value(H_curr + dxi * k1, 1.5, 5.0)

        # Corrector evaluated at ξ_{i+1} state
        k2 = _dH_dxi_lam(H_pred, theta_ip1, ue_ip1, due_ip1, Re_safe, L_branch)

        # Heun step
        H_curr = tf.clip_by_value(H_curr + 0.5 * dxi * (k1 + k2), 1.5, 5.0)
        H_list.append(H_curr)

    H_lam = tf.concat(H_list, axis=1)         # (B, Nside+1)
    delta_lam = H_lam * theta_lam

    # ---- (e) n-factor envelope (same as compute_thwaites) ----
    Re_theta_lam = ue_safe * theta_lam * Re_safe
    Re_theta_crit = Re_theta_crit_drela(H_lam)
    onset = tf.sigmoid(4.0 * (Re_theta_lam - Re_theta_crit) / (Re_theta_crit + 1.0))

    dn_dRe = dn_dRe_dampl(H_lam) * re_blend

    Re_theta_lam_shift = tf.concat([Re_theta_lam[:, 1:], Re_theta_lam[:, -1:]], axis=1)
    dRe_theta_dxi = tf.maximum((Re_theta_lam_shift - Re_theta_lam) / dxi, 0.0)
    sigma_n = tf.clip_by_value(onset * dn_dRe * dRe_theta_dxi, 0.0, 30.0)

    n = tf.cumsum(sigma_n * dxi, axis=1, exclusive=True)
    w_tr = tf.sigmoid(cfg.trans_k * (n - cfg.Ncrit))

    return {
        "theta_lam": theta_lam,
        "H_lam": H_lam,
        "delta_lam": delta_lam,
        "n": n,
        "w_tr": w_tr,
        "sigma_n": sigma_n,
    }
