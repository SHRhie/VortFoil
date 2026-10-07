# -*- coding: utf-8 -*-
"""
BL closure (Cf, f_θ, f_δ) and residual computations.

Extracted verbatim from steady51.py.  Standalone functions — `re_blend`,
`Ncrit`, `trans_k` and the ξ-grid are passed explicitly via `BLPhysicsConfig`.

The energy-equation derivation cancels the 1/θ singularity:
    dδ*/dξ = H · f_θ + θ · f_H
where  θ · f_H = numerator / (dH*/dH)  with the θ multiplier already absorbed
into the numerator (see steady51.py:2030–2090 for full derivation).
"""
from __future__ import annotations
from dataclasses import dataclass

import tensorflow as tf

from .interp import central_diff_1d
from .bl_closures import (
    Hstar_from_H_lam, Hstar_from_H_turb,    # noqa: F401  (kept for legacy imports)
    Cd_lam, Cd_turb,                         # noqa: F401
    dn_dRe_dampl,
    Re_theta_crit_drela,
)
# v51-fix [2026-05-08] — XFOIL 1:1 turbulent closures replace the
# simplified Drela forms in bl_closures.py.  Plateau diagnosis showed
# that the missing C_τ-dependent outer-layer term in our old Cd_turb
# (=0.5·Cf·H* + 0.0056/(H_c-0.6)) was the main cause; XFOIL's
# Cd_turb has an extra S²·(0.995-Us) outer term plus a viscous
# contribution that we were missing entirely.
from .xfoil_closures import (
    cf_turb_xfoil,
    hstar_turb_xfoil,
    cd_turb_xfoil,
    s_eq_xfoil,
    lag_rhs_xfoil,
    delta_thickness_xfoil,
    hkin,
)

DTYPE = tf.float32


@dataclass
class BLPhysicsConfig:
    """Runtime physics parameters for BL closure."""
    Ncrit: float = 7.0
    trans_k: float = 2.0
    re_blend: float = 1.0      # 0 = Re_delta*, 1 = Re_theta


# ============================================================
# bl_closure — main physics: f_theta, f_theta_Hstar, f_S, Cf, Hstar
#
# v51-fix [2026-05-10] D-revert — n-factor reinstated.
#
# Earlier attempt (D): replace n-factor with operator-learned ξ_tr head.
# Result: degenerate solutions in BOTH directions:
#   • shadow off  →  ξ_tr → 1  (fully laminar trivializes lam-ODE residual)
#   • shadow on   →  ξ_tr → 0  (fully turb trivializes shadow residual)
#   • boundary-anchored w_tr → ξ_tr → 0 still finds sharp-LE transition path
# The fundamental issue: any operator-controlled w_tr provides a loophole.
#
# Resolution: n-factor envelope (Drela CL2) determines transition from
# the BL state itself.  This is PHYSICS (well-established model), not a
# hardcoded constraint — the operator still learns θ, H, S, and the
# transition position EMERGES from those predictions via Re_θ growth.
# ============================================================
def bl_closure(ue, due, theta, delta, H, Re, n, L_branch,
                xi_1d, cfg: BLPhysicsConfig, S=None, J_xi=None):
    """
    BL closure in ξ-coordinates.

    Momentum (von Kármán):
        f_θ      = L·C_f/2  −  (H+2)·θ·u_e'/u_e
    Energy in (θH*) form:
        f_θH*   = 2·L·C_D  −  2·H*·θ·u_e'/u_e
    Lag (XFOIL CL2):
        f_S     = lag_rhs_xfoil(...)               (S = √C_τ)

    Parameters
    ----------
    n : (B, Nside+1)  cumulative n-factor (instability amplification),
        from solve_laminar_bl_full_2eq.  w_tr derived inside this
        function as sigmoid(trans_k · (n - Ncrit)).

    Returns
    -------
    f_theta, f_theta_Hstar, f_delta_drela, f_S, sigma_n, w_tr, Cf, Hstar
        all (B, Nside+1) tensors.
    """
    ue_safe = tf.maximum(ue, 1e-4)
    theta_safe = tf.maximum(theta, 1e-6)
    delta_safe = tf.maximum(delta, 1e-6)
    Re_safe = tf.maximum(Re, 1.0)

    # Re_delta ↔ Re_theta blend
    Re_theta_raw = ue_safe * theta_safe * Re_safe
    Re_delta_raw = ue_safe * delta_safe * Re_safe
    Re_theta = tf.maximum(
        (1.0 - cfg.re_blend) * Re_delta_raw + cfg.re_blend * Re_theta_raw, 1.0)

    # Transition weight from n-factor (physics-derived)
    w_tr = tf.sigmoid(cfg.trans_k * (n - cfg.Ncrit))

    # ============================================================
    # v51-fix [2026-05-08] — turb branch swapped to XFOIL 1:1 closures
    # (laminar branch unchanged — Falkner-Skan form happens to match
    #  XFOIL CFL exactly for HK<5.5, which is the only relevant range).
    #
    # OLD turbulent  : Ludwieg-Tillmann 1949
    #     Cf = 0.246·10^(-1.561·H)·Re_θ^(-0.268)
    #     H* = 1.505 + 4/(H_c+0.25) - 4/1.85 + 0.007·max(H_c-1.6, 0)²
    #     Cd = 0.5·Cf·H* + 0.0056/(H_c-0.6)
    # NEW turbulent  : XFOIL CFT/HST/BLVAR (Drela)
    #     Cf = (Coles + Drela mods)
    #     H* = (attached/separated branches, Re_θ-dependent)
    #     Cd = 0.5·Cf·Us·DFAC  +  S²·(0.995-Us)  +  0.15·(0.995-Us)²/RT
    # The Cd is the critical change — it now contains a non-zero
    # outer-layer term S²·(0.995-Us).  S = √Cτ uses the equilibrium
    # value S_eq(H, Re_θ) (algebraic) here; the lag ODE for S can be
    # added in a second iteration if needed.
    # ============================================================
    Re_theta_safe = tf.maximum(Re_theta, 1.0)

    # Laminar Cf — Falkner-Skan (matches XFOIL CFL for HK<5.5)
    H_cf = tf.clip_by_value(H, 1.02, 5.3)
    FC = 0.0727 * tf.pow(5.5 - H_cf, 3.0) / (H_cf + 1.0) - 0.07
    FC = tf.maximum(FC, 0.001)
    Cf_lam_v = FC / Re_theta_safe

    # Turbulent Cf — XFOIL CFT
    HK = hkin(H)        # incompressible (Msq=0): HK = H
    Cf_turb_v = tf.maximum(cf_turb_xfoil(HK, Re_theta_safe), 1e-6)

    Cf = (1.0 - w_tr) * Cf_lam_v + w_tr * Cf_turb_v

    # LE Cf mask — suppress spike at ξ ≈ 0 from clamped Re_θ.
    # Disabled via OP2_NO_LE_MASK=1 for pure-equation experiments.
    import os as _os
    if _os.environ.get("OP2_NO_LE_MASK", "0") != "1":
        xi_le_mask_cf = tf.sigmoid(40.0 * (xi_1d[None, :] - 0.05))
        Cf = Cf * xi_le_mask_cf

    # ----- Jacobian J(ξ) = ds_arc/dξ for arc-length BL form -----
    # If caller supplies J_xi (per-ξ, shape (B, N)), use it (variable
    # Jacobian — proper parametric mode).  Else fall back to constant
    # L_branch broadcast (legacy/arc-uniform mode).
    if J_xi is None:
        J_xi_eff = L_branch                # (B, 1) broadcast
    else:
        J_xi_eff = J_xi                    # (B, N)

    # ----- Momentum: dθ/dξ -----
    H_for_theta = tf.stop_gradient(H)
    f_theta_lam = (J_xi_eff * 0.5 * Cf_lam_v
                   - theta_safe * (2.0 + H_for_theta) * due / (ue_safe + 1e-4))
    f_theta_turb = (J_xi_eff * 0.5 * Cf_turb_v
                    - theta_safe * (2.0 + H_for_theta) * due / (ue_safe + 1e-4))
    f_theta = (1.0 - w_tr) * f_theta_lam + w_tr * f_theta_turb

    # ----- Energy: dδ*/dξ = H·f_theta + θ·f_H -----
    # Laminar H*, Cd — kept (Drela lam form is close enough to XFOIL HSL/DIL
    #                  for HK<4.35; the dominant plateau is in the turb branch)
    Hstar_lam_v = Hstar_from_H_lam(H)
    Cd_lam_v = Cd_lam(H, Re_theta, Cf_lam_v)

    # Turbulent H*, Cd — XFOIL HST + BLVAR (3-component Cd with C_τ)
    Hstar_turb_v = hstar_turb_xfoil(HK, Re_theta_safe)
    S_eq_v = s_eq_xfoil(Hstar_turb_v, HK, H, Re_theta_safe)
    # v51-fix [2026-05-08] OPTION B — use trainable S if provided
    # (3-eq turb), else equilibrium S_eq (algebraic, A path).
    S_for_cd = S if S is not None else S_eq_v
    Cd_turb_v = cd_turb_xfoil(Hstar_turb_v, HK, H, Cf_turb_v, S_for_cd,
                              Re_theta_safe)

    Hstar = (1.0 - w_tr) * Hstar_lam_v + w_tr * Hstar_turb_v
    Cd = (1.0 - w_tr) * Cd_lam_v + w_tr * Cd_turb_v

    # ============================================================
    # Review (2) — Energy integral in (θH*) form (primary, no stiffness)
    #
    #   d(θH*)/dξ = 2·L·C_D − 2·H*·θ · (1/u_e)·(du_e/dξ)
    #
    # No dH*/dH division.  This is the primary energy residual.
    # ============================================================
    theta_for_E = tf.stop_gradient(theta_safe)
    f_theta_Hstar = (
        2.0 * J_xi_eff * Cd
        - 2.0 * Hstar * theta_for_E * due / (ue_safe + 1e-4)
    )

    # ============================================================
    # δ* mass-defect residual (Drela form, with strict clipping)
    #
    # Why both R_θH* and R_δ:
    #   The (θH*)-form alone constrains the product θ·H* but leaves
    #   δ* (a separate ADAF head) free.  Empirically this caused 7x
    #   over-prediction of δ*  (avg H_L1 jumped from 0.80 → 1.92).
    #
    # Adding the original Drela δ*-form as a *secondary* residual
    # constrains δ* magnitude directly:
    #
    #   f_δ = H · f_θ + θ · f_H,    where  θ·f_H = numerator / (dH*/dH)
    #
    # The dH*/dH division is the stiffness that review (2) wanted to
    # avoid.  We mitigate by:
    #   (a) tighter H clip [1.1, 4.0]  — kept inside attached regime
    #   (b) 5× stricter dHs_dH floor (0.02 → 0.10)
    #   (c) tight clip on θ·f_H ([−0.5, +0.5], was [−5, +5])
    #
    # Net effect: stiff at H≈4 still suppressed, but the energy form
    # (R_θH*) does the heavy lifting in those edge cases.  In the bulk
    # of the BL the two residuals agree (mathematically equivalent),
    # and R_δ pins δ*'s absolute scale.
    # ============================================================
    H_c = tf.clip_by_value(H, 1.1, 4.0)
    dHs_dH_lam = tf.where(
        H_c < 4.0,
        0.076 * (1.0 - 16.0 / tf.square(H_c + 1e-6)),
        0.040 * tf.ones_like(H_c),
    )
    eps_dh = 0.01
    Hs_p = hstar_turb_xfoil(HK + eps_dh, Re_theta_safe)
    Hs_m = hstar_turb_xfoil(HK - eps_dh, Re_theta_safe)
    dHs_dH_turb = (Hs_p - Hs_m) / (2.0 * eps_dh)
    dHs_dH = (1.0 - w_tr) * dHs_dH_lam + w_tr * dHs_dH_turb
    dHs_dH_safe = tf.where(
        tf.abs(dHs_dH) < 0.10,             # 5x stricter than 0.02
        tf.sign(dHs_dH + 1e-8) * 0.10,
        dHs_dH,
    )
    numerator_drela = (
        J_xi_eff * (2.0 * Cd - Hstar * 0.5 * Cf)
        + Hstar * H * theta_for_E * due / (ue_safe + 1e-4)
    )
    theta_times_fH = numerator_drela / dHs_dH_safe
    theta_times_fH = tf.clip_by_value(theta_times_fH, -0.5, 0.5)   # tight (was -5, 5)
    H_sg = tf.stop_gradient(H)
    f_theta_sg = tf.stop_gradient(f_theta)
    f_delta_drela = H_sg * f_theta_sg + theta_times_fH

    # ============================================================
    # v51-fix [2026-05-08] OPTION B — Lag ODE for S = √Cτ (turb only)
    #
    # f_S = lag_rhs_xfoil(S, S_eq, δ, ue, due, Cf, HK, H, RT)
    # Masked by w_tr so that lam region has zero residual contribution
    # (S in lam is unused).
    # ============================================================
    if S is not None:
        delta_full = delta_thickness_xfoil(theta_safe, delta_safe, HK)
        S_for_lag = tf.maximum(S, 1e-6)
        f_S_turb = lag_rhs_xfoil(S_for_lag, S_eq_v, delta_full,
                                  ue_safe, due, Cf_turb_v,
                                  HK, H, Re_theta_safe, ALD=1.0)
        # Stability clip — calibrated to typical XFOIL dS/dξ magnitudes
        f_S_turb = tf.clip_by_value(f_S_turb, -0.5, 0.5)
        f_S = w_tr * f_S_turb     # lam → 0
    else:
        f_S = tf.zeros_like(f_theta)

    # ----- Transition amplification (Drela CL2 envelope) -----
    # v51-fix [2026-05-10] C-1 (α-4) — σ_n clip 100 → 50.
    # Earlier 30→100 was based on flawed reasoning (transition delayed).
    # XFOIL plot shows transition fires LATER than our model (xi≈0.85
    # vs ours xi≈0.65) — meaning OUR transition is too fast, not too
    # slow.  Lowered σ_n clip restores rate limit; combined with α-3's
    # bubble dissipation (slower H growth → slower Re_θ growth → onset
    # delayed), the transition position shifts toward XFOIL.
    Re_theta_crit = Re_theta_crit_drela(H)
    onset = tf.sigmoid(4.0 * (Re_theta - Re_theta_crit) / (Re_theta_crit + 1.0))
    dn_dRe = dn_dRe_dampl(H) * cfg.re_blend
    dRe_ds = Re_safe * ue_safe
    sigma_n_per_s = onset * dn_dRe * dRe_ds
    sigma_n = J_xi_eff * sigma_n_per_s
    sigma_n = tf.clip_by_value(sigma_n, 0.0, 50.0)

    return f_theta, f_theta_Hstar, f_delta_drela, f_S, sigma_n, w_tr, Cf, Hstar


# ============================================================
# Residuals (with masks applied)
# ============================================================
def compute_bl_residuals(ue, due, theta, theta_x, delta, delta_x,
                          H, n, Re, L_branch, xi_1d, cfg: BLPhysicsConfig,
                          S=None, S_x=None, J_xi=None):
    """
    Physics-Informed BL residuals (n-factor transition):

        R_θ      = θ_x      − f_θ                     (momentum)
        R_θH*    = (θH*)_x  − f_θH*                  (energy)
        R_S      = S_x      − f_S                     (lag)

    v51-fix [2026-05-10] D-revert — n-factor restored.  n input is the
    cumulative instability amplification from solve_laminar_bl_full_2eq;
    bl_closure derives w_tr from it.  Operator learns BL state (θ, H, S);
    transition position emerges from the state via n-factor.

    Masking policy:
      * R_θ, R_θH* : multiplied by w_tr·ue·xi_mask  (turb-restricted;
                     the lam BL state is enforced by the lam ODE solver
                     so the lam region has no learnable signal here).
                     Mask is now safe: w_tr is physics-derived, not
                     operator-controlled, so cannot be trivialized.
      * R_S       : multiplied by ue·xi_mask only, NOT by w_tr.
                     S develops smoothly through the transition region.
    """
    f_theta, f_theta_Hstar, f_delta, f_S, sigma_n, w_tr, Cf, Hstar = bl_closure(
        ue, due, theta, delta, H, Re, n, L_branch, xi_1d, cfg, S=S, J_xi=J_xi)

    # ----- R_θ -----
    R_theta_raw = theta_x - f_theta

    # ----- R_θH* -----  (review 2 — primary energy)
    theta_Hstar = theta * Hstar
    dxi_scalar = tf.cast(xi_1d[1] - xi_1d[0], theta_Hstar.dtype)
    theta_Hstar_x = central_diff_1d(theta_Hstar, dxi_scalar)
    R_theta_Hstar_raw = theta_Hstar_x - f_theta_Hstar

    # ----- R_δ -----  (Drela form, secondary; pins δ* magnitude)
    # R_δ_consistency (chain rule) was algebraically trivial:
    # δ* = θ·H by definition, so its product-rule derivative is
    # automatically satisfied — no learning signal.  Switching back
    # to the Drela direct-residual form (with strict dH*/dH clipping)
    # provides actual constraint on δ*'s magnitude.
    R_delta_raw = delta_x - f_delta

    # Masks — w_tr is physics-derived (n-factor), so masking R_θ / R_θH*
    # by w_tr is safe and gives clean turb-restricted signal.
    xi_grid = xi_1d[None, :]
    xi_mask = tf.clip_by_value((xi_grid - 0.02) / 0.06, 0.0, 1.0)

    R_theta = R_theta_raw * w_tr * ue * xi_mask
    R_theta_Hstar = R_theta_Hstar_raw * w_tr * ue * xi_mask
    R_delta = R_delta_raw * w_tr * ue * xi_mask

    # ----- R_S -----  (transition-focused weighting, Tier A-3)
    # v51-fix [2026-05-10] Tier A-3 — R_S amplified near transition zone.
    # User guidance: "S Lag should respond immediately at N_crit".
    # w_tr·(1-w_tr) peaks at w_tr=0.5 (= transition midpoint), giving
    # an automatic locator of the transition zone.  Weight = 1 + α·focus
    # amplifies R_S there without adding a hardcoded position.
    if S is not None and S_x is not None:
        R_S_raw = S_x - f_S
        trans_focus = w_tr * (1.0 - w_tr)            # max 0.25 at w_tr=0.5
        S_weight = 1.0 + 4.0 * trans_focus           # max ~2.0 at transition
        R_S = R_S_raw * ue * xi_mask * S_weight
    else:
        R_S_raw = tf.zeros_like(R_theta_raw)
        R_S = tf.zeros_like(R_theta_raw)

    return {
        "R_theta": R_theta,
        "R_theta_Hstar": R_theta_Hstar,
        "R_delta": R_delta,
        "R_S": R_S,
        "R_theta_raw": R_theta_raw,
        "R_theta_Hstar_raw": R_theta_Hstar_raw,
        "R_delta_raw": R_delta_raw,
        "R_S_raw": R_S_raw,
        "w_tr": w_tr, "Cf": Cf, "Hstar": Hstar,
    }


def compute_shadow_turb_residuals(ue, due,
                                    theta_turb, theta_turb_x,
                                    delta_turb, delta_turb_x,
                                    Re, L_branch, xi_1d,
                                    cfg: BLPhysicsConfig, J_xi=None):
    """
    Shadow turbulent BL residuals — anchors θ_turb / H_turb to satisfy
    turbulent BL physics in the attached-turb regime.

    v51-fix [2026-05-10] D-revert+ — restored with H clip [1.1, 2.0].
    Earlier removed (claim: clip biased H_turb), then noted as actually
    helpful (without anchor, H_turb pins at decoder upper bound, Cf→0,
    H_L1 = 0.81 → 1.11 regression).  Restored with slightly looser clip
    than 14th iter's [1.1, 1.8]:

      Attached turb : H ≈ 1.4–1.8  (~85% of post-transition chord)
      Mild separation : H up to ~2.0  (mid-APG, recovery regions)
      [Strong separation H>2.5: NOT anchored by shadow — operator's
       prediction floats freely there, but blend with lam ODE which
       already covers separation handles those regions]

    Trade-off accepted: clip biases H_turb toward attached regime,
    which is mostly correct for our Re∈[3e5, 1e6] airfoils where
    attached turb dominates post-transition.
    """
    ue_safe = tf.maximum(ue, 1e-4)
    theta_safe = tf.maximum(theta_turb, 1e-6)
    delta_safe = tf.maximum(delta_turb, 1e-6)
    Re_safe = tf.maximum(Re, 1.0)

    H_turb = tf.clip_by_value(delta_safe / theta_safe, 1.0, 5.0)

    Re_theta_raw = ue_safe * theta_safe * Re_safe
    Re_delta_raw = ue_safe * delta_safe * Re_safe
    Re_theta = tf.maximum(
        (1.0 - cfg.re_blend) * Re_delta_raw + cfg.re_blend * Re_theta_raw, 1.0)

    # Attached-turb-focused shadow: H clipped to [1.1, 2.0] for closure eval.
    H_shd = tf.clip_by_value(H_turb, 1.1, 2.0)
    HK_shd = hkin(H_shd)
    Cf_turb_v = tf.maximum(cf_turb_xfoil(HK_shd, Re_theta), 1e-6)

    xi_le_mask = tf.sigmoid(40.0 * (xi_1d[None, :] - 0.05))
    Cf_turb_v = Cf_turb_v * xi_le_mask

    J_xi_eff = L_branch if J_xi is None else J_xi

    H_sg = tf.stop_gradient(H_shd)
    f_theta = (J_xi_eff * 0.5 * Cf_turb_v
               - theta_safe * (2.0 + H_sg) * due / (ue_safe + 1e-4))

    Hstar = hstar_turb_xfoil(HK_shd, Re_theta)
    S_eq_v = s_eq_xfoil(Hstar, HK_shd, H_shd, Re_theta)
    Cd = cd_turb_xfoil(Hstar, HK_shd, H_shd, Cf_turb_v, S_eq_v, Re_theta)

    # ----- Energy in (θH*) form (review 2) -----
    theta_sg = tf.stop_gradient(theta_safe)
    f_theta_Hstar = (
        2.0 * J_xi_eff * Cd
        - 2.0 * Hstar * theta_sg * due / (ue_safe + 1e-4)
    )

    # (θH*)_x via grid central diff on shadow turb θ·H*
    theta_Hstar_shd = theta_turb * Hstar
    dxi_scalar = tf.cast(xi_1d[1] - xi_1d[0], theta_Hstar_shd.dtype)
    theta_Hstar_x = central_diff_1d(theta_Hstar_shd, dxi_scalar)

    R_theta = theta_turb_x - f_theta
    R_theta_Hstar = theta_Hstar_x - f_theta_Hstar

    xi_mask = tf.clip_by_value((xi_1d[None, :] - 0.02) / 0.06, 0.0, 1.0)
    R_theta = R_theta * ue_safe * xi_mask
    R_theta_Hstar = R_theta_Hstar * ue_safe * xi_mask

    return {
        "R_theta_shadow": R_theta,
        "R_theta_Hstar_shadow": R_theta_Hstar,
    }
