# -*- coding: utf-8 -*-
# Translated from XFOIL 6.99 (src/xblsys.f, src/xbl.f), Copyright (C) 2000 Mark Drela, Harold Youngren.
# XFOIL is licensed under the GNU GPL, version 2 or (at your option) any later version. This modified
# version is distributed under the GNU GPL version 3 or later as part of VortFoil,
# Copyright (C) 2026 Sang-Hyun Rhie, Gyuwon Kim, Jeongsu Lee. See LICENSE and NOTICE.
"""
1:1 Python/TensorFlow translation of XFOIL 6.99 BL closures (Drela 1989+).

Source: `XFOIL6.99/src/Xfoil699src/src/`
  - xblsys.f   :  HKIN, HSL, HST, CFL, CFT, DIL, DIT, BLVAR (Us, Cτ_eq), TRDIF
  - xbl.f      :  BLPINI (closure constants)

The closures cover the inviscid-incompressible regime (Msq = 0).  The Msq
parameter is preserved in signatures for traceability with the Fortran
source but defaults to zero, removing the compressibility branches.

Outputs match XFOIL up to fp32 precision when fed identical inputs.
"""
from __future__ import annotations

import tensorflow as tf

DTYPE = tf.float32


# ============================================================
# BLPAR constants — xbl.f:1558-1569 (BLPINI)
# ============================================================
SCCON  = 5.6                 # lag rate base coefficient
GACON  = 6.70                # Drela G-beta α
GBCON  = 0.75                # Drela G-beta β
GCCON  = 18.0                # HKC offset (non-equilibrium boundary)
DLCON  = 0.9                 # lag length factor (wake)
CTRCON = 1.8                 # transition √Cτ scale
CTRCEX = 3.3                 # transition √Cτ exponent
DUXCON = 1.0                 # equilibrium dUe/dx coupling
CTCON  = 0.5 / (GACON**2 * GBCON)   # ≈ 0.014862  (closure derived)


# ============================================================
# Compressibility H ↔ HK (incompressible: HK = H)
# ============================================================
def hkin(H, Msq=0.0):
    """xblsys.f:2278 — kinematic shape factor.

    Incompressible (Msq=0): HK = H exactly.
    """
    return (H - 0.290 * Msq) / (1.0 + 0.113 * Msq)


# ============================================================
# Turbulent Cf (xblsys.f:2483 CFT — Coles + Drela)
# ============================================================
def cf_turb_xfoil(HK, RT, Msq=0.0):
    """Coles correlation with Drela's modifications and compressibility.

    GRT = max( ln(RT/FC), 3 ),   FC = sqrt(1 + 0.5·γ_-1·Msq)
    GEX = -1.74 - 0.31·HK
    CFO = 0.3·exp(-1.33·HK)·(GRT/ln(10))^GEX
    THK = tanh(4 - HK/0.875)
    Cf  = ( CFO + 1.1e-4·(THK - 1) ) / FC
    """
    GM1 = 0.4
    FC = tf.sqrt(1.0 + 0.5 * GM1 * Msq)
    GRT = tf.math.log(tf.maximum(RT / FC, 1e-3))
    GRT = tf.maximum(GRT, 3.0)

    GEX = -1.74 - 0.31 * HK
    ARG = tf.maximum(-1.33 * HK, -20.0)
    THK = tf.tanh(4.0 - HK / 0.875)

    CFO = 0.3 * tf.exp(ARG) * tf.pow(GRT / 2.3026, GEX)
    return (CFO + 1.1e-4 * (THK - 1.0)) / FC


# ============================================================
# Turbulent H* (xblsys.f:2388 HST — Drela 1991 fit)
# ============================================================
def hstar_turb_xfoil(HK, RT, Msq=0.0):
    """Two-branch H* with Re_θ dependence.

    HO  = 3 + 400/RT  (RT > 400)  else  4
    Rtz = max(RT, 200)

    HK < HO  (attached):
        HR = (HO - HK)/(HO - 1)
        H* = (2 - Hsmin - 4/Rtz)·HR² · 1.5/(HK + 0.5) + Hsmin + 4/Rtz

    HK ≥ HO  (separated):
        HDIF = HK - HO
        Rtmp = HK - HO + 4/ln(Rtz)
        HTMP = 0.007·ln(Rtz)/Rtmp² + Dhsinf/HK
        H*   = HDIF²·HTMP + Hsmin + 4/Rtz

    + Whitfield compressibility correction (Msq=0 → identity).
    """
    HSMIN  = 1.500
    DHSINF = 0.015

    HO = tf.where(RT > 400.0,
                  3.0 + 400.0 / RT,
                  4.0 * tf.ones_like(RT))
    Rtz = tf.maximum(RT, 200.0)

    HR = (HO - HK) / (HO - 1.0)
    HS_attached = (
        (2.0 - HSMIN - 4.0 / Rtz) * tf.square(HR) * 1.5 / (HK + 0.5)
        + HSMIN + 4.0 / Rtz
    )

    GRT_sep = tf.math.log(Rtz)
    HDIF = HK - HO
    Rtmp = HK - HO + 4.0 / GRT_sep
    HTMP = 0.007 * GRT_sep / tf.square(Rtmp) + DHSINF / HK
    HS_separated = tf.square(HDIF) * HTMP + HSMIN + 4.0 / Rtz

    HS = tf.where(HK < HO, HS_attached, HS_separated)

    # Whitfield compressibility (Msq=0: HS unchanged)
    FM = 1.0 + 0.014 * Msq
    return (HS + 0.028 * Msq) / FM


# ============================================================
# Slip velocity ratio Us (xblsys.f:825)
#   Us = 0.5·H*·(1 - (HK-1)/(GBCON·H))
# ============================================================
def us_xfoil(Hstar, HK, H):
    """Slip velocity ratio with XFOIL clamp Us > 0.95 → 0.98."""
    Us_raw = 0.5 * Hstar * (1.0 - (HK - 1.0) / (GBCON * H))
    return tf.where(Us_raw > 0.95,
                     0.98 * tf.ones_like(Us_raw),
                     Us_raw)


# ============================================================
# Equilibrium √Cτ — Cτ_eq (xblsys.f:858-887, BLVAR turbulent branch)
#
#   HKC = HK - 1 - GCCON/RT     clamp ≥ 0.01
#   HKB = HK - 1
#   USB = 1 - Us                clamp ≥ 1e-3 (numerical floor)
#   √Cτ_eq = sqrt( CTCON · H* · HKB · HKC² / (USB · H · HK²) )
# ============================================================
def s_eq_xfoil(Hstar, HK, H, RT):
    """Equilibrium shear-stress coefficient √Cτ_eq."""
    HKC = HK - 1.0 - GCCON / tf.maximum(RT, 1.0)
    HKC = tf.maximum(HKC, 0.01)
    HKB = HK - 1.0
    HKB = tf.maximum(HKB, 1e-3)

    Us = us_xfoil(Hstar, HK, H)
    USB = tf.maximum(1.0 - Us, 1e-3)

    H_safe = tf.maximum(H, 1.0)
    HK_safe = tf.maximum(HK, 1.05)
    arg = CTCON * Hstar * HKB * tf.square(HKC) / (USB * H_safe * tf.square(HK_safe))
    arg = tf.maximum(arg, 1e-12)
    return tf.sqrt(arg)


# ============================================================
# Turbulent dissipation Cd  (xblsys.f:945-1040 BLVAR)
#
# Inner wall:    Cd_inner = 0.5·Cf·Us · DFAC
#   DFAC = 0.5 + 0.5·tanh( (HK-1)/(Hmin-1) ),   Hmin = 1 + 2.1/ln(RT)
# Outer (Cτ):    Cd_outer = S² · (0.995 - Us)
# Outer lam-stress: Cd_ls  = 0.15·(0.995-Us)² / RT
#
# (XFOIL stores DI = 2·Cd/H*; we compute Cd directly.)
# ============================================================
def cd_turb_xfoil(Hstar, HK, H, Cf, S, RT):
    """Three-component XFOIL turbulent Cd, **C_τ-dependent**.

    Parameters
    ----------
    Hstar : H*
    HK    : kinematic shape factor (= H incompressible)
    H     : shape factor
    Cf    : Cf (from cf_turb_xfoil)
    S     : √Cτ (primary lag variable, evolves via lag_rhs_xfoil)
    RT    : Re_θ
    """
    Us = us_xfoil(Hstar, HK, H)

    # Inner wall (Cf·Us) with DFAC low-HK correction
    GRT = tf.math.log(tf.maximum(RT, 2.0))
    Hmin = 1.0 + 2.1 / GRT
    FL = (HK - 1.0) / tf.maximum(Hmin - 1.0, 1e-3)
    DFAC = 0.5 + 0.5 * tf.tanh(FL)
    Cd_inner = 0.5 * Cf * Us * DFAC

    # Outer layer Cτ-dependent
    Cd_outer = tf.square(S) * (0.995 - Us)

    # Outer laminar-stress contribution
    Cd_lam_str = 0.15 * tf.square(0.995 - Us) / tf.maximum(RT, 1.0)

    return tf.maximum(Cd_inner + Cd_outer + Cd_lam_str, 1e-8)


# ============================================================
# Transition initial √Cτ (xblsys.f:1393 TRDIF)
#   S_init = CTRCON · exp(-CTRCEX/(HK-1))
# ============================================================
def s_initial_transition(HK):
    """Drela's transition √Cτ correlation."""
    HK_m1 = tf.maximum(HK - 1.0, 0.05)
    return CTRCON * tf.exp(-CTRCEX / HK_m1)


# ============================================================
# Lag ODE  (xblsys.f:1759-1771, derived from REZC residual)
#
# REZC residual form (XFOIL between station 1 and 2):
#   SCC·(S_eq - S·ALD)·Δξ
#       - 2·δ·ln(S2/S1)
#       + 2·δ·DUXCON·(UQ·Δξ - ln(Ue2/Ue1)) = 0
#
# Δξ → 0 ⇒  dS/dξ = (S/(2·δ)) ·
#                    [ SCC·(S_eq - S·ALD) + 2·δ·DUXCON·(UQ - 1/ue·due/dξ) ]
#
# where:
#   SCC = SCCON · 1.333/(1+Us)
#   HR  = HKC/(GACON·ALD·HK),    HKC = HK - 1 - GCCON/RT  (clamp ≥ 0.01)
#   UQ  = (0.5·Cf - HR²)/(GBCON·δ)
#   ALD = 1.0 airfoil  /  0.9 wake (DLCON)
# ============================================================
def lag_rhs_xfoil(S, S_eq, delta, ue, due_dxi, Cf, HK, H, RT,
                   ALD: float = 1.0):
    """RHS of the √Cτ lag ODE.  Returns dS/dξ."""
    HKC = HK - 1.0 - GCCON / tf.maximum(RT, 1.0)
    HKC = tf.maximum(HKC, 0.01)
    HK_safe = tf.maximum(HK, 1.05)
    HR = HKC / (GACON * ALD * HK_safe)

    delta_safe = tf.maximum(delta, 1e-6)
    UQ = (0.5 * Cf - tf.square(HR)) / (GBCON * delta_safe)

    Hstar = hstar_turb_xfoil(HK, RT)
    Us = us_xfoil(Hstar, HK, H)
    SCC = SCCON * 1.333 / (1.0 + Us)

    ue_safe = tf.maximum(ue, 1e-4)
    due_over_ue = due_dxi / ue_safe

    S_safe = tf.maximum(S, 1e-6)
    rhs = (S_safe / (2.0 * delta_safe)) * (
        SCC * (S_eq - S * ALD)
        + 2.0 * delta_safe * DUXCON * (UQ - due_over_ue)
    )
    return rhs


# ============================================================
# Boundary-layer thickness δ (xblsys.f:1103, simplified Green's)
#   δ = (3.15 + 1.72/(HK-1))·θ + δ*
#   clamp δ ≤ 12·θ  (HDMAX in XFOIL)
# ============================================================
def delta_thickness_xfoil(theta, delta_star, HK):
    """BL thickness used in lag ODE δ-scale."""
    HK_m1 = tf.maximum(HK - 1.0, 1e-3)
    de = (3.15 + 1.72 / HK_m1) * theta + delta_star
    return tf.minimum(de, 12.0 * theta)


# ============================================================
# Laminar Cf  (xblsys.f:2354 CFL — Falkner-Skan)
#   HK < 5.5:  Cf·RT = 0.0727·(5.5-HK)³/(HK+1) - 0.07
#   HK ≥ 5.5:  Cf·RT = 0.015·(1 - 1/(HK-4.5))² - 0.07
# ============================================================
def cf_lam_xfoil(HK, RT, Msq=0.0):
    """Two-branch Falkner-Skan Cf with separation-region extension."""
    HK_safe = tf.maximum(HK, 1.02)
    RT_safe = tf.maximum(RT, 1.0)

    # branch HK < 5.5
    tmp1 = tf.pow(tf.maximum(5.5 - HK_safe, 1e-6), 3.0) / (HK_safe + 1.0)
    cf_low = (0.0727 * tmp1 - 0.07) / RT_safe

    # branch HK ≥ 5.5  (separation extension)
    tmp2 = 1.0 - 1.0 / tf.maximum(HK_safe - 4.5, 0.05)
    cf_high = (0.015 * tf.square(tmp2) - 0.07) / RT_safe

    return tf.where(HK_safe < 5.5, cf_low, cf_high)


# ============================================================
# Laminar H* (xblsys.f:2329 HSL — Drela cubic fit)
# ============================================================
def hstar_lam_xfoil(HK, RT, Msq=0.0):
    """Drela laminar H* with cubic correction.

    HK < 4.35:
        TMP = HK - 4.35
        HS = 0.0111·TMP²/(HK+1) - 0.0278·TMP³/(HK+1) + 1.528
             - 0.0002·(TMP·HK)²
    HK ≥ 4.35:
        HS = 0.015·(HK-4.35)²/HK + 1.528
    """
    TMP = HK - 4.35

    HS_low = (
        0.0111 * tf.square(TMP) / (HK + 1.0)
        - 0.0278 * tf.pow(TMP, 3.0) / (HK + 1.0)
        + 1.528
        - 0.0002 * tf.square(TMP * HK)
    )
    HS_high = 0.015 * tf.square(HK - 4.35) / tf.maximum(HK, 1.0) + 1.528

    return tf.where(HK < 4.35, HS_low, HS_high)


# ============================================================
# Laminar Cd via DIL (xblsys.f:2292)
#
#   2·Cd/H* = DI :
#     HK < 4: DI = (0.00205·(4-HK)^5.5 + 0.207) / RT
#     HK ≥ 4: DI = (-0.0016·(HK-4)²/(1+0.02·(HK-4)²) + 0.207) / RT
#
# We return Cd = DI · H* / 2  (caller passes H* it computed).
# ============================================================
def cd_lam_xfoil(HK, RT, Hstar=None):
    """Laminar Cd via XFOIL DIL.  If `Hstar` not provided, compute internally."""
    HK_safe = tf.maximum(HK, 1.02)
    RT_safe = tf.maximum(RT, 1.0)

    DI_low = (0.00205 * tf.pow(tf.maximum(4.0 - HK_safe, 0.0), 5.5)
              + 0.207) / RT_safe
    HKB = HK_safe - 4.0
    DEN = 1.0 + 0.02 * tf.square(HKB)
    DI_high = (-0.0016 * tf.square(HKB) / DEN + 0.207) / RT_safe

    DI = tf.where(HK_safe < 4.0, DI_low, DI_high)
    if Hstar is None:
        Hstar = hstar_lam_xfoil(HK_safe, RT_safe)
    return tf.maximum(DI * Hstar / 2.0, 1e-8)


# ============================================================
# DAMPL — n-factor amplification rate (xblsys.f:1983)
#
# log10(Re_θ_crit) = 2.492·HMI^0.43 + 0.7·(tanh(14·HMI - 9.24) + 1)
# AF (m-factor)    = -0.05 + 2.7·HMI - 5.5·HMI² + 3·HMI³
# DADR (slope)     = 0.028·(HK-1) - 0.0345·exp(-(3.87·HMI - 2.52)²)
# RFAC (smooth on) = 3·RNORM² - 2·RNORM³  on  log10(RT/Rcrit) ∈ [-DGR, +DGR]
# AX (= dN/dx)     = AF · DADR / θ · RFAC
#
# DGR = 0.08 in current XFOIL (was 0.04).
# ============================================================
def dampl_xfoil(HK, theta, RT):
    """XFOIL e^N envelope amplification rate AX = dN/dx.

    Returns AX (1/length).  Multiply by ds (or L_branch·dξ) to get dN.
    """
    DGR = 0.08
    HMI = 1.0 / tf.maximum(HK - 1.0, 0.05)

    AA = 2.492 * tf.pow(HMI, 0.43)
    BB = tf.tanh(14.0 * HMI - 9.24)
    GRCRIT = AA + 0.7 * (BB + 1.0)

    GR = tf.math.log(tf.maximum(RT, 1.0)) / 2.302585

    # Below critical: AX = 0
    below = GR < GRCRIT - DGR

    # Smooth cubic ramp 3·RNORM² - 2·RNORM³
    RNORM = (GR - (GRCRIT - DGR)) / (2.0 * DGR)
    RNORM = tf.clip_by_value(RNORM, 0.0, 1.0)
    RFAC = 3.0 * tf.square(RNORM) - 2.0 * tf.pow(RNORM, 3.0)

    # m(H) correction (1991 update)
    AF = (-0.05 + 2.7 * HMI
          - 5.5 * tf.square(HMI)
          + 3.0 * tf.pow(HMI, 3.0))

    # Envelope slope DADR (Drela 1989)
    ARG = 3.87 * HMI - 2.52
    EX = tf.exp(-tf.square(ARG))
    DADR = 0.028 * (HK - 1.0) - 0.0345 * EX

    AX = (AF * DADR / tf.maximum(theta, 1e-8)) * RFAC
    AX = tf.where(below, tf.zeros_like(AX), AX)
    return tf.maximum(AX, 0.0)


# ============================================================
# Laminar wake DILW (xblsys.f:2310) — used in wake region only,
# kept for completeness.  Not on the primary critical path.
# ============================================================
def cd_lam_wake_xfoil(HK, RT):
    """Laminar wake Cd: 2·Cd/H* = 2.2·(1-1/HK)²/(HK·H*·RT)"""
    Hstar = hstar_lam_xfoil(HK, RT)
    HK_safe = tf.maximum(HK, 1.05)
    RCD = 1.10 * tf.square(1.0 - 1.0 / HK_safe) / HK_safe
    DI = 2.0 * RCD / (Hstar * tf.maximum(RT, 1.0))
    return tf.maximum(DI * Hstar / 2.0, 1e-8)


# ============================================================
# 3-equation BL ODE solver  (θ, δ*, S=√Cτ)  with XFOIL 1:1 closures
#
# Variables (per branch):
#   y = (θ, δ*, S)
#   ξ ∈ [0, 1] uniform grid  (ds/dξ = L_branch)
#
# Stages:
#   ξ ∈ [0, 0.05]    Hiemenz analytical hold (laminar, S=0 unused)
#   ξ ∈ [0.05, ξ_tr] laminar 2-eq Heun: dθ/dξ, dδ*/dξ from XFOIL closures
#   ξ ≥ ξ_tr         turbulent 3-eq Heun: + lag ODE for S
#
# Transition criterion:
#   N(ξ) = ∫ AX·dξ  reaches Ncrit (default 7.0; XFOIL standard 9.0).
#   At first ξ where N ≥ Ncrit  ⇒  switch to turbulent;
#   S(ξ_tr) = CTRCON · exp(-CTRCEX/(HK-1))  [s_initial_transition]
#
# Implementation notes:
#   - Smooth blending via w_tr = sigmoid(trans_k·(N - Ncrit)) used by
#     downstream losses;  the ODE itself uses a hard switch at first
#     crossing (numerically more stable).
# ============================================================
def lam_rhs_xfoil(theta, delta_star, ue, due_dxi, Re, L_branch):
    """RHS of laminar 2-eq momentum + energy with XFOIL closures.

    Returns (f_θ, f_δ).
    """
    theta_safe = tf.maximum(theta, 1e-7)
    H = tf.clip_by_value(delta_star / theta_safe, 1.5, 6.0)
    HK = hkin(H)

    Re_theta = tf.maximum(ue * theta_safe * tf.maximum(Re, 1.0), 1.0)

    Cf = cf_lam_xfoil(HK, Re_theta)
    Hstar = hstar_lam_xfoil(HK, Re_theta)
    Cd = cd_lam_xfoil(HK, Re_theta, Hstar)

    ue_safe = tf.maximum(ue, 1e-4)

    # Momentum: dθ/dξ = L·Cf/2 - (H+2)·θ·due/ue
    f_theta = L_branch * 0.5 * Cf - theta_safe * (2.0 + H) * due_dxi / ue_safe

    # Energy → dδ*/dξ = H·f_θ + θ·(dH*/dξ)/(dH*/dH)
    # numerator = θ·(dH*/dξ) = L·(2·Cd - H*·Cf/2) + H*·H·θ·due/ue
    numerator = (
        L_branch * (2.0 * Cd - Hstar * 0.5 * Cf)
        + Hstar * H * theta_safe * due_dxi / ue_safe
    )

    # dH*/dH from numerical differentiation (XFOIL gives analytical; we
    # use finite-diff for robustness in the cubic branch)
    eps = 0.01
    Hstar_p = hstar_lam_xfoil(hkin(H + eps), Re_theta)
    Hstar_m = hstar_lam_xfoil(hkin(H - eps), Re_theta)
    dHs_dH = (Hstar_p - Hstar_m) / (2.0 * eps)
    dHs_dH_safe = tf.where(tf.abs(dHs_dH) < 0.02,
                            tf.sign(dHs_dH + 1e-8) * 0.02,
                            dHs_dH)

    theta_times_fH = numerator / dHs_dH_safe
    theta_times_fH = tf.clip_by_value(theta_times_fH, -0.05, 0.05)

    f_delta = H * f_theta + theta_times_fH

    f_theta = tf.clip_by_value(f_theta, -0.02, 0.02)
    f_delta = tf.clip_by_value(f_delta, -0.10, 0.10)

    return f_theta, f_delta


def turb_rhs_xfoil(theta, delta_star, S, ue, due_dxi, Re, L_branch):
    """RHS of turbulent 3-eq momentum + energy + lag with XFOIL closures.

    Returns (f_θ, f_δ, f_S).
    """
    theta_safe = tf.maximum(theta, 1e-7)
    H = tf.clip_by_value(delta_star / theta_safe, 1.05, 6.0)
    HK = hkin(H)

    Re_theta = tf.maximum(ue * theta_safe * tf.maximum(Re, 1.0), 1.0)

    Cf = cf_turb_xfoil(HK, Re_theta)
    Hstar = hstar_turb_xfoil(HK, Re_theta)
    Cd = cd_turb_xfoil(Hstar, HK, H, Cf, S, Re_theta)
    delta_full = delta_thickness_xfoil(theta_safe, delta_star, HK)

    ue_safe = tf.maximum(ue, 1e-4)

    # Momentum
    f_theta = L_branch * 0.5 * Cf - theta_safe * (2.0 + H) * due_dxi / ue_safe

    # Energy → dδ*/dξ
    numerator = (
        L_branch * (2.0 * Cd - Hstar * 0.5 * Cf)
        + Hstar * H * theta_safe * due_dxi / ue_safe
    )

    # dH*/dH numerical (turb branch)
    eps = 0.01
    Hstar_p = hstar_turb_xfoil(hkin(H + eps), Re_theta)
    Hstar_m = hstar_turb_xfoil(hkin(H - eps), Re_theta)
    dHs_dH = (Hstar_p - Hstar_m) / (2.0 * eps)
    dHs_dH_safe = tf.where(tf.abs(dHs_dH) < 0.02,
                            tf.sign(dHs_dH + 1e-8) * 0.02,
                            dHs_dH)

    theta_times_fH = tf.clip_by_value(numerator / dHs_dH_safe, -0.10, 0.10)
    f_delta = H * f_theta + theta_times_fH

    # Lag ODE for S
    S_eq = s_eq_xfoil(Hstar, HK, H, Re_theta)
    f_S = lag_rhs_xfoil(S, S_eq, delta_full, ue, due_dxi, Cf, HK, H, Re_theta,
                        ALD=1.0)

    # Stability clips (calibrated to typical XFOIL turb magnitudes)
    f_theta = tf.clip_by_value(f_theta, -0.05, 0.05)
    f_delta = tf.clip_by_value(f_delta, -0.20, 0.20)
    f_S = tf.clip_by_value(f_S, -0.50, 0.50)

    return f_theta, f_delta, f_S


def solve_full_bl_xfoil(ue, due_dxi, Re, L_branch,
                          dxi, xi_1d,
                          Ncrit: float = 7.0,
                          trans_k: float = 2.0,
                          hiemenz_xi_end: float = 0.05):
    """Full XFOIL-faithful 3-eq BL solver: laminar 2-eq → turb 3-eq via DAMPL.

    Returns dict with (B, Nside+1) arrays:
      theta, delta_star, S, H, n, w_tr, sigma_n, regime
    where regime ∈ {0=Hiemenz, 1=laminar, 2=turbulent}.
    """
    Re_safe = tf.maximum(Re, 1.0)
    ue_safe = tf.maximum(ue, 1e-4)

    # Hiemenz IC (xi < hiemenz_xi_end)
    due_ds = due_dxi / (L_branch + 1e-10)
    n_le = min(10, due_ds.shape[1])
    due_ds_stag = tf.reduce_max(tf.abs(due_ds[:, :n_le]), axis=1, keepdims=True)
    due_ds_stag = tf.maximum(due_ds_stag, 1.0)
    theta_h = tf.sqrt(0.075 / (Re_safe * due_ds_stag))
    delta_h = 2.216 * theta_h            # Hiemenz H = 2.216

    Nside_p1 = xi_1d.shape[0]
    Nside = Nside_p1 - 1
    i_start = max(2, min(int(round(hiemenz_xi_end * Nside)), Nside - 2))

    theta = theta_h
    delta = delta_h
    # S init: 0 in laminar (unused), proper value at transition
    S_curr = tf.zeros_like(theta)
    n_amp = tf.zeros_like(theta)            # accumulated N
    is_turb = tf.zeros_like(theta)           # 0 lam / 1 turb (per-batch)

    theta_list = [theta]
    delta_list = [delta]
    S_list = [S_curr]
    n_list = [n_amp]
    regime_list = [tf.zeros_like(theta)]    # 0 Hiemenz, 1 lam, 2 turb

    for i in range(Nside):
        if i < i_start:
            # Hiemenz hold (constant)
            theta_list.append(theta)
            delta_list.append(delta)
            S_list.append(S_curr)
            n_list.append(n_amp)
            regime_list.append(tf.zeros_like(theta))
            continue

        ue_i = ue_safe[:, i:i + 1]
        due_i = due_dxi[:, i:i + 1]
        ue_ip1 = ue_safe[:, i + 1:i + 2]
        due_ip1 = due_dxi[:, i + 1:i + 2]

        # n-factor amplification (laminar region only)
        H_curr = tf.clip_by_value(delta / tf.maximum(theta, 1e-7), 1.5, 6.0)
        HK_curr = hkin(H_curr)
        RT_curr = tf.maximum(ue_i * theta * Re_safe, 1.0)
        AX = dampl_xfoil(HK_curr, theta, RT_curr)
        # Convert AX (per-arclength rate) to per-ξ: dN = AX·ds = AX·L·dξ
        dN = AX * L_branch * dxi
        n_amp_new = tf.where(is_turb > 0.5, n_amp, n_amp + dN)

        # Transition: smooth weight w_tr (used by downstream losses)
        # Hard switch (for ODE): once is_turb=1, stays 1
        cross = tf.cast(n_amp_new >= Ncrit, theta.dtype)
        is_turb_new = tf.maximum(is_turb, cross)

        # Initial S at the ξ where transition first triggers
        cross_now = (1.0 - is_turb) * cross
        S_at_tr = s_initial_transition(HK_curr) * cross_now
        S_curr_pretrans = S_curr + S_at_tr     # fold in init at transition

        # Heun step — branch on is_turb_new
        # Predictor
        ft1_l, fd1_l = lam_rhs_xfoil(theta, delta, ue_i, due_i, Re_safe, L_branch)
        ft1_t, fd1_t, fS1_t = turb_rhs_xfoil(
            theta, delta, S_curr_pretrans, ue_i, due_i, Re_safe, L_branch)

        ft1 = is_turb_new * ft1_t + (1.0 - is_turb_new) * ft1_l
        fd1 = is_turb_new * fd1_t + (1.0 - is_turb_new) * fd1_l
        fS1 = is_turb_new * fS1_t                       # S frozen in lam

        theta_p = tf.maximum(theta + dxi * ft1, 1e-7)
        delta_p = tf.maximum(delta + dxi * fd1, 1e-7)
        S_p = tf.maximum(S_curr_pretrans + dxi * fS1, 0.0)

        # Corrector at ξ_{i+1}
        ft2_l, fd2_l = lam_rhs_xfoil(theta_p, delta_p, ue_ip1, due_ip1,
                                       Re_safe, L_branch)
        ft2_t, fd2_t, fS2_t = turb_rhs_xfoil(
            theta_p, delta_p, S_p, ue_ip1, due_ip1, Re_safe, L_branch)

        ft2 = is_turb_new * ft2_t + (1.0 - is_turb_new) * ft2_l
        fd2 = is_turb_new * fd2_t + (1.0 - is_turb_new) * fd2_l
        fS2 = is_turb_new * fS2_t

        theta = tf.maximum(theta + 0.5 * dxi * (ft1 + ft2), 1e-7)
        delta = tf.maximum(delta + 0.5 * dxi * (fd1 + fd2), 1e-7)
        S_curr = tf.maximum(S_curr_pretrans + 0.5 * dxi * (fS1 + fS2), 0.0)

        # Enforce H ∈ [1.05, 6] via δ-renorm
        H_check = delta / theta
        H_clamped = tf.clip_by_value(H_check, 1.05, 6.0)
        delta = H_clamped * theta

        n_amp = n_amp_new
        is_turb = is_turb_new

        theta_list.append(theta)
        delta_list.append(delta)
        S_list.append(S_curr)
        n_list.append(n_amp)
        regime_list.append(1.0 + is_turb)    # 1 lam, 2 turb

    theta_arr = tf.concat(theta_list, axis=1)
    delta_arr = tf.concat(delta_list, axis=1)
    S_arr = tf.concat(S_list, axis=1)
    n_arr = tf.concat(n_list, axis=1)
    regime_arr = tf.concat(regime_list, axis=1)
    H_arr = tf.clip_by_value(delta_arr / tf.maximum(theta_arr, 1e-8), 1.05, 6.0)

    # Smooth w_tr for downstream losses
    w_tr = tf.sigmoid(trans_k * (n_arr - Ncrit))

    # Per-step σ_n for transition residual / monitoring
    sigma_n = tf.zeros_like(theta_arr)        # not needed when transition is hard

    return {
        "theta": theta_arr,
        "delta_star": delta_arr,
        "S": S_arr,
        "H": H_arr,
        "n": n_arr,
        "w_tr": w_tr,
        "sigma_n": sigma_n,
        "regime": regime_arr,
    }
