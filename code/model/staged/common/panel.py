# -*- coding: utf-8 -*-
"""
Linear vortex / source panel kernels with analytical influence coefficients.

Self-influence convention (CCW airfoil contour, left-hand-normal eta_loc):
    eta_loc > 0  ⟹  inside airfoil
    eta_loc < 0  ⟹  outside (physical flow region)

For the self-panel (i==j) eta_loc = 0 exactly; we force eta to -eta_min
(slightly outside) so the analytical limit returns the *exterior* velocity:
    I1 = -π,  I2 = 0
    vortex:  u_ξ_self = -γ_mid/2,   u_η_self = -Δγ/(2π)
    source:  u_ξ_self = -Δσ/(2π),  u_η_self = -σ_mid/2

This is the standard Hess–Smith self-influence; it was missing in the
original v51e (both diagonal masks zeroed it out) and is the single
biggest physics fix in the v51-fix lineage.

Extracted from steady51.py.  Numerical equivalence is enforced by
staged/common/_smoke_test.py.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

DTYPE = tf.float32
INV_2PI = tf.constant(1.0 / (2.0 * np.pi), dtype=DTYPE)


def _self_mask(B, Np, dtype=DTYPE):
    """Returns (1, Np, Np) tensor with 1 on diagonal, 0 elsewhere."""
    return tf.eye(Np, dtype=dtype)[None, :, :]


# ============================================================
# Analytical influence-coefficient integrals
# ============================================================
def linear_panel_integrals(geom: dict, eta_min: float = 1e-6):
    """
    Compute analytical I1, I2 and panel-local coordinates for all
    (collocation i, source panel j) pairs.

    For each panel j with endpoints (x1j, y1j) → (x2j, y2j) and a field
    point at (xc_i, yc_i):

        ξ_loc  = (P - P1) · t̂      (along panel)
        η_loc  = (P - P1) · n̂_left  (left-hand normal — INWARD for CCW)

    Analytical integrals (exact for a *linear* vortex / source panel):

        I1 = atan(ξ/η) - atan((ξ-L)/η)         [subtended angle]
        I2 = ½ · ln(r1²/r2²)                    [log distance ratio]

    Self-panel: ξ = L/2, η = 0 → forced to -eta_min (exterior side)
    so the limit gives I1 = -π, I2 = 0 (correct Hess–Smith jump).

    Returns
    -------
    I1, I2     : (B, Np_eval, Np_src)
    xi_loc     : (B, Np_eval, Np_src)
    eta_clamped: (B, Np_eval, Np_src)
    Lp         : (B, Np_src)
    """
    x1 = geom["x1"]; y1 = geom["y1"]
    x2 = geom["x2"]; y2 = geom["y2"]
    xc = geom["xc"]; yc = geom["yc"]

    dxp = x2 - x1
    dyp = y2 - y1
    Lp = tf.sqrt(dxp * dxp + dyp * dyp + 1e-12)   # (B, Np)

    tx = dxp / Lp
    ty = dyp / Lp

    # Vector P_i - P1_j
    dx_ip = xc[:, :, None] - x1[:, None, :]   # (B, Ni, Nj)
    dy_ip = yc[:, :, None] - y1[:, None, :]

    TX = tx[:, None, :]
    TY = ty[:, None, :]
    xi_loc = dx_ip * TX + dy_ip * TY            # along panel
    eta_loc = -dx_ip * TY + dy_ip * TX          # left-hand normal (inward)

    LP = Lp[:, None, :]

    # Clamp |eta| to eta_min for numerical stability of atan / log
    eta_clamped = tf.where(
        eta_loc >= 0,
        tf.maximum(eta_loc, eta_min),
        tf.minimum(eta_loc, -eta_min),
    )

    # Self-panel correction: force eta to -eta_min (exterior side)
    Np = tf.shape(xi_loc)[1]
    self_mask = _self_mask(tf.shape(xi_loc)[0], Np, dtype=eta_clamped.dtype)
    eta_clamped = eta_clamped - 2.0 * eta_min * self_mask

    # Distances squared
    r1_sq = xi_loc * xi_loc + eta_clamped * eta_clamped
    xi_m_L = xi_loc - LP
    r2_sq = xi_m_L * xi_m_L + eta_clamped * eta_clamped

    r1_sq = tf.maximum(r1_sq, 1e-30)
    r2_sq = tf.maximum(r2_sq, 1e-30)

    I1 = tf.atan(xi_loc / eta_clamped) - tf.atan(xi_m_L / eta_clamped)
    I2 = 0.5 * tf.math.log(r1_sq / r2_sq)
    I2 = tf.clip_by_value(I2, -10.0, 10.0)

    return I1, I2, xi_loc, eta_clamped, Lp


# ============================================================
# Vortex panel — induced velocity (global frame)
# ============================================================
def induced_velocity_from_gamma_linear(geom: dict, gamma_nodes,
                                        eta_min: float = 1e-6):
    """
    Velocity (u, v) induced at every panel midpoint by linear vortex
    panels of strength γ at the nodes.

    γ(τ) = γ_L·(1 - τ/L) + γ_R·(τ/L) on each panel.

    Returns
    -------
    u, v : (B, Np)
    """
    I1, I2, xi_loc, eta_loc, Lp = linear_panel_integrals(geom, eta_min)

    gamma_L = gamma_nodes[:, :-1]     # (B, Np)
    gamma_R = gamma_nodes[:, 1:]

    gL = gamma_L[:, None, :]          # (B, 1, Nj)
    gR = gamma_R[:, None, :]
    dg = gR - gL                       # Δγ
    LP = Lp[:, None, :]

    g_at_xi = gL + dg * xi_loc / LP

    inv2pi = INV_2PI
    u_xi_ij = inv2pi * (g_at_xi * I1 - dg * eta_loc / LP * I2)
    u_eta_ij = -inv2pi * (g_at_xi * I2 + dg * eta_loc / LP * I1 - dg)

    # Panel j tangent / normal in global frame  → rotate per-panel local → global
    dxp = geom["x2"] - geom["x1"]
    dyp = geom["y2"] - geom["y1"]
    txj = dxp / Lp
    tyj = dyp / Lp
    TXj = txj[:, None, :]
    TYj = tyj[:, None, :]

    u_ij_global = u_xi_ij * TXj - u_eta_ij * TYj
    v_ij_global = u_xi_ij * TYj + u_eta_ij * TXj

    u = tf.reduce_sum(u_ij_global, axis=-1)
    v = tf.reduce_sum(v_ij_global, axis=-1)
    return u, v


# ============================================================
# Source panel — induced velocity (global frame)
# ============================================================
def induced_velocity_from_source_linear(geom: dict, sigma_nodes,
                                          eta_min: float = 1e-6):
    """
    Velocity (u, v) at every panel midpoint induced by linear source panels
    of strength σ at the nodes.  σ(τ) linear between σ_L and σ_R.
    """
    I1, I2, xi_loc, eta_loc, Lp = linear_panel_integrals(geom, eta_min)

    sigma_L = sigma_nodes[:, :-1]
    sigma_R = sigma_nodes[:, 1:]

    sL = sigma_L[:, None, :]
    sR = sigma_R[:, None, :]
    ds = sR - sL
    LP = Lp[:, None, :]

    s_at_xi = sL + ds * xi_loc / LP

    inv2pi = INV_2PI
    u_xi_ij = inv2pi * (s_at_xi * I2 + ds * eta_loc / LP * I1 - ds)
    u_eta_ij = inv2pi * (s_at_xi * I1 - ds * eta_loc / LP * I2)

    dxp = geom["x2"] - geom["x1"]
    dyp = geom["y2"] - geom["y1"]
    TXj = (dxp / Lp)[:, None, :]
    TYj = (dyp / Lp)[:, None, :]

    u_ij_global = u_xi_ij * TXj - u_eta_ij * TYj
    v_ij_global = u_xi_ij * TYj + u_eta_ij * TXj

    u = tf.reduce_sum(u_ij_global, axis=-1)
    v = tf.reduce_sum(v_ij_global, axis=-1)
    return u, v


# ============================================================
# Wake source panels (XFOIL-style, no self-influence concerns since
# wake is downstream of airfoil — no overlap)
# ============================================================
def induced_velocity_from_wake_source_analytical(
    airfoil_xc, airfoil_yc, wake_geom: dict, sigma_wake_nodes,
    inv_2pi=INV_2PI,
):
    """
    Velocity at airfoil collocation points induced by linear source panels
    placed in the wake.  Same closed-form atan / log integrals as for the
    airfoil panels (kernel rotated 90° vs vortex).

    Parameters
    ----------
    airfoil_xc, airfoil_yc : (B, Np)
    wake_geom              : dict with x1, y1, tx, ty, Lp  (B, Nw)
    sigma_wake_nodes       : (B, Nw+1)
    inv_2pi                : scalar

    Returns
    -------
    u_src_wake, v_src_wake : (B, Np)
    """
    x1_w = wake_geom["x1"]
    y1_w = wake_geom["y1"]
    Lp = wake_geom["Lp"]
    tx_w = wake_geom["tx"]
    ty_w = wake_geom["ty"]

    sigma_L = sigma_wake_nodes[:, :-1]
    sigma_R = sigma_wake_nodes[:, 1:]

    dx_raw = airfoil_xc[:, :, None] - x1_w[:, None, :]
    dy_raw = airfoil_yc[:, :, None] - y1_w[:, None, :]

    TX = tx_w[:, None, :]
    TY = ty_w[:, None, :]
    LP = Lp[:, None, :]

    xi_loc = dx_raw * TX + dy_raw * TY
    eta_loc = dx_raw * TY - dy_raw * TX     # (ty, -tx) convention

    eta_min = 0.1 * tf.reduce_mean(LP, axis=-1, keepdims=True)
    eta_abs = tf.abs(eta_loc)
    eta_sign = tf.sign(eta_loc)
    eta_clamped = eta_sign * tf.maximum(eta_abs, eta_min)

    r1_sq = xi_loc ** 2 + eta_clamped ** 2 + 1e-20
    xi_mL = xi_loc - LP
    r2_sq = xi_mL ** 2 + eta_clamped ** 2 + 1e-20

    I1 = tf.atan2(eta_clamped, xi_loc) - tf.atan2(eta_clamped, xi_mL)
    I2 = 0.5 * tf.math.log(r1_sq / r2_sq)

    sL = sigma_L[:, None, :]
    sR = sigma_R[:, None, :]
    ds_sigma = sR - sL

    s_at_xi = sL + ds_sigma * xi_loc / LP
    u_xi_ij = inv_2pi * (s_at_xi * I2 + ds_sigma * eta_clamped / LP * I1 - ds_sigma)
    u_eta_ij = inv_2pi * (s_at_xi * I1 - ds_sigma * eta_clamped / LP * I2)

    u_ij_global = u_xi_ij * TX - u_eta_ij * TY
    v_ij_global = u_xi_ij * TY + u_eta_ij * TX

    u_src_wake = tf.reduce_sum(u_ij_global, axis=-1)
    v_src_wake = tf.reduce_sum(v_ij_global, axis=-1)
    return u_src_wake, v_src_wake


# ============================================================
# High-level surface velocity helpers
# ============================================================
def compute_inviscid_surface_velocity(geom: dict, gamma_nodes, alpha_deg,
                                       eta_min: float = 1e-6) -> dict:
    """
    Inviscid surface velocity = U∞ + γ-vortex induced.
    """
    pi_over_180 = tf.constant(np.pi / 180.0, dtype=DTYPE)
    alpha_rad = alpha_deg * pi_over_180
    Ux = tf.cos(alpha_rad)
    Uy = tf.sin(alpha_rad)

    u_ind, v_ind = induced_velocity_from_gamma_linear(geom, gamma_nodes, eta_min)

    Ut = (Ux + u_ind) * geom["tx"] + (Uy + v_ind) * geom["ty"]
    Un = (Ux + u_ind) * geom["nx"] + (Uy + v_ind) * geom["ny"]

    return {"Ut": Ut, "Un": Un, "u_ind": u_ind, "v_ind": v_ind}


def compute_surface_velocity_coupled(geom: dict, gamma_nodes, sigma_nodes,
                                      alpha_deg, eta_min: float = 1e-6) -> dict:
    """
    Full coupled surface velocity = U∞ + vortex(γ) + source(σ).
    Used in BL/coupled stages.
    """
    pi_over_180 = tf.constant(np.pi / 180.0, dtype=DTYPE)
    alpha_rad = alpha_deg * pi_over_180
    Ux = tf.cos(alpha_rad)
    Uy = tf.sin(alpha_rad)

    u_v, v_v = induced_velocity_from_gamma_linear(geom, gamma_nodes, eta_min)
    u_s, v_s = induced_velocity_from_source_linear(geom, sigma_nodes, eta_min)

    u_total = Ux + u_v + u_s
    v_total = Uy + v_v + v_s

    Ut = u_total * geom["tx"] + v_total * geom["ty"]
    Un = u_total * geom["nx"] + v_total * geom["ny"]

    return {
        "Ut": Ut, "Un": Un,
        "u_vort": u_v, "v_vort": v_v,
        "u_src": u_s, "v_src": v_s,
    }
