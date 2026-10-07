# -*- coding: utf-8 -*-
"""
Source-related primitives for the viscous correction stage.

  - compute_source_strength :   σ(s) = d(ue · δ*)/ds   on contour nodes
  - compute_wake_bl_analytical: analytical wake BL  H(s) = 1 + (H₀-1)·e^(-s/L)
                                                     σ(s) = θ · dH/ds

Extracted verbatim from steady51.py.
"""
from __future__ import annotations

import tensorflow as tf

from .interp import interp1d_common_grid, central_diff_1d_nonuniform

DTYPE = tf.float32


def compute_source_strength(Ut_panels, delta_star_nodes, ds_panels,
                              s_panels_1d, s_nodes_1d):
    """
    Source strength on contour nodes:  σ(s) = d(ue · δ*)/ds.

    ue is recovered from |Ut| (sign of Ut depends on contour-tangent direction;
    magnitude is the physical edge speed).  Mass-defect derivative uses the
    non-uniform central-difference operator with the actual panel arc-lengths
    `ds_panels`.

    Parameters
    ----------
    Ut_panels        : (B, Np)        contour Ut at panel midpoints
    delta_star_nodes : (B, Nnode)     δ* at contour nodes (already mapped from branches)
    ds_panels        : (B, Np)        actual panel arc-lengths
    s_panels_1d      : (Np,)          panel-midpoint s grid (uniform)
    s_nodes_1d       : (Nnode,)       contour-node s grid (uniform)

    Returns
    -------
    sigma_nodes : (B, Nnode)
    """
    B = tf.shape(Ut_panels)[0]
    Nnode = tf.shape(s_nodes_1d)[0]
    s_nodes_2d = tf.broadcast_to(s_nodes_1d[None, :], [B, Nnode])
    Ut_at_nodes = interp1d_common_grid(s_panels_1d, Ut_panels, s_nodes_2d)
    ue_nodes = tf.abs(Ut_at_nodes)

    mass_defect = ue_nodes * delta_star_nodes
    sigma_nodes = central_diff_1d_nonuniform(mass_defect, ds_panels)
    # ============================================================
    # v51-fix [2026-05-08] — σ magnitude clip
    #
    # Diagnosed root cause of Stage-4 Cp / Cd_p explosion at strong-APG
    # cases (NACA 0012 α=8° R=1e6: |σ|_max=559, picard_res=12, Cp_min=
    # −420, Cd_p=1647).  Across the validation set, attached-flow cases
    # have |σ|_max ∈ [0.4, 25].  The blow-up sequence is:
    #
    #   strong LE suction (|Ut|≈2.2) + large δ* (≈0.19, near-stall)
    #   → mass_defect (ue·δ*) has a sharp LE gradient
    #   → cosine-spaced ds_min ≈ 0.005 amplifies central-diff
    #   → σ spikes O(500)
    #   → induced velocity spike via the source kernel
    #   → Picard fixed-point self-amplifies (even at relax=0.2)
    #
    # Clipping σ at ±50 (= 2× observed attached-flow max of 25) leaves
    # all healthy cases untouched and breaks the divergence feedback
    # loop in the stiff cases.  Cl computation does NOT use σ
    # (Cl = 2·∮γ ds), so Cl predictions are bit-exact unchanged; only
    # Cp / Cd_p / Ut from the *coupled* path are affected.
    #
    # Op3.predict_coupled also uses this helper (single forward, no
    # Picard) — its attached-flow σ is already < 25 so the clip is a
    # no-op there.  At inference / validation time this fix is enough;
    # no retrain required.
    # ============================================================
    sigma_nodes = tf.clip_by_value(sigma_nodes, -50.0, 50.0)
    return sigma_nodes


def compute_wake_bl_analytical(theta_u_te, theta_l_te,
                                delta_u_te, delta_l_te,
                                te_gap, wake_s_nodes, L_relax: float = 1.0):
    """
    Analytical wake boundary-layer evolution.

      θ_wake(s) ≈ θ_u + θ_l               (no friction in wake → θ const)
      H_wake(s) = 1 + (H₀ - 1)·exp(-s / L_relax)
      σ_wake(s) = θ · dH/ds

    H₀ is clipped to [1, 6] to prevent explosion when θ ≈ 0 during early
    BL training (te_gap / 1e-8 ≫ 1).

    Returns
    -------
    sigma_wake_nodes : (B, n_wake+1)
    theta_wake_far   : (B, 1)
    delta_wake_nodes : (B, n_wake+1)
    """
    theta_init = theta_u_te + theta_l_te
    delta_init = delta_u_te + delta_l_te + te_gap

    H_init_raw = delta_init / tf.maximum(theta_init, 1e-8)
    H_init = tf.clip_by_value(H_init_raw, 1.0, 6.0)

    s = wake_s_nodes[None, :]
    decay = tf.exp(-s / L_relax)
    H_wake = 1.0 + (H_init - 1.0) * decay
    delta_wake_nodes = H_wake * theta_init

    dH_ds = -(H_init - 1.0) / L_relax * decay
    sigma_wake_nodes = theta_init * dH_ds
    theta_wake_far = theta_init
    return sigma_wake_nodes, theta_wake_far, delta_wake_nodes
