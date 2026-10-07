# -*- coding: utf-8 -*-
"""
Branch-side sampling (upper/lower) from the closed contour.

Given the inviscid tangential velocity Ut on contour panel midpoints and a
stagnation parameter s_stag, samples ue_inv(ξ) along the upper and lower
branch in branch-local coordinates ξ ∈ [0, 1] (ξ=0 at stagnation, ξ=1 at TE).

Sign convention (matches steady51):
    upper:   contour goes TE → LE → stag, so panel tangent points "upstream"
             when traveling LE→stag.  Therefore  ue_inv_u = -Ut(s_u).
    lower:   contour goes stag → TE, panel tangent matches flow direction.
             Therefore  ue_inv_l = +Ut(s_l).
"""
from __future__ import annotations

import tensorflow as tf

from .interp import interp1d_common_grid

DTYPE = tf.float32


def _batched_piecewise_interp(x_query, x_data, y_data):
    """Per-batch linear interpolation.

    x_query : (B, M)   query points
    x_data  : (B, N)   monotone-increasing x grid per batch
    y_data  : (B, N)   values at x_data per batch
    returns : (B, M)
    """
    idx_hi = tf.searchsorted(x_data, x_query, side="left")
    N = tf.shape(x_data)[1]
    idx_hi = tf.clip_by_value(idx_hi, 1, N - 1)
    idx_lo = idx_hi - 1
    x0 = tf.gather(x_data, idx_lo, batch_dims=1)
    x1 = tf.gather(x_data, idx_hi, batch_dims=1)
    y0 = tf.gather(y_data, idx_lo, batch_dims=1)
    y1 = tf.gather(y_data, idx_hi, batch_dims=1)
    t = (x_query - x0) / (x1 - x0 + 1e-12)
    return y0 + t * (y1 - y0)


def sample_branch_ue_arc_uniform(Ut_base, s_stag,
                                   s_panels_1d, s_nodes_1d,
                                   ds_panels, xi_1d):
    """Arc-length-uniform branch sampling.

    ξ ∈ [0,1] now corresponds to a uniform arc-length step from the
    stagnation point along each branch.  This matches the BL equation's
    arc-length normalization (L_branch · dξ = ds_arc), fixing the
    parametric-vs-arc mismatch (cosine-spaced panels → 4× spatial
    distortion at mid-chord under the old linear-in-s mapping).

    Parameters
    ----------
    Ut_base     : (B, Np)        contour Ut at panel midpoints
    s_stag      : (B, 1)         stagnation parameter
    s_panels_1d : (Np,)          panel-midpoint parametric s values
    s_nodes_1d  : (Np+1,)        panel-boundary parametric s values
    ds_panels   : (B, Np)        per-panel arc length
    xi_1d       : (Nside+1,)     branch ξ grid on [0, 1]
    """
    s_query_min = s_panels_1d[0]
    s_query_max = s_panels_1d[-1]
    s_stag_clip = tf.clip_by_value(s_stag, s_query_min, s_query_max)

    # Cumulative arc at nodes
    cum_arc = tf.concat([tf.zeros_like(ds_panels[:, :1]),
                            tf.cumsum(ds_panels, axis=1)], axis=1)  # (B, Np+1)
    L_total = cum_arc[:, -1:]                                          # (B, 1)
    arc_at_stag = interp1d_common_grid(s_nodes_1d, cum_arc, s_stag_clip)  # (B, 1)
    L_upper = tf.maximum(arc_at_stag, 1e-6)
    L_lower = tf.maximum(L_total - arc_at_stag, 1e-6)

    xi = xi_1d[None, :]                                              # (1, Nside+1)
    # Arc-position on contour (cum_arc frame) for each ξ:
    # Upper branch ξ=0 at stag (arc=arc_at_stag) → ξ=1 at TE_upper (arc=0)
    arc_u = (1.0 - xi) * arc_at_stag
    # Lower branch ξ=0 at stag (arc=arc_at_stag) → ξ=1 at TE_lower (arc=L_total)
    arc_l = arc_at_stag + xi * L_lower

    # Inverse: arc → s_panels via batched piecewise linear interp of
    # (cum_arc_nodes → s_nodes).  Then sample Ut at those s_panels.
    s_nodes_2d = tf.broadcast_to(s_nodes_1d[None, :], tf.shape(cum_arc))
    s_u = _batched_piecewise_interp(arc_u, cum_arc, s_nodes_2d)
    s_l = _batched_piecewise_interp(arc_l, cum_arc, s_nodes_2d)
    s_u = tf.clip_by_value(s_u, s_query_min, s_query_max)
    s_l = tf.clip_by_value(s_l, s_query_min, s_query_max)

    Ut_u = interp1d_common_grid(s_panels_1d, Ut_base, s_u)
    Ut_l = interp1d_common_grid(s_panels_1d, Ut_base, s_l)

    return {
        "xi_u": tf.broadcast_to(xi, tf.shape(s_u)),
        "xi_l": tf.broadcast_to(xi, tf.shape(s_l)),
        "s_u": s_u, "s_l": s_l,
        "Ut_u": Ut_u, "Ut_l": Ut_l,
        "ue_inv_u": -Ut_u, "ue_inv_l": Ut_l,
        "s_stag_clip_bl": s_stag_clip,
        "arc_at_stag": arc_at_stag,
        "L_upper": L_upper, "L_lower": L_lower,
    }


def sample_branch_ue_from_contour(Ut_base, s_stag,
                                    s_panels_1d, xi_1d):
    """
    Sample upper/lower branch ue from contour Ut.

    Parameters
    ----------
    Ut_base    : (B, Np)        contour Ut at panel midpoints
    s_stag     : (B, 1)         stagnation parameter (clipped externally if needed)
    s_panels_1d: (Np,)          panel-midpoint s values (uniform)
    xi_1d      : (Nside+1,)     branch ξ grid on [0, 1]

    Returns
    -------
    dict with keys: xi_u, xi_l, s_u, s_l, Ut_u, Ut_l, ue_inv_u, ue_inv_l,
                     s_stag_clip_bl
    """
    s_query_min = s_panels_1d[0]
    s_query_max = s_panels_1d[-1]

    s_stag_clip = tf.clip_by_value(s_stag, s_query_min, s_query_max)
    xi = xi_1d[None, :]

    s_u = s_stag_clip * (1.0 - xi)
    s_l = s_stag_clip + (1.0 - s_stag_clip) * xi

    s_u = tf.clip_by_value(s_u, s_query_min, s_query_max)
    s_l = tf.clip_by_value(s_l, s_query_min, s_query_max)

    Ut_u = interp1d_common_grid(s_panels_1d, Ut_base, s_u)
    Ut_l = interp1d_common_grid(s_panels_1d, Ut_base, s_l)

    ue_inv_u = -Ut_u
    ue_inv_l = Ut_l

    return {
        "xi_u": tf.broadcast_to(xi, tf.shape(s_u)),
        "xi_l": tf.broadcast_to(xi, tf.shape(s_l)),
        "s_u": s_u, "s_l": s_l,
        "Ut_u": Ut_u, "Ut_l": Ut_l,
        "ue_inv_u": ue_inv_u, "ue_inv_l": ue_inv_l,
        "s_stag_clip_bl": s_stag_clip,
    }


def map_delta_to_contour_nodes(delta_u, delta_l, s_stag,
                                 s_nodes_1d, xi_1d):
    """
    Map δ*(ξ) from upper/lower branches back to the closed contour nodes.

    Contour: upper TE (s=0) → LE (s≈s_stag) → lower TE (s=1).
    Upper branch: ξ=0 at stagnation (s=s_stag), ξ=1 at upper TE (s=0).
    Lower branch: ξ=0 at stagnation (s=s_stag), ξ=1 at lower TE (s=1).

    Returns
    -------
    delta_star_nodes : (B, Nnode)
    """
    s_2d = s_nodes_1d[None, :]                              # (1, Nnode)

    xi_u_query = 1.0 - s_2d / (s_stag + 1e-8)
    xi_u_query = tf.clip_by_value(xi_u_query, 0.0, 1.0)

    xi_l_query = (s_2d - s_stag) / (1.0 - s_stag + 1e-8)
    xi_l_query = tf.clip_by_value(xi_l_query, 0.0, 1.0)

    delta_u_at_nodes = interp1d_common_grid(xi_1d, delta_u, xi_u_query)
    delta_l_at_nodes = interp1d_common_grid(xi_1d, delta_l, xi_l_query)

    w_upper = tf.sigmoid(200.0 * (s_stag - s_2d))           # smooth step
    return w_upper * delta_u_at_nodes + (1.0 - w_upper) * delta_l_at_nodes
