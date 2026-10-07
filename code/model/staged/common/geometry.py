# -*- coding: utf-8 -*-
"""
Geometry primitives for airfoils — NACA 4-digit (legacy) and Kulfan CST
(Phase A generalization).

NACA 4-digit functions are kept verbatim from steady51.py (a NACA4 instance
remains a 3-parameter sample of the larger CST subspace; the two
representations are interchangeable via `naca4_to_cst_coeffs`).

All functions are TensorFlow ops — gradients propagate to shape_params /
CST coefficients and alpha_rad inputs.
"""
from __future__ import annotations

from math import comb

import numpy as np
import tensorflow as tf

DTYPE = tf.float32

# CST defaults — Kulfan class function exponents fix LE sqrt singularity
# and sharp (C^0) TE closure analytically.
CST_N1_DEFAULT = 0.5
CST_N2_DEFAULT = 1.0
CST_N_BERN_DEFAULT = 8     # Bernstein coefficients per surface → 16-dim shape


# ============================================================
# cosine spacing
# ============================================================
def cosine_spacing(n: int) -> np.ndarray:
    """Cosine-clustered spacing on [0, 1] with n points (LE & TE clustered)."""
    beta = np.linspace(0.0, np.pi, n, dtype=np.float32)
    return 0.5 * (1.0 - np.cos(beta))


# ============================================================
# NACA 4-digit thickness + camber
# ============================================================
def naca4_geometry_batch_tf(shape_params, n_surface_pts: int = 81):
    """
    Compute NACA 4-digit airfoil upper / lower surface coordinates.

    Parameters
    ----------
    shape_params : (B, 3)  =  [m (camber), p (camber location), t (thickness)]
    n_surface_pts : int    points per surface (cosine-clustered)

    Returns
    -------
    xu, yu : (B, n_surface_pts)  upper surface  (LE → TE)
    xl, yl : (B, n_surface_pts)  lower surface  (LE → TE)
    """
    m = shape_params[:, 0:1]
    p = shape_params[:, 1:2]
    t = shape_params[:, 2:3]

    x_np = cosine_spacing(n_surface_pts)[None, :]
    x = tf.constant(x_np, dtype=DTYPE)

    yt = 5.0 * t * (
        0.2969 * tf.sqrt(tf.maximum(x, 1e-12))
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1015 * x ** 4
    )

    eps = tf.constant(1e-8, dtype=DTYPE)
    left = x < tf.maximum(p, eps)

    yc_left = m / tf.maximum(p ** 2, eps) * (2.0 * p * x - x ** 2)
    yc_right = m / tf.maximum((1.0 - p) ** 2, eps) * (
        (1.0 - 2.0 * p) + 2.0 * p * x - x ** 2
    )
    yc = tf.where(left, yc_left, yc_right)

    dyc_left = 2.0 * m / tf.maximum(p ** 2, eps) * (p - x)
    dyc_right = 2.0 * m / tf.maximum((1.0 - p) ** 2, eps) * (p - x)
    dyc = tf.where(left, dyc_left, dyc_right)

    theta = tf.atan(dyc)

    xu = x - yt * tf.sin(theta)
    yu = yc + yt * tf.cos(theta)
    xl = x + yt * tf.sin(theta)
    yl = yc - yt * tf.cos(theta)

    return xu, yu, xl, yl


# ============================================================
# Kulfan CST (Class-Shape Transformation) geometry
# ============================================================
def _bernstein_basis_tf(x_row, n_bern: int):
    """
    Bernstein polynomial basis of degree (n_bern - 1), evaluated at x_row.

    Parameters
    ----------
    x_row : tf.Tensor of shape (P,)  — x positions on [0, 1]
    n_bern : number of basis polynomials (degree = n_bern - 1)

    Returns
    -------
    B : tf.Tensor of shape (P, n_bern)
        B[p, i] = C(n-1, i) * x[p]^i * (1 - x[p])^(n-1-i)
    """
    n = n_bern - 1
    binom = tf.constant([comb(n, i) for i in range(n_bern)], dtype=DTYPE)  # (n_bern,)

    x_col = x_row[:, None]                                                  # (P, 1)
    i_row = tf.range(n_bern, dtype=DTYPE)[None, :]                          # (1, n_bern)

    # Clip for numeric safety — endpoints handled by class function below.
    x_safe = tf.clip_by_value(x_col, 1e-12, 1.0 - 1e-12)
    one_m  = 1.0 - x_safe

    return binom[None, :] * (x_safe ** i_row) * (one_m ** (n - i_row))


def cst_geometry_batch_tf(
    coeffs,
    n_surface_pts: int = 81,
    n_bern: int = CST_N_BERN_DEFAULT,
    N1: float = CST_N1_DEFAULT,
    N2: float = CST_N2_DEFAULT,
):
    """
    Compute Kulfan CST airfoil upper / lower surface coordinates.

        y_u(x) = +C(x) · sum_i a_u[i] · B_{i, n-1}(x)
        y_l(x) = -C(x) · sum_i a_l[i] · B_{i, n-1}(x)
        C(x)   = x^N1 · (1 - x)^N2

    Defaults (N1=0.5, N2=1.0) impose a sqrt-singular LE and a sharp,
    C^0-closed TE analytically; this is the same prior NeuralFoil uses
    and is consistent with the JPA trunk LE singularity in Op1.

    Parameters
    ----------
    coeffs : (B, 2 * n_bern)
        First n_bern entries → upper-surface Bernstein weights a_u.
        Next  n_bern entries → lower-surface Bernstein weights a_l.
    n_surface_pts : int     points per surface (cosine-clustered, same grid
                            as `naca4_geometry_batch_tf`)
    n_bern : int            Bernstein coefficients per surface
                            (default 8 → 16-dim shape)
    N1, N2 : float          class-function exponents

    Returns
    -------
    xu, yu : (B, n_surface_pts)  upper surface  (LE → TE)
    xl, yl : (B, n_surface_pts)  lower surface  (LE → TE)
    """
    a_u = coeffs[:, :n_bern]            # (B, n_bern)
    a_l = coeffs[:, n_bern:2 * n_bern]  # (B, n_bern)

    x_np = cosine_spacing(n_surface_pts)
    x = tf.constant(x_np, dtype=DTYPE)              # (P,)

    # Class function — pins LE & TE to y=0 (sharp TE).
    x_safe = tf.maximum(x, 0.0)
    C = (x_safe ** N1) * ((1.0 - x_safe) ** N2)     # (P,)

    B = _bernstein_basis_tf(x, n_bern)              # (P, n_bern)

    # Shape function S(x) = B · a, then y = C * S per surface.
    S_u = tf.einsum("pi,bi->bp", B, a_u)            # (B, P)
    S_l = tf.einsum("pi,bi->bp", B, a_l)            # (B, P)

    yu = C[None, :] * S_u
    yl = -C[None, :] * S_l

    # --- Kulfan-18 extension (backward-compatible) ---------------------------
    # If coeffs carries 2 extra DOF beyond 2*n_bern, interpret them as the
    # NeuralFoil / aerosandbox Kulfan leading-edge weight (LEW) + trailing-edge
    # thickness (TE), matching get_kulfan_coordinates EXACTLY:
    #     y_u += LEW·x·(1-x)^(n_bern+0.5) + x·TE/2
    #     y_l += LEW·x·(1-x)^(n_bern+0.5) - x·TE/2
    # Coeffs of width 2*n_bern (=16) skip this block (identical to CST16), so
    # existing OP1/OP2 weights + prep_uiuc_cst are unaffected.
    _total = coeffs.shape[-1]
    if _total is not None and int(_total) >= 2 * n_bern + 2:
        lew = coeffs[:, 2 * n_bern:2 * n_bern + 1]              # (B, 1)
        te = coeffs[:, 2 * n_bern + 1:2 * n_bern + 2]           # (B, 1)
        lem = (x * (1.0 - x) ** (n_bern + 0.5))[None, :]        # (1, P) LE camber mode
        te_term = (0.5 * x)[None, :]                            # (1, P) TE half-thickness
        yu = yu + lew * lem + te * te_term
        yl = yl + lew * lem - te * te_term
    # ------------------------------------------------------------------------

    # Upper and lower share x grid (sharp TE, both surfaces span [0, 1]).
    B_batch = tf.shape(coeffs)[0]
    x_tile = tf.tile(x[None, :], [B_batch, 1])      # (B, P)

    return x_tile, yu, x_tile, yl


# ============================================================
# NACA 4-digit  →  Kulfan CST  least-squares converter
# ============================================================
def _naca4_y_thin(m, p, t, x):
    """
    NACA 4-digit y_upper / y_lower on the raw x grid (no surface deflection).
        y_u(x) = yc(x) + yt(x)
        y_l(x) = yc(x) - yt(x)
    The deflected form (yu = yc + yt·cos θ etc.) differs at O(t) near LE and
    is intentionally dropped here so the result lives on the CST x-grid.

    Inputs are numpy scalars/arrays; output: (yu, yl) at x.
    """
    yt = 5.0 * t * (
        0.2969 * np.sqrt(np.maximum(x, 1e-12))
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1015 * x ** 4
    )
    eps = 1e-8
    left = x < max(p, eps)
    yc = np.where(
        left,
        m / max(p ** 2, eps) * (2.0 * p * x - x ** 2),
        m / max((1.0 - p) ** 2, eps) * ((1.0 - 2.0 * p) + 2.0 * p * x - x ** 2),
    )
    return yc + yt, yc - yt


def naca4_to_cst_coeffs(
    shape_params,
    n_bern: int = CST_N_BERN_DEFAULT,
    n_fit_pts: int = 200,
    N1: float = CST_N1_DEFAULT,
    N2: float = CST_N2_DEFAULT,
):
    """
    Least-squares fit each NACA 4-digit airfoil to Kulfan CST coefficients.

        y_surface(x_i)  ≈  C(x_i) · sum_j a_j · B_j(x_i)

    Solved per-surface (upper / lower) via a single shared design matrix.

    Parameters
    ----------
    shape_params : array_like (B, 3)  — NACA4 [m, p, t]
    n_bern       : Bernstein coefficients per surface (default 8)
    n_fit_pts    : sample points for LSQ fit (cosine-clustered)
    N1, N2       : class-function exponents — must match cst_geometry_batch_tf

    Returns
    -------
    coeffs : np.ndarray (B, 2 * n_bern)   — [a_u | a_l]  (numpy float32)

    Notes
    -----
    NACA4 uses the no-deflection (thin) y convention here; reconstruction
    error on the *deflected* surface is dominated by the O(t) LE offset
    that CST drops. Both representations remain valid airfoils.
    """
    shape_params = np.asarray(shape_params, dtype=np.float32)
    if shape_params.ndim != 2 or shape_params.shape[1] != 3:
        raise ValueError(f"shape_params must be (B, 3), got {shape_params.shape}")

    x = cosine_spacing(n_fit_pts).astype(np.float64)
    C = (np.maximum(x, 0.0) ** N1) * ((1.0 - x) ** N2)         # (P,)

    n = n_bern - 1
    binom = np.array([comb(n, i) for i in range(n_bern)], dtype=np.float64)

    x_safe = np.clip(x, 1e-12, 1.0 - 1e-12)
    i_row = np.arange(n_bern, dtype=np.float64)[None, :]
    B_basis = binom[None, :] * (x_safe[:, None] ** i_row) * (
        (1.0 - x_safe)[:, None] ** (n - i_row)
    )                                                          # (P, n_bern)

    # Design matrix A[p, j] = C(x_p) * B_j(x_p)  — same for upper / lower.
    A = C[:, None] * B_basis                                   # (P, n_bern)

    out = np.empty((shape_params.shape[0], 2 * n_bern), dtype=np.float32)
    for k, (m, p, t) in enumerate(shape_params):
        yu, yl = _naca4_y_thin(float(m), float(p), float(t), x)
        a_u, *_ = np.linalg.lstsq(A, yu, rcond=None)
        # y_lower = -C * S_l in cst_geometry_batch_tf, so fit (-yl) onto A.
        a_l, *_ = np.linalg.lstsq(A, -yl, rcond=None)
        out[k, :n_bern]              = a_u.astype(np.float32)
        out[k, n_bern:2 * n_bern]    = a_l.astype(np.float32)

    return out


# ----- TF-side NACA4 -> CST conversion (for Op2 -> Op1 bridge) ---------------
def _make_naca4_to_cst_pinv(
    n_bern: int, n_fit_pts: int, N1: float, N2: float,
):
    """Return (x_grid_np, pinv_np) where  a = pinv @ y_surface  recovers the
    Bernstein coefficients in least-squares sense.  Cached per parameter set."""
    x = cosine_spacing(n_fit_pts).astype(np.float64)
    C = (np.maximum(x, 0.0) ** N1) * ((1.0 - x) ** N2)            # (P,)
    n = n_bern - 1
    binom = np.array([comb(n, i) for i in range(n_bern)], dtype=np.float64)
    x_safe = np.clip(x, 1e-12, 1.0 - 1e-12)
    i_row = np.arange(n_bern, dtype=np.float64)[None, :]
    B_basis = binom[None, :] * (x_safe[:, None] ** i_row) * (
        (1.0 - x_safe)[:, None] ** (n - i_row)
    )                                                              # (P, n_bern)
    A = C[:, None] * B_basis                                       # (P, n_bern)
    pinv = np.linalg.pinv(A).astype(np.float32)                    # (n_bern, P)
    return x.astype(np.float32), pinv


_NACA4_TO_CST_PINV_CACHE: dict = {}


def naca4_to_cst_coeffs_tf(
    shape_params,
    n_bern: int = CST_N_BERN_DEFAULT,
    n_fit_pts: int = 200,
    N1: float = CST_N1_DEFAULT,
    N2: float = CST_N2_DEFAULT,
):
    """
    TF-graph-friendly NACA4 -> CST converter.

    Same projection as `naca4_to_cst_coeffs` but uses a precomputed
    pseudo-inverse of the design matrix, so the per-batch cost reduces to
    one matmul per surface.  Differentiable wrt shape_params.

    Parameters
    ----------
    shape_params : tf.Tensor (B, 3)  -- NACA4 [m, p, t]

    Returns
    -------
    coeffs : tf.Tensor (B, 2 * n_bern)
    """
    key = (n_bern, n_fit_pts, float(N1), float(N2))
    if key not in _NACA4_TO_CST_PINV_CACHE:
        _NACA4_TO_CST_PINV_CACHE[key] = _make_naca4_to_cst_pinv(*key)
    x_np, pinv_np = _NACA4_TO_CST_PINV_CACHE[key]
    x = tf.constant(x_np, dtype=DTYPE)                             # (P,)
    pinv = tf.constant(pinv_np, dtype=DTYPE)                       # (n_bern, P)

    m = shape_params[:, 0:1]                                       # (B, 1)
    p = shape_params[:, 1:2]
    t = shape_params[:, 2:3]

    # No-deflection (thin) y_upper / y_lower on the same cosine grid.
    x_b = x[None, :]                                               # (1, P)
    yt = 5.0 * t * (
        0.2969 * tf.sqrt(tf.maximum(x_b, 1e-12))
        - 0.1260 * x_b
        - 0.3516 * x_b ** 2
        + 0.2843 * x_b ** 3
        - 0.1015 * x_b ** 4
    )                                                              # (B, P)
    eps = tf.constant(1e-8, dtype=DTYPE)
    left = x_b < tf.maximum(p, eps)
    yc_left = m / tf.maximum(p ** 2, eps) * (2.0 * p * x_b - x_b ** 2)
    yc_right = m / tf.maximum((1.0 - p) ** 2, eps) * (
        (1.0 - 2.0 * p) + 2.0 * p * x_b - x_b ** 2
    )
    yc = tf.where(left, yc_left, yc_right)                         # (B, P)

    yu = yc + yt
    yl = yc - yt
    # cst_geometry_batch_tf uses y_lower = -C * S_l, so fit (-yl) onto basis.
    a_u = tf.einsum("ip,bp->bi", pinv, yu)                         # (B, n_bern)
    a_l = tf.einsum("ip,bp->bi", pinv, -yl)
    return tf.concat([a_u, a_l], axis=1)                           # (B, 2*n_bern)


# ============================================================
# Closed contour: upper TE -> LE -> lower TE  (CCW, outward normal = right-hand)
# ============================================================
def _stitch_contour_tf(xu, yu, xl, yl, n_surface_pts: int) -> dict:
    """
    Stitch upper/lower-surface coordinate arrays into a CCW closed-loop
    airfoil contour with right-hand outward normals.

    Shared by `build_single_loop_contour_tf` (NACA4) and
    `build_single_loop_contour_cst_tf` (Kulfan CST) -- both produce
    (xu, yu, xl, yl) on the same cosine grid so the stitching is identical.
    """
    # Reverse upper (LE→TE) → (TE→LE) so contour starts at upper TE
    x_upper_rev = tf.reverse(xu, axis=[1])
    y_upper_rev = tf.reverse(yu, axis=[1])

    # Lower already LE→TE
    x_lower_fwd = xl
    y_lower_fwd = yl

    # Stitch — drop duplicate LE point ([0])
    x_nodes = tf.concat([x_upper_rev, x_lower_fwd[:, 1:]], axis=1)
    y_nodes = tf.concat([y_upper_rev, y_lower_fwd[:, 1:]], axis=1)

    x1 = x_nodes[:, :-1]
    y1 = y_nodes[:, :-1]
    x2 = x_nodes[:, 1:]
    y2 = y_nodes[:, 1:]

    dx = x2 - x1
    dy = y2 - y1
    ds = tf.sqrt(dx * dx + dy * dy + 1e-12)

    tx = dx / ds
    ty = dy / ds

    Nside = n_surface_pts - 1

    # Outward normal of CCW contour (right-hand rotation of tangent)
    nx = ty
    ny = -tx

    xc = 0.5 * (x1 + x2)
    yc = 0.5 * (y1 + y2)

    B = tf.shape(xu)[0]
    Np = 2 * Nside

    s_nodes_np = np.linspace(0.0, 1.0, Np + 1, dtype=np.float32)
    s_panels_np = 0.5 * (s_nodes_np[:-1] + s_nodes_np[1:])

    s_nodes = tf.tile(tf.constant(s_nodes_np[None, :], dtype=DTYPE), [B, 1])
    s_panels = tf.tile(tf.constant(s_panels_np[None, :], dtype=DTYPE), [B, 1])

    return {
        "x_nodes": x_nodes,
        "y_nodes": y_nodes,
        "x1": x1, "y1": y1,
        "x2": x2, "y2": y2,
        "xc": xc, "yc": yc,
        "tx": tx, "ty": ty,
        "nx": nx, "ny": ny,
        "ds": ds,
        "s_nodes": s_nodes,
        "s_panels": s_panels,
        "Nside": Nside,
        "Np": Np,
    }


def build_single_loop_contour_tf(shape_params, n_surface_pts: int = 81) -> dict:
    """
    Build a single closed-loop airfoil contour from NACA 4-digit shape
    parameters (m, p, t).

    See `_stitch_contour_tf` for the returned dict structure.

    Used by Op2 (which still consumes the legacy 3-dim shape vector).
    Op1 has migrated to `build_single_loop_contour_cst_tf`.
    """
    xu, yu, xl, yl = naca4_geometry_batch_tf(shape_params, n_surface_pts=n_surface_pts)
    return _stitch_contour_tf(xu, yu, xl, yl, n_surface_pts)


def build_single_loop_contour_cst_tf(
    coeffs,
    n_surface_pts: int = 81,
    n_bern: int = CST_N_BERN_DEFAULT,
    N1: float = CST_N1_DEFAULT,
    N2: float = CST_N2_DEFAULT,
) -> dict:
    """
    Build a single closed-loop airfoil contour from Kulfan CST coefficients
    (B, 2 * n_bern).  Phase A.4 Op1 entry point.

    See `_stitch_contour_tf` for the returned dict structure.
    """
    xu, yu, xl, yl = cst_geometry_batch_tf(
        coeffs, n_surface_pts=n_surface_pts,
        n_bern=n_bern, N1=N1, N2=N2)
    return _stitch_contour_tf(xu, yu, xl, yl, n_surface_pts)


# ============================================================
# Wake panels in freestream direction (XFOIL-style wake panel array)
# ============================================================
def build_wake_geometry_tf(base_geom: dict, alpha_rad, n_wake: int = 4) -> dict:
    """
    Build wake panels extending from TE in freestream direction.

    Geometric spacing: panel lengths grow by factor 1.5.

    Parameters
    ----------
    base_geom : dict from build_single_loop_contour_tf
    alpha_rad : (B, 1) AoA in radians
    n_wake    : int    number of wake panels

    Returns
    -------
    dict with x1,y1,x2,y2,xc,yc,tx,ty,ds,Lp  (B, n_wake)
    """
    x_te = 0.5 * (base_geom["x1"][:, :1] + base_geom["x2"][:, -1:])
    y_te = 0.5 * (base_geom["y1"][:, :1] + base_geom["y2"][:, -1:])

    dx_fs = tf.cos(alpha_rad)
    dy_fs = tf.sin(alpha_rad)

    lengths = [0.3 * (1.5 ** k) for k in range(n_wake)]
    cum = [0.0]
    for L in lengths:
        cum.append(cum[-1] + L)
    cum_tf = tf.constant(cum, dtype=DTYPE)

    x_nodes = x_te + cum_tf[None, :] * dx_fs
    y_nodes = y_te + cum_tf[None, :] * dy_fs

    x1_w = x_nodes[:, :-1]
    y1_w = y_nodes[:, :-1]
    x2_w = x_nodes[:, 1:]
    y2_w = y_nodes[:, 1:]
    xc_w = 0.5 * (x1_w + x2_w)
    yc_w = 0.5 * (y1_w + y2_w)
    dxp = x2_w - x1_w
    dyp = y2_w - y1_w
    ds_w = tf.sqrt(dxp ** 2 + dyp ** 2 + 1e-12)
    tx_w = dxp / ds_w
    ty_w = dyp / ds_w

    return {
        "x1": x1_w, "y1": y1_w,
        "x2": x2_w, "y2": y2_w,
        "xc": xc_w, "yc": yc_w,
        "ds": ds_w, "tx": tx_w, "ty": ty_w,
        "Lp": ds_w,
    }
