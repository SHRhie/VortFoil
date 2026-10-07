# -*- coding: utf-8 -*-
"""
Random airfoil + alpha + Re samplers for staged training.

Two samplers are exposed:

  sample_random_shapes()      -- legacy 3-dim NACA 4-digit  (m, p, t)
                                 kept for backward compatibility with the
                                 stages that have not yet migrated to CST.

  sample_random_cst_shapes()  -- Phase A.3 generalization.  Anchors the
                                 distribution on NACA 4-digit (so eps=0
                                 reproduces the legacy training set
                                 exactly, in CST space), then perturbs
                                 each Bernstein coefficient by
                                 eps * sigma_k * N(0, 1).  Configurable
                                 reject-sampling guards against self-
                                 intersection.

Both samplers share the (alpha, Re) distribution so flow-condition
behavior matches across stages.
"""
from __future__ import annotations

import numpy as np

# Empirical per-Bernstein-coefficient std of the NACA 4-digit family after
# projection onto the Kulfan CST basis (N1=0.5, N2=1.0, n_bern=8).
# Measured from n=2000 NACA4 samples; used as the radial perturbation
# scale in sample_random_cst_shapes so that eps=1 corresponds to roughly
# "one NACA4 standard deviation per coefficient".
_CST_STD_UPPER = np.array(
    [0.0428, 0.0770, 0.0492, 0.0872, 0.0464, 0.0970, 0.0581, 0.0865],
    dtype=np.float32,
)
_CST_STD_LOWER = np.array(
    [0.0426, 0.0760, 0.0485, 0.0866, 0.0459, 0.0964, 0.0578, 0.0858],
    dtype=np.float32,
)
CST_SIGMA_DEFAULT = np.concatenate([_CST_STD_UPPER, _CST_STD_LOWER])  # (16,)


def sample_random_shapes(
    n_samples: int,
    m_range=(0.00, 0.06),
    p_range=(0.20, 0.60),
    t_range=(0.08, 0.18),
    symmetric_fraction: float = 0.30,
    low_camber_fraction: float = 0.10,
    alpha_range=(-6.0, 12.0),
    near_zero_alpha_fraction: float = 0.15,
    Re_range=(5e4, 2e6),
):
    """Returns (shapes, alphas, Res) numpy arrays of shape (n,3), (n,1), (n,1)."""
    shapes = []
    alphas = []
    Res = []

    for _ in range(n_samples):
        r = np.random.rand()
        if r < symmetric_fraction:
            m = 0.0
            p = 0.4
        elif r < symmetric_fraction + low_camber_fraction:
            m = np.random.uniform(0.0, 0.02)
            p = np.random.uniform(*p_range)
        else:
            m = np.random.uniform(*m_range)
            p = np.random.uniform(*p_range)

        t = np.random.uniform(*t_range)

        if np.random.rand() < near_zero_alpha_fraction:
            alpha = np.random.uniform(-2.0, 2.0)
        else:
            alpha = np.random.uniform(*alpha_range)

        Re = 10.0 ** np.random.uniform(np.log10(Re_range[0]), np.log10(Re_range[1]))

        shapes.append([m, p, t])
        alphas.append([alpha])
        Res.append([Re])

    return (
        np.asarray(shapes, dtype=np.float32),
        np.asarray(alphas, dtype=np.float32),
        np.asarray(Res, dtype=np.float32),
    )


# ============================================================
# Phase A.3 -- CST sampler (NACA4-anchored + radial perturbation)
# ============================================================
def _check_no_self_intersect(coeffs, n_check_pts: int = 81, tol: float = 1e-4):
    """
    Returns boolean mask (B,) -- True for shapes whose upper surface stays
    above the lower surface (with tolerance `tol` for numerical wiggle).

    Uses cst_geometry_batch_tf for the evaluation so the geometry produced
    here is exactly the same as the one used downstream by Op1.
    """
    import tensorflow as tf
    from staged.common.geometry import cst_geometry_batch_tf

    _, yu, _, yl = cst_geometry_batch_tf(
        tf.constant(coeffs), n_surface_pts=n_check_pts)
    gap = yu.numpy() - yl.numpy()      # (B, P)
    return gap.min(axis=1) > -tol


def sample_random_cst_shapes(
    n_samples: int,
    perturb_eps: float = 0.0,
    sigma=None,
    alpha_range=(-6.0, 12.0),
    near_zero_alpha_fraction: float = 0.15,
    Re_range=(5e4, 2e6),
    naca4_m_range=(0.00, 0.06),
    naca4_p_range=(0.20, 0.60),
    naca4_t_range=(0.08, 0.18),
    symmetric_fraction: float = 0.30,
    low_camber_fraction: float = 0.10,
    max_reject_passes: int = 4,
):
    """
    NACA4-anchored CST sampler (Phase A.3, design S2).

    Step 1: draw an anchor NACA 4-digit (m, p, t) from the legacy
            sample_random_shapes distribution.
    Step 2: project onto Kulfan CST (16 Bernstein coefficients).
    Step 3: add Gaussian perturbation  delta ~ N(0, (perturb_eps * sigma)^2)
            per coefficient, where sigma defaults to the empirical per-
            coefficient std of the NACA4-derived CST distribution.
    Step 4: reject self-intersecting shapes; resample anchors for the
            rejected slots up to `max_reject_passes` rounds.  Any shape
            still failing after the budget is returned with delta=0
            (i.e. the unperturbed NACA4 projection, which is guaranteed
            to be a valid airfoil).

    perturb_eps=0.0 reproduces NACA4-in-CST exactly (no exploration).
    perturb_eps=1.0 spreads each Bernstein coefficient by ~1 NACA4-std.

    Returns
    -------
    coeffs : (n, 16) float32   CST [a_u_0..7, a_l_0..7]
    alphas : (n, 1)  float32   AoA in degrees
    Res    : (n, 1)  float32   Reynolds number
    """
    from staged.common.geometry import naca4_to_cst_coeffs

    if sigma is None:
        sigma = CST_SIGMA_DEFAULT
    sigma = np.asarray(sigma, dtype=np.float32)
    if sigma.shape != (16,):
        raise ValueError(f"sigma must have shape (16,), got {sigma.shape}")

    # --- Anchor NACA4 shapes (same distribution as the legacy sampler) ---
    naca_shapes, alphas, Res = sample_random_shapes(
        n_samples,
        m_range=naca4_m_range,
        p_range=naca4_p_range,
        t_range=naca4_t_range,
        symmetric_fraction=symmetric_fraction,
        low_camber_fraction=low_camber_fraction,
        alpha_range=alpha_range,
        near_zero_alpha_fraction=near_zero_alpha_fraction,
        Re_range=Re_range,
    )

    anchor_coeffs = naca4_to_cst_coeffs(naca_shapes, n_bern=8)   # (n, 16)

    if perturb_eps <= 0.0:
        return anchor_coeffs.astype(np.float32), alphas, Res

    # --- Perturb + reject self-intersecting shapes ---
    sigma_eps = (sigma * perturb_eps)[None, :]                    # (1, 16)
    coeffs = anchor_coeffs + np.random.randn(*anchor_coeffs.shape).astype(
        np.float32) * sigma_eps

    valid = _check_no_self_intersect(coeffs)
    for _ in range(max_reject_passes):
        if valid.all():
            break
        n_bad = int((~valid).sum())
        # Re-perturb the anchor; do NOT re-roll the anchor itself so the
        # (alpha, Re) joint distribution remains the legacy distribution.
        coeffs[~valid] = anchor_coeffs[~valid] + np.random.randn(
            n_bad, 16).astype(np.float32) * sigma_eps
        valid = _check_no_self_intersect(coeffs)

    # Final fallback: any still-invalid slot reverts to the unperturbed
    # NACA4 projection (a guaranteed valid airfoil).
    coeffs[~valid] = anchor_coeffs[~valid]

    return coeffs.astype(np.float32), alphas, Res


# ============================================================
# UIUC-anchored mixed sampler (NF-scale dataset)
# ============================================================
_UIUC_CACHE = {"cst": None, "names": None}


def _load_uiuc_anchors(path: str = "staged/data/uiuc_cst.npz"):
    if _UIUC_CACHE["cst"] is None:
        from pathlib import Path
        z = np.load(Path(path))
        _UIUC_CACHE["cst"]   = z["cst"].astype(np.float32)
        _UIUC_CACHE["names"] = z["names"]
    return _UIUC_CACHE["cst"], _UIUC_CACHE["names"]


def sample_uiuc_mixed_shapes(
    n_samples: int,
    perturb_eps: float = 0.3,
    sigma=None,
    uiuc_frac: float = 0.5,
    naca4_frac: float = 0.4,    # uiuc_frac + naca4_frac + (random_cst_frac) = 1
    alpha_range=(-10.0, 18.0),
    near_zero_alpha_fraction: float = 0.10,
    Re_range=(3e4, 1e8),
    naca4_m_range=(0.00, 0.07),
    naca4_p_range=(0.20, 0.70),
    naca4_t_range=(0.06, 0.20),
    symmetric_fraction: float = 0.20,
    low_camber_fraction: float = 0.10,
    uiuc_path: str = "staged/data/uiuc_cst.npz",
    max_reject_passes: int = 4,
):
    """Mixed-anchor CST sampler for large-scale dataset generation.

    Three anchor sources combined into one sample stream:
      - UIUC (~1595 real airfoils, projected to CST(8))   -- weight uiuc_frac
      - NACA4 (procedural, broad m/p/t grid)              -- weight naca4_frac
      - Random small CST perturbations of the above       -- residual

    α and Re ranges are wider than the legacy sampler (NeuralFoil-style).

    Returns
    -------
    coeffs : (n, 16) float32
    alphas : (n, 1)  float32  in degrees
    Res    : (n, 1)  float32
    source : (n,)    uint8  (0=UIUC, 1=NACA4, 2=random-CST)
    """
    if sigma is None:
        sigma = CST_SIGMA_DEFAULT
    sigma = np.asarray(sigma, dtype=np.float32)

    # ---- Per-slot flow conditions (shared α/Re distribution) ----
    near_zero_n = int(round(n_samples * near_zero_alpha_fraction))
    near_zero_a = np.random.uniform(-2.0, 2.0, size=near_zero_n)
    wide_a      = np.random.uniform(alpha_range[0], alpha_range[1],
                                       size=n_samples - near_zero_n)
    alphas = np.concatenate([near_zero_a, wide_a]).astype(np.float32)
    np.random.shuffle(alphas)
    alphas = alphas[:, None]

    Res = (10.0 ** np.random.uniform(np.log10(Re_range[0]),
                                        np.log10(Re_range[1]),
                                        size=n_samples)).astype(np.float32)
    Res = Res[:, None]

    # ---- Anchor source assignment ----
    n_uiuc   = int(round(n_samples * uiuc_frac))
    n_naca4  = int(round(n_samples * naca4_frac))
    n_random = n_samples - n_uiuc - n_naca4
    source = np.concatenate([
        np.zeros(n_uiuc, dtype=np.uint8),
        np.ones(n_naca4, dtype=np.uint8),
        np.full(n_random, 2, dtype=np.uint8),
    ])
    np.random.shuffle(source)

    coeffs = np.zeros((n_samples, 16), dtype=np.float32)

    # UIUC anchors (sample with replacement from the 1595 real airfoils)
    if n_uiuc > 0:
        uiuc_cst, _ = _load_uiuc_anchors(uiuc_path)
        idx_uiuc = np.random.randint(0, uiuc_cst.shape[0], size=n_uiuc)
        anchors_uiuc = uiuc_cst[idx_uiuc]
        mask_u = (source == 0)
        coeffs[mask_u] = anchors_uiuc

    # NACA4 anchors
    if n_naca4 > 0:
        from staged.common.geometry import naca4_to_cst_coeffs
        naca_shapes, _, _ = sample_random_shapes(
            n_naca4,
            m_range=naca4_m_range, p_range=naca4_p_range,
            t_range=naca4_t_range,
            symmetric_fraction=symmetric_fraction,
            low_camber_fraction=low_camber_fraction,
            alpha_range=alpha_range,
            near_zero_alpha_fraction=0.0,
            Re_range=Re_range,
        )
        anchors_n = naca4_to_cst_coeffs(naca_shapes, n_bern=8)
        coeffs[source == 1] = anchors_n

    # Random-CST anchors: draw 50/50 from UIUC and NACA4 pools, then
    # perturb with larger eps for diversity
    if n_random > 0:
        mask_r = (source == 2)
        n_r = int(mask_r.sum())
        # Mix anchors for the random-perturb slots
        if n_uiuc > 0 or n_naca4 > 0:
            uiuc_cst, _ = _load_uiuc_anchors(uiuc_path)
            n_r_u = n_r // 2
            n_r_n = n_r - n_r_u
            idx_ru = np.random.randint(0, uiuc_cst.shape[0], size=n_r_u)
            anchors_ru = uiuc_cst[idx_ru]
            from staged.common.geometry import naca4_to_cst_coeffs
            naca_shapes_r, _, _ = sample_random_shapes(
                n_r_n,
                m_range=naca4_m_range, p_range=naca4_p_range,
                t_range=naca4_t_range,
                symmetric_fraction=symmetric_fraction,
                low_camber_fraction=low_camber_fraction,
                alpha_range=alpha_range,
                near_zero_alpha_fraction=0.0,
                Re_range=Re_range,
            )
            anchors_rn = naca4_to_cst_coeffs(naca_shapes_r, n_bern=8)
            anchors_r = np.concatenate([anchors_ru, anchors_rn], axis=0)
            np.random.shuffle(anchors_r)
            coeffs[mask_r] = anchors_r

    # ---- Perturb (skip if eps=0) ----
    if perturb_eps > 0.0:
        sigma_eps = (sigma * perturb_eps)[None, :]
        anchors_kept = coeffs.copy()
        # Anchors (UIUC/NACA4): small eps perturbation (or zero for purity)
        # Random source: full perturbation
        eps_per_slot = np.where(source == 2,
                                  perturb_eps,
                                  perturb_eps * 0.3).astype(np.float32)
        noise = np.random.randn(*coeffs.shape).astype(np.float32) * (
            (sigma[None, :]) * eps_per_slot[:, None])
        coeffs = coeffs + noise

        valid = _check_no_self_intersect(coeffs)
        for _ in range(max_reject_passes):
            if valid.all():
                break
            n_bad = int((~valid).sum())
            coeffs[~valid] = anchors_kept[~valid] + np.random.randn(
                n_bad, 16).astype(np.float32) * (
                    sigma[None, :] * eps_per_slot[~valid, None])
            valid = _check_no_self_intersect(coeffs)
        # Final fallback: revert to clean anchor
        coeffs[~valid] = anchors_kept[~valid]

    return coeffs.astype(np.float32), alphas, Res, source
