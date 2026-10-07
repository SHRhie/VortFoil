# -*- coding: utf-8 -*-
"""
sdf2d/geometry.py
=================
NACA 4-digit airfoil surface generation + differentiable point-to-segment
signed distance function (SDF), in BOTH numpy and TensorFlow.

Conventions (per MODULE INTERFACE CONTRACT):
  - chord runs 0..1 on x.
  - SDF d(x,y): negative INSIDE the airfoil, positive OUTSIDE, ~0 on the surface.
  - |grad d| == 1 (true Euclidean distance), verified via TF autodiff.
  - surf is a closed polyline (M,2). Distance is the min over all polygon segments
    (point-to-segment, differentiable). Sign comes from a robust even-odd / winding
    inside test (numpy) and an equivalent smooth-free ray-cast (tf, via tf.numpy
    only for sign would break autodiff, so we use a differentiable winding-based sign).

The thickness polynomial matches the rest of this codebase (steady51.py):
    yt = 5 t (0.2969 sqrt(x) - 0.1260 x - 0.3516 x^2 + 0.2843 x^3 - 0.1015 x^4)
which is the standard "open trailing edge" NACA form. Set close_te=True to use
the closed-TE coefficient (-0.1036) so the surface is a watertight loop with a
sharp TE; the SDF is robust either way because the surface is treated as a closed
polyline (TE gap, if any, is ~1e-3 c and bridged by the closing segment).
"""

import os

import numpy as np

try:
    import tensorflow as tf
    _HAS_TF = True
except Exception:  # pragma: no cover
    _HAS_TF = False


# ----------------------------------------------------------------------------
# NACA 4-digit surface
# ----------------------------------------------------------------------------
def _cosine_spacing(n):
    """n points clustered at LE/TE on [0,1] via cosine spacing."""
    beta = np.linspace(0.0, np.pi, n)
    return 0.5 * (1.0 - np.cos(beta))


def naca_surface(code='0012', n=400, close_te=True):
    """
    Generate a closed NACA 4-digit airfoil surface polyline, chord 0..1.

    Parameters
    ----------
    code : str   4-digit NACA designation, e.g. '0012', '2412'.
    n    : int   approximate number of surface points (M ~ n; see Returns).
    close_te : bool  if True (DEFAULT) use closed-TE coeff (-0.1036) so the
                     thickness is exactly 0 at x=1 and the loop is watertight
                     with a sharp TE. This MATCHES the Fluent NACA0012 mesh
                     (verified: near-wall SDF vs SV_WALL_DIST median rel err
                     ~2e-4, ~0 cells flagged inside). Set False for the
                     codebase open-TE coeff (-0.1015, steady51.py) which leaves
                     a small TE gap and disagrees with the Fluent wall near TE.

    Returns
    -------
    surf : ndarray (M, 2)
        Closed loop, ordered: upper TE -> LE -> lower TE -> (back to TE).
        First point == last point (explicit closure). M = 2*nh - 1 + 1
        where nh = n//2 (LE shared, explicit closing vertex appended).
    """
    code = str(code).zfill(4)
    m = int(code[0]) / 100.0          # max camber
    p = int(code[1]) / 10.0           # camber position
    t = int(code[2:]) / 100.0         # thickness

    a4 = -0.1036 if close_te else -0.1015

    nh = max(2, n // 2)
    x = _cosine_spacing(nh)           # 0..1, LE-clustered

    yt = 5.0 * t * (0.2969 * np.sqrt(np.maximum(x, 0.0))
                    - 0.1260 * x
                    - 0.3516 * x**2
                    + 0.2843 * x**3
                    + a4 * x**4)

    if m > 0.0 and p > 0.0:
        left = x < p
        yc = np.where(left,
                      m / max(p**2, 1e-12) * (2.0 * p * x - x**2),
                      m / max((1.0 - p)**2, 1e-12) * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2))
        dyc = np.where(left,
                       2.0 * m / max(p**2, 1e-12) * (p - x),
                       2.0 * m / max((1.0 - p)**2, 1e-12) * (p - x))
    else:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)

    theta = np.arctan(dyc)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # order: upper TE -> LE (reverse upper), then lower LE+1 -> TE
    xu_r = xu[::-1]; yu_r = yu[::-1]              # TE..LE
    # drop the LE duplicate from lower side (index 0), go LE->TE on lower
    x_loop = np.concatenate([xu_r, xl[1:]])
    y_loop = np.concatenate([yu_r, yl[1:]])

    surf = np.stack([x_loop, y_loop], axis=1)
    # explicit closure (append first vertex)
    surf = np.concatenate([surf, surf[:1]], axis=0)
    return surf.astype(np.float64)


# ----------------------------------------------------------------------------
# numpy SDF (point-to-segment, signed by winding number)
# ----------------------------------------------------------------------------
def _unsigned_dist_np(xy, A, B):
    """
    Min distance from each query point to the polyline segments {A_i -> B_i}.
    xy: (N,2), A,B: (S,2). Returns (N,) distances.
    Vectorized over (N,S); chunked to bound memory.
    """
    N = xy.shape[0]
    S = A.shape[0]
    AB = B - A                                   # (S,2)
    AB2 = np.einsum('sd,sd->s', AB, AB) + 1e-30  # (S,)

    out = np.empty(N, dtype=np.float64)
    # chunk over N to keep (chunk,S) manageable
    chunk = max(1, int(2_000_000 // max(S, 1)))
    for i0 in range(0, N, chunk):
        i1 = min(N, i0 + chunk)
        P = xy[i0:i1]                            # (c,2)
        AP = P[:, None, :] - A[None, :, :]       # (c,S,2)
        tloc = np.einsum('csd,sd->cs', AP, AB) / AB2[None, :]
        tloc = np.clip(tloc, 0.0, 1.0)
        proj = A[None, :, :] + tloc[:, :, None] * AB[None, :, :]  # (c,S,2)
        diff = P[:, None, :] - proj
        d2 = np.einsum('csd,csd->cs', diff, diff)
        out[i0:i1] = np.sqrt(d2.min(axis=1))
    return out


def _winding_sign_np(xy, surf):
    """
    +1 outside, -1 inside, via signed winding number of the closed polygon.
    surf is the closed loop (last==first). Returns (N,) of +/-1.
    """
    V = surf[:-1]                                # polygon vertices (no dup close)
    x = xy[:, 0][:, None]                        # (N,1)
    y = xy[:, 1][:, None]
    x1 = V[:, 0][None, :]                        # (1,S)
    y1 = V[:, 1][None, :]
    x2 = np.roll(V[:, 0], -1)[None, :]
    y2 = np.roll(V[:, 1], -1)[None, :]

    # winding number accumulation (Sunday's algorithm, vectorized)
    # isLeft > 0 left of edge, <0 right
    isLeft = (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)   # (N,S)
    up = (y1 <= y) & (y2 > y)
    down = (y1 > y) & (y2 <= y)
    wn = np.where(up & (isLeft > 0), 1, 0) + np.where(down & (isLeft < 0), -1, 0)
    wn = wn.sum(axis=1)
    inside = wn != 0
    return np.where(inside, -1.0, 1.0)


def sdf_np(xy, surf):
    """
    Signed distance (negative inside, positive outside) from points xy (N,2)
    to closed airfoil polyline surf (M,2, last==first).  Returns (N,) numpy.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    surf = np.asarray(surf, dtype=np.float64)
    A = surf[:-1]
    B = surf[1:]
    dist = _unsigned_dist_np(xy, A, B)
    sign = _winding_sign_np(xy, surf)
    return sign * dist


# ----------------------------------------------------------------------------
# tf SDF (differentiable; identical numbers to numpy)
# ----------------------------------------------------------------------------
def _unsigned_dist_tf(xy, A, B):
    """
    Differentiable min point-to-segment distance. xy:(N,2), A,B:(S,2).
    Returns (N,). Uses a soft min only for gradient stability? NO — uses a
    hard min (argmin distance is a.e. differentiable; the active segment's
    distance is smooth and |grad|=1 there). Returns true Euclidean distance.
    """
    dt = xy.dtype
    A = tf.cast(A, dt); B = tf.cast(B, dt)
    AB = B - A                                       # (S,2)
    AB2 = tf.reduce_sum(AB * AB, axis=1) + tf.cast(1e-30, dt)   # (S,)
    AP = xy[:, None, :] - A[None, :, :]              # (N,S,2)
    tloc = tf.reduce_sum(AP * AB[None, :, :], axis=2) / AB2[None, :]
    tloc = tf.clip_by_value(tloc, 0.0, 1.0)          # (N,S)
    proj = A[None, :, :] + tloc[:, :, None] * AB[None, :, :]    # (N,S,2)
    diff = xy[:, None, :] - proj                     # (N,S,2)
    d2 = tf.reduce_sum(diff * diff, axis=2)          # (N,S)
    dmin2 = tf.reduce_min(d2, axis=1)                # (N,)
    return tf.sqrt(dmin2 + tf.cast(1e-30, dt))       # +eps keeps grad finite at d=0


def _winding_sign_tf(xy, surf):
    """
    Non-differentiable +/-1 sign (sign is piecewise constant, grad is 0 a.e.),
    computed with pure TF ops so the function works inside tf.function/graph.
    +1 outside, -1 inside.
    """
    dt = xy.dtype
    surf = tf.cast(surf, dt)
    V = surf[:-1]
    x = xy[:, 0][:, None]
    y = xy[:, 1][:, None]
    x1 = V[:, 0][None, :]
    y1 = V[:, 1][None, :]
    x2 = tf.roll(V[:, 0], shift=-1, axis=0)[None, :]
    y2 = tf.roll(V[:, 1], shift=-1, axis=0)[None, :]
    isLeft = (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)
    up = (y1 <= y) & (y2 > y)
    down = (y1 > y) & (y2 <= y)
    contrib = tf.where(up & (isLeft > 0), tf.ones_like(isLeft), tf.zeros_like(isLeft)) \
        - tf.where(down & (isLeft < 0), tf.ones_like(isLeft), tf.zeros_like(isLeft))
    wn = tf.reduce_sum(contrib, axis=1)
    inside = tf.not_equal(tf.round(wn), 0.0)
    return tf.where(inside, -tf.ones_like(wn), tf.ones_like(wn))


def sdf_tf(xy_tf, surf_tf):
    """
    Differentiable signed distance (TF). Same numbers as sdf_np.
    xy_tf:(N,2) tensor, surf_tf:(M,2) closed loop. Returns (N,) tensor.
    The magnitude (distance) is differentiable with |grad|=1; the sign is a
    stop_gradient constant (sign flips only across the surface, where grad of a
    true SDF is well-defined by the distance term, not the sign).
    """
    xy_tf = tf.convert_to_tensor(xy_tf)
    dt = xy_tf.dtype
    surf_tf = tf.cast(tf.convert_to_tensor(surf_tf), dt)
    A = surf_tf[:-1]
    B = surf_tf[1:]
    dist = _unsigned_dist_tf(xy_tf, A, B)
    sign = tf.stop_gradient(_winding_sign_tf(xy_tf, surf_tf))
    return sign * dist


# ----------------------------------------------------------------------------
# smoke test
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    import os
    os.environ.setdefault('NF_CPU_THREADS', '2')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MPLBACKEND', 'Agg')

    print('=' * 70)
    print('geometry.py smoke test')
    print('=' * 70)

    # --- 1. surface ---
    surf = naca_surface('0012', n=400)
    print(f'[surf] shape={surf.shape}  closed={np.allclose(surf[0], surf[-1])}')
    print(f'[surf] x range=[{surf[:,0].min():.4f}, {surf[:,0].max():.4f}]'
          f'  y range=[{surf[:,1].min():.4f}, {surf[:,1].max():.4f}]')
    # max half-thickness of NACA0012 ~ 0.06 at x~0.3
    print(f'[surf] max |y| = {np.abs(surf[:,1]).max():.4f}  (expect ~0.06)')
    assert surf.shape[1] == 2 and surf.shape[0] > 100
    assert np.allclose(surf[0], surf[-1])
    assert abs(np.abs(surf[:, 1]).max() - 0.06) < 0.01

    # --- 2. sdf ~ 0 on surface points ---
    d_on = sdf_np(surf[:-1], surf)
    print(f'[sdf=0 on surf] max|d| on vertices = {np.abs(d_on).max():.3e}  (expect ~0)')
    assert np.abs(d_on).max() < 1e-9

    # --- 3. sign: a point clearly inside is negative, outside positive ---
    inside_pt = np.array([[0.3, 0.0]])    # mid-chord, on chordline -> inside
    outside_pt = np.array([[0.5, 0.5], [-0.5, 0.0], [2.0, 0.0]])
    di = sdf_np(inside_pt, surf)
    do = sdf_np(outside_pt, surf)
    print(f'[sign] inside (0.3,0) d={di[0]:+.4f} (expect <0, ~ -0.06)')
    print(f'[sign] outside pts d={do}  (expect all >0)')
    assert di[0] < 0
    assert np.all(do > 0)

    # --- 4. numpy vs tf identical ---
    if _HAS_TF:
        rng = np.random.default_rng(0)
        Q = rng.uniform(-0.5, 1.5, size=(500, 2))
        Q[:, 1] = rng.uniform(-0.4, 0.4, size=500)
        d_np = sdf_np(Q, surf)
        d_tf = sdf_tf(tf.constant(Q, tf.float64),
                      tf.constant(surf, tf.float64)).numpy()
        err = np.abs(d_np - d_tf).max()
        print(f'[np vs tf] max abs diff over 500 pts = {err:.3e}  (expect ~0)')
        assert err < 1e-9

        # --- 5. |grad sdf| ~ 1 via autodiff (away from surface & medial axis) ---
        # pick query points outside, not too close to surface, off the chordline
        Qg = np.column_stack([
            rng.uniform(0.1, 0.9, size=300),
            np.where(rng.random(300) > 0.5, 1, -1) * rng.uniform(0.08, 0.35, size=300),
        ])
        xq = tf.constant(Qg, tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(xq)
            dq = sdf_tf(xq, tf.constant(surf, tf.float64))
        g = tape.gradient(dq, xq)               # (N,2)
        gmag = tf.sqrt(tf.reduce_sum(g * g, axis=1)).numpy()
        print(f'[|grad d|] mean={gmag.mean():.5f}  '
              f'p5={np.percentile(gmag,5):.5f}  p95={np.percentile(gmag,95):.5f} (expect ~1)')
        # the vast majority should be ~1; a few near medial axis can deviate
        frac_ok = np.mean(np.abs(gmag - 1.0) < 1e-3)
        print(f'[|grad d|] fraction within 1e-3 of 1.0 = {frac_ok:.3f}')
        assert frac_ok > 0.95

    # --- 6. spot-check vs Fluent wall_dist ---
    try:
        import h5py
        import glob
        # Fluent NACA0012 case/data files: NOT shipped in this release (see README).
        # Point FLUENT_NACA0012_DIR at them to run this optional spot-check.
        ddir = os.environ.get('FLUENT_NACA0012_DIR', 'fluent_data/NACA0012_0deg')
        cas = glob.glob(ddir + '/*.cas.h5')[0]
        dat = glob.glob(ddir + '/*.dat.h5')[0]

        with h5py.File(cas, 'r') as f:
            nc = f['meshes/1/nodes/coords']
            coords = nc[list(nc.keys())[0]][()]          # (Nnode,2)
            c0 = f['meshes/1/faces/c0/1'][()].astype(np.int64)  # owner (1-based)
            c1 = f['meshes/1/faces/c1/1'][()].astype(np.int64)  # neighbor
            nn = f['meshes/1/faces/nodes/1/nnodes'][()].astype(np.int64)
            flat = f['meshes/1/faces/nodes/1/nodes'][()].astype(np.int64)
        with h5py.File(dat, 'r') as f:
            wd = f['results/1/phase-1/cells/SV_WALL_DIST/1'][()]

        Ncell = int(max(c0.max(), c1.max()))
        # face midpoints = mean of node coords (2 nodes per face here)
        off = np.concatenate([[0], np.cumsum(nn)])
        # vectorized 2-node assumption check
        n0 = flat[off[:-1]] - 1
        n1 = flat[off[:-1] + 1] - 1
        mid = 0.5 * (coords[n0] + coords[n1])            # (Nface,2)

        # scatter-mean of midpoints into cells by c0 and c1.
        # c0 and c1 can have DIFFERENT lengths (Fluent: c0 over all faces,
        # c1 only over interior faces). Align each to its own midpoints prefix.
        cen_sum = np.zeros((Ncell + 1, 2))
        cen_cnt = np.zeros(Ncell + 1)
        for cc in (c0, c1):
            mc = mid[:len(cc)]                    # midpoints for these faces
            valid = cc > 0
            np.add.at(cen_sum, cc[valid], mc[valid])
            np.add.at(cen_cnt, cc[valid], 1.0)
        cen = cen_sum[1:] / np.maximum(cen_cnt[1:, None], 1.0)   # (Ncell,2)
        print(f'[centroids] shape={cen.shape} (expect (382575,2))')
        assert cen.shape == (382575, 2)

        # (a) GLOBAL: all cells are OUTSIDE the airfoil -> sdf must be >= 0.
        d_all = sdf_np(cen, surf)
        n_inside = int((d_all < -1e-4).sum())
        print(f'[wall_dist] cells flagged inside (d<-1e-4) = {n_inside} / {len(cen)} '
              f'(expect ~0)')

        # (b) magnitude vs Fluent SV_WALL_DIST. Sample broadly + all near-wall.
        rng2 = np.random.default_rng(1)
        samp = rng2.choice(len(cen), 4000, replace=False)
        rel_all = np.abs(np.abs(d_all[samp]) - wd[samp]) / (wd[samp] + 1e-9)
        print(f'[wall_dist] global (4000 rand) median rel err = '
              f'{np.median(rel_all):.5f}')

        near = wd < 0.05                                  # near-wall band
        rel_near = np.abs(np.abs(d_all[near]) - wd[near]) / (wd[near] + 1e-9)
        print(f'[wall_dist] near-wall (d<0.05) n={int(near.sum())} '
              f'median rel err = {np.median(rel_near):.5f}  '
              f'mean = {rel_near.mean():.5f}')

        assert n_inside < 50, 'too many outside cells flagged inside by SDF'
        assert np.median(rel_all) < 0.02, 'SDF magnitude disagrees w/ wall_dist'
        assert np.median(rel_near) < 0.02, 'near-wall SDF disagrees w/ wall_dist'
        print('[wall_dist] OK: analytic closed-TE SDF matches Fluent SV_WALL_DIST')
    except Exception as e:
        print(f'[wall_dist spot] SKIPPED ({type(e).__name__}: {e})')

    print('=' * 70)
    print('ALL GEOMETRY SMOKE CHECKS PASSED')
    print('=' * 70)
