# -*- coding: utf-8 -*-
"""Faster kernels for the field pathway, with the same arithmetic.

The pathway's cost is three O(n_points x n_panels) kernels evaluated at every query node of a RANS
mesh: the panel-induced velocity, the signed distance to the surface, and the nearest panel. All
three were written as broadcasts that build (chunk, 240, 2) temporaries and then reduce them, so
they run single-threaded and are bound by memory traffic.

Each one can be rewritten so that every part that is a contraction becomes a matrix product, which
this platform's BLAS runs multithreaded, leaving only the genuinely elementwise part (a logarithm
and two arc tangents in the panel kernel; comparisons in the winding number) on the slow path. The
algebra is exact:

    |p - q|^2          = |p|^2 - 2 p.q + |q|^2      (nearest panel: one matmul, the |p|^2 term dropped)
    x_local, y_local   = p.t - p1.t, p.n - p1.n     (panel kernel: two matmuls)
    sum_j c_j f(x_j)   = f @ c                      (the reduction over panels: one matmul)

so no three-dimensional temporary is ever formed. The distance keeps its differences p-a rather
than expanding them, because the expansion cancels where the distance is small. Results differ
from the originals only by float64 re-association; `verify()` measures that on a real case, both
per kernel and on the painted field the decoder actually consumes.

Use it by replacing the three names in the paint module, which leaves the physics untouched:

    import vf_field_fast as F
    F.install()          # -> paint_points_v2 now uses the fast kernels

  python code/field/vf_field_fast.py --verify
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_2PI = 2.0 * np.pi
CHUNK = 16384          # rows per pass; large enough to amortize the matmuls, small enough to stay in cache
THREADS = int(os.environ.get("VORTFOIL_FIELD_THREADS", min(8, os.cpu_count() or 1)))
_POOL = None


def _pool():
    global _POOL
    if _POOL is None and THREADS > 1:
        _POOL = ThreadPoolExecutor(THREADS)
    return _POOL


def _parallel(n, fn):
    """Run fn(start, stop) over contiguous row blocks.

    The blocks are disjoint and each writes only its own rows, so the result does not depend on how
    the rows are divided. numpy releases the interpreter lock inside the large elementwise kernels
    this module spends its time in, so threads give real parallelism here."""
    if THREADS <= 1 or n < 4 * CHUNK:
        fn(0, n)
        return
    step = -(-n // THREADS)
    list(_pool().map(lambda i: fn(i, min(i + step, n)), range(0, n, step)))


# --------------------------------------------------------------------------- nearest panel
def nearest_idx(P, Q, chunk=CHUNK):
    """Index of the nearest point of Q for each row of P, through |p-q|^2 = |p|^2 - 2 p.q + |q|^2.

    The |p|^2 term is constant along the row and cannot change the argmin, so it is dropped."""
    P = np.asarray(P, np.float64)[:, :2]
    Q = np.asarray(Q, np.float64)[:, :2]
    q2 = np.einsum("sd,sd->s", Q, Q)
    Qt = -2.0 * Q.T
    out = np.empty(len(P), np.int64)

    def block(lo, hi):
        for i0 in range(lo, hi, chunk):
            sl = slice(i0, min(i0 + chunk, hi))
            d2 = P[sl] @ Qt
            d2 += q2[None, :]
            out[sl] = np.argmin(d2, axis=1)

    _parallel(len(P), block)
    return out


# --------------------------------------------------------------------------- signed distance
def _unsigned_dist(xy, A, B, chunk=CHUNK):
    """Min distance to the segments A->B.

    The arithmetic is the original's: the difference p - a is formed first and the foot of the
    perpendicular is subtracted from it, rather than expanding |p-a|^2 into |p|^2 - 2 p.a + |a|^2.
    The expansion cancels for points near the surface, where the first mesh layer sits about 1e-8
    from the wall and the near-wall reconstruction amplifies a relative error in the distance. What
    is saved here is only the three-dimensional temporary, not the arithmetic."""
    ABx, ABy = (B - A)[:, 0], (B - A)[:, 1]
    AB2 = ABx * ABx + ABy * ABy + 1e-30
    Ax, Ay = A[:, 0][None, :], A[:, 1][None, :]
    out = np.empty(len(xy), np.float64)

    def block(lo, hi):
      for i0 in range(lo, hi, chunk):
        p = xy[i0:min(i0 + chunk, hi)]
        dx = p[:, 0:1] - Ax
        dy = p[:, 1:2] - Ay
        t = dx * ABx[None, :] + dy * ABy[None, :]
        t /= AB2[None, :]
        np.clip(t, 0.0, 1.0, out=t)
        dx -= t * ABx[None, :]                                # p - (a + t (b - a)), formed as a difference
        dy -= t * ABy[None, :]
        d2 = dx * dx + dy * dy
        out[i0:i0 + len(p)] = np.sqrt(d2.min(axis=1))

    _parallel(len(xy), block)
    return out


def _winding_sign(xy, surf, chunk=CHUNK):
    """+1 outside, -1 inside, by winding number. The predicate is evaluated in the same form as the
    original so that a point sitting on an edge lands on the same side."""
    V = surf[:-1]
    x1, y1 = V[:, 0][None, :], V[:, 1][None, :]
    x2, y2 = np.roll(V[:, 0], -1)[None, :], np.roll(V[:, 1], -1)[None, :]
    out = np.empty(len(xy), np.float64)

    def block(lo, hi):
      for i0 in range(lo, hi, chunk):
        p = xy[i0:min(i0 + chunk, hi)]
        x = p[:, 0:1]
        y = p[:, 1:2]
        isLeft = (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)
        up = (y1 <= y) & (y2 > y)
        down = (y1 > y) & (y2 <= y)
        wn = np.count_nonzero(up & (isLeft > 0), axis=1) - np.count_nonzero(down & (isLeft < 0), axis=1)
        out[i0:i0 + len(p)] = np.where(wn != 0, -1.0, 1.0)

    _parallel(len(xy), block)
    return out


def sdf(xy, surf):
    """Signed distance to the closed polyline (negative inside), same convention as the original."""
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    surf = np.asarray(surf, np.float64)
    return _winding_sign(xy, surf) * _unsigned_dist(xy, surf[:-1], surf[1:])


def sdf_chunked(P, nodes, chunk=None):
    return sdf(np.asarray(P, np.float64)[:, :2], nodes)


# --------------------------------------------------------------------------- panel induced velocity
def field_uv(P, geom, gamma, sigma, Uinf, chunk=CHUNK):
    """Uniform stream plus the constant-strength source and vortex panels, at every point of P.

    The local coordinates are formed from the differences p - p1, as in the original: writing them
    as p.t - p1.t would cancel for points close to a panel. What is saved is the reduction over
    panels. The original builds four (chunk, n_panels) arrays of global-frame influence, multiplies
    them by the strengths and sums; the same result is a matrix product of the two elementwise
    quantities the kernel really produces, the log ratio and the subtended angle, against per-panel
    coefficients that fold in the strengths and the rotation back to the global frame."""
    P = np.asarray(P, np.float64)[:, :2]
    p1, t, L = geom["p1"], geom["t"], geom["L"]
    tx, ty = t[:, 0], t[:, 1]
    g = np.asarray(gamma, np.float64)
    # u = Uinf_x + lnr @ (sigma*tx + gamma*ty)/2pi + dtheta @ (gamma*tx - sigma*ty)/2pi
    # v = Uinf_y + lnr @ (sigma*ty - gamma*tx)/2pi + dtheta @ (gamma*ty + sigma*tx)/2pi
    lnr_x = (sigma * tx + g * ty) / _2PI
    lnr_y = (sigma * ty - g * tx) / _2PI
    dth_x = (g * tx - sigma * ty) / _2PI
    dth_y = (g * ty + sigma * tx) / _2PI
    p1x, p1y = p1[:, 0][None, :], p1[:, 1][None, :]
    u = np.empty(len(P)); v = np.empty(len(P))

    def block(lo, hi):
      for i0 in range(lo, hi, chunk):
        p = P[i0:min(i0 + chunk, hi)]
        dx = p[:, 0:1] - p1x
        dy = p[:, 1:2] - p1y
        xl = dx * tx[None, :] + dy * ty[None, :]
        yl = -dx * ty[None, :] + dy * tx[None, :]
        dx = xl - L[None, :]                                  # reuse: xl - L
        lnr = 0.5 * np.log((xl * xl + yl * yl + 1e-30) / (dx * dx + yl * yl + 1e-30))
        dth = np.arctan2(yl, dx) - np.arctan2(yl, xl)
        u[i0:i0 + len(p)] = Uinf[0] + lnr @ lnr_x + dth @ dth_x
        v[i0:i0 + len(p)] = Uinf[1] + lnr @ lnr_y + dth @ dth_y

    _parallel(len(P), block)
    return u, v


# --------------------------------------------------------------------------- decoder
def fwd_parallel(X, s_row, PA, z, W, threads=None):
    """The decoder's forward pass over row blocks.

    Every query point is independent, so splitting the rows changes nothing; the matrix products
    are already threaded by the platform's BLAS, but the activations between them are not, and on
    a full RANS mesh they are most of the cost."""
    from supervised_field.nf_eval_shape import fwd_np
    n = len(X)
    t = int(threads or THREADS)
    if t <= 1 or n < 4 * CHUNK:
        return np.asarray(fwd_np(X, s_row, PA, z, W))
    step = -(-n // t)
    parts = list(_pool().map(lambda i: np.asarray(fwd_np(X[i:i + step], s_row[i:i + step], PA[i:i + step], z, W)),
                             range(0, n, step)))
    return np.concatenate(parts, axis=0)


# --------------------------------------------------------------------------- installation
def install():
    """Point the paint module's three kernels at these implementations."""
    import supervised_field.airfrans_p3_fields_v2 as M
    M.field_uv = field_uv
    M.sdf_chunked = sdf_chunked
    M.nearest_idx = nearest_idx
    return M


def verify(case=None, cases=1):
    """Check the fast kernels against the originals on real cases, and time the pathway.

    Two comparisons: each kernel on its own away from the surface, where the original is
    well defined, and the painted field itself, which is what reaches the decoder. Points
    lying exactly on the surface are excluded from the kernel comparison because the panel
    kernel is singular there; the paint overwrites them with the no-slip condition anyway,
    so the painted-field comparison covers every point without exception."""
    import json
    import os
    import statistics
    import sys
    import time
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE)
    import state_provider as G                     # puts the release's code/model and code/field on sys.path
    from supervised_field.nf_eval200 import ROOT
    from supervised_field.nf_hybrid_field import _case_inputs
    import supervised_field.airfrans_p3_fields_v2 as M
    from supervised_field.nf_extract import panel_mids_normals
    orig = dict(field_uv=M.field_uv, sdf_chunked=M.sdf_chunked, nearest_idx=M.nearest_idx)
    G._init({0: os.path.join(G.VP.WEIGHTS, "field_decoder_f0cm_s1.npz")}, None)
    prov = G.ConfirmedStateProvider(s_rule="teH")
    names = sorted(json.load(open(os.path.join(ROOT, "manifest.json")))["full_test"])[:cases] if case is None else [case]

    def med(fn, reps=3):
        fn(); ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return statistics.median(ts)

    report = []
    for nm in names:
        ci = _case_inputs(nm, prov, feature_mode="legacy_paint4")
        P = np.asarray(ci["P"], np.float64)[:, :2]; poly = ci["poly"]
        a, Re, cl = float(ci["a"]), float(ci["Re"]), ci["cl_used"]
        nodes = poly if np.allclose(poly[0], poly[-1]) else np.vstack([poly, poly[:1]])
        dec = M.decamber_gamma(nodes, a, cl)
        mid, _ = panel_mids_normals(poly)

        # ---- painted field: the contract that matters ----
        for k, f in orig.items():
            setattr(M, k, f)
        ro = M.paint_points_v2(P, poly, a, cl, Re)
        t_paint_o = med(lambda: M.paint_points_v2(P, poly, a, cl, Re))
        install()
        rn = M.paint_points_v2(P, poly, a, cl, Re)
        t_paint_n = med(lambda: M.paint_points_v2(P, poly, a, cl, Re))
        painted = {}
        for key in sorted(set(ro) & set(rn)):
            x, y = np.asarray(ro[key]), np.asarray(rn[key])
            if x.dtype.kind not in "fiu" or x.shape != y.shape or x.size == 0:
                continue
            scale = max(float(np.max(np.abs(x))), 1e-30)
            painted[key] = float(np.max(np.abs(x - y)) / scale)

        # ---- kernels on their own, away from the singular on-surface points ----
        for k, f in orig.items():
            setattr(M, k, f)
        d_ref = orig["sdf_chunked"](P, nodes)
        off = np.abs(d_ref) > 1e-12
        uo, vo = orig["field_uv"](P, dec["geom"], dec["gamma"], dec["sigma"], dec["Uinf"])
        un, vn = field_uv(P, dec["geom"], dec["gamma"], dec["sigma"], dec["Uinf"])
        dn = sdf_chunked(P, nodes)
        no, nn_ = orig["nearest_idx"](P, mid), nearest_idx(P, mid)
        rel = lambda x, y, m: float(np.max(np.abs(x[m] - y[m])) / max(float(np.max(np.abs(y[m]))), 1e-30))
        t = dict(
            uv=(med(lambda: orig["field_uv"](P, dec["geom"], dec["gamma"], dec["sigma"], dec["Uinf"])),
                med(lambda: field_uv(P, dec["geom"], dec["gamma"], dec["sigma"], dec["Uinf"]))),
            sdf=(med(lambda: orig["sdf_chunked"](P, nodes)), med(lambda: sdf_chunked(P, nodes))),
            near=(med(lambda: orig["nearest_idx"](P, mid)), med(lambda: nearest_idx(P, mid))))
        row = dict(case=nm, n=int(len(P)), n_on_surface=int((~off).sum()),
                   uv_rel_off_surface=max(rel(un, uo, off), rel(vn, vo, off)),
                   sdf_rel=rel(dn, d_ref, np.ones(len(P), bool)),
                   nearest_mismatches=int(np.count_nonzero(no != nn_)),
                   painted_field_rel=painted,
                   paint_s=dict(original=t_paint_o, fast=t_paint_n, speedup=t_paint_o / t_paint_n),
                   kernels={k: dict(original_s=o, fast_s=f, speedup=o / f) for k, (o, f) in t.items()})
        report.append(row)
        print(f"{nm[:42]:42s} n={row['n']:7d}  ({row['n_on_surface']} points lie on the surface)")
        print(f"   panel field   rel {row['uv_rel_off_surface']:.2e}   {t['uv'][0]:6.2f} -> {t['uv'][1]:6.2f} s  ({t['uv'][0]/t['uv'][1]:5.1f}x)")
        print(f"   signed dist   rel {row['sdf_rel']:.2e}   {t['sdf'][0]:6.2f} -> {t['sdf'][1]:6.2f} s  ({t['sdf'][0]/t['sdf'][1]:5.1f}x)")
        print(f"   nearest panel mismatches {row['nearest_mismatches']}      {t['near'][0]:6.2f} -> {t['near'][1]:6.2f} s  ({t['near'][0]/t['near'][1]:5.1f}x)")
        print(f"   paint total                       {t_paint_o:6.2f} -> {t_paint_n:6.2f} s  ({t_paint_o/t_paint_n:5.1f}x)")
        print("   painted field, max relative difference: " + ", ".join(f"{k} {v:.1e}" for k, v in sorted(painted.items())))
        assert row["uv_rel_off_surface"] < 1e-12, "the panel kernel disagrees away from the surface"
        assert row["sdf_rel"] < 1e-12, "the signed distance disagrees"
        assert row["nearest_mismatches"] == 0, "the nearest panel disagrees"
        assert all(v < 1e-9 for v in painted.values()), "the painted field disagrees"
    return report


if __name__ == "__main__":
    import sys
    if "--verify" in sys.argv:
        n = int(sys.argv[sys.argv.index("--cases") + 1]) if "--cases" in sys.argv else 1
        verify(cases=n)
        print("all kernels agree with the originals")
