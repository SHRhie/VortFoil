# -*- coding: utf-8 -*-
"""supervised_field/nf_extract.py — training-tensor extraction for the MARIO-class
conditional neural field with the v2 physics paint as a residual prior.

For each of the 200 AirfRANS scarce_train cases (full RANS volume fields):
  * subsample 12,000 NON-surface volume nodes without replacement (one global
    rng seed 0, cases in sorted order) PLUS all surface nodes up to 1,500
    (random-subsample surface if more) -> ~13k rows/case, ~2.6M total;
  * evaluate the v2 paint (airfrans_p3_fields_v2.paint_points_v2, cl_XTR via the
    md5-cached airfrans_p1_labels.run_one) at exactly those nodes;
  * per-point features (float32, 13):
      [0] x   [1] y   [2] sdf d (signed, NEGATIVE inside; sdf2d.geometry.sdf_np
      on the closed naca_generator polyline)   [3] nx [4] ny (nearest-surface-
      panel OUTWARD normal)   [5] is_surface   [6] alpha_deg/15
      [7] 2*(logRe-log2e6)/(log6e6-log2e6)-1   [8:12] padded params/PSCALE
      [12] is5  (PSCALE/padding identical to airfrans_head.feats)
  * PAINT (rows,4) = [u_paint, v_paint (m/s), p_paint (kinematic), nut_paint
    (kinematic)] — dimensionalized exactly like the v2 eval (u,v *= U,
    p = cp*0.5*U^2, nut *= U), with u/v/nut zeroed AT surface nodes (v2
    convention);
  * TRUTH (rows,4) = [ux, uy (m/s), p kinematic, nut kinematic].

Output: supervised_field/_nf_train.npz with X, PAINT, TRUTH, case_id and a case
manifest (names, U, alpha, Re, params4, n_params, cl_xtr) + skipped-case names.
"""
import os, json, time
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XFOIL_EXE", os.path.expanduser("~/src/xfoil/bin/xfoil"))
import numpy as np
from multiprocessing import Pool

import airfrans as af
from supervised_field.airfrans_p3_fields_v2 import (
    case_params, analytic_polyline, paint_points_v2, nearest_idx)
from supervised_field.airfrans_p1_labels import run_one

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AIRFRANS_ROOT",   # external dataset, not shipped with this release
                     os.path.join(os.path.dirname(HERE), "airfrans_data", "Dataset"))
OUT = os.path.join(HERE, "_nf_train.npz")
NU_AIR = 1.56e-5
N_VOL = 12000
N_SURF_MAX = 1500
PSCALE = np.array([6.0, 8.0, 20.0, 20.0], np.float64)   # airfrans_head convention
A_NORM = 15.0
RE_LO, RE_HI = np.log(2e6), np.log(6e6)

FEAT_NAMES = ["x", "y", "sdf", "nx", "ny", "is_surf", "alpha/15", "re_norm",
              "p0/6", "p1/8", "p2/20", "p3/20", "is5"]
PAINT_NAMES = ["u_paint", "v_paint", "p_paint", "nut_paint"]
TRUTH_NAMES = ["ux", "uy", "p", "nut"]


def panel_mids_normals(nodes):
    """Panel midpoints + OUTWARD unit normals of the closed polyline (zero-length
    panels dropped)."""
    A, B = nodes[:-1], nodes[1:]
    L = np.linalg.norm(B - A, axis=1)
    ok = L > 1e-12
    A, B, L = A[ok], B[ok], L[ok]
    mid = 0.5 * (A + B)
    t = (B - A) / L[:, None]
    n = np.column_stack([t[:, 1], -t[:, 0]])
    ctr = mid.mean(0)
    s = np.sign(np.sum((mid - ctr) * n, axis=1, keepdims=True)); s[s == 0] = 1.0
    return mid, n * s


def main():
    with open(os.path.join(ROOT, "manifest.json")) as f:
        names = sorted(json.load(f)["scarce_train"])
    assert len(names) == 200, len(names)

    tasks = []
    for nm in names:
        U, a, prm = case_params(nm)
        tasks.append((prm, a, U * 1.0 / NU_AIR))
    print(f"[nf] XFOIL XTR cl labels for {len(tasks)} scarce_train cases "
          f"(md5-cached, Pool(6))...", flush=True)
    t0 = time.time()
    with Pool(6) as pool:
        labels = pool.map(run_one, tasks)
    n_ok = sum(1 for l in labels if l["cl"] is not None)
    print(f"[nf] labels: {n_ok}/{len(labels)} converged  ({time.time()-t0:.0f}s)",
          flush=True)

    rng = np.random.default_rng(0)
    Xs, Ps, Ts, Cs = [], [], [], []
    man = dict(names=[], U=[], alpha=[], Re=[], params4=[], n_params=[], cl_xtr=[],
               n_rows=[], n_surf_rows=[])
    skipped = []
    t0 = time.time()
    for i, (nm, lab) in enumerate(zip(names, labels)):
        U, a, prm = case_params(nm)
        Re = U * 1.0 / NU_AIR
        if lab["cl"] is None:
            skipped.append(nm)
            print(f"  SKIP {nm} (XFOIL XTR failed, a={a:+.2f})", flush=True)
            continue
        cl_visc = float(lab["cl"])

        sim = af.Simulation(root=ROOT, name=nm)
        P_all = np.asarray(sim.position, np.float64)
        V_all = np.asarray(sim.velocity, np.float64)
        p_all = np.asarray(sim.pressure, np.float64).ravel()
        nt_all = np.asarray(sim.nu_t, np.float64).ravel()
        surf_mask = np.asarray(sim.surface, bool)

        # ---- subsample: all surface nodes (cap 1500) + 12k non-surface nodes ----
        si = np.flatnonzero(surf_mask)
        if len(si) > N_SURF_MAX:
            si = rng.choice(si, N_SURF_MAX, replace=False)
        vi = np.flatnonzero(~surf_mask)
        vsel = rng.choice(vi, min(N_VOL, len(vi)), replace=False)
        sel = np.concatenate([si, vsel])
        P = P_all[sel]
        is_s = surf_mask[sel]

        # ---- v2 paint at exactly these nodes ----
        poly = analytic_polyline(prm)
        r = paint_points_v2(P, poly, a, cl_visc, Re)
        u_paint = r["u"] * U
        v_paint = r["v"] * U
        p_paint = r["cp"] * 0.5 * U**2
        nut_paint = r["nut"] * U
        u_paint[is_s] = 0.0; v_paint[is_s] = 0.0; nut_paint[is_s] = 0.0  # v2 conv.

        # ---- features ----
        mid, nrm = panel_mids_normals(poly)
        nn = nearest_idx(P, mid)
        prm4 = np.zeros(4); prm4[:len(prm)] = prm
        is5 = float(len(prm) == 4)
        an = a / A_NORM
        re_n = 2.0 * (np.log(Re) - RE_LO) / (RE_HI - RE_LO) - 1.0
        n_pt = len(P)
        X = np.column_stack([
            P[:, 0], P[:, 1], r["d"], nrm[nn, 0], nrm[nn, 1],
            is_s.astype(np.float64),
            np.full(n_pt, an), np.full(n_pt, re_n),
            np.tile(prm4 / PSCALE, (n_pt, 1)),
            np.full(n_pt, is5)]).astype(np.float32)

        Xs.append(X)
        Ps.append(np.column_stack([u_paint, v_paint, p_paint,
                                   nut_paint]).astype(np.float32))
        Ts.append(np.column_stack([V_all[sel, 0], V_all[sel, 1], p_all[sel],
                                   nt_all[sel]]).astype(np.float32))
        cid = len(man["names"])
        Cs.append(np.full(n_pt, cid, np.int32))
        man["names"].append(nm); man["U"].append(U); man["alpha"].append(a)
        man["Re"].append(Re); man["params4"].append(prm4)
        man["n_params"].append(len(prm)); man["cl_xtr"].append(cl_visc)
        man["n_rows"].append(n_pt); man["n_surf_rows"].append(int(is_s.sum()))

        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  [{i+1:3d}/200] {nm[:44]:44s} rows={n_pt} "
                  f"surf={int(is_s.sum())} leak={r['vn_leak']:.3f}  "
                  f"{el:.0f}s elapsed, ETA {el/(i+1)*(200-i-1):.0f}s", flush=True)

    X = np.concatenate(Xs); PA = np.concatenate(Ps)
    TR = np.concatenate(Ts); CI = np.concatenate(Cs)
    np.savez_compressed(
        OUT, X=X, PAINT=PA, TRUTH=TR, case_id=CI,
        names=np.array(man["names"]), U=np.array(man["U"]),
        alpha=np.array(man["alpha"]), Re=np.array(man["Re"]),
        params4=np.array(man["params4"]), n_params=np.array(man["n_params"]),
        cl_xtr=np.array(man["cl_xtr"]), n_rows=np.array(man["n_rows"]),
        n_surf_rows=np.array(man["n_surf_rows"]),
        skipped=np.array(skipped), feat_names=np.array(FEAT_NAMES),
        paint_names=np.array(PAINT_NAMES), truth_names=np.array(TRUTH_NAMES),
        pscale=PSCALE, a_norm=A_NORM, re_lo=RE_LO, re_hi=RE_HI)

    print(f"\n[nf] cases={len(man['names'])}/200 (skipped {len(skipped)}: "
          f"{skipped})", flush=True)
    print(f"[nf] X{X.shape} PAINT{PA.shape} TRUTH{TR.shape} -> {OUT} "
          f"({os.path.getsize(OUT)/1e6:.0f} MB)", flush=True)
    # sanity: finite, sdf sign, surface truth
    for nmx, arr in (("X", X), ("PAINT", PA), ("TRUTH", TR)):
        print(f"[nf] {nmx} finite={np.isfinite(arr).all()} "
              f"min={arr.min(0)} max={arr.max(0)}", flush=True)
    sm = X[:, 5] > 0.5
    print(f"[nf] surface rows {int(sm.sum())}  |sdf|surf max={np.abs(X[sm,2]).max():.2e}"
          f"  truth |u|surf max={np.abs(TR[sm,0]).max():.2e}", flush=True)


if __name__ == "__main__":
    main()
