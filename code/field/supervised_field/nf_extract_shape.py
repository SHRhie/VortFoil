# -*- coding: utf-8 -*-
"""nf_extract_shape.py — per-case SHAPE TOKENS for the geometry-conditioned encoder.

For each case we take the REAL surface node set (NOT analytic_polyline = naca_generator,
which would smuggle NACA params back in = leakage), resample to M points with LE-weighted
density, and normalize to a shared chord-relative frame.  Output is aligned (same order)
to the `names` array of the matching training npz so the trainer can gather by case_id.

  --dataset airfrans : AirfRANS full_train-800 -> sim.surface nodes -> _nf_shape_tokens_airfrans.npz
  --smoke N          : only first N cases, no save (GATE 0 sanity)

Token features C=2: chord-normalized (x', y').  (curvature/arc-length grafts deferred.)
"""
import os, sys, json, time
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AIRFRANS_ROOT",   # external dataset, not shipped with this release
                     os.path.join(os.path.dirname(HERE), "airfrans_data", "Dataset"))
M = 256          # points per shape token cloud
LE_W = 0.05      # LE-density weight floor: w ~ 1/(x'+LE_W)


def resample_le(xy, m, seed):
    """xy (N,2) raw surface coords -> (m,2) chord-normalized, LE-weighted subsample.
    Order-free (PointNet max-pool is permutation invariant)."""
    x = xy[:, 0]; y = xy[:, 1]
    xmin, xmax = x.min(), x.max()
    chord = max(xmax - xmin, 1e-9)
    xn = (x - xmin) / chord          # [0,1]
    yn = y / chord                   # keep camber sign/scale
    P = np.column_stack([xn, yn]).astype(np.float32)
    n = len(P)
    rng = np.random.RandomState(seed)
    w = 1.0 / (xn + LE_W)            # denser near LE (xn~0)
    w = w / w.sum()
    if n >= m:
        idx = rng.choice(n, size=m, replace=False, p=w)
    else:                            # pad by weighted resample with replacement
        idx = np.concatenate([np.arange(n), rng.choice(n, size=m - n, replace=True, p=w)])
    return P[idx]


def airfrans_surfaces(names, smoke):
    import airfrans as af
    toks = []; t0 = time.time()
    for i, nm in enumerate(names):
        sim = af.Simulation(root=ROOT, name=nm)
        Pos = np.asarray(sim.position, np.float64)
        sm = np.asarray(sim.surface, bool)
        xy = Pos[sm][:, :2]
        toks.append(resample_le(xy, M, seed=i))
        if (i + 1) % 100 == 0 or i + 1 == len(names):
            print(f"  [airfrans {i+1}/{len(names)}] surf_nodes~{sm.sum()} ({time.time()-t0:.0f}s)", flush=True)
    return np.stack(toks)


def main():
    ds = sys.argv[sys.argv.index("--dataset") + 1] if "--dataset" in sys.argv else "airfrans"
    smoke = "--smoke" in sys.argv
    nsm = int(sys.argv[sys.argv.index("--smoke") + 1]) if smoke and \
        sys.argv.index("--smoke") + 1 < len(sys.argv) and \
        sys.argv[sys.argv.index("--smoke") + 1].isdigit() else 5

    if ds == "airfrans":
        train_npz = os.path.join(HERE, "_nf_train_v5_full800.npz")
        names = [str(x) for x in np.load(train_npz, allow_pickle=True)["names"]]
        from supervised_field import WEIGHTS as _W
        out = os.path.join(_W, "airfrans_surface_tokens.npz")   # release name
        if smoke:
            names = names[:nsm]
        toks = airfrans_surfaces(names, smoke)
    elif ds == "airfrans_all":
        # ALL 1000 AirfRANS cases (full_train U full_test) — needed for OOD reynolds/aoa splits
        train_npz = "manifest(full_train+full_test)"
        with open(os.path.join(ROOT, "manifest.json")) as f:
            man = json.load(f)
        names = sorted(set(man["full_train"]) | set(man["full_test"]))
        out = os.path.join(HERE, "_nf_shape_tokens_airfrans_all.npz")
        if smoke:
            names = names[:nsm]
        toks = airfrans_surfaces(names, smoke)
    else:
        raise SystemExit("--dataset airfrans|airfrans_all")

    # ---- GATE 0 sanity ----
    print(f"[shape] dataset={ds} tokens shape={toks.shape} (n,M,C) "
          f"x'[{toks[...,0].min():.3f},{toks[...,0].max():.3f}] "
          f"y'[{toks[...,1].min():.3f},{toks[...,1].max():.3f}]", flush=True)
    assert toks.shape[1:] == (M, 2), toks.shape
    assert np.isfinite(toks).all()
    # NOT analytic_polyline assert: tokens come from sim.surface (real)
    print(f"[shape] source = REAL surface nodes (sim.surface), "
          f"NOT analytic_polyline(naca_generator).", flush=True)
    if smoke:
        print(f"[shape] SMOKE ({len(names)} cases) — not saving.", flush=True)
        return
    np.savez_compressed(out, surf_tokens=toks.astype(np.float32),
                        names=np.array(names), M=M, n_feat=2)
    print(f"[shape] wrote {out} ({os.path.getsize(out)/1e6:.1f} MB, {len(names)} cases, "
          f"aligned to {os.path.basename(train_npz)} names order)", flush=True)


if __name__ == "__main__":
    main()
