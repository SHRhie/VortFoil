# -*- coding: utf-8 -*-
"""Reproduce this build's recorded scores, so an installed package can prove it is the deployed model."""
import os

import numpy as np


def _gate(z):
    v = z["valid"] & (np.abs(z["cl_visc"][:, 0]) <= 3) & (z["cd_visc"][:, 0] > 1e-4) & (z["cd_visc"][:, 0] <= 0.1)
    for k in ("theta_u_te", "theta_l_te", "dstar_u_te", "dstar_l_te", "ue_te"):
        v &= np.isfinite(z[k][:, 0]) & ((z[k][:, 0] != 0) if "theta" in k else True)
    return v


def run(mode="graph", device="cpu", data=None, tol=2e-4):
    import vortfoil
    m = vortfoil.manifest()
    data = data or os.environ.get("VORTFOIL_EVAL_DATA")
    if not data or not os.path.exists(data):
        print("selfcheck needs the evaluation grid: pass --data holdout_targets.npz "
              "(distributed with the repository, not with the wheel)")
        return 2
    z = np.load(data, allow_pickle=True)
    keep = _gate(z)
    vf = vortfoil.load_model(mode=mode, device=device)
    r = vf.predict(cst=z["cst"][keep], alpha=z["alpha"][keep].ravel(), Re=z["Re"][keep].ravel())
    cl_t, cd_t = z["cl_visc"][keep, 0], z["cd_visc"][keep, 0]
    r2 = lambda p, t: float(1 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2))
    ref = m["evaluation"]["all_validity_filtered_rows"]
    got = {"cl": r2(r.cl, cl_t), "cl_gamma": r2(r.cl_gamma, cl_t), "cd": r2(np.clip(r.cd, 1e-4, 0.1), cd_t)}
    print(vf, f"\n rows {int(keep.sum())} (expected {ref['n']})")
    bad = 0
    for k, v in got.items():
        want = ref["scores"].get(k)
        ok = want is None or abs(v - float(want)) <= tol
        bad += not ok
        print(f"  {k:9s} {v:.6f}" + (f"   recorded {want}   {'ok' if ok else 'MISMATCH'}" if want else ""))
    print(" SELFCHECK", "PASS" if not bad else "FAIL")
    return 0 if not bad else 1
