# -*- coding: utf-8 -*-
"""Regenerate holdout-schema targets on the Xeon with the NARROWED convergence flag.

Row-for-row identical to staged/scripts/gen_holdout_targets.py (same held-out UIUC
shapes, n_crit, n_panels, TE index rule, output keys) except:
  * coordinates arrive precomputed (no TensorFlow on this host),
  * `converged` comes from xfoil_bl_narrow, which tests for the GLOBAL
    "VISCAL:  Convergence failed" instead of any "onvergence failed" substring,
  * two extra diagnostic arrays are saved so the change can be audited:
    station_march_failures (per row) and viscal_failed (per row).

Usage:
  python gen_targets_win.py --alphas -4,-2,0,2,4,6,8,10,12 --Res 2e5,5e5,1e6,3e6
                            --out repro_narrow.npz --cache_prefix rn
"""
import argparse, hashlib, os, sys, time
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("XFOIL_EXE", os.path.join(os.path.expanduser("~"), "nfcpu", "xfoil.exe"))
import xfoil_bl_narrow as XB

Z = np.load(os.path.join(HERE, "sep_eval_coords.npz"), allow_pickle=True)
N_PANELS, MAX_ITER, TIMEOUT = 200, 200, 90.0
N_CRIT = float(Z["n_crit"])
# Windows uses spawn: workers re-import this module, so module-level state set
# inside main() is NOT visible to them.  Every worker argument travels in the task.


def _te_index(x_branch):
    return -2 if (len(x_branch) >= 2 and x_branch[-1] > 1.0 + 1e-6) else -1


def run_one(task):
    j, i, a, re, prefix = task
    coords = (Z["xu"][i], Z["yu"][i], Z["xl"][i], Z["yl"][i])
    try:
        r = XB.run_xfoil_visc_bl(
            naca="", alpha_deg=a, Re=re, coords=coords,
            cache_label=prefix + hashlib.md5(Z["cst"][i].tobytes()).hexdigest()[:10],
            n_panels=N_PANELS, n_crit=N_CRIT, max_iter=MAX_ITER, timeout=TIMEOUT)
    except Exception:
        r = None
    if r is None:
        return j, None, 0, 1
    smf = int(r.get("station_march_failures", 0))
    vf = int(bool(r.get("viscal_failed", False)))
    if not r.get("converged", False):
        return j, None, smf, vf
    try:
        iu, il = _te_index(r["x_upper"]), _te_index(r["x_lower"])
        # OP1 surface truth, same recipe as staged/scripts/gen_xfoil_all_targets.py:
        # Cp = 1 - ue^2 per side, interpolated onto the OP1 240-panel midpoints; s_stag = argmax Cp in s in (0.4,0.6)
        xc = Z["xc"][i].astype(np.float64); sp = Z["s_panels"].astype(np.float64); mu = sp < 0.5
        cpu = 1.0 - r["ue_upper"].astype(np.float64) ** 2; cpl = 1.0 - r["ue_lower"].astype(np.float64) ** 2
        cp = np.zeros(240, np.float32)
        cp[mu] = np.interp(xc[mu], r["x_upper"].astype(np.float64), cpu); cp[~mu] = np.interp(xc[~mu], r["x_lower"].astype(np.float64), cpl)
        mle = (sp > 0.4) & (sp < 0.6); s_stag = float(sp[int(np.argmax(np.where(mle, cp, -np.inf)))])
        # XFOIL BL state at NeuralFoil's 32 x/c stations (LE->TE), per side, for a station-matched comparison
        nx_ = Z["nf_x"].astype(np.float64)
        xu_, xl_ = r["x_upper"].astype(np.float64), r["x_lower"].astype(np.float64)
        blst = np.stack([np.interp(nx_, xu_, r["theta_upper"]), np.interp(nx_, xl_, r["theta_lower"]),
                         np.interp(nx_, xu_, r["dstar_upper"]), np.interp(nx_, xl_, r["dstar_lower"])]).astype(np.float32)   # (4,32)
        return j, (float(r.get("cl", np.nan)), float(r.get("cd", np.nan)),
                   float(r["theta_upper"][iu]), float(r["theta_lower"][il]),
                   float(r["dstar_upper"][iu]), float(r["dstar_lower"][il]),
                   abs(float(r["ue_upper"][iu])),
                   float(r.get("x_tr_top") or 1.0), float(r.get("x_tr_bot") or 1.0), cp, s_stag, float(r.get("cm", np.nan)), blst), smf, vf
    except Exception:
        return j, None, smf, vf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", required=True)
    ap.add_argument("--Res", default="2e5,5e5,1e6,3e6")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache_prefix", default="rn")
    ap.add_argument("--workers", type=int, default=60)
    a = ap.parse_args()
    alphas = [float(x) for x in a.alphas.split(",")]
    Res = [float(x) for x in a.Res.split(",")]
    rows = [(i, al, re) for i in range(len(Z["cst"])) for al in alphas for re in Res]
    tasks = [(j, i, al, re, a.cache_prefix) for j, (i, al, re) in enumerate(rows)]
    M = len(rows)
    nw = max(1, min(60, a.workers))          # Windows Pool: 64-handle limit
    print("rows %d  workers %d  n_crit %.1f  alphas %s" % (M, nw, N_CRIT, alphas), flush=True)
    t0 = time.time(); res = {}; smf = np.zeros(M, np.int32); vf = np.zeros(M, np.int32)
    with Pool(nw) as p:
        for k, (j, v, s, f) in enumerate(p.imap_unordered(run_one, tasks, chunksize=4)):
            res[j] = v; smf[j] = s; vf[j] = f
            if (k + 1) % 250 == 0:
                print("  %d/%d valid %d  %.0fs" % (k + 1, M,
                      sum(1 for x in res.values() if x is not None), time.time() - t0), flush=True)
    F = lambda: np.zeros((M, 1), np.float32)
    cst = np.zeros((M, 16), np.float32); names = np.empty(M, dtype=object)
    al_, re_ = F(), F(); cl, cd = F(), F()
    tut, tlt, dut, dlt, uet, xtt, xtb = (F() for _ in range(7))
    cpp = np.zeros((M, 240), np.float32); sst = F(); cmv = F(); bls = np.zeros((M, 4, 32), np.float32)
    valid = np.zeros(M, bool)
    for j, (i, alv, rev) in enumerate(rows):
        cst[j] = Z["cst"][i]; names[j] = str(Z["names"][i]); al_[j, 0] = alv; re_[j, 0] = rev
        v = res.get(j)
        if v is None:
            continue
        (cl[j, 0], cd[j, 0], tut[j, 0], tlt[j, 0],
         dut[j, 0], dlt[j, 0], uet[j, 0], xtt[j, 0], xtb[j, 0], cpp[j], sst[j, 0], cmv[j, 0], bls[j]) = v
        valid[j] = True
    np.savez(os.path.join(HERE, a.out),
             cst=cst, names=names.astype(str), alpha=al_, Re=re_,
             cl_visc=cl, cd_visc=cd, theta_u_te=tut, theta_l_te=tlt,
             dstar_u_te=dut, dstar_l_te=dlt, ue_te=uet,
             x_tr_top=xtt, x_tr_bot=xtb, valid=valid,
             cp_panels=cpp, s_stag=sst, cm_visc=cmv, xc=Z["xc"], s_panels=Z["s_panels"], bl_at_nf_x=bls, nf_x=Z["nf_x"],
             station_march_failures=smf, viscal_failed=vf)
    print("done %.0fs  valid %d/%d  viscal_failed %d" % (
        time.time() - t0, int(valid.sum()), M, int(vf.sum())), flush=True)
    for alv in alphas:
        m = (al_[:, 0] == np.float32(alv))
        print("  alpha %6.1f valid %3d/%3d" % (alv, int(valid[m].sum()), int(m.sum())), flush=True)


if __name__ == "__main__":
    main()
