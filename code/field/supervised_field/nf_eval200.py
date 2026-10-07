# -*- coding: utf-8 -*-
"""supervised_field/nf_eval200.py — CITABLE full_test evaluation of the v1
alpha-gated residual neural field (_nf_residual_v1.npz) on ALL 200 AirfRANS
full_test cases (full volume, ALL mesh nodes — MARIO protocol).

Per case:
  1. cl for the v2 paint: XFOIL XTR label via airfrans_p1_labels.run_one
     (md5-cached).  If XFOIL fails/non-converged, FALL BACK to the 5-member
     ensemble head cl (airfrans_head_ens.py members, numpy forward from
     best_weights/airfrans_head_ens_v2.npz) — every fallback is documented in
     the per-case row (cl_source) and summarized.
  2. v2 paint at all nodes (airfrans_p3_fields_v2.paint_points_v2), v2 surface
     convention (u, v, nut zeroed on surface nodes — same as training PAINT).
  3. v1 net numpy forward from the saved npz (artifact-verified path), then
     pred[ux,uy,p] = w(alpha)*PAINT + net*std + t_mean;  pred[nut] = net*std +
     t_mean (always scratch).  Gate w stored in the npz: w=1 for a<=8 deg,
     smoothstep 3t^2-2t^3 with t=(11-a)/3 for 8<a<11, w=0 for a>=11.
  4. Standardized MSE (stds from _airfrans_v2_trainstats.npz, the SAME stats
     as the v2 paint eval / MARIO table conversion) per field over the full
     volume + surface-p MSE on surface nodes (volume-p std).  Paint-only MSE
     recorded too (reference).

Aggregates: all-200 means, alpha<10 / alpha>=10 splits.  Comparison rows in
the JSON: v0 20-case means (_nf_eval.json) and MARIO's published numbers (raw).
-> supervised_field/_nf_eval200.json
"""
import os, json, time
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XFOIL_EXE", os.path.expanduser("~/src/xfoil/bin/xfoil"))
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AIRFRANS_ROOT",   # external dataset, not shipped with this release
                     os.path.join(os.path.dirname(HERE), "airfrans_data", "Dataset"))
W_V1 = os.path.join(HERE, "_nf_residual_v1.npz")
from supervised_field import WEIGHTS as _W   # <release>/weights
STATS_NPZ = os.path.join(_W, "airfrans_train_stats.npz")
ENS_PATHS = [os.path.join(HERE, "best_weights", "airfrans_head_ens_v2.npz"),
             os.path.join(HERE, "_airfrans_head_ens.npz")]
V0_JSON = os.path.join(HERE, "_nf_eval.json")
OUT = os.path.join(HERE, "_nf_eval200.json")
NU_AIR = 1.56e-5
FIELDS = ("ux", "uy", "p", "nut")
KEYS = list(FIELDS) + ["p_surf"]
MARIO_PUBLISHED = dict(ux=1.52e-3, uy=1.13e-3, p=2.4e-3, nut=9.6e-3,
                        p_surf=2.7e-2)


# ---------------------------------------------------------------------------
# v1 net + gate (numpy, from the saved npz)
# ---------------------------------------------------------------------------
def np_forward(wz, X, chunk=65536):
    B = wz["fourier_B"]; nl = int(wz["n_layers"])
    out = np.empty((len(X), len(FIELDS)), np.float32)
    for i0 in range(0, len(X), chunk):
        x = X[i0:i0 + chunk]
        ph = x[:, :2] @ B
        h = np.concatenate([np.sin(ph), np.cos(ph), x[:, 2:]], axis=1)
        for i in range(nl):
            h = h @ wz[f"W{i}"] + wz[f"b{i}"]
            if i < nl - 1:
                h = h * (1.0 / (1.0 + np.exp(-h)))           # swish
        out[i0:i0 + chunk] = h
    return out


def gate_w(alpha_deg, a_lo, a_hi):
    t = np.clip((a_hi - float(alpha_deg)) / (a_hi - a_lo), 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))


# ---------------------------------------------------------------------------
# ensemble-head cl fallback (numpy forward of the 5 members; replicates
# airfrans_head.feats — same PSCALE/A_NORM/RE_LO/RE_HI as nf_extract — without
# importing TensorFlow)
# ---------------------------------------------------------------------------
def head_feats(prm, alpha_deg, Re, PSCALE, A_NORM, RE_LO, RE_HI):
    prm4 = np.zeros(4); prm4[:len(prm)] = prm
    is5 = float(len(prm) == 4)
    an = alpha_deg / A_NORM
    re = 2.0 * (np.log(Re) - RE_LO) / (RE_HI - RE_LO) - 1.0
    return np.concatenate([prm4 / PSCALE, [is5, an, an * an, abs(an), re]]
                          ).astype(np.float32)[None, :]


def ens_cl(ez, x9):
    """Mean cl over the n_members MLP members (9->128x4 swish->2; out[:,0]=cl)."""
    cls = []
    for i in range(int(ez["n_members"])):
        h = x9
        for j in range(0, 10, 2):
            h = h @ ez[f"m{i}_w{j}"] + ez[f"m{i}_w{j+1}"]
            if j < 8:
                h = h * (1.0 / (1.0 + np.exp(-h)))           # swish
        cls.append(float(h[0, 0]))
    return float(np.mean(cls))


def main():
    import airfrans as af
    from supervised_field.airfrans_p3_fields_v2 import (
        case_params, analytic_polyline, paint_points_v2)
    from supervised_field.nf_extract import (
        panel_mids_normals, nearest_idx, PSCALE, A_NORM, RE_LO, RE_HI)
    from supervised_field.airfrans_p1_labels import run_one

    t_all = time.time()
    wz = np.load(W_V1, allow_pickle=True)
    assert str(wz["variant"]) == "residual_gated_v1", str(wz["variant"])
    std = np.asarray(wz["std"], np.float64)
    t_mean = np.asarray(wz["t_mean"], np.float64)
    a_lo, a_hi = float(wz["gate_a_lo"]), float(wz["gate_a_hi"])
    sz = np.load(STATS_NPZ, allow_pickle=True)
    std_chk = np.array([float(sz[k][1]) for k in FIELDS])
    assert np.allclose(std, std_chk), (std, std_chk)         # stats consistency

    ens_path = next(p for p in ENS_PATHS if os.path.exists(p))
    ez = np.load(ens_path)
    print(f"[eval200] v1 weights {os.path.basename(W_V1)} | gate "
          f"[{a_lo},{a_hi}] deg | ens fallback {ens_path}", flush=True)

    with open(os.path.join(ROOT, "manifest.json")) as f:
        names = sorted(json.load(f)["full_test"])
    assert len(names) == 200, len(names)

    # ---- XFOIL XTR cl labels (md5-cached) ----
    tasks = [(case_params(nm)[2], case_params(nm)[1],
              case_params(nm)[0] / NU_AIR) for nm in names]
    print(f"[eval200] XFOIL XTR labels for {len(tasks)} full_test cases "
          f"(cached, Pool(6))...", flush=True)
    t0 = time.time()
    with Pool(6) as pool:
        labels = pool.map(run_one, tasks)
    n_ok = sum(1 for l in labels if l["cl"] is not None)
    print(f"[eval200] labels: {n_ok}/200 converged ({time.time()-t0:.0f}s); "
          f"{200-n_ok} will use the ensemble-head cl fallback", flush=True)

    rows, fallbacks = [], []
    t0 = time.time()
    for i, (nm, lab) in enumerate(zip(names, labels)):
        U, a, prm = case_params(nm)
        Re = U / NU_AIR
        if lab["cl"] is not None:
            cl_used, cl_src = float(lab["cl"]), "xfoil_xtr"
        else:
            x9 = head_feats(prm, a, Re, PSCALE, A_NORM, RE_LO, RE_HI)
            cl_used, cl_src = ens_cl(ez, x9), "ens_head_fallback"
            fallbacks.append(dict(name=nm, alpha=a, cl_ens=cl_used))
            print(f"  FALLBACK {nm} a={a:+.2f}: XFOIL XTR failed -> ensemble "
                  f"head cl={cl_used:.4f}", flush=True)

        sim = af.Simulation(root=ROOT, name=nm)
        P = np.asarray(sim.position, np.float64)
        TR = np.column_stack([np.asarray(sim.velocity, np.float64),
                              np.asarray(sim.pressure, np.float64).reshape(-1, 1),
                              np.asarray(sim.nu_t, np.float64).reshape(-1, 1)])
        sm = np.asarray(sim.surface, bool)
        n_pt = len(P)

        # ---- v2 paint (training PAINT convention incl. surface zeroing) ----
        poly = analytic_polyline(prm)
        r = paint_points_v2(P, poly, a, cl_used, Re)
        PA = np.column_stack([r["u"] * U, r["v"] * U,
                              r["cp"] * 0.5 * U**2, r["nut"] * U])
        PA[sm, 0] = 0.0; PA[sm, 1] = 0.0; PA[sm, 3] = 0.0

        # ---- features (identical recipe to nf_extract / v0 eval) ----
        mid, nrm = panel_mids_normals(poly)
        nn = nearest_idx(P, mid)
        prm4 = np.zeros(4); prm4[:len(prm)] = prm
        X = np.column_stack([
            P[:, 0], P[:, 1], r["d"], nrm[nn, 0], nrm[nn, 1],
            sm.astype(np.float64),
            np.full(n_pt, a / A_NORM),
            np.full(n_pt, 2.0 * (np.log(Re) - RE_LO) / (RE_HI - RE_LO) - 1.0),
            np.tile(prm4 / PSCALE, (n_pt, 1)),
            np.full(n_pt, float(len(prm) == 4))]).astype(np.float32)

        # ---- v1 prediction: gated paint + destandardized net ----
        w = gate_w(a, a_lo, a_hi)
        net = np_forward(wz, X) * std[None, :] + t_mean[None, :]
        pred = net.copy()
        pred[:, :3] += w * PA[:, :3]                 # ux,uy,p gated residual
        # nut column = net only (always scratch)

        def mses(pr):
            e = ((pr - TR) / std[None, :]) ** 2
            d = {k: float(v) for k, v in zip(FIELDS, e.mean(0))}
            d["p_surf"] = float(np.mean(((pr[sm, 2] - TR[sm, 2]) / std[2]) ** 2))
            return d

        m_pa, m_v1 = mses(PA), mses(pred)
        rows.append(dict(name=nm, U=U, alpha=a, Re=Re, n_nodes=n_pt,
                         n_surf=int(sm.sum()), cl_used=cl_used,
                         cl_source=cl_src, gate_w=w,
                         vn_leak=float(r["vn_leak"]),
                         mse_paint=m_pa, mse_v1=m_v1))
        if (i + 1) % 10 == 0 or (i + 1) == len(names):
            el = time.time() - t0
            print(f"  [{i+1:3d}/200] {nm[:44]:44s} a={a:+6.2f} w={w:.2f} | "
                  f"v1 ux {m_v1['ux']:.3e} uy {m_v1['uy']:.3e} p "
                  f"{m_v1['p']:.3e} nut {m_v1['nut']:.3e} ps "
                  f"{m_v1['p_surf']:.3e} | {el:.0f}s, ETA "
                  f"{el/(i+1)*(200-i-1):.0f}s", flush=True)

    # ---- aggregates ----
    al = np.array([r["alpha"] for r in rows])
    lo, hi = al < 10.0, al >= 10.0

    def agg(tag, mask):
        sel = [r for r, m in zip(rows, mask) if m]
        return {k: float(np.mean([r[tag][k] for r in sel])) for k in KEYS}

    full_v1 = agg("mse_v1", np.ones(len(rows), bool))
    lo_v1 = agg("mse_v1", lo)
    hi_v1 = agg("mse_v1", hi)
    full_pa = agg("mse_paint", np.ones(len(rows), bool))
    lo_pa, hi_pa = agg("mse_paint", lo), agg("mse_paint", hi)

    v0 = json.load(open(V0_JSON))["mse_std_means"] if os.path.exists(V0_JSON) \
        else None
    out = dict(
        n_evaluated=len(rows), n_alpha_lt10=int(lo.sum()),
        n_alpha_ge10=int(hi.sum()),
        n_xfoil_ok=n_ok, n_cl_fallback=len(fallbacks), fallback_cases=fallbacks,
        weights=os.path.basename(W_V1), stats_npz=os.path.basename(STATS_NPZ),
        ens_fallback=os.path.relpath(ens_path, HERE),
        gate=dict(a_lo=a_lo, a_hi=a_hi,
                  rule="w=1 a<=8; smoothstep 3t^2-2t^3, t=(11-a)/3, 8<a<11; "
                       "w=0 a>=11; applies to ux,uy,p only, nut always scratch"),
        mse_v1=dict(all200=full_v1, alpha_lt10=lo_v1, alpha_ge10=hi_v1),
        mse_paint=dict(all200=full_pa, alpha_lt10=lo_pa, alpha_ge10=hi_pa),
        comparison=dict(
            v0_20case_means=v0,
            mario_published=MARIO_PUBLISHED,
            note="raw standardized MSE; MARIO printed-table conversion x1e-2 "
                 "for ux/uy/p, x1e-1 for nut/p_surf. v0_20case_means = "
                 "_nf_eval.json (first 20 sorted full_test only, 13k-row "
                 "subsampled training); this file = ALL 200 full_test, "
                 "full-res-trained v1."),
        cases=rows,
        eval_protocol="full volume, ALL mesh nodes (MARIO protocol); "
                      "standardized with _airfrans_v2_trainstats stds; p_surf "
                      "on surface nodes with the volume-p std; pred = "
                      "w(alpha)*PAINT + destd(net) for ux/uy/p, destd(net) "
                      "for nut; paint uses XFOIL-XTR cl, ensemble-head cl "
                      "where XFOIL fails (documented per case).")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, default=float)

    print(f"\n[eval200] {len(rows)} cases ({int(lo.sum())} alpha<10, "
          f"{int(hi.sum())} alpha>=10), {len(fallbacks)} cl fallbacks, "
          f"{(time.time()-t_all)/60:.1f} min total", flush=True)
    for tag, d in (("v1 all200", full_v1), ("v1 a<10", lo_v1),
                   ("v1 a>=10", hi_v1), ("paint all200", full_pa)):
        print(f"  {tag:12s} " + "  ".join(f"{k}={d[k]:.4e}" for k in KEYS),
              flush=True)
    if v0:
        for tag in ("residual", "scratch", "paint"):
            print(f"  v0-{tag:8s} " + "  ".join(f"{k}={v0[tag][k]:.4e}"
                                                for k in KEYS), flush=True)
    print("  mario-raw    " + "  ".join(f"{k}={MARIO_PUBLISHED[k]:.4e}"
                                        for k in KEYS), flush=True)
    print(f"[eval200] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
