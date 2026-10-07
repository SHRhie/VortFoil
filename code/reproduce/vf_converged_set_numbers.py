# -*- coding: utf-8 -*-
"""Every attached-set number the manuscript quotes, recomputed on the CONVERGED subset (2026-09-20).
Converged = XFOIL's global Newton iteration converged (viscal_failed false) and the moment is finite; the
same rows for every model, because the flag is a property of the truth label, not of a prediction.  The
separated set has no unconverged rows, so its numbers are unchanged and are recomputed here for one file.
Definitions are ports of the sealed surface_contract_cst16.py / surface_cp_stag_cst16.py / eval_op2_cd2.py.
  /opt/anaconda3/envs/tf210/bin/python VortFoil/code/vf_converged_set_numbers.py
"""
import os, sys, json, datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); CLONE = os.path.join(REPO, ".neurofoil_clone"); sys.path.insert(0, CLONE)
import tensorflow as tf; tf.config.set_visible_devices([], "GPU")
from staged.op1_inviscid import build_op1
from staged.op2_bl_surrogate import build_op2_surrogate
from staged.common.geometry import build_single_loop_contour_cst_tf
from staged.common.panel import compute_inviscid_surface_velocity
from staged.common.wake_ode_tf import wake_ode_cd_tf
from staged.common.lift_closure_tf import cl_closed_tf, load_lift_closure
A = os.path.join(REPO, "VortFoil_v2/artifacts/op2_cdloss_20260917T044259Z/raw")
W1 = os.path.join(CLONE, "staged/weights/op1_supervised_eps0.50.scr_bd128.npz"); W2 = os.path.join(REPO, "VortFoil_v2/artifacts/op2_cdloss_20260917T044259Z/weights/CANDIDATE_op2_m325_BT.npz")
LIFT = load_lift_closure(os.path.join(REPO, "VortFoil_v2/artifacts/lift_closure_20260920/weights/lift_closure_poly2.npz"))
op1 = build_op1(n_surface_pts=121, branch_dim=128); op2 = build_op2_surrogate(n_surface_pts=121, enable_wake=False, branch_dim=128, branch_depth=3, trunk_depth=2, N_m=16)
z2 = np.zeros((2, 16), np.float32); a2 = np.zeros((2, 1), np.float32); r2_ = np.full((2, 1), 1e6, np.float32)
op1.compute_losses(tf.constant(z2), tf.constant(a2), tf.constant(r2_), training=False); op2.predict_bl(tf.constant(z2), tf.constant(a2), tf.constant(r2_), training=False)
ck = np.load(W1); kk = sorted(x for x in ck.files if x.startswith("v_")); op1.field_decoder.build((None, int(ck[kk[-6]].shape[0]))); op1.load_weights(W1); op2.load_weights(W2, strict=False)
PI = np.pi; r2f = lambda p, t: float(1 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2))


@tf.function(reduce_retracing=True)
def _fwd(c, a, r):
    """graph mode, exactly as the sealed surface_contract_cst16.py: the Kutta cancellation is
    execution-order sensitive at float32 epsilon, so the sealed script's mode is reproduced here."""
    g = build_single_loop_contour_cst_tf(c, n_surface_pts=121); gg = op1.predict_gamma(c, a, Re=r, training=False)
    inv = compute_inviscid_surface_velocity(g, gg["gamma_nodes"], a); Ut, Un = inv["Ut"], inv["Un"]; cp = 1.0 - Ut ** 2
    ca, sa = tf.cos(a * PI / 180.0), tf.sin(a * PI / 180.0); xc, yc, ds, nx, ny = g["xc"], g["yc"], g["ds"], g["nx"], g["ny"]
    cl_circ = 2.0 * tf.reduce_sum(gg["gamma_panels"] * ds, 1); ue = 0.5 * (tf.abs(Ut[:, 0]) + tf.abs(Ut[:, -1]))
    fx = -tf.reduce_sum(cp * nx * ds, 1); fy = -tf.reduce_sum(cp * ny * ds, 1)
    bl = op2.predict_bl(c, a, r, training=False); th = bl["theta_u"][:, -1] + bl["theta_l"][:, -1]; dsx = bl["delta_u"][:, -1] + bl["delta_l"][:, -1]
    gn = gg["gamma_nodes"]
    return dict(cp=cp, ut=Ut, un_rms=tf.sqrt(tf.reduce_mean(Un ** 2, 1)), ut_rms=tf.sqrt(tf.reduce_mean(Ut ** 2, 1)),
                kutta=tf.abs(gn[:, :1] + gn[:, -1:])[:, 0], cl_circ=cl_circ,
                cl_press=-sa[:, 0] * fx + ca[:, 0] * fy, cm_circ=2.0 * tf.reduce_sum(((0.25 - xc) * ca - yc * sa) * gg["gamma_panels"] * ds, 1),
                cm_press=tf.reduce_sum((-(xc - 0.25) * (-cp * ny) + yc * (-cp * nx)) * ds, 1), theta_te=th, dstar_te=dsx, ue_te=ue,
                cd_fitted=wake_ode_cd_tf(th, dsx, ue, closure="fitted_poly2"), cd_legacy=wake_ode_cd_tf(th, dsx, ue, closure="legacy_ode"),
                cd_sy=wake_ode_cd_tf(th, dsx, ue, closure="squire_young"),
                cl_deployed=cl_closed_tf(cl_circ, bl["theta_u"][:, -1], bl["theta_l"][:, -1], bl["delta_u"][:, -1], bl["delta_l"][:, -1], ue, LIFT))


def forward(cst, al, re, B=512):
    """all sheet readouts + boundary-layer trailing-edge state + the three drag closures + the deployed lift"""
    o = {}
    for i in range(0, len(cst), B):
        d = _fwd(tf.constant(cst[i:i+B]), tf.constant(al[i:i+B]), tf.constant(re[i:i+B]))
        for k, v in d.items(): o.setdefault(k, []).append(np.asarray(v.numpy(), np.float64))
    return {k: np.concatenate(v) for k, v in o.items()}


def gate(z):
    v = z["valid"] & (np.abs(z["cl_visc"][:, 0]) <= 3) & (z["cd_visc"][:, 0] > 1e-4) & (z["cd_visc"][:, 0] <= 0.1)
    for k in ["theta_u_te", "theta_l_te", "dstar_u_te", "dstar_l_te", "ue_te"]: v &= np.isfinite(z[k][:, 0]) & ((z[k][:, 0] != 0) if "theta" in k else True)
    return v


def crossing(ut1, sp):
    idx = np.flatnonzero(np.diff(np.sign(ut1)) != 0)
    if idx.size == 0: return np.nan, 0
    j = idx[np.argmin(np.abs(sp[idx] - 0.5))]; u0, u1 = ut1[j], ut1[j + 1]; t = 0.0 if u1 == u0 else -u0 / (u1 - u0)
    return sp[j] + t * (sp[j + 1] - sp[j]), int(idx.size)


def contract(P, cl_t, m):
    gl = (P["cl_circ"] - P["cl_press"])[m]; gm = (P["cm_circ"] - P["cm_press"])[m]; t = cl_t[m]
    rms_t = float(np.sqrt(np.mean(t ** 2))); mabs = float(np.mean(np.abs(t)))
    return dict(n=int(m.sum()), kutta_residual_max=float(P["kutta"][m].max()), kutta_residual_median=float(np.median(P["kutta"][m])),
                cl_route_gap_rmse=float(np.sqrt(np.mean(gl ** 2))), cl_route_gap_pct_of_rms_truth=float(100 * np.sqrt(np.mean(gl ** 2)) / rms_t),
                cl_route_gap_pct_of_mean_abs_truth=float(100 * np.sqrt(np.mean(gl ** 2)) / mabs), cm_route_gap_rmse=float(np.sqrt(np.mean(gm ** 2))),
                tangency_un_rms_median=float(np.median(P["un_rms"][m])), tangency_un_rms_pct_of_ut_rms_median=float(100 * np.median(P["un_rms"][m] / P["ut_rms"][m])),
                cl_circ_r2=r2f(P["cl_circ"][m], t), cl_press_r2=r2f(P["cl_press"][m], t), cl_deployed_r2=r2f(P["cl_deployed"][m], t))


def drag_block(P, cd_t, m):
    out = {}
    for nm, key in (("fitted", "cd_fitted"), ("marched", "cd_legacy"), ("squire_young", "cd_sy")):
        p = np.clip(P[key][m], 1e-4, 0.1); t = cd_t[m]
        out[nm] = dict(cd_raw_r2=r2f(p, t), cd_log_r2=r2f(np.log10(p), np.log10(t)), cd_medape_pct=float(100 * np.median(np.abs(p / t - 1))))
    return out


res = dict(schema="CONVERGED_SET_NUMBERS_V1", utc=datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
           definition="attached set restricted to rows whose XFOIL solution converged (viscal_failed false) and has a finite moment; identical rows for every model compared; the separated set has no unconverged rows",
           weights=dict(op1=os.path.basename(W1), op2=os.path.basename(W2)), sets={})
# ---------------- attached ----------------
z = np.load(os.path.join(CLONE, "staged/data/holdout_targets.npz"), allow_pickle=True); v = gate(z)
T = np.load(os.path.join(A, "attached_surface_targets_v2.npz"), allow_pickle=True); key = lambda n, a, r: f"{str(n)}|{float(a):.6g}|{float(r):.6g}"
tk = {key(n, a, r): i for i, (n, a, r) in enumerate(zip(T["names"], T["alpha"][:, 0], T["Re"][:, 0]))}
al = z["alpha"][v].astype(np.float32); re = z["Re"][v].astype(np.float32); idx = np.array([tk[key(n, a, r)] for n, a, r in zip(z["names"][v], al[:, 0], re[:, 0])])
conv = (~T["viscal_failed"][idx].astype(bool)) & np.isfinite(T["cm_visc"][idx, 0])
cl_t = z["cl_visc"][v, 0].astype(np.float64); cd_t = z["cd_visc"][v, 0].astype(np.float64); cm_t = T["cm_visc"][idx, 0].astype(np.float64)
cp_t = T["cp_panels"][idx].astype(np.float64); s_t = T["s_stag"][idx, 0].astype(np.float64); sp = T["s_panels"].astype(np.float64)
P = forward(z["cst"][v].astype(np.float32), al, re); nf = np.load(os.path.join(A, "nf_full_attached.npz")); assert len(nf["CL"]) == len(cl_t) == 2737
assert not np.allclose(P["cl_circ"], cl_t) and not np.allclose(P["cp"], cp_t) and P["kutta"].max() < 1e-4
cr = np.array([crossing(u, sp) for u in P["ut"]]); s_p, ncross = cr[:, 0], cr[:, 1].astype(int); err_panels = np.abs(s_p - s_t) * 240.0
cp_ok = np.linalg.norm(cp_t, axis=1) > 0; assert (cp_ok == conv).all(), "the rows carrying an XFOIL Cp are exactly the converged rows"
rel_l2 = np.linalg.norm(P["cp"] - cp_t, axis=1) / np.maximum(np.linalg.norm(cp_t, axis=1), 1e-30); rmse_case = np.sqrt(np.mean((P["cp"] - cp_t) ** 2, 1))
nfcd = np.clip(nf["CD"], 1e-4, 0.1)
bands = {"all": np.ones(len(al), bool), "alpha<0": al[:, 0] < 0, "0<=alpha<8": (al[:, 0] >= 0) & (al[:, 0] < 8), "alpha>=8": al[:, 0] >= 8}
for label, base in (("attached_converged", conv), ("attached_all_gated", np.ones(len(al), bool))):
    surf_base = base & cp_ok   # Cp/stagnation only where XFOIL returned a surface solution
    b = {}
    for bn, bm in bands.items():
        m = bm & base
        b[bn] = dict(n=int(m.sum()), cl_deployed=r2f(P["cl_deployed"][m], cl_t[m]), cl_gamma=r2f(P["cl_circ"][m], cl_t[m]), cl_nf=r2f(nf["CL"][m], cl_t[m]),
                     cd=r2f(np.clip(P["cd_fitted"][m], 1e-4, 0.1), cd_t[m]), cd_nf=r2f(nfcd[m], cd_t[m]),
                     cm_press=r2f(P["cm_press"][m], cm_t[m]), cm_circ=r2f(P["cm_circ"][m], cm_t[m]), cm_nf=r2f(nf["CM"][m], cm_t[m]))
    m = base
    res["sets"][label] = dict(bands=b, contract=contract(P, cl_t, m), drag_closures=drag_block(P, cd_t, m),
                              surface=dict(n=int(surf_base.sum()), cp_rel_l2_median=float(np.median(rel_l2[surf_base])), cp_rel_l2_p90=float(np.percentile(rel_l2[surf_base], 90)),
                                           cp_rmse_case_median=float(np.median(rmse_case[surf_base])), stag_unique_crossing_rate=float(np.mean(ncross[surf_base] == 1)),
                                           stag_zero_crossing_rate=float(np.mean(ncross[surf_base] == 0)), stag_multi_crossing_rate=float(np.mean(ncross[surf_base] > 1)),
                                           stag_err_panels_median=float(np.nanmedian(err_panels[surf_base])), stag_within_one_panel=float(np.nanmean(err_panels[surf_base] < 1.0)),
                                           cm_press_r2=r2f(P["cm_press"][surf_base], cm_t[surf_base]), cm_circ_r2=r2f(P["cm_circ"][surf_base], cm_t[surf_base])))
res["attached_unconverged"] = dict(total=int((~conv).sum()), by_band={bn: int(((~conv) & bm).sum()) for bn, bm in bands.items()})
# ---------------- separated (no unconverged rows) ----------------
zs = np.load(os.path.join(A, "sep_eval_targets_v2.npz"), allow_pickle=True); vs = gate(zs); als = zs["alpha"][vs].astype(np.float32)
convs = (~zs["viscal_failed"][vs].astype(bool)) & np.isfinite(zs["cm_visc"][vs, 0])
Ps = forward(zs["cst"][vs].astype(np.float32), als, zs["Re"][vs].astype(np.float32)); nfs = np.load(os.path.join(A, "nf_full_separated.npz")); assert len(nfs["CL"]) == int(vs.sum()) == 1930
cls_t = zs["cl_visc"][vs, 0].astype(np.float64); cds_t = zs["cd_visc"][vs, 0].astype(np.float64); cms_t = zs["cm_visc"][vs, 0].astype(np.float64)
H = (zs["dstar_u_te"][vs, 0] + zs["dstar_l_te"][vs, 0]) / (zs["theta_u_te"][vs, 0] + zs["theta_l_te"][vs, 0]); nfscd = np.clip(nfs["CD"], 1e-4, 0.1)
bs = {"all": np.ones(len(als), bool), "H<6": H < 6, "6<=H<10": (H >= 6) & (H < 10), "H>=10": H >= 10}
res["sets"]["separated"] = dict(n_unconverged=int((~convs).sum()),
    bands={bn: dict(n=int(m.sum()), cl_deployed=r2f(Ps["cl_deployed"][m], cls_t[m]), cl_gamma=r2f(Ps["cl_circ"][m], cls_t[m]), cl_nf=r2f(nfs["CL"][m], cls_t[m]),
                    cd=r2f(np.clip(Ps["cd_fitted"][m], 1e-4, 0.1), cds_t[m]), cd_nf=r2f(nfscd[m], cds_t[m]),
                    cm_press=r2f(Ps["cm_press"][m], cms_t[m]), cm_nf=r2f(nfs["CM"][m], cms_t[m])) for bn, m in bs.items()},
    drag_closures=drag_block(Ps, cds_t, np.ones(len(als), bool)),
    drag_closures_H10=drag_block(Ps, cds_t, H >= 10))
for lab in ("attached_converged", "attached_all_gated"):
    d = res["sets"][lab]; print(f"--- {lab} (n={d['contract']['n']})")
    for bn, b in d["bands"].items(): print(f"   {bn:11s} n={b['n']:5d} Cl {b['cl_deployed']:.4f}/{b['cl_nf']:.4f}  Cd {b['cd']:.4f}/{b['cd_nf']:.4f}  Cm {b['cm_press']:.4f}/{b['cm_nf']:.4f}")
    print("   contract:", {k: round(vv, 5) for k, vv in d["contract"].items() if k != "n"})
    print("   surface:", {k: round(vv, 5) for k, vv in d["surface"].items() if k != "n"})
    print("   closures:", {k: {k2: round(v2, 4) for k2, v2 in vv.items()} for k, vv in d["drag_closures"].items()})
print("--- separated"); [print(f"   {bn:9s} n={b['n']:5d} Cl {b['cl_deployed']:.4f}/{b['cl_nf']:.4f}  Cd {b['cd']:.4f}/{b['cd_nf']:.4f}  Cm {b['cm_press']:.4f}/{b['cm_nf']:.4f}") for bn, b in res["sets"]["separated"]["bands"].items()]
print("   closures:", {k: {k2: round(v2, 4) for k2, v2 in vv.items()} for k, vv in res["sets"]["separated"]["drag_closures"].items()}); print("   H>=10:", {k: round(vv["cd_raw_r2"], 4) for k, vv in res["sets"]["separated"]["drag_closures_H10"].items()})
print("unconverged:", res["attached_unconverged"])
fp = os.path.join(REPO, "VortFoil/tables", f"converged_set_numbers_{res['utc']}.json"); json.dump(res, open(fp, "w"), indent=1); print("wrote", fp)
