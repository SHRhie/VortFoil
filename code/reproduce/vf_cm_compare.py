# -*- coding: utf-8 -*-
"""Moment coefficient of the deployed sheet (pressure-integral readout about the quarter chord) against XFOIL and
NeuralFoil xlarge on the SAME rows as the paper's coefficient tables (attached 2,737 / separated 1,930), plus the
finite-moment subsets the surface-results text uses.  2026-09-20.  Run: /opt/anaconda3/envs/tf210/bin/python VortFoil/code/vf_cm_compare.py"""
import os, sys, json, datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..")); CLONE = os.path.join(REPO, ".neurofoil_clone"); sys.path.insert(0, CLONE)
import tensorflow as tf; tf.config.set_visible_devices([], "GPU")
from staged.op1_inviscid import build_op1
from staged.common.geometry import build_single_loop_contour_cst_tf
from staged.common.panel import compute_inviscid_surface_velocity
A = os.path.join(REPO, "VortFoil_v2/artifacts/op2_cdloss_20260917T044259Z/raw"); W1 = os.path.join(CLONE, "staged/weights/op1_supervised_eps0.50.scr_bd128.npz")
op1 = build_op1(n_surface_pts=121, branch_dim=128); z2 = np.zeros((2, 16), np.float32)
op1.compute_losses(tf.constant(z2), tf.constant(np.zeros((2, 1), np.float32)), tf.constant(np.full((2, 1), 1e6, np.float32)), training=False)
ck = np.load(W1); k = sorted(x for x in ck.files if x.startswith("v_")); op1.field_decoder.build((None, int(ck[k[-6]].shape[0]))); op1.load_weights(W1)

def gate(z):
    v = z["valid"] & (np.abs(z["cl_visc"][:, 0]) <= 3) & (z["cd_visc"][:, 0] > 1e-4) & (z["cd_visc"][:, 0] <= 0.1)
    for kk in ["theta_u_te", "theta_l_te", "dstar_u_te", "dstar_l_te", "ue_te"]: v &= np.isfinite(z[kk][:, 0]) & ((z[kk][:, 0] != 0) if "theta" in kk else True)
    return v
def cm_press(cst, al, re):
    out = []
    for i in range(0, len(cst), 512):
        c, a, r = tf.constant(cst[i:i+512]), tf.constant(al[i:i+512]), tf.constant(re[i:i+512])
        g = build_single_loop_contour_cst_tf(c, n_surface_pts=121); gg = op1.predict_gamma(c, a, Re=r, training=False)
        cp = 1 - compute_inviscid_surface_velocity(g, gg["gamma_nodes"], a)["Ut"] ** 2
        out.append(tf.reduce_sum((-(g["xc"] - 0.25) * (-cp * g["ny"]) + g["yc"] * (-cp * g["nx"])) * g["ds"], 1).numpy())
    return np.concatenate(out).astype(np.float64)
r2 = lambda p, t: float(1 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2))
res = dict(schema="CM_COMPARE_V1", utc=datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"), readout="pressure-integral moment about x/c=0.25 from the sheet's panel Cp (cm_press of the contract script)", sets={})
# ---- attached: gate on holdout_targets (2737); truth Cm from the Xeon surface file v2 matched by name/alpha/Re; NF from nf_full_attached.npz (aligned to the gate)
z = np.load(os.path.join(CLONE, "staged/data/holdout_targets.npz"), allow_pickle=True); v = gate(z); nf = np.load(os.path.join(A, "nf_full_attached.npz")); assert len(nf["CM"]) == v.sum() == 2737
T = np.load(os.path.join(A, "attached_surface_targets_v2.npz"), allow_pickle=True); key = lambda n, a, r: f"{str(n)}|{float(a):.6g}|{float(r):.6g}"
tk = {key(n, a, r): i for i, (n, a, r) in enumerate(zip(T["names"], T["alpha"][:, 0], T["Re"][:, 0]))}
al = z["alpha"][v].astype(np.float32); re = z["Re"][v].astype(np.float32); idx = np.array([tk.get(key(n, a, r), -1) for n, a, r in zip(z["names"][v], al[:, 0], re[:, 0])])
assert (idx >= 0).all(); cm_t = T["cm_visc"][idx, 0].astype(np.float64); fin = np.isfinite(cm_t); vf_ok = ~T["viscal_failed"][idx].astype(bool)
P = cm_press(z["cst"][v].astype(np.float32), al, re); assert not np.allclose(P[fin], cm_t[fin])
bands = {"all": np.ones(len(al), bool), "alpha<0": al[:, 0] < 0, "0<=alpha<8": (al[:, 0] >= 0) & (al[:, 0] < 8), "alpha>=8": al[:, 0] >= 8}
def block(P, N, T_, masks):
    return {name: dict(n=int(m.sum()), cm_phys=r2(P[m], T_[m]), cm_nf=r2(N[m], T_[m]), rmse_phys=float(np.sqrt(np.mean((P[m]-T_[m])**2))), rmse_nf=float(np.sqrt(np.mean((N[m]-T_[m])**2)))) for name, m in masks.items() if m.sum() > 10}
res["sets"]["attached_gated"] = dict(note="same 2,737 rows as the coefficient tables; rows with non-finite XFOIL Cm dropped", **block(P, nf["CM"], cm_t, {k2: m & fin for k2, m in bands.items()}))
res["sets"]["attached_finite_converged"] = dict(note="additionally excluding rows XFOIL flags as viscal_failed (global Newton not converged)", **block(P, nf["CM"], cm_t, {k2: m & fin & vf_ok for k2, m in bands.items()}))
print("attached n_fin", int(fin.sum()), "n_fin&conv", int((fin & vf_ok).sum()), "| gated all:", {k2: round(res['sets']['attached_gated'][k2][x], 4) for k2 in ("all",) for x in ("cm_phys", "cm_nf")}, "| converged:", {x: round(res['sets']['attached_finite_converged']['all'][x], 4) for x in ("cm_phys", "cm_nf")})
# ---- separated: gate on sep_eval_targets_v2 (1930); NF from nf_full_separated.npz (aligned to the v1 gate: verify identical)
zs = np.load(os.path.join(A, "sep_eval_targets_v2.npz"), allow_pickle=True); zs1 = np.load(os.path.join(A, "sep_eval_targets.npz"), allow_pickle=True); vs = gate(zs); vs1 = gate(zs1)
print("sep gate v2", int(vs.sum()), "v1", int(vs1.sum()), "identical:", bool((vs == vs1).all())); nfs = np.load(os.path.join(A, "nf_full_separated.npz")); assert len(nfs["CM"]) == vs1.sum() == 1930
als = zs["alpha"][vs].astype(np.float32); cms_t = zs["cm_visc"][vs, 0].astype(np.float64); fins = np.isfinite(cms_t); vfs_ok = ~zs["viscal_failed"][vs].astype(bool)
Ps = cm_press(zs["cst"][vs].astype(np.float32), als, zs["Re"][vs].astype(np.float32)); assert not np.allclose(Ps[fins], cms_t[fins])
bs = {"all": np.ones(len(als), bool), "alpha<0": als[:, 0] < 0, "alpha>=8": als[:, 0] >= 8}
res["sets"]["separated_gated"] = dict(note="same 1,930 rows as the coefficient tables; non-finite XFOIL Cm dropped", **block(Ps, nfs["CM"], cms_t, {k2: m & fins for k2, m in bs.items()}))
res["sets"]["separated_finite_converged"] = dict(note="additionally excluding viscal_failed rows", **block(Ps, nfs["CM"], cms_t, {k2: m & fins & vfs_ok for k2, m in bs.items()}))
print("separated n_fin", int(fins.sum()), "n_fin&conv", int((fins & vfs_ok).sum()), "| gated all:", {x: round(res['sets']['separated_gated']['all'][x], 4) for x in ("cm_phys", "cm_nf")}, "| converged:", {x: round(res['sets']['separated_finite_converged']['all'][x], 4) for x in ("cm_phys", "cm_nf")})
for s_ in ("attached_gated", "attached_finite_converged"): print(s_, {b: (v_['n'], round(v_['cm_phys'], 3), round(v_['cm_nf'], 3)) for b, v_ in res["sets"][s_].items() if b != "note"})
fp = os.path.join(REPO, "VortFoil/tables", f"cm_compare_{res['utc']}.json"); json.dump(res, open(fp, "w"), indent=1); print("wrote", fp)
