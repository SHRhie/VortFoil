"""Where does the physics route (OP1 scr_bd128 + OP2 m325_BT + fitted_poly2) beat NeuralFoil xlarge, if anywhere?
Per-regime R2 for Cl, Cd, Cm on both evaluation sets, plus the trailing-edge BL state (theta, dstar, ue) of both
models against XFOIL. Measurement only; every number saved. Usage: nf_regime_search.py <out.json>"""
import json, os, sys
import numpy as np
from itertools import combinations_with_replacement as cwr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # <release>/code
import vortfoil_paths as VP; VP.add_paths()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import tensorflow as tf; tf.config.set_visible_devices([], "GPU")
from staged.op1_inviscid import build_op1
from staged.op2_bl_surrogate import build_op2_surrogate
from staged.common.geometry import build_single_loop_contour_cst_tf
from staged.common.panel import compute_inviscid_surface_velocity
R = VP.RELEASE_ROOT; W = VP.WEIGHTS + os.sep; DT = tf.float32
op1 = build_op1(n_surface_pts=121, branch_dim=128); op2 = build_op2_surrogate(n_surface_pts=121, enable_wake=False, branch_dim=128, branch_depth=3, trunk_depth=2, N_m=16)
c0 = tf.constant(np.zeros((2, 16), np.float32)); a0 = tf.constant(np.zeros((2, 1), np.float32)); r0 = tf.constant(np.full((2, 1), 1e6, np.float32))
op1.compute_losses(c0, a0, r0, training=False); op2.predict_bl(c0, a0, r0, training=False)
ck = np.load(W + "op1_supervised_eps0.50.scr_bd128.npz"); kk = sorted(x for x in ck.files if x.startswith("v_"))
op1.field_decoder.build((None, int(ck[kk[-6]].shape[0]))); op1.load_weights(W + "op1_supervised_eps0.50.scr_bd128.npz")
op2.load_weights(W + "op2_m325_BT.npz", strict=False)   # = op2_final_eps0.50_direct.m325_BT.npz
cc = np.load(os.path.join(VP.WEIGHTS, "cd_closure_poly2.npz")); mu, sd, w = cc["mu"], cc["sd"], cc["w"]
def poly2(t, H, u):
    F = np.c_[np.log(np.maximum(t, 1e-12)), H, u]; X = np.c_[tuple([np.ones(len(F))] + [F[:, list(k)].prod(1) for d in (1, 2) for k in cwr(range(3), d)])]
    return np.clip(10 ** (((X - mu) / sd) @ w), 1e-4, 0.1)
def physics(cst, al, re):
    N = len(cst); o = {k: np.zeros(N) for k in ("cl", "cm", "ue", "th", "ds")}
    for i in range(0, N, 512):
        c, a, r = tf.constant(cst[i:i+512]), tf.constant(al[i:i+512]), tf.constant(re[i:i+512])
        g = build_single_loop_contour_cst_tf(c, n_surface_pts=121); gg = op1.predict_gamma(c, a, Re=r, training=False)
        inv = compute_inviscid_surface_velocity(g, gg["gamma_nodes"], a); Ut = inv["Ut"]; cp = 1 - Ut ** 2
        xc, yc, ds, nx, ny = g["xc"], g["yc"], g["ds"], g["nx"], g["ny"]
        o["cl"][i:i+512] = (2.0 * tf.reduce_sum(gg["gamma_panels"] * ds, 1)).numpy()
        o["cm"][i:i+512] = tf.reduce_sum((-(xc - 0.25) * (-cp * ny) + yc * (-cp * nx)) * ds, 1).numpy()
        o["ue"][i:i+512] = 0.5 * (tf.abs(Ut[:, 0]) + tf.abs(Ut[:, -1])).numpy()
        bl = op2.predict_bl(c, a, r, training=False)
        o["th"][i:i+512] = (bl["theta_u"][:, -1] + bl["theta_l"][:, -1]).numpy(); o["ds"][i:i+512] = (bl["delta_u"][:, -1] + bl["delta_l"][:, -1]).numpy()
    o["cd"] = poly2(o["th"], o["ds"] / o["th"], o["ue"]); return o
r2 = lambda p, t: float(1 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2)) if len(t) > 10 else float("nan")
def gate(z):
    v = z["valid"] & (np.abs(z["cl_visc"][:, 0]) <= 3) & (z["cd_visc"][:, 0] > 1e-4) & (z["cd_visc"][:, 0] <= 0.1)
    for k in ["theta_u_te", "theta_l_te", "dstar_u_te", "dstar_l_te", "ue_te"]: v &= np.isfinite(z[k][:, 0]) & ((z[k][:, 0] != 0) if "theta" in k else True)
    return v
OUT = {}
# NeuralFoil prediction dumps (nf_full_*.npz, from run_nf_full.py) and the v1 target decks
# (attached_surface_targets.npz, sep_eval_targets.npz) are NOT shipped in this release; see README.
RAW = os.environ.get("VORTFOIL_RAW", VP.DATA_EVAL)
for name, tgt, nff, extra in (("attached", os.path.join(VP.DATA_EVAL, "holdout_targets.npz"), RAW + "/nf_full_attached.npz", RAW + "/attached_surface_targets.npz"),
                              ("separated", RAW + "/sep_eval_targets.npz", RAW + "/nf_full_separated.npz", None)):
    z = np.load(tgt, allow_pickle=True); v = gate(z); nf = np.load(nff)
    cst = z["cst"][v].astype(np.float32); al = z["alpha"][v].astype(np.float32); re = z["Re"][v].astype(np.float32)
    cl_t = z["cl_visc"][v, 0]; cd_t = z["cd_visc"][v, 0]; th_t = z["theta_u_te"][v, 0] + z["theta_l_te"][v, 0]; ds_t = z["dstar_u_te"][v, 0] + z["dstar_l_te"][v, 0]; ue_t = z["ue_te"][v, 0]
    H = ds_t / th_t; P = physics(cst, al, re); assert len(nf["CL"]) == len(cl_t)
    nf_cd = np.clip(nf["CD"], 1e-4, 0.1); nf_th = nf["theta_u"][:, -1] + nf["theta_l"][:, -1]; nf_ds = nf["H_u"][:, -1] * nf["theta_u"][:, -1] + nf["H_l"][:, -1] * nf["theta_l"][:, -1]; nf_ue = 0.5 * (nf["ue_u"][:, -1] + nf["ue_l"][:, -1])
    # Cm truth: attached from the Xeon surface file (match by name/alpha/Re); separated has cm_visc in the targets
    if extra:
        T = np.load(extra, allow_pickle=True); key = lambda n, a, r: f"{str(n)}|{float(a):.6g}|{float(r):.6g}"
        tk = {key(n, a, r): i for i, (n, a, r) in enumerate(zip(T["names"], T["alpha"][:, 0], T["Re"][:, 0]))}
        idx = np.array([tk.get(key(n, a, r), -1) for n, a, r in zip(z["names"][v], al[:, 0], re[:, 0])]); mcm = idx >= 0
        cm_t = np.where(mcm, T["cm_visc"][np.maximum(idx, 0), 0], np.nan); mcm &= np.isfinite(cm_t)
    elif "cm_visc" in z.files:
        cm_t = z["cm_visc"][v, 0]; mcm = np.isfinite(cm_t)
    else:
        cm_t = np.full(len(cl_t), np.nan); mcm = np.zeros(len(cl_t), bool)   # no Cm truth in this file
    regimes = {"all": np.ones(len(cl_t), bool), "alpha<0": al[:, 0] < 0, "0<=alpha<8": (al[:, 0] >= 0) & (al[:, 0] < 8), "alpha>=8": al[:, 0] >= 8,
               "Re<=5e5": re[:, 0] <= 5e5, "Re>=1e6": re[:, 0] >= 1e6, "H<2.5": H < 2.5, "2.5<=H<4": (H >= 2.5) & (H < 4), "4<=H<6": (H >= 4) & (H < 6), "H>=6": H >= 6,
               "Cd<0.010": cd_t < 0.01, "0.010<=Cd<0.020": (cd_t >= 0.01) & (cd_t < 0.02), "Cd>=0.020": cd_t >= 0.02}
    res = {}
    for rn, m in regimes.items():
        if m.sum() < 30: continue
        d = {"n": int(m.sum()),
             "cl_phys": r2(P["cl"][m], cl_t[m]), "cl_nf": r2(nf["CL"][m], cl_t[m]),
             "cd_phys": r2(P["cd"][m], cd_t[m]), "cd_nf": r2(nf_cd[m], cd_t[m]),
             "cm_phys": r2(P["cm"][m & mcm], cm_t[m & mcm]), "cm_nf": r2(nf["CM"][m & mcm], cm_t[m & mcm]), "n_cm": int((m & mcm).sum()),
             "theta_medrel_phys": float(100 * np.median(np.abs(P["th"][m] / th_t[m] - 1))), "theta_medrel_nf": float(100 * np.median(np.abs(nf_th[m] / th_t[m] - 1))),
             "dstar_medrel_phys": float(100 * np.median(np.abs(P["ds"][m] / ds_t[m] - 1))), "dstar_medrel_nf": float(100 * np.median(np.abs(nf_ds[m] / ds_t[m] - 1))),
             "ue_medrel_phys": float(100 * np.median(np.abs(P["ue"][m] / ue_t[m] - 1))), "ue_medrel_nf": float(100 * np.median(np.abs(nf_ue[m] / ue_t[m] - 1))),
             "theta_r2_phys": r2(P["th"][m], th_t[m]), "theta_r2_nf": r2(nf_th[m], th_t[m])}
        res[rn] = d
    OUT[name] = res
    print(f"\n=== {name} ===\n{'regime':16s} {'n':>5s} | {'Cl ph/nf':>15s} | {'Cd ph/nf':>15s} | {'Cm ph/nf':>15s} | {'θ% ph/nf':>13s} | {'δ*% ph/nf':>13s} | {'ue% ph/nf':>13s}")
    for rn, d in res.items():
        print(f"{rn:16s} {d['n']:5d} | {d['cl_phys']:7.4f}/{d['cl_nf']:7.4f} | {d['cd_phys']:7.4f}/{d['cd_nf']:7.4f} | {d['cm_phys']:7.4f}/{d['cm_nf']:7.4f} | {d['theta_medrel_phys']:5.2f}/{d['theta_medrel_nf']:6.2f} | {d['dstar_medrel_phys']:5.2f}/{d['dstar_medrel_nf']:6.2f} | {d['ue_medrel_phys']:5.2f}/{d['ue_medrel_nf']:6.2f}")
json.dump(OUT, open(sys.argv[1], "w"), indent=1)
