"""Surface-representation contract for the SINGLE confirmed model (OP1 scr_bd128, CST-16, 19-input),
on the CST-16 attached holdout (holdout_targets.npz, same gate as the drag route, n=2737).
Port of the research tree's gate1_tangency_percase.py's forward and metric definitions (Un RMS per unit
freestream; Cl_press = -oint Cp n.y_wind ds with Cp = 1-Ut^2; Cl_circ = 2 oint gamma ds; Cm full arm;
Kutta = |gamma_0 + gamma_Np|; LOOP_SIGN = +1 fixed by contour orientation, guarded not fitted).
Cp RMSE / stagnation are added by surface_cp_stag_cst16.py once the XFOIL Cp truth exists.
Usage: surface_contract_cst16.py <out.npz> <out.json>"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # <release>/code
import vortfoil_paths as VP; VP.add_paths()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import tensorflow as tf; tf.config.set_visible_devices([], "GPU")
from staged.op1_inviscid import build_op1
from staged.common.geometry import build_single_loop_contour_cst_tf
from staged.common.panel import compute_inviscid_surface_velocity
DT = tf.float32; PI = np.pi; W = os.path.join(VP.WEIGHTS, "op1_supervised_eps0.50.scr_bd128.npz")
z = np.load(os.path.join(VP.DATA_EVAL, "holdout_targets.npz"), allow_pickle=True)
v = z["valid"] & (np.abs(z["cl_visc"][:, 0]) <= 3) & (z["cd_visc"][:, 0] > 1e-4) & (z["cd_visc"][:, 0] <= 0.1)
for k in ["theta_u_te", "theta_l_te", "dstar_u_te", "dstar_l_te", "ue_te"]:
    v &= np.isfinite(z[k][:, 0]) & ((z[k][:, 0] != 0) if "theta" in k else True)
cst = z["cst"][v].astype(np.float32); al = z["alpha"][v].astype(np.float32); re = z["Re"][v].astype(np.float32)
cl_t = z["cl_visc"][v, 0].astype(np.float64); N = len(cst); print("gated n =", N)
op1 = build_op1(n_surface_pts=121, branch_dim=128)
op1.compute_losses(tf.constant(cst[:2]), tf.constant(al[:2]), tf.constant(re[:2]), training=False)
_ck = np.load(W); _k = sorted(x for x in _ck.files if x.startswith("v_"))
op1.field_decoder.build((None, int(_ck[_k[-6]].shape[0]))); op1.load_weights(W)
@tf.function(reduce_retracing=True)
def forward(c, a, r):
    geom = build_single_loop_contour_cst_tf(c, n_surface_pts=121)
    gg = op1.predict_gamma(c, a, Re=r, training=False)
    inv = compute_inviscid_surface_velocity(geom, gg["gamma_nodes"], a)
    ds, xc, yc, nx, ny = geom["ds"], geom["xc"], geom["yc"], geom["nx"], geom["ny"]
    Un, Ut = inv["Un"], inv["Ut"]; cp = 1.0 - Ut ** 2
    ca = tf.cos(a * PI / 180.0); sa = tf.sin(a * PI / 180.0)
    cl_circ = 2.0 * tf.reduce_sum(gg["gamma_panels"] * ds, axis=1, keepdims=True)
    arm = (0.25 - xc) * ca - yc * sa
    cm_circ = 2.0 * tf.reduce_sum(arm * gg["gamma_panels"] * ds, axis=1, keepdims=True)
    fx = -tf.reduce_sum(cp * nx * ds, axis=1, keepdims=True); fy = -tf.reduce_sum(cp * ny * ds, axis=1, keepdims=True)
    cl_press = -sa * fx + ca * fy
    cm_press = tf.reduce_sum((-(xc - 0.25) * (-cp * ny) + yc * (-cp * nx)) * ds, axis=1, keepdims=True)
    gn = gg["gamma_nodes"]
    return {"un_rms": tf.sqrt(tf.reduce_mean(Un ** 2, 1, keepdims=True)), "un_max": tf.reduce_max(tf.abs(Un), 1, keepdims=True),
            "ut_rms": tf.sqrt(tf.reduce_mean(Ut ** 2, 1, keepdims=True)), "cl_circ": cl_circ, "cm_circ": cm_circ,
            "cl_press": cl_press, "cm_press": cm_press, "kutta": tf.abs(gn[:, :1] + gn[:, -1:]),
            "cp": cp, "ut": Ut, "xc": xc, "ds": ds}
P = {}
for i in range(0, N, 512):
    o = forward(tf.constant(cst[i:i+512]), tf.constant(al[i:i+512]), tf.constant(re[i:i+512]))
    for k, val in o.items(): P.setdefault(k, []).append(np.asarray(val.numpy(), np.float64))
P = {k: np.concatenate(vv) for k, vv in P.items()}
for k in ("un_rms", "un_max", "ut_rms", "cl_circ", "cm_circ", "cl_press", "cm_press", "kutta"): P[k] = P[k].ravel()
r2 = lambda p, t: float(1 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2))
LOOP_SIGN = 1.0
for nm in ("cl_circ", "cl_press"):
    rp, rm = r2(LOOP_SIGN * P[nm], cl_t), r2(-LOOP_SIGN * P[nm], cl_t)
    assert rp >= rm, ("GUARD: loop-orientation sign is not the winning orientation for %s (%.6f < %.6f)" % (nm, rp, rm))
assert not np.allclose(P["cl_circ"], cl_t, atol=1e-6), "GUARD: prediction equals truth"
assert np.max(P["kutta"]) < 1e-4, "GUARD: Kutta sum not structural"; assert np.median(P["un_rms"]) > 1e-6, "GUARD: Un ~ 0"
gap_l = P["cl_circ"] - P["cl_press"]; gap_m = P["cm_circ"] - P["cm_press"]
rms_t = float(np.sqrt(np.mean(cl_t ** 2))); mabs_t = float(np.mean(np.abs(cl_t)))
S = {"model": "OP1 scr_bd128 (CST-16, 19-input) -- the confirmed single model", "weights_file": W, "n": int(N),
     "gate": "holdout_targets.npz, same gate as the drag route",
     "kutta_residual_max": float(np.max(P["kutta"])), "kutta_residual_median": float(np.median(P["kutta"])),
     "cl_route_gap_rmse": float(np.sqrt(np.mean(gap_l ** 2))), "cl_route_gap_pct_of_rms_truth": float(100 * np.sqrt(np.mean(gap_l ** 2)) / rms_t),
     "cl_route_gap_pct_of_mean_abs_truth": float(100 * np.sqrt(np.mean(gap_l ** 2)) / mabs_t),
     "cm_route_gap_rmse": float(np.sqrt(np.mean(gap_m ** 2))),
     "tangency_un_rms_median": float(np.median(P["un_rms"])), "tangency_un_rms_pct_of_ut_rms_median": float(100 * np.median(P["un_rms"] / P["ut_rms"])),
     "tangency_un_max_median": float(np.median(P["un_max"])),
     "cl_circ_r2": r2(P["cl_circ"], cl_t), "cl_press_r2": r2(P["cl_press"], cl_t),
     "cl_circ_rmse": float(np.sqrt(np.mean((P["cl_circ"] - cl_t) ** 2))), "cl_press_rmse": float(np.sqrt(np.mean((P["cl_press"] - cl_t) ** 2)))}
np.savez(sys.argv[1], cst=cst, alpha=al, Re=re, cl_truth=cl_t, **{k: P[k] for k in P})
json.dump(S, open(sys.argv[2], "w"), indent=1); print(json.dumps(S, indent=1))
