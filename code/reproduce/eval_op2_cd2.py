"""Evaluate the physics drag route for a given OP2 weight file, on the published clone holdout.

Mirrors the research tree's cdgate_deployed_state.py's gate, chain and wake march exactly. The only thing
that varies between calls is which OP2 checkpoint is loaded. Measurement only: no training, no
XFOIL, no deployed file written.
Usage: eval_op2_cd.py <op2_weight_name> <out_json> [op1_weight_name]"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # <release>/code
import vortfoil_paths as VP; VP.add_paths()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
from staged.common.geometry import build_single_loop_contour_cst_tf
from staged.common.panel import compute_inviscid_surface_velocity
from staged.common.wake_ode_tf import wake_ode_cd_tf
from staged.op1_inviscid import build_op1
from op2_dense_surrogate import build_op2_surrogate

OP2_W = sys.argv[1]; OUT = sys.argv[2]
OP1_W = sys.argv[3] if len(sys.argv) > 3 else "op1_supervised_eps0.50.scr_bd128.npz"
# argv[1] is the OP2 weight file name inside weights/, e.g. op2_m325_BT.npz
HEAD_MODE = sys.argv[4] if len(sys.argv) > 4 else "bt"
CLOSURE = sys.argv[5] if len(sys.argv) > 5 else "squire_young"
D = VP.DATA_EVAL + os.sep; W = VP.WEIGHTS + os.sep
DT = tf.float32; NSP = 121


def r2(p, t):
    p, t = np.asarray(p, np.float64), np.asarray(t, np.float64)
    return float(1 - np.sum((p - t) ** 2) / max(np.sum((t - t.mean()) ** 2), 1e-30))


def r2log(p, t):
    return r2(np.log10(np.clip(p, 1e-9, None)), np.log10(np.clip(t, 1e-9, None)))


z = np.load(D + "holdout_targets.npz", allow_pickle=True)
v = z["valid"] & (np.abs(z["cl_visc"][:, 0]) <= 3) & (z["cd_visc"][:, 0] > 1e-4) & (z["cd_visc"][:, 0] <= 0.1)
for k in ["theta_u_te", "theta_l_te", "dstar_u_te", "dstar_l_te", "ue_te"]:
    v &= np.isfinite(z[k][:, 0]) & ((z[k][:, 0] != 0) if "theta" in k else True)
cst = z["cst"][v].astype(np.float32); al = z["alpha"][v]; re = z["Re"][v]
cd_t = z["cd_visc"][v, 0]; cl_t = z["cl_visc"][v, 0]
N = len(cst)

op1 = build_op1(n_surface_pts=NSP, branch_dim=128)
op2 = build_op2_surrogate(n_surface_pts=NSP, enable_wake=False, branch_dim=128,
                          branch_depth=3, trunk_depth=2, N_m=16, head_mode=HEAD_MODE)
op1.compute_losses(tf.constant(cst[:2], DT), tf.constant(al[:2], DT), tf.constant(re[:2], DT), training=False)
op2.predict_bl(tf.constant(cst[:2], DT), tf.constant(al[:2], DT), tf.constant(re[:2], DT), training=False)
# OP1's field_decoder is a Sequential with no input_shape and compute_losses never calls it,
# so trainable_variables_list() raises before load_weights can run. Build it from the checkpoint's
# own field_h0 kernel width. This changes no math: the decoder is not used by the drag route.
_ck = np.load(W + OP1_W)
_keys = sorted(k for k in _ck.files if k.startswith('v_'))
_fin = int(_ck[_keys[-6]].shape[0])          # field_h0 kernel is (F_in, 64), 6th from the end
op1.field_decoder.build((None, _fin))
op1.load_weights(W + OP1_W)
op2.load_weights((OP2_W if os.path.isabs(OP2_W) else W + OP2_W), strict=False)

thu = np.zeros(N); thl = np.zeros(N); dsu = np.zeros(N); dsl = np.zeros(N); ue_p = np.zeros(N); cl_p = np.zeros(N)
for s in range(0, N, 1024):
    e = min(s + 1024, N)
    c = tf.constant(cst[s:e], DT); a = tf.constant(al[s:e], DT); r = tf.constant(re[s:e], DT)
    geom = build_single_loop_contour_cst_tf(c, n_surface_pts=NSP)
    g = op1.predict_gamma(c, a, Re=r, training=False)
    Ut = compute_inviscid_surface_velocity(geom, g["gamma_nodes"], a)["Ut"]
    ue_p[s:e] = 0.5 * (tf.abs(Ut[:, 0]) + tf.abs(Ut[:, -1])).numpy()
    cl_p[s:e] = (2.0 * tf.reduce_sum(g["gamma_panels"] * geom["ds"], axis=1)).numpy()
    bl = op2.predict_bl(c, a, r, training=False)
    thu[s:e] = bl["theta_u"][:, -1].numpy(); thl[s:e] = bl["theta_l"][:, -1].numpy()
    dsu[s:e] = bl["delta_u"][:, -1].numpy(); dsl[s:e] = bl["delta_l"][:, -1].numpy()

th_p = thu + thl; ds_p = dsu + dsl
cd_p = wake_ode_cd_tf(tf.constant(th_p, DT), tf.constant(ds_p, DT), tf.constant(ue_p, DT),
                      closure=CLOSURE).numpy()
cd_c = np.clip(cd_p, 1e-4, 0.1)
# per-state relative errors against the holdout's own truth TE state
th_t = z["theta_u_te"][v, 0] + z["theta_l_te"][v, 0]
ds_t = z["dstar_u_te"][v, 0] + z["dstar_l_te"][v, 0]
ue_t = z["ue_te"][v, 0]
rel = lambda p, t: float(np.median(np.abs(p / np.maximum(t, 1e-12) - 1)))      # noqa: E731
R = {"op2_weights": OP2_W, "head_mode": HEAD_MODE, "closure": CLOSURE, "op1_weights": OP1_W, "n": int(N),
     "cd_raw_r2": r2(cd_c, cd_t), "cd_log_r2": r2log(cd_c, cd_t),
     "cd_medape_pct": float(100 * np.median(np.abs(cd_c / cd_t - 1))),
     "nonfinite_cd_pred": int((~np.isfinite(cd_p)).sum()),
     "clamped_rows": int(((cd_p < 1e-4) | (cd_p > 0.1)).sum()),
     "te_state_median_rel_err": {"theta_TE": rel(th_p, th_t), "dstar_TE": rel(ds_p, ds_t), "ue_TE": rel(ue_p, ue_t)},
     "te_state_r2": {"theta_TE": r2(th_p, th_t), "dstar_TE": r2(ds_p, ds_t), "ue_TE": r2(ue_p, ue_t)},
     "cl_raw_r2": r2(cl_p, cl_t), "cl_mae": float(np.mean(np.abs(cl_p - cl_t))),
     "by_alpha_band": {b: {"n": int(m.sum()), "cd_raw_r2": (r2(cd_c[m], cd_t[m]) if m.sum() > 10 else None)}
                       for b, m in (("a<0", al[:, 0] < 0), ("0<=a<8", (al[:, 0] >= 0) & (al[:, 0] < 8)),
                                    ("a>=8", al[:, 0] >= 8))},
     "gate": "holdout_targets.npz, the exact gate of max_phys_lever_a_clip.py / cdgate_deployed_state.py"}
json.dump(R, open(OUT, "w"), indent=1)
print(json.dumps({k: R[k] for k in ("op2_weights", "n", "cd_raw_r2", "cd_log_r2", "cd_medape_pct",
                                    "te_state_median_rel_err")}, indent=1))
