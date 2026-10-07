# -*- coding: utf-8 -*-
"""Lift decambering closure test (2026-09-20): can a closure in the predicted TRAILING-EDGE STATE alone close the gap
between the sheet's circulation lift and XFOIL's viscous lift?  Same discipline as the drag closure: features are
predicted state variables only (no shape, incidence or Reynolds number), ridge-fitted on TRAINING-deck rows, frozen,
evaluated on the attached holdout (2,737 gated rows) and the separated set (1,930 gated rows).
    dCl = w . phi(cl_circ, theta_u,TE, theta_l,TE, delta*_u,TE, delta*_l,TE, u_e,TE)      (degree-2 polynomial, standardized)
Also a physics-minimal variant dCl = a (delta*_u - delta*_l) + b (asymmetric TE displacement = effective decambering).
Distinct from the 2026-07-10 gamma_bl probe (which added a boundary-layer circulation to the sheet on an earlier OP1).
  /opt/anaconda3/envs/tf210/bin/python VortFoil/code/vf_cl_decamber_closure.py [--ntrain 40000]
"""
import os, sys, json, datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CLONE = os.path.join(REPO, ".neurofoil_clone"); sys.path.insert(0, CLONE)
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
from staged.op1_inviscid import build_op1
from staged.op2_bl_surrogate import build_op2_surrogate
from staged.common.geometry import build_single_loop_contour_cst_tf
from staged.common.panel import compute_inviscid_surface_velocity
from staged.common.lift_closure_tf import poly2_features_tf, cl_closed_tf, load_lift_closure, FEATURES
W1 = os.path.join(CLONE, "staged/weights/op1_supervised_eps0.50.scr_bd128.npz")
W2 = os.path.join(REPO, "VortFoil_v2/artifacts/op2_cdloss_20260917T044259Z/weights/CANDIDATE_op2_m325_BT.npz")
ART = os.path.join(REPO, "VortFoil_v2/artifacts/op2_cdloss_20260917T044259Z")
NSP = 121; DT = tf.float32
ntrain = sys.argv[sys.argv.index("--ntrain") + 1] if "--ntrain" in sys.argv else "40000"; ntrain = None if ntrain == "all" else int(ntrain)
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(REPO, "VortFoil/tables"); os.makedirs(OUT, exist_ok=True)

op1 = build_op1(n_surface_pts=NSP, branch_dim=128); op2 = build_op2_surrogate(n_surface_pts=NSP, enable_wake=False, branch_dim=128, branch_depth=3, trunk_depth=2, N_m=16)
z2 = np.zeros((2, 16), np.float32); a2 = np.zeros((2, 1), np.float32); r2_ = np.full((2, 1), 1e6, np.float32)
op1.compute_losses(tf.constant(z2), tf.constant(a2), tf.constant(r2_), training=False); op2.predict_bl(tf.constant(z2), tf.constant(a2), tf.constant(r2_), training=False)
ck = np.load(W1); k = sorted(x for x in ck.files if x.startswith("v_")); op1.field_decoder.build((None, int(ck[k[-6]].shape[0]))); op1.load_weights(W1); op2.load_weights(W2, strict=False)


def state(cst, al, re, B=1024):
    """predicted state per row: cl_circ, theta_u, theta_l, dstar_u, dstar_l (TE), ue_TE"""
    out = []
    for s in range(0, len(cst), B):
        c = tf.constant(cst[s:s + B].astype(np.float32)); a = tf.constant(al[s:s + B].reshape(-1, 1).astype(np.float32)); r = tf.constant(re[s:s + B].reshape(-1, 1).astype(np.float32))
        geom = build_single_loop_contour_cst_tf(c, n_surface_pts=NSP); g = op1.predict_gamma(c, a, Re=r, training=False)
        Ut = compute_inviscid_surface_velocity(geom, g["gamma_nodes"], a)["Ut"]
        cl = 2.0 * tf.reduce_sum(g["gamma_panels"] * geom["ds"], axis=1); ue = 0.5 * (tf.abs(Ut[:, 0]) + tf.abs(Ut[:, -1]))
        bl = op2.predict_bl(c, a, r, training=False)
        out.append(np.stack([cl.numpy(), bl["theta_u"].numpy()[:, -1], bl["theta_l"].numpy()[:, -1], bl["delta_u"].numpy()[:, -1], bl["delta_l"].numpy()[:, -1], ue.numpy()], 1).astype(np.float64))
    return np.concatenate(out)


def gate(cl, cd, valid):
    return valid.astype(bool) & (np.abs(cl) <= 3) & (cd > 1e-4) & (cd <= 0.1)


# ---- training rows (deck), subsampled ----
D = np.load(os.path.join(CLONE, "staged/data/xfoil_nf_scale_merged_clean.npz"))
tv = gate(D["train_cl_visc"].ravel(), D["train_cd_visc"].ravel(), D["train_valid"].ravel()); idx = np.flatnonzero(tv); rng = np.random.default_rng(0); idx = idx if ntrain is None else np.sort(rng.choice(idx, min(ntrain, len(idx)), replace=False))
Str = state(D["train_cst"][idx], D["train_alpha"].ravel()[idx], D["train_Re"].ravel()[idx]); cl_tr = D["train_cl_visc"].ravel()[idx].astype(np.float64)
print(f"train rows {len(idx)} of {int(tv.sum())} gated", flush=True)
# ---- evaluation sets ----
H = np.load(os.path.join(CLONE, "staged/data/holdout_targets.npz"), allow_pickle=True); hv = gate(H["cl_visc"][:, 0], H["cd_visc"][:, 0], H["valid"].ravel())
Sh = state(H["cst"][hv], H["alpha"][hv].ravel(), H["Re"][hv].ravel()); cl_h = H["cl_visc"][hv, 0].astype(np.float64)
S_ = np.load(os.path.join(ART, "raw/sep_eval_targets_v2.npz")); sv = gate(S_["cl_visc"][:, 0], S_["cd_visc"][:, 0], S_["valid"].ravel())
Ss = state(S_["cst"][sv], S_["alpha"][sv].ravel(), S_["Re"][sv].ravel()); cl_s = S_["cl_visc"][sv, 0].astype(np.float64)
r2 = lambda p, t: float(1 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2))
assert not np.allclose(Sh[:, 0], cl_h)

def poly2(F):
    return poly2_features_tf(tf.constant(F, tf.float64)).numpy()

def fit_ridge(X, y, lam=1e-2):
    mu = X.mean(0); sd = X.std(0) + 1e-12; mu[0] = 0; sd[0] = 1; Xs = (X - mu) / sd
    w = np.linalg.solve(Xs.T @ Xs + lam * len(Xs) * np.eye(X.shape[1]), Xs.T @ y); return mu, sd, w

def apply(X, mu, sd, w): return ((X - mu) / sd) @ w

feat_names = ["cl_circ", "theta_u", "theta_l", "dstar_u", "dstar_l", "ue_te"]
res = dict(schema="CL_DECAMBER_CLOSURE_TEST_V1", utc=datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"), n_train=int(len(idx)), n_holdout=int(hv.sum()), n_separated=int(sv.sum()),
           features=feat_names, target="cl_xfoil - cl_circ", baseline=dict(cl_circ_r2_holdout=r2(Sh[:, 0], cl_h), cl_circ_r2_separated=r2(Ss[:, 0], cl_s), neuralfoil_r2_holdout=0.9937, neuralfoil_r2_separated=0.9993), variants={})
def report(name, Xtr, Xh, Xs, lam=1e-2):
    mu, sd, w = fit_ridge(Xtr, cl_tr - Str[:, 0], lam)
    ph = Sh[:, 0] + apply(Xh, mu, sd, w); ps = Ss[:, 0] + apply(Xs, mu, sd, w); ptr = Str[:, 0] + apply(Xtr, mu, sd, w)
    res["variants"][name] = dict(n_coef=int(len(w)), lam=lam, r2_train=r2(ptr, cl_tr), r2_holdout=r2(ph, cl_h), r2_separated=r2(ps, cl_s),
                                 rmse_holdout=float(np.sqrt(np.mean((ph - cl_h) ** 2))), rmse_holdout_baseline=float(np.sqrt(np.mean((Sh[:, 0] - cl_h) ** 2))),
                                 bias_holdout_before=float(np.mean(Sh[:, 0] - cl_h)), bias_holdout_after=float(np.mean(ph - cl_h)), w_standardized=w.tolist())
    print(f"{name:34s} holdout R2 {r2(Sh[:,0],cl_h):.4f} -> {r2(ph,cl_h):.4f} | separated {r2(Ss[:,0],cl_s):.4f} -> {r2(ps,cl_s):.4f} | train {r2(ptr,cl_tr):.4f} | RMSE holdout {res['variants'][name]['rmse_holdout_baseline']:.4f} -> {res['variants'][name]['rmse_holdout']:.4f}", flush=True)
report("poly2 all six state variables", poly2(Str), poly2(Sh), poly2(Ss))
# ---- deploy: coefficients of the six-state poly2 variant, frozen ----
mu, sd, w = fit_ridge(poly2(Str), cl_tr - Str[:, 0]); alpha_h = H["alpha"][hv].ravel().astype(np.float64)
cf = os.path.join(OUT, "lift_closure_poly2.npz"); np.savez(cf, mu=mu, sd=sd, w=w, features=np.array(FEATURES), lam=1e-2, n_train=len(idx), schema="LIFT_CLOSURE_POLY2_V1", utc=res["utc"])
coef = load_lift_closure(cf); f32 = lambda S: cl_closed_tf(*[tf.constant(S[:, i], tf.float32) for i in range(6)], coef).numpy().astype(np.float64)
cl_h_closed = Sh[:, 0] + apply(poly2(Sh), mu, sd, w); cl_s_closed = Ss[:, 0] + apply(poly2(Ss), mu, sd, w)
assert np.max(np.abs(f32(Sh) - cl_h_closed)) < 2e-4 and np.max(np.abs(f32(Ss) - cl_s_closed)) < 2e-4, "float32 TF closure != float64 fit"
assert not np.allclose(cl_h_closed, cl_h) and not np.allclose(cl_s_closed, cl_s)
bands = {"alpha<0": alpha_h < 0, "0<=alpha<8": (alpha_h >= 0) & (alpha_h < 8), "alpha>=8": alpha_h >= 8}
res["deployed"] = dict(coefficients=os.path.abspath(cf), n_coef=int(len(w)), tf_float32_max_abs_dev=float(max(np.max(np.abs(f32(Sh) - cl_h_closed)), np.max(np.abs(f32(Ss) - cl_s_closed)))),
    holdout=dict(n=int(hv.sum()), r2_circ=r2(Sh[:, 0], cl_h), r2_closed=r2(cl_h_closed, cl_h), rmse_circ=float(np.sqrt(np.mean((Sh[:, 0] - cl_h) ** 2))), rmse_closed=float(np.sqrt(np.mean((cl_h_closed - cl_h) ** 2))),
                 mae_circ=float(np.mean(np.abs(Sh[:, 0] - cl_h))), mae_closed=float(np.mean(np.abs(cl_h_closed - cl_h))), bias_circ=float(np.mean(Sh[:, 0] - cl_h)), bias_closed=float(np.mean(cl_h_closed - cl_h)),
                 bands={k: dict(n=int(m.sum()), r2_circ=r2(Sh[m, 0], cl_h[m]), r2_closed=r2(cl_h_closed[m], cl_h[m])) for k, m in bands.items()}),
    separated=dict(n=int(sv.sum()), r2_circ=r2(Ss[:, 0], cl_s), r2_closed=r2(cl_s_closed, cl_s), rmse_circ=float(np.sqrt(np.mean((Ss[:, 0] - cl_s) ** 2))), rmse_closed=float(np.sqrt(np.mean((cl_s_closed - cl_s) ** 2))),
                   mae_circ=float(np.mean(np.abs(Ss[:, 0] - cl_s))), mae_closed=float(np.mean(np.abs(cl_s_closed - cl_s)))))
np.savez(os.path.join(OUT, "lift_closure_preds.npz"), holdout_cl_truth=cl_h, holdout_cl_circ=Sh[:, 0], holdout_cl_closed=cl_h_closed, holdout_alpha=alpha_h, holdout_Re=H["Re"][hv].ravel(), holdout_names=H["names"][hv], holdout_state=Sh,
         separated_cl_truth=cl_s, separated_cl_circ=Ss[:, 0], separated_cl_closed=cl_s_closed, separated_alpha=S_["alpha"][sv].ravel(), separated_state=Ss)
print("deployed:", json.dumps({k: v for k, v in res["deployed"].items() if k != "coefficients"}, indent=None)[:1500], flush=True)
report("linear all six", np.c_[np.ones(len(Str)), Str], np.c_[np.ones(len(Sh)), Sh], np.c_[np.ones(len(Ss)), Ss])
d_tr = (Str[:, 3] - Str[:, 4])[:, None]; d_h = (Sh[:, 3] - Sh[:, 4])[:, None]; d_s = (Ss[:, 3] - Ss[:, 4])[:, None]
report("physics-minimal a*(dstar_u-dstar_l)+b", np.c_[np.ones(len(Str)), d_tr], np.c_[np.ones(len(Sh)), d_h], np.c_[np.ones(len(Ss)), d_s], lam=1e-6)
report("poly2 BL state only (no cl_circ)", poly2(Str[:, 1:]), poly2(Sh[:, 1:]), poly2(Ss[:, 1:]))
report("constant offset only", np.ones((len(Str), 1)), np.ones((len(Sh), 1)), np.ones((len(Ss), 1)), lam=1e-9)
fp = os.path.join(OUT, f"cl_decamber_closure_{res['utc']}.json"); json.dump(res, open(fp, "w"), indent=1); print("wrote", fp)
