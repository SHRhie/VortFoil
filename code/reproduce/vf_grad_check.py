# -*- coding: utf-8 -*-
"""Gradient verification for the confirmed surface chain (2026-09-20): reverse-mode shape gradients of the sheet's
lift readout and of the closure drag with respect to the 16 CST coefficients, checked against central finite
differences on held-out cases.  Exact TF 2.10 environment of the paper's numbers:
  /opt/anaconda3/envs/tf210/bin/python VortFoil/code/vf_grad_check.py [--cases 64] [--eps 1e-3]
Reports, per quantity: median / p90 / max relative error of the 16-vector (||g_ad - g_fd|| / ||g_fd||), the
component-wise sign agreement, and the finite-difference step sensitivity (eps and eps/2), so the reader can see
that the AD gradient is the derivative of the function actually evaluated, not a smoothed proxy.
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
from staged.common.wake_ode_tf import wake_ode_cd_tf
from staged.common.lift_closure_tf import cl_closed_tf, load_lift_closure

W1 = os.path.join(CLONE, "staged/weights/op1_supervised_eps0.50.scr_bd128.npz")
W2 = os.path.join(REPO, "VortFoil_v2/artifacts/op2_cdloss_20260917T044259Z/weights/CANDIDATE_op2_m325_BT.npz")
NSP = 121
LIFT = load_lift_closure(os.path.join(REPO, "VortFoil_v2/artifacts/lift_closure_20260920/weights/lift_closure_poly2.npz"))
ncase = int(sys.argv[sys.argv.index("--cases") + 1]) if "--cases" in sys.argv else 64
eps = float(sys.argv[sys.argv.index("--eps") + 1]) if "--eps" in sys.argv else 1e-3

z = np.load(os.path.join(CLONE, "staged/data/holdout_targets.npz"), allow_pickle=True)
v = z["valid"].astype(bool).ravel(); idx = np.flatnonzero(v)[:: max(1, int(v.sum()) // ncase)][:ncase]
cst = z["cst"][idx].astype(np.float64); al = z["alpha"][idx].reshape(-1, 1).astype(np.float32); re = z["Re"][idx].reshape(-1, 1).astype(np.float32)
op1 = build_op1(n_surface_pts=NSP, branch_dim=128); op2 = build_op2_surrogate(n_surface_pts=NSP, enable_wake=False, branch_dim=128, branch_depth=3, trunk_depth=2, N_m=16)
c0 = tf.constant(cst[:2].astype(np.float32)); op1.compute_losses(c0, tf.constant(al[:2]), tf.constant(re[:2]), training=False); op2.predict_bl(c0, tf.constant(al[:2]), tf.constant(re[:2]), training=False)
ck = np.load(W1); k = sorted(x for x in ck.files if x.startswith("v_")); op1.field_decoder.build((None, int(ck[k[-6]].shape[0]))); op1.load_weights(W1); op2.load_weights(W2, strict=False)
A, R = tf.constant(al), tf.constant(re)


def chain(c):
    """c: (N,16) float32 -> (Cl, Cd) each (N,) -- the paper's surface chain (sheet readout + BL operator + fitted closure)"""
    geom = build_single_loop_contour_cst_tf(c, n_surface_pts=NSP); g = op1.predict_gamma(c, A, Re=R, training=False)
    Ut = compute_inviscid_surface_velocity(geom, g["gamma_nodes"], A)["Ut"]
    cl = 2.0 * tf.reduce_sum(g["gamma_panels"] * geom["ds"], axis=1); ue = 0.5 * (tf.abs(Ut[:, 0]) + tf.abs(Ut[:, -1]))
    bl = op2.predict_bl(c, A, R, training=False); th = bl["theta_u"][:, -1] + bl["theta_l"][:, -1]; ds = bl["delta_u"][:, -1] + bl["delta_l"][:, -1]
    cd = wake_ode_cd_tf(th, ds, ue, closure="fitted_poly2")   # signature is (theta_TE, delta*_TE, ue_TE): the closure forms H_TE itself
    clv = cl_closed_tf(cl, bl["theta_u"][:, -1], bl["theta_l"][:, -1], bl["delta_u"][:, -1], bl["delta_l"][:, -1], ue, LIFT)   # deployed lift
    return cl, cd, clv


def ad_grads(c64):
    # tf.Variable (auto-watched) rather than tape.watch on a constant: with the latter the boundary-layer path
    # intermittently returned None in this environment (TF 2.10 eager); variables were always recorded.
    # NOTE (TF 2.10, macOS arm64, CPU): asking tape.gradient AFTER the `with` block intermittently returned None for
    # every quantity of this chain, while the same gradients asked INSIDE the (persistent) tape context are always
    # returned; the finite-difference check below is what certifies them.  Gradients are therefore taken inside.
    c = tf.Variable(c64.astype(np.float32))
    with tf.GradientTape(persistent=True) as tape:
        cl, cd, clv = chain(c)
        g_cl, g_cd, g_clv = tape.gradient(tf.reduce_sum(cl), c), tape.gradient(tf.reduce_sum(cd), c), tape.gradient(tf.reduce_sum(clv), c)
    assert g_cl is not None and g_cd is not None and g_clv is not None, "gradient not connected"
    return g_cl.numpy().astype(np.float64), g_cd.numpy().astype(np.float64), g_clv.numpy().astype(np.float64)


def fd_grads(c64, h):
    """central differences, one CST component at a time, evaluated in float32 like the deployed chain"""
    N = len(c64); gcl = np.zeros((N, 16)); gcd = np.zeros((N, 16)); gclv = np.zeros((N, 16)); f64 = lambda t: t.numpy().astype(np.float64)
    for j in range(16):
        cp, cm = c64.copy(), c64.copy(); cp[:, j] += h; cm[:, j] -= h
        clp, cdp, clvp = chain(tf.constant(cp.astype(np.float32))); clm, cdm, clvm = chain(tf.constant(cm.astype(np.float32)))
        gcl[:, j] = (f64(clp) - f64(clm)) / (2 * h); gcd[:, j] = (f64(cdp) - f64(cdm)) / (2 * h); gclv[:, j] = (f64(clvp) - f64(clvm)) / (2 * h)
    return gcl, gcd, gclv


def stats(g_ad, g_fd):
    rel = np.linalg.norm(g_ad - g_fd, axis=1) / np.maximum(np.linalg.norm(g_fd, axis=1), 1e-30)
    sign = np.mean(np.sign(g_ad) == np.sign(g_fd))
    comp = np.abs(g_ad - g_fd) / np.maximum(np.abs(g_fd), 1e-3 * np.abs(g_fd).max(axis=1, keepdims=True))
    return dict(rel_err_median=float(np.median(rel)), rel_err_p90=float(np.percentile(rel, 90)), rel_err_max=float(rel.max()), sign_agreement=float(sign),
                componentwise_rel_err_median=float(np.median(comp)), grad_norm_median=float(np.median(np.linalg.norm(g_ad, axis=1))))


g_cl, g_cd, g_clv = ad_grads(cst)
out = dict(schema="VF_GRAD_CHECK_V2", utc=datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"), tf=tf.__version__, precision="float32 chain, float64 differences",
           n_cases=int(len(idx)), quantities="Cl = 2*sum(gamma*ds) (sheet readout); Cd = fitted closure of the boundary-layer trailing-edge state and the sheet's u_e,TE; Cl(closed) = deployed lift = sheet Cl + trailing-edge-state lift closure", eps=eps, results={})
for h in (eps, eps / 2):
    f_cl, f_cd, f_clv = fd_grads(cst, h)
    out["results"][f"eps={h:g}"] = dict(dCl_dCST16=stats(g_cl, f_cl), dCd_dCST16=stats(g_cd, f_cd), dClclosed_dCST16=stats(g_clv, f_clv))
    print(f"   dCl(closed): rel median {out['results'][f'eps={h:g}']['dClclosed_dCST16']['rel_err_median']:.2e} p90 {out['results'][f'eps={h:g}']['dClclosed_dCST16']['rel_err_p90']:.2e} max {out['results'][f'eps={h:g}']['dClclosed_dCST16']['rel_err_max']:.2e} sign {out['results'][f'eps={h:g}']['dClclosed_dCST16']['sign_agreement']:.4f}", flush=True)
    print(f"eps={h:g}  dCl: rel median {out['results'][f'eps={h:g}']['dCl_dCST16']['rel_err_median']:.2e} p90 {out['results'][f'eps={h:g}']['dCl_dCST16']['rel_err_p90']:.2e} max {out['results'][f'eps={h:g}']['dCl_dCST16']['rel_err_max']:.2e} sign {out['results'][f'eps={h:g}']['dCl_dCST16']['sign_agreement']:.4f} | "
          f"dCd: rel median {out['results'][f'eps={h:g}']['dCd_dCST16']['rel_err_median']:.2e} p90 {out['results'][f'eps={h:g}']['dCd_dCST16']['rel_err_p90']:.2e} max {out['results'][f'eps={h:g}']['dCd_dCST16']['rel_err_max']:.2e} sign {out['results'][f'eps={h:g}']['dCd_dCST16']['sign_agreement']:.4f}", flush=True)
od = os.path.join(REPO, "VortFoil/tables/speed"); fp = os.path.join(od, f"_grad_check_{out['utc']}.json"); json.dump(out, open(fp, "w"), indent=1); print("wrote", fp)
