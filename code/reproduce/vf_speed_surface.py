# -*- coding: utf-8 -*-
"""Inference cost of the CONFIRMED surface chain (2026-09-20) -- OP1 sheet operator + readouts, OP2 boundary-layer
operator, fitted closure -- per case, as deployed (eager TF, CPU), at several batch sizes, plus reverse-mode
shape gradients.  Run in the exact environment of the sealed numbers (TF 2.10):
  /opt/anaconda3/envs/tf210/bin/python VortFoil/code/vf_speed_surface.py
Methodology (lessons of the 2026-08-29 speed section): median over repeats after warm-up; host load recorded before
and after; every absolute is reported together with the batch size; within-run ratios (grad/forward, chain/OP1)
are the stable statistics.  Inputs = the attached holdout grid (data of the paper), replicated to fill batches.
"""
import os, sys, json, time, datetime, statistics, platform
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
WL = os.path.join(REPO, "VortFoil_v2/artifacts/lift_closure_20260920/weights/lift_closure_poly2.npz")
DT = tf.float32; NSP = 121
LIFT = load_lift_closure(WL)
BATCHES = (1, 64, 1024, 2737); REPEATS = {1: 50, 64: 20, 1024: 8, 2737: 5}


def load():
    z = np.load(os.path.join(CLONE, "staged/data/holdout_targets.npz"), allow_pickle=True)
    cst, al, re = z["cst"].astype(np.float32), z["alpha"].reshape(-1, 1).astype(np.float32), z["Re"].reshape(-1, 1).astype(np.float32)
    op1 = build_op1(n_surface_pts=NSP, branch_dim=128)
    op2 = build_op2_surrogate(n_surface_pts=NSP, enable_wake=False, branch_dim=128, branch_depth=3, trunk_depth=2, N_m=16)
    op1.compute_losses(tf.constant(cst[:2]), tf.constant(al[:2]), tf.constant(re[:2]), training=False)
    op2.predict_bl(tf.constant(cst[:2]), tf.constant(al[:2]), tf.constant(re[:2]), training=False)
    ck = np.load(W1); k = sorted(x for x in ck.files if x.startswith("v_")); op1.field_decoder.build((None, int(ck[k[-6]].shape[0]))); op1.load_weights(W1)
    op2.load_weights(W2, strict=False)
    return cst, al, re, op1, op2


def make_batch(cst, al, re, B):
    idx = np.resize(np.arange(len(cst)), B)
    return tf.constant(cst[idx]), tf.constant(al[idx]), tf.constant(re[idx])


def op1_readouts(op1, c, a, r):
    geom = build_single_loop_contour_cst_tf(c, n_surface_pts=NSP)
    g = op1.predict_gamma(c, a, Re=r, training=False)
    Ut = compute_inviscid_surface_velocity(geom, g["gamma_nodes"], a)["Ut"]
    cl = 2.0 * tf.reduce_sum(g["gamma_panels"] * geom["ds"], axis=1)
    cp = 1.0 - Ut ** 2
    ue_te = 0.5 * (tf.abs(Ut[:, 0]) + tf.abs(Ut[:, -1]))
    # stagnation: unique sign change of Ut along the loop (numpy, as the paper's readout)
    s = np.sign(Ut.numpy()); nchg = np.sum(s[:, 1:] * s[:, :-1] < 0, axis=1)
    return cl, cp, ue_te, nchg


def op2_te(op2, c, a, r):
    bl = op2.predict_bl(c, a, r, training=False)
    th = bl["theta_u"][:, -1] + bl["theta_l"][:, -1]; ds = bl["delta_u"][:, -1] + bl["delta_l"][:, -1]
    return th, ds, bl          # (theta_TE, delta*_TE): the closure forms H_TE = delta*/theta itself


def closure(th, ds, ue):
    return wake_ode_cd_tf(th, ds, ue, closure="fitted_poly2")   # signature (theta_TE, delta_TE, ue_TE)


def lift_closure(cl, bl, ue):
    return cl_closed_tf(cl, bl["theta_u"][:, -1], bl["theta_l"][:, -1], bl["delta_u"][:, -1], bl["delta_l"][:, -1], ue, LIFT)


def timeit(fn, reps):
    fn(); fn()                                     # warm-up (eager: first call builds kernels)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts)


def main():
    load0 = os.getloadavg(); cst, al, re, op1, op2 = load()
    out = dict(schema="VF_SPEED_SURFACE_V2", utc=datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"), host=platform.platform(), cpu_count=os.cpu_count(),
               tf=tf.__version__, threads=dict(intra=tf.config.threading.get_intra_op_parallelism_threads(), inter=tf.config.threading.get_inter_op_parallelism_threads()),
               weights=dict(op1=os.path.basename(W1), op2=os.path.basename(W2), lift_closure=os.path.basename(WL)), mode="eager CPU, as deployed", n_rows_source=int(len(cst)),
               loadavg_start=load0, per_batch={})
    for B in BATCHES:
        c, a, r = make_batch(cst, al, re, B); reps = REPEATS[B]
        t_op1_cl, _ = timeit(lambda: (lambda g: 2.0 * tf.reduce_sum(g["gamma_panels"] * build_single_loop_contour_cst_tf(c, n_surface_pts=NSP)["ds"], axis=1))(op1.predict_gamma(c, a, Re=r, training=False)), reps)
        t_op1_full, _ = timeit(lambda: op1_readouts(op1, c, a, r), reps)
        t_op2, _ = timeit(lambda: op2_te(op2, c, a, r), reps)
        def chain():
            cl, cp, ue, n = op1_readouts(op1, c, a, r); th, ds, bl = op2_te(op2, c, a, r); return closure(th, ds, ue), lift_closure(cl, bl, ue)
        t_chain, _ = timeit(chain, reps)
        cv = tf.Variable(c)   # gradients are taken INSIDE the tape context (see vf_grad_check.py note) and asserted non-None
        def grad_cl():
            with tf.GradientTape() as tape:
                g = op1.predict_gamma(cv, a, Re=r, training=False); geom = build_single_loop_contour_cst_tf(cv, n_surface_pts=NSP)
                cl = 2.0 * tf.reduce_sum(g["gamma_panels"] * geom["ds"], axis=1); gr = tape.gradient(tf.reduce_sum(cl), cv)
            assert gr is not None, "dCl gradient not connected"; return gr
        def grad_cd():
            with tf.GradientTape() as tape:
                g = op1.predict_gamma(cv, a, Re=r, training=False); geom = build_single_loop_contour_cst_tf(cv, n_surface_pts=NSP)
                Ut = compute_inviscid_surface_velocity(geom, g["gamma_nodes"], a)["Ut"]; ue = 0.5 * (tf.abs(Ut[:, 0]) + tf.abs(Ut[:, -1]))
                th, ds, _ = op2_te(op2, cv, a, r); cd = closure(th, ds, ue); gr = tape.gradient(tf.reduce_sum(cd), cv)
            assert gr is not None, "dCd gradient not connected"; return gr
        def grad_clv():   # deployed lift = circulation lift + trailing-edge-state closure: shares the BL operator's backward pass
            with tf.GradientTape() as tape:
                g = op1.predict_gamma(cv, a, Re=r, training=False); geom = build_single_loop_contour_cst_tf(cv, n_surface_pts=NSP)
                Ut = compute_inviscid_surface_velocity(geom, g["gamma_nodes"], a)["Ut"]; ue = 0.5 * (tf.abs(Ut[:, 0]) + tf.abs(Ut[:, -1]))
                cl = 2.0 * tf.reduce_sum(g["gamma_panels"] * geom["ds"], axis=1); _, _, bl = op2_te(op2, cv, a, r); clv = lift_closure(cl, bl, ue); gr = tape.gradient(tf.reduce_sum(clv), cv)
            assert gr is not None, "dCl(closed) gradient not connected"; return gr
        t_gcl, _ = timeit(grad_cl, max(3, reps // 2)); t_gclv, _ = timeit(grad_clv, max(3, reps // 2)); t_gcd, _ = timeit(grad_cd, max(3, reps // 2))
        us = lambda t: 1e6 * t / B
        out["per_batch"][str(B)] = dict(repeats=reps,
            op1_cl_us_per_case=us(t_op1_cl), op1_full_readout_us_per_case=us(t_op1_full), op2_te_us_per_case=us(t_op2),
            full_chain_us_per_case=us(t_chain), grad_dCl_dCST16_us_per_case=us(t_gcl), grad_dClclosed_dCST16_us_per_case=us(t_gclv), grad_dCd_dCST16_us_per_case=us(t_gcd),
            batch_wall_full_chain_ms=1e3 * t_chain, ratios=dict(grad_cl_over_op1_cl=t_gcl / t_op1_cl, grad_clclosed_over_chain=t_gclv / t_chain, grad_cd_over_chain=t_gcd / t_chain, chain_over_op1_full=t_chain / t_op1_full))
        print(f"B={B:5d}  OP1+Cl {us(t_op1_cl):9.1f} us  OP1 full readout {us(t_op1_full):9.1f} us  OP2 {us(t_op2):9.1f} us  chain {us(t_chain):9.1f} us  "
              f"dCl {us(t_gcl):9.1f} us  dCl(closed) {us(t_gclv):9.1f} us  dCd {us(t_gcd):9.1f} us   (batch wall {1e3*t_chain:.1f} ms)", flush=True)
    out["loadavg_end"] = os.getloadavg()
    od = os.path.join(REPO, "VortFoil/tables/speed"); os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, f"_speed_confirmed_surface_{out['utc']}.json"); json.dump(out, open(fp, "w"), indent=1); print("wrote", fp)


if __name__ == "__main__":
    main()
