# -*- coding: utf-8 -*-
"""Surface-chain inference cost under different IMPLEMENTATION CONDITIONS (2026-09-20): eager vs tf.function graph
mode, CPU vs GPU, at the batch sizes of the paper's speed table.  Same models, weights, inputs and readouts as
vf_speed_surface.py (whose eager-CPU numbers are the deployed reference); this script adds the conditions so the
paper can state that the deployed eager-CPU figures are an upper bound on cost, not a property of the model.
  Mac (CPU):  /opt/anaconda3/envs/tf210/bin/python VortFoil/code/vf_speed_surface_modes.py --device cpu --tag mac_m5pro
  GPU host:   python vf_speed_surface_modes.py --root . --device gpu --tag rtx4060ti     (layout: root/staged, root/weights, root/data)
Timing: median over repeats after two warm-up calls (graph mode: the warm-ups include tracing); every timed call
ends with a host copy of one output so GPU work is synchronized.  No XLA (jit_compile) anywhere.
"""
import os, sys, json, time, datetime, statistics, platform, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None, help="portable layout: <root>/staged (package), <root>/weights/*.npz, <root>/data/holdout_targets.npz")
ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
ap.add_argument("--modes", default="eager,graph")
ap.add_argument("--tag", default="host")
ap.add_argument("--batches", default="1,64,1024,2737")
ap.add_argument("--out", default=None)
args = ap.parse_args()
if args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
if args.root:
    ROOT = os.path.abspath(args.root); sys.path.insert(0, ROOT)
    W1 = os.path.join(ROOT, "weights", "op1_supervised_eps0.50.scr_bd128.npz"); W2 = os.path.join(ROOT, "weights", "CANDIDATE_op2_m325_BT.npz")
    WL = os.path.join(ROOT, "weights", "lift_closure_poly2.npz"); DATA = os.path.join(ROOT, "data", "holdout_targets.npz"); OUT = args.out or ROOT
else:
    REPO = os.path.abspath(os.path.join(HERE, "..", "..")); CLONE = os.path.join(REPO, ".neurofoil_clone"); sys.path.insert(0, CLONE)
    W1 = os.path.join(CLONE, "staged/weights/op1_supervised_eps0.50.scr_bd128.npz")
    W2 = os.path.join(REPO, "VortFoil_v2/artifacts/op2_cdloss_20260917T044259Z/weights/CANDIDATE_op2_m325_BT.npz")
    WL = os.path.join(REPO, "VortFoil_v2/artifacts/lift_closure_20260920/weights/lift_closure_poly2.npz")
    DATA = os.path.join(CLONE, "staged/data/holdout_targets.npz"); OUT = args.out or os.path.join(REPO, "VortFoil/tables/speed")
import tensorflow as tf
if args.device == "cpu":
    tf.config.set_visible_devices([], "GPU")
else:
    gpus = tf.config.list_physical_devices("GPU"); assert gpus, "no GPU visible"
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
from staged.op1_inviscid import build_op1
from staged.op2_bl_surrogate import build_op2_surrogate
from staged.common.geometry import build_single_loop_contour_cst_tf
from staged.common.panel import compute_inviscid_surface_velocity
from staged.common.wake_ode_tf import wake_ode_cd_tf
from staged.common.lift_closure_tf import cl_closed_tf, load_lift_closure

NSP = 121; BATCHES = [int(b) for b in args.batches.split(",")]; REPEATS = {1: 50, 64: 20, 1024: 8, 2737: 5}
z = np.load(DATA, allow_pickle=True)
cst, al, re = z["cst"].astype(np.float32), z["alpha"].reshape(-1, 1).astype(np.float32), z["Re"].reshape(-1, 1).astype(np.float32)
op1 = build_op1(n_surface_pts=NSP, branch_dim=128)
op2 = build_op2_surrogate(n_surface_pts=NSP, enable_wake=False, branch_dim=128, branch_depth=3, trunk_depth=2, N_m=16)
op1.compute_losses(tf.constant(cst[:2]), tf.constant(al[:2]), tf.constant(re[:2]), training=False)
op2.predict_bl(tf.constant(cst[:2]), tf.constant(al[:2]), tf.constant(re[:2]), training=False)
ck = np.load(W1); k = sorted(x for x in ck.files if x.startswith("v_")); op1.field_decoder.build((None, int(ck[k[-6]].shape[0]))); op1.load_weights(W1)
op2.load_weights(W2, strict=False); LIFT = load_lift_closure(WL)


def sheet_cl(c, a, r):
    g = op1.predict_gamma(c, a, Re=r, training=False)
    return 2.0 * tf.reduce_sum(g["gamma_panels"] * build_single_loop_contour_cst_tf(c, n_surface_pts=NSP)["ds"], axis=1)


def chain(c, a, r):
    """every surface output: Cp (240 panels), stagnation crossing count, u_e,TE, BL TE state, deployed Cl, Cd"""
    geom = build_single_loop_contour_cst_tf(c, n_surface_pts=NSP); g = op1.predict_gamma(c, a, Re=r, training=False)
    Ut = compute_inviscid_surface_velocity(geom, g["gamma_nodes"], a)["Ut"]
    cl_g = 2.0 * tf.reduce_sum(g["gamma_panels"] * geom["ds"], axis=1); cp = 1.0 - Ut ** 2
    ue = 0.5 * (tf.abs(Ut[:, 0]) + tf.abs(Ut[:, -1]))
    s = tf.sign(Ut); ncross = tf.reduce_sum(tf.cast(s[:, 1:] * s[:, :-1] < 0, tf.int32), axis=1)
    bl = op2.predict_bl(c, a, r, training=False)
    th = bl["theta_u"][:, -1] + bl["theta_l"][:, -1]; ds = bl["delta_u"][:, -1] + bl["delta_l"][:, -1]
    cd = wake_ode_cd_tf(th, ds, ue, closure="fitted_poly2")
    cl = cl_closed_tf(cl_g, bl["theta_u"][:, -1], bl["theta_l"][:, -1], bl["delta_u"][:, -1], bl["delta_l"][:, -1], ue, LIFT)
    return cl, cd, cp, ncross, ue, th, ds


def grad_cl(c, a, r):
    with tf.GradientTape() as tape:
        tape.watch(c); cl = chain(c, a, r)[0]; gr = tape.gradient(tf.reduce_sum(cl), c)
    return gr


def grad_cd(c, a, r):
    with tf.GradientTape() as tape:
        tape.watch(c); cd = chain(c, a, r)[1]; gr = tape.gradient(tf.reduce_sum(cd), c)
    return gr


def timeit(fn, args_, reps):
    def call():
        out = fn(*args_); o = out[0] if isinstance(out, (tuple, list)) else out; return o.numpy()   # host copy = device sync
    call(); call()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); call(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


dev = tf.config.list_logical_devices("GPU")[0].name if args.device == "gpu" else "/CPU:0"
out = dict(schema="VF_SPEED_SURFACE_MODES_V1", utc=datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"), tag=args.tag, host=platform.platform(), cpu_count=os.cpu_count(),
           tf=tf.__version__, device=args.device, gpus=[d.name for d in tf.config.list_physical_devices("GPU")], weights=dict(op1=os.path.basename(W1), op2=os.path.basename(W2), lift_closure=os.path.basename(WL)),
           note="same chain as vf_speed_surface.py (deployed reference = eager CPU); graph = tf.function without XLA; medians after warm-up; GPU timings include the host copy of one output", modes={})
for mode in args.modes.split(","):
    fns = dict(sheet_cl=sheet_cl, chain=chain, grad_cl=grad_cl, grad_cd=grad_cd)
    if mode == "graph":
        fns = {k: tf.function(v, reduce_retracing=True) for k, v in fns.items()}
    res = {}
    with tf.device(dev):
        for B in BATCHES:
            idx = np.resize(np.arange(len(cst)), B); c, a, r = tf.constant(cst[idx]), tf.constant(al[idx]), tf.constant(re[idx]); reps = REPEATS.get(B, 5)
            try:
                gr = fns["grad_cl"](c, a, r); assert gr is not None, "dCl gradient None"
                t = {k: timeit(fns[k], (c, a, r), reps if k in ("sheet_cl", "chain") else max(3, reps // 2)) for k in fns}
            except Exception as e:   # a graph-mode trace failure is a result too
                res[str(B)] = dict(error=repr(e)[:300]); print(f"[{mode}/{args.device}] B={B} ERROR {repr(e)[:200]}", flush=True); continue
            us = lambda x: 1e6 * x / B
            res[str(B)] = dict(sheet_cl_us_per_case=us(t["sheet_cl"]), full_chain_us_per_case=us(t["chain"]), grad_dCl_dCST16_us_per_case=us(t["grad_cl"]), grad_dCd_dCST16_us_per_case=us(t["grad_cd"]),
                               batch_wall_full_chain_ms=1e3 * t["chain"], ratios=dict(grad_cl_over_chain=t["grad_cl"] / t["chain"], grad_cd_over_chain=t["grad_cd"] / t["chain"]))
            print(f"[{mode}/{args.device}] B={B:5d}  sheet Cl {us(t['sheet_cl']):9.2f} us  chain {us(t['chain']):9.2f} us  dCl {us(t['grad_cl']):9.2f} us  dCd {us(t['grad_cd']):9.2f} us  (batch wall {1e3*t['chain']:.1f} ms)", flush=True)
    out["modes"][mode] = res
os.makedirs(OUT, exist_ok=True); fp = os.path.join(OUT, f"_speed_modes_{args.tag}_{args.device}_{out['utc']}.json"); json.dump(out, open(fp, "w"), indent=1); print("wrote", fp)
