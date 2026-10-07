"""Evaluate the physics drag route + NeuralFoil on an arbitrary holdout-schema targets npz.
Measurement only.  Usage: eval_on_targets.py <targets.npz> <out.npz> [op2_weights]"""
import os, sys
import numpy as np
from itertools import combinations_with_replacement as cwr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # <release>/code
import vortfoil_paths as VP; VP.add_paths()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import tensorflow as tf; tf.config.set_visible_devices([], "GPU")
from staged.common.geometry import build_single_loop_contour_cst_tf, cst_geometry_batch_tf
from staged.common.panel import compute_inviscid_surface_velocity
from staged.common.wake_ode_tf import wake_ode_cd_tf
from staged.op1_inviscid import build_op1
from staged.op2_bl_surrogate import build_op2_surrogate

TGT, OUT = sys.argv[1], sys.argv[2]
OP2_W = sys.argv[3] if len(sys.argv) > 3 else "op2_m325_BT.npz"   # = op2_final_eps0.50_direct.m325_BT.npz
W = VP.WEIGHTS + os.sep; DT = tf.float32; NSP = 121

z = np.load(TGT, allow_pickle=True)
v = z["valid"] & (np.abs(z["cl_visc"][:, 0]) <= 3) & (z["cd_visc"][:, 0] > 1e-4) & (z["cd_visc"][:, 0] <= 0.1)
for k in ["theta_u_te", "theta_l_te", "dstar_u_te", "dstar_l_te", "ue_te"]:
    v &= np.isfinite(z[k][:, 0]) & ((z[k][:, 0] != 0) if "theta" in k else True)
cst = z["cst"][v].astype(np.float32); al = z["alpha"][v]; re = z["Re"][v]; N = len(cst)
cd_t = z["cd_visc"][v, 0].astype(np.float64); cl_t = z["cl_visc"][v, 0].astype(np.float64)
th_t = (z["theta_u_te"][v, 0] + z["theta_l_te"][v, 0]).astype(np.float64)
ds_t = (z["dstar_u_te"][v, 0] + z["dstar_l_te"][v, 0]).astype(np.float64)
ue_t = z["ue_te"][v, 0].astype(np.float64)
print(f"rows {N}")

op1 = build_op1(n_surface_pts=NSP, branch_dim=128)
op2 = build_op2_surrogate(n_surface_pts=NSP, enable_wake=False, branch_dim=128,
                          branch_depth=3, trunk_depth=2, N_m=16)
op1.compute_losses(tf.constant(cst[:2], DT), tf.constant(al[:2], DT), tf.constant(re[:2], DT), training=False)
op2.predict_bl(tf.constant(cst[:2], DT), tf.constant(al[:2], DT), tf.constant(re[:2], DT), training=False)
_ck = np.load(W + "op1_supervised_eps0.50.scr_bd128.npz")
_k = sorted(x for x in _ck.files if x.startswith("v_"))
op1.field_decoder.build((None, int(_ck[_k[-6]].shape[0])))
op1.load_weights(W + "op1_supervised_eps0.50.scr_bd128.npz")
op2.load_weights(W + OP2_W, strict=False)

thp = np.zeros(N); dsp = np.zeros(N); uep = np.zeros(N); clp = np.zeros(N)
for s in range(0, N, 1024):
    e = min(s + 1024, N)
    c = tf.constant(cst[s:e], DT); a = tf.constant(al[s:e], DT); r = tf.constant(re[s:e], DT)
    geom = build_single_loop_contour_cst_tf(c, n_surface_pts=NSP)
    g = op1.predict_gamma(c, a, Re=r, training=False)
    Ut = compute_inviscid_surface_velocity(geom, g["gamma_nodes"], a)["Ut"]
    uep[s:e] = 0.5 * (tf.abs(Ut[:, 0]) + tf.abs(Ut[:, -1])).numpy()
    clp[s:e] = (2.0 * tf.reduce_sum(g["gamma_panels"] * geom["ds"], axis=1)).numpy()
    bl = op2.predict_bl(c, a, r, training=False)
    thp[s:e] = (bl["theta_u"][:, -1] + bl["theta_l"][:, -1]).numpy()
    dsp[s:e] = (bl["delta_u"][:, -1] + bl["delta_l"][:, -1]).numpy()

cc = np.load(os.path.join(VP.WEIGHTS, "cd_closure_poly2.npz")); mu, sd, w = cc["mu"], cc["sd"], cc["w"]
def clo(th, H, ue):
    F = np.c_[np.log(np.maximum(th, 1e-12)), H, ue]
    X = np.c_[tuple([np.ones(len(F))] + [F[:, list(k)].prod(1) for d in (1, 2) for k in cwr(range(3), d)])]
    return np.clip(10 ** (((X - mu) / sd) @ w), 1e-4, 0.1)
ode = lambda t, d, u: np.clip(wake_ode_cd_tf(tf.constant(t, DT), tf.constant(d, DT),
                                             tf.constant(u, DT)).numpy().astype(np.float64), 1e-4, 0.1)

# NeuralFoil lives in its own venv (nf_benchmark_033); coordinates are dumped here
# and predictions are filled in by run_nf_on.py, then merged.
xu, yu, xl, yl = cst_geometry_batch_tf(tf.constant(cst, DT), n_surface_pts=NSP)
XY = np.concatenate([np.stack([xu.numpy()[:, ::-1], yu.numpy()[:, ::-1]], -1),
                     np.stack([xl.numpy()[:, 1:], yl.numpy()[:, 1:]], -1)], 1).astype(np.float64)
np.savez(OUT.replace(".npz", "_nfin.npz"), coords=XY, alpha=al[:, 0], Re=re[:, 0])
_nf = OUT.replace(".npz", "_nfout.npz")
if os.path.exists(_nf):
    _z = np.load(_nf); cd_nf = np.clip(_z["cd_nf"], 1e-4, 0.1); cl_nf = _z["cl_nf"]; conf = _z["conf"]
else:
    cd_nf = np.full(N, np.nan); cl_nf = np.full(N, np.nan); conf = np.full(N, np.nan)
    print("NeuralFoil predictions not present yet -- run run_nf_on.py then re-run")


np.savez(OUT, thp=thp, dsp=dsp, uep=uep, clp=clp, th_t=th_t, ds_t=ds_t, ue_t=ue_t,
         cd_t=cd_t, cl_t=cl_t, alpha=al[:, 0], Re=re[:, 0],
         cd_fit=clo(thp, dsp / thp, uep), cd_ode=ode(thp, dsp, uep),
         cd_fit_truthstate=clo(th_t, ds_t / th_t, ue_t),
         cd_nf=cd_nf, cl_nf=cl_nf, conf=conf)
r2 = lambda p, t, m: float(1 - np.sum((p[m] - t[m]) ** 2) / np.sum((t[m] - t[m].mean()) ** 2))
d = np.load(OUT); H = ds_t / th_t
print(f"\n{'구간':14s} {'n':>5s} {'phys(fit)':>10s} {'phys(ODE)':>10s} {'NF':>8s} {'천장(truth상태)':>15s} {'NF conf':>8s}")
for nm, m in [("전체", np.ones(N, bool)), ("H<6", H < 6), ("H>=6", H >= 6),
              ("H>=10", H >= 10), ("a>=15", al[:, 0] >= 15)]:
    if m.sum() < 10: continue
    print(f"{nm:14s} {m.sum():5d} {r2(d['cd_fit'],cd_t,m):10.4f} {r2(d['cd_ode'],cd_t,m):10.4f} "
          f"{r2(cd_nf,cd_t,m):8.4f} {r2(d['cd_fit_truthstate'],cd_t,m):15.4f} {np.median(conf[m]):8.3f}")
print(f"\nCl R²  physics {r2(clp,cl_t,np.ones(N,bool)):.4f}   NeuralFoil {r2(cl_nf,cl_t,np.ones(N,bool)):.4f}")
