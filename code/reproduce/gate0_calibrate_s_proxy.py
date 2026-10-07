# -*- coding: utf-8 -*-
"""Gate 0 pre-step (2026-09-18): calibrate the OP2 -> separation-scalar proxy on the 800 TRAIN cases.

The 2-case smoke test showed that frac(H>3) over all nodes saturates s=1 on attached cases: OP2's
laminar/transitional H (3.5-4 before transition) is not separation.  XFOIL's s = clip(frac(Cf<0)/0.10,0,1)
is stored per TRAIN case in supervised_field/_nf_train_v5_full800.npz (keys s, f_sep_xfoil), disjoint
from full_test, so the proxy rule is CHOSEN HERE ON TRAIN ONLY and then frozen for the 200-case gate.
Candidates (all from the confirmed OP2 H = delta*/theta on the 121-node per-surface x/c grid):
  frac   : s = clip(frac(H>h over nodes with x/c>=x0, both surfaces)/0.10, 0, 1)
  teramp : s = clip((max_side H_TE - h0)/w, 0, 1)
  tail   : s = clip((contiguous separated length back from the TE, H>h, both sides, summed)/0.10, 0, 1)
Selection = min MAE vs XFOIL s on the 800 train cases (sign agreement reported alongside).
Also reports OP1 readout cl vs XFOIL cl_xtr on the same 800 cases (diagnostic only).
"""
import os, sys, json, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.argv = [sys.argv[0]]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # <release>/code
import vortfoil_paths as VP; REPO = VP.add_paths()
import state_provider as G
import tensorflow as tf
prov = G.ConfirmedStateProvider()
# the 800-case training corpus is NOT shipped in this release; see README
TRAIN_CORPUS = os.environ.get("VORTFOIL_TRAIN_CORPUS",
                              os.path.join(VP.FIELD, "supervised_field", "_nf_train_v5_full800.npz"))
z = np.load(TRAIN_CORPUS, mmap_mode="r", allow_pickle=True)
names, prm4, npar, al, Re = z["names"], z["params4"], z["n_params"], z["alpha"], z["Re"]
s_x, f_x, cl_x, clok = np.asarray(z["s"], np.float64), np.asarray(z["f_sep_xfoil"], np.float64), np.asarray(z["cl_xtr"], np.float64), np.asarray(z["xfoil_cl_ok"])
N = len(names); CST = np.zeros((N, 16), np.float32); AE = np.zeros((N, 1), np.float32); RR = np.asarray(Re, np.float32)[:, None]; ferr = np.zeros(N)
for i in range(N):
    poly = prov.polyline(np.asarray(prm4[i, :npar[i]], np.float64)); c, rot, e = prov._cst(poly); CST[i] = c; AE[i, 0] = al[i] - rot; ferr[i] = e
Hu = np.zeros((N, 121)); Hl = np.zeros((N, 121)); THu = np.zeros((N, 121)); THl = np.zeros((N, 121)); CL = np.zeros(N)
for s in range(0, N, 200):
    e = min(s + 200, N); C = tf.constant(CST[s:e]); A = tf.constant(AE[s:e]); R = tf.constant(RR[s:e])
    g = prov.op1.predict_gamma(C, A, Re=R, training=False); geom = prov.loop(C, n_surface_pts=121)
    CL[s:e] = (2.0 * tf.reduce_sum(g["gamma_panels"] * geom["ds"], axis=1)).numpy()
    bl = prov.op2.predict_bl(C, A, R, training=False)
    THu[s:e] = bl["theta_u"].numpy(); THl[s:e] = bl["theta_l"].numpy()
    Hu[s:e] = bl["delta_u"].numpy() / np.maximum(THu[s:e], 1e-9); Hl[s:e] = bl["delta_l"].numpy() / np.maximum(THl[s:e], 1e-9)
xc = np.linspace(0, 1, 121)
def score(sp):
    sp = np.asarray(sp); return dict(MAE=float(np.mean(np.abs(sp - s_x))), sign_agree=float(np.mean((sp > 0) == (s_x > 0))),
                                     frac_nonzero=float(np.mean(sp > 0)), xfoil_frac_nonzero=float(np.mean(s_x > 0)), R2_fsep=float(1 - np.sum((sp - s_x) ** 2) / np.sum((s_x - s_x.mean()) ** 2)))
cands = {}
for h in (2.8, 3.0, 3.2, 3.5, 4.0):
    for x0 in (0.0, 0.1, 0.3, 0.5, 0.7):
        m = xc >= x0; ok_u = THu > 1e-6; ok_l = THl > 1e-6
        f = (np.sum((Hu > h) & m & ok_u, 1) + np.sum((Hl > h) & m & ok_l, 1)) / np.maximum(np.sum(m & ok_u, 1) + np.sum(m & ok_l, 1), 1)
        cands[f"frac_h{h}_x{x0}"] = np.clip(f / 0.10, 0, 1)
    # tail: contiguous separated run back from the TE, each side, as a fraction of the 121 nodes
    def tail(H):
        sep = H > h; out = np.zeros(len(H))
        for i in range(len(H)):
            k = 0
            while k < 121 and sep[i, 120 - k]: k += 1
            out[i] = k / 120.0
        return out
    cands[f"tail_h{h}"] = np.clip((tail(Hu) + tail(Hl)) / 2.0 / 0.10, 0, 1)
Hte = np.maximum(Hu[:, -1], Hl[:, -1])
for h0 in (2.4, 2.6, 2.8, 3.0):
    for w in (0.5, 1.0, 1.5, 2.0):
        cands[f"teramp_h{h0}_w{w}"] = np.clip((Hte - h0) / w, 0, 1)
tab = {k: score(v) for k, v in cands.items()}
best = min(tab, key=lambda k: tab[k]["MAE"])
zero = score(np.zeros(N))
cl = dict(R2=float(1 - np.sum((CL - cl_x)[clok] ** 2) / np.sum((cl_x[clok] - cl_x[clok].mean()) ** 2)), median_abs_err=float(np.median(np.abs(CL - cl_x)[clok])),
          median_rel_err=float(np.median((np.abs(CL - cl_x) / np.maximum(np.abs(cl_x), 0.05))[clok])), n_xfoil_ok=int(clok.sum()), bias=float(np.mean((CL - cl_x)[clok])))
out = dict(schema="GATE0_S_PROXY_CALIBRATION_V1", train_cases=int(N), selection_rule="min MAE vs XFOIL s on the 800 full_train cases (disjoint from full_test)",
           selected=best, selected_score=tab[best], all_zero_baseline=zero, xfoil_s_stats=dict(frac_nonzero=float(np.mean(s_x > 0)), mean=float(s_x.mean())),
           top10=sorted(((k, v["MAE"], v["sign_agree"]) for k, v in tab.items()), key=lambda t: t[1])[:10], cl_op1_vs_xfoil_train=cl,
           cst_fit_err_max=float(ferr.max()), re_range=[float(Re.min()), float(Re.max())], candidates=tab)
os.makedirs(os.path.join(VP.RESULTS, "gate0_calibration"), exist_ok=True)
json.dump(out, open(os.path.join(VP.RESULTS, "gate0_calibration", "calibration.json"), "w"), indent=1)
print("XFOIL s: frac nonzero %.3f mean %.3f | all-zero baseline MAE %.4f" % (out["xfoil_s_stats"]["frac_nonzero"], out["xfoil_s_stats"]["mean"], zero["MAE"]))
for k, mae, sa in out["top10"]: print(f"  {k:24s} MAE {mae:.4f}  sign-agree {sa:.3f}")
print("selected:", best, json.dumps(tab[best])); print("OP1 cl vs XFOIL cl_xtr (train):", json.dumps(cl)); print("cst fit err max %.2e, Re %.2e..%.2e" % (ferr.max(), Re.min(), Re.max()))
