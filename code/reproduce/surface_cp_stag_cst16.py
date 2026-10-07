"""Cp, stagnation and Cm readouts of the confirmed single model (OP1 scr_bd128, CST-16) against XFOIL truth
generated on the same 240-panel midpoint grid (raw/attached_surface_targets.npz, Xeon, recipe of
gen_xfoil_all_targets.py: Cp = 1-ue^2, s_stag = argmax Cp in s in (0.4,0.6)).
Stagnation crossing exactly as stag_recompute.py / vf_talk_stag_accuracy_fig.py: sign changes of Ut along the
loop, the one nearest s=0.5, linear interpolation; unique = exactly one sign change.
Also writes the paper figures vf_cp_dist_cst16.{pdf,png} and vf_stagnation_cst16.{pdf,png} + numbers json.
Usage: surface_cp_stag_cst16.py <out.json>"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # <release>/code
import vortfoil_paths as VP; VP.add_paths()
sys.path.insert(0, os.path.join(VP.CODE, "figures"))   # plot_style.py
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from plot_style import apply_ncfe_style; apply_ncfe_style()
import tensorflow as tf; tf.config.set_visible_devices([], "GPU")
from staged.op1_inviscid import build_op1
from staged.common.geometry import build_single_loop_contour_cst_tf
from staged.common.panel import compute_inviscid_surface_velocity
R = VP.RELEASE_ROOT; FIG = VP.FIGURES
DT = tf.float32; W = os.path.join(VP.WEIGHTS, "op1_supervised_eps0.50.scr_bd128.npz")
# attached_surface_targets.npz (the v1 XFOIL Cp deck) is NOT shipped in this release; see README.
# Only attached_surface_targets_v2.npz ships. Override with ATTACHED_CP_TARGETS.
CP_TARGETS = os.environ.get("ATTACHED_CP_TARGETS", os.path.join(VP.DATA_EVAL, "attached_surface_targets.npz"))
T = np.load(CP_TARGETS, allow_pickle=True)
v = T["valid"] & (np.abs(T["cl_visc"][:, 0]) <= 3) & (T["cd_visc"][:, 0] > 1e-4) & (T["cd_visc"][:, 0] <= 0.1) & np.isfinite(T["cm_visc"][:, 0])
cst = T["cst"][v].astype(np.float32); al = T["alpha"][v].astype(np.float32); re = T["Re"][v].astype(np.float32); names = T["names"][v]
cp_t = T["cp_panels"][v].astype(np.float64); s_t = T["s_stag"][v, 0].astype(np.float64); cm_t = T["cm_visc"][v, 0].astype(np.float64)
cl_t = T["cl_visc"][v, 0].astype(np.float64); sp = T["s_panels"].astype(np.float64); N = len(cst); print("gated n =", N)
op1 = build_op1(n_surface_pts=121, branch_dim=128)
op1.compute_losses(tf.constant(cst[:2]), tf.constant(al[:2]), tf.constant(re[:2]), training=False)
_ck = np.load(W); _k = sorted(x for x in _ck.files if x.startswith("v_")); op1.field_decoder.build((None, int(_ck[_k[-6]].shape[0]))); op1.load_weights(W)
CP, UT, CMc, CMp, XCl = [], [], [], [], []
for i in range(0, N, 512):
    c, a, r = tf.constant(cst[i:i+512]), tf.constant(al[i:i+512]), tf.constant(re[i:i+512])
    geom = build_single_loop_contour_cst_tf(c, n_surface_pts=121); gg = op1.predict_gamma(c, a, Re=r, training=False)
    inv = compute_inviscid_surface_velocity(geom, gg["gamma_nodes"], a); Ut = inv["Ut"]; cp = 1.0 - Ut ** 2
    ca = tf.cos(a * np.pi / 180); sa = tf.sin(a * np.pi / 180); xc, yc, ds, nx, ny = geom["xc"], geom["yc"], geom["ds"], geom["nx"], geom["ny"]
    CMc.append((2.0 * tf.reduce_sum(((0.25 - xc) * ca - yc * sa) * gg["gamma_panels"] * ds, 1)).numpy())
    CMp.append(tf.reduce_sum((-(xc - 0.25) * (-cp * ny) + yc * (-cp * nx)) * ds, 1).numpy())
    CP.append(cp.numpy()); UT.append(Ut.numpy()); XCl.append(xc.numpy())
cp_p = np.concatenate(CP).astype(np.float64); ut = np.concatenate(UT).astype(np.float64); XC = np.concatenate(XCl); cm_c = np.concatenate(CMc); cm_p = np.concatenate(CMp)
assert cp_p.shape == cp_t.shape and not np.allclose(cp_p, cp_t)
r2 = lambda p, t: float(1 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2))
def crossing(ut1):
    idx = np.flatnonzero(np.diff(np.sign(ut1)) != 0)
    if idx.size == 0: return np.nan, 0
    j = idx[np.argmin(np.abs(sp[idx] - 0.5))]; u0, u1 = ut1[j], ut1[j + 1]; t = 0.0 if u1 == u0 else -u0 / (u1 - u0)
    return sp[j] + t * (sp[j + 1] - sp[j]), int(idx.size)
cr = np.array([crossing(u) for u in ut]); s_p, ncross = cr[:, 0], cr[:, 1].astype(int)
err_panels = np.abs(s_p - s_t) * 240.0
rmse_case = np.sqrt(np.mean((cp_p - cp_t) ** 2, 1)); rel_l2 = np.linalg.norm(cp_p - cp_t, axis=1) / np.linalg.norm(cp_t, axis=1)
S = {"model": "OP1 scr_bd128 (CST-16, 19-input)", "n": int(N), "truth": "raw/attached_surface_targets.npz (Xeon XFOIL, Ncrit 7, 240-panel midpoints)",
     "cp_rmse_pooled": float(np.sqrt(np.mean((cp_p - cp_t) ** 2))), "cp_rmse_case_median": float(np.median(rmse_case)),
     "cp_rel_l2_case_median": float(np.median(rel_l2)), "cp_rel_l2_case_p90": float(np.percentile(rel_l2, 90)),
     "stag_unique_crossing_rate": float(np.mean(ncross == 1)), "stag_zero_crossing_rate": float(np.mean(ncross == 0)),
     "stag_err_panels_median": float(np.nanmedian(err_panels)), "stag_err_panels_p90": float(np.nanpercentile(err_panels, 90)),
     "stag_within_1_panel_rate": float(np.nanmean(err_panels <= 1.0)),
     "cm_circ_r2": r2(cm_c, cm_t), "cm_press_r2": r2(cm_p, cm_t), "cm_circ_rmse": float(np.sqrt(np.mean((cm_c - cm_t) ** 2))), "cm_press_rmse": float(np.sqrt(np.mean((cm_p - cm_t) ** 2)))}
# ---- figures ----
sel = [i for i in range(N) if abs(al[i, 0] - 4.0) < 1e-6 and abs(re[i, 0] - 1e6) < 1]; seen, pick = set(), []
for i in sel:
    if names[i] not in seen: seen.add(names[i]); pick.append(i)
    if len(pick) == 4: break
fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.9), sharey=True)
for a_, i in zip(ax, pick):
    a_.plot(XC[i, :120], cp_t[i, :120], "k-", lw=0.8, label="XFOIL"); a_.plot(XC[i, 120:], cp_t[i, 120:], "k-", lw=0.8)
    a_.plot(XC[i, :120], cp_p[i, :120], "-", color="#b2182b", lw=0.8, label="sheet readout"); a_.plot(XC[i, 120:], cp_p[i, 120:], "-", color="#b2182b", lw=0.8)
    a_.invert_yaxis(); a_.set_title(f"{names[i]}  rel$L_2$ {rel_l2[i]:.3f}", fontsize=7); a_.set_xlabel("$x/c$")
ax[0].set_ylabel("$C_p$"); ax[0].legend(frameon=False, fontsize=6.5)
for e in ("pdf", "png"): fig.savefig(f"{FIG}/vf_cp_dist_cst16.{e}", dpi=400, bbox_inches="tight")
plt.close(fig)
fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.0)); k = pick[0]; m = (sp > 0.4) & (sp < 0.6)
ax[0].plot(sp[m], ut[k, m], "-", color="#b2182b", lw=0.9, label="$U_t(s)$ sheet"); ax[0].axhline(0, color="k", lw=0.5)
ax[0].axvline(s_t[k], color="k", ls="--", lw=0.7, label="XFOIL stagnation node"); ax[0].axvline(s_p[k], color="#2166ac", ls=":", lw=0.9, label="sheet crossing")
ax[0].set_xlabel("arc-length $s$"); ax[0].set_ylabel("$U_t/U_\\infty$"); ax[0].legend(frameon=False, fontsize=6); ax[0].set_title(f"{names[k]}, $\\alpha=4^\\circ$", fontsize=7)
ax[1].hist(np.clip(err_panels[np.isfinite(err_panels)], 0, 6), bins=np.arange(0, 6.25, 0.25), color="0.4"); ax[1].set_xlabel("|crossing $-$ XFOIL node| (panels)"); ax[1].set_ylabel("cases")
ax[1].set_title(f"unique crossing {100*S['stag_unique_crossing_rate']:.2f}%, median {S['stag_err_panels_median']:.2f} panels", fontsize=7)
for e in ("pdf", "png"): fig.savefig(f"{FIG}/vf_stagnation_cst16.{e}", dpi=400, bbox_inches="tight")
S["figure_cases_alpha4_Re1e6"] = [str(names[i]) for i in pick]
json.dump(S, open(sys.argv[1], "w"), indent=1); json.dump(S, open(f"{FIG}/vf_surface_cst16_numbers.json", "w"), indent=1); print(json.dumps(S, indent=1))
