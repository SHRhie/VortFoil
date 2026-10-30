# -*- coding: utf-8 -*-
"""RANS field figure of the two-column manuscript (v3): RANS truth (top row) and a field decoder (bottom row) for u_par, u_perp,
p/rho and nu_t on one AirfRANS test case, 7.22 in wide (figure* text width), 8/7 pt text; with it, the appendix companion
<stem>_nutlog (nu_t/nu on a log scale, one column).

Case rule: among the 66 test cases at alpha >= 8 deg, the median (lower middle, rank 33) of the decoder's mean four-channel
standardized all-mesh MSE (score files' normalization). The deployed decoder (--decoder cmx_s0, the manuscript's figure) reads
its per-case MSEs from results/stepmatch_score_20261015T070933Z_per_case.npz (naca_cmx_s0); the AirfRANS-only decoder
(--decoder f0cm_s1) from results/step2_eval_f0cm_*/per_case.npz. Colours, frame (wind frame, airfoil at its true incidence),
u_par centring and the +-U_inf^2/2 pressure limits are those of code/figures/vf_field_f2_contour.py, whose helpers are
imported here. The manuscript's figures/vf_field_cmx_median_hi{,_nutlog}.pdf (numbers vf_field_cmx_median_hi_numbers.json):
  python code/field/step2_store_preds.py --weights-tag cmx --seed 0 --name airFoil2D_SST_49.099_14.068_3.006_4.916_1.0_11.985
  python code/figures/vf_field_contour_v3.py figures --decoder cmx_s0
(the prediction goes to results/preds_cmx_s0/; it needs the AirfRANS dataset, AIRFRANS_ROOT, as every field evaluation does)
"""
import os, sys, json, glob, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # <release>/code
import vortfoil_paths as VP; VP.add_paths()
REL = VP.RELEASE_ROOT
PERCASE = os.path.join(VP.RESULTS, "stepmatch_score_20261015T070933Z_per_case.npz")   # the step-matched scoring's per-case MSEs
PREDS = {t: os.path.join(VP.RESULTS, f"preds_{t}") for t in ("f0cm_s1", "cmx_s0")}   # code/field/step2_store_preds.py
spec = importlib.util.spec_from_file_location("vfc", os.path.join(HERE, "vf_field_f2_contour.py"))
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
plt = F.plt
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "savefig.bbox": None})


def pick_case(tag="f0cm_s1"):
    if tag == "f0cm_s1":
        z = np.load(sorted(glob.glob(os.path.join(REL, "results/step2_eval_f0cm_*/per_case.npz")))[0], allow_pickle=True)
        m = np.mean([z[f"mse_all_{c}_s1"] for c in ("ux", "uy", "p", "nut")], 0)
    else:
        z = np.load(PERCASE, allow_pickle=True)
        m = z[f"naca_{tag}"].mean(1)
    case = [str(c) for c in z["case"]]
    a = np.array([float(c.split("_")[3]) for c in case])
    hi = np.flatnonzero(a >= 8.0); order = hi[np.argsort(m[hi], kind="stable")]
    k = order[(len(order) - 1) // 2]
    return case[k], dict(n_alpha_ge_8=int(len(hi)), rank_in_band=int((len(order) - 1) // 2 + 1), mean4_mse=float(m[k]),
                         rank_overall=int((m < m[k]).sum()) + 1)


def main(out, tag="f0cm_s1"):
    name, sel = pick_case(tag)
    stem = f"vf_field_{tag.split('_')[0]}_median_hi"
    z = np.load(os.path.join(PREDS[tag], f"eval_{name}.npz"), allow_pickle=False)
    xy, pred, lab = z["query_xy"].astype(np.float64), z["prediction"].astype(np.float64), z["label"].astype(np.float64)
    assert not np.allclose(pred, lab) and str(z["case_name"]) == name
    alpha = float(name.split("_")[3]); U = float(name.split("_")[2])
    xy0 = xy; xy, lab = F.to_wind(xy0, lab, alpha); _, pred = F.to_wind(xy0, pred, alpha)
    F.WIN.update(F.WIN_TP)
    poly = F.surface_polygon(xy, lab); x, y = xy[:, 0], xy[:, 1]
    win = (x > F.WIN["xlim"][0] - 0.2) & (x < F.WIN["xlim"][1] + 0.2) & (np.abs(y) < F.WIN["ylim"][1] + 0.2)
    tri = F.masked_tri(x, y, poly)
    fig, axes = plt.subplots(2, 4, figsize=(7.22, 2.55), constrained_layout=True)
    nums = dict(case_name=name, selection=sel, arm=str(z["arm"]), checkpoint_sha256=str(z["checkpoint_sha256"]), frame="wind",
                alpha_deg=alpha, channels={})
    labels = {"ux": r"$u_{\parallel}$ [m/s]", "uy": r"$u_{\perp}$ [m/s]", "p": r"$p/\rho$ [m$^2$/s$^2$]", "nut": r"$\nu_t$ [m$^2$/s]"}
    for j, (k, _, cmap, kind) in enumerate(F.ROWS):
        t, p = lab[:, j], pred[:, j]
        if k == "ux":
            a = float(np.percentile(np.abs(np.r_[t[win], p[win]] - U), 99)); vmin, vmax = U - a, U + a; rule = "U_inf +- p99 |u - U_inf| (both rows)"
        elif k == "uy":
            vmin, vmax = F.limits(t, "sym", win); rule = "+- p99 |RANS|"
        elif k == "p":
            q = 0.5 * U ** 2; vmin, vmax = -q, q; rule = "+-U_inf^2/2 (C_p = +-1)"
        else:
            vmin, vmax = 0.0, float(np.percentile(t[win], 99)); rule = "linear, 0 to the 99th percentile of the RANS nu_t in the window (both rows)"
        for i, (val, row) in enumerate(((t, "RANS"), (p, "decoder"))):
            cf = F.panel(axes[i, j], tri, val, vmin, vmax, cmap, "seq" if k != "nut" else "lin", poly)
        axes[0, j].set_title(labels[k])
        cb = fig.colorbar(cf, ax=axes[:, j], orientation="horizontal", shrink=0.9, pad=0.02, aspect=25)
        cb.set_ticks([v for v in F.MaxNLocator(nbins=4).tick_values(vmin, vmax) if vmin <= v <= vmax]); cb.ax.tick_params(labelsize=7)
        if k == "nut":
            cb.formatter.set_powerlimits((-2, 2)); cb.update_ticks()
        m = win & ~np.isnan(t)
        nums["channels"][k] = dict(colour_rule=rule, range=[float(vmin), float(vmax)],
                                   window_MSE_over_sigma2=float(np.mean((p[m] - t[m]) ** 2) / F.SIG[k] ** 2))
    axes[0, 0].set_ylabel("RANS"); axes[1, 0].set_ylabel("decoder")
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{stem}.{ext}"), dpi=400)
    json.dump(nums, open(os.path.join(out, f"{stem}_numbers.json"), "w"), indent=1)
    # appendix companion: the eddy-viscosity ratio nu_t/nu of the same case on a logarithmic scale (one column, 3.49 in), RANS above,
    # decoder below. Lower limit nu_t = nu (the professor, 10-15: levels like 1e-7 and 1e-6 m^2/s are both zero); values below it,
    # non-positive predictions included, are drawn in the lowest colour (F.panel clips at vmin, so there are no masked holes).
    from supervised_field.airfrans_p3_fields_v2 import NU_AIR
    t, p = lab[:, 3] / NU_AIR, pred[:, 3] / NU_AIR
    vmin, vmax = 1.0, float(np.percentile(t[win][t[win] > 0], 99.5))
    fig2, ax2 = plt.subplots(2, 1, figsize=(3.49, 2.55), constrained_layout=True)
    for a_, val, row in zip(ax2, (t, p), ("RANS", "decoder")):
        cf = F.panel(a_, tri, val, vmin, vmax, "YlGnBu", "log", poly); a_.set_ylabel(row)
    cb = fig2.colorbar(cf, ax=ax2, shrink=0.9, pad=0.02)
    cb.set_ticks([10.0 ** e for e in range(0, int(np.floor(np.log10(vmax))) + 1)])
    cb.set_label(r"$\nu_t/\nu$"); cb.ax.tick_params(labelsize=7)
    for ext in ("pdf", "png"):
        fig2.savefig(os.path.join(out, f"{stem}_nutlog.{ext}"), dpi=400)
    nums["nut_log_companion"] = dict(rule="log scale of nu_t/nu, nu = 1.56e-5 m^2/s (AirfRANS): from 1 to the 99.5th percentile of the RANS "
                                          "ratio in the window; values below 1, non-positive predictions included, take the lowest colour",
                                     range=[vmin, vmax], window_fraction_below_1=dict(rans=float(np.mean(t[win] < 1)), decoder=float(np.mean(p[win] < 1))))
    json.dump(nums, open(os.path.join(out, f"{stem}_numbers.json"), "w"), indent=1)
    print(json.dumps({k: (v if k != "channels" else {c: round(d["window_MSE_over_sigma2"], 5) for c, d in v.items()}) for k, v in nums.items()}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[sys.argv.index("--decoder") + 1] if "--decoder" in sys.argv else "f0cm_s1")
