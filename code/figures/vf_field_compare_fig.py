# -*- coding: utf-8 -*-
"""The manuscript's field figures in one format (2.2.0): vf_field_naca (a NACA test case, in distribution) and vf_field_nonnaca
(two non-NACA SST sections, geometry coverage), with the companions vf_field_naca_other and vf_field_nonnaca_other for the
remaining channels.

Main figures: velocity magnitude |u|/U_inf and pressure coefficient C_p = p/(rho U_inf^2/2) in the wind frame (airfoil at its true
incidence, to_wind of vf_field_f2_contour.py), and wall C_p along the chord. One block per case, rows RANS and the decoder(s); colour
ranges shared by every row of a block: |u|/U_inf coolwarm centred on the free stream, 1 +- the 99th percentile of ||u|/U_inf - 1|
over all rows in the window; C_p RdYlBu_r on +-1. Companions: u_par/U_inf (centred on 1, same rule), u_perp/U_inf (+- the 99th
percentile of the RANS |u_perp|/U_inf) and nu_t/nu linear from 0 to the 99th percentile of RANS in the window.
Data:
  NACA: the deployed decoder's stored prediction for airFoil2D_SST_49.099_14.068_3.006_4.916_1.0_11.985 (needs the AirfRANS
        dataset, AIRFRANS_ROOT):
          python code/field/step2_store_preds.py --weights-tag cmx --seed 0 --name airFoil2D_SST_49.099_14.068_3.006_4.916_1.0_11.985
        Without it the NACA figures are skipped.
  non-NACA: results/vf_field_nonnaca/ ships the plotted arrays of the two sections (RANS truth, the deployed co-trained decoder
        cmx_s0 and the step-matched AirfRANS-only decoder sml2_s2 on the RANS mesh; xy and truth stored in float32 exactly) and
        selection.json, the rule that picked them among the 46 non-NACA test sections.
  python code/figures/vf_field_compare_fig.py [--out DIR]     (default figures/; numbers: figures/vf_field_compare_numbers.json)
"""
import importlib.util
import json
import os
import sys

import numpy as np
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                    # <release>/code
import vortfoil_paths as VP  # noqa: E402

OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else VP.FIGURES
NACA_CASE = "airFoil2D_SST_49.099_14.068_3.006_4.916_1.0_11.985"
NACA_NPZ = os.path.join(VP.RESULTS, "preds_cmx_s0", f"eval_{NACA_CASE}.npz")
NONNACA = os.path.join(VP.RESULTS, "vf_field_nonnaca")
spec = importlib.util.spec_from_file_location("vfc", os.path.join(HERE, "vf_field_f2_contour.py"))
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
plt = F.plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix", "font.size": 8,
                     "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
                     "xtick.direction": "in", "ytick.direction": "in", "savefig.bbox": None})
NU = 1.56e-5
F.WIN.update(xlim=(-0.4, 1.6), ylim=(-0.6, 0.6))
STYLE = {"RANS": dict(color="k", lw=0.8, ls="-", zorder=4), "AirfRANS only": dict(color="tab:blue", lw=1.0, ls="--", zorder=2),
         "AirfRANS + UniFoil": dict(color="tab:red", lw=1.0, ls="-", zorder=3), "VortFoil": dict(color="tab:red", lw=1.0, ls="-", zorder=3)}
ROWLAB = {"AirfRANS only": "AirfRANS\nonly", "AirfRANS + UniFoil": "AirfRANS\n+ UniFoil"}
SEL = json.load(open(os.path.join(NONNACA, "selection.json")))
NAMES = {"mh102": "MH 102", "rc1064c": "NASA/Langley RC10-64C"}
sci = lambda v: r"%.2f\times10^{%d}" % (v / 10 ** np.floor(np.log10(v)), np.floor(np.log10(v)))


def case_naca():
    z = np.load(NACA_NPZ)
    name = str(z["case_name"])
    lab, pred = z["label"].astype(np.float64), z["prediction"].astype(np.float64)
    U, alpha = float(name.split("_")[2]), float(name.split("_")[3])
    wall = (np.abs(lab[:, 0]) < 1e-9) & (np.abs(lab[:, 1]) < 1e-9)
    return dict(name=name, label="NACA five-digit, 12.0% thick", U=U, alpha=alpha, Re=U / NU, xy=z["query_xy"].astype(np.float64),
                wall=wall, rows={"RANS": lab, "VortFoil": pred})


def case_nonnaca(role):
    c = SEL["cases"][role]
    z = np.load(os.path.join(NONNACA, f"{c['case']}.npz"))
    assert np.isclose(float(z["alpha"]), c["alpha"]) and str(z["role"]) == role
    return dict(name=c["case"], label=NAMES[c["case"].split("_")[-1]], U=float(z["U"]), alpha=float(z["alpha"]), Re=float(z["Re"]),
                xy=z["xy"].astype(np.float64), wall=z["wall"], sel=c,          # float32 on disk, exact: widened as drawn
                rows={"RANS": z["truth"].astype(np.float64), "AirfRANS only": z["control"], "AirfRANS + UniFoil": z["cotrained"]})


def channels(C, kind):
    """wind-frame scalar fields per row, colour range and colormap per column"""
    xy0, U = C["xy"], C["U"]
    xy, _ = F.to_wind(xy0, C["rows"]["RANS"], C["alpha"])
    W = {k: F.to_wind(xy0, v, C["alpha"])[1] for k, v in C["rows"].items()}
    x, y = xy[:, 0], xy[:, 1]
    win = (x > F.WIN["xlim"][0] - 0.2) & (x < F.WIN["xlim"][1] + 0.2) & (np.abs(y) < F.WIN["ylim"][1] + 0.2)
    if kind == "main":
        val = {"umag": {k: np.hypot(v[:, 0], v[:, 1]) / U for k, v in W.items()}, "cp": {k: v[:, 2] / (0.5 * U * U) for k, v in W.items()}}
        top = float(np.percentile(np.concatenate([v[win] for v in val["umag"].values()]), 99))
        assert top > 1.0
        cols = [("umag", r"$|\mathbf{u}|/U_\infty$", "coolwarm", (0.0, top, 1.0)), ("cp", r"$C_p$", "RdYlBu_r", (-1.0, 1.0, None))]
    else:
        val = {"upar": {k: v[:, 0] / U for k, v in W.items()}, "uperp": {k: v[:, 1] / U for k, v in W.items()},
               "nut": {k: v[:, 3] / NU for k, v in W.items()}}
        a = float(np.percentile(np.abs(np.concatenate([v[win] for v in val["upar"].values()]) - 1.0), 99))
        b = float(np.percentile(np.abs(val["uperp"]["RANS"][win]), 99))
        cols = [("upar", r"$u_{\parallel}/U_\infty$", "coolwarm", (1 - a, 1 + a, None)), ("uperp", r"$u_{\perp}/U_\infty$", "coolwarm", (-b, b, None)),
                ("nut", r"$\nu_t/\nu$", "YlGnBu", (0.0, float(np.percentile(val["nut"]["RANS"][win], 99)), None))]
    return xy, win, val, cols


def loop(C):
    """wall nodes as a closed loop in the body frame: angle about the mid-thickness point at mid-chord (perimeter check)"""
    s = C["xy"][C["wall"]]
    m = np.abs(s[:, 0] - 0.5) < 0.03
    cy = 0.5 * (s[m, 1].max() + s[m, 1].min())
    o = np.argsort(np.arctan2(s[:, 1] - cy, s[:, 0] - 0.5))
    o = np.r_[o, o[:1]]
    per = float(np.sum(np.hypot(*np.diff(s[o], axis=0).T)))
    assert 1.9 < per < 2.4, (C["name"], per)
    return s[o], o


def wall_cp(C):
    s, o = loop(C)
    return s[:, 0], {k: v[C["wall"], 2][o] / (0.5 * C["U"] ** 2) for k, v in C["rows"].items()}


def panel(ax, tri, val, lim, cmap, poly):
    """F.panel (linear) with an optional two-slope norm: |u|/U_inf from 0 to its p99 with the free stream (1) at the centre colour"""
    vmin, vmax, vc = lim
    if vc is None:
        levels, norm = np.linspace(vmin, vmax, 24), None
    else:
        levels, norm = np.r_[np.linspace(vmin, vc, 12), np.linspace(vc, vmax, 13)[1:]], TwoSlopeNorm(vc, vmin, vmax)
    cf = ax.tricontourf(tri, val, levels=levels, cmap=cmap, norm=norm, extend="both")
    cf.set_rasterized(True)
    ax.fill(poly[:, 0], poly[:, 1], color="0.25", zorder=5)
    ax.set_xlim(*F.WIN["xlim"]); ax.set_ylim(*F.WIN["ylim"]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    return cf


def block(sf, C, kind, title, nums):
    xy, win, val, cols = channels(C, kind)
    rows = list(C["rows"])
    poly = F.to_wind(loop(C)[0], np.zeros((len(loop(C)[0]), 2)), C["alpha"])[0]
    tri = F.masked_tri(xy[:, 0], xy[:, 1], poly)
    nr, nc = len(rows), len(cols)
    gs = sf.add_gridspec(nr + (kind == "main"), nc, height_ratios=[1] * nr + ([1.45] if kind == "main" else []))
    ax = [[sf.add_subplot(gs[i, j]) for j in range(nc)] for i in range(nr)]
    out = dict(case=C["name"], alpha=C["alpha"], Re=C["Re"], U=C["U"], rows=rows, columns={})
    for j, (k, lab, cmap, lim) in enumerate(cols):
        vmin, vmax = lim[:2]
        for i, r in enumerate(rows):
            cf = panel(ax[i][j], tri, val[k][r], lim, cmap, poly)
        cb = sf.colorbar(cf, ax=[ax[i][j] for i in range(nr)], location="top", shrink=0.92, pad=0.02, aspect=22)
        cb.set_label(lab, labelpad=2)
        cb.set_ticks([v for v in F.MaxNLocator(nbins=4).tick_values(vmin, vmax) if vmin - 1e-12 <= v <= vmax + 1e-12])
        cb.ax.tick_params(labelsize=6.5, direction="in")
        out["columns"][k] = dict(range=[vmin, vmax], window_mse_over_rans_var={r: float(np.mean((val[k][r][win] - val[k]["RANS"][win]) ** 2)
                                                                                      / np.var(val[k]["RANS"][win])) for r in rows[1:]})
    for i, r in enumerate(rows):
        ax[i][0].set_ylabel(ROWLAB.get(r, r), fontsize=7, linespacing=0.95)
    if kind == "main":
        aw = sf.add_subplot(gs[nr, :])
        xw, cpw = wall_cp(C)
        for r in rows:
            aw.plot(xw, cpw[r], label=r, marker="", **STYLE[r])          # marker "": the release style cycles markers
        aw.invert_yaxis()
        aw.set_xlim(0, 1)
        aw.set_xlabel(r"$x/c$", labelpad=1)
        aw.set_ylabel(r"Wall $C_p$")
        if C["alpha"] > 8:                                     # the empty side of the curves
            aw.legend(loc="upper right", frameon=False, handlelength=2.2)
        else:                                                  # one row along the empty band below the lower surface
            aw.legend(loc="lower center", ncol=len(rows), frameon=False, handlelength=2.0, columnspacing=1.0)
        aw.tick_params(direction="in", top=True, right=True)
        out["wall_cp_rmse"] = {r: float(np.sqrt(np.mean((cpw[r] - cpw["RANS"]) ** 2))) for r in rows[1:]}
    sf.suptitle(title, fontsize=8)
    nums.append(out)


def title(C, tag=""):
    return (rf"{tag}{C['label']}, $\alpha={C['alpha']:.2f}^\circ$, $Re={sci(C['Re'])}$")


def main():
    nums = {}
    naca = case_naca() if os.path.exists(NACA_NPZ) else None
    if naca is None:
        print(f"{NACA_NPZ} not found: the NACA figures are skipped (see the docstring); the numbers file is not written")
    rep, dif = case_nonnaca("representative"), case_nonnaca("difficult")
    for kind, suffix, h1, h2 in (("main", "", 3.0, 3.75), ("other", "_other", 2.15, 2.75)):
        nums[kind] = []
        if naca is not None:
            fig = plt.figure(figsize=(3.49, h1), layout="constrained")
            block(fig, naca, kind, title(naca), nums[kind])
            for ext in ("pdf", "png"):
                fig.savefig(os.path.join(OUT, f"vf_field_naca{suffix}.{ext}"), dpi=400)
            plt.close(fig)
        fig = plt.figure(figsize=(7.22, h2), layout="constrained")
        sfs = fig.subfigures(1, 2, wspace=0.03)
        block(sfs[0], rep, kind, title(rep, "(a) "), nums[kind])
        block(sfs[1], dif, kind, title(dif, "(b) "), nums[kind])
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT, f"vf_field_nonnaca{suffix}.{ext}"), dpi=400)
        plt.close(fig)
    nums["selection"] = SEL
    if naca is not None:
        json.dump(nums, open(os.path.join(OUT, "vf_field_compare_numbers.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in nums.items() if k != "selection"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
