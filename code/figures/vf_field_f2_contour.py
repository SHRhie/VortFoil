"""Field contour figure: RANS truth against a field decoder's stored prediction on AirfRANS full_test cases.

Data: prediction files written by code/field/step2_store_preds.py, one per case, eval_{index:04d}.npz (or
eval_<case name>.npz with --name) with prediction, label, query_xy, case_index, case_name, checkpoint_sha256 and arm,
on the AirfRANS point cloud (including the no-slip surface nodes, which give the airfoil polygon without any
generator). Without --case, the cases and their indices are those of results/fig_case_selection.json (p10 / p50 / p90
of per-case u_x error).

Figure discipline: repo generator + saved result files + assert not allclose(pred, truth); numbers written next to
the PDF.

Fig. 6 of the manuscript (figures/vf_field_f0cm_tp_pos.pdf and vf_field_f0cm_tp_pos_numbers.json), from the shipped
weights of the deployed operator-conditioned decoder:
  python code/field/step2_store_preds.py --weights-tag f0cm --seed 1 --name airFoil2D_SST_39.31_8.883_4.707_0.0_12.552
  python code/figures/vf_field_f2_contour.py --preds results/preds_f0cm_s1 --tag vf_field_f0cm_tp --layout truth_pred \
      --frame wind --label "operator-conditioned decoder" --case airFoil2D_SST_39.31_8.883_4.707_0.0_12.552 \
      --case-tag pos --p-qinf --print
(--print: drawn at the manuscript text width, 5.40 in, with 11 pt titles/labels and 10 pt colour-bar ticks; the shipped
figure. Without it the figure is 10 x 8 in, and --font-scale X enlarges its text. --p-qinf: pressure limits
+-U_inf^2/2, C_p = +-1.) The case is the test case at alpha >= 8 deg with the lowest mean of the four standardized
all-mesh MSEs of the deployed run (results/step2_eval_f0cm_20260925T213247Z/per_case.npz, mse_all_*_s1): rank 14 of 200.

Author constraints (2026-09-18): no white empty regions inside the plotting window and no colormap
with a black/near-black background. Hence: the long-edge Delaunay mask is relaxed to 0.5 so the
far-field is filled; diverging channels use coolwarm (light-grey centre, not white); nu_t uses
YlGnBu on a log scale (light low end) with decade ticks instead of magma. Pressure uses RdYlBu_r. u (u_par in the wind
frame) uses coolwarm centred on the free stream (author 2026-10-12, was viridis): red faster, blue slower.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.path import Path as MplPath

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # <release>/code
import vortfoil_paths as VP; ROOT = VP.add_paths()
PREDS = os.path.join(VP.RESULTS, "preds_f0cm_s1")   # step2_store_preds.py output for Fig. 6
SEL_JSON = os.environ.get("VORTFOIL_FIG_CASE_SELECTION", os.path.join(VP.RESULTS, "fig_case_selection.json"))
OUT = VP.FIGURES
sys.path.insert(0, HERE)   # plot_style.py sits beside this file
try:
    from plot_style import apply_ncfe_style
    apply_ncfe_style()
except Exception:
    pass
plt.rcParams["savefig.bbox"] = None   # exact 10x8 in canvas (plot_style sets a tight bbox, which would crop it)

ROWS = [("ux", r"$u_x$ [m/s]", "coolwarm", "seq"),         # centred on the free stream (author 2026-10-12): limits set in make()
        ("uy", r"$u_y$ [m/s]", "coolwarm", "sym"),          # grey centre, no white void
        ("p", r"$p/\rho$ [m$^2$/s$^2$]", "RdYlBu_r", "sym"),  # pale-yellow centre; distinct from u_y (author: no shared colormap)
        ("nut", r"$\nu_t$ [m$^2$/s]", "YlGnBu", "log")]
WIN = dict(xlim=(-0.4, 1.6), ylim=(-0.6, 0.6))
WIN_TP = dict(xlim=(-0.5, 2.0), ylim=(-0.6, 0.6))   # truth/pred layout: wider window (more wake) fills the 5:4 canvas without the error column
SIG = dict(ux=28.82679722, uy=28.66885471, p=2101.85340294, nut=0.0027139)   # full-800 train std (AirfRANS convention)


def surface_polygon(xy, lab):
    w = (np.abs(lab[:, 0]) < 1e-9) & (np.abs(lab[:, 1]) < 1e-9)
    s = xy[w]
    ang = np.arctan2(s[:, 1], s[:, 0] - 0.5)          # airfoil is star-shaped about mid-chord
    s = s[np.argsort(ang)]
    return np.vstack([s, s[:1]])


def masked_tri(x, y, poly):
    tri = mtri.Triangulation(x, y)
    t = tri.triangles
    cx, cy = x[t].mean(1), y[t].mean(1)
    inside = MplPath(poly).contains_points(np.column_stack([cx, cy]))
    d = lambda i, j: np.hypot(x[t[:, i]] - x[t[:, j]], y[t[:, i]] - y[t[:, j]])
    longe = np.maximum.reduce([d(0, 1), d(1, 2), d(2, 0)]) > 0.5   # relaxed: no far-field holes
    tri.set_mask(inside | longe)
    return tri


def limits(val, kind, win):
    v = val[win]
    if kind == "seq":
        return np.percentile(v, 1), np.percentile(v, 99)
    if kind == "sym":
        a = np.percentile(np.abs(v), 99); return -a, a
    v = v[v > 0]; return max(np.percentile(v, 5), 1e-7), np.percentile(v, 99.5)


def panel(ax, tri, val, vmin, vmax, cmap, kind, poly):
    if kind == "log":
        cf = ax.tricontourf(tri, np.clip(val, vmin, None), levels=np.logspace(np.log10(vmin), np.log10(vmax), 24),
                            norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, extend="both")
    else:
        cf = ax.tricontourf(tri, val, levels=np.linspace(vmin, vmax, 24), cmap=cmap, extend="both")
    cf.set_rasterized(True)
    ax.fill(poly[:, 0], poly[:, 1], color="0.25", zorder=5)
    ax.set_xlim(*WIN["xlim"]); ax.set_ylim(*WIN["ylim"]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    return cf


PRED_LABEL = "operator-conditioned decoder"


LAYOUT = "truth_pred_err"   # or "truth_pred" (--layout truth_pred): no error column


PRINT = False       # --print: text width 5.40 in, 11 pt titles/labels, 10 pt colour-bar ticks, included 1:1 (2026-10-07)
FONT_SCALE = 1.0    # --font-scale X: titles/labels/colour-bar ticks x X; above 1 the linear colour bars get <= 5 round ticks
FRAME = "wind"      # 2026-09-26 author rule: airfoil at its true incidence (freestream along +x), --frame body = old view
PIVOT = np.array([0.25, 0.0])
P_QINF = False      # --p-qinf (2026-10-12): pressure colour limits +-U_inf^2/2 (C_p = +-1) instead of the |p| p99, which a strong
                    # suction peak widens until the field sits in the pale centre (alpha 8.9: p99 = 3.96 U_inf^2/2; case 128: 0.98)


def to_wind(xy, V, alpha_deg):
    """rotate coordinates by -alpha about the quarter chord and the velocity channels (cols 0,1) with them;
    p/rho and nu_t are frame-invariant. The query window of the stored files (x -2.2..4.2, |y| 1.6) covers the plot window."""
    a = np.radians(alpha_deg); c_, s_ = np.cos(a), np.sin(a); d = xy - PIVOT
    xr = np.c_[c_ * d[:, 0] + s_ * d[:, 1], -s_ * d[:, 0] + c_ * d[:, 1]] + PIVOT
    Vr = V.copy(); Vr[:, 0] = c_ * V[:, 0] + s_ * V[:, 1]; Vr[:, 1] = -s_ * V[:, 0] + c_ * V[:, 1]
    return xr, Vr


def make(case_file, tag, prefix="vf_field"):
    z = np.load(case_file, allow_pickle=False)
    xy, pred, lab = z["query_xy"].astype(np.float64), z["prediction"].astype(np.float64), z["label"].astype(np.float64)
    assert not np.allclose(pred, lab), "prediction equals truth -- wrong file"
    alpha = float(str(z["case_name"]).split("_")[3])
    if FRAME == "wind":
        xy0 = xy; xy, lab = to_wind(xy0, lab, alpha); _, pred = to_wind(xy0, pred, alpha)
    poly = surface_polygon(xy, lab)
    x, y = xy[:, 0], xy[:, 1]
    win = (x > WIN["xlim"][0] - 0.2) & (x < WIN["xlim"][1] + 0.2) & (np.abs(y) < WIN["ylim"][1] + 0.2)
    tri = masked_tri(x, y, poly)
    ncol = 3 if LAYOUT == "truth_pred_err" else 2
    if ncol == 2:
        WIN.update(WIN_TP)
    fig, axes = plt.subplots(4, ncol, figsize=(5.40, 4.32) if PRINT else (10, 8), constrained_layout=True)
    nums = {"case_name": str(z["case_name"]), "case_index": int(z["case_index"]), "arm": str(z["arm"]), "checkpoint_sha256": str(z["checkpoint_sha256"]),
            "frame": FRAME, "alpha_deg": alpha, "velocity_components": "along/normal to the freestream" if FRAME == "wind" else "body x/y", "channels": {}}
    for i, (k, lab_txt, cmap, kind) in enumerate(ROWS):
        t, p = lab[:, i], pred[:, i]
        vmin, vmax = limits(t, kind, win)
        if k == "p" and P_QINF:   # shared by both columns; the suction peak below C_p = -1 saturates (extend arrow)
            q = 0.5 * float(nums["case_name"].split("_")[2]) ** 2; vmin, vmax = -q, q
            nums["p_colour_limits"] = {"rule": "+-U_inf^2/2 (C_p = +-1)", "range": [-q, q]}
        if k == "ux":   # 2026-10-12 author: coolwarm centred on the free stream, red faster, blue slower; the sym rule (p99)
            # about that centre over both columns' window values, shared by both columns
            u0 = float(nums["case_name"].split("_")[2]) * (1.0 if FRAME == "wind" else float(np.cos(np.radians(alpha))))
            a = float(np.percentile(np.abs(np.r_[t[win], p[win]] - u0), 99)); vmin, vmax = u0 - a, u0 + a
            nums["u_colour_limits"] = {"rule": ("U_inf" if FRAME == "wind" else "U_inf cos(alpha)") + " +- p99 of |u - centre| over both columns' window values",
                                       "centre": u0, "range": [vmin, vmax]}
        e = np.abs(p - t); emax = np.percentile(e[win], 99)
        cols = ((t, "RANS (AirfRANS)"), (p, PRED_LABEL)) + (((e, r"$|$error$|$"),) if ncol == 3 else ())
        for j, (val, ttl) in enumerate(cols):
            ax = axes[i, j]
            if j < 2:
                cf = panel(ax, tri, val, vmin, vmax, cmap, kind, poly)
            else:
                cf = panel(ax, tri, val, 0.0, emax, "YlOrBr", "seq", poly)
            if i == 0:
                ax.set_title(ttl, fontsize=11 if PRINT else 11 * FONT_SCALE)
            if j == ncol - 1 or (ncol == 3 and j == 1):
                cb = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
                if kind == "log" and j < 2:   # decade ticks; the default picks contour levels and prints e.g. 1.21609x10^-4
                    cb.set_ticks([10.0 ** e for e in range(int(np.ceil(np.log10(vmin))), int(np.floor(np.log10(vmax))) + 1)])
                elif (FONT_SCALE > 1 or PRINT) and j < 2:   # enlarged labels: fewer, rounder ticks so they do not collide
                    cb.set_ticks([t for t in MaxNLocator(nbins=5).tick_values(vmin, vmax) if vmin <= t <= vmax])
                cb.ax.tick_params(labelsize=10 if PRINT else 7.5 * FONT_SCALE)
        if FRAME == "wind" and k in ("ux", "uy"):     # components parallel / perpendicular to U_inf (u_x, u_y rotated by alpha)
            lab_txt = {"ux": r"$u_{\parallel}$ [m/s]", "uy": r"$u_{\perp}$ [m/s]"}[k]
        axes[i, 0].set_ylabel(lab_txt, fontsize=11 if PRINT else 11 * FONT_SCALE)
        m = win & ~np.isnan(t)
        nums["channels"][k] = {"window_MSE_over_sigma2": float(np.mean((p[m] - t[m]) ** 2) / SIG[k] ** 2),
                               "window_p99_abs_error": float(emax), "truth_range_p1_p99": [float(v) for v in limits(t, kind, win)]}
    prm = nums["case_name"].split("_")
    if LAYOUT == "truth_pred_err":   # publication rule (2026-07-05): no main title; the truth/pred figure leaves it to the caption
        fig.suptitle(f"AirfRANS full test case {nums['case_index']}: $U_\\infty$={prm[2]} m/s, $\\alpha$={prm[3]}$^\\circ$  ({tag})", fontsize=11)
    out = os.path.join(OUT, f"{prefix}_{tag}")
    fig.savefig(out + ".pdf", dpi=400 if PRINT else 200); fig.savefig(out + ".png", dpi=400 if PRINT else 110)   # print: rasterized fills at 400 dpi at 1:1 size; plt.close(fig)
    return nums


if __name__ == "__main__":
    # --preds DIR --tag PREFIX [--label TEXT]: render another decoder's stored predictions (same 3 cases, same layout)
    preds_dir = sys.argv[sys.argv.index("--preds") + 1] if "--preds" in sys.argv else PREDS
    prefix = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "vf_field"
    if "--label" in sys.argv:
        PRED_LABEL = sys.argv[sys.argv.index("--label") + 1]
    if "--layout" in sys.argv:
        LAYOUT = sys.argv[sys.argv.index("--layout") + 1]
    if "--frame" in sys.argv:
        FRAME = sys.argv[sys.argv.index("--frame") + 1]
    if "--print" in sys.argv:
        PRINT = True
    if "--font-scale" in sys.argv:
        FONT_SCALE = float(sys.argv[sys.argv.index("--font-scale") + 1])
    P_QINF = "--p-qinf" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None      # p10 | p50 | p90
    sel = json.load(open(SEL_JSON))["selected_full_test_cases"]; tags = ("p10", "p50", "p90")
    if "--case" in sys.argv:   # 2026-10-12: one case by AirfRANS name (step2_store_preds.py --name) -> eval_<name>.npz
        sel, tags = [(sys.argv[sys.argv.index("--case") + 1], None, None)], (sys.argv[sys.argv.index("--case-tag") + 1],)
    allnums = {}
    for (ci, name, _), tag in zip(sel, tags):
        if only and tag != only:
            continue
        allnums[tag] = make(os.path.join(preds_dir, f"eval_{ci:04d}.npz" if isinstance(ci, int) else f"eval_{ci}.npz"), tag, prefix)
        print(tag, allnums[tag]["case_index"], {k: round(v["window_MSE_over_sigma2"], 5) for k, v in allnums[tag]["channels"].items()})
    if "--case" in sys.argv:   # its own numbers file, so the p10/p50/p90 one is left as it is
        prefix = f"{prefix}_{tags[0]}"
    json.dump(allnums, open(os.path.join(OUT, f"{prefix}_numbers.json"), "w"), indent=1)
