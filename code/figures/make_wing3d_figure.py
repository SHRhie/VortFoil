# -*- coding: utf-8 -*-
"""2.5D wing against 3D RANS: the manuscript figure and table (experimental extension of the manuscript).

Reads results/wing3d/wing3d_results.json (the research assembly of the 3D RANS runs and the 2.5D results, with the deployed
decoder in the RANS state; or --results) and writes into figures/ (or --out):
  fig_wing3d_validation.pdf / .png   one figure at the manuscript's print size: 5.40 x 4.32 in (text width, 5:4), 11/10 pt, 400 dpi
  figure_wing3d.tex                  the figure float with its caption draft (\\label{fig:wing3d})
  table_wing3d.tex                   the booktabs table with its caption draft (\\label{tab:wing3d}; needs \\fittab of main.tex)
Figure data: (a) spanwise loading and (b) section C_p of NACA 2412 at 6 deg, from the results file; (c) every section that has
runs. Sections without runs are skipped in (c) and shown as '--' in the table. The written table is read back (column meanings
from its own header) and every number compared with the results file; exit status 1 on a mismatch.
The manuscript figure: python code/figures/make_wing3d_figure.py --v3   [--results J] [--out DIR]
"""
import os, re, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")                                                 # no GUI windows
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib import font_manager as fm
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

ap = argparse.ArgumentParser()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # <release>/code
import vortfoil_paths as VP
ap.add_argument("--results", default=os.path.join(VP.RESULTS, "wing3d", "wing3d_results.json"))
ap.add_argument("--out", default=VP.FIGURES, help="output folder (default: figures/)")
ap.add_argument("--v3", action="store_true", help="two-column journal layout (2026-10-15): 7.22 in wide (figure*), 8/7 pt")
args = ap.parse_args()
OUT = args.out or os.path.dirname(os.path.abspath(args.results))
os.makedirs(OUT, exist_ok=True)
R = json.load(open(args.results))
SEC, WING = R["sections"], R["wing"]

ORDER = ("NACA 0012", "NACA 0018", "NACA 2412", "NACA 4412", "NACA 23012")   # table rows and panel (c) groups
ORDER += tuple(n for n in SEC if n not in ORDER)                             # a section this list does not know is added, never dropped
ALPHAS = ("2", "6", "10")                                                    # incidences (deg), as the results file keys them
STATES = ("sst", "xfoil")                                                    # plot order: SST state first
COL = dict(sst="#2a78d6", xfoil="#eb6834")                                   # categorical slots 1-2 (validated on white), RANS = black
NAME = dict(sst="SST state", xfoil="XFOIL state")
NTIP = 2                                                                     # outermost RANS strips flagged in (a)

try:
    fm.findfont("Times New Roman", fallback_to_default=False)
except ValueError:
    print("note: Times New Roman is not installed; the serif fallback is used")
plt.rcParams.update({
    "font.family": ["Times New Roman", "STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8 if args.v3 else 11, "axes.labelsize": 8 if args.v3 else 11, "xtick.labelsize": 7 if args.v3 else 10,
    "ytick.labelsize": 7 if args.v3 else 10, "legend.fontsize": 7 if args.v3 else 10,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "pdf.fonttype": 42, "savefig.dpi": 400, "savefig.bbox": None})


def pending(ax):
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", va="center")


def run(name, a):
    return SEC.get(name, {}).get("alphas", {}).get(a, {})


def steady(name, a):
    """the RANS run settled (assembler's rule); True when the run is not there, so nothing is marked"""
    return (run(name, a).get("rans", {}).get("convergence") or {}).get("steady", True)


UNSTEADY = []                                                                 # (name, incidence) of the RANS runs that did not settle


def err(name, a, s, k):
    """quasi-3D error of C_L / C_D (k) relative to RANS in percent, or None if the run is not there"""
    q = run(name, a).get("quasi3d", {}).get(s)
    return None if q is None else 100 * q["rel_err_" + k]


# ---------------------------------------------------------------- figure
Wd = 7.22 if args.v3 else 5.40
LM, RM, GAP = (0.52, 0.10, 0.62) if args.v3 else (0.62, 0.14, 0.74)        # inches: margins, gap between (a) and (b)
TOP, HT, MID, HC, HB = (0.20, 1.75, 0.52, 1.30, 0.42) if args.v3 else (0.26, 1.50, 0.70, 1.30, 0.56)                          # top margin (panel letters), row heights and gaps, bottom margin (labels of (c))
Hd = TOP + HT + MID + HC + HB
fig = plt.figure(figsize=(Wd, Hd))
WA = (Wd - LM - RM - GAP) / 2


def axes(l, b, w, h):
    return fig.add_axes([l / Wd, b / Hd, w / Wd, h / Hd])


aa, ab, ac = axes(LM, HB + HC + MID, WA, HT), axes(LM + WA + GAP, HB + HC + MID, WA, HT), axes(LM, HB, Wd - LM - RM, HC)
for ax, t in zip((aa, ab, ac), "abc"):
    ax.text(0, 1.03, f"({t})", transform=ax.transAxes, ha="left", va="bottom")
for ax in (aa, ab):
    ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))

# (a) spanwise loading, NACA 2412 at 6 deg
L = R.get("spanwise_loading_naca2412_6deg")
if L:
    y, rans = np.array(L["two_y_over_b"]), np.array(L["rans"])
    for s, ls, lw in (("sst", "-", 2.4), ("xfoil", "--", 1.1)):                # SST wide underneath, XFOIL dashed on top: the two nearly coincide
        if s in L["quasi3d"]:
            aa.plot(y, L["quasi3d"][s], ls, color=COL[s], lw=lw, zorder=2 + (s == "xfoil"))
    aa.plot(y, rans, "-", color="k", lw=0.6, zorder=4)
    aa.plot(y[:-NTIP], rans[:-NTIP], "o", color="k", ms=3.0, zorder=5)
    aa.plot(y[-NTIP:], rans[-NTIP:], "x", color="k", ms=5.0, mew=1.1, clip_on=False, zorder=5)
    aa.set_xlim(0, 1); aa.set_ylim(0, 1.0)
    aa.set_xlabel(r"$2y/b$"); aa.set_ylabel(r"$c\,c_l/\bar{c}$")
    aa.text(0.97, 0.96, r"NACA 2412, $%g^\circ$" % L["alpha"], transform=aa.transAxes, ha="right", va="top")
    aa.legend(handles=[Line2D([], [], color="k", lw=0.6, marker="o", ms=3.0, label="3D RANS")]
              + [Line2D([], [], color=COL[s], ls="-" if s == "sst" else "--", lw=2.4 if s == "sst" else 1.1, label=NAME[s]) for s in STATES if s in L["quasi3d"]],
              loc="lower left", frameon=False, handlelength=1.6, labelspacing=0.25, borderaxespad=0.3)
else:
    pending(aa)

# (b) section C_p at 2y/b = 0.49 (the strip nearest eta = 0.5), RANS against the SST-state strip
C = R.get("section_cp_naca2412_6deg_eta05")
if C:
    for sfc in ("upper", "lower"):
        ab.plot(C["rans"][sfc]["x"], C["rans"][sfc]["cp"], "o", color="k", ms=2.2, mew=0)
        if "sst_strip" in C:
            ab.plot(C["sst_strip"][sfc]["x"], C["sst_strip"][sfc]["cp"], "-", color=COL["sst"], lw=1.4)
    ab.set_xlim(0, 1); ab.set_ylim(1.1, -1.45)                               # C_p axis inverted
    ab.set_xlabel(r"$x/c$"); ab.set_ylabel(r"$C_p$")
    ab.text(0.97, 0.96, r"$2y/b=%.2f$" % C["two_y_over_b"], transform=ab.transAxes, ha="right", va="top")
    ab.legend(handles=[Line2D([], [], color="k", ls="none", marker="o", ms=3.0, label="3D RANS")]
              + ([Line2D([], [], color=COL["sst"], lw=1.4, label=NAME["sst"])] if "sst_strip" in C else []),
              loc="lower right", frameon=False, handlelength=1.6, labelspacing=0.25, borderaxespad=0.2)
else:
    pending(ab)

# (c) quasi-3D error of C_L (circles) and C_D (squares) against RANS, per available section and incidence
groups = [n for n in ORDER if any(err(n, a, s, "CL") is not None for a in ALPHAS for s in STATES)]
SLOT, DS = (-0.3, 0.0, 0.3), 0.065                                           # incidence slots within a group; offset of the two states within a slot
pts = {(s, k, ok): ([], []) for s in STATES for k in ("CL", "CD") for ok in (True, False)}
for g, n in enumerate(groups):
    for a, sl in zip(ALPHAS, SLOT):
        for s, d in zip(STATES, (-DS, DS)):
            for k in ("CL", "CD"):
                v = err(n, a, s, k)
                if v is not None:
                    pts[s, k, steady(n, a)][0].append(g + sl + d); pts[s, k, steady(n, a)][1].append(v)
ac.axhline(0, color="k", lw=0.8, zorder=1)
for g in range(1, len(groups), 2):
    ac.axvspan(g - 0.5, g + 0.5, color="#f1f1f1", lw=0, zorder=0)
for (s, k, ok), (x, v) in pts.items():                                        # faded: the RANS run did not settle
    ac.plot(x, v, ls="none", marker="o" if k == "CL" else "s", ms=4.3, mew=1.1, color=COL[s],
            mfc=COL[s] if s == "sst" else "white", alpha=1.0 if ok else 0.35, zorder=3)
if groups:
    allv = [x for _, ys in pts.values() for x in ys]
    ac.set_ylim(10 * np.floor((min(allv + [0]) - 4) / 10), 10 * np.ceil((max(allv + [0]) + 4) / 10))     # >= 4 % clear of the extreme marker
    ac.yaxis.set_major_locator(MultipleLocator(20))
    ac.set_xlim(-0.5, len(groups) - 0.5)
    ac.set_xticks([g + sl for g in range(len(groups)) for sl in SLOT])
    ac.set_xticklabels([r"$%s^\circ$" % a for _ in groups for a in ALPHAS])
    for g, n in enumerate(groups):
        ac.annotate(n, (g, 0), xycoords=ac.get_xaxis_transform(), xytext=(0, -15 if args.v3 else -19), textcoords="offset points", ha="center", va="top")
else:
    pending(ac)
ac.set_ylabel("Relative error (%)"); ac.yaxis.set_minor_locator(MultipleLocator(10))
mk = lambda s, m: Line2D([], [], ls="none", marker=m, ms=4.3, mew=1.1, color=COL[s], mfc=COL[s] if s == "sst" else "white")
ac.legend(handles=[(mk(s, "o"), mk(s, "s")) for s in STATES], labels=[NAME[s] for s in STATES],
          handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)}, loc="lower left", bbox_to_anchor=(0.07, 1.0),
          ncol=2, frameon=False, handlelength=2.0, columnspacing=1.4, borderaxespad=0.2, borderpad=0.1)
ac.text(1.0, 1.03, r"circles: $C_L$,  squares: $C_D$", transform=ac.transAxes, ha="right", va="bottom")

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_wing3d_validation.{ext}"), dpi=400)
plt.close(fig)

# ---------------------------------------------------------------- captions and table
def put(name, text):
    open(os.path.join(OUT, name), "w").write(text)


mc = sorted({round(s["mesh_cells"] / 1e6, 1) for s in SEC.values() if s.get("mesh_cells")})
CELLS = (("$%.1f$" % mc[0] if len(mc) == 1 else "$%.1f$--$%.1f$" % (mc[0], mc[-1])) + " million") if mc else "about $1.2$ million"
iters = sorted({run(n, a)["rans"]["iterations"] for n in SEC for a in ALPHAS if "rans" in run(n, a)})
ITERS = ("$%d$" % iters[0] if len(iters) == 1 else "$%d$--$%d$" % (iters[0], iters[-1])) if iters else "$2000$"
FRAME = (r"2.5D results take the section aerodynamics from the released VortFoil models in two reference states: the XFOIL state "
         r"(the vortex-sheet surface branch trained on XFOIL) and the SST state (the RANS field decoder trained on AirfRANS "
         r"$k$--$\omega$ SST, wall pressure integrated plus wall-function friction); the sections are coupled by the classical "
         r"Phillips--Snyder nonlinear lifting line with a relaxed free wake, the induced drag taken in the Trefftz plane. "
         r"The reference is a 3D RANS solution (OpenFOAM v2112 simpleFoam, $k$--$\omega$ SST, wall functions) on a coarse "
         r"snappyHexMesh mesh of " + CELLS + r" cells, half model, " + ITERS + r" iterations; "
         + (r"a finer-mesh check for NACA 2412 is given in the text." if R.get("mesh_check_naca2412") else r"mesh sensitivity is pending."))
PLAN = (r"span $%g$~m, chord $%g\to%g$~m, $%g^\circ$ washout, $%g$~m/s, $Re=%g\times10^{6}\to%g\times10^{6}$ from root to tip, planform area $S=%g$~m$^2$"
        % (WING["span_m"], WING["root_chord_m"], WING["tip_chord_m"], WING["washout_deg"], WING["V_ms"],
           WING["Re_root"] / 1e6, WING["Re_tip"] / 1e6, WING["S_ref_m2"]))
nst, hb = WING["strips_full_span"], WING["span_m"] / 2                      # the lifting line's cosine-spaced strips: widths (cm) of the outermost ones
TIPW = tuple(100 * hb * (np.cos(j * np.pi / nst) - np.cos((j + 1) * np.pi / nst)) for j in range(NTIP))
CAPF = (r"2.5D wing against 3D RANS on the planform of Table~\ref{tab:wing3d}. (a)~Spanwise loading $c\,c_l/\bar c$ "
        r"(local chord $c$, section lift coefficient $c_l$, $\bar c=S/b$) against $2y/b$ (root $0$, tip $1$) for the NACA 2412 wing at "
        r"root incidence $\alpha=%g^\circ$: RANS strips (black; the crosses are the two outermost strips, $%.1f$ and $%.1f$~cm wide, where the RANS "
        r"loading far exceeds the lifting line's), the 2.5D model in the SST state (blue) and the XFOIL state (orange, dashed). "
        r"(b)~Section $C_p$ at $2y/b=%.2f$ of the same wing: RANS (black dots) and the SST-state strip at its effective incidence of "
        r"$%.1f^\circ$ (blue line). (c)~Relative error $(\text{2.5D}-\text{RANS})/\text{RANS}$ of the 2.5D $C_L$ (circles) and $C_D$ (squares) at root incidences $2^\circ$, $6^\circ$ "
        r"and $10^\circ$ for each NACA section; filled: SST state, open: XFOIL state%s. "
        % ((L or {}).get("alpha", np.nan), *TIPW, (C or {}).get("two_y_over_b", np.nan), (C or {}).get("sst_strip_alpha_eff_deg", np.nan),
           r"; faded: the RANS solution does not settle (Table~\ref{tab:wing3d})" if any(not steady(n, a) for n in SEC for a in ALPHAS) else ""))
put("figure_wing3d.tex",
    "% Draft figure float (experimental extension): the figure comes from the generator next to it and wing3d_results.json.\n"
    + r"\begin{figure}[tbp]\centering" + "\n" + r"\includegraphics[width=\linewidth]{fig_wing3d_validation.pdf}" + "\n"
    + r"\caption{" + CAPF + FRAME + "}\n" + r"\label{fig:wing3d}" + "\n" + r"\end{figure}" + "\n")

MISS = [0]


def num(v, nd, sign=False):
    if v is None:
        MISS[0] += 1
        return "--"
    return f"${v:+.{nd}f}$" if sign else f"${v:.{nd}f}$"


rows = []
for n in ORDER:
    for i, a in enumerate(ALPHAS):
        r = run(n, a).get("rans")
        if not steady(n, a):
            UNSTEADY.append((n, a))
        c = [n if i == 0 else "", f"${a}$" if steady(n, a) else f"${a}^{{\\dagger}}$", num(r and r["CL"], 4), num(r and r["CD"], 5)]
        for s in STATES:
            c += [num(err(n, a, s, k), 1, True) for k in ("CL", "CD")]
        rows.append(" & ".join(c) + r" \\")
    rows.append(r"\addlinespace")
rows.pop()
stats = []
for b, lab in (("2-6 deg", r"$\alpha=2$--$6^\circ$"), ("10 deg", r"$\alpha=10^\circ$")):
    st = [R["summary_statistics"].get(s, {}).get(b) for s in STATES]
    ns = [x["n"] if x else 0 for x in st]
    c = []
    for x in st:
        c += [num(x and 100 * x["mean_abs_rel_err_CL"], 1), num(x and 100 * x["mean_rel_err_CD"], 1, True)]
    nl = "$n=%d$" % ns[0] if len(set(ns)) == 1 else "$n=%d$ SST, $%d$ XFOIL" % tuple(ns)      # the case counts differ if a state lacks runs
    stats.append(r"\multicolumn{4}{l}{Mean, %s (%s)} & " % (lab, nl) + " & ".join(c) + r" \\")
def sci(v):
    e = int(np.floor(np.log10(v)))
    return r"%.0f\times10^{%d}" % (v / 10 ** e, e)


CONV = [run(n, a)["rans"]["convergence"] for n in SEC for a in ALPHAS if (run(n, a).get("rans") or {}).get("convergence")]
OKC = [c for c in CONV if c.get("steady", True)]
DAG = (r"$^{\dagger}$The RANS solution does not settle: over the last quarter of the iterations its $C_L$ and $C_D$ vary by "
       + "; ".join(r"$%.1f\%%$ and $%.1f\%%$ (%s at $%s^\circ$, final initial residual $%s$)"
                   % (100 * c["CL_rel_range_last25pct"], 100 * c["CD_rel_range_last25pct"], n, a, sci(max(c["final_initial_residuals"].values())))
                   for n, a in UNSTEADY for c in [run(n, a)["rans"]["convergence"]])
       + r", against at most $%.2f\%%$ and $%s$ in the other runs; it is left out of the means."
       % (100 * max(max(c["CL_rel_range_last25pct"], c["CD_rel_range_last25pct"]) for c in OKC), sci(max(max(c["final_initial_residuals"].values()) for c in OKC)))
       ) if UNSTEADY else ""
CAPT = (r"3D RANS against the 2.5D wing: " + PLAN + r"; five NACA sections at root incidences $\alpha=2^\circ$, $6^\circ$ and $10^\circ$. "
        r"Coefficients refer to $S$; the RANS values are means over the last $10\%$ of the iterations, and "
        r"$\varepsilon_C=100\,(C_{\text{2.5D}}/C_{\text{RANS}}-1)$ is the 2.5D error in percent. " + FRAME +
        r" In the mean rows the $C_L$ entry is the mean absolute error and the $C_D$ entry the mean signed error over the available section--incidence cases"
        + (r" whose RANS solution settles. " + DAG if UNSTEADY else ".")
        + (" Dashes: no run yet." if MISS[0] else ""))
put("table_wing3d.tex",
    "% Draft table (experimental extension): generated from wing3d_results.json by the figure generator next to it; '--' = no run yet.\n"
    + r"\begin{table}[tbp]\centering" + "\n" + r"\caption{" + CAPT + "}\n" + r"\label{tab:wing3d}" + "\n" + r"\fittab{%" + "\n"
    + r"\begin{tabular}{llrrrrrr}" + "\n" + r"\toprule" + "\n"
    + r"& & \multicolumn{2}{c}{3D RANS} & \multicolumn{2}{c}{SST state} & \multicolumn{2}{c}{XFOIL state} \\" + "\n"
    + r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}" + "\n"
    + r"Section & $\alpha$ (deg) & $C_L$ & $C_D$ & $\varepsilon_{C_L}$ (\%) & $\varepsilon_{C_D}$ (\%) & $\varepsilon_{C_L}$ (\%) & $\varepsilon_{C_D}$ (\%) \\" + "\n"
    + r"\midrule" + "\n" + "\n".join(rows) + "\n" + r"\midrule" + "\n" + "\n".join(stats) + "\n"
    + r"\bottomrule" + "\n" + r"\end{tabular}}" + "\n" + r"\end{table}" + "\n")


def check_table(tex):
    """read the written table back (each column's meaning from its own header) and compare every number with the results file at
    its printed precision: catches swapped columns, wrong rows, wrong signs. Returns (numbers checked, mismatches)"""
    rows = [[c.strip() for c in l.rstrip()[:-2].split("&")] for l in tex.splitlines() if l.rstrip().endswith("\\\\")]
    grp = [g for g in re.findall(r"multicolumn\{2\}\{c\}\{([^}]*)\}", tex) for _ in (0, 1)]       # group name of each of the six value columns
    col = [(g, re.search(r"C_(.)", h).group(1)) for g, h in zip(grp, next(r for r in rows if r[0] == "Section")[2:])]
    key, bad, name, nchk = {"SST state": "sst", "XFOIL state": "xfoil"}, [], None, 0
    for r in rows:
        if len(r) == 8 and re.fullmatch(r"\$\d+(\^\{\\dagger\})?\$", r[1]):                       # a section / incidence row
            name, a = r[0] or name, re.match(r"\$(\d+)", r[1]).group(1)
            want = [(run(name, a).get("rans", {}).get("C" + k), 4 + (k == "D")) if g == "3D RANS" else (err(name, a, key[g], "C" + k), 1)
                    for g, k in col]
            cells = r[2:]
        elif r[0].startswith("\\multicolumn{4}"):                                                         # a mean row: |C_L| mean, C_D mean, per state
            b = "2-6 deg" if "--" in r[0] else "10 deg"
            sx = {g: R["summary_statistics"].get(key[g], {}).get(b) for g, _ in col[2:]}
            want = [(sx[g] and 100 * sx[g]["mean_abs_rel_err_CL" if k == "L" else "mean_rel_err_CD"], 1) for g, k in col[2:]]
            cells = r[1:]
        else:
            continue
        for (g, k), c, (v, nd) in zip(col[len(col) - len(cells):], cells, want):
            nchk += 1
            if (c == "--") != (v is None) or (v is not None and abs(float(c.strip("$")) - v) > 0.5 * 10 ** -nd + 1e-9):
                bad.append((name, g, k, c, v))
    return nchk, bad


nchk, bad = check_table(open(os.path.join(OUT, "table_wing3d.tex")).read())
print(f"sections in panel (c): {', '.join(groups) or 'none'}; missing: {', '.join(n for n in ORDER if n not in groups) or 'none'}")
print("wrote fig_wing3d_validation.pdf/.png, figure_wing3d.tex, table_wing3d.tex in", OUT)
print(f"table read back: {nchk} cells checked against the results file, {len(bad)} mismatches")
for m in bad:
    print("  MISMATCH", m)
sys.exit(1 if bad else 0)
