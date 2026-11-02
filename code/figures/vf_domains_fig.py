# -*- coding: utf-8 -*-
"""Geometry coverage and training domains (the manuscript's figure vf_domains), 7.22 in wide (figure*).

The figure draws random samples (seed 0) of data sets this release does not ship (the XFOIL training decks, UniFoil, the
AirfRANS cases and the further SST sets); the research generator (VortFoil/code/vf_domains_fig_v3.py) saved every plotted array
to results/vf_domains_points.npz, and this script draws the figure from that file with the same plot() function, reproducing
figures/vf_domains.png pixel for pixel. The figure's numbers (set sizes and ranges) are figures/vf_domains_numbers.json.
  python code/figures/vf_domains_fig.py [out_dir]          (default: figures/)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # <release>/code
import vortfoil_paths as VP

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "xtick.direction": "in",
                     "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "savefig.dpi": 400, "savefig.bbox": None})
D = ".neurofoil_clone/staged/data"
RC = "VortFoil/artifacts/rans_correction_20261014"

TITLES = ("NACA\nAirfRANS (SST)", "UIUC\nXFOIL training", "UIUC, 80 held out\nXFOIL evaluation", "CST$_{16}$ rows\nXFOIL training",
          "UniFoil\nRANS (SA)", "UIUC\nRANS (SST)")


def plot(P, out):
    """the figure from the plotted arrays alone (the release's code/figures/vf_domains_fig.py carries the same function)"""
    split = lambda j: np.split(np.c_[P[f"s{j}_x"], P[f"s{j}_y"]], np.cumsum(P[f"s{j}_n"])[:-1])
    fig = plt.figure(figsize=(7.22, 3.05), constrained_layout=True)
    gs = fig.add_gridspec(2, 6, height_ratios=[0.36, 1.0])
    for j, ttl in enumerate(TITLES):
        ax = fig.add_subplot(gs[0, j])
        for q in split(j):
            if j == 4:                                           # UniFoil surface points (unordered tokens): drawn as points
                ax.plot(q[:, 0], q[:, 1], ".", color="#d95f02", ms=0.6, alpha=0.6)
            else:
                ax.plot(q[:, 0], q[:, 1], "-", color="0.25", lw=0.35, alpha=0.6)
        ax.set_aspect("equal"); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.13, 0.17); ax.axis("off")
        ax.set_title(f"({'abcdef'[j]}) {ttl}", fontsize=7)
    ax = fig.add_subplot(gs[1, :])
    ax.scatter(P["xfoil_alpha"], P["xfoil_Re"], s=1.2, color="0.70", lw=0, rasterized=True, label="XFOIL boundary-layer training rows (6,000 of 325,370)")
    ax.scatter(P["uni_alpha"], P["uni_Re"], s=1.6, color="#d95f02", lw=0, alpha=0.7, rasterized=True, label="UniFoil, SA (4,000 of 17,073)")
    tr, te = ~P["af_test"], P["af_test"]
    ax.scatter(P["af_alpha"][tr], P["af_Re"][tr], s=4, color="#1f78b4", lw=0, label="AirfRANS training + validation (800), SST")
    ax.scatter(P["af_alpha"][te], P["af_Re"][te], s=7, facecolor="none", edgecolor="#1f78b4", lw=0.6, label="AirfRANS test (200), SST")
    ax.scatter(P["uiuc_alpha"], P["uiuc_Re"], s=9, marker="^", color="#33a02c", lw=0, label="UIUC sections, SST (107)")
    ax.scatter(P["hi_alpha"], P["hi_Re"], s=9, marker="D", color="#e31a1c", lw=0, label=f"high-incidence NACA, SST ({len(P['hi_alpha'])})")
    ax.scatter(P["rx_alpha"], P["rx_Re"], s=9, marker="s", color="#6a3d9a", lw=0, label=f"Reynolds-extension NACA, SST ({len(P['rx_alpha'])})")
    ax.set_yscale("log"); ax.set_xlim(-11, 19); ax.set_ylim(4e4, 1.2e7)
    ax.set_xlabel(r"$\alpha$ [deg]"); ax.set_ylabel(r"$Re$")
    ax.text(0.005, 1.02, "(g)", transform=ax.transAxes, fontsize=8, va="bottom")
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="none", ncol=3, markerscale=2.5, handletextpad=0.3, columnspacing=1.2, borderaxespad=0.2)
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"vf_domains.{ext}"), dpi=400)


if __name__ == "__main__":
    plot(dict(np.load(os.path.join(VP.RESULTS, "vf_domains_points.npz"))), sys.argv[1] if len(sys.argv) > 1 else VP.FIGURES)
