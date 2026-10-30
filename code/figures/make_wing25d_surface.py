# -*- coding: utf-8 -*-
"""2.5D wing, surface only: the manuscript figure of the surface pressure the 2.5D model puts on the wing.

Each strip carries its section's C_p (the surface model of the chosen state at the strip's effective incidence),
drawn on the wing's upper and lower surfaces in the wind frame at true scale; no wake, no axes. Case: the full-scale
NACA 2412 wing of Table/Figure "wing3d" (span 10 m, chord 1.5 -> 0.75 m, 2 deg washout, 45 m/s) at alpha 6 deg in the
RANS state, from the saved 2.5D result (results/wing3d/wing_viz3d_n2412_sst.json: the research wing solver with the deployed
decoder, research folder wing_nll_20261010/naca_library_cmx/n2412/sst). Colours as in the application
(RdYlBu_r, C_p -1.5 .. 1 with white at 0). Writes fig_wing25d_surface.{pdf,png} and fig_wing25d_surface_numbers.json
into figures/ (or --out). The manuscript figure: python code/figures/make_wing25d_surface.py --v3
"""
import os, sys, json, hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # <release>/code
import vortfoil_paths as VP
HERE = VP.RELEASE_ROOT
SRC = os.path.join(VP.RESULTS, "wing3d", "wing_viz3d_n2412_sst.json")
ALPHA, ELEV, AZIM = 6.0, 35.0, -20.0           # seen from behind and above: the upper surface, leading edge at the far side
V3 = "--v3" in sys.argv     # 2026-10-15: two-column journal layout, 7.22 in wide (figure*), 8 pt; --out DIR for the outputs
OUTD = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else VP.FIGURES
plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 8 if V3 else 10, "savefig.dpi": 400,
                     "xtick.direction": "in", "ytick.direction": "in"})

raw = open(SRC, "rb").read(); rec = json.loads(raw)
c = next(x for x in rec["cases"] if abs(x["alpha"] - ALPHA) < 1e-9)
U, L = c["surface"]["upper"], c["surface"]["lower"]
norm, cmap = colors.TwoSlopeNorm(vmin=-1.5, vcenter=0.0, vmax=1.0), plt.get_cmap("RdYlBu_r")

fig = plt.figure(figsize=(7.22, 2.74) if V3 else (5.40, 2.05))
ax = fig.add_axes([-0.05, -0.62, 0.94, 2.25], projection="3d")      # larger than the canvas: a 3D axes keeps wide margins
for S in (L, U):                                                    # lower first, the upper surface is the one in view
    ax.plot_surface(np.array(S["x"]), np.array(S["y"]), np.array(S["z"]), facecolors=cmap(norm(np.array(S["cp"]))),
                    rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
X = np.r_[np.ravel(U["x"]), np.ravel(L["x"])]; Y = np.r_[np.ravel(U["y"]), np.ravel(L["y"])]; Z = np.r_[np.ravel(U["z"]), np.ravel(L["z"])]
ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)), zoom=0.86)    # true scale
ax.view_init(elev=ELEV, azim=AZIM); ax.set_axis_off()
cax = fig.add_axes([0.862, 0.14, 0.022, 0.72])
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
cb.set_label(r"$C_p$"); cb.ax.tick_params(direction="in")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUTD, f"fig_wing25d_surface.{ext}"), dpi=400)

cp = np.r_[np.ravel(U["cp"]), np.ravel(L["cp"])]
json.dump(dict(figure="fig_wing25d_surface", source=os.path.relpath(SRC, HERE), source_sha256=hashlib.sha256(raw).hexdigest(),
               case=dict(section="NACA 2412", alpha_deg=ALPHA, V_ms=rec["wing"]["V"], state=rec["wing"].get("sections_state"),
                         span_m=rec["wing"]["span"], CL=c["CL"], CDi=c["CDi"], CD0=c["CD0"], e=c["e"]),
               strips=len(U["cp"]), points_per_strip_side=len(U["cp"][0]), cp_min=float(cp.min()), cp_max=float(cp.max()),
               colour=dict(cmap="RdYlBu_r", vmin=-1.5, vcenter=0.0, vmax=1.0), view=dict(elev=ELEV, azim=AZIM, frame="wind", scale="true")),
          open(os.path.join(OUTD, "fig_wing25d_surface_numbers.json"), "w"), indent=1)
print("wrote fig_wing25d_surface.{pdf,png} and its numbers; C_p", round(float(cp.min()), 3), "to", round(float(cp.max()), 3))
