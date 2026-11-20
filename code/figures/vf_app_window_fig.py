# -*- coding: utf-8 -*-
"""The manuscript's application figure (2.2.0) from one run of the desktop application: (a) the application window as captured;
(b) the surface pressure and boundary-layer panels and (c) the coefficient list and warnings of the same result, cropped from the
figure the application saved with Save figure (400 dpi), where the text is legible at print size.
Inputs: results/app_window/app_window_native.png (the window as captured on macOS, light mode) and app_figure.png (its saved figure).
  python code/figures/vf_app_window_fig.py [--out DIR]     (default figures/)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                    # <release>/code
import vortfoil_paths as VP  # noqa: E402

SRC = os.path.join(VP.RESULTS, "app_window")
FIG = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else VP.FIGURES
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 8})

raw = Image.open(os.path.join(SRC, "app_window_native.png")).convert("RGBA")
win = Image.new("RGBA", raw.size, "white")
win.alpha_composite(raw)                                     # the window's transparent rounded corners on white, not black
win = np.asarray(win.convert("RGB"))
full = Image.open(os.path.join(SRC, "app_figure.png")).convert("RGB")
assert full.size == (5200, 4200), full.size                  # the crops below are in this figure's pixels
surf = np.asarray(full.crop((0, 2850, 3660, 4200)))          # bottom row: surface pressure, boundary layer and C_f
coef = np.asarray(full.crop((3630, 2880, 5200, 4060)))       # the coefficient list and warnings

W, M = 7.22, 0.24                                             # M: left margin of (b) and (c) for their labels
h_a = W * win.shape[0] / win.shape[1]
wb = 0.58 * W
h_b = wb * surf.shape[0] / surf.shape[1]
wc = W - wb - 2 * M
h_c = wc * coef.shape[0] / coef.shape[1]
hb = max(h_b, h_c)
fig = plt.figure(figsize=(W, h_a + hb + 0.12))
H = h_a + hb + 0.12
axes = [fig.add_axes([0, (hb + 0.12) / H, 1, h_a / H]), fig.add_axes([M / W, (hb - h_b) / H, wb / W, h_b / H]),
        fig.add_axes([(wb + 2 * M) / W, (hb - h_c) / H, wc / W, h_c / H])]
for k, (ax, im, lab) in enumerate(zip(axes, (win, surf, coef), ("(a)", "(b)", "(c)"))):
    ax.imshow(im, interpolation="none")
    ax.set_axis_off()
    if k == 0:                                                 # on the window's title bar
        ax.text(0.003, 0.995, lab, transform=ax.transAxes, va="top", ha="left", fontsize=8,           # 8 pt, as every panel title
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="0.3", lw=0.5))
    else:                                                      # in the margin to the left
        ax.text(-0.02, 1.0, lab, transform=ax.transAxes, va="top", ha="right", fontsize=8)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"vf_app_window.{ext}"), dpi=400)
print("height", round(H, 2), "in")
