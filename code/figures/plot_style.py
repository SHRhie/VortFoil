# -*- coding: utf-8 -*-
"""
NCFE 2026 figure style -- camera-ready conference defaults.

  - Times New Roman, mathtext = stix
  - 9-point body, 8-point ticks
  - Inward ticks on all four spines
  - B&W only -- distinguish cases by linestyle + marker, not color
  - 600 dpi save default, tight bbox

Usage:
    from staged.scripts.plot_style import apply_ncfe_style
    apply_ncfe_style()
"""
from __future__ import annotations

import matplotlib.pyplot as plt


def apply_ncfe_style():
    """Mutate matplotlib rcParams in place.  Idempotent."""
    plt.rcParams.update({
        "font.family":       "Times New Roman",
        "mathtext.fontset":  "stix",
        "font.size":         9,
        "axes.linewidth":    0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.top":         True,
        "ytick.right":       True,
        "savefig.dpi":       600,
        "savefig.bbox":      "tight",
        "lines.linewidth":   1.2,
        "axes.prop_cycle":   (
            plt.cycler(color=[
                "k", "0.4", "0.6", "0.2", "k", "0.4", "0.6",
            ]) +
            plt.cycler(linestyle=[
                "-", "--", "-.", ":", "-", "--", "-.",
            ]) +
            plt.cycler(marker=[
                "o", "s", "^", "D", "v", "<", ">",
            ])
        ),
    })


if __name__ == "__main__":
    apply_ncfe_style()
    print("NCFE style applied.")
