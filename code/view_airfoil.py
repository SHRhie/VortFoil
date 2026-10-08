# -*- coding: utf-8 -*-
"""One window for one airfoil: VortFoil's flow field and surface results together.

    python code/view_airfoil.py                                        # asks for the airfoil, incidence and Reynolds number
    python code/view_airfoil.py --airfoil 2412 --alpha 4 --Re 3e6
    python code/view_airfoil.py --airfoil path/to/clarky.dat --alpha 4 --Re 3e6 --save clarky.png

The airfoil is a NACA 4- or 5-digit designation or a coordinate file (Selig or Lednicer .dat). The window shows
  left   the RANS field from the field decoder: velocity along and across the freestream, C_p and nu_t, with the
         section at its incidence (wind frame);
  right  the surface pressure of the vortex sheet, the boundary-layer thicknesses of both surfaces, and the
         coefficients (C_l, C_d, C_m, the circulation lift, the stagnation point), with every range warning.
The surface results come from the vortfoil package (trained on XFOIL), the field from the decoder trained on AirfRANS
k-omega SST fields. They are two models with their own training domains, and each warns about its own.
--save writes the figure; --no-show skips the window (for scripts or machines without a display).
"""
import os, sys, re, argparse, warnings
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "field"))


def ask(prompt, default, cast=str):
    """Prompt for a value, keeping the default on an empty answer and asking again on a bad one."""
    while True:
        a = input(f"{prompt} [{default}]: ").strip() or str(default)
        try:
            return cast(a)
        except ValueError:
            print(f"  not understood: {a!r}")


def section(airfoil):
    import predict_field as PF
    if re.fullmatch(r"\d{4,5}", airfoil.strip()):
        return PF.naca_coords(airfoil.strip()), f"NACA {airfoil.strip()}"
    if not os.path.exists(airfoil):
        raise SystemExit(f"{airfoil}: not a NACA 4/5-digit designation and no such coordinate file")
    return PF.read_coords(airfoil), os.path.splitext(os.path.basename(airfoil))[0]


def compute(loop, alpha, Re, h):
    """Surface results (vortfoil package) and the field (predict_field), with their warnings."""
    import vortfoil
    import predict_field as PF
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        r = vortfoil.load_model().predict(coords=loop, alpha=alpha, Re=Re)
    surf_w = [str(w.message) for w in ws if issubclass(w.category, vortfoil.RangeWarning)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PF.FieldRangeWarning)          # kept in meta["warnings"]
        out, meta = PF.predict(loop, alpha, Re, h=h)
    return r, surf_w, out, meta


def figure(name, alpha, Re, r, surf_w, out, meta):
    import matplotlib.pyplot as plt
    import vortfoil
    import predict_field as PF
    plt.rcParams.update(PF.STYLE)
    fig = plt.figure(figsize=(16, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.25, 1])
    gl = gs[0].subgridspec(2, 2); gr = gs[1].subgridspec(3, 1, height_ratios=[1.15, 1.0, 1.05])
    axf = [fig.add_subplot(gl[i, j]) for i in range(2) for j in range(2)]
    PF.draw_fields(fig, axf, out, meta)
    for ax in axf[2:]:
        ax.set_xlabel("x / c")
    for ax in axf[::2]:
        ax.set_ylabel("y / c")

    # surface pressure of the sheet: panels 0-119 run TE -> LE on the upper surface, 120-239 LE -> TE on the lower
    ax = fig.add_subplot(gr[0])
    x, cp = r.x_cp[0], r.cp[0]
    ax.plot(x[:120], cp[:120], color="#b2182b", lw=1.2, label="upper")
    ax.plot(x[120:], cp[120:], color="#2166ac", lw=1.2, label="lower")
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("x / c"); ax.set_ylabel(r"$C_p$")
    ax.set_title("surface pressure (vortex sheet)"); ax.legend(frameon=False)

    # boundary-layer thicknesses of both surfaces
    ax = fig.add_subplot(gr[1])
    xb = vortfoil.BL_X
    for k, (lab, col) in enumerate((("upper", "#b2182b"), ("lower", "#2166ac"))):
        ax.semilogy(xb[1:], r.dstar[0, k, 1:], "-", color=col, lw=1.2, label=rf"$\delta^*$ {lab}")
        ax.semilogy(xb[1:], r.theta[0, k, 1:], "--", color=col, lw=1.2, label=rf"$\theta$ {lab}")
    top = float(max(r.dstar[0].max(), r.theta[0].max()))
    ax.set_xlim(0, 1); ax.set_ylim(1e-5, 2.0 * top); ax.set_xlabel("x / c"); ax.set_ylabel("thickness / c")
    ax.set_title("boundary layer (displacement, momentum thickness)"); ax.legend(frameon=False, ncol=2, fontsize=8)

    # coefficients and warnings
    ax = fig.add_subplot(gr[2]); ax.axis("off")
    lines = [f"{name}    alpha {meta['alpha_used']:.2f} deg    Re {Re:.3g}", "",
             f"C_l  {r.cl[0]: .4f}    (circulation lift {r.cl_gamma[0]: .4f})",
             f"C_d  {r.cd[0]: .5f}",
             f"C_m  {r.cm[0]: .4f}    (quarter chord)",
             f"stagnation at {r.stagnation[0]:.4f} of the arc length",
             f"field inputs: C_l^gamma {meta['cl_gamma']:.4f}, s_sep {meta['s_sep']:.3f}", ""]
    warn = [f"surface: {m}" for m in surf_w] + [f"field: {m}" for m in meta["warnings"]]
    lines += ["warnings:"] + ([f"- {m}" for m in warn] if warn else ["- none: inside both training domains"])
    import textwrap
    txt = "\n".join(textwrap.fill(l, 78, subsequent_indent="  ") if l.startswith("- ") else l for l in lines)
    ax.text(0.0, 1.0, txt, va="top", ha="left", family="monospace", fontsize=8.5, transform=ax.transAxes)
    fig.suptitle(rf"VortFoil — {name},  $\alpha$ = {meta['alpha_used']:.2f}$^\circ$,  Re = {Re:.3g}"
                 "   (left: RANS field decoder, wind frame; right: surface model)", fontsize=12)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--airfoil", help="NACA 4/5-digit designation (e.g. 2412, 23012) or a coordinate .dat file")
    ap.add_argument("--alpha", type=float, help="incidence, deg")
    ap.add_argument("--Re", type=float, help="Reynolds number")
    ap.add_argument("--h", type=float, default=0.005, help="field grid spacing in chords (default %(default)s)")
    ap.add_argument("--save", help="also write the figure to this file (png, pdf, ...)")
    ap.add_argument("--no-show", action="store_true", help="do not open a window")
    a = ap.parse_args(argv)
    if a.airfoil is None or a.alpha is None or a.Re is None:
        print("VortFoil viewer: an airfoil, an incidence and a Reynolds number (empty answer keeps the default).")
    airfoil = a.airfoil or ask("airfoil (NACA 4/5 digits or a .dat file)", "2412")
    alpha = a.alpha if a.alpha is not None else ask("angle of attack, deg", 4.0, float)
    Re = a.Re if a.Re is not None else ask("Reynolds number", 3e6, float)
    import matplotlib
    if a.no_show:
        matplotlib.use("Agg")
    loop, name = section(airfoil)
    print(f"computing {name} at alpha {alpha:g} deg, Re {Re:.3g} ...", flush=True)
    r, surf_w, out, meta = compute(loop, alpha, Re, a.h)
    fig = figure(name, alpha, Re, r, surf_w, out, meta)
    if a.save:
        fig.savefig(a.save, dpi=150)
        print("wrote", a.save)
    import matplotlib.pyplot as plt
    if not a.no_show:
        if matplotlib.get_backend().lower() in ("agg", "pdf", "svg", "ps", "cairo", "template"):
            path = a.save or f"vortfoil_{re.sub(r'[^A-Za-z0-9]+', '_', name)}.png"
            if not a.save:
                fig.savefig(path, dpi=150)
            print(f"no interactive display (matplotlib backend {matplotlib.get_backend()}); the figure is in {path}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
