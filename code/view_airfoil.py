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
--save writes the figure (400 dpi); --no-show skips the window (for scripts or machines without a display). On an
upright screen the panels are stacked (--layout). Resizing the window zooms the figure; its layout stays as drawn.
"""
import os, sys, re, argparse, warnings
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vortfoil_paths  # noqa: E402
vortfoil_paths.add_paths()            # this checkout's vortfoil package (src) and code/field, ahead of other installs


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


_MODEL = None


def compute(loop, alpha, Re, h, family=None, decoder=None):
    """Surface results (vortfoil package) and the field (predict_field), with their warnings."""
    global _MODEL
    import vortfoil
    import predict_field as PF
    if _MODEL is None:
        _MODEL = vortfoil.load_model()                        # once per process (the application runs many cases)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        r = _MODEL.predict(coords=loop, alpha=alpha, Re=Re)
    surf_w = [str(w.message) for w in ws if issubclass(w.category, vortfoil.RangeWarning)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PF.FieldRangeWarning)          # kept in meta["warnings"]
        out, meta = PF.predict(loop, alpha, Re, h=h, family=family, wall=True, decoder=decoder or PF.DECODER)
    return r, surf_w, out, meta


DESIGNS = {"landscape": (16.0, 7.6), "square": (13.0, 10.5), "portrait": (9.5, 12.0)}   # drawn at these sizes (in); windows zoom
WRAP = {"landscape": 78, "square": 58, "portrait": 100}   # characters per line of the coefficients box
DESIGN = DESIGNS["landscape"]


def screen_is_portrait():
    """True on a screen turned upright (taller than wide); False when that cannot be told."""
    try:
        import tkinter
        r = tkinter.Tk(); r.withdraw()
        w, h = r.winfo_screenwidth(), r.winfo_screenheight()
        r.destroy()
        return h > w
    except Exception:
        return False


def figure(name, alpha, Re, r, surf_w, out, meta, fig=None, layout="landscape"):
    """The result figure in one of three layouts: 'landscape' (fields left, surface right), 'square' (fields on top,
    surface pressure, boundary layer and coefficients in a row below) or 'portrait' (fields, then the two surface
    panels, then the coefficients). With `fig`, it is cleared and redrawn (the application reuses its figure)."""
    import matplotlib.pyplot as plt
    import vortfoil
    import predict_field as PF
    plt.rcParams.update(PF.STYLE)
    if fig is None:
        fig = plt.figure(figsize=DESIGNS[layout], constrained_layout=True)
    else:
        fig.clf()
        fig.set_size_inches(*DESIGNS[layout], forward=False)
        fig.set_layout_engine("constrained")
    if layout == "landscape":
        gs = fig.add_gridspec(1, 2, width_ratios=[2.25, 1])
        gl = gs[0].subgridspec(2, 2); gr = gs[1].subgridspec(3, 1, height_ratios=[1.15, 1.0, 1.05])
        cell = dict(cp=gr[0], bl=gr[1], text=gr[2])
    elif layout == "square":
        gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0])
        gl = gs[0].subgridspec(2, 2); gm = gs[1].subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.9])
        cell = dict(cp=gm[0], bl=gm[1], text=gm[2])
    else:
        gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.0, 0.6])
        gl = gs[0].subgridspec(2, 2); gm = gs[1].subgridspec(1, 2)
        cell = dict(cp=gm[0], bl=gm[1], text=gs[2])
    axf = [fig.add_subplot(gl[i, j]) for i in range(2) for j in range(2)]
    PF.draw_fields(fig, axf, out, meta)
    for ax in axf[2:]:
        ax.set_xlabel("x / c")
    for ax in axf[::2]:
        ax.set_ylabel("y / c")

    # surface pressure of the sheet: panels 0-119 run TE -> LE on the upper surface, 120-239 LE -> TE on the lower
    ax = fig.add_subplot(cell["cp"])
    x, cp = r.x_cp[0], r.cp[0]
    ax.plot(x[:120], cp[:120], color="#b2182b", lw=1.2, label="upper")
    ax.plot(x[120:], cp[120:], color="#2166ac", lw=1.2, label="lower")
    wl = meta.get("wall")
    if wl:                                                    # the field decoder's wall pressure, in its RANS state
        xw, cw, up = np.asarray(wl["x"]), np.asarray(wl["cp"]), np.asarray(wl["upper"], bool)
        for m, col in ((up, "#b2182b"), (~up, "#2166ac")):
            o = np.argsort(xw[m])
            ax.plot(xw[m][o], cw[m][o], "--", color=col, lw=0.9, alpha=0.8, label="RANS state (field)" if col == "#b2182b" else None)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("x / c"); ax.set_ylabel(r"$C_p$")
    ax.set_title("surface pressure: vortex sheet (XFOIL state); field decoder (RANS state)"); ax.legend(frameon=False)

    # boundary-layer thicknesses of both surfaces
    ax = fig.add_subplot(cell["bl"])
    xb = vortfoil.BL_X
    for k, (lab, col) in enumerate((("upper", "#b2182b"), ("lower", "#2166ac"))):
        ax.semilogy(xb[1:], r.dstar[0, k, 1:], "-", color=col, lw=1.2, label=rf"$\delta^*$ {lab}")
        ax.semilogy(xb[1:], r.theta[0, k, 1:], "--", color=col, lw=1.2, label=rf"$\theta$ {lab}")
    top = float(max(r.dstar[0].max(), r.theta[0].max()))
    ax.set_xlim(0, 1); ax.set_ylim(1e-5, 2.0 * top); ax.set_xlabel("x / c"); ax.set_ylabel("thickness / c")
    ax.set_title("boundary layer (displacement, momentum thickness)"); ax.legend(frameon=False, ncol=2, fontsize=8)

    # coefficients and warnings
    ax = fig.add_subplot(cell["text"]); ax.axis("off")
    lines = [f"{name}    alpha {meta['alpha_used']:.2f} deg    Re {Re:.3g}", "",
             f"C_l  {r.cl[0]: .4f}    (circulation lift {r.cl_gamma[0]: .4f})",
             f"C_d  {r.cd[0]: .5f}",
             f"C_m  {r.cm[0]: .4f}    (quarter chord)",
             f"stagnation at {r.stagnation[0]:.4f} of the arc length",
             f"field inputs: C_l^gamma {meta['cl_gamma']:.4f}, s_sep {meta['s_sep']:.3f}"]
    if meta.get("wall"):
        wl = meta["wall"]
        lines += [f"field, RANS state: C_l {wl['cl']: .4f}, C_d {wl['cd']: .5f}",
                  f"  (pressure {wl['cd_pressure']:.5f}, friction {wl['cd_friction']:.5f}, wall function)"]
    lines.append("")
    warn = [f"surface: {m}" for m in surf_w] + [f"field: {m}" for m in meta["warnings"]]
    lines += ["warnings:"] + ([f"- {m}" for m in warn] if warn else ["- none: inside both training domains"])
    import textwrap
    txt = "\n".join(textwrap.fill(l, WRAP[layout], subsequent_indent="  ") if l.startswith("- ") else l for l in lines)
    ax.text(0.0, 1.0, txt, va="top", ha="left", family="monospace", fontsize=8.5, transform=ax.transAxes)
    where = {"landscape": "left: RANS field decoder, wind frame; right: surface model",
             "square": "top: RANS field decoder, wind frame; bottom row: surface model",
             "portrait": "top: RANS field decoder, wind frame; below: surface model"}[layout]
    if meta.get("decoder") == "field_decoder_rext_s2.npz":                # experimental (branch exp/rext-decoder)
        where = "field: Re-extended decoder, experimental; " + where
    fig.suptitle(rf"VortFoil — {name},  $\alpha$ = {meta['alpha_used']:.2f}$^\circ$,  Re = {Re:.3g}"
                 + ("   " if layout == "landscape" else "\n") + f"({where})", fontsize=12)
    return fig


def export_csv(folder, name, alpha, Re, r, surf_w, out, meta):
    """Every number of one result as CSV files in `folder` (created if needed), with a text file that explains their
    columns. Returns the paths written."""
    import csv
    import vortfoil
    import predict_field as PF
    os.makedirs(folder, exist_ok=True)
    a = meta["alpha_used"]
    base = os.path.join(folder, re.sub(r"[^A-Za-z0-9.+-]+", "_", f"vortfoil_{name}_a{a:g}_Re{Re:.3g}"))
    paths = []

    def table(suffix, header, cols):
        p = f"{base}_{suffix}.csv"
        np.savetxt(p, np.column_stack(cols), delimiter=",", header=",".join(header), comments="", fmt="%.8g")
        paths.append(p)

    p = f"{base}_coefficients.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["airfoil", "alpha_given_deg", "alpha_used_deg", "Re", "Cl", "Cl_circulation", "Cd", "Cm_quarter_chord",
                    "stagnation_arc_fraction", "ue_TE_over_U", "theta_TE_over_c", "dstar_TE_over_c", "field_Cl_gamma",
                    "field_s_sep", "field_Cl_RANS", "field_Cd_RANS", "field_Cd_RANS_pressure", "field_Cd_RANS_friction", "warnings"])
        wl = meta.get("wall") or dict(cl=np.nan, cd=np.nan, cd_pressure=np.nan, cd_friction=np.nan)
        w.writerow([name, alpha, a, Re] + [f"{float(v):.8g}" for v in (r.cl[0], r.cl_gamma[0], r.cd[0], r.cm[0], r.stagnation[0],
                                                                       r.ue_te[0], r.theta_te[0], r.dstar_te[0], meta["cl_gamma"], meta["s_sep"],
                                                                       wl["cl"], wl["cd"], wl["cd_pressure"], wl["cd_friction"])]
                   + [" | ".join([f"surface: {m}" for m in surf_w] + [f"field: {m}" for m in meta["warnings"]])])
    paths.append(p)
    side = np.where(np.arange(r.cp.shape[1]) < r.cp.shape[1] // 2, 0, 1)
    table("surface", ["panel", "lower_surface", "x_over_c", "Cp"], [np.arange(r.cp.shape[1]), side, r.x_cp[0], r.cp[0]])
    table("boundary_layer", ["x_over_c", "theta_upper_over_c", "theta_lower_over_c", "dstar_upper_over_c", "dstar_lower_over_c"],
          [vortfoil.BL_X, r.theta[0, 0], r.theta[0, 1], r.dstar[0, 0], r.dstar[0, 1]])
    g = {k: np.asarray(out[k]).ravel() for k in ("x", "y", "x_body", "y_body", "inside", "u_over_U", "v_over_U", "cp", "nut_over_Uc", "u", "v", "p", "nut")}
    table("field", ["x_wind", "y_wind", "x_body", "y_body", "inside_section", "u_over_U", "v_over_U", "Cp", "nut_over_Uc",
                    "u_m_s", "v_m_s", "p_kinematic_m2_s2", "nut_m2_s"],
          [g["x"], g["y"], g["x_body"], g["y_body"], g["inside"].astype(int), g["u_over_U"], g["v_over_U"], g["cp"], g["nut_over_Uc"],
           g["u"], g["v"], g["p"], g["nut"]])
    if meta.get("wall"):
        wl = meta["wall"]
        table("wall", ["x_body", "y_body", "upper_surface", "Cp_RANS", "Cf_RANS"], [wl["x"], wl["y"], np.asarray(wl["upper"], int), wl["cp"], wl["cf"]])
    sec = np.asarray(meta["section"])
    xb, yb = PF.wind_to_body(sec[:, 0], sec[:, 1], a) if meta.get("frame", "wind") == "wind" else (sec[:, 0], sec[:, 1])
    table("section", ["x_body", "y_body"], [xb, yb])
    p = f"{base}_about.txt"
    with open(p, "w") as f:
        f.write(f"""VortFoil result: {name}, incidence {a:g} deg, Reynolds number {Re:.6g}
Chord 1 m, kinematic viscosity {meta.get('nu', 1.56e-5):g} m2/s, freestream speed U = Re nu = {meta.get('U', Re * 1.56e-5):.6g} m/s.

_coefficients.csv    the surface model (vortex sheet + boundary-layer operator + closures, trained on XFOIL): C_l (with the
                     lift closure) and the circulation lift alone, C_d, C_m about the quarter chord, the stagnation point
                     as a fraction of the arc length, the trailing-edge state, and the field decoder's inputs
                     (circulation lift C_l^gamma and separation measure s_sep), the field decoder's lift and drag in its
                     RANS state (drag = pressure + friction from a wall function); every range warning in the last column.
_surface.csv         C_p of the vortex sheet at the 240 panel midpoints: panels 0-119 run from the trailing edge to the
                     leading edge on the upper surface (lower_surface 0), 120-239 back on the lower surface (1).
_boundary_layer.csv  momentum (theta) and displacement (dstar) thickness of both surfaces at x/c = 0..1 (121 stations).
_field.csv           the RANS field of the decoder trained on AirfRANS k-omega SST at every grid point: x_wind, y_wind
                     (freestream along +x, section at its incidence), the same point in the chord frame (x_body, y_body),
                     inside_section (1: inside the section, values nan), velocity along and across the freestream, C_p and
                     nu_t/(U c), then the same in SI units (u, v in m/s, kinematic pressure p/rho in m2/s2, nu_t in m2/s).
_wall.csv            the field decoder's wall values at the section's nodes, in its RANS state (k-omega SST): x_body,
                     y_body, upper_surface (1) or lower (0), C_p, and C_f from a wall function (the decoder's velocity
                     2e-4 chord off the wall, Spalding's law; > 0 where the near-wall flow runs toward the trailing edge).
_section.csv         the section the field was computed for, in the chord frame.
""")
    paths.append(p)
    return paths


def scale_with_window(fig):
    """Make a window resize zoom the figure as a whole instead of re-flowing it. The layout is computed once at the
    design size and frozen; on every resize the axes keep their design places scaled by s = min(width, height ratio)
    and centred (letterboxed when the window's aspect ratio differs), and fonts, lines and ticks scale by s.
    Returns the resize handler (also connected to the canvas, replacing an earlier one)."""
    fig.draw_without_rendering()                              # constrained layout at the current (design) size
    fig.set_layout_engine("none")
    W0, H0 = fig.get_size_inches()
    boxes = [(ax, ax.get_position().bounds) for ax in fig.axes]
    texts = [fig._suptitle] if fig._suptitle else []
    texts += list(fig.texts)
    places = [(t, t.get_position()) for t in texts]           # figure-fraction positions of figure-level texts
    for ax in fig.axes:
        texts += [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label,
                  ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()] + list(ax.texts)
    fonts = [(t, t.get_fontsize()) for t in texts]
    lines = [(l, l.get_linewidth(), l.get_markersize()) for ax in fig.axes for l in ax.lines]
    # a legend's paddings and handle lengths are fixed when it is built, so it is rebuilt at the scaled font size,
    # pinned (upper-left corner, axes fraction) where the design layout put it ("best" could otherwise move it)
    renderer = fig.canvas.get_renderer()
    legends = [(ax, lg.get_window_extent(renderer).transformed(ax.transAxes.inverted()), getattr(lg, "_ncols", getattr(lg, "_ncol", 1)),
                lg.get_frame_on(), lg.get_texts()[0].get_fontsize())
               for ax in fig.axes for lg in [ax.get_legend()] if lg is not None and lg.get_texts()]
    edges = [(p, p.get_linewidth()) for ax in fig.axes for p in list(ax.patches) + list(ax.spines.values())]
    ticks = []
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for which, get in (("major", axis.get_major_ticks), ("minor", axis.get_minor_ticks)):
                tk = get()
                if tk:
                    t0 = tk[0]
                    ticks.append((axis, which, t0.label1.get_fontsize(), t0.tick1line.get_markersize(), t0.tick1line.get_markeredgewidth()))

    def on_resize(event=None):
        W, H = fig.get_size_inches()
        s = min(W / W0, H / H0)
        ox, oy = (W - W0 * s) / 2, (H - H0 * s) / 2
        for ax, (x, y, w, h) in boxes:
            ax.set_position([(ox + x * W0 * s) / W, (oy + y * H0 * s) / H, w * W0 * s / W, h * H0 * s / H])
        for t, (x, y) in places:
            t.set_position(((ox + x * W0 * s) / W, (oy + y * H0 * s) / H))
        for t, f in fonts:
            t.set_fontsize(f * s)
        for l, lw, ms in lines:
            l.set_linewidth(lw * s); l.set_markersize(ms * s)
        for p, lw in edges:
            p.set_linewidth(lw * s)
        for axis, which, f, ln, wd in ticks:
            axis.set_tick_params(which=which, labelsize=f * s, length=ln * s, width=wd * s)
        for ax, bb, ncols, frame, f in legends:
            ax.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(bb.x0, bb.y1), bbox_transform=ax.transAxes,
                      borderaxespad=0.0, ncols=ncols, frameon=frame, fontsize=f * s)
        fig.canvas.draw_idle()

    old = getattr(fig, "_vf_resize_cid", None)
    if old is not None:
        fig.canvas.mpl_disconnect(old)
    fig._vf_resize_cid = fig.canvas.mpl_connect("resize_event", on_resize)
    return on_resize


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--airfoil", help="NACA 4/5-digit designation (e.g. 2412, 23012) or a coordinate .dat file")
    ap.add_argument("--alpha", type=float, help="incidence, deg")
    ap.add_argument("--Re", type=float, help="Reynolds number")
    ap.add_argument("--h", type=float, default=0.005, help="field grid spacing in chords (default %(default)s)")
    ap.add_argument("--save", help="also write the figure to this file (png, pdf, ...)")
    ap.add_argument("--no-show", action="store_true", help="do not open a window")
    ap.add_argument("--csv", metavar="DIR", help="also write every number of the result as CSV files in DIR")
    ap.add_argument("--layout", choices=("auto", "landscape", "square", "portrait"), default="auto",
                    help="auto: portrait on an upright screen, else landscape")
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
    r, surf_w, out, meta = compute(loop, alpha, Re, a.h, family="naca" if re.fullmatch(r"\d{4,5}", airfoil.strip()) else None)
    layout = a.layout if a.layout != "auto" else ("portrait" if not a.no_show and screen_is_portrait() else "landscape")
    fig = figure(name, alpha, Re, r, surf_w, out, meta, layout=layout)
    if a.save:
        fig.savefig(a.save, dpi=400)
        print("wrote", a.save)
    if a.csv:
        for p in export_csv(a.csv, name, alpha, Re, r, surf_w, out, meta):
            print("wrote", p)
    import matplotlib.pyplot as plt
    if not a.no_show:
        if matplotlib.get_backend().lower() in ("agg", "pdf", "svg", "ps", "cairo", "template"):
            path = a.save or f"vortfoil_{re.sub(r'[^A-Za-z0-9]+', '_', name)}.png"
            if not a.save:
                fig.savefig(path, dpi=400)
            print(f"no interactive display (matplotlib backend {matplotlib.get_backend()}); the figure is in {path}")
        else:
            scale_with_window(fig)
            plt.show()


if __name__ == "__main__":
    main()
