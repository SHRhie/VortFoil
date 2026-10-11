# -*- coding: utf-8 -*-
"""One window for one airfoil: VortFoil's flow field and surface results together.

    python code/view_airfoil.py                                        # asks for the airfoil, incidence and Reynolds number
    python code/view_airfoil.py --airfoil 2412 --alpha 4 --Re 3e6
    python code/view_airfoil.py --airfoil path/to/clarky.dat --alpha 4 --Re 3e6 --save clarky.png

The airfoil is a NACA 4- or 5-digit designation or a coordinate file (Selig or Lednicer .dat). The window shows
  left   the RANS field from the field decoder: velocity along and across the freestream, C_p and nu_t, with the
         section at its incidence (wind frame);
  right  the surface pressure, the boundary layer of both surfaces and the coefficients, in two states: the XFOIL state
         (the vortfoil package: vortex sheet and boundary-layer operator, trained on XFOIL; C_l, C_d, C_m, the circulation
         lift, the stagnation point) and the RANS state (the field decoder's wall values, trained on AirfRANS k-omega SST
         fields: C_l, C_d with its pressure and friction parts, C_m, the boundary layer read from its flow, and the
         wall-function C_f), with every range warning.
The two states have their own training domains, and each warns about its own. figure(..., states=...) shows either state or
both (the application's two boxes); a state outside its training range (meta["xfoil_in_range"], meta["rans_in_range"]) is
drawn faded and labelled, and so are the field panels when the RANS state is the one outside.
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
    """Surface results (vortfoil package) and the field (predict_field), with their warnings. Both states are computed:
    the XFOIL state (r, the surface model) and the RANS state (meta: the field decoder's wall values meta["wall"] and boundary
    layer meta["boundary_layer"]). meta["xfoil_in_range"] / meta["rans_in_range"]: the incidence and the Reynolds number lie
    inside that model's training ranges (incidence and Re only: shape warnings, such as the surface model's 16-parameter fit
    and trailing-edge gap or the decoder's family and shape distance, do not count)."""
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
        out, meta = PF.predict(loop, alpha, Re, h=h, family=family, wall=True, bl=True, decoder=decoder or PF.DECODER)
    meta["xfoil_in_range"] = not bool(r.info["outside_training_range"][0])
    meta["rans_in_range"] = meta["in_training_range"]
    return r, surf_w, out, meta


DESIGNS = {"landscape": (16.0, 7.6), "square": (13.0, 10.5), "portrait": (9.5, 12.0)}   # drawn at these sizes (in); windows zoom
WRAP = {"landscape": 66, "square": 50, "portrait": 100}   # characters per line of the coefficients box
DESIGN = DESIGNS["landscape"]
STATES = ("xfoil", "rans")                                # what the surface panels can show: the surface model, the field decoder's wall values
STATE_NAME = {"xfoil": "XFOIL state", "rans": "RANS state"}
UP, LO, FADE = "#b2182b", "#2166ac", 0.35                 # upper and lower surface; the alpha of a state outside its training range
LINE, UNDER, FILL = 1.15, 0.35, 0.9                         # the coefficients box: line pitch (em), the extra after a line with an underscore (C_l, s_sep: it touched the next line's letters on a screen) and the share of the width a line may take
CHECKED_X = 0.1                                           # the figure shows the RANS-state boundary layer and C_f from x/c = 0.1, where the layer is checked against RANS (predict_field.boundary_layer)


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


def look(state, inside):
    """Line style of a state's curves in the surface panels: the XFOIL state solid, the RANS state dashed; a state outside its
    training range is drawn faded (alpha FADE) and dashed in its own way (XFOIL dashes, RANS dots)."""
    if inside:
        return dict(ls="-", lw=1.2, alpha=1.0) if state == "xfoil" else dict(ls="--", lw=0.9, alpha=0.8)
    return dict(ls=(0, (5, 2)) if state == "xfoil" else (0, (2, 1.5)), lw=1.2, alpha=FADE)


def figure(name, alpha, Re, r, surf_w, out, meta, fig=None, layout="landscape", states=STATES):
    """The result figure in one of three layouts: 'landscape' (fields left, surface right), 'square' (fields on top,
    surface pressure, boundary layer and coefficients in a row below) or 'portrait' (fields, then the two surface
    panels, then the coefficients). With `fig`, it is cleared and redrawn (the application reuses its figure).
    states: which states the surface panels and the coefficients show: 'xfoil' (the vortex sheet and boundary-layer operator),
    'rans' (the field decoder's wall pressure, boundary layer and wall-function C_f), or both (default). A state outside its
    training range (meta["xfoil_in_range"], meta["rans_in_range"]) is drawn faded, with "(outside training range)" in its
    legend entry and its numbers pale. The four field panels are the RANS field whatever the states; when the RANS state is outside
    its training range they are faded too (with their colour bars) and labelled."""
    import textwrap
    import matplotlib.pyplot as plt
    import vortfoil
    import predict_field as PF
    states = (states,) if isinstance(states, str) else tuple(states)
    if not states or not set(states) <= set(STATES):
        raise ValueError(f"states: 'xfoil', 'rans' or both (got {states!r})")
    states = tuple(s for s in STATES if s in states)
    inr = {"xfoil": bool(meta.get("xfoil_in_range", True)), "rans": bool(meta.get("rans_in_range", True))}
    tag = lambda s: STATE_NAME[s] + ("" if inr[s] else " (outside training range)")
    wl = meta.get("wall") if "rans" in states else None
    bl = meta.get("boundary_layer") if "rans" in states else None
    prime = [s for s in states if inr[s]] or list(states)               # the axes' limits follow the states inside their training range
    plt.rcParams.update(PF.STYLE)
    if fig is None:
        fig = plt.figure(figsize=DESIGNS[layout], constrained_layout=True)
    else:
        fig.clf()
        fig.set_size_inches(*DESIGNS[layout], forward=False)
        fig.set_layout_engine("constrained")
    if layout == "landscape":
        gs = fig.add_gridspec(1, 2, width_ratios=[2.25, 1])
        gl = gs[0].subgridspec(2, 2); gr = gs[1].subgridspec(3, 1, height_ratios=[0.85, 1.5, 1.05] if bl else [1.15, 1.0, 1.05])
        cell = dict(cp=gr[0], bl=gr[1], text=gr[2])
    elif layout == "square":
        gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0])
        gl = gs[0].subgridspec(2, 2); gm = gs[1].subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.9])
        cell = dict(cp=gm[0], bl=gm[1], text=gm[2])
    else:
        gs = fig.add_gridspec(3, 1, height_ratios=[1.85, 1.0, 0.8])
        gl = gs[0].subgridspec(2, 2); gm = gs[1].subgridspec(1, 2)
        cell = dict(cp=gm[0], bl=gm[1], text=gs[2])
    axf = [fig.add_subplot(gl[i, j]) for i in range(2) for j in range(2)]
    PF.draw_fields(fig, axf, out, meta)
    for ax in axf[2:]:
        ax.set_xlabel("x / c")
    for ax in axf[::2]:
        ax.set_ylabel("y / c")
    if not inr["rans"]:                                  # the field is the RANS state's too: faded (with its colour bar) and labelled like its curves
        for ax in axf:
            for c in ax.collections:
                c.set_alpha(FADE)
                if c.colorbar is not None:
                    c.colorbar.solids.set_alpha(FADE)
            ax.text(0.5, 0.97, "outside training range", transform=ax.transAxes, ha="center", va="top", fontsize=8, color="0.2")

    # surface pressure. XFOIL state: the sheet's panels 0-119 run TE -> LE on the upper surface, 120-239 LE -> TE on the lower;
    # RANS state: the field decoder's wall nodes
    ax = fig.add_subplot(cell["cp"])
    cps = []
    if "xfoil" in states:
        x, cp, st = r.x_cp[0], r.cp[0], look("xfoil", inr["xfoil"])
        cps += [cp] if "xfoil" in prime else []
        ax.plot(x[:120], cp[:120], color=UP, **st)
        ax.plot(x[120:], cp[120:], color=LO, **st)
        ax.plot([], [], color="0.25", label=tag("xfoil"), **st)
    if wl:
        xw, cw, up, st = np.asarray(wl["x"]), np.asarray(wl["cp"]), np.asarray(wl["upper"], bool), look("rans", inr["rans"])
        cps += [cw] if "rans" in prime else []
        for m, col in ((up, UP), (~up, LO)):
            o = np.argsort(xw[m])
            ax.plot(xw[m][o], cw[m][o], color=col, **st)
        ax.plot([], [], color="0.25", label=tag("rans"), **st)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("x / c"); ax.set_ylabel(r"$C_p$")
    if len(prime) < len(states) and cps:                                  # a faded state may run off the scale
        lo, hi = float(np.nanmin(np.r_[tuple(cps)])), float(np.nanmax(np.r_[tuple(cps)]))
        ax.set_ylim(hi + 0.05 * (hi - lo), lo - 0.05 * (hi - lo))
    ax.set_title("surface pressure: upper (red), lower (blue)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=9)

    # boundary layer: the thicknesses of both surfaces (XFOIL state: lines; RANS state: dots, from x/c 0.1 on, where it is checked
    # against RANS) and, below, the RANS state's wall-function C_f
    if bl:
        sub = cell["bl"].subgridspec(2, 1, height_ratios=[2.3, 1.0])
        ax = fig.add_subplot(sub[0]); axc = fig.add_subplot(sub[1], sharex=ax)
        ax.tick_params(labelbottom=False)
    else:
        ax, axc = fig.add_subplot(cell["bl"]), None
    axb, pts = ax, []                                     # pts: (x, thickness) of the states inside their training range, which set the axis
    if "xfoil" in states:
        xb, a = vortfoil.BL_X, look("xfoil", inr["xfoil"])["alpha"]
        for k, col in enumerate((UP, LO)):
            ax.plot(xb[1:], r.dstar[0, k, 1:], "-", color=col, lw=1.2, alpha=a)
            ax.plot(xb[1:], r.theta[0, k, 1:], "--", color=col, lw=1.2, alpha=a)
            pts += [(xb[1:], r.dstar[0, k, 1:]), (xb[1:], r.theta[0, k, 1:])] if "xfoil" in prime else []
        ax.plot([], [], color="0.25", lw=1.2, alpha=a, label=tag("xfoil") + (": " if inr["xfoil"] else "\n") + r"$\delta^*$ solid, $\theta$ dashed")
    if bl:
        a = 0.9 if inr["rans"] else FADE
        cf, read = [], False
        for side, col in (("upper", UP), ("lower", LO)):
            b = bl[side]
            m = np.asarray(b["x_over_c"], float) >= CHECKED_X
            x = np.asarray(b["x_over_c"], float)[m]
            for q, hollow in (("dstar", False), ("theta", True)):
                v = np.asarray(b[q], float)[m]
                ax.plot(x, v, "o", ls="none", ms=2.4, mew=0.7, color=col, mfc="none" if hollow else col, alpha=a)
                read = read or bool(np.isfinite(v).any())
                pts += [(x, v)] if "rans" in prime else []
            axc.plot(x, np.asarray(b["Cf"], float)[m], color=col, ls="-" if inr["rans"] else "--", lw=1.0, alpha=1.0 if inr["rans"] else FADE)
            cf.append(np.asarray(b["Cf"], float)[m])
        ax.plot([], [], marker="o" if read else "none", ls="none", ms=3, color="0.25", alpha=a,           # no dots to show: the entry says so
                label=tag("rans") + (": " if inr["rans"] else "\n") + (r"$\delta^*$ filled, $\theta$ open dots" if read else "no layer read"))
        lo, hi = min(float(np.nanmin(np.r_[tuple(cf)])), 0.0), float(np.nanmax(np.r_[tuple(cf)]))
        axc.set_ylim(lo - 0.06 * (hi - lo), hi + 0.12 * (hi - lo)); axc.axhline(0.0, color="0.7", lw=0.6, zorder=0)
        axc.set_ylabel(r"$C_f$"); axc.set_xlabel("x / c")
        axc.set_title(r"wall-function $C_f$, RANS state" + ("" if inr["rans"] else " (outside training range)"), loc="right", fontsize=8, color="0.25", pad=2)
    else:
        ax.set_xlabel("x / c")
    xs, ys = (np.concatenate([np.asarray(p[i], float) for p in pts]) if pts else np.zeros(0) for i in (0, 1))
    ok = np.isfinite(ys) & (ys > 0)
    ax.set_yscale("log"); ax.set_xlim(0, 1); ax.set_ylim(6e-6, 3.0 * (ys[ok].max() if ok.any() else 1e-2)); ax.set_ylabel("thickness / c")   # the 1e-5 tick clear of the corner
    ax.set_title("boundary layer: upper (red), lower (blue)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8, loc="upper left")

    # coefficients and warnings: a line, and whether it is pale (the state is outside its training range)
    ax = fig.add_subplot(cell["text"]); ax.axis("off")
    lines = [(f"{name}    alpha {meta['alpha_used']:.2f} deg    Re {Re:.3g}", False), ("", False)]
    if "xfoil" in states:
        pale = not inr["xfoil"]
        lines += [("XFOIL state: outside training range" if pale else "XFOIL state (surface model)", pale),
                  (f"  C_l {r.cl[0]: .4f}   C_d {r.cd[0]: .5f}   C_m {r.cm[0]: .4f}", pale),
                  (f"  circulation lift {r.cl_gamma[0]:.4f}, stagnation {r.stagnation[0]:.4f}", pale)]
    if wl:
        pale = not inr["rans"]
        lines += [("RANS state: outside training range" if pale else "RANS state (field decoder, wall function)", pale),
                  (f"  C_l {wl['cl']: .4f}   C_d {wl['cd']: .5f}   C_m {wl['cm']: .4f}", pale),
                  (f"  pressure {wl['cd_pressure']:.5f} + friction {wl['cd_friction']:.5f} (C_d)", pale),
                  (f"  field inputs: C_l^gamma {meta['cl_gamma']:.4f}, s_sep {meta['s_sep']:.3f}", pale)]
        if bl:
            lines += [("  boundary layer, x/c 0.1 to 1: " + ("not checked here" if pale else "checked vs RANS"), pale)]
            if not pale and meta["alpha_used"] >= 8.0:     # 1 test case in 3 from 8 deg on (field_boundary_layer_validation.json)
                lines += [("  lower surface: often 20%+ off at this alpha", False)]
    lines += [("C_m about the quarter chord, nose up positive", False), ("", False), ("warnings:", False)]
    warn = [f"surface: {m}" for m in surf_w] + [f"field: {m}" for m in meta["warnings"]]
    lines += [(l, False) for m in (warn or ["none: inside both training domains"]) for l in
              textwrap.wrap(f"- {m}", WRAP[layout], subsequent_indent="  ")]
    field = "RANS field decoder, wind frame" + ("" if inr["rans"] else ", outside its range")
    surface = "XFOIL and RANS states" if len(states) == 2 else STATE_NAME[states[0]] + ("" if inr[states[0]] else ", outside its range")
    where = {"landscape": f"left: {field}; right: surface, {surface}",
             "square": f"top: {field}; bottom row: surface, {surface}",
             "portrait": f"top: {field}; below: surface, {surface}"}[layout]
    if meta.get("decoder") == "field_decoder_rext_s2.npz":                # experimental (branch exp/rext-decoder)
        where = "field: Re-extended decoder, experimental; " + where
    fig.suptitle(rf"VortFoil — {name},  $\alpha$ = {meta['alpha_used']:.2f}$^\circ$,  Re = {Re:.3g}"
                 + ("   " if layout == "landscape" else "\n") + f"({where})", fontsize=12)
    fig.get_layout_engine().execute(fig)                                         # the cells' sizes
    # The boundary-layer legend sits in the upper left, and the axis is raised until the curves under it are clear of it (those to its
    # right may rise beside it). Where that would take more than two decades (thick layers), the legend goes where it fits ("best"),
    # in a box, since it then lies over something.
    lg = axb.get_legend()
    if lg and ok.any():
        bb, ab = lg.get_window_extent(), axb.get_window_extent()
        share, x1 = (bb.height + 8.0 * fig.dpi / 72) / ab.height, (bb.width + 4.0 * fig.dpi / 72) / ab.width + 0.01
        lo, t_all = np.log10(6e-6), np.log10(ys[ok].max())
        t_left = np.log10(max(ys[ok & (xs <= x1)].max(initial=0.0), 6e-6))
        up = max((t_left - share * lo) / (1.0 - share), t_all + 0.15)
        far = up > t_all + 2.0
        if far:
            lg.remove()
            axb.legend(frameon=True, fontsize=8, loc="best")
        axb.set_ylim(6e-6, 10 ** (t_all + 0.5 if far else up))
    # the type: 8.5 pt, smaller when the lines are many or long for the cell the layout gives them, and a list of warnings that would
    # not fit even at 6 pt ends in a pointer to the CSV file. One text per line, on a pitch of LINE em (UNDER more below a line with an
    # underscore), so that a pale line stays in line. The width is measured (matplotlib 3.11 rounds a character's advance to whole
    # pixels, 3.10 does not), and FILL of the cell is all it may take: a window zoom changes the rounding.
    H = ax.get_position().height * fig.get_figheight() * 72                      # points
    em = lambda ls: np.cumsum([0.0] + [LINE + (UNDER if "_" in t else 0.0) for t, _ in ls])     # in em: where each line starts, and the total
    keep = max(12, int(np.searchsorted(em(lines)[1:], 0.97 * H / 6.0, side="right")))            # lines that fit at 6 pt
    if len(lines) > keep:
        lines = lines[:keep - 1] + [("- ... (the rest is in the CSV coefficients)", False)]
    rows = em(lines)
    widest, fs, cell = max((t for t, _ in lines), key=len), min(8.5, 0.97 * H / rows[-1]), ax.get_window_extent().width
    for _ in range(4):
        probe = ax.text(0.0, 1.0, widest, family="monospace", fontsize=fs, transform=ax.transAxes)
        w = probe.get_window_extent().width
        probe.remove()
        if w <= FILL * cell:
            break
        fs *= 0.99 * FILL * cell / w
    fs = max(6.0, fs)
    for (txt, pale), y in zip(lines, rows):
        if txt:
            ax.text(0.0, 1.0 - (y + 0.9) * fs / H, txt, va="baseline", ha="left", family="monospace", fontsize=fs,
                    color="0.6" if pale else "k", transform=ax.transAxes).set_in_layout(False)
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
                    "field_s_sep", "field_Cl_RANS", "field_Cd_RANS", "field_Cd_RANS_pressure", "field_Cd_RANS_friction", "warnings",
                    "field_Cm_RANS", "XFOIL_state_in_training_range", "RANS_state_in_training_range"])
        wl = meta.get("wall") or dict(cl=np.nan, cd=np.nan, cd_pressure=np.nan, cd_friction=np.nan, cm=np.nan)
        w.writerow([name, alpha, a, Re] + [f"{float(v):.8g}" for v in (r.cl[0], r.cl_gamma[0], r.cd[0], r.cm[0], r.stagnation[0],
                                                                       r.ue_te[0], r.theta_te[0], r.dstar_te[0], meta["cl_gamma"], meta["s_sep"],
                                                                       wl["cl"], wl["cd"], wl["cd_pressure"], wl["cd_friction"])]
                   + [" | ".join([f"surface: {m}" for m in surf_w] + [f"field: {m}" for m in meta["warnings"]]), f"{float(wl['cm']):.8g}",
                      int(meta.get("xfoil_in_range", True)), int(meta.get("rans_in_range", True))])
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
    if meta.get("boundary_layer"):                                # the RANS state's: upper surface stations, then lower, each leading edge to trailing edge
        bl = meta["boundary_layer"]
        col = lambda q: np.r_[np.asarray(bl["upper"][q], float), np.asarray(bl["lower"][q], float)]
        table("boundary_layer_rans", ["upper_surface", "x_over_c", "Ue_over_U", "dstar_over_c", "theta_over_c", "H", "delta_over_c", "Cf", "separated"],
              [np.r_[np.ones(len(bl["upper"]["theta"])), np.zeros(len(bl["lower"]["theta"]))], col("x_over_c"), col("Ue"), col("dstar"), col("theta"),
               col("H"), col("delta"), col("Cf"), col("separated")])
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
                     RANS state (drag = pressure + friction from a wall function); every range warning in the warnings
                     column; then the RANS state's C_m about the quarter chord (field_Cm_RANS, nose up positive, from the
                     wall pressure and wall-function friction of _wall.csv) and, for each state, 1 if the incidence and the
                     Reynolds number lie inside its training range, 0 if not (the figure draws a state outside its range
                     faded).
_surface.csv         C_p of the vortex sheet at the 240 panel midpoints: panels 0-119 run from the trailing edge to the
                     leading edge on the upper surface (lower_surface 0), 120-239 back on the lower surface (1).
_boundary_layer.csv  momentum (theta) and displacement (dstar) thickness of both surfaces at x/c = 0..1 (121 stations), of the
                     surface model (the XFOIL state).
_field.csv           the RANS field of the decoder trained on AirfRANS k-omega SST at every grid point: x_wind, y_wind
                     (freestream along +x, section at its incidence), the same point in the chord frame (x_body, y_body),
                     inside_section (1: inside the section, values nan), velocity along and across the freestream, C_p and
                     nu_t/(U c), then the same in SI units (u, v in m/s, kinematic pressure p/rho in m2/s2, nu_t in m2/s).
_wall.csv            the field decoder's wall values at the section's nodes, in its RANS state (k-omega SST): x_body,
                     y_body, upper_surface (1) or lower (0), C_p, and C_f from a wall function (the decoder's velocity
                     2e-4 chord off the wall, Spalding's law; > 0 where the near-wall flow runs toward the trailing edge).
_boundary_layer_rans.csv
                     the boundary layer of the RANS state, read from the field decoder's flow on wall-normal rays (0.1
                     chord long) at every second section node with x/c from 0.01 to 0.99, the upper surface first, each
                     surface from the leading edge to the trailing edge: upper_surface (1 or 0), x_over_c, Ue_over_U (the
                     speed at the layer edge), dstar_over_c and theta_over_c (displacement and momentum thickness), H =
                     dstar/theta, delta_over_c (the edge distance), Cf (the wall function's, as in _wall.csv) and
                     separated (1: the flow 2e-4 chord off the wall runs against the outer flow; 0 ahead of the
                     stagnation point, where the attached flow near the nose runs toward the leading edge). nan: no layer
                     is read at that station (the ray crosses the section, no edge is found, fewer than 7 samples lie in
                     the layer, the edge lies beyond 0.8 of the ray, or a thickness is <= 0). Isolated stations just
                     behind the leading edge can be outliers, a very large H where theta is near zero (H = 138 at x/c
                     0.12 on the lower surface of NACA 2412 at 14 deg). The same extraction on the AirfRANS RANS fields
                     of the 200 test cases and on the decoder's fields differs, on attached stations at x/c 0.1 to 0.95,
                     by a median of 4% (upper surface) and 6% (lower) in theta and dstar. Case by case the lower surface
                     varies more: its median exceeds 20% in 31 of the 200 cases and 50% in 13, mostly from 8 deg on (21
                     of those 66 cases, and 9 of their 17 thin sections, t/c < 9%, against 10 of the 134 cases below 8
                     deg); single stations there can be off several times over. The upper surface's median exceeds 20% in
                     5 cases and 50% in none. At x/c < 0.1 the median differs by 8-10% (upper) and 27-48% (lower):
                     indicative there, so the figure starts at x/c 0.1. Outside the decoder's training range the layer is
                     often not read at all.
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
