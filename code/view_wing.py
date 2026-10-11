# -*- coding: utf-8 -*-
"""The result window of the 3D wing (code/wing_nll.py): the wing surface coloured by a section quantity, the spanwise
distribution with a draggable section cut, the section at the cut, the wing polar and the numbers, in the three layouts of
view_airfoil (landscape, square, portrait); and the CSV export of every number, or of a chosen subset.

    view = WingView(fig, res, layout, options, on_cut)       res = wing_nll.compute(...); on_cut(index, eta); nothing is drawn yet
    view.draw()                                              builds the figure (then view_airfoil.scale_with_window(fig))
    view.set_option("surface", "H")                          another quantity: redraws (scale_with_window again), keeps the view and the cut
    view.set_cut(0.4)                                        moves the cut to the strip nearest 2y/b = 0.4 (artists only)
    export_csv(folder, res, parts=["summary", "polar"], cut_index=view.cut_index)

Press and drag on the span panel to move the cut (not while the toolbar's pan or zoom tool is on). In the 3D panel, at true scale,
filling its cell at the first view and inside it at every turn: drag to turn the wing, a double click is back to the first view.
While the mouse is down only what moves is redrawn, on the figure as it was.
The colour of every surface point is its strip's 2D section value at the strip's effective incidence (strip theory), not a 3D
pressure solution. The numbers turn red when the lifting line found no solution (residual above NEWTON_TOL) or the SST state is far
outside its decoder's Reynolds range.
"""
import os, re, sys, csv, gc, textwrap, functools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import view_airfoil as VA  # noqa: E402     (adds this checkout's vortfoil package and code/field to sys.path)

SURFACE_QUANTITIES = {"xfoil": ["C_p", "theta/c", "delta*/c", "H"], "sst": ["C_p", "C_f"]}
SPAN_QUANTITIES = ["c_l", "c*c_l/c_bar", "alpha_eff", "alpha_ind", "Gamma", "c_d", "c_m", "Re", "chord", "twist"]
POLAR_PLOTS = ["C_L vs alpha", "C_D vs C_L", "L/D vs alpha", "C_M vs alpha"]
DEFAULTS = {"surface": "C_p", "span": "c_l", "polar": "C_L vs alpha", "cut_eta": 0.5}
CSV_PARTS = {"summary": ("summary", "the wing's numbers and warnings (one row)"),
             "stations": ("stations", "the wing definition (one row per station)"),
             "spanwise": ("spanwise", "every per-strip value along the span (one row per strip)"),
             "polar": ("polar", "the wing's coefficients over the incidence range"),
             "surface": ("surface", "the 3D surface points with every colour quantity"),
             "section_tables": ("section_tables", "each strip's 2D section c_l, c_d, c_m over its incidence table"),
             "cut": ("cut", "the raw values of the cut strip (surface pressure, boundary layer or wall values)")}
WRAP, MAX_DRAWN = 56, 64             # text line width; strips drawn
RECT, FILL, PAD = (0.012, 0.014, 0.976, 0.972), 0.9, 9   # layout rectangle (slack for the pads a zoom does not scale); share of its cell a text may fill; x tick pad (pt)
NEWTON_TOL = 1e-6                     # a lifting-line residual above this is no solution (wing_nll warns at the same value): the numbers turn red
VIEW = (34.0, 225.0)                 # elevation, azimuth (deg) of the first 3D view: the research viewer's 'oblique' (from upstream, above, the left wing nearer)
TURN, FIT3D = 0.458, 0.9              # deg turned per pixel dragged (the viewer's 0.008 rad); share of its cell the wing fills at VIEW (at most, at any turn)
UP, LO = "#b2182b", "#2166ac"         # upper and lower surface (as view_airfoil)

# colour quantity: (label, CSV column, what it is)
QUANT = {"C_p": (r"$C_p$", "Cp", "pressure coefficient of the strip's section at its effective incidence (the sheet's in the XFOIL state, "
                                 "the field decoder's wall pressure in the SST state)"),
         "theta/c": (r"$\theta/c$", "theta_over_c", "momentum thickness / chord of the strip's boundary layer (XFOIL state)"),
         "delta*/c": (r"$\delta^*/c$", "dstar_over_c", "displacement thickness / chord of the strip's boundary layer (XFOIL state)"),
         "H": (r"$H=\delta^*/\theta$", "H", "shape factor delta*/theta (nan where either thickness is <= 0, near the leading edge)"),
         "C_f": (r"$C_f$", "Cf", "skin friction coefficient from the decoder's wall function (> 0: the near-wall flow runs toward the "
                                 "trailing edge; SST state)")}
# spanwise quantity: (strips key, axis label, legend label of the line, extra dashed lines [(key, legend label, colour, only in state)])
SPAN = {"c_l": ("cl", r"$c_l$", None, []), "c*c_l/c_bar": ("cl_c_over_cbar", r"$c\,c_l/\bar c$", "loading", []),
        "alpha_eff": ("alpha_eff_deg", r"$\alpha_{eff}$ (deg)", "effective", [("alpha_geo_deg", "geometric", "0.45", None)]),
        "alpha_ind": ("alpha_ind_deg", r"$\alpha_{ind}$ (deg)", None, []), "Gamma": ("gamma", r"$\Gamma$ (m$^2$/s)", None, []),
        "c_d": ("cd", r"$c_d$", "total", [("cd_pressure", "pressure", LO, "sst"), ("cd_friction", "friction", "#4d9221", "sst")]),
        "c_m": ("cm", r"$c_m$ (about $c/4$, nose up +)", None, []), "Re": ("Re", "Re", None, []), "chord": ("chord", "chord (m)", None, []),
        "twist": ("twist_deg", "twist (deg)", None, [])}
# polar plot: (x key, y key, x label, y label); the keys are the polar's and the coefficients'
POLAR = {"C_L vs alpha": ("alpha", "CL", r"$\alpha$ (deg)", r"$C_L$"), "C_D vs C_L": ("CL", "CD", r"$C_L$", r"$C_D$"),
         "L/D vs alpha": ("alpha", "LD", r"$\alpha$ (deg)", "L/D"), "C_M vs alpha": ("alpha", "CM", r"$\alpha$ (deg)", r"$C_M$ (root $c/4$, MAC)")}


def _e(x):
    """3.4e6 style number"""
    return f"{x:.3g}".replace("e+0", "e").replace("e+", "e")


def short(text, n):
    """text cut to n characters with '...' (at a space when there is one: shorten gives only '...' for one long word)"""
    t = textwrap.shorten(text, n, placeholder="...")
    return t if t != "..." else text[:n - 3] + "..."


def _raw(res, q, k):
    """strip k's values of colour quantity q at the model's own stations: [(x/c, values) of the upper, of the lower surface]"""
    d = res["strip_detail"]
    if res["state"] == "sst":                                        # wall nodes (chord frame), any order
        x, v, up = d["x"][k], d["cp" if q == "C_p" else "cf"][k], d["upper"][k]
        return [(x[m][o], v[m][o]) for m in (up, ~up) for o in [np.argsort(x[m])]]
    if q == "C_p":                                                   # sheet panels: 0-119 upper TE -> LE, 120-239 lower LE -> TE
        return [(d["x_cp"][k][s], d["cp"][k][s]) for s in (slice(0, 120), slice(120, 240))]
    th, ds = np.asarray(d["theta"][k], np.float64), np.asarray(d["dstar"][k], np.float64)      # (2, 121): upper, lower
    v = {"theta/c": th, "delta*/c": ds, "H": np.where((th > 0) & (ds > 0), ds / np.where(th > 0, th, 1.0), np.nan)}[q]
    return [(d["BL_X"], v[0]), (d["BL_X"], v[1])]


def range_message(res):
    """the sentence that says the SST state's numbers mean nothing here (more than half the strips outside the decoder's Reynolds range), else None"""
    return (res.get("sst_range") or {}).get("message")


@functools.lru_cache(None)
def _quad_collection():
    """The surface's collection class: mplot3d's Poly3DCollection; before matplotlib 3.11 with the projection and the depth sort of its
    quads done in numpy (it does them one polygon at a time in Python, ~17 us each: 50 ms of every redraw of the 3,000-quad wing)."""
    import matplotlib
    from matplotlib.collections import PolyCollection
    from mpl_toolkits.mplot3d import proj3d
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if matplotlib.__version_info__ >= (3, 11):
        return Poly3DCollection

    class Quads(Poly3DCollection):
        def __init__(self, quads, *a, **k):               # quads: (N, 4, 3)
            super().__init__(quads, *a, **k)
            self._quads = np.asarray(quads, np.float64)

        def do_3d_projection(self):
            P = self._quads
            x, y, z = proj3d.proj_transform(P[..., 0].ravel(), P[..., 1].ravel(), P[..., 2].ravel(), self.axes.M)
            zq = z.reshape(-1, 4).mean(1)
            i = np.argsort(-zq, kind="stable")            # furthest first, as zsort 'average'
            PolyCollection.set_verts(self, np.stack([x, y], -1).reshape(-1, 4, 2)[i], True)
            self._facecolors2d, self._edgecolors2d = self._facecolor3d[i], self._edgecolor3d[i]
            return float(zq.min())
    return Quads


class WingView:
    """The wing result in one figure (draw), with a draggable section cut (cut_index, cut_eta). res: wing_nll.compute's dict.
    options: surface (a SURFACE_QUANTITIES item of the result's state; C_p when it is not one), span, polar, cut_eta (2y/b,
    the strip nearest to it) and view ((elevation, azimuth) in deg of the 3D view: `pose`). on_cut(index, eta) is
    called when a mouse drag moves the cut to another strip (not by set_cut). layout (a view_airfoil.DESIGNS key) may be set
    before draw()."""

    def __init__(self, fig, res, layout="landscape", options=None, on_cut=None):
        old = getattr(fig, "wing_view", None)
        if old is not None:                                          # the figure is reused: its earlier view lets go
            old.disconnect()
        self.fig, self.res, self.layout, self.on_cut = fig, res, layout, on_cut
        o = dict(DEFAULTS, **{k: v for k, v in (options or {}).items() if v is not None})     # other keys are not read
        if o["surface"] not in SURFACE_QUANTITIES[res["state"]]:
            o["surface"] = "C_p"                                      # e.g. 'H' left over from the other state
        self.surface, self.span, self.polar = (self._check(k, o[k]) for k in ("surface", "span", "polar"))
        self._pose = tuple(float(a) for a in (o.get("view") or VIEW)[:2])      # (elevation, azimuth)
        self.cut_index = self._nearest(self._check("cut_eta", o["cut_eta"])); self.cut_eta = float(res["strips"]["eta"][self.cut_index])
        self.ax3d = self.ax_span = self._turn = self._bg = None
        self._drag, self._anim = False, ()
        cv = fig.canvas
        self._cids = [cv.mpl_connect(e, f) for e, f in (("button_press_event", self._press), ("motion_notify_event", self._motion),
                                                         ("button_release_event", self._release))]
        fig.wing_view = self

    @property
    def pose(self):
        """the 3D view as it is now (the user turns it): (elevation, azimuth in deg); pass it as options['view'] to keep it"""
        ax = self.ax3d
        return (ax.elev, ax.azim) if ax is not None else self._pose

    # ------------------------------------------------------------------ options and the cut
    def _check(self, key, value):
        ok = {"surface": SURFACE_QUANTITIES[self.res["state"]], "span": SPAN_QUANTITIES, "polar": POLAR_PLOTS}
        if key == "cut_eta":
            try:
                v = float(value)
            except (TypeError, ValueError):
                v = float("nan")
            if not np.isfinite(v):
                raise ValueError(f"cut_eta: 2y/b must be a number from -1 to 1 (got {value!r})")
            return v
        if key not in ok:
            raise ValueError(f"unknown display option '{key}' (surface, span, polar, cut_eta)")
        if value not in ok[key]:
            raise ValueError(f"{key}: '{value}' is not one of {', '.join(ok[key])}")
        return value

    def _nearest(self, eta):
        return int(np.argmin(np.abs(self.res["strips"]["eta"] - np.clip(eta, -1.0, 1.0))))

    def set_option(self, key, value):
        """Change a display option and redraw (no recomputation). 'cut_eta' moves the cut like set_cut. After a redraw, call
        view_airfoil.scale_with_window(fig) again, as after draw()."""
        value = self._check(key, value)
        if key == "cut_eta":
            return self.set_cut(value)
        setattr(self, key, value)
        self.draw()

    def set_cut(self, eta):
        """Move the cut to the strip nearest 2y/b = eta (clipped to -1..1) and update its artists; on_cut is not called."""
        self._cut(self._check("cut_eta", eta))

    def _cut(self, eta, notify=False):
        k = self._nearest(eta)
        if k == self.cut_index:
            return
        self.cut_index, self.cut_eta = k, float(self.res["strips"]["eta"][k])
        if self.ax_span is not None:
            self._update_cut()
        if notify and self.on_cut is not None:
            self.on_cut(k, self.cut_eta)

    def _update_cut(self):
        k, eta = self.cut_index, self.cut_eta
        if self._drag:                                    # a drag moves the cut: only what moves is redrawn, on the figure as it was
            self._fast_on(self._cut_artists(), self.fig.bbox)
        self._vline.set_xdata([eta, eta]); self._dot.set_data([eta], [self._yspan[k]])
        self._outline.set_data_3d(*self._loop3d(k))
        for ln, (x, v) in zip(self._sec, _raw(self.res, self.surface, k)):
            ln.set_data(x, v)
        self._poly.set_xy(self._loop2d(k)); self._info.set_text(self._label(k))
        self.fig.canvas.draw_idle()

    def _cut_artists(self):
        return (self._vline, self._dot, *self._sec, self._poly, self._info, self._outline)

    # ------------------------------------------------------------------ the 3D view
    def reset_view(self):
        """the first view: VIEW and the first limits, where a toolbar tool moved them (a double click in the 3D panel)"""
        self._fast_off(False)
        ax = self.ax3d
        ax.view_init(elev=VIEW[0], azim=VIEW[1])
        for set_, lim in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), self._lim0):
            set_(*lim)
        self.fig.canvas.draw_idle()

    def _fast_on(self, anim, box):
        """Until _fast_off, a draw redraws only the artists `anim` (blitted into `box`) on the figure as it is now: ~10 ms a step,
        not the whole figure's ~70 (every panel's text again). The canvas's draw is shadowed, not draw_idle: Tk's idle draws still merge."""
        cv, anim = self.fig.canvas, tuple(anim)
        if self._bg is not None:
            if self._anim == anim:
                return
            self._fast_off(False)
        if not (hasattr(cv, "copy_from_bbox") and hasattr(cv, "restore_region")):      # not an Agg canvas: whole redraws
            return
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        self._anim, self._box, self._moved = anim, box, False
        for a in anim:
            a.set_animated(True)                          # left out of the draw that takes the background
        FigureCanvasAgg.draw(cv)                          # render only: the window keeps the figure it shows until the first fast draw
        self._bg = cv.copy_from_bbox(self.fig.bbox)
        cv.draw = self._fast_draw
        gc.freeze()                                       # a collection of the big heap (TensorFlow's objects too) would stop a frame for ~70 ms

    def _fast_draw(self):
        cv, self._moved = self.fig.canvas, True
        cv.restore_region(self._bg)
        for a in self._anim:
            self.fig.draw_artist(a)
        cv.blit(self._box)

    def _fast_off(self, redraw=True):
        """Back to whole draws, one now when something was drawn fast (redraw)."""
        if self._bg is None:
            return
        cv, self._bg = self.fig.canvas, None
        gc.unfreeze()
        cv.__dict__.pop("draw", None)
        for a in self._anim:
            a.set_animated(False)
        self._anim = ()
        if redraw and self._moved:                        # (a click that moved nothing leaves the window as it is)
            cv.draw_idle()

    # ------------------------------------------------------------------ mouse
    def _moving_tools(self):
        tb = getattr(self.fig.canvas, "toolbar", None)
        return tb is not None and str(getattr(tb, "mode", "")) != ""          # pan or zoom: the mouse is theirs

    def _press(self, ev):
        if ev.button != 1 or self.ax_span is None or self._moving_tools():
            return
        if ev.inaxes is self.ax3d:                        # turn the wing (here); a double click: the first view
            if ev.dblclick:
                self.reset_view()
            else:
                self._turn = (ev.x, ev.y, self.ax3d.elev, self.ax3d.azim)
                self._fast_on((self.ax3d,), self.ax3d.bbox)           # ready before the first move: no delay at the start of a drag
        elif ev.inaxes is self.ax_span:
            self._drag = True
            self._motion(ev)

    def _motion(self, ev):
        if ev.x is None:
            return
        if self._turn:
            x, y, elev, azim = self._turn                 # TURN deg a pixel, like the research viewer; not over the poles
            self.ax3d.view_init(elev=float(np.clip(elev - (ev.y - y) * TURN, -89.5, 89.5)), azim=azim - (ev.x - x) * TURN)
            self.fig.canvas.draw_idle()
        elif self._drag:                                  # through the axes' transform: also outside the axes and after a window zoom
            self._cut(self.ax_span.transData.inverted().transform((ev.x, ev.y))[0], notify=True)

    def _release(self, ev):
        self._drag = self._turn = None
        self._fast_off()

    def disconnect(self):
        """Remove this view's mouse connections (before the figure is used for something else)."""
        self._fast_off(False)
        for c in self._cids:
            self.fig.canvas.mpl_disconnect(c)
        self._cids, self._drag, self._turn = [], False, None

    # ------------------------------------------------------------------ the figure
    def draw(self):
        """(Re)build the whole figure for self.layout and the options, as view_airfoil.figure does (clear, design size,
        constrained layout). The wing keeps its turn. Returns the figure."""
        import matplotlib
        import predict_field as PF
        if self.layout not in VA.DESIGNS:
            raise ValueError(f"layout: '{self.layout}' is not one of {', '.join(VA.DESIGNS)}")
        matplotlib.rcParams.update(PF.STYLE)
        fig, res = self.fig, self.res
        self._fast_off(False)                                         # the figure is built anew: no cached background, the view as it is
        self._pose, self.ax3d = self.pose, None
        fig.clf()
        fig.set_size_inches(*VA.DESIGNS[self.layout], forward=False)
        fig.set_layout_engine("constrained", w_pad=0.1, h_pad=0.14, wspace=0.03, hspace=0.03, rect=RECT)
        self._lim = self._limits(self.surface)
        c = self._cells()
        self.ax3d = self._wing3d(c["w3d"])
        self.ax_span = fig.add_subplot(c["span"]); self._span(self.ax_span)
        head = c["head"].subgridspec(1, 2, width_ratios=[0.22, 0.78])
        self._section(fig.add_subplot(c["sec"]), fig.add_subplot(head[0]), fig.add_subplot(head[1]))
        pol = fig.add_subplot(c["polar"]); self._polarplot(pol)
        ax = fig.add_subplot(c["text"]); ax.axis("off")
        ax.set_in_layout(self.layout != "portrait")                   # portrait: the layout would cut its height by the polar's labels (set below)
        self._text = ax.text(0.0, 1.0, self._summary(), va="top", ha="left", family="monospace", fontsize=8.5, linespacing=1.1, transform=ax.transAxes)
        self._text.set_in_layout(False)                               # the cell's size decides the text's, not the other way round
        if res["newton"]["residual"] > NEWTON_TOL or range_message(res):
            self._text.set_color(UP)                                  # no lifting-line solution, or a decoder far outside its Reynolds range: not numbers to read
        name = short(res["wing"]["name"], 80).replace("$", r"\$")      # _fit below shrinks a long one to the figure's width
        state = "SST state" if res["state"] == "sst" else "XFOIL state"
        fig.suptitle(rf"VortFoil wing — {name},  $\alpha$ = {res['alpha']:.2f}$^\circ$,  V = {res['V']:g} m/s,  {state}" + "\n"
                     + "lifting line on VortFoil sections; colour: each strip's 2D section values; wind frame, true scale", fontsize=12)
        fig.draw_without_rendering()                                  # the layout, to fit the texts
        if self.layout == "portrait":                                 # right of the polar, from its top down to the figure's lower edge
            p = pol.get_position(); x = p.x1 + 0.02
            ax.set_position([x, RECT[1], RECT[0] + RECT[2] - x, p.y1 - RECT[1]])
        self._fit(fig._suptitle, fig)
        self._fit(self._text, ax, keep_lines=14 + (5 if range_message(res) else 0), floor=6.0)    # 14: down to the coefficients and 'warnings:' (5: the lead sentence)
        self._fit(self._info, self._info.axes, self._widest())
        return fig

    def _cells(self):
        """the grid cells of the panels: w3d, span, head (the cut's label and outline), sec (its plot), polar, text"""
        f, lay = self.fig, self.layout
        if lay == "landscape":      # 3D | span over the section | polar over the numbers
            gs = f.add_gridspec(1, 3, width_ratios=[1.65, 1.0, 1.1])
            a, b = gs[1].subgridspec(3, 1, height_ratios=[1.0, 0.44, 1.1]), gs[2].subgridspec(2, 1, height_ratios=[1.0, 1.1])
            return dict(w3d=gs[0], span=a[0], head=a[1], sec=a[2], polar=b[0], text=b[1])
        if lay == "square":         # 3D with the numbers beside it, a row of span | section | polar below
            gs = f.add_gridspec(2, 1, height_ratios=[1.5, 1.0])
            t, b = gs[0].subgridspec(1, 2, width_ratios=[1.3, 1.0]), gs[1].subgridspec(1, 3, width_ratios=[1.0, 1.15, 1.0])
            m = b[1].subgridspec(2, 1, height_ratios=[0.42, 1.0])
            return dict(w3d=t[0], text=t[1], span=b[0], head=m[0], sec=m[1], polar=b[2])
        gs = f.add_gridspec(3, 1, height_ratios=[1.4, 1.0, 0.9])     # 3D across the top, span | section, polar | numbers
        a, b = gs[1].subgridspec(1, 2), gs[2].subgridspec(1, 2)
        m = a[1].subgridspec(2, 1, height_ratios=[0.66, 1.0])
        return dict(w3d=gs[0], span=a[0], head=m[0], sec=m[1], polar=b[0], text=b[1])

    def _fit(self, txt, ax, sample=None, keep_lines=0, floor=5.0):
        """Shrink a text's font (to `floor` pt) until it fills at most FILL of its axes (or the figure): text sizes do not scale
        exactly with a zoom. `sample` is a text at least as large as any it will show. keep_lines > 0: when it is still too tall,
        its last lines (warnings) go, down to that many (the CSV summary has them all)."""
        keep = txt.get_text()
        if sample is not None:
            txt.set_text(sample)
        for _ in range(4):
            bb, ab = txt.get_window_extent(), ax.get_window_extent()
            f = min(1.0, FILL * ab.width / bb.width, FILL * ab.height / bb.height)
            if f > 0.995:
                break
            txt.set_fontsize(max(floor, txt.get_fontsize() * f * 0.97))
        txt.set_text(keep)
        lines, k = keep.split("\n"), keep.count("\n") + 1
        while keep_lines and k > keep_lines and txt.get_window_extent().height > FILL * ab.height:
            k -= 1
            txt.set_text("\n".join(lines[:k] + ["... (the rest is in the CSV summary)"]))

    def _widest(self):
        """the longest of the cut labels, line by line (a text at least as wide as any cut's)"""
        rows = [self._label(k).split("\n") for k in range(self.res["wing"]["n"])]
        return "\n".join(max((r[i] for r in rows), key=len) for i in range(len(rows[0])))

    # ---- colours
    def _limits(self, q):
        """the quantity's range over every strip, without the outliers: the section axis, and the colour limits (not C_p's)"""
        v = np.concatenate([np.ravel(a) for k in range(self.res["wing"]["n"]) for _, a in _raw(self.res, q, k)])
        v = v[np.isfinite(v)]
        if v.size == 0:
            return 0.0, 1.0
        lo, hi = np.percentile(v, [0.1, 99.9] if q == "C_p" else [1, 99])
        lo = 0.0 if q in ("theta/c", "delta*/c") or (q == "C_f" and lo > 0) else lo      # thicknesses from zero (noise below it at the nose)
        return (float(lo), float(hi)) if hi - lo > 1e-12 else (float(lo) - 0.5, float(lo) + 0.5)

    def _look(self, q):
        import matplotlib
        from matplotlib import colors
        cmap = lambda name: matplotlib.colormaps[name].with_extremes(bad="0.85")        # nan (H at the nose) is grey
        if q == "C_p":
            return cmap("RdYlBu_r"), colors.TwoSlopeNorm(vmin=-1.5, vcenter=0.0, vmax=1.0)   # white-yellow = freestream
        lo, hi = self._lim
        if q == "C_f" and lo < -0.15 * hi:                            # a real reversed flow shows (a diverging map needs room below zero)
            return cmap("RdBu_r"), colors.TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
        return cmap("viridis"), colors.Normalize(vmin=lo, vmax=hi)

    # ---- 3D
    def _loop3d(self, k):
        S = self.res["surface"]
        return [np.r_[S["upper"][c][k], S["lower"][c][k][::-1]] for c in "XYZ"]   # the section at strip k: upper LE -> TE, lower TE -> LE

    def _wing3d(self, cell):
        from matplotlib import cm, ticker
        from mpl_toolkits.mplot3d import proj3d
        S, w, q = self.res["surface"], self.res["wing"], self.surface
        ax = self.fig.add_subplot(cell, projection="3d")
        cmap, norm = self._look(q)
        step = -(-w["n"] // MAX_DRAWN)         # ponytail: above MAX_DRAWN strips every step-th strip is drawn (a redraw is ~0.3 s at 200)
        rows = np.unique(np.r_[np.arange(0, w["n"], step), w["n"] - 1])
        quads, vals = [], []
        for side in ("upper", "lower"):
            P, v = np.stack([S[side][c][rows] for c in "XYZ"], -1), S[side][q][rows]
            corner = lambda a: np.stack([a[:-1, :-1], a[:-1, 1:], a[1:, 1:], a[1:, :-1]], 2)
            quads.append(corner(P).reshape(-1, 4, 3))
            vals.append(np.ma.masked_invalid(corner(v)).mean(axis=2).filled(np.nan).ravel())      # a quad of NaN corners is grey
        quads = np.concatenate(quads)
        fc = cmap(norm(np.ma.masked_invalid(np.concatenate(vals))))
        ax.add_collection3d(_quad_collection()(quads, facecolors=fc, edgecolors=fc, linewidths=0.2, rasterized=True))
        U, L = S["upper"], S["lower"]
        x, z = (np.r_[U[c].ravel(), L[c].ravel()] for c in "XZ"); b2 = 0.5 * w["b"]
        ax.set_xlim(x.min(), x.max()); ax.set_ylim(-b2, b2); ax.set_zlim(z.min(), z.max())
        ax.set_box_aspect((np.ptp(x), 2 * b2, np.ptp(z))); ax.set_proj_type("ortho"); ax.set_axis_off()    # true scale: the box has the data's proportions
        ax.disable_mouse_rotation()                                   # the left button turns the wing here (_motion); mplot3d's own turn, move and zoom are off
        self._outline, = ax.plot(*self._loop3d(self.cut_index), color="k", lw=1.8, zorder=10)  # above the surface (collections sort below)
        # mplot3d draws in a square inside its cell, its window sized for the box's circumscribed sphere (any turn fits). Here the window
        # is the cell's shape, centred on the wing at VIEW, where the wing fills FIT3D of its limiting dimension; it grows only as far as
        # a turn needs to keep the wing within FIT3D of it, and is never smaller (no zoom to bring a clipped wing back; circles stay circles)
        ax.view_init(elev=VIEW[0], azim=VIEW[1])
        pts = np.concatenate([np.stack([a[c].ravel() for c in "XYZ"], -1) for a in (U, L)])
        px, py, _ = proj3d.proj_transform(pts[:, 0], pts[:, 1], pts[:, 2], ax.get_proj())
        cx, cy, ex, ey = (px.max() + px.min()) / 2, (py.max() + py.min()) / 2, np.ptp(px) / 2, np.ptp(py) / 2

        def fill_cell(position=None):                                 # Axes3D.apply_aspect, called by every draw
            pos = ax.get_position(original=True) if position is None else position
            ax._set_position(pos, "active")
            cell_px = pos.transformed(ax.get_figure().transSubfigure)
            if cell_px.width <= 0 or cell_px.height <= 0:                 # a collapsed cell (a layout that failed): nothing to fit
                return
            r = cell_px.width / cell_px.height
            X, Y, _ = proj3d.proj_transform(pts[:, 0], pts[:, 1], pts[:, 2], ax.get_proj())      # the wing at the pose it has now
            h = max(ey, ex / r, np.abs(Y - cy).max(), np.abs(X - cx).max() / r, 1e-9) / FIT3D   # half the window's height (projection units); its width is r times that
            ax.viewLim.intervalx, ax.viewLim.intervaly = (cx - r * h, cx + r * h), (cy - h, cy + h)
        ax.apply_aspect = fill_cell
        self.ax3d, self._lim0 = ax, (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())
        ax.view_init(elev=self._pose[0], azim=self._pose[1])
        side = self.layout == "square"                                # in the wide, short cells the bar goes below the wing; beside it in the square layout
        cb = self.fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, location="right" if side else "bottom", shrink=0.6, pad=0.02, aspect=25 if side else 40,
                               label=QUANT[q][0])
        if q != "C_p":                                                # a few ticks with short labels (thicknesses are ~1e-3)
            tv = ticker.MaxNLocator(6).tick_values(*self._lim)
            cb.set_ticks(tv[(tv >= self._lim[0]) & (tv <= self._lim[1])])    # inside the range: a tick outside would widen the bar
            (cb.ax.yaxis if side else cb.ax.xaxis).set_major_formatter(ticker.FormatStrFormatter("%.3g"))
        ax.set_title(f"{QUANT[q][0]} on the wing: drag to turn, double-click to reset", fontsize=10)
        return ax

    # ---- span, section, polar
    def _span(self, ax):
        s, k, state = self.res["strips"], self.cut_index, self.res["state"]
        key, label, main, extra = SPAN[self.span]
        ax.plot(s["eta"], s[key], color=UP, lw=1.2, label=main)
        for ek, el, col, only in extra:
            if only in (None, state):
                ax.plot(s["eta"], s[ek], "--", color=col, lw=1.0, label=el)
        if key == "cl_c_over_cbar":                                   # the elliptic loading with the same C_L
            e = np.linspace(-1, 1, 201)
            ax.plot(e, 4 / np.pi * self.res["coef"]["CL"] * np.sqrt(1 - e ** 2), "--", color="0.5", lw=1.0, label="elliptic, same $C_L$")
        self._yspan = np.asarray(s[key])
        self._vline = ax.axvline(self.cut_eta, color="k", lw=1.0, ls="--", zorder=3)
        self._dot, = ax.plot([self.cut_eta], [self._yspan[k]], "o", color="k", ms=4, zorder=4)
        ax.set_xlim(-1, 1); ax.set_xlabel(r"$2y/b$"); ax.set_ylabel(label); ax.tick_params(axis="x", pad=PAD)   # pad: a y tick at the corner (a zoom does not scale it)
        ax.set_title("drag to move the section cut", fontsize=10)
        n = len(ax.get_legend_handles_labels()[0])
        if n > 1:                                                     # the legend in a band above the curves: no line runs under it
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + (0.1 + 0.1 * -(-n // 2)) * (hi - lo))
            ax.legend(fontsize=8, framealpha=0.9, edgecolor="0.8", loc="upper center", ncols=2)      # a frame: the cut line may pass behind it

    def _loop2d(self, k):
        L = self.res["loops"]
        return np.asarray(L["coords"][L["index"][k]])

    def _label(self, k):
        s, L = self.res["strips"], self.res["loops"]
        sec = short(L["label"][L["index"][k]], 44).replace("$", r"\$")        # no mathtext from a file name
        return (f"2y/b {s['eta'][k]:.2f}, y {s['y'][k]:.2f} m, c {s['chord'][k]:.3g} m, Re {_e(s['Re'][k])}\n"
                f"alpha_eff {s['alpha_eff_deg'][k]:.1f} deg, c_l {s['cl'][k]:.3f}, c_d {s['cd'][k]:.4f}, c_m {s['cm'][k]:.3f}\n"
                f"alpha_geo {s['alpha_geo_deg'][k]:.1f} deg (twist {s['twist_deg'][k]:.1f}), alpha_ind {s['alpha_ind_deg'][k]:.2f} deg\n"
                f"section: {sec}")

    def _section(self, ax, thumb, info):
        q, k = self.surface, self.cut_index
        self._sec = [ax.plot(x, v, color=col, lw=1.2)[0] for col, (x, v) in zip((UP, LO), _raw(self.res, q, k))]
        lo, hi = self._lim
        pad = 0.05 * (hi - lo)
        ax.set_xlim(0, 1); ax.set_ylim((hi + pad, lo - pad) if q == "C_p" else (lo - pad, hi + pad))   # C_p axis inverted
        ax.tick_params(axis="x", pad=PAD); ax.set_xlabel("x / c"); ax.set_ylabel(QUANT[q][0])
        ax.set_title(f"{QUANT[q][0]} at the cut: upper (red), lower (blue)", fontsize=10)
        if lo < 0 < hi:
            ax.axhline(0.0, color="0.7", lw=0.6, zorder=0)
        r = max(np.abs(np.asarray(c)[:, 1]).max() for c in self.res["loops"]["coords"]) * 1.1     # the section's outline, unit chord
        self._poly, = thumb.fill(*self._loop2d(k).T, color="0.8", ec="0.2", lw=0.8)
        thumb.set_xlim(-0.03, 1.03); thumb.set_ylim(-r, r); thumb.set_aspect("equal"); thumb.axis("off")
        info.axis("off")
        self._info = info.text(0.0, 0.5, self._label(k), va="center", ha="left", fontsize=9, transform=info.transAxes)
        self._info.set_in_layout(False)

    def _polarplot(self, ax):
        p, c = self.res["polar"], self.res["coef"]
        xk, yk, xl, yl = POLAR[self.polar]
        ax.plot(p[xk], p[yk], color=UP, lw=1.2)
        ax.plot([self.res["alpha"] if xk == "alpha" else c[xk]], [c[yk]], "o", color="k", ms=5)
        ax.tick_params(axis="x", pad=PAD); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title("wing polar, dot: this incidence", fontsize=10)

    # ---- numbers
    def _summary(self):
        r, w, c = self.res, self.res["wing"], self.res["coef"]
        sst, msg = r["state"] == "sst", range_message(r)
        L = ([textwrap.fill(msg, WRAP), ""] if msg else []) + [short(w["name"], WRAP),
             "SST state (RANS field decoder, fully turbulent)" if sst else "XFOIL state (VortFoil surface model, free transition)",
             f"V {r['V']:g} m/s, alpha {r['alpha']:.2f} deg (root chord)",
             f"span {w['b']:.4g} m, S {w['S']:.4g} m^2, AR {w['AR']:.2f}",
             f"MAC {w['MAC']:.4g} m, taper {w['taper']:.3g}, {w['n']} strips",
             f"Re {_e(w['Re_root'])} (root), {_e(w['Re_tip'])} (tip)", "",
             f"C_L  {c['CL']: .4f}   C_Di {c['CDi']: .5f}   e " + (f"{c['e']:.3f}" if np.isfinite(c["e"]) else "-"),
             f"C_D0 {c['CD0']: .5f}   C_D  {c['CD']: .5f}   L/D {c['LD']:.1f}"]
        if sst:
            L.append(f"  C_D0 = pressure {c['CD0_pressure']:.5f} + friction {c['CD0_friction']:.5f}")
        L += [f"C_M  {c['CM']: .4f}  (root c/4, MAC; nose up +)",
              f"Newton {r['newton']['iterations']} steps, residual {r['newton']['residual']:.0e}"
              + (f", {c['strips_past_stall']} past stall" if c["strips_past_stall"] else ""), "", "warnings:"]
        L += [f"- {m}" for m in sorted(r["warnings"], key=lambda m: "did not converge" not in m)] or ["- none: inside the models' training domains"]   # no solution first
        return "\n".join(textwrap.fill(l, WRAP, subsequent_indent="  ") if l.startswith("- ") else l for l in L).replace("$", r"\$")   # no mathtext


# ------------------------------------------------------------------------------------------- CSV
def _tables(res, k):
    """the CSV parts of a result as {part: [(column, what it is, values (one per row))]}; k: the cut strip"""
    w, c, s, st, n = res["wing"], res["coef"], res["strips"], res["state"], res["wing"]["n"]
    sst, sec, D = st == "sst", res["loops"], res["strip_detail"]
    one = lambda name, text, v: (name, text, [v])
    T = {}
    T["summary"] = [one("name", "the wing's name", w["name"]),
                    one("state", "section state: xfoil (VortFoil surface model) or sst (RANS field decoder)", st),
                    one("V_m_s", "freestream speed", res["V"]), one("alpha_deg", "incidence of the root chord", res["alpha"]),
                    one("nu_m2_s", "kinematic viscosity in Re = V c / nu", res["nu"]),
                    one("decoder", "field decoder weights file (SST state; empty in the XFOIL state)", res["decoder"] or ""),
                    one("n_strips", "number of strips", n), one("span_m", "span b", w["b"]),
                    one("S_m2", "planform area S = sum of c ds over the strips (along the dihedral)", w["S"]), one("AR", "aspect ratio b^2/S", w["AR"]),
                    one("MAC_m", "mean aerodynamic chord sum(c^2 ds)/S, the reference length of C_M", w["MAC"]),
                    one("taper", "tip chord / root chord", w["taper"]), one("c_root_m", "root chord", w["c_root"]), one("c_tip_m", "tip chord", w["c_tip"]),
                    one("Re_root", "Reynolds number of the root chord", w["Re_root"]), one("Re_tip", "Reynolds number of the tip chord", w["Re_tip"]),
                    one("x_ref_m", "moment reference point (root quarter chord), x in the body frame", w["x_ref"]),
                    one("z_ref_m", "moment reference point, z in the body frame", w["z_ref"]),
                    one("CL", "lift coefficient: the horseshoe vortices' force rho Gamma_i V_i x dl_i, perpendicular to the freestream", c["CL"]),
                    one("CDi", "induced drag coefficient (Trefftz plane)", c["CDi"]),
                    one("CD0", "profile drag coefficient (the sections' c_d at their effective incidence)", c["CD0"])]
    if sst:
        T["summary"] += [one("CD0_pressure", "pressure part of CD0 (SST state)", c["CD0_pressure"]), one("CD0_friction", "wall-function friction part of CD0 (SST state)", c["CD0_friction"])]
    T["summary"] += [one("CD", "total drag coefficient CDi + CD0", c["CD"]), one("L_over_D", "CL / CD", c["LD"]),
                     one("e", "span efficiency CL^2/(pi AR CDi); nan when CDi <= 0", c["e"]),
                     one("CM", "pitching moment coefficient about the root quarter chord, reference length MAC, nose up positive", c["CM"]),
                     one("strips_past_stall", "strips beyond their section's largest-c_l incidence", c["strips_past_stall"]),
                     one("newton_iterations", "Newton steps of the lifting-line solution", res["newton"]["iterations"]),
                     one("newton_residual", "its residual, relative to the largest strip's lift", res["newton"]["residual"]),
                     one("warnings", "range and convergence warnings, joined by ' | ' (the SST state's Reynolds-range sentence first)",
                         " | ".join(([range_message(res)] if range_message(res) else []) + res["warnings"]))]
    S0 = res["defn"]["stations"]
    T["stations"] = [("y_m", "spanwise position of the station on the right half (the left half mirrors it)", [t["y"] for t in S0]),
                     ("chord_m", "chord", [t["chord"] for t in S0]), ("twist_deg", "twist, nose up positive, about the quarter chord", [t["twist"] for t in S0]),
                     ("x_c4_m", "quarter-chord x in the body frame (sweep)", [t["x"] for t in S0]),
                     ("z_c4_m", "quarter-chord z in the body frame (dihedral)", [t["z"] for t in S0]),
                     ("section", "the station's section: NACA digits, a UIUC name or a file", [t["section"] for t in S0])]
    strip = np.arange(n)
    sp = [("strip", "strip index, 0 at the left tip (y < 0) to n-1 at the right tip", strip), ("y_m", "spanwise position of the strip's control point", s["y"]),
          ("eta", "2y/b", s["eta"]), ("chord_m", "chord", s["chord"]), ("twist_deg", "twist (nose up positive)", s["twist_deg"]),
          ("x_c4_m", "quarter-chord x, body frame", s["x_c4"]), ("z_c4_m", "quarter-chord z, body frame", s["z_c4"]),
          ("Re", "Reynolds number V c / nu", s["Re"]), ("alpha_geo_deg", "geometric incidence: wing incidence + twist", s["alpha_geo_deg"]),
          ("alpha_eff_deg", "effective incidence: the angle of the local flow (with the induced velocity) to the section chord", s["alpha_eff_deg"]),
          ("alpha_ind_deg", "induced incidence alpha_geo - alpha_eff", s["alpha_ind_deg"]), ("Gamma_m2_s", "circulation of the strip's horseshoe vortex", s["gamma"]),
          ("cl", "section lift coefficient at the effective incidence", s["cl"]), ("cd", "section drag coefficient", s["cd"]),
          ("cm", "section pitching moment about its quarter chord, nose up positive", s["cm"]),
          ("cl_c_over_cbar", "loading c c_l / c_bar, c_bar = S/b", s["cl_c_over_cbar"])]
    if sst:
        sp += [("cd_pressure", "pressure part of cd (SST state)", s["cd_pressure"]), ("cd_friction", "wall-function friction part of cd (SST state)", s["cd_friction"])]
    T["spanwise"] = sp + [("section", "the strip's section (blends between stations of different sections)", [sec["label"][i] for i in sec["index"]])]
    p = res["polar"]
    T["polar"] = [("alpha_deg", "incidence of the root chord", p["alpha"]), ("CL", "lift coefficient", p["CL"]), ("CDi", "induced drag coefficient", p["CDi"]),
                  ("CD0", "profile drag coefficient", p["CD0"]), ("CD", "total drag coefficient", p["CD"]), ("CM", "pitching moment about the root quarter chord (MAC), nose up positive", p["CM"]),
                  ("L_over_D", "CL / CD", p["LD"]), ("e", "span efficiency", p["e"])]
    U, L = res["surface"]["upper"], res["surface"]["lower"]
    m = len(res["surface"]["x_c"])
    qs = [q for q in U if q not in "XYZ"]
    T["surface"] = ([("strip", "strip index (see spanwise)", np.repeat(strip, m).tolist() * 2), ("eta", "2y/b of the strip", np.repeat(s["eta"], m).tolist() * 2),
                     ("side", "upper or lower surface", ["upper"] * (n * m) + ["lower"] * (n * m)),
                     ("x_over_c", "chordwise position x/c of the point (chord frame)", np.tile(res["surface"]["x_c"], 2 * n))]
                    + [(f"{a}_m", f"wind-frame {a} of the point: {what}", np.r_[np.ravel(U[a]), np.ravel(L[a])])
                       for a, what in (("X", "along the freestream"), ("Y", "spanwise (right wing > 0)"), ("Z", "up"))]
                    + [(QUANT[q][1], QUANT[q][2], np.r_[np.ravel(U[q]), np.ravel(L[q])]) for q in qs])
    t = res["tables"]
    a = t["alpha_deg"]
    cols = ["cl", "cd", "cm"] + (["cd_pressure", "cd_friction"] if sst else [])
    T["section_tables"] = ([("strip", "strip index", np.repeat(strip, len(a))), ("eta", "2y/b of the strip", np.repeat(s["eta"], len(a))),
                            ("Re", "the strip's Reynolds number", np.repeat(s["Re"], len(a))),
                            ("alpha_deg", f"section incidence from its chord, the table grid ({a[0]:g} to {a[-1]:g} deg, step {a[1] - a[0]:g})", np.tile(a, n))]
                           + [(q, {"cl": "section lift coefficient", "cd": "section drag coefficient", "cm": "section moment about the quarter chord, nose up positive",
                                   "cd_pressure": "pressure part of cd", "cd_friction": "friction part of cd"}[q], np.ravel(t[q])) for q in cols])
    nn = len(D["x_cp"][k]) if not sst else len(D["x"][k])
    pad = lambda v: np.r_[np.ravel(v), np.full(nn - np.size(v), np.nan)]
    head = [("strip", "the cut strip's index", np.full(nn, k)), ("eta", "its 2y/b", np.full(nn, s["eta"][k]))]
    if sst:
        T["cut"] = head + [("node", "wall node index, in the decoder's order around the section", np.arange(nn)),
                           ("side", "upper or lower surface", np.where(D["upper"][k], "upper", "lower")),
                           ("x_over_c", "node x/c (chord frame)", D["x"][k]), ("y_over_c", "node y/c (chord frame)", D["y"][k]),
                           ("Cp", "the decoder's wall pressure coefficient", D["cp"][k]), ("Cf", "its wall-function skin friction coefficient", D["cf"][k])]
    else:
        th, ds = np.asarray(D["theta"][k], np.float64), np.asarray(D["dstar"][k], np.float64)
        H = lambda i: np.where((th[i] > 0) & (ds[i] > 0), ds[i] / np.where(th[i] > 0, th[i], 1.0), np.nan)
        T["cut"] = head + [("panel", "sheet panel index: 0-119 run from the trailing edge to the leading edge on the upper surface, 120-239 back on the lower", np.arange(nn)),
                           ("side", "upper or lower surface", np.where(np.arange(nn) < nn // 2, "upper", "lower")),
                           ("x_over_c", "panel midpoint x/c (chord frame)", D["x_cp"][k]), ("Cp", "the vortex sheet's C_p at the panel", D["cp"][k]),
                           ("bl_x_over_c", f"boundary-layer station x/c ({len(D['BL_X'])} stations; rows beyond that are nan)", pad(D["BL_X"])),
                           ("theta_upper_over_c", "momentum thickness / chord, upper surface", pad(th[0])), ("theta_lower_over_c", "momentum thickness / chord, lower surface", pad(th[1])),
                           ("dstar_upper_over_c", "displacement thickness / chord, upper surface", pad(ds[0])), ("dstar_lower_over_c", "displacement thickness / chord, lower surface", pad(ds[1])),
                           ("H_upper", "shape factor delta*/theta, upper surface (nan where either thickness is <= 0)", pad(H(0))),
                           ("H_lower", "shape factor delta*/theta, lower surface", pad(H(1)))]
    return T


def _fmt(v):
    return f"{v:.8g}" if isinstance(v, (float, np.floating)) else str(v)


def export_csv(folder, res, parts=None, cut_index=None, name=None):
    """The chosen parts of a wing result (CSV_PARTS keys; default all) as CSV files `<base>_<suffix>.csv` in `folder` (created if
    needed), with `<base>_about.txt` that explains every column of the files written. cut_index: the cut strip of the 'cut' part
    (default the strip nearest 2y/b = 0.5); name: the wing's name in the file names (default the result's). Returns the paths."""
    parts = list(CSV_PARTS) if parts is None else list(parts)
    bad = [p for p in parts if p not in CSV_PARTS]
    if bad:
        raise ValueError(f"unknown CSV part '{bad[0]}' (choose from {', '.join(CSV_PARTS)})")
    parts = [p for p in CSV_PARTS if p in parts]
    n = res["wing"]["n"]
    k = int(np.argmin(np.abs(res["strips"]["eta"] - DEFAULTS["cut_eta"]))) if cut_index is None else int(cut_index)
    if not 0 <= k < n:
        raise ValueError(f"cut_index: a strip number from 0 to {n - 1} (got {cut_index})")
    name = res["wing"]["name"] if name is None else name
    os.makedirs(folder, exist_ok=True)
    base = os.path.join(folder, re.sub(r"(?:[^\w.+-]|_)+", "_", f"vortfoil_wing_{name}_a{res['alpha']:g}_V{res['V']:g}_{res['state']}"))   # \w: a name in any script stays one
    T, paths = _tables(res, k), []
    for part in parts:
        paths.append(f"{base}_{CSV_PARTS[part][0]}.csv")
        with open(paths[-1], "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow([c[0] for c in T[part]])
            wr.writerows(zip(*[[_fmt(v) for v in c[2]] for c in T[part]]))
    s, w, sst = res["strips"], res["wing"], res["state"] == "sst"
    txt = [f"VortFoil wing result: {name}, incidence {res['alpha']:g} deg of the root chord, speed {res['V']:g} m/s, " + ("SST state." if sst else "XFOIL state."),
           "Sections: " + ("the RANS field decoder's wall values (trained on AirfRANS k-omega SST, fully turbulent): c_l and c_d from the wall pressure "
                           "and wall-function friction, c_m from the same wall data; the section tables may be interpolated in log10(Re) between tables "
                           "on a fixed Reynolds-number grid." if sst else
                           "the VortFoil surface model (trained on XFOIL, free transition): c_l, c_d, c_m, the sheet's C_p and the boundary layer."),
           f"Wing: a numerical nonlinear lifting line (Phillips and Snyder) with straight trailing legs along the freestream. Each strip carries its section at "
           f"its own Re = V c / nu (nu = {res['nu']:g} m^2/s) and at its effective incidence; C_D = C_Di (Trefftz plane) + C_D0 (the sections' drag).",
           f"{n} strips, cosine-spaced along the span; strip 0 is at the left tip (y < 0), 2y/b runs from -1 to 1. Coefficients use S = {w['S']:.6g} m^2 and "
           "the dynamic pressure 1/2 V^2 (density 1; Gamma is in m^2/s).",
           "Frames: body frame: x along the root chord (downstream), y spanwise, z up. Wind frame (the surface points): X along the freestream, Y spanwise, "
           "Z up, the wing pitched nose up by the incidence.",
           f"Signs: c_m and C_M are positive nose up; c_m is about the section's quarter chord, C_M about the root quarter chord (x {w['x_ref']:g}, z {w['z_ref']:g} m) "
           "with the mean aerodynamic chord as reference length; twist is positive nose up.",
           "Lengths in m, angles in deg. Numbers have 8 significant digits; nan means not defined."]
    for part in parts:
        extra = f" (strip {k}, counted from 0 at the left tip: 2y/b {s['eta'][k]:.4f}, y {s['y'][k]:.4f} m)" if part == "cut" else ""
        txt += ["", f"{os.path.basename(paths[parts.index(part)])}: {CSV_PARTS[part][1]}{extra}"]
        txt += [textwrap.fill(f"{col}: {what}", 118, initial_indent="  ", subsequent_indent="      ") for col, what, _ in T[part]]
    paths.append(base + "_about.txt")
    with open(paths[-1], "w", encoding="utf-8") as f:
        f.write("\n".join(txt) + "\n")
    return paths


def figure(res, fig=None, layout="landscape", options=None):
    """The wing figure (static convenience for scripts and tests); with `fig` it is cleared and redrawn. The view stays
    reachable as fig.wing_view (its mouse connections live as long as it does)."""
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.figure(figsize=VA.DESIGNS[layout], constrained_layout=True)
    WingView(fig, res, layout, options).draw()
    return fig
