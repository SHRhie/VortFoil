# -*- coding: utf-8 -*-
"""The 3D wing's result window (code/view_wing.py): the three layouts at their design sizes with no overlapping or clipped
text, every colour / span / polar option in both section states, the 3D panel (true scale, the first view, the wing filling its
cell and staying in it at every turn, turn / double click by mouse, only what moves redrawn while the mouse is down), the draggable section cut
(synthetic mouse events, also after a window zoom), and the CSV export of all values or a chosen subset. No window is opened (Agg)."""
import csv
import itertools
import os
import pathlib
import sys
import time

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib.backend_bases import MouseEvent  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "code", "field"))

import predict_field as PF  # noqa: E402
import view_airfoil as VA  # noqa: E402
import view_wing as VW  # noqa: E402
import wing_nll as WG  # noqa: E402

FULL, KINKED = "Full-scale wing (span 10 m)", "Kinked wing (4415 to 2412, dihedral)"
N_AXES = 8                      # 3D, colour bar, span, section, outline, section label, polar, numbers
PARTS = list(VW.CSV_PARTS)


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """small results of both states, built once: the XFOIL state on the full-scale wing (12 strips), the RANS state on the
    kinked wing (8 strips, a coarse incidence table, a scratch table cache)"""
    mp = pytest.MonkeyPatch()
    mp.setattr(WG, "CACHE", str(tmp_path_factory.mktemp("wing_tables")))
    mp.setattr(WG, "_MEM", {})
    mp.setattr(WG, "TABLE_S", np.arange(-8.0, 16.001, 2.0))
    xf = WG.compute(dict(WG.PRESETS[FULL], n=12), 45.0, 6.0)
    ss = WG.compute(dict(WG.PRESETS[KINKED], n=8), 60.0, 6.0, "sst")
    mp.undo()
    far = {"xfoil": WG.compute(dict(WG.PRESETS[FULL], n=8), 45.0, 28.0)}                       # incidences far outside the states' training ranges
    mp = pytest.MonkeyPatch()
    mp.setattr(WG, "CACHE", str(tmp_path_factory.mktemp("wing_tables_far")))
    mp.setattr(WG, "_MEM", {})
    mp.setattr(WG, "TABLE_S", np.arange(-8.0, 16.001, 4.0))
    far["sst"] = WG.compute(dict(WG.PRESETS[FULL], n=8), 45.0, 24.0, "sst")
    mp.undo()
    return {"xfoil": xf, "sst": ss, "far_xfoil": far["xfoil"], "far_sst": far["sst"]}


@pytest.fixture(params=["xfoil", "sst"])
def res(request, results):
    return results[request.param]


def make(res, layout="landscape", options=None, dpi=100, **kw):
    fig = Figure(figsize=VA.DESIGNS[layout], dpi=dpi)
    FigureCanvasAgg(fig)
    view = VW.WingView(fig, res, layout, options, **kw)
    view.draw()
    return fig, view


def send(fig, name, x, y, button=1, **kw):
    fig.canvas.callbacks.process(name, MouseEvent(name, fig.canvas, x, y, button=button, **kw))


def span_point(view, eta):
    """display coordinates of 2y/b = eta on the span panel (through the axes' transform, so valid after a window zoom)"""
    return view.ax_span.transData.transform((eta, 0.5 * sum(view.ax_span.get_ylim())))


def drag(fig, view, etas, release=True):
    """press at the first 2y/b on the span panel, move through the others, release"""
    send(fig, "button_press_event", *span_point(view, etas[0]))
    for e in etas[1:]:
        send(fig, "motion_notify_event", *span_point(view, e))
    if release:
        send(fig, "button_release_event", *span_point(view, etas[-1]))


def nearest(res, eta):
    return int(np.argmin(np.abs(res["strips"]["eta"] - eta)))


def text_problems(fig):
    """what is wrong with the figure's text: outside the figure, overlapping another text, or (the numbers, the cut label) outside
    its cell. Drawn texts only: the suptitle, titles, axis labels, tick labels in view, axes texts and legends."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []
    if fig._suptitle is not None:
        boxes.append(("suptitle", fig._suptitle.get_window_extent(r)))
    for i, ax in enumerate(fig.axes):
        for name, t in ([("title", ax.title)] + ([("xlabel", ax.xaxis.label), ("ylabel", ax.yaxis.label)] if ax.axison else [])
                        + [(f"text{j}", t) for j, t in enumerate(ax.texts)]):
            if t.get_text().strip():
                boxes.append((f"ax{i}.{name}", t.get_window_extent(r)))
        if ax.get_legend():
            boxes.append((f"ax{i}.legend", ax.get_legend().get_window_extent(r)))
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())) if ax.axison else ():
            for lab, loc in zip(axis.get_majorticklabels(), axis.get_majorticklocs()):
                if min(lim) <= loc <= max(lim) and lab.get_text().strip():
                    boxes.append((f"ax{i}.tick{loc:g}", lab.get_window_extent(r)))
    W, H = fig.bbox.width, fig.bbox.height
    bad = [f"outside the figure: {n}" for n, b in boxes if b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1]
    for (n1, b1), (n2, b2) in itertools.combinations(boxes, 2):
        if min(b1.x1, b2.x1) - 1 > max(b1.x0, b2.x0) + 1 and min(b1.y1, b2.y1) - 1 > max(b1.y0, b2.y0) + 1:
            bad.append(f"overlap: {n1} / {n2}")
    for t in (fig.wing_view._text, fig.wing_view._info):
        a, b = t.axes.get_window_extent(), t.get_window_extent()
        if b.x0 < a.x0 - 1 or b.x1 > a.x1 + 1 or b.y0 < a.y0 - 1 or b.y1 > a.y1 + 1:
            bad.append(f"text outside its cell: {t.get_text()[:24]!r}")
    return bad


# ------------------------------------------------------------------------------------------- figure
@pytest.mark.parametrize("dpi", [100, 72, 96])                    # 72 and 96: the screens' Tk (macOS, Windows)
@pytest.mark.parametrize("layout", list(VA.DESIGNS))
def test_layouts_at_design_size(res, layout, dpi):
    """Each layout builds at its design size with its eight axes, and its text neither overlaps nor is clipped."""
    fig, view = make(res, layout, dpi=dpi)
    assert tuple(fig.get_size_inches()) == VA.DESIGNS[layout] and len(fig.axes) == N_AXES and view.ax3d.name == "3d"
    title = fig._suptitle.get_text()
    assert title.count("\n") == 1 and title.endswith("wind frame, true scale") and "×" not in title       # no exaggeration of the heights
    assert ("RANS state" if res["state"] == "sst" else "XFOIL state") in title
    assert text_problems(fig) == []
    assert max(len(line) for line in view._text.get_text().split("\n")) <= VW.WRAP == 56         # the numbers' line width


@pytest.mark.parametrize("layout", list(VA.DESIGNS))
def test_text_fits_in_the_fallback_font_and_with_long_texts(results, layout, monkeypatch):
    """Without Times New Roman (Linux) the wider fallback font must fit too, and so must a long name and many long warnings."""
    monkeypatch.setitem(PF.STYLE, "font.serif", ["DejaVu Serif"])
    xf = results["xfoil"]
    long = dict(xf, warnings=[f"warning number {i}: " + "a long message that needs wrapping " * 3 for i in range(5)],
                wing=dict(xf["wing"], name="A very long wing name that goes on and on " * 3 + "$x_1$"),
                loops=dict(xf["loops"], label=["a_file$with$dollars_and_a_very_long_name_that_goes_on_and_on"]))
    for r in (xf, results["sst"], long):
        fig, view = make(r, layout)
        assert text_problems(fig) == [] and view._info.get_text().startswith("2y/b") and len(view._info.get_text().split("\n")) == 4
    last = view._info.get_text().split("\n")[-1]                  # a section name from a file: no mathtext
    assert "\\$" in last and "$" not in last.replace("\\$", "")


def test_a_lifting_line_without_solution_is_marked(results):
    """A residual above the tolerance turns the numbers red and puts the non-convergence warning first; a solution stays black."""
    xf = results["xfoil"]
    assert xf["newton"]["residual"] < VW.NEWTON_TOL
    bad = dict(xf, newton=dict(xf["newton"], residual=8.0),
               warnings=["alpha 6 deg: 22 of 40 strips beyond their section's stall angle", "the lifting line did not converge at 6 deg (residual 8e+00 after 60 Newton steps)"])
    fig, view = make(bad)
    lines = view._text.get_text().split("\n")
    assert view._text.get_color() == VW.UP and lines[lines.index("warnings:") + 1].startswith("- the lifting line did not converge")
    assert text_problems(fig) == [] and make(xf)[1]._text.get_color() != VW.UP


def test_text_after_a_window_zoom(res):
    """The zoomed figure (view_airfoil.scale_with_window) keeps its text clear of the edges and of its neighbours down to 0.6."""
    for layout, (W, H) in VA.DESIGNS.items():
        fig, _ = make(res, layout)
        on_resize = VA.scale_with_window(fig)
        for s in (0.6, 1.5):
            fig.set_size_inches(W * s, H * s, forward=False)
            on_resize()
            assert text_problems(fig) == [], (layout, s)


def test_every_option_draws(res):
    """Every colour quantity of the state, every spanwise quantity and every polar plot draws and is labelled."""
    state = res["state"]
    fig, view = make(res)
    for q in VW.SURFACE_QUANTITIES[state]:
        view.set_option("surface", q)
        assert len(fig.axes) == N_AXES and fig.axes[1].get_xlabel() == VW.QUANT[q][0] == fig.axes[3].get_ylabel()    # colour bar, section
        assert fig.axes[3].get_title() == f"{VW.QUANT[q][0]} at the cut: upper (red), lower (blue)"
    for q in VW.SPAN_QUANTITIES:
        view.set_option("span", q)
        assert view.ax_span.get_ylabel() == VW.SPAN[q][1] and len(view.ax_span.lines) >= 3                  # curve, cut line, cut dot
    for q in VW.POLAR_PLOTS:
        view.set_option("polar", q)
        ax = fig.axes[6]
        assert (ax.get_xlabel(), ax.get_ylabel()) == VW.POLAR[q][2:] and len(ax.lines) == (2 if res["polar"]["in_range"].all() else 5)   # curve, the dot at this incidence (+ the faded part, its hollow points, the legend's sample)
    view.set_option("span", "c_d")
    assert len(view.ax_span.lines) == (5 if state == "sst" else 3)                                           # the SST split adds two lines
    assert text_problems(fig) == []
    for key, value in (("surface", "H" if state == "sst" else "C_f"), ("span", "lift"), ("polar", "x"), ("cut_eta", "left"), ("color", 1)):
        with pytest.raises(ValueError):
            view.set_option(key, value)


@pytest.mark.parametrize("layout", list(VA.DESIGNS))
def test_colour_bar_is_labelled_and_beside_or_below_the_wing(res, layout):
    """The colour bar names the quantity; it stands beside the 3D view in the square layout (the wing gets the cell's height), below it otherwise."""
    fig, view = make(res, layout)
    side = layout == "square"
    for q in VW.SURFACE_QUANTITIES[res["state"]]:
        view.set_option("surface", q)
        bar, wing, box = fig.axes[1], view.ax3d.get_position(), fig.axes[1].get_position()
        assert (bar.get_ylabel(), bar.get_xlabel()) == ((VW.QUANT[q][0], "") if side else ("", VW.QUANT[q][0]))
        assert box.x0 >= wing.x1 - 1e-6 if side else box.y1 <= wing.y0 + 1e-6
    assert text_problems(fig) == []


def test_stale_surface_option_falls_back_to_cp(results):
    assert make(results["sst"], options={"surface": "H"})[1].surface == "C_p"
    assert make(results["xfoil"], options={"surface": "theta/c", "span": None})[1].surface == "theta/c"


def test_set_option_keeps_the_view_and_the_cut(res):
    fig, view = make(res, options={"view": (12.0, -70.0), "cut_eta": -0.3})
    assert tuple(round(a, 6) for a in view.pose) == (12, -70)                               # a view of two numbers: elevation and azimuth
    view.ax3d.view_init(elev=50, azim=20)                      # the user turns the wing
    assert tuple(round(a, 6) for a in view.pose) == (50, 20)
    assert tuple(round(a, 6) for a in make(res, options={"view": view.pose})[1].pose) == (50, 20)          # a new view can keep it
    k = view.cut_index
    view.set_option("polar", "C_D vs C_L")
    view.set_option("surface", "C_p")
    assert tuple(round(a, 6) for a in view.pose) == (50, 20) and view.cut_index == k == nearest(res, -0.3)
    on_resize = VA.scale_with_window(fig)                       # as the application does: a zoomed, frozen figure is redrawn
    fig.set_size_inches(11, 5, forward=False)
    on_resize()
    view.layout = "portrait"                                    # a relayout draws the same view again
    view.draw()
    assert tuple(fig.get_size_inches()) == VA.DESIGNS["portrait"] and tuple(round(a, 6) for a in view.pose) == (50, 20)
    assert text_problems(fig) == [] and "$" not in view._text.get_text().replace("\\$", "")
    view.layout = "tall"
    with pytest.raises(ValueError, match="layout"):
        view.draw()


# ------------------------------------------------------------------------------------------- the cut
def test_drag_moves_the_cut_and_calls_back(res):
    calls = []
    fig, view = make(res, on_cut=lambda k, e: calls.append((k, e)))
    eta, last = res["strips"]["eta"], len(res["strips"]["eta"]) - 1
    assert view.cut_index == nearest(res, 0.5) and view.cut_eta == eta[view.cut_index]
    drag(fig, view, [-0.8, -0.4, 0.1], release=False)
    k = nearest(res, 0.1)
    assert view.cut_index == k and calls[-1] == (k, eta[k]) and len(calls) == 3
    send(fig, "motion_notify_event", *view.ax_span.transData.transform((5.0, 0.0)))      # outside the axes, still pressed: the tip strip
    assert view.cut_index == last
    send(fig, "button_release_event", 1, 1)
    n = len(calls)
    send(fig, "motion_notify_event", *span_point(view, 0.0))                              # after the release nothing moves
    assert view.cut_index == last and len(calls) == n
    e = view.cut_eta                                           # the artists follow: cut line and dot, outline on the wing, label
    assert list(view._vline.get_xdata()) == [e, e] and view._dot.get_xdata()[0] == e
    Y = view._outline.get_data_3d()[1]
    assert np.allclose(Y[0], res["surface"]["upper"]["Y"][last, 0]) and f"2y/b {e:.2f}" in view._info.get_text()
    fig.canvas.draw()


def test_drag_ignored_in_other_panels_with_the_toolbar_in_use_or_with_another_button(res):
    from matplotlib.backend_bases import NavigationToolbar2
    fig, view = make(res)
    tb = NavigationToolbar2(fig.canvas)                               # the toolbar's own logic (no window), as fig.canvas.toolbar
    k = view.cut_index
    send(fig, "button_press_event", *view.ax3d.bbox.get_points().mean(0))                # a press in the 3D panel turns the wing
    send(fig, "motion_notify_event", *span_point(view, -0.5))
    send(fig, "button_release_event", 1, 1)
    assert view.cut_index == k and not view._drag
    for tool in (tb.pan, tb.zoom):                                    # the pan or the zoom tool is active: the mouse is the tool's
        tool()
        assert str(tb.mode) != ""
        drag(fig, view, [0.3, -0.5])
        assert view.cut_index == k
        tool()
        assert str(tb.mode) == ""
    send(fig, "button_press_event", *span_point(view, 0.3), button=3)                    # a right press is not a drag
    send(fig, "motion_notify_event", *span_point(view, -0.5))
    send(fig, "button_release_event", *span_point(view, -0.5), button=3)
    assert view.cut_index == k
    drag(fig, view, [0.3, -0.5])
    assert view.cut_index == nearest(res, -0.5)


@pytest.mark.parametrize("layout", list(VA.DESIGNS))
def test_drag_after_the_window_zoom(res, layout):
    """The pointer goes through the axes' transform: the drag works on a zoomed, letterboxed figure."""
    fig, view = make(res, layout)
    on_resize = VA.scale_with_window(fig)
    W, H = VA.DESIGNS[layout]
    for s, k in ((0.6, 2), (1.5, -3), (1.0, 4)):               # to the centre of a strip: a point between two would be a toss-up
        target = res["strips"]["eta"][k]
        fig.set_size_inches(W * s, H * s, forward=False)
        on_resize()
        drag(fig, view, [0.0, target / 2, target])
        assert view.cut_index == k % len(res["strips"]["eta"]), (layout, s)
        fig.canvas.draw()


def test_cut_update_touches_artists_only_and_is_quick(res):
    fig, view = make(res)
    axes, ntext, nlines = list(fig.axes), len(fig.texts), [len(a.lines) for a in fig.axes]
    drawn = []
    fig.canvas.draw_idle = lambda *a, **k: drawn.append(1)       # what Tk would defer
    ks = [view.cut_index]
    t = time.perf_counter()
    for e in np.linspace(-0.9, 0.9, 25):
        view.set_cut(e)
        ks.append(view.cut_index)
    dt = (time.perf_counter() - t) / 25
    moves = sum(a != b for a, b in zip(ks, ks[1:]))
    assert fig.axes == axes and len(fig.texts) == ntext and [len(a.lines) for a in fig.axes] == nlines     # nothing rebuilt
    assert moves > 3 and len(drawn) == moves and dt < 0.05, dt                                             # one redraw request per move
    t = time.perf_counter()
    fig.canvas.draw()
    assert time.perf_counter() - t < 1.0                          # a whole step (generous bound; ~0.1 s on the Mac)
    view.set_cut(5.0)
    assert view.cut_index == len(res["strips"]["eta"]) - 1
    view.set_cut(-5.0)
    assert view.cut_index == 0
    with pytest.raises(ValueError, match="cut_eta"):
        view.set_cut(float("nan"))


def test_a_press_before_the_first_draw_is_ignored(res):
    fig = Figure(figsize=VA.DESIGNS["landscape"])
    FigureCanvasAgg(fig)
    view = VW.WingView(fig, res)                                  # nothing drawn: no span panel yet
    send(fig, "button_press_event", 5, 5)
    send(fig, "motion_notify_event", 6, 6)
    assert not view._drag and view.ax_span is None


def test_set_cut_does_not_call_back(res):
    calls = []
    _, view = make(res, on_cut=lambda k, e: calls.append(k))
    view.set_cut(-0.6)
    assert calls == [] and view.cut_index == nearest(res, -0.6)


def test_disconnect_and_reuse_of_the_figure(res):
    fig, view = make(res)
    view.disconnect()
    k = view.cut_index
    drag(fig, view, [-0.5, -0.6])
    assert view.cut_index == k
    v1 = VW.WingView(fig, res)                                   # a view on the same figure lets the earlier one go
    v1.draw()
    v2 = VW.WingView(fig, res)
    v2.draw()
    drag(fig, v2, [0.2, -0.6])
    assert v2.cut_index == nearest(res, -0.6) and v1._cids == []


def test_static_figure_convenience(res):
    import matplotlib.pyplot as plt
    fig = VW.figure(res, layout="square", options={"span": "c*c_l/c_bar"})
    assert tuple(fig.get_size_inches()) == VA.DESIGNS["square"] and fig.wing_view.span == "c*c_l/c_bar"
    plt.close(fig)
    fig2 = Figure()
    assert VW.figure(res, fig=fig2) is fig2 and len(fig2.axes) == N_AXES


# ------------------------------------------------------------------------------------------- the 3D panel
def project(view, pts):
    """display coordinates of 3D points (n, 3) of the wing's data frame at the figure's current state"""
    from mpl_toolkits.mplot3d import proj3d
    view.fig.canvas.draw()
    x, y, _ = proj3d.proj_transform(pts[:, 0], pts[:, 1], pts[:, 2], view.ax3d.M)
    return view.ax3d.transData.transform(np.c_[x, y])


def wing_points(res):
    """every point the wing view must show: both surfaces"""
    S = res["surface"]
    return np.concatenate([np.stack([a[c].ravel() for c in "XYZ"], -1) for a in (S["upper"], S["lower"])])


def centre3d(view):
    """the 3D panel's centre in whole pixels (a mouse event has no fractions)"""
    return tuple(int(v) for v in view.ax3d.get_window_extent().get_points().mean(0))


def turn(view, dx, dy):
    """a left-button drag in the 3D panel by (dx, dy) pixels (up is +dy, as the canvas counts), not released"""
    x, y = centre3d(view)
    send(view.fig, "button_press_event", x, y)
    send(view.fig, "motion_notify_event", x + dx, y + dy)
    return x, y


def test_first_view_is_the_viewers_oblique_one_at_true_scale(res):
    """The first view is the research viewer's 'oblique' one (elevation 34, azimuth 225 in the same X, Y, Z): the freestream axis points
    right and up on the screen, the span axis (the right wing) left and up, Z straight up. The box has the data's proportions: no
    exaggeration of the heights."""
    from mpl_toolkits.mplot3d import proj3d
    fig, view = make(res)
    ax = view.ax3d
    assert view.pose == (34.0, 225.0) and VW.VIEW == (34.0, 225.0)
    assert len(ax.collections) == 1 and len(ax.lines) == 1                     # the surface and the cut's outline: no trailing lines
    assert "double-click to reset" in ax.get_title() and "wheel" not in ax.get_title()     # the hint says what the panel does
    fig.canvas.draw()
    o, *axes = [proj3d.proj_transform(*(np.array([float(c)]) for c in p), ax.M)[:2] for p in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))]
    (xr, xu), (yr, yu), (zr, zu) = [(float(a[0][0] - o[0][0]), float(a[1][0] - o[1][0])) for a in axes]
    assert xr > 0 and xu > 0 and yr < 0 and yu > 0 and abs(zr) < 1e-12 and zu > 0
    box, lim = np.array(ax.get_box_aspect()), np.array([np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())])
    assert np.allclose(box / box[1], lim / lim[1], rtol=1e-6)                  # the box has the limits' proportions: equal scale in X, Y and Z
    P = wing_points(res)
    for i, get in enumerate((ax.get_xlim, ax.get_ylim, ax.get_zlim)):          # and the limits hold every point of the wing
        assert get()[0] <= P[:, i].min() + 1e-9 and P[:, i].max() <= get()[1] + 1e-9


@pytest.mark.parametrize("layout", list(VA.DESIGNS))
def test_the_wing_fills_its_cell(res, layout):
    """At the first view the wing takes 90% of the limiting dimension of the 3D panel, inside it, in the cell's own shape (not mplot3d's
    square), at the design size and after a window zoom of another shape too."""
    fig, view = make(res, layout)
    P = wing_points(res)
    W, H = VA.DESIGNS[layout]
    on_resize = VA.scale_with_window(fig)
    for size in ((W, H), (W * 0.6, H * 0.6), (W * 1.4, H * 0.8), (W * 0.8, H * 1.3)):
        fig.set_size_inches(*size, forward=False)
        on_resize()
        d, bb = project(view, P), view.ax3d.get_window_extent()
        share = max(np.ptp(d[:, 0]) / bb.width, np.ptp(d[:, 1]) / bb.height)
        assert 0.89 < share < 0.91, (size, share)
        assert d[:, 0].min() >= bb.x0 - 1 and d[:, 0].max() <= bb.x1 + 1 and d[:, 1].min() >= bb.y0 - 1 and d[:, 1].max() <= bb.y1 + 1
    fig.set_size_inches(W, H, forward=False); on_resize()
    ax = view.ax3d
    bb = ax.get_window_extent()
    assert bb.width > 1.1 * bb.height                                          # the cell's own shape: wider than tall, mplot3d's would be a square
    assert ax.get_position().bounds == ax.get_position(original=True).bounds


@pytest.mark.parametrize("layout", list(VA.DESIGNS))
def test_the_wing_stays_in_its_cell_at_every_turn(res, layout):
    """There is no zoom, so the window follows the turn: at any elevation and azimuth the wing lies inside the cell (filling at most
    90% of its limiting dimension, as at the first view) and the window is never smaller than the first view's (the wing keeps its
    scale while turned and shrinks only where a turn would take it out of the cell)."""
    fig, view = make(res, layout)
    ax, P = view.ax3d, wing_points(res)
    fig.canvas.draw()
    h0 = np.ptp(ax.viewLim.intervaly)
    for el, az in ((0, 180), (0, 270), (90, 270), (60, 225), (-34, 45)):         # along the chord, along the span, down on it, steep, from below
        ax.view_init(elev=el, azim=az)
        d, bb = project(view, P), ax.get_window_extent()
        assert d[:, 0].min() >= bb.x0 - 1 and d[:, 0].max() <= bb.x1 + 1 and d[:, 1].min() >= bb.y0 - 1 and d[:, 1].max() <= bb.y1 + 1, (el, az)
        assert max(np.ptp(d[:, 0]) / bb.width, np.ptp(d[:, 1]) / bb.height) < 0.91, (el, az)
        assert np.ptp(ax.viewLim.intervaly) >= h0 * (1 - 1e-9), (el, az)


def test_the_view_turns_and_starts_over_by_mouse(res):
    """Left drag turns (0.458 deg a pixel, not over the poles, also when the pointer leaves the panel); a double click is back to the
    first view, its limits too where a toolbar tool moved them; the wheel, the right drag and the middle drag do nothing (there is no
    zoom or move); the span panel's cut is not moved by any of it."""
    fig, view = make(res)
    ax, k = view.ax3d, view.cut_index
    e0, a0 = view.pose
    first = lambda: all(np.allclose(get(), lim) for get, lim in zip((ax.get_xlim, ax.get_ylim, ax.get_zlim), view._lim0))
    x, y = turn(view, 100, 50)
    assert view.pose == pytest.approx((e0 - 50 * VW.TURN, a0 - 100 * VW.TURN))
    send(fig, "motion_notify_event", x - 3000, y + 5000)                          # far outside the panel: the turn goes on, not past the poles
    assert view.pose == pytest.approx((-89.5, a0 + 3000 * VW.TURN))
    send(fig, "motion_notify_event", x, y - 9000)                                  # (the turn is from where the press was)
    assert view.pose == pytest.approx((89.5, a0))
    send(fig, "button_release_event", x, y)
    assert "draw" not in vars(fig.canvas) and not ax.get_animated() and view.cut_index == k
    cx, cy = centre3d(view)
    fig.canvas.callbacks.process("scroll_event", MouseEvent("scroll_event", fig.canvas, cx, cy, step=3))     # the wheel: nothing
    for button in (3, 2):                                                          # the right and the middle drag: nothing (mplot3d's zoom and move are off)
        send(fig, "button_press_event", cx, cy, button=button)
        assert "draw" not in vars(fig.canvas) and view._bg is None                # and they start no fast redraw
        send(fig, "motion_notify_event", cx, cy - 40, button=button)
        send(fig, "button_release_event", cx, cy - 40, button=button)
        assert view.pose == pytest.approx((89.5, a0)) and first() and "draw" not in vars(fig.canvas)
    ax.set_xlim(*(np.array(ax.get_xlim()) + 1.0))                                  # what the toolbar's pan tool does to the limits
    assert not first()
    send(fig, "button_press_event", cx, cy, dblclick=True)                        # double click: the first view, centred again
    assert view.pose == pytest.approx((34.0, 225.0)) and first()
    send(fig, "button_release_event", cx, cy)
    assert "draw" not in vars(fig.canvas) and view.cut_index == k


def test_the_tools_of_the_toolbar_keep_the_mouse(res):
    """While the toolbar's pan or zoom tool is on, a drag over the 3D panel does not turn it (the tool's own)."""
    from matplotlib.backend_bases import NavigationToolbar2
    fig, view = make(res)
    tb = NavigationToolbar2(fig.canvas)
    pose = view.pose
    for tool in (tb.pan, tb.zoom):
        tool()
        x, y = turn(view, 80, 20)
        send(fig, "button_release_event", x, y)
        assert view.pose == pose and "draw" not in vars(fig.canvas)
        tool()
    x, y = turn(view, 80, 20)
    send(fig, "button_release_event", x, y)
    assert view.pose != pose


def whole_draw(view):
    """what a whole draw gives at the figure's state now, with the artists a fast redraw leaves out drawn too (then put back)"""
    cv = view.fig.canvas
    for a in view._anim:
        a.set_animated(False)
    FigureCanvasAgg.draw(cv)
    img = np.asarray(cv.buffer_rgba()).copy()
    for a in view._anim:
        a.set_animated(True)
    return img


def test_only_what_moves_is_redrawn_while_the_mouse_is_down(res):
    """A turn redraws only the 3D panel, a cut drag only the cut's artists, on the figure as it was: no other panel is drawn,
    the frame is the whole draw's (the turn to the pixel), and the release is one whole draw that puts it all back."""
    fig, view = make(res)
    cv, ax = fig.canvas, view.ax3d
    drawn = []
    for a in fig.axes:
        if a is not ax:                                                           # a spy on every other panel's draw
            real = a.draw
            a.draw = lambda r, real=real, a=a: drawn.append(a) or real(r)
    x, y = turn(view, 30, 10)
    assert view._anim == (ax,) and ax.get_animated() and "draw" in vars(cv)
    drawn.clear()
    for i in range(1, 6):
        send(fig, "motion_notify_event", x + 30 + 17 * i, y + 10 + 6 * i)
        assert not drawn                                                          # no other panel was drawn
        assert np.array_equal(np.asarray(cv.buffer_rgba()), whole_draw(view))     # the frame is the whole draw's
        drawn.clear()                                                             # (that comparison drew them)
    send(fig, "button_release_event", x, y)
    assert "draw" not in vars(cv) and not ax.get_animated() and view._anim == () and view._bg is None
    assert drawn                                                                  # the release: one whole draw (every panel)
    # the cut drag: the cut's artists only; the frame differs from the whole draw only where the cut's lines cross the panels' frames
    cut = view._cut_artists()
    send(fig, "button_press_event", *span_point(view, 0.8))
    send(fig, "motion_notify_event", *span_point(view, 0.1))
    assert view._anim == cut and all(a.get_animated() for a in cut) and not ax.get_animated() and view.cut_index == nearest(res, 0.1)
    drawn.clear()
    for e in (-0.2, -0.6):
        send(fig, "motion_notify_event", *span_point(view, e))
        assert not drawn
        diff = np.abs(np.asarray(cv.buffer_rgba()).astype(int) - whole_draw(view).astype(int)).max(-1)
        assert (diff > 0).mean() < 2e-3                                           # a few pixels where a curve meets the frame
        drawn.clear()
    send(fig, "button_release_event", *span_point(view, -0.6))
    assert not any(a.get_animated() for a in cut) and view._anim == () and "draw" not in vars(cv)
    before = np.asarray(cv.buffer_rgba()).copy()
    FigureCanvasAgg.draw(cv)
    assert np.array_equal(before, np.asarray(cv.buffer_rgba()))                   # what the release left is a whole draw
    turn(view, 5, 5)                                                              # a session that is cut short by the view's end
    view.disconnect()
    assert "draw" not in vars(cv) and not ax.get_animated() and view._bg is None


def test_the_garbage_collector_sits_out_a_fast_redraw(res):
    """A collection of the application's big heap would stop a frame of a drag for ~70 ms: the objects that exist when it starts are
    frozen until the mouse is up (a few thousand frames of garbage do not grow memory)."""
    import gc
    fig, view = make(res)
    frozen = gc.get_freeze_count()
    x, y = turn(view, 10, 10)
    assert gc.get_freeze_count() > frozen
    send(fig, "button_release_event", x, y)
    assert gc.get_freeze_count() == frozen
    turn(view, 10, 10)
    view.disconnect()
    assert gc.get_freeze_count() == frozen                                       # also when the view ends first


def test_a_click_that_moves_nothing_redraws_nothing(res):
    """The background of a drag is taken when the button goes down (no delay at the first move); a click that moves nothing leaves the
    window as it is: nothing is drawn when it ends, and a double click is the first view."""
    fig, view = make(res)
    spied = []
    real = view.ax_span.draw
    view.ax_span.draw = lambda r: spied.append(1) or real(r)
    x, y = centre3d(view)
    send(fig, "button_press_event", x, y)
    assert "draw" in vars(fig.canvas) and view._anim == (view.ax3d,)                 # ready for the drag
    spied.clear()
    send(fig, "button_release_event", x, y)
    assert not spied and "draw" not in vars(fig.canvas) and not view.ax3d.get_animated() and view.pose == (34.0, 225.0)
    send(fig, "button_press_event", x, y); send(fig, "button_release_event", x, y)      # a double click: the second press is the reset
    send(fig, "button_press_event", x, y, dblclick=True); send(fig, "button_release_event", x, y)
    assert view.pose == pytest.approx((34.0, 225.0)) and "draw" not in vars(fig.canvas)


def test_a_rebuild_ends_a_fast_redraw(res):
    fig, view = make(res)
    x, y = turn(view, 20, 20)
    assert "draw" in vars(fig.canvas)
    view.set_option("polar", "C_D vs C_L")                                          # the figure is built anew: no cached background
    assert "draw" not in vars(fig.canvas) and view._bg is None and view.pose == pytest.approx((34 - 20 * VW.TURN, 225 - 20 * VW.TURN))


def test_the_vectorised_surface_equals_mplot3ds_own(results, monkeypatch):
    """Before matplotlib 3.11 the surface is drawn by a collection that projects and sorts its quads in numpy: the same picture as
    mplot3d's own Poly3DCollection, at several views (not one that looks along the span: there quads of equal depth are sorted in
    another order, a few pixels)."""
    if matplotlib.__version_info__ >= (3, 11):
        pytest.skip("matplotlib 3.11's own collection is vectorised")
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    res = results["xfoil"]
    fast, ref = make(res)[1], None
    assert type(fast.ax3d.collections[0]).__name__ == "Quads"
    imgs = []
    for az in (225, 40, 330):
        fast.ax3d.view_init(elev=30, azim=az); fast.fig.canvas.draw()
        imgs.append(np.asarray(fast.fig.canvas.buffer_rgba()).copy())
    monkeypatch.setattr(VW, "_quad_collection", lambda: Poly3DCollection)
    ref = make(res)[1]
    assert type(ref.ax3d.collections[0]) is Poly3DCollection
    for az, im in zip((225, 40, 330), imgs):
        ref.ax3d.view_init(elev=30, azim=az); ref.fig.canvas.draw()
        assert np.abs(np.asarray(ref.fig.canvas.buffer_rgba()).astype(int) - im.astype(int)).max() == 0


def test_the_sst_numbers_are_flagged_outside_the_decoders_range(results, tmp_path):
    """More than half the strips outside the decoder's Reynolds range: the sentence leads the numbers (red), in every layout, and is the
    first of the CSV summary's warnings; the result's numbers are all still there."""
    ss = results["sst"]
    msg = ("RANS state outside the field decoder's Reynolds range (2e6-6e6) on 8 of 8 strips: the RANS-state numbers are not meaningful here; "
           "use the XFOIL state")
    assert ss["sst_range"]["message"] is None and VW.range_message(ss) is None            # this wing is inside it
    bad = dict(ss, sst_range=dict(ss["sst_range"], message=msg))
    assert VW.range_message(bad) == msg and VW.range_message(results["xfoil"]) is None
    for layout in VA.DESIGNS:
        fig, view = make(bad, layout)
        txt = view._text.get_text()
        assert txt.replace("\n", " ").startswith(msg) and view._text.get_color() == VW.UP and "C_L" in txt and "warnings:" in txt
        assert max(len(line) for line in txt.split("\n")) <= VW.WRAP and text_problems(fig) == []
        assert make(ss, layout)[1]._text.get_color() != VW.UP
    row = dict(zip(*read(VW.export_csv(tmp_path, bad, parts=["summary"])[0])))
    assert row["warnings"].startswith(msg) and "RANS decoder" not in msg


def test_the_span_legend_covers_no_curve(res):
    """The legend of the spanwise plots stands in a band above the curves: none of them (the cut line aside) runs under it."""
    for layout in VA.DESIGNS:
        fig, view = make(res, layout)
        on_resize = VA.scale_with_window(fig)
        W, H = VA.DESIGNS[layout]
        for scale in (1.0, 0.6):
            fig.set_size_inches(W * scale, H * scale, forward=False); on_resize()
            for q in ("alpha_eff", "c*c_l/c_bar", "c_d"):
                view.span = q
                view.draw(); on_resize = VA.scale_with_window(fig)
                fig.set_size_inches(W * scale, H * scale, forward=False); on_resize()
                fig.canvas.draw()
                ax = view.ax_span
                if ax.get_legend() is None:                                      # one curve: no legend (c_d in the XFOIL state)
                    assert q == "c_d" and res["state"] == "xfoil"
                    continue
                bb = ax.get_legend().get_window_extent()
                for ln in ax.lines:
                    if ln is view._vline or ln is view._dot:
                        continue
                    d = ax.transData.transform(np.c_[ln.get_xdata(), ln.get_ydata()])
                    assert not ((d[:, 0] >= bb.x0) & (d[:, 0] <= bb.x1) & (d[:, 1] >= bb.y0) & (d[:, 1] <= bb.y1)).any(), (layout, scale, q)


def test_names_in_any_script_make_distinct_files(res, tmp_path):
    a = VW.export_csv(tmp_path / "a", res, parts=["summary"], name="직사각형 날개")
    b = VW.export_csv(tmp_path / "b", res, parts=["summary"], name="테이퍼 날개")
    assert os.path.basename(a[0]).startswith("vortfoil_wing_직사각형_날개_a6_V") and os.path.basename(b[0]).startswith("vortfoil_wing_테이퍼_날개_a6_V")
    assert os.path.basename(VW.export_csv(tmp_path / "c", res, parts=["summary"], name="My wing/1 (x)")[0]).startswith("vortfoil_wing_My_wing_1_x_a6_V")


def test_a_long_name_stays_whole_in_the_title(results):
    xf = results["xfoil"]
    for name in ("Model wing (span 2 m), NACA 2412, scale x 5", "Full-scale wing (span 10 m), NACA 2412, scale x 5 and a longer tail than that"):
        fig, view = make(dict(xf, wing=dict(xf["wing"], name=name)), "square")
        assert name in fig._suptitle.get_text() and text_problems(fig) == []


# ------------------------------------------------------------------------------------------- CSV
def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def test_export_all_parts(res, tmp_path):
    paths = VW.export_csv(tmp_path / "out", res)
    names = [os.path.basename(p) for p in paths]
    n, m, sst = res["wing"]["n"], len(res["tables"]["alpha_deg"]), res["state"] == "sst"
    assert all(a.endswith("_" + b) for a, b in zip(names, [VW.CSV_PARTS[p][0] + ".csv" for p in PARTS] + ["about.txt"])) and len(names) == 8
    assert all(a.startswith("vortfoil_wing_") and f"_a6_V{res['V']:g}_{res['state']}_" in a and " " not in a for a in names)
    about = pathlib.Path(paths[-1]).read_text(encoding="utf-8")
    rows = dict(zip(PARTS, [read(p) for p in paths[:-1]]))
    k = nearest(res, 0.5)                                         # the default cut strip
    assert {p: len(t) - 1 for p, t in rows.items()} == dict(
        summary=1, stations=len(res["defn"]["stations"]), spanwise=n, polar=len(res["polar"]["alpha"]), surface=2 * n * len(res["surface"]["x_c"]),
        section_tables=n * m, cut=len(res["strip_detail"]["x"][k]) if sst else 240)
    for part, table in rows.items():                              # every column is explained, every row is complete
        assert all(f"\n  {col}: " in about for col in table[0]) and all(len(r) == len(table[0]) for r in table), part
    col = lambda part, name: [r[rows[part][0].index(name)] for r in rows[part][1:]]
    s = dict(zip(*rows["summary"]))
    assert abs(float(s["CL"]) - res["coef"]["CL"]) < 1e-7 and s["state"] == res["state"] and float(s["alpha_deg"]) == 6.0 and ("CD0_pressure" in s) == sst
    assert ("Cf" in rows["surface"][0]) == sst and ("theta_over_c" in rows["surface"][0]) != sst and float(col("spanwise", "eta")[0]) < -0.9
    assert np.allclose(np.array(col("spanwise", "cl"), float), res["strips"]["cl"], atol=1e-7) and col("spanwise", "strip")[:2] == ["0", "1"]
    assert col("surface", "side")[0] == "upper" and col("surface", "side")[n * len(res["surface"]["x_c"])] == "lower"
    assert float(col("polar", "CL")[0]) == pytest.approx(res["polar"]["CL"][0], abs=1e-7)
    assert ("RANS state." if sst else "XFOIL state.") in about and "nose up" in about and "Wind frame" in about
    assert col("polar", "in_training_range") == [str(int(v)) for v in res["polar"]["in_range"]] and s["in_training_range"] == str(int(res["coef"]["in_range"]))     # the flags
    assert "\n  in_training_range: 1: every strip's" in about and "the figure draws these points hollow" in about


def test_export_chosen_parts_only(res, tmp_path):
    folder = tmp_path / "some"
    paths = VW.export_csv(folder, res, parts=["polar", "cut", "summary"], cut_index=3, name="My wing/1")
    assert [os.path.basename(p).rsplit("_", 1)[1] for p in paths] == ["summary.csv", "polar.csv", "cut.csv", "about.txt"]
    assert sorted(os.listdir(folder)) == sorted(os.path.basename(p) for p in paths)                             # nothing else was written
    assert all(os.path.basename(p).startswith("vortfoil_wing_My_wing_1_a6_V") for p in paths)
    cut = read(paths[2])
    assert {r[0] for r in cut[1:]} == {"3"} and abs(float(cut[1][1]) - res["strips"]["eta"][3]) < 1e-6
    about = pathlib.Path(paths[-1]).read_text(encoding="utf-8")
    assert "_polar.csv" in about and "_cut.csv" in about and "strip 3" in about
    assert "_surface.csv" not in about and "_spanwise.csv" not in about and "section_tables" not in about       # only what was written
    with pytest.raises(ValueError, match="unknown CSV part 'everything'"):
        VW.export_csv(folder, res, parts=["polar", "everything"])
    with pytest.raises(ValueError, match="cut_index"):
        VW.export_csv(folder, res, parts=["cut"], cut_index=res["wing"]["n"])
    assert len(VW.export_csv(tmp_path / "one", res, parts=("stations",))) == 2


# ------------------------------------------------------------------------------------------- the state's training range
def test_polar_marks_the_points_outside_the_training_range(results):
    """The polar's points that the backend flags outside the state's training range are drawn faded, dashed and hollow, named in a
    legend; a polar inside it has no legend and the figure looks as before; the dot of this incidence is hollow when it is outside."""
    for key in ("xfoil", "sst", "far_xfoil", "far_sst"):
        res = results[key]
        ok = np.asarray(res["polar"]["in_range"])
        for q in VW.POLAR_PLOTS:
            fig, view = make(res, options={"polar": q})
            ax = fig.axes[6]
            xk, yk = VW.POLAR[q][:2]
            x, y = np.asarray(res["polar"][xk]), np.asarray(res["polar"][yk])
            solid, dot = ax.lines[0], ax.lines[-1]
            assert np.array_equal(np.isnan(solid.get_ydata()), ~ok) and solid.get_linestyle() == "-" and solid.get_alpha() is None   # solid where inside
            assert dot.get_marker() == "o" and (dot.get_markerfacecolor() == "k") == bool(res["coef"]["in_range"])
            if ok.all():
                assert ax.get_legend() is None and len(ax.lines) == 2
                continue
            faded, hollow, sample = ax.lines[1], ax.lines[2], ax.lines[3]
            assert faded.get_alpha() == VA.FADE and faded.get_linestyle() == "--" and np.allclose(faded.get_xdata(), x) and np.allclose(faded.get_ydata(), y)
            assert hollow.get_markerfacecolor() == "white" and np.allclose(hollow.get_xdata(), x[~ok]) and np.allclose(hollow.get_ydata(), y[~ok])
            assert [t.get_text() for t in ax.get_legend().get_texts()] == ["inside the training range", "outside the training range"]
            assert text_problems(fig) == [], (key, q)
    for key, state in (("xfoil", "xfoil"), ("sst", "sst")):                        # the polar spans the state's training incidences
        a = results[key]["polar"]["alpha"]
        assert (a[0], a[-1]) == ((-10.0, 18.0) if state == "xfoil" else (-5.0, 15.0))
    old = dict(results["xfoil"], polar={k: v for k, v in results["xfoil"]["polar"].items() if k != "in_range"})      # a result without flags
    assert make(old)[0].axes[6].get_legend() is None
    xf = results["xfoil"]                                                          # no point inside: one entry, the outside one
    none = dict(xf, polar=dict(xf["polar"], in_range=np.zeros(len(xf["polar"]["alpha"]), bool)), coef=dict(xf["coef"], in_range=False))
    fig, view = make(none)
    assert [t.get_text() for t in fig.axes[6].get_legend().get_texts()] == ["outside the training range"] and text_problems(fig) == []


@pytest.mark.parametrize("layout", list(VA.DESIGNS))
def test_numbers_say_when_the_requested_incidence_is_outside_the_training_range(results, layout):
    """Outside the state's training range (a strip's incidence or Reynolds number outside it, or a strip past stall) the numbers say so
    right after the incidence, with the ranges; inside it they say nothing; the RANS state's lead sentence about a Reynolds number far
    outside covers that case alone."""
    for key, name, rng in (("far_xfoil", "XFOIL", "strip incidence -10 to 18 deg, Re 5e4 to 1e7"), ("far_sst", "RANS", "strip incidence -5 to 15 deg, Re 2e6 to 6e6")):
        res = results[key]
        assert not res["coef"]["in_range"]
        fig, view = make(res, layout)
        said = f"Outside the {name} state's training range ({rng}, no strip past stall): the numbers are extrapolated."
        lines = view._text.get_text().split("\n")
        i = [j for j, s in enumerate(lines) if s.startswith("Outside the")][0]
        assert lines[i - 1].startswith("V ") and " ".join(s.strip() for s in lines[i:i + 4])[:len(said)] == said and max(map(len, lines)) <= VW.WRAP
        assert text_problems(fig) == []
    for key in ("xfoil", "sst"):
        assert results[key]["coef"]["in_range"] and "Outside the" not in make(results[key], layout)[1]._text.get_text()
    ss = results["sst"]
    covered = dict(ss, coef=dict(ss["coef"], in_range=False), sst_range=dict(ss["sst_range"], message="RANS state outside the field decoder's Reynolds range (2e6-6e6) on 8 of 8 strips: the RANS-state numbers are not meaningful here; use the XFOIL state"))
    fig, view = make(covered, layout)
    assert "Outside the" not in view._text.get_text() and view._text.get_text().startswith("RANS state outside the field decoder's Reynolds range")
    assert text_problems(fig) == []


def test_option_tables():
    assert PARTS == ["summary", "stations", "spanwise", "polar", "surface", "section_tables", "cut"]
    assert all(len(v) == 2 and v[0] and v[1].isascii() for v in VW.CSV_PARTS.values())
    assert VW.SURFACE_QUANTITIES == {"xfoil": ["C_p", "theta/c", "delta*/c", "H"], "sst": ["C_p", "C_f"]}
    assert VW.POLAR_PLOTS == ["C_L vs alpha", "C_D vs C_L", "L/D vs alpha", "C_M vs alpha"] and len(VW.SPAN_QUANTITIES) == 10
