# -*- coding: utf-8 -*-
"""The field route for any section (code/field/predict_field.py) runs from the repository's own files: no AirfRANS
dataset and no XFOIL. Its accuracy against RANS is checked by code/field/validate_predict_field.py, not here."""
import os
import sys
import warnings

import numpy as np
import pytest

import vortfoil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "code", "field"))

NACA2412_CST = np.array([0.1721, 0.1595, 0.2032, 0.1581, 0.2017, 0.1668, 0.1830, 0.1752,
                         0.1101, 0.0913, 0.1129, 0.0777, 0.1069, 0.0847, 0.0932, 0.0967], np.float32)


@pytest.fixture(scope="module")
def PF():
    import vortfoil_paths
    vortfoil_paths.add_paths()
    import predict_field
    return predict_field


@pytest.fixture(scope="module")
def section():
    x, yu, yl = vortfoil.geometry.surface(NACA2412_CST, 121)          # 241 nodes: the sparse (resampled) route
    return np.r_[np.c_[x[::-1], yu[::-1]], np.c_[x[1:], yl[1:]]]


def test_field_route_inside_the_domain(PF, section):
    with warnings.catch_warnings():
        warnings.simplefilter("error", PF.FieldRangeWarning)
        out, meta = PF.predict(section, 4.0, 3e6, window=(-0.5, 1.5, -0.5, 0.5), h=0.05, family="naca")
    fluid = ~out["inside"]
    for k in ("u", "v", "p", "nut"):
        assert np.isfinite(out[k][fluid]).all(), k
    assert meta["section_resampled"] and len(meta["section"]) == 441
    assert 0.3 < meta["cl_gamma"] < 1.2 and 0.0 <= meta["s_sep"] <= 1.0
    up, lo = (out["y"] > 0.08) & (np.abs(out["x"] - 0.3) < 0.1), (out["y"] < -0.08) & (np.abs(out["x"] - 0.3) < 0.1)
    assert np.nanmean(out["cp"][up]) < np.nanmean(out["cp"][lo])        # suction above a lifting section
    assert abs(np.nanmean(out["u_over_U"][:, 0]) - 1.0) < 0.15          # near-freestream at the upstream edge


def test_field_route_warns_outside_the_domain(PF, section):
    with pytest.warns(PF.FieldRangeWarning, match="Reynolds"):
        PF.predict(section, 4.0, 1e5, h=0.1, family="naca")
    with pytest.warns(PF.FieldRangeWarning, match="incidence"):
        PF.predict(section, 20.0, 3e6, h=0.1, family="naca")


def test_family_warning(PF, section):
    """Coordinates warn that the decoder knows only NACA 4/5-digit sections; a NACA designation does not."""
    with pytest.warns(PF.FieldRangeWarning, match="NACA 4- and 5-digit sections only"):
        PF.predict(section, 4.0, 3e6, h=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PF.FieldRangeWarning)
        _, meta = PF.predict(PF.naca_coords("2412"), 4.0, 3e6, h=0.1, family="naca")
    assert meta["family"] == "naca" and not meta["warnings"]


def test_open_trailing_edge_is_closed(PF):
    x, yu, yl = vortfoil.geometry.surface(NACA2412_CST, 121)
    loop = np.r_[np.c_[x[::-1], yu[::-1] + 0.002 * x[::-1]], np.c_[x[1:], yl[1:] - 0.002 * x[1:]]]   # 0.4% chord gap
    g, shift = PF.close_trailing_edge(loop)
    assert np.allclose(g[0], g[-1]) and abs(shift - 0.002) < 1e-9


def test_closed_loop_with_a_coarse_trailing_edge_panel_stays_in_frame(PF):
    loop = PF.naca_coords("2412")[::20]                                   # closed, last panel 2.5% chord long
    g, rot, _, _ = PF.field_frame(loop)
    assert rot == 0.0 and np.allclose(g, loop[:-1])
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        _, meta = PF.predict(loop, 4.0, 3e6, window=(-0.5, 1.5, -0.5, 0.5), h=0.1)
    assert meta["te_gap"] == 0.0 and not any("gap" in str(w.message) for w in ws)


def test_pool_workers_follow_the_callers_errstate(PF):
    """The fast kernels' worker threads use the caller's np.errstate, which numpy keeps per thread."""
    import vf_field_fast as F

    def bad(lo, hi):
        np.ones(1) / np.zeros(1)
    with np.errstate(divide="raise"), pytest.raises(FloatingPointError):
        F._parallel(4 * F.CHUNK, bad)
    with np.errstate(divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("error")
        F._parallel(4 * F.CHUNK, bad)



def test_wall_values_in_the_rans_state(PF):
    """The decoder's wall values (surface()): no lift on a symmetric section at zero incidence, a stagnation C_p near 1,
    NACA 2412 at 4 deg near its RANS lift, and a drag of pressure plus wall-function friction near the fully turbulent
    level. Their accuracy against RANS is checked by code/field/validate_field_surface.py."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PF.FieldRangeWarning)
        sym = PF.surface(PF.naca_coords("0012"), 0.0, 3e6, family="naca")
        r = PF.surface(PF.naca_coords("2412"), 4.0, 3e6, family="naca")
    assert abs(sym["cl"]) < 0.01
    assert np.isfinite(r["cp"]).all() and 0.95 < r["cp"].max() < 1.05
    assert 0.60 < r["cl"] < 0.70 and 0.0 < r["cd_pressure"] < 0.01
    assert r["upper"].sum() > 100 and (~r["upper"]).sum() > 100
    # drag = pressure + wall-function friction: fully turbulent friction of a 12% section near 2 x 0.0037 at Re 3e6
    assert 0.007 < sym["cd"] < 0.015 and 0.4 < sym["cd_friction"] / sym["cd"] < 0.9
    assert 0.007 < r["cd"] < 0.02 and abs(r["cd"] - r["cd_pressure"] - r["cd_friction"]) < 1e-12
    assert np.isfinite(r["cf"]).all() and np.mean(r["cf"] > 0) > 0.9 and 10 < r["yplus_median"] < 80


def _wall_relative_edge(n, u, thresh=0.01):
    """The edge test the research gate replaced (first n where |du/dn| < thresh |du/dn|_wall): defined here only, as the
    mutant arm of the edge test below."""
    g = np.abs(np.gradient(u, n))
    k = np.where(g < thresh * g[0])[0]
    k = k[k >= 3]
    return int(k[0]) if len(k) else None


def test_boundary_layer_extraction(PF):
    """The RANS-state boundary layer's extractor (predict_field.layer_edge, layer_state, bl_states) on profiles with known
    answers, after the research audit's unit tests: the edge follows the layer and not the Reynolds number (U1: the viscous
    scale moves 10x under a fixed layer, the edge <= 10%, the wall-relative test >= 2x); the integrals start at the wall (U6:
    a linear layer gives theta = delta/6, delta* = delta/2); over a curved front, where the inviscid speed falls off the wall,
    u_t / U_I reads the layer and u_t alone does not; reversed near-wall flow is flagged (U5) and an edge in the outer fifth of
    the ray is not read (U3)."""
    delta, n = 1e-2, np.linspace(0.0, 6e-2, 60000)[1:]
    outer = np.where(n < delta, np.sin(0.5 * np.pi * np.minimum(n / delta, 1.0)) ** 2, 1.0)
    edges = [(n[PF.layer_edge(n, 0.2 * np.tanh(n / eps) + 0.8 * outer)], n[_wall_relative_edge(n, 0.2 * np.tanh(n / eps) + 0.8 * outer)])
             for eps in (2e-5, 6e-5, 2e-4)]
    new, old = np.array(edges).T
    assert new.max() / new.min() <= 1.10 and old.max() / old.min() >= 2.0
    N, U = PF.BL_N, 40.0
    zero = np.zeros((1, len(N)))
    lin = PF.bl_states(N, U * np.minimum(N / delta, 1.0)[None], zero, zero, U)
    assert abs(lin["theta"][0] / (delta / 6) - 1) < 0.02 and abs(lin["dstar"][0] / (delta / 2) - 1) < 0.02
    assert abs(lin["H"][0] - 3.0) < 0.05 and abs(lin["Ue"][0] - 1.0) < 1e-12 and not lin["separated"][0]
    f = np.sin(0.5 * np.pi * np.minimum(N / 3e-3, 1.0))                   # a thin layer, theta = (2/pi - 1/2) 3e-3
    ui = U * (1.0 + 0.4 * np.exp(-N / 0.01))                              # inviscid speed falling off a curved wall
    p = 0.5 * (U ** 2 - ui ** 2)[None]                                    # Bernoulli, freestream total pressure
    cur, raw = PF.bl_states(N, (ui * f)[None], zero, p, U), PF.bl_states(N, (ui * f)[None], zero, zero, U)
    assert abs(cur["theta"][0] / ((2 / np.pi - 0.5) * 3e-3) - 1) < 0.02 and abs(cur["Ue"][0] - ui[N <= cur["delta"][0]][-1] / U) < 1e-12
    assert not abs(raw["theta"][0] / ((2 / np.pi - 0.5) * 3e-3) - 1) < 0.5          # NaN or far off: the gate's reading of u_t
    rev = U * (np.sin(0.5 * np.pi * np.minimum(N / 0.02, 1.0)) - 0.15 * np.exp(-N / 2e-3))
    pin = U * np.minimum(N / 0.09, 1.0)
    st = PF.bl_states(N, np.r_[rev[None], pin[None]], np.r_[zero, zero], np.r_[zero, zero], U)
    assert list(st["separated"]) == [True, False] and st["H"][0] > 2.7 and np.isnan(st["theta"][1])     # 2.66 without the reversal
    # with n_upper (bl_rays' stations: 2 upper, then 4 lower, each from the leading edge) the run of flagged stations from the
    # leading edge on the surface of the highest near-wall pressure lies ahead of the stagnation point and is not flagged; a
    # reversal on the other surface, or after an attached station, is
    att = U * np.sin(0.5 * np.pi * np.minimum(N / 0.02, 1.0))
    prof = np.array([rev, att, rev, rev, att, rev])
    pw = np.zeros_like(prof); pw[:, 0] = [0.0, 0.0, 30.0, 70.0, 20.0, 0.0]
    assert list(PF.bl_states(N, prof, 0 * prof, pw, U)["separated"]) == [True, False, True, True, False, True]
    assert list(PF.bl_states(N, prof, 0 * prof, pw, U, 2)["separated"]) == [True, False, False, False, False, True]


def test_boundary_layer_of_a_section(PF, monkeypatch):
    """boundary_layer() on NACA 2412 at 4 deg, Re 3e6: the stations, predict(bl=True) giving the same numbers, C_f exactly
    surface()'s at the station nodes, an attached layer growing downstream with a turbulent shape factor, its edge well inside
    the ray (U3), and theta, delta* settling as the edge fraction moves 0.98-0.999 (U4: the decoder's own sensitivity; on the
    RANS fields the median spread is 3%, code/field/validate_field_boundary_layer.py). Its accuracy against RANS is checked
    there. At 12 deg the lower-surface stations ahead of the stagnation point, whose near-wall flow runs toward the leading
    edge, are not flagged separated."""
    loop = PF.naca_coords("2412")
    with warnings.catch_warnings():
        warnings.simplefilter("error", PF.FieldRangeWarning)                  # inside the domain: no warning
        b = PF.boundary_layer(loop, 4.0, 3e6, family="naca")
        out, meta = PF.predict(loop, 4.0, 3e6, h=0.25, family="naca", wall=True, bl=True)
        s = PF.surface(loop, 4.0, 3e6, family="naca")
    assert meta["in_training_range"] and b["meta"]["in_training_range"]
    for side in ("upper", "lower"):
        d, m = b[side], meta["boundary_layer"][side]
        assert set(d) == {"x_over_c", "theta", "dstar", "H", "Ue", "delta", "Cf", "separated"} == set(m)
        assert all(np.array_equal(d[k], np.asarray(m[k]), equal_nan=True) for k in d)
        x = d["x_over_c"]
        assert 80 < len(x) < 110 and np.all(np.diff(x) > 0) and x[0] >= 0.01 and x[-1] <= 0.99
        j = [int(np.flatnonzero((s["x"] == xi) & (s["upper"] == (side == "upper")))[0]) for xi in x]
        assert np.array_equal(d["Cf"], s["cf"][j])
        aft = (x >= 0.1) & (x <= 0.95)
        assert np.isfinite(d["theta"][aft]).all() and not d["separated"][aft].any() and 1.2 < np.median(d["H"][aft]) < 1.8
        assert np.mean(d["theta"][x > 0.8]) > 2.0 * np.mean(d["theta"][aft & (x < 0.3)])
        assert np.median(d["delta"][aft]) / PF.BL_BAND <= 0.5
    lo = PF.boundary_layer(loop, 12.0, 3e6, family="naca")["lower"]
    assert (lo["Cf"][:3] < 0).all() and not lo["separated"].any()
    edge, ladder = PF.layer_edge, []
    for frac in (0.98, 0.99, 0.995, 0.999):
        monkeypatch.setattr(PF, "layer_edge", lambda n, u, frac=frac: edge(n, u, frac))
        r = PF.boundary_layer(loop, 4.0, 3e6, family="naca")
        ladder.append([np.r_[r["upper"][q], r["lower"][q]] for q in ("theta", "dstar")])
    for L in np.moveaxis(np.array(ladder), 1, 0):
        ok = np.isfinite(L).all(0)
        assert np.median((L.max(0) - L.min(0))[ok] / np.median(L, axis=0)[ok]) <= 0.12


def test_old_edge_criterion_is_not_in_the_release():
    """No file of the release carries the wall-relative edge test (research audit U2); the scan finds it in a synthetic line."""
    import glob, re
    pat = r"<\s*[0-9.eE+-]+\s*\*\s*[A-Za-z_][A-Za-z_0-9.]*\s*\[\s*0\s*\]"
    assert re.search(pat, "k = np.where(g < 0.01 * g[0])[0]")
    files = glob.glob(os.path.join(ROOT, "code", "**", "*.py"), recursive=True) + glob.glob(os.path.join(ROOT, "src", "**", "*.py"), recursive=True)
    assert len(files) > 20 and not [f for f in files if re.search(pat, open(f, encoding="utf-8", errors="replace").read())]


def test_viewer_state_flags(PF):
    """view_airfoil.compute flags each state's training range by incidence and Reynolds number (the surface model -10..18 deg,
    5e4..1e7; the field decoder -5..15 deg, 2e6..6e6); the decoder's family note does not count; the RANS-state boundary layer
    comes with the result."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import view_airfoil as V
    loop = PF.naca_coords("2412")
    for alpha, Re, xfoil, rans in ((4.0, 3e6, True, True), (17.0, 3e6, True, False), (4.0, 2e7, False, False), (-12.0, 1e6, False, False)):
        r, sw, out, meta = V.compute(loop, alpha, Re, 0.25, "naca")
        assert (meta["xfoil_in_range"], meta["rans_in_range"]) == (xfoil, rans), (alpha, Re)
        assert set(meta["boundary_layer"]) == {"upper", "lower"} and len(meta["boundary_layer"]["upper"]["theta"]) > 80
    r, sw, out, meta = V.compute(loop, 4.0, 3e6, 0.25)                       # plain coordinates: the family warning only
    assert meta["xfoil_in_range"] and meta["rans_in_range"] and any("NACA 4- and 5-digit" in m for m in meta["warnings"])

def test_viewer_window_resize_zooms_the_figure():
    """Resizing the window scales the frozen design layout (letterboxed) instead of re-flowing it."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import view_airfoil as V
    loop, name = V.section("2412")
    r, sw, out, meta = V.compute(loop, 4.0, 3e6, 0.05)
    fig = V.figure(name, 4.0, 3e6, r, sw, out, meta)
    on_resize = V.scale_with_window(fig)
    W0, H0 = fig.get_size_inches()
    pos0 = np.array([ax.get_position().bounds for ax in fig.axes]); f0 = fig.axes[0].title.get_fontsize()
    fig.set_size_inches(W0 / 2, H0 / 2); on_resize()
    assert np.allclose([ax.get_position().bounds for ax in fig.axes], pos0) and abs(fig.axes[0].title.get_fontsize() - f0 / 2) < 1e-9
    fig.set_size_inches(W0, 2 * H0); on_resize()
    pos = np.array([ax.get_position().bounds for ax in fig.axes])
    assert abs(fig.axes[0].title.get_fontsize() - f0) < 1e-9 and np.allclose(pos[:, 3], pos0[:, 3] / 2)


def test_viewer_layouts():
    """The three layouts of the result figure build at their design sizes, with the states' panels: 4 fields, 4 colour bars, C_p,
    the boundary layer (and below it the RANS state's C_f), the text."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import view_airfoil as V
    loop, name = V.section("2412")
    res = V.compute(loop, 4.0, 3e6, 0.05)
    for layout, size in V.DESIGNS.items():
        for states, n in ((("xfoil", "rans"), 12), (("xfoil",), 11), (("rans",), 12), ("rans", 12)):
            fig = V.figure(name, 4.0, 3e6, *res, layout=layout, states=states)
            assert tuple(fig.get_size_inches()) == size and len(fig.axes) == n, (layout, states)
            matplotlib.pyplot.close(fig)
    for bad in ((), ("sst",), ("xfoil", "x"), ""):
        with pytest.raises(ValueError, match="states"):
            V.figure(name, 4.0, 3e6, *res, states=bad)


def text_problems(fig):
    """what is wrong with the figure's text: outside the figure, overlapping another text, or (the coefficients) outside its cell. Drawn
    texts only: the suptitle, titles, axis labels, tick labels in view, texts in axes and legends."""
    import itertools
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [("suptitle", fig._suptitle.get_window_extent(r))] if fig._suptitle is not None else []
    for i, ax in enumerate(fig.axes[:-1]):
        for name, t in ([("title", ax.title), ("xlabel", ax.xaxis.label), ("ylabel", ax.yaxis.label)] + [(f"text{j}", t) for j, t in enumerate(ax.texts)]):
            if t.get_text().strip():
                boxes.append((f"ax{i}.{name}", t.get_window_extent(r)))
        if ax.get_legend():
            boxes.append((f"ax{i}.legend", ax.get_legend().get_window_extent(r)))
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            for lab, loc in zip(axis.get_majorticklabels(), axis.get_majorticklocs()):
                if min(lim) <= loc <= max(lim) and lab.get_text().strip():
                    boxes.append((f"ax{i}.tick{loc:g}", lab.get_window_extent(r)))
    W, H = fig.bbox.width, fig.bbox.height
    bad = [f"outside the figure: {n}" for n, b in boxes if b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1]
    bad += [f"overlap: {n1} / {n2}" for (n1, b1), (n2, b2) in itertools.combinations(boxes, 2)
            if min(b1.x1, b2.x1) - 1 > max(b1.x0, b2.x0) + 1 and min(b1.y1, b2.y1) - 1 > max(b1.y0, b2.y0) + 1]
    cell, lines = fig.axes[-1].get_window_extent(r), [t.get_window_extent(r) for t in fig.axes[-1].texts]
    bad += [f"coefficients outside their cell: {b}" for b in lines if b.x0 < cell.x0 - 1 or b.x1 > cell.x1 + 1 or b.y0 < cell.y0 - 1 or b.y1 > cell.y1 + 1]
    bad += [f"coefficient lines overlap: {i}" for i, (a, b) in enumerate(zip(lines, lines[1:])) if b.y1 > a.y0 + 1]
    return bad


@pytest.fixture(scope="module")
def cases(PF):
    """the viewer's results on NACA 2412 (a coarse field): 4 deg at Re 3e6 (inside both training ranges), 12 deg at Re 3e6 (inside both,
    with the RANS-state lower-surface caveat), 17 deg at Re 3e6 (outside the field decoder's), 4 deg at Re 5e5 (outside it by Reynolds
    number, its numbers meaningless), 20 deg at Re 3e5 (outside both), -10 deg at Re 5e4 (outside the decoder's, no layer read at any
    station)"""
    import view_airfoil as V
    loop = PF.naca_coords("2412")
    out = {k: ("NACA 2412", a, Re) + V.compute(loop, a, Re, 0.25, "naca") for k, (a, Re) in
           {"in": (4.0, 3e6), "in_high": (12.0, 3e6), "rans_out": (17.0, 3e6), "rans_out_re": (4.0, 5e5), "both_out": (20.0, 3e5)}.items()}
    # no layer read anywhere: the AirfRANS-only decoder of 2.0.0 gives that state at -10 deg, Re 5e4 (the co-trained one reads a few)
    f0cm = os.path.join(os.path.dirname(PF.DECODER), "field_decoder_f0cm_s1.npz")
    out["none_read"] = ("NACA 2412", -10.0, 5e4) + V.compute(loop, -10.0, 5e4, 0.25, "naca", decoder=f0cm)
    return out


def legend_hits(fig):
    """How many plotted points of the boundary-layer panel's curves drawn at full strength (a state inside its training range) lie under
    its legend."""
    import view_airfoil as V
    fig.canvas.draw()
    bl = fig.axes[9]
    bb, n = bl.get_legend().get_window_extent(), 0
    for ln in bl.lines:
        if len(ln.get_xdata()) and ln.get_alpha() != V.FADE:
            xy = bl.transData.transform(np.c_[ln.get_xdata(), ln.get_ydata()].astype(float))
            xy = xy[np.isfinite(xy).all(axis=1)]
            n += int(((xy[:, 0] > bb.x0) & (xy[:, 0] < bb.x1) & (xy[:, 1] > bb.y0) & (xy[:, 1] < bb.y1)).sum())
    return n


@pytest.mark.parametrize("serif", [["Times New Roman", "DejaVu Serif"], ["DejaVu Serif"]])          # the second: without Times New Roman (Linux)
@pytest.mark.parametrize("layout", ["landscape", "square", "portrait"])
def test_viewer_text_fits_in_every_layout_and_state(PF, cases, layout, serif, monkeypatch):
    """Each state set in each layout, inside both ranges and outside them, with the long warnings of a section that is not a NACA one:
    nothing overlaps or leaves the figure, and the coefficients stay in their cell (a long list gets smaller type, then a pointer to the CSV)."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import view_airfoil as V
    monkeypatch.setitem(PF.STYLE, "font.serif", serif)
    for key, case in cases.items():
        for states in (("xfoil", "rans"), ("xfoil",), ("rans",)):
            fig = V.figure(*case, layout=layout, states=states)
            assert text_problems(fig) == [], (key, states)
            matplotlib.pyplot.close(fig)
    name, a, Re, r, sw, out, meta = cases["both_out"]
    many = [f"a long surface warning number {i} that needs wrapping, " * 3 for i in range(8)]
    fig = V.figure(name, a, Re, r, many, out, meta, layout=layout)
    assert text_problems(fig) == [] and fig.axes[-1].texts[-1].get_text().startswith("- ... (the rest is in the CSV")
    matplotlib.pyplot.close(fig)


@pytest.mark.parametrize("serif", [["Times New Roman", "DejaVu Serif"], ["DejaVu Serif"]])          # the second: without Times New Roman (Linux)
@pytest.mark.parametrize("layout", ["landscape", "square", "portrait"])
def test_viewer_text_after_a_window_zoom(PF, cases, layout, serif, monkeypatch):
    """The zoomed, frozen figure (scale_with_window) keeps the states' text clear of the edges and of its neighbours from 0.8 to 1.5 times
    its design size (the released figure's axis-label pads do not shrink: below 0.7 its y labels leave the figure), with the legends as
    they were (they are rebuilt at the scaled type size)."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import view_airfoil as V
    monkeypatch.setitem(PF.STYLE, "font.serif", serif)
    fig = V.figure(*cases["rans_out"], layout=layout)
    legends = [[t.get_text() for t in ax.get_legend().get_texts()] for ax in fig.axes if ax.get_legend()]
    W, H = V.DESIGNS[layout]
    on_resize = V.scale_with_window(fig)
    for s in (0.8, 1.5):
        fig.set_size_inches(W * s, H * s, forward=False)
        on_resize()
        assert text_problems(fig) == [] and [[t.get_text() for t in ax.get_legend().get_texts()] for ax in fig.axes if ax.get_legend()] == legends, (layout, s)
        assert legend_hits(fig) == 0, (layout, s)
    matplotlib.pyplot.close(fig)


def test_viewer_states_in_range(cases):
    """Inside both training ranges the two states are drawn at full strength (XFOIL solid, RANS dashed) and named in the legends;
    one state alone shows only its own curves, numbers and legend."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import view_airfoil as V
    name, a, Re, r, sw, out, meta = cases["in"]
    fig = V.figure(*cases["in"], layout="square")
    cp, bl, cf, txt = fig.axes[8], fig.axes[9], fig.axes[10], fig.axes[-1]
    assert [t.get_text() for t in cp.get_legend().get_texts()] == ["XFOIL state", "RANS state"]
    assert [(ln.get_linestyle(), ln.get_alpha()) for ln in cp.lines[:2]] == [("-", 1.0)] * 2 and [ln.get_linestyle() for ln in cp.lines[3:5]] == ["--"] * 2
    assert [t.get_text() for t in bl.get_legend().get_texts()] == [r"XFOIL state: $\delta^*$ solid, $\theta$ dashed", r"RANS state: $\delta^*$ filled, $\theta$ open dots"]
    assert bl.get_xlim() == (0.0, 1.0) and cf.get_ylabel() == r"$C_f$" and cf.get_xlabel() == "x / c" and bl.get_xlabel() == ""      # one x axis for the two
    x = np.concatenate([ln.get_xdata() for ln in cf.lines[:2]])
    assert x.min() >= V.CHECKED_X and len(cf.lines[0].get_xdata()) > 70                      # the RANS layer from x/c 0.1
    assert "outside" not in "".join(t.get_text() for t in txt.texts) and {t.get_color() for t in txt.texts} == {"k"}
    assert not any(ax.texts for ax in fig.axes[:4]) and {c.get_alpha() for ax in fig.axes[:8] for c in ax.collections} == {None}      # the field panels at full strength
    lines = [t.get_text() for t in txt.texts]
    assert lines[1] == "XFOIL state (surface model)" and "RANS state (field decoder, wall function)" in lines
    assert any(f"C_l {r.cl[0]: .4f}" in s for s in lines) and any(f"C_l {meta['wall']['cl']: .4f}" in s for s in lines)
    assert any(f"{meta['wall']['cd_pressure']:.5f} + friction {meta['wall']['cd_friction']:.5f}" in s for s in lines)
    assert any(f"C_m {meta['wall']['cm']: .4f}" in s for s in lines) and "XFOIL and RANS states" in fig._suptitle.get_text()
    caveat = lambda case, states: any(t.get_text().startswith("  lower surface: often 20%") for t in V.figure(*case, layout="square", states=states).axes[-1].texts)
    assert not caveat(cases["in"], V.STATES) and caveat(cases["in_high"], V.STATES) and not caveat(cases["in_high"], ("xfoil",))    # from 8 deg on
    for states, gone, kept in ((("xfoil",), "RANS", "XFOIL"), (("rans",), "XFOIL", "RANS")):
        one = V.figure(*cases["in"], layout="square", states=states)
        joined = "".join(t.get_text() for t in one.axes[-1].texts) + "".join(t.get_text() for ax in one.axes for lg in [ax.get_legend()] if lg for t in lg.get_texts())
        assert gone not in joined and kept in joined and gone not in one._suptitle.get_text().split("surface, ")[1], states
    matplotlib.pyplot.close("all")


def test_viewer_state_outside_its_training_range_is_faded(cases):
    """A state outside its training range is drawn at alpha 0.35 (dashed), with '(outside training range)' in its legend entries and its
    numbers pale under a heading that says so; the state inside is unchanged; the axes follow the state that is inside (a decoder's
    numbers outside its range may be nonsense and must not squash the others)."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import view_airfoil as V
    name, a, Re, r, sw, out, meta = cases["rans_out"]
    assert meta["xfoil_in_range"] and not meta["rans_in_range"]
    fig = V.figure(*cases["rans_out"], layout="square")
    cp, bl, cf, txt = fig.axes[8], fig.axes[9], fig.axes[10], fig.axes[-1]
    assert [t.get_text() for t in cp.get_legend().get_texts()] == ["XFOIL state", "RANS state (outside training range)"]
    assert [ln.get_alpha() for ln in cp.lines[:2]] == [1.0, 1.0] and [ln.get_alpha() for ln in cp.lines[3:5]] == [V.FADE, V.FADE]
    assert len({str(V.look(s, inside)["ls"]) for s in V.STATES for inside in (True, False)}) == 4       # four looks: a faded state keeps its own dashes
    assert "outside training range" in bl.get_legend().get_texts()[1].get_text() and "outside" not in bl.get_legend().get_texts()[0].get_text()
    dots = [ln for ln in bl.lines if ln.get_marker() == "o" and len(ln.get_xdata())]
    assert dots and {ln.get_alpha() for ln in dots} == {V.FADE}
    assert [ln.get_alpha() for ln in bl.lines if ln.get_linestyle() in ("-", "--") and len(ln.get_xdata())] == [1.0] * 4       # the XFOIL lines
    assert all(ln.get_alpha() == V.FADE for ln in cf.lines[:2]) and "outside training range" in cf.get_title(loc="right") and not cf.texts
    for ax in fig.axes[:4]:                                      # the field is the RANS state's too: faded, with its colour bar, and labelled
        assert [t.get_text() for t in ax.texts] == ["outside training range"] and {c.get_alpha() for c in ax.collections} == {V.FADE}
    from matplotlib.collections import QuadMesh
    assert {c.get_alpha() for cax in fig.axes[4:8] for c in cax.collections if isinstance(c, QuadMesh)} == {V.FADE}
    inside = V.figure(*cases["in"], layout="square")
    assert not any(ax.texts for ax in inside.axes[:4]) and {c.get_alpha() for ax in inside.axes[:8] for c in ax.collections} == {None}
    pale = [t.get_text() for t in txt.texts if t.get_color() != "k"]
    assert pale[0] == "RANS state: outside training range" and len(pale) == 5 and all("XFOIL" not in s for s in pale)
    assert [t.get_text() for t in txt.texts if t.get_color() == "k"][1] == "XFOIL state (surface model)"
    assert "RANS field decoder, wind frame, outside its range" in fig._suptitle.get_text() and text_problems(fig) == []
    both = cases["both_out"]
    assert not both[6]["xfoil_in_range"] and not both[6]["rans_in_range"]
    fig = V.figure(*both, layout="portrait")
    cp, txt = fig.axes[8], fig.axes[-1]
    assert [t.get_text() for t in cp.get_legend().get_texts()] == ["XFOIL state (outside training range)", "RANS state (outside training range)"]
    assert {ln.get_alpha() for ln in cp.lines} == {V.FADE}                                      # both faded, still told apart (four looks above)
    pale = [t.get_text() for t in txt.texts if t.get_color() != "k"]
    assert [s for s in pale if s.endswith("outside training range")] == ["XFOIL state: outside training range", "RANS state: outside training range"] and len(pale) == 8
    one = V.figure(*both, layout="portrait", states=("xfoil",))
    assert [t.get_text() for t in one.axes[8].get_legend().get_texts()] == ["XFOIL state (outside training range)"]
    # the axes follow the state that is inside: at Re 5e5 the decoder's C_p runs off the scale of the sheet's
    name, a, Re, r, sw, out, meta = cases["rans_out_re"]
    fig = V.figure(*cases["rans_out_re"], layout="square")
    lo, hi = sorted(fig.axes[8].get_ylim())
    assert lo > r.cp[0].min() - 1.0 and hi < r.cp[0].max() + 1.0 and np.nanmax(np.abs(meta["wall"]["cp"])) > 3
    only = V.figure(*cases["rans_out_re"], layout="square", states=("rans",))
    assert max(only.axes[8].get_ylim()) - min(only.axes[8].get_ylim()) > 5                          # alone, it sets its own scale
    matplotlib.pyplot.close("all")


@pytest.mark.parametrize("serif", [["Times New Roman", "DejaVu Serif"], ["DejaVu Serif"]])          # the second: without Times New Roman (Linux)
@pytest.mark.parametrize("layout", ["landscape", "square", "portrait"])
def test_viewer_boundary_layer_legend_is_clear_of_the_data(PF, cases, layout, serif, monkeypatch):
    """The boundary-layer panel's legend keeps clear of the curves of the states inside their training range, whichever states are shown,
    thin layers or thick, at any incidence and Reynolds number (the axis grows by up to two decades above the data under it), in either
    serif (DejaVu Serif, where Times New Roman is missing, sets the legend some 20% wider)."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import view_airfoil as V
    monkeypatch.setitem(PF.STYLE, "font.serif", serif)
    for key, case in cases.items():
        for states in (("xfoil", "rans"), ("xfoil",), ("rans",)):
            fig = V.figure(*case, layout=layout, states=states)
            assert legend_hits(fig) == 0, (key, states)
            lo, hi = fig.axes[9].get_ylim()
            top = max([np.nanmax(ln.get_ydata()) for ln in fig.axes[9].lines if len(ln.get_xdata()) and ln.get_alpha() != V.FADE and np.isfinite(ln.get_ydata()).any()] or [0.0])
            assert lo == 6e-6 and (top == 0.0 or top < hi <= 100.0 * top * 1.0001), (key, states, hi, top)         # the data are in, and at most two decades of room above them
            matplotlib.pyplot.close(fig)


def test_viewer_boundary_layer_legend_says_when_no_layer_is_read(cases):
    """Where the decoder reads no layer at any station (-10 deg at Re 5e4) the RANS state's legend entry says so, with no dot sample for dots
    that are not there; where it does read one, the entry describes its dots."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import view_airfoil as V
    meta = cases["none_read"][6]
    assert all(not np.isfinite(meta["boundary_layer"][s][q]).any() for s in ("upper", "lower") for q in ("theta", "dstar"))
    for states, n in ((("xfoil", "rans"), 2), (("rans",), 1)):
        lg = V.figure(*cases["none_read"], layout="square", states=states).axes[9].get_legend()
        texts = [t.get_text() for t in lg.get_texts()]
        assert texts[-1] == "RANS state (outside training range)\nno layer read" and len(texts) == n and "dots" not in "".join(texts)
        handle = (getattr(lg, "legend_handles", None) or lg.legendHandles)[-1]
        assert str(handle.get_marker()).lower() == "none" and handle.get_linestyle() in ("None", "none", "")
    lg = V.figure(*cases["in"], layout="square").axes[9].get_legend()
    assert [t.get_text() for t in lg.get_texts()][-1] == r"RANS state: $\delta^*$ filled, $\theta$ open dots"
    assert (getattr(lg, "legend_handles", None) or lg.legendHandles)[-1].get_marker() == "o"
    assert not lg.get_frame_on() and V.figure(*cases["none_read"], layout="square").axes[9].get_legend().get_frame_on()   # thick layers: no room above, so a boxed legend where it fits
    matplotlib.pyplot.close("all")


def test_viewer_coefficient_lines_leave_room_under_an_underscore(cases):
    """The underscores of C_l, C_d and s_sep touched the letters of the next line on a screen: the line below a line with an underscore
    starts V.UNDER em further down; the other lines are V.LINE apart."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import view_airfoil as V
    for layout in V.DESIGNS:
        fig = V.figure(*cases["in_high"], layout=layout)
        ax = fig.axes[-1]
        ts = list(ax.texts)
        em = (ax.get_position().height * fig.get_figheight() * 72) / ts[0].get_fontsize()          # the cell's height in em
        steps = [((a.get_position()[1] - b.get_position()[1]) * em, a.get_text()) for a, b in zip(ts, ts[1:])]
        under = [d for d, t in steps if "_" in t and d < 2.0]
        plain = [d for d, t in steps if "_" not in t and d < 2.0]
        assert len(under) >= 3 and len(plain) >= 5, layout                    # the block has C_l ... lines, field inputs, and plain lines between them
        assert np.allclose(under, V.LINE + V.UNDER, atol=1e-6) and np.allclose(plain, V.LINE, atol=1e-6), layout
        matplotlib.pyplot.close(fig)


def test_csv_export(tmp_path):
    """Every number of a result as CSV files: one row of coefficients, 240 panels, 121 stations, every grid point."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import view_airfoil as V
    loop, name = V.section("2412")
    r, sw, out, meta = V.compute(loop, 4.0, 3e6, 0.05)
    paths = V.export_csv(tmp_path, name, 4.0, 3e6, r, sw, out, meta)
    rows = {p.rsplit("_", 1)[1]: sum(1 for _ in open(p)) - 1 for p in paths if p.endswith(".csv")}
    bl = meta["boundary_layer"]
    assert rows == {"coefficients.csv": 1, "surface.csv": 240, "layer.csv": 121, "field.csv": out["x"].size, "wall.csv": len(meta["wall"]["x"]),
                    "section.csv": len(meta["section"]), "rans.csv": len(bl["upper"]["theta"]) + len(bl["lower"]["theta"])} and any(p.endswith("_about.txt") for p in paths)


def test_csv_export_of_the_rans_state_boundary_layer(tmp_path):
    """The RANS state's boundary layer is its own file (upper surface stations, then lower, each from the leading edge), its C_f the wall
    function's, explained in the about text; the surface model's boundary-layer file and the others are as they were."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import view_airfoil as V
    loop, name = V.section("2412")
    r, sw, out, meta = V.compute(loop, 4.0, 3e6, 0.05, "naca")
    paths = V.export_csv(tmp_path, name, 4.0, 3e6, r, sw, out, meta)
    assert sorted(os.path.basename(p).split("Re3e+06_")[1] for p in paths) == sorted(
        ["about.txt", "boundary_layer.csv", "boundary_layer_rans.csv", "coefficients.csv", "field.csv", "section.csv", "surface.csv", "wall.csv"])
    p = [p for p in paths if p.endswith("boundary_layer_rans.csv")][0]
    d = np.genfromtxt(p, delimiter=",", names=True)
    assert d.dtype.names == ("upper_surface", "x_over_c", "Ue_over_U", "dstar_over_c", "theta_over_c", "H", "delta_over_c", "Cf", "separated")
    bl, up = meta["boundary_layer"], d["upper_surface"] == 1
    n = len(bl["upper"]["theta"])
    assert up[:n].all() and not up[n:].any() and np.all(np.diff(d["x_over_c"][up]) > 0) and np.all(np.diff(d["x_over_c"][~up]) > 0)
    assert np.allclose(d["theta_over_c"][up], bl["upper"]["theta"], rtol=1e-7, equal_nan=True) and np.allclose(d["H"][~up], bl["lower"]["H"], rtol=1e-7, equal_nan=True)
    aft = (d["x_over_c"] > 0.1) & (d["x_over_c"] < 0.95)
    assert np.isfinite(d["theta_over_c"][aft]).all() and d["separated"].sum() == 0 and 1.2 < np.median(d["H"][aft]) < 1.8
    w = meta["wall"]                                              # C_f is the wall values' at those nodes
    wx, wu = np.asarray(w["x"]), np.asarray(w["upper"])
    j = [int(np.flatnonzero(wu == bool(u))[np.argmin(np.abs(wx[wu == bool(u)] - x))]) for x, u in zip(d["x_over_c"], d["upper_surface"])]     # the node (x is written to 8 digits)
    assert np.allclose(np.abs(wx[j] - d["x_over_c"]), 0, atol=1e-7) and np.allclose(d["Cf"], np.asarray(w["cf"])[j], rtol=1e-6)
    about = open([p for p in paths if p.endswith("_about.txt")][0]).read()
    assert "_boundary_layer_rans.csv" in about and "wall-normal rays" in about and "x/c < 0.1" in about and "the XFOIL state" in about
    flat = " ".join(about.split())                                  # (the lines are wrapped)
    for said in ("the ray crosses the section", "no edge is found", "fewer than 7 samples", "beyond 0.8 of the ray", "a thickness is <= 0", "can be outliers",
                 "every range warning in the warnings column", "field_Cm_RANS", "1 if the incidence and the Reynolds number lie inside its training range"):
        assert said in flat, said                                   # why a station is nan, that one can be an outlier, and the new columns


def test_csv_export_coefficients_carry_the_rans_state_moment_and_the_range_flags(cases, tmp_path):
    """_coefficients.csv keeps its 19 columns in place (warnings the 19th) and appends the RANS state's C_m, which the figure shows and no
    other file holds, and 1/0 for whether each state's training range holds the incidence and Reynolds number."""
    import csv
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import view_airfoil as V
    old = ["airfoil", "alpha_given_deg", "alpha_used_deg", "Re", "Cl", "Cl_circulation", "Cd", "Cm_quarter_chord", "stagnation_arc_fraction",
           "ue_TE_over_U", "theta_TE_over_c", "dstar_TE_over_c", "field_Cl_gamma", "field_s_sep", "field_Cl_RANS", "field_Cd_RANS",
           "field_Cd_RANS_pressure", "field_Cd_RANS_friction", "warnings"]
    for key, flags in (("in", ("1", "1")), ("rans_out", ("1", "0")), ("both_out", ("0", "0"))):
        name, a, Re, r, sw, out, meta = cases[key]
        paths = V.export_csv(tmp_path / key, name, a, Re, r, sw, out, meta)
        rows = list(csv.DictReader(open([p for p in paths if p.endswith("_coefficients.csv")][0])))
        assert len(rows) == 1 and list(rows[0]) == old + ["field_Cm_RANS", "XFOIL_state_in_training_range", "RANS_state_in_training_range"], key
        row = rows[0]
        assert float(row["field_Cm_RANS"]) == pytest.approx(meta["wall"]["cm"], rel=1e-7) and float(row["Cm_quarter_chord"]) == pytest.approx(float(r.cm[0]), rel=1e-6)
        assert (row["XFOIL_state_in_training_range"], row["RANS_state_in_training_range"]) == flags and (key == "in") == (row["warnings"] == "")
    meta = dict(cases["in"][6], wall=None)                                     # no wall values (the field route without them): the moment is nan, the row still written
    name, a, Re, r, sw, out, _ = cases["in"]
    row = next(csv.DictReader(open([p for p in V.export_csv(tmp_path / "nowall", name, a, Re, r, sw, out, meta) if p.endswith("_coefficients.csv")][0])))
    assert row["field_Cm_RANS"] == "nan" and row["field_Cl_RANS"] == "nan"


def test_application_helpers():
    """The application's database list parser and the field panels' fill of the section interior (no window needed)."""
    import vortfoil_app as A
    html = ('<a href="coord/2032c.dat">2032c.dat</a> \\ Dillner 20-32-C low Reynolds number airfoil \\ <br>'
            '<a href="coord/clarky.dat">clarky.dat</a> \\ Clark Y \\ <br><a href="other.html">x</a>')
    assert A.uiuc_index(html) == [("2032c", "Dillner 20-32-C low Reynolds number airfoil"), ("clarky", "Clark Y")]
    import predict_field as PF
    f = np.arange(30.0).reshape(5, 6); g = f.copy(); g[1:4, 2:4] = np.nan
    h = PF._fill_inside(g)
    assert np.isfinite(h).all() and np.array_equal(h[~np.isnan(g)], f[~np.isnan(g)])


def test_vortfoil_command_runs_the_viewer(tmp_path):
    """The installed command's routing: `vortfoil view ...` draws one case without a window."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from vortfoil import cli
    out = tmp_path / "view.png"
    cli.main(["view", "--airfoil", "2412", "--alpha", "4", "--Re", "3e6", "--h", "0.05", "--no-show", "--save", str(out)])
    assert out.exists() and out.stat().st_size > 20000


def test_viewer_writes_one_figure(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")                      # the viewer's one extra dependency
    matplotlib.use("Agg")
    import view_airfoil
    out = tmp_path / "view.png"
    view_airfoil.main(["--airfoil", "2412", "--alpha", "4", "--Re", "3e6", "--h", "0.05", "--no-show", "--save", str(out)])
    assert out.exists() and out.stat().st_size > 20000
