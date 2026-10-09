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
    """The three layouts of the result figure build at their design sizes."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import view_airfoil as V
    loop, name = V.section("2412")
    res = V.compute(loop, 4.0, 3e6, 0.05)
    for layout, size in V.DESIGNS.items():
        fig = V.figure(name, 4.0, 3e6, *res, layout=layout)
        assert tuple(fig.get_size_inches()) == size and len(fig.axes) == 11      # 4 fields, 4 colour bars, C_p, BL, text
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
    assert rows == {"coefficients.csv": 1, "surface.csv": 240, "layer.csv": 121, "field.csv": out["x"].size,
                    "wall.csv": len(meta["wall"]["x"]), "section.csv": len(meta["section"])} and any(p.endswith("_about.txt") for p in paths)


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
