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
        out, meta = PF.predict(section, 4.0, 3e6, window=(-0.5, 1.5, -0.5, 0.5), h=0.05)
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
        PF.predict(section, 4.0, 1e5, h=0.1)
    with pytest.warns(PF.FieldRangeWarning, match="incidence"):
        PF.predict(section, 20.0, 3e6, h=0.1)


def test_open_trailing_edge_is_closed(PF):
    x, yu, yl = vortfoil.geometry.surface(NACA2412_CST, 121)
    loop = np.r_[np.c_[x[::-1], yu[::-1] + 0.002 * x[::-1]], np.c_[x[1:], yl[1:] - 0.002 * x[1:]]]   # 0.4% chord gap
    g, shift = PF.close_trailing_edge(loop)
    assert np.allclose(g[0], g[-1]) and abs(shift - 0.002) < 1e-9
