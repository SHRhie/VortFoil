# -*- coding: utf-8 -*-
"""The 3D view of a wing (code/wing_flow.py): the page's data without a flow field and with the inviscid one on a coarse
wing, the composite field's own checks, the relaxed free wake, the viscous field's guards, the page's data blocks; and, with
VORTFOIL_SLOW=1, the research full-scale wing's viscous field with the rolled-up wake in both section states
(VORTFOIL_RESEARCH_WING: the research's artifacts/wing_nll_20261010 folder, by default next to this checkout)."""
import base64
import gzip
import json
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "code", "field"))

import wing_flow as WF  # noqa: E402
import wing_nll as WG  # noqa: E402

ART = os.environ.get("VORTFOIL_RESEARCH_WING", os.path.join(ROOT, "..", "airfoil_operator", "VortFoil", "artifacts", "wing_nll_20261010"))


@pytest.fixture(scope="module")
def res():
    """the model wing on 12 strips, XFOIL state, 20 m/s, 4 deg (a few seconds)"""
    return WG.compute(dict(WG.PRESETS["Model wing (span 2 m)"], n=12), 20.0, 4.0)


def case_of(res):
    s = res["strips"]
    return dict(alpha=res["alpha"], y=s["y"], chord=s["chord"], gamma=s["gamma"], cl=s["cl"], alpha_eff=s["alpha_eff_deg"])


def test_page_record_and_free_wake(res):
    seen = []
    d = WF.view_data(res, "none", free_wake=True, progress=lambda f, t: seen.append(f))
    rec, n, c0 = d["wing"], res["wing"]["n"], res["coef"]
    json.dumps(d, allow_nan=False)                                       # what write_page packs
    assert d["field"] is None and rec["focus"] == res["alpha"] and len(rec["cases"]) == 1
    c = rec["cases"][0]
    assert all(c[k] == c0[k] for k in ("CL", "CDi", "CD0", "CD", "CM", "LD", "e")) and c["alpha"] == res["alpha"]
    assert rec["wing"]["section"] == "NACA 2412" and [s["section"] for s in rec["wing"]["stations"]] == ["NACA 2412"] * 2
    assert rec["wing"]["name"] == res["wing"]["name"] and rec["wing"]["state"] == "xfoil" and rec["warnings"] == res["warnings"]
    lead = WF.view_data(dict(res, sst_range=dict(message="SST state outside ...")), "none")["wing"]["warnings"]
    assert lead == ["SST state outside ..."] + res["warnings"]           # wing_nll's sst_range message (B4) leads the page's list
    for side in ("upper", "lower"):
        assert all(np.shape(c["surface"][side][k]) == (n, WG.NX) for k in ("x", "y", "z", "cp"))
    assert len(c["trailing"]) == n + 1 and abs(sum(t["strength"] for t in c["trailing"])) < 1e-9
    al = [p["alpha"] for p in rec["polar"]]
    focus = rec["polar"][al.index(res["alpha"])]
    assert al == sorted(al) and al.count(res["alpha"]) == 1 and focus["CL"] == c0["CL"] and len(focus["y"]) == len(focus["cl"]) == n
    # the relaxed wake: from the trailing edges downstream, carried down by the downwash; it moves C_L by < 1%
    Wk = np.array(c["wake"])
    _, T, _ = WF.nodes(WG.Wing(stations=res["defn"]["stations"], n=n), res["alpha"])
    assert Wk.shape == (n + 1, WF.K + 1, 3) and np.allclose(Wk[:, 0], T, atol=1e-5) and np.all(np.diff(Wk[..., 0], axis=1) > 0)
    assert np.mean(Wk[:, -1, 2]) < np.mean(Wk[:, 0, 2])
    cl_free = float(re.search(r"C_L ([-\d.]+) \(straight", rec["wake_model"]).group(1))
    assert abs(cl_free / c0["CL"] - 1) < 0.01
    assert seen and np.all(np.diff(seen) >= 0) and 0 <= seen[0] and seen[-1] <= 1


def test_composite_field(res):
    """the research checks of the composite field: the panel method, tangency on the strips, the far field"""
    sec = WF.Section("2412")
    us, vs, uv, vv = WF.panel_vel(sec.cpts, sec.X0, sec.X1)
    for a in (0.0, np.radians(4)):
        s = np.cos(a) * sec.sol[0] + np.sin(a) * sec.sol[1]
        with np.errstate(all="ignore"):                                  # Accelerate BLAS raises spurious FP flags
            u, v = np.cos(a) + us @ s[:-1] + s[-1] * uv.sum(1), np.sin(a) + vs @ s[:-1] + s[-1] * vv.sum(1)
        assert np.max(np.abs(u * sec.nv[:, 0] + v * sec.nv[:, 1])) < 1e-8
    assert 0.22 < sec.cl_xy[0] < 0.32 and 6.3 < sec.cl_xy[1] < 7.3                     # NACA 2412: C_l(0), slope per rad
    n = res["wing"]["n"]
    fl = WF.Flow(WG.Wing(stations=res["defn"]["stations"], n=n), res["V"], case_of(res), *WF.strip_sections(res))
    for j in range(n):                       # tangency at every strip's control points (per unit freestream: |V| -> 0 at stagnation)
        ca, sa, c = np.cos(fl.ag[j]), np.sin(fl.ag[j]), fl.c[j]
        xi, ze = sec.cpts[:, 0] - 0.25, sec.cpts[:, 1]
        V = fl.vel(np.c_[c * (xi * ca + ze * sa) + fl.Qs[j, 0], np.full(len(xi), fl.y[j]), c * (-xi * sa + ze * ca) + fl.Qs[j, 2]])
        vn = (V[:, 0] * ca - V[:, 2] * sa) * sec.nv[:, 0] + (V[:, 0] * sa + V[:, 2] * ca) * sec.nv[:, 1]
        assert np.max(np.abs(vn[1:-1])) < 1e-6                               # trailing-edge panels carry Kutta
    far = fl.vel(np.array([[-6.0, 0.3, 0.5], [0.3, 4.0, 0.0]]))           # far upstream, far outboard: the freestream
    assert np.all(np.abs(far - WF.EX) < 0.01)


def blocks(path):
    """the page's two data blocks, unpacked as the browser does (an empty block: None)"""
    page = open(path, encoding="utf-8").read()
    assert "__WING_DATA__" not in page and "__FIELD_DATA__" not in page and not os.path.exists(path + ".part")
    b = {k: re.search(rf'id="{k}-data">([^<]*)<', page).group(1).strip() for k in ("wing", "field")}
    return {k: json.loads(gzip.decompress(base64.b64decode(v))) if v else None for k, v in b.items()}


def test_inviscid_streamlines(res, monkeypatch, tmp_path):
    monkeypatch.setattr(WF, "STEPS", 150)                                # short lines: the data, not the picture
    with pytest.raises(RuntimeError, match="cancelled"):
        WF.view_data(res, "inviscid", cancel=lambda: True)
    d = WF.view_data(res, "inviscid")
    f = d["field"]
    lines = f["cases"][0]["lines"]
    assert f["kind"] == "inviscid" and f["decoder_warnings"] == [] and f["cases"][0]["alpha"] == res["alpha"] and len(lines) == 100
    assert all(len(l["x"]) == len(l["y"]) == len(l["z"]) == len(l["cp"]) >= 2 and np.isfinite(l["cp"]).all() for l in lines)
    q = np.concatenate([l["cp"] for l in lines])
    assert -3 < q.min() < -0.2 and q.max() < 1.0                             # suction over the wing, C_p <= 1
    assert blocks(WF.write_page(d, str(tmp_path / "inviscid.html")))["field"] == json.loads(json.dumps(f))


def test_viscous_guards(res, monkeypatch):
    """the viscous field is refused, before any work, below 3/4 of the decoder's Re range (the model wing at 20 m/s: Re 2e5-4e5,
    where its streamlines end inside the wing); decoder messages the SST state's result lists already are not repeated"""
    with pytest.raises(ValueError, match=r"down to Re 2e5 .* Choose Inviscid streamlines, or a speed of at least 1\d\d m/s"):
        WF.view_data(res, "viscous", progress=lambda f, t: pytest.fail("work started"))
    monkeypatch.setattr(WF, "flow_field", lambda *a: dict(kind="inviscid", decoder_warnings=["a", "b"], cases=[]))
    assert WF.view_data(dict(res, warnings=["SST decoder: a"]), "inviscid")["field"]["decoder_warnings"] == ["b"]


def test_write_page_round_trip(res, tmp_path):
    d = WF.view_data(res, "none")
    got = blocks(WF.write_page(d, str(tmp_path / "pages" / "wing.html")))
    assert got["field"] is None                                          # no flow field: an empty block
    assert got["wing"] == json.loads(json.dumps(d["wing"])) and os.path.expanduser("~") not in json.dumps(got["wing"])


@pytest.mark.skipif(not os.environ.get("VORTFOIL_SLOW"), reason="research regression, minutes: set VORTFOIL_SLOW=1")
@pytest.mark.skipif(not os.path.isdir(ART), reason="the research outputs (VORTFOIL_RESEARCH_WING) are not here")
@pytest.mark.parametrize("folder, state", [("full_scale", "xfoil"), ("full_scale_sst", "sst")])
def test_research_full_scale_viscous_free_wake(folder, state):
    """The research full-scale wing (NACA 2412, 45 m/s, 6 deg, relaxed wake, viscous field) in the section state it used. Its
    own stored state through flow_field gives its outputs exactly; the release wing (wing_nll.compute, whose C_L differs by
    up to 5e-4) through view_data stays within 2 mm and 1 byte of plane colour (the design prototype's acceptance: >= 99%
    equal bytes)."""
    import predict_field as PF
    R = json.load(open(os.path.join(ART, folder, "wing_viz3d.json")))
    ref = next(c for c in json.load(open(os.path.join(ART, folder, "wing_viscous.json")))["cases"] if c["alpha"] == 6.0)
    rc = next(c for c in R["cases"] if c["alpha"] == 6.0)
    case = dict(alpha=6.0, **{k: rc[k] for k in ("y", "chord", "gamma", "cl", "alpha_eff", "wake")})
    one = np.zeros(40, int)
    f = WF.flow_field("viscous", WG.Wing(stations=R["wing"]["stations"], n=40), 45.0, case, [WF.Section("2412")], one,
                      dict(coords=[PF.naca_coords("2412")], index=one, family=["naca"]), PF.DECODER)["cases"][0]
    assert f["lines"] == ref["lines"] and f["planes"] == ref["planes"] and f["sheet"] == ref["sheet"]

    res = WG.compute(WG.PRESETS["Full-scale wing (span 10 m)"], 45.0, 6.0, state)
    f = WF.view_data(res, "viscous", free_wake=True)["field"]["cases"][0]
    assert [len(l["x"]) for l in f["lines"]] == [len(l["x"]) for l in ref["lines"]]
    assert max(np.max(np.abs(np.array(a[k]) - np.array(b[k]))) for a, b in zip(f["lines"], ref["lines"]) for k in "xyz") < 2e-3
    for a, b in zip(f["planes"], ref["planes"]):
        qa, qb = (np.frombuffer(base64.b64decode(p["q"]), np.uint8).astype(int) for p in (a, b))
        assert np.mean(qa == qb) > 0.99 and np.max(np.abs(qa - qb)) <= 1
    assert np.max(np.abs(np.array(f["sheet"]["q"]) - np.array(ref["sheet"]["q"]))) <= 2e-3
