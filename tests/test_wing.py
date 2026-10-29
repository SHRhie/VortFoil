# -*- coding: utf-8 -*-
"""The application's 3D wing (code/wing_nll.py): the lifting line's checks, the XFOIL and SST section states, the wing
definitions. The research regression (both states, minutes on a cold cache) runs with VORTFOIL_SLOW=1."""
import json
import os
import sys

import copy
import warnings

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "code", "field"))

import wing_nll as WG  # noqa: E402

FULL, KINKED = "Full-scale wing (span 10 m)", "Kinked wing (4415 to 2412, dihedral)"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(WG, "CACHE", str(tmp_path / "wing_tables"))
    monkeypatch.setattr(WG, "_MEM", {})
    return tmp_path / "wing_tables"


def test_lifting_line_checks():
    WG.check()


def test_solve_prefers_a_root_inside_the_table():
    """Outside its table the polar continues C_l linearly (research, 2026-10-15); solve() takes a root with every strip inside the
    table, trying the clamped form first. On the default wing the continued polar alone settles the tip strips near 72 deg."""
    res = WG.compute(WG.PRESETS[FULL], 45.0, 6.0)
    t, d = res["tables"], res["defn"]
    pol = WG.table_polar(dict(alpha=np.radians(t["alpha_deg"]), cl=t["cl"], cd=t["cd"], cm=t["cm"]))
    w = WG.Wing(stations=d["stations"], n=d["n"])
    edge = np.full(w.n, pol.hi)
    assert np.allclose(pol(edge + 0.1)[0], pol(edge)[0] + 0.1 * pol(edge)[1]) and np.allclose(pol.clamped(edge + 0.1)[0], pol(edge)[0])
    bare = lambda ai: pol(ai)                       # the continued form without its table range: the first converged root
    s0, s = WG.solve(w, 45.0, 6.0, bare), WG.solve(w, 45.0, 6.0, pol)
    assert max(s0["alpha_eff"]) > 60 and s0["residual"] < 1e-6                         # the spurious tip root
    assert -10 < min(s["alpha_eff"]) and max(s["alpha_eff"]) < 18 and abs(s["CL"] - 0.6265) < 3e-3


def test_naca_23012_wing_rans_state():
    """The research's failure case (2026-10-15): with the deployed decoder a NACA 23012 tip strip settled at -62 deg on a table
    over -8..16 deg. Tables over the decoder's training incidences and the in-table root give the research's re-solved C_L
    (0.1958, 0.5421, 0.8912 at 2, 6, 10 deg; free wake) with every strip inside the table."""
    d = copy.deepcopy(WG.PRESETS[FULL])
    for st in d["stations"]:
        st["section"] = "23012"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = WG.compute(d, 45.0, 6.0, "sst")
    al = r["polar"]["alpha"]
    cl = [r["polar"]["CL"][int(np.flatnonzero(al == a)[0])] for a in (2.0, 6.0, 10.0)]
    assert np.allclose(cl, (0.1958, 0.5421, 0.8912), atol=4e-3), cl
    ae = np.asarray(r["strips"]["alpha_eff_deg"])
    assert -5 < ae.min() and ae.max() < 15 and r["coef"]["in_range"]


def test_full_scale_wing_xfoil_state():
    """The research full-scale wing at 45 m/s and 6 deg (C_L 0.6265 there, free wake; straight legs here), and the result's
    layout that the figure and the application read."""
    res = WG.compute(WG.PRESETS[FULL], 45.0, 6.0)
    c, n = res["coef"], 40
    assert abs(c["CL"] - 0.6265) < 3e-3 and res["warnings"] == [] and res["newton"]["residual"] < 1e-6
    assert abs(c["CD"] - c["CDi"] - c["CD0"]) < 1e-15 and 0.95 < c["e"] < 1.0 and -0.06 < c["CM"] < -0.02
    w = res["wing"]
    assert w["b"] == 10.0 and abs(w["S"] - 11.25) < 0.01 and abs(w["MAC"] - 7 / 6) < 2e-3 and w["taper"] == 0.5
    s = res["strips"]
    assert all(np.shape(v) == (n,) for v in s.values()) and np.allclose(s["alpha_ind_deg"], s["alpha_geo_deg"] - s["alpha_eff_deg"])
    assert abs(np.sum(s["cl_c_over_cbar"] * np.diff(res["trailing"]["y"])) / w["b"] / c["CL"] - 1) < 0.01   # loading integral
    assert res["tables"]["cl"].shape == (n, len(WG.TABLE_X)) and res["polar"]["CL"].shape == res["polar"]["alpha"].shape
    for side in ("upper", "lower"):
        S = res["surface"][side]
        assert set(S) == {"X", "Y", "Z", "C_p", "theta/c", "delta*/c", "H"} and all(v.shape == (n, WG.NX) for v in S.values())
    assert res["surface"]["upper"]["C_p"].min() < -0.5 < 0.5 < res["surface"]["lower"]["C_p"].max()
    assert res["strip_detail"]["cp"].shape == (n, 240) and res["strip_detail"]["theta"].shape == (n, 2, 121)
    L = res["loops"]
    assert L["label"] == ["NACA 2412"] and L["family"] == ["naca"] and np.all(L["index"] == 0)
    t = res["trailing"]
    assert all(v.shape == (n + 1,) for v in t.values()) and abs(t["strength"].sum()) < 1e-9 * np.abs(t["strength"]).max()


@pytest.mark.parametrize("change, message", [
    (dict(stations=[dict(y=0, chord=1, section="2412")]), "at least two"),
    (dict(stations=[dict(y=0.5, chord=1, section="2412"), dict(y=1, chord=1, section="2412")]), "station 1 y"),
    (dict(stations=[dict(y=0, chord=1, section="2412"), dict(y=0, chord=1, section="2412")]), "station 2 y"),
    (dict(stations=[dict(y=0, chord=1, section="2412"), dict(y=1, chord=0, section="2412")]), "station 2 chord"),
    (dict(stations=[dict(y=0, chord=1, section="2412"), dict(y=1, chord="x", section="2412")]), "station 2 chord"),
    (dict(stations=[dict(y=0, chord=1, section="2412"), dict(y=1, chord=float("nan"), section="2412")]), "station 2 chord"),
    (dict(stations=[dict(y=0, chord=1, section="2412"), dict(y=1, section="2412")]), "missing 'chord'"),
    (dict(stations=[dict(y=0, chord=1, section="2412"), dict(y=1, chord=1, section=" ")]), "station 2 section"),
    (dict(n=4), "n:"), (dict(n=40.5), "n:"), (dict(V=0), "V:"), (dict(alpha=31), "alpha:"),
])
def test_validate_names_the_bad_field(change, message):
    with pytest.raises(ValueError, match=message):
        WG.validate(dict(WG.PRESETS[FULL], **change))


def test_definitions(tmp_path):
    """Every preset is valid; JSON round trip; a coordinate-file section is relative to its definition; sections the
    application resolves ('uiuc:<name>', 'current') come in through `sections`; speed and incidence are checked."""
    for d in WG.PRESETS.values():
        assert WG.validate(d)["n"] == 40
    assert WG.validate(dict(WG.PRESETS[FULL], stations=[dict(y=0, chord=1, section="naca 4412"), dict(y=3, chord=1, section="NACA4412")]))["stations"][1]["section"] == "4412"
    import predict_field as PF
    np.savetxt(tmp_path / "sec.dat", PF.naca_coords("4412"), header="NACA 4412 as a file", comments="")
    d = dict(WG.PRESETS[FULL], stations=[dict(s, section="sec.dat") for s in WG.PRESETS[FULL]["stations"]])
    WG.save_wing(d, tmp_path / "w.json")
    assert json.load(open(tmp_path / "w.json"))["stations"][0]["section"] == "sec.dat"
    w = WG.load_wing(tmp_path / "w.json")
    assert w["stations"][0]["section"] == str(tmp_path / "sec.dat") and w["V"] == 45.0
    (tmp_path / "bom.json").write_bytes(b"\xef\xbb\xbf" + (tmp_path / "w.json").read_bytes().replace(b"\n", b"\r\n"))   # Notepad's 'UTF-8 with BOM'
    assert WG.load_wing(tmp_path / "bom.json") == w
    (tmp_path / "u16.json").write_text((tmp_path / "w.json").read_text(), encoding="utf-16")
    with pytest.raises(ValueError, match="not UTF-8 text"):
        WG.load_wing(tmp_path / "u16.json")
    r = WG.compute(w, 45.0, 4.0)
    assert r["loops"]["label"] == ["sec"] and r["loops"]["family"] == [None] and r["sst_range"] is None
    cur = dict(w, stations=[dict(s, section="current") for s in w["stations"]])
    r2 = WG.compute(cur, 45.0, 4.0, sections={"current": PF.naca_coords("4412")}, labels={"current": "my foil"})
    assert r2["loops"]["label"] == ["my foil"] and {s["section"] for s in r2["defn"]["stations"]} == {"my foil"} and abs(r2["coef"]["CL"] - r["coef"]["CL"]) < 1e-12
    clark = WG.PRESETS["Clark Y wing"]
    with pytest.raises(ValueError, match="uiuc:clarky"):
        WG.compute(clark, 30.0, 4.0)
    r = WG.compute(clark, 30.0, 4.0, sections={"uiuc:clarky": PF.naca_coords("4412")})
    assert r["loops"]["label"] == ["clarky"] and abs(r["coef"]["CL"] - WG.compute(w, 30.0, 4.0)["coef"]["CL"]) < 1e-12
    with pytest.raises(ValueError, match="V:"):
        WG.compute(w, -1.0, 4.0)
    with pytest.raises(ValueError, match="state"):
        WG.compute(w, 45.0, 4.0, state="rans")


def test_wing_warnings():
    """Aspect ratio, sweep and non-convergence far past stall are reported in English."""
    r = WG.compute(WG.PRESETS["Swept 30 deg"], 30.0, 4.0)
    assert any("sweep is not modelled by the lifting line" in m for m in r["warnings"])
    assert WG.compute(WG.PRESETS["Rectangular, AR 4"], 30.0, 4.0)["warnings"] == []            # AR 4: the limit itself
    assert WG.compute(dict(WG.PRESETS["Rectangular, AR 4"], n=37), 30.0, 4.0)["warnings"] == []  # b^2/S = 3.999999999999999 there
    short = dict(WG.PRESETS["Rectangular, AR 4"], stations=[dict(s, y=s["y"] * 0.5) for s in WG.PRESETS["Rectangular, AR 4"]["stations"]])
    assert any(m.startswith("aspect ratio 2.00 < 4") for m in WG.compute(short, 30.0, 4.0)["warnings"])
    r = WG.compute(WG.PRESETS[FULL], 45.0, 25.0)
    assert any("beyond their section's stall angle" in m for m in r["warnings"]) and len(r["warnings"]) <= 8
    assert any(m.startswith("the lifting line did not converge at 25 deg") for m in r["warnings"])


def test_sst_state(cache, monkeypatch):
    """RANS state on a small kinked wing (coarse incidence grid): drag split, wall values on the surface, grouped decoder
    warnings, the disk cache, progress and cancel."""
    monkeypatch.setattr(WG, "TABLE_S", np.arange(-8.0, 16.001, 2.0))
    d = dict(WG.PRESETS[KINKED], n=8)
    prog = []
    r = WG.compute(d, 60.0, 6.0, "sst", progress=lambda f, s: prog.append(f))
    c = r["coef"]
    assert 0.65 < c["CL"] < 0.8 and abs(c["CD0_pressure"] + c["CD0_friction"] - c["CD0"]) < 1e-12 and -0.12 < c["CM"] < -0.04
    assert r["warnings"] == [] and r["sst_range"] == dict(n=8, below=0, above=0, lo=2e6, hi=6e6, message=None)
    assert r["loops"]["family"] == ["naca"] * 4 and list(r["loops"]["index"]) == [0, 1, 2, 3, 3, 2, 1, 0]   # 4415/2412 blends are NACA
    assert set(r["surface"]["upper"]) == {"X", "Y", "Z", "C_p", "C_f"} and np.isfinite(r["surface"]["lower"]["C_f"]).all()
    assert len(r["strip_detail"]["cp"]) == 8 and r["tables"]["cd_friction"].shape == (8, len(WG.TABLE_S))
    assert prog[-1] == 1.0 and all(b >= a for a, b in zip(prog, prog[1:])) and r["timing"]["built"] == 4
    assert len(os.listdir(cache)) == 4
    WG._MEM.clear()
    r2 = WG.compute(d, 60.0, 6.0, "sst")
    assert r2["timing"]["built"] == 0 and r2["coef"]["CL"] == c["CL"]
    import predict_field as PF
    monkeypatch.setattr(PF, "FIELD_RE", (7e6, 1e7))                     # every strip below the range: said first, and still computed
    r3 = WG.compute(d, 60.0, 6.0, "sst")
    assert r3["sst_range"]["below"] == 8 and r3["sst_range"]["message"].startswith(
        "RANS state outside the field decoder's Reynolds range (7e6-1e7) on 8 of 8 strips: the RANS-state numbers are not meaningful here")
    assert r3["coef"]["CL"] == c["CL"] and "RANS decoder: 8 of 8 strips at Re below its training range 7e6-1e7" in r3["warnings"]
    with pytest.raises(RuntimeError, match="cancelled"):
        WG.compute(d, 60.0, 6.0, "sst", cancel=lambda: True)


def test_polar_follows_the_state_training_range(cache, monkeypatch):
    """The wing polar spans the state's training incidences (XFOIL -10..18 deg, SST -5..15 deg, widened to the requested
    incidence). The coefficients and every polar point carry in_range: every strip's incidence and Re inside the state's
    training ranges and no strip past its section's stall."""
    d = dict(WG.PRESETS[FULL], n=12)
    r = WG.compute(d, 45.0, 6.0)
    p = r["polar"]
    assert list(p["alpha"]) == list(np.arange(-10.0, 18.5, 1.0)) and r["coef"]["in_range"] is True
    assert p["in_range"].dtype == bool and p["in_range"].shape == p["alpha"].shape and p["in_range"][p["alpha"] == 6.0][0]
    hi = WG.compute(d, 45.0, 22.0)
    assert hi["polar"]["alpha"][-1] == 22.0 and not hi["coef"]["in_range"] and not hi["polar"]["in_range"][-1]
    orig = WG.table_polar

    def early_stall(tab, deg=12):                       # every section stalls at 3 deg: the stall rule alone decides
        f = orig(tab, deg)
        f.alpha_stall = np.full_like(f.alpha_stall, np.radians(3.0))
        return f
    monkeypatch.setattr(WG, "table_polar", early_stall)
    st = WG.compute(d, 45.0, 6.0)
    a, ok = st["polar"]["alpha"], st["polar"]["in_range"]
    assert ok[(a >= -8) & (a <= 1)].all() and not ok[a >= 5].any() and not st["coef"]["in_range"]
    monkeypatch.setattr(WG, "table_polar", orig)
    monkeypatch.setattr(WG, "TABLE_S", np.arange(-8.0, 16.001, 2.0))
    k = dict(WG.PRESETS[KINKED], n=8)
    s = WG.compute(k, 60.0, 6.0, "sst")
    assert list(s["polar"]["alpha"]) == list(np.arange(-5.0, 15.5, 1.0)) and s["coef"]["in_range"] and s["polar"]["in_range"][5:].all()
    import predict_field as PF
    monkeypatch.setattr(PF, "FIELD_RE", (7e6, 1e7))                     # every strip below the decoder's Re range
    s = WG.compute(k, 60.0, 6.0, "sst")
    assert not s["coef"]["in_range"] and not s["polar"]["in_range"].any()


def test_sst_warnings_are_grouped():
    import predict_field as PF
    lo = "Reynolds number 1e+06 lies outside the decoder's training range 2e+06 to 6e+06"
    out = WG._sst_warnings([[lo], [lo, PF.FAMILY_NOTE_AIRFRANS], [], ["incidence 15.50 deg lies outside ..."]], [1e6, 1e6, 3e6, 7e6], 2e6, 6e6)
    assert out == ["RANS decoder: 2 of 4 strips at Re below its training range 2e6-6e6",
                   "RANS decoder: 1 of 4 strips are not NACA 4/5-digit sections (the decoder's training family): read their values as indicative",
                   "RANS decoder: 1 of 4 strips at incidences outside its training range -5 to 15 deg"]


def test_sst_moment_against_the_xfoil_state():
    """The RANS state's c_m from the decoder's wall pressure and friction is close to the surface model's for NACA 2412."""
    import warnings
    import predict_field as PF
    loop, al = PF.naca_coords("2412"), np.arange(0.0, 8.001, 2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm_x = np.asarray(WG._model().predict(coords=loop, alpha=al, Re=3e6).cm).ravel()
        cm_s = np.array([PF.surface(loop, a, 3e6, family="naca")["cm"] for a in al])
    assert np.all(np.abs(cm_s - cm_x) < 0.02), (cm_s, cm_x)


def test_input_ranges():
    """The 3D Flow box's ranges, without a model call: the speeds that keep every strip inside each state's training Re (compute()'s own
    strips: their chords at the control points times the scale, NU_AIR 1.5e-5, so they follow the strip count), the states' strip
    incidences, a typed speed judged per state, the RANS decoder's family (NACA 4/5-digit; 'current' is the Airfoil box's section) and
    the sentences the box shows."""
    import re
    import predict_field as PF
    strips = lambda d: WG.Wing(stations=WG.validate(d)["stations"], n=WG.validate(d)["n"]).c          # compute()'s strip chords
    r = WG.input_ranges(WG.PRESETS[FULL])                                           # stations 1.5 to 0.75 m; its 40 strips 1.471 to 0.751 m
    x, s, c = r["xfoil"], r["sst"], strips(WG.PRESETS[FULL])
    assert r["chord"] == (c.min(), c.max()) and 0.75 < c.min() < 0.751 and 1.47 < c.max() < 1.471 and r["naca"] and x["possible"] and s["possible"]
    assert np.allclose(x["V"], (5e4 * 1.5e-5 / c.min(), 1e7 * 1.5e-5 / c.max()), rtol=1e-11, atol=0)
    assert np.allclose(s["V"], (2e6 * 1.5e-5 / c.min(), 6e6 * 1.5e-5 / c.max()), rtol=1e-11, atol=0)
    assert (x["label"], x["alpha"], x["Re"], s["label"], s["alpha"], s["Re"]) == (
        "XFOIL state", (-10.0, 18.0), (5e4, 1e7), "RANS state", (-5.0, 15.0), (2e6, 6e6))
    assert x["Re_at_V"] is x["inside"] is s["Re_at_V"] is s["inside"] is r["sentences"]["at_V"] is r["sentences"]["family"] is None
    assert r["text"] == ("Speeds that keep every strip inside the training Re: XFOIL state 1-102 m/s, RANS state 40-61.2 m/s. "
                         "Strip incidence: XFOIL -10 to 18 deg, RANS -5 to 15 deg.")
    at = lambda V, d=WG.PRESETS[FULL], **kw: WG.input_ranges(d, V=V, **kw)
    r = at(45)
    assert r["xfoil"]["inside"] and r["sst"]["inside"] and np.allclose(r["sst"]["Re_at_V"], (45 * c.min() / 1.5e-5, 45 * c.max() / 1.5e-5))
    assert r["sentences"]["at_V"] == "45 m/s: inside both." and r["text"].endswith("RANS -5 to 15 deg. 45 m/s: inside both.")
    for V, said in ((30, "30 m/s: XFOIL state only; RANS state below its Re range at the outboard strips."),
                    (61, "61 m/s: inside both."),                                   # above the stations' 60 m/s: no strip reaches the root chord
                    (61.3, "61.3 m/s: XFOIL state only; RANS state above its Re range at the inboard strips."),
                    (70, "70 m/s: XFOIL state only; RANS state above its Re range at the inboard strips."),
                    (10, "10 m/s: XFOIL state only; RANS state below its Re range on every strip."),
                    (0.5, "0.5 m/s: XFOIL state below its Re range on every strip; RANS state below its Re range on every strip.")):
        assert at(V)["sentences"]["at_V"] == said
    for V, state in ((1.0, "xfoil"), (102.0, "xfoil"), (40.0, "sst"), (61.2, "sst"), (x["V"][0], "xfoil"), (s["V"][1], "sst")):
        assert at(V)[state]["inside"], (V, state)                                  # a bound, as typed or as computed, is inside
    assert WG.input_ranges(dict(WG.PRESETS[FULL], n=16))["sentences"]["speeds"].endswith("XFOIL state 0.996-105 m/s, RANS state 39.9-63 m/s.")
    m = at(20, WG.PRESETS["Model wing (span 2 m)"])                                 # stations 0.3 to 0.15 m
    assert m["sentences"]["speeds"] == "Speeds that keep every strip inside the training Re: XFOIL state 5-510 m/s, RANS state 200-306 m/s."
    assert m["sentences"]["at_V"] == "20 m/s: XFOIL state only; RANS state below its Re range on every strip." and m["xfoil"]["inside"]
    h = WG.input_ranges(WG.PRESETS[FULL], scale=0.5)                               # half the size: twice the speeds
    assert h["chord"] == (c.min() / 2, c.max() / 2) and np.allclose(h["xfoil"]["V"], 2 * np.array(x["V"])) and np.allclose(h["sst"]["V"], 2 * np.array(s["V"]))
    assert "XFOIL state 2-204 m/s, RANS state 80-122 m/s." in h["text"]
    st = WG.PRESETS[KINKED]["stations"]                                             # a kinked wing with a UIUC section at its kink
    k = at(60, dict(WG.PRESETS[KINKED], stations=[st[0], dict(st[1], section="uiuc:e387"), st[2]]))
    assert not k["naca"] and WG.input_ranges(WG.PRESETS[KINKED])["naca"] and k["sentences"]["family"] == (
        "RANS state: SST-trained on NACA 4/5-digit sections only; this wing's other sections give indicative values; prefer the XFOIL state.")
    assert k["text"].endswith(k["sentences"]["family"]) and k["sst"]["inside"] and "RANS state 59.9-60.7 m/s." in k["text"]
    near3 = WG.input_ranges(dict(stations=[dict(y=0, chord=1.5, section="2412"), dict(y=5, chord=0.487, section="2412")]))
    assert near3["sst"]["possible"] and "RANS state about 61.6 m/s only." in near3["text"]                    # strip chord ratio just under 3
    cur = dict(WG.PRESETS[FULL], stations=[dict(t, section="current") for t in WG.PRESETS[FULL]["stations"]])
    assert not WG.input_ranges(cur)["naca"] and WG.input_ranges(cur, current_is_naca=True)["naca"]
    assert WG.input_ranges(dict(cur, stations=[dict(t, section="NACA 4412") for t in cur["stations"]]))["naca"]
    imp = at(60, dict(stations=[dict(y=0, chord=1.5, section="2412"), dict(y=5, chord=0.4, section="2412")]))   # stations' chord ratio 3.75
    assert imp["xfoil"]["possible"] and not imp["sst"]["possible"] and imp["sst"]["V"][0] > imp["sst"]["V"][1]
    assert imp["sentences"]["speeds"].endswith("XFOIL state 1.88-102 m/s, RANS state no speed (strip chord ratio 3.63 > its Re ratio 3).")
    assert imp["sentences"]["at_V"] == "60 m/s: XFOIL state only; RANS state below its Re range at the outboard strips."
    for d, said in (((1.0, 1.5, 0.5), "80 m/s: XFOIL state only; RANS state above its Re range at the mid-span strips."),   # the widest chord at y = 2 of 5
                    ((1.5, 1.5, 0.75), "80 m/s: XFOIL state only; RANS state above its Re range at the inboard strips.")):  # a constant-chord inboard panel
        wing = dict(stations=[dict(y=y, chord=ch, section="2412") for y, ch in zip((0, 2, 5), d)])
        assert at(80, wing)["sentences"]["at_V"] == said
    assert np.allclose(WG.input_ranges(WG.PRESETS[FULL], decoder=PF.DECODER_REXT)["sst"]["V"], (1e6 * 1.5e-5 / c.min(), 1e7 * 1.5e-5 / c.max()))
    for c0, c1 in ((1.2, 0.9), (1.3, 0.7), (0.37, 0.13)):                         # a shown bound, typed back in, is inside
        d = dict(stations=[dict(y=0, chord=c0, section="2412"), dict(y=4, chord=c1, section="2412")])
        for state, label in (("xfoil", "XFOIL"), ("sst", "RANS")):
            lo, hi = re.search(label + r" state ([0-9.]+)-([0-9.]+) m/s", WG.input_ranges(d)["text"]).groups()
            assert at(float(lo), d)[state]["inside"] and at(float(hi), d)[state]["inside"], (c0, c1, state, lo, hi)
    for kw, message in ((dict(scale=0), "Scale x"), (dict(scale="x"), "Scale x"), (dict(V=-1), "Speed"), (dict(V="fast"), "Speed"),
                        (dict(scale="1e-320"), "too small or too large")):                                           # no OverflowError
        with pytest.raises(ValueError, match=message):
            WG.input_ranges(WG.PRESETS[FULL], **kw)
    with pytest.raises(ValueError, match="chord: inf to inf m after Scale x, too small or too large"):                          # 1e300 m x 1e10
        WG.input_ranges(dict(stations=[dict(y=0, chord=1e300, section="2412"), dict(y=1, chord=1e300, section="2412")]), scale=1e10)
    with pytest.raises(ValueError, match="at least two"):
        WG.input_ranges(dict(stations=[dict(y=0, chord=1, section="2412")]))


@pytest.mark.parametrize("state", ["xfoil", "sst"])
def test_input_ranges_bound_the_strip_reynolds_warnings(state, cache, monkeypatch):
    """At the speeds input_ranges gives for a state, compute() has no strip Reynolds-number warning in it; 1% below the lower one and 1%
    above the upper one it has one (16 strips, coarse tables: the warnings come from the strips' own states, not the tables)"""
    monkeypatch.setattr(WG, "TABLE_X", np.arange(-10.0, 18.001, 2.0))
    monkeypatch.setattr(WG, "TABLE_S", np.arange(-8.0, 16.001, 2.0))
    d = dict(WG.PRESETS[FULL], n=16)
    lo, hi = WG.input_ranges(d)[state]["V"]
    re_warnings = lambda V: [m for m in WG.compute(d, V, 4.0, state)["warnings"] if "Reynolds numbers lie outside" in m or " at Re " in m]
    assert re_warnings(lo) == re_warnings(hi) == [] and len(re_warnings(0.99 * lo)) == len(re_warnings(1.01 * hi)) == 1


def test_newton_restarts_from_zero_circulation(cache, monkeypatch):
    """The NACA 0009 full-scale wing at 1 deg (XFOIL state): from the usual start Newton drifted to a non-physical root at the tip
    strips (71 deg after 60 steps) and the polar point read 'outside the training range'; it now restarts from zero circulation
    and converges. A lifting line that does not converge is never a result in range."""
    import warnings
    d = dict(WG.PRESETS["Full-scale wing (span 10 m)"]); d["stations"] = [dict(st, section="0009") for st in d["stations"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = WG.compute(d, 45.0, 1.0, "xfoil")
    c, ae = r["coef"], np.asarray(r["strips"]["alpha_eff_deg"])
    assert c["in_range"] and c["strips_past_stall"] == 0 and abs(c["CL"] - 0.0122) < 5e-4 and -1.0 < ae.min() and ae.max() < 1.0
    p = r["polar"]
    assert p["in_range"][list(p["alpha"]).index(1.0)]
    orig = WG.solve
    def failing(w, V, a, pol, *args, **kw):                   # one polar point that does not converge
        s = orig(w, V, a, pol, *args, **kw)
        return dict(s, residual=1.0) if a == 5.0 else s
    monkeypatch.setattr(WG, "solve", failing)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = WG.compute(d, 45.0, 1.0, "xfoil")["polar"]
    al = list(p["alpha"])
    assert not p["in_range"][al.index(5.0)] and p["in_range"][al.index(4.0)] and p["in_range"][al.index(6.0)]


@pytest.mark.skipif(not os.environ.get("VORTFOIL_SLOW"), reason="research regression, minutes: set VORTFOIL_SLOW=1")
def test_research_regression(cache):
    WG.check(regression=True)
