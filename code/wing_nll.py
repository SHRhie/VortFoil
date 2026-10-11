# -*- coding: utf-8 -*-
"""The application's 3D wing: a quasi-3D wing on VortFoil sections by Phillips & Snyder's numerical nonlinear lifting line
(J. Aircraft 37(4), 2000). Ported from the research code (airfoil_operator, VortFoil/code/wing_nll.py, wing_viz3d.py and
wing_rans3d_case.py as of ac28ffe) with straight trailing legs (the research free wake moves C_L by about 1e-4).

The wing is a table of stations along its right half (root at y = 0 first; lengths in m): chord, twist (deg, nose up about
the quarter chord), quarter-chord x (sweep) and z (dihedral) and a section, linear in between; the left half mirrors it. It
is n spanwise horseshoe vortices on the quarter-chord line (cosine-spaced nodes, legs along the freestream). Each strip
carries the section at its span position at its own Reynolds number Re = V c / NU_AIR. Its 2D polar is tabulated in one of
two section states:
  xfoil  the VortFoil surface model (trained on XFOIL, free transition): C_l, C_d, C_m; the strips between stations of
         different sections blend the stations' 16-parameter (CST) fits linearly;
  sst    the RANS field decoder's wall values (trained on AirfRANS k-omega SST, fully turbulent; predict_field.surface):
         C_l, C_d from the wall pressure and wall-function friction, C_m from the same wall data; the strips between
         stations of different sections blend the sections' coordinates linearly (as the research 3D RANS geometry).
The vortex lifting law |V_i x dl_i| Gamma_i = 1/2 V^2 c_i ds_i C_l(alpha_i) is solved by Newton (V_i includes every vortex's
induced velocity, alpha_i is measured in the section's plane from its chord). Lift from rho sum Gamma_i (V_i x dl_i),
induced drag in the Trefftz plane, profile drag from the sections' C_d at their effective incidence. Valid for straight to
moderately swept wings of moderate to high aspect ratio in attached flow; sweep, tip and stall flows are not modelled.

Frames and signs. Body frame: x along the root chord (downstream), y spanwise (right wing y > 0), z up. Wind frame (the 3D
surfaces): X along the freestream, Y spanwise, Z up, the wing pitched nose up by alpha. Section c_m is about the quarter
chord and wing C_M about the root quarter-chord point with the mean aerodynamic chord as reference length; both are positive
nose up (the surface model's convention). Coefficients use the strips' area S = sum c ds (along the dihedral).

  python code/wing_nll.py --check                  analytic and consistency checks (a few seconds)
  python code/wing_nll.py --check --regression     plus the research results of both states (minutes on a cold cache)
"""
import os, re, sys, json, time, hashlib, warnings
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vortfoil_paths  # noqa: E402
vortfoil_paths.add_paths()            # this checkout's vortfoil package (src) and code/field, ahead of other installs

NU_AIR = 1.5e-5                 # m^2/s, air at about 20 C
AR_MIN, SWEEP_MAX_DEG = 4.0, 10.0      # lifting-line validity: aspect ratio; quarter-chord sweep (its effects are not modelled)
TABLE_X = np.arange(-10.0, 18.001, 0.25)  # XFOIL state: section incidences tabulated (deg), the surface model's training range
TABLE_S = np.arange(-8.0, 16.001, 1.0)    # RANS state: section incidences tabulated (deg; research 0.5)
RE_GRID = 0.05                  # RANS state: per shape, tables on this log10(Re) grid (decades) where that needs fewer tables
                                # than the strips' own Re values, linear in log10(Re); None: every table at a strip's Re.
# Why (Mac M5 Pro, serial, 0.088 s per decoder call): the research tables (0.5 deg, every strip's Re) took 88 s for the
# 'Full-scale wing' and 89 s for the 'Kinked wing'; TABLE_S and RE_GRID take 19 s and 43 s (the kinked wing's 15 lofted strip
# shapes need a table each) and reproduce them within 2.4e-4 in C_L and 0.42% in C_D0 at 2, 6 and 10 deg (accepted: 2e-3, 2%).
NX = 41                                   # chordwise points per surface in the 3D view
CACHE = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".cache"), "vortfoil", "wing_tables")
SST_COLS = ("cl", "cd", "cm", "cd_pressure", "cd_friction")
_VERSION = "wing-tables-1"              # part of every table key: bump when a table's content changes
_MEM = {}                               # tables of this process, by key (both states)


def horseshoe_v(P, A, B, u):
    """induced velocity at points P (n,3) of unit-strength horseshoes (m) A->B with legs along u: (n, m, 3)"""
    r1 = P[:, None, :] - A[None]; r2 = P[:, None, :] - B[None]
    n1 = np.linalg.norm(r1, axis=-1); n2 = np.linalg.norm(r2, axis=-1)
    c12 = np.cross(r1, r2); den = n1 * n2 * (n1 * n2 + np.sum(r1 * r2, axis=-1))
    bound = np.where((np.linalg.norm(c12, axis=-1) > 1e-12 * n1 * n2)[..., None],
                     (n1 + n2)[..., None] * c12 / np.where(np.abs(den) > 1e-300, den, 1.0)[..., None], 0.0)
    leg2 = np.cross(u, r2) / (n2 * (n2 - r2 @ u))[..., None]
    leg1 = np.cross(u, r1) / (n1 * (n1 - r1 @ u))[..., None]
    return (leg2 + bound - leg1) / (4 * np.pi)


class WingRangeWarning(UserWarning):
    """The wing, or a flight condition, lies outside what this lifting line with streamwise VortFoil sections models."""


def is_naca(sec):
    return isinstance(sec, str) and re.fullmatch(r"\d{4,5}", sec) is not None


def section_coords(sec, sections=None):
    """a station's section as a loop (Selig order): an array as given, a key of `sections` (the application's resolved
    'current' and 'uiuc:<name>' sections), a NACA 4/5-digit designation (the AirfRANS generator, as the 2D analysis draws
    it) or a coordinate file (Selig or Lednicer)"""
    if not isinstance(sec, str):
        return np.asarray(sec, np.float64)
    import predict_field as PF
    if sections and sec in sections:
        return np.asarray(sections[sec], np.float64)
    if is_naca(sec):
        return PF.naca_coords(sec)
    if sec == "current" or sec.startswith("uiuc:"):
        raise ValueError(f"section '{sec}': not available here (the application resolves 'current' and 'uiuc:<name>')")
    if not os.path.isfile(sec):
        raise ValueError(f"section '{sec}': not a NACA 4/5-digit designation and no such coordinate file")
    return PF.read_coords(sec)


class Wing:
    """A symmetric wing given by stations along its right half, root (y = 0) first: chord, twist (deg, nose up, about the
    quarter chord), quarter-chord position x (sweep) and z (dihedral), and section; linear between stations, the section as
    a linear blend of the neighbouring stations' shapes. Sections are streamwise (x-z planes). The keyword form (span,
    c_root, taper, twist_tip, sweep, root, tip) is the two-station wing; elliptic=True gives the elliptic chord law.
    coords: the stations' section loops, when already resolved (else section_coords of each station's section)."""

    def __init__(self, span=None, c_root=None, taper=1.0, twist_tip=0.0, sweep=0.0, root="2412", tip="2412", n=40, elliptic=False,
                 stations=None, coords=None):
        if stations is None:
            b2 = 0.5 * span
            stations = [dict(y=0.0, chord=c_root, twist=0.0, x=0.0, z=0.0, section=root),
                        dict(y=b2, chord=c_root * taper, twist=twist_tip, x=b2 * np.tan(np.radians(sweep)), z=0.0, section=tip)]
        self.stations, self.coords = [dict(s) for s in stations], coords
        self.sy = np.array([s["y"] for s in stations], np.float64)
        if self.sy[0] != 0 or np.any(np.diff(self.sy) <= 0):
            raise ValueError("stations: the root at y = 0 first, then increasing y")
        b2 = float(self.sy[-1]); self.b, self.n, self.elliptic, self._cst = 2 * b2, n, elliptic, None
        th = np.linspace(0, np.pi, n + 1); y = -b2 * np.cos(th)                           # cosine nodes, left tip -> right tip
        ym = -b2 * np.cos(0.5 * (th[:-1] + th[1:]))                                       # control points at theta midpoints (Phillips)
        xn, zn = self.at(y, "x"), self.at(y, "z")                                         # quarter-chord line
        self.A = np.c_[xn[:-1], y[:-1], zn[:-1]]; self.B = np.c_[xn[1:], y[1:], zn[1:]]
        self.dl = self.B - self.A
        f = ((ym - y[:-1]) / np.diff(y))[:, None]; self.P = self.A + f * self.dl
        self.y, self.yn, self.ds = ym, y, np.hypot(np.diff(y), np.diff(zn))               # strip span in the y-z plane
        self.c, self.cn = self.chord(ym), self.chord(y)
        self.twist, self.twn = np.radians(self.at(ym, "twist")), np.radians(self.at(y, "twist"))
        self.ua = np.c_[np.cos(self.twist), np.zeros(n), -np.sin(self.twist)]           # chordwise (LE -> TE)
        un = np.cross(self.ua, np.c_[np.zeros(n), np.diff(y), np.diff(zn)] / self.ds[:, None])
        self.un = un / np.linalg.norm(un, axis=1)[:, None]                                # section normal (up), tilted by dihedral
        self.S = float(np.sum(self.c * self.ds)); self.AR = self.b ** 2 / self.S
        self.mac = float(np.sum(self.c ** 2 * self.ds) / self.S)                          # mean aerodynamic chord
        self.ref = np.array([stations[0]["x"], 0.0, stations[0]["z"]], np.float64)        # moment reference: root quarter chord
        self.eta = np.abs(ym) / b2
        self._warn()

    def at(self, y, key):
        """a station quantity, linear in |y| between stations"""
        return np.interp(np.abs(y), self.sy, [s[key] for s in self.stations])

    def chord(self, y):
        if self.elliptic:
            return self.stations[0]["chord"] * np.sqrt(np.clip(1 - (np.abs(y) / self.sy[-1]) ** 2, 0, 1))
        return self.at(y, "chord")

    def blend(self, y):
        """for spanwise positions y: the station k below |y| and the weight of station k + 1"""
        a = np.abs(np.asarray(y, np.float64))
        k = np.clip(np.searchsorted(self.sy, a, side="right") - 1, 0, len(self.sy) - 2)
        return k, np.clip((a - self.sy[k]) / (self.sy[k + 1] - self.sy[k]), 0.0, 1.0)

    def cst(self, y=None):
        """CST16 of the sections at span positions y (default: the strips): the stations' fits, blended linearly"""
        if self._cst is None:
            from vortfoil import geometry
            from vortfoil.api import FIT_ERROR_WARN, TE_GAP_WARN
            self._cst, self._rot = [], []
            for k, s in enumerate(self.stations):
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):     # Accelerate BLAS raises spurious FP flags
                    c, info = geometry.fit_cst16(self.coords[k] if self.coords is not None else section_coords(s["section"]))
                if info["max_abs_error_excluding_te"] > FIT_ERROR_WARN or info["te_gap"] > TE_GAP_WARN:
                    warnings.warn(f"section {os.path.basename(str(s['section']))}: the 16-parameter form misses it by {100 * info['max_abs_error_excluding_te']:.2f}% "
                                  f"chord (trailing-edge gap {100 * info['te_gap']:.2f}%); the loads are for the fitted shape", WingRangeWarning, stacklevel=2)
                self._cst.append(c); self._rot.append(info["rotation_deg"])
        k, w = self.blend(self.y if y is None else y)
        C = np.array(self._cst)
        return (1 - w)[:, None] * C[k] + w[:, None] * C[k + 1]

    def chord_rotation(self, y=None):
        """the fits' chord rotation (deg) at span positions y, blended as cst(). A fit measures its chord from the node farthest
        from the trailing edge, so VortFoil is asked for the section's incidence minus this (its predict does the same for
        coordinates): -0.10 deg for NACA 2412, -0.54 deg for 23018, 0 for symmetric sections"""
        self.cst()
        k, w = self.blend(self.y if y is None else y)
        R = np.array(self._rot)
        return (1 - w) * R[k] + w * R[k + 1]

    def _warn(self):                    # compared as printed: b^2/S of an AR 4 wing is 3.999999999999999 for some n
        if round(self.AR, 2) < AR_MIN:
            warnings.warn(f"aspect ratio {self.AR:.2f} < {AR_MIN:g}: lifting-line loads are unreliable", WingRangeWarning, stacklevel=3)
        sweep = np.degrees(np.arctan(np.diff([s["x"] for s in self.stations]) / np.diff(self.sy)))
        if round(float(np.max(np.abs(sweep))), 1) > SWEEP_MAX_DEG:
            warnings.warn(f"quarter-chord sweep up to {np.max(np.abs(sweep)):.1f} deg > {SWEEP_MAX_DEG:g}: sweep is not modelled by the "
                          "lifting line (streamwise sections), the loads are unreliable", WingRangeWarning, stacklevel=3)


# ------------------------------------------------------------------------------------------- definitions
def _st(y, chord, twist, x, z, section):
    return dict(y=y, chord=chord, twist=twist, x=x, z=z, section=section)


def _preset(name, description, V, *stations):
    return name, dict(name=name, description=description, V=V, n=40, stations=[_st(*s) for s in stations])


PRESETS = dict((
    _preset("Model wing (span 2 m)", "the research demo wing: taper 0.5, washout 2 deg, NACA 2412 (AR 8.9)", 20.0,
            (0.0, 0.30, 0.0, 0.0, 0.0, "2412"), (1.0, 0.15, -2.0, 0.0, 0.0, "2412")),
    _preset("Full-scale wing (span 10 m)", "full-scale planform: taper 0.5, washout 2 deg, NACA 2412 (AR 8.9); the research "
            "3D RANS case", 45.0, (0.0, 1.5, 0.0, 0.0, 0.0, "2412"), (5.0, 0.75, -2.0, 0.0, 0.0, "2412")),
    _preset("Kinked wing (4415 to 2412, dihedral)", "kinked: inboard panel c 1.5->1.3 NACA 4415 flat; outboard c 1.3->0.5 "
            "lofted to NACA 2412, washout 3 deg, dihedral 6 deg", 60.0, (0.0, 1.5, 0.0, 0.0, 0.0, "4415"),
            (2.0, 1.3, -1.0, 0.0, 0.0, "4415"), (5.0, 0.5, -3.0, 0.0, 0.3153127057970294, "2412")),
    _preset("Rectangular, AR 6", "rectangular, AR 6, untwisted, NACA 2412", 30.0,
            (0.0, 1.5, 0.0, 0.0, 0.0, "2412"), (4.5, 1.5, 0.0, 0.0, 0.0, "2412")),
    _preset("Rectangular, AR 4", "rectangular, AR 4 (lifting-line limit), untwisted, NACA 2412", 30.0,
            (0.0, 1.5, 0.0, 0.0, 0.0, "2412"), (3.0, 1.5, 0.0, 0.0, 0.0, "2412")),
    _preset("Washout 6 deg", "taper 0.5, washout 6 deg, NACA 2412 (AR 8.9)", 30.0,
            (0.0, 1.5, 0.0, 0.0, 0.0, "2412"), (5.0, 0.75, -6.0, 0.0, 0.0, "2412")),
    _preset("Clark Y wing", "taper 0.5, washout 2 deg, Clark Y (UIUC coordinates, not NACA)", 30.0,
            (0.0, 1.5, 0.0, 0.0, 0.0, "uiuc:clarky"), (5.0, 0.75, -2.0, 0.0, 0.0, "uiuc:clarky")),
    _preset("Swept 30 deg", "taper 0.5, washout 2 deg, quarter-chord sweep 30 deg: outside the lifting line's range "
            "(documents the gap)", 30.0, (0.0, 1.5, 0.0, 0.0, 0.0, "2412"), (5.0, 0.75, -2.0, 2.886751345948128, 0.0, "2412")),
))


def _num(name, v, ok, what):
    """v as a finite float satisfying ok, else ValueError naming the field"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: not a number ({v!r})") from None
    if not (np.isfinite(f) and ok(f)):
        raise ValueError(f"{name}: must be {what} (got {f:g})")
    return f


def validate(defn):
    """a wing definition -> a checked copy (numbers as floats, twist/x/z default 0, n default 40, 'NACA 2412' -> '2412');
    ValueError with an English message naming the bad field"""
    if not isinstance(defn, dict):
        raise ValueError("a wing definition is a JSON object with 'stations'")
    st = defn.get("stations")
    if not isinstance(st, (list, tuple)) or len(st) < 2:
        raise ValueError("stations: give at least two (the root at y = 0 and the tip)")
    out = []
    for k, s in enumerate(st):
        where = f"station {k + 1}"
        if not isinstance(s, dict):
            raise ValueError(f"{where}: not an object")
        miss = [f for f in ("y", "chord", "section") if f not in s]
        if miss:
            raise ValueError(f"{where}: missing '{miss[0]}'")
        e = {f: _num(f"{where} {f}", s.get(f, 0.0), lambda v: True, "a number") for f in ("y", "chord", "twist", "x", "z")}
        _num(f"{where} chord", e["chord"], lambda v: v > 0, "positive (m)")
        sec = s.get("section")
        if not isinstance(sec, str) or not sec.strip():
            raise ValueError(f"{where} section: give a NACA 4/5-digit designation, 'current', 'uiuc:<name>' or a file path")
        m = re.fullmatch(r"(?i)\s*(?:naca)?\s*(\d{4,5})\s*", sec)
        e["section"] = m.group(1) if m else sec.strip()
        if out and e["y"] <= out[-1]["y"]:
            raise ValueError(f"{where} y: must increase from the root to the tip (got {e['y']:g} after {out[-1]['y']:g})")
        out.append(e)
    if out[0]["y"] != 0:
        raise ValueError(f"station 1 y: the root station must be at y = 0 (got {out[0]['y']:g})")
    d = dict(defn, stations=out, name=str(defn.get("name", "wing")), description=str(defn.get("description", "")),
             n=int(_num("n", defn.get("n", 40), lambda v: v == int(v) and 8 <= v <= 200, "a whole number of strips from 8 to 200")))
    if "V" in defn:
        d["V"] = _num("V", defn["V"], lambda v: v > 0, "a positive speed (m/s)")
    if "alpha" in defn:
        d["alpha"] = _num("alpha", defn["alpha"], lambda a: abs(a) <= 30, "between -30 and 30 deg")
    return d


def load_wing(path):
    """a wing definition from JSON (the research wing_defs format); a section that is not a NACA designation, 'current' or
    'uiuc:<name>' is a coordinate file, relative to the definition's folder"""
    with open(path, encoding="utf-8-sig") as fh:                         # with or without the BOM Windows editors write
        try:
            d = json.load(fh)
        except json.JSONDecodeError as e:
            raise ValueError(f"{os.path.basename(path)}: not valid JSON ({e})") from None
        except UnicodeDecodeError:
            raise ValueError(f"{os.path.basename(path)}: not UTF-8 text (save it as UTF-8)") from None
    d = validate(d)
    base = os.path.dirname(os.path.abspath(path))
    for s in d["stations"]:
        if not (is_naca(s["section"]) or s["section"] == "current" or s["section"].startswith("uiuc:")):
            s["section"] = os.path.normpath(os.path.join(base, s["section"]))
    return d


def save_wing(defn, path):
    """the definition as JSON in the research wing_defs format"""
    d = validate(defn)
    keep = {k: d[k] for k in ("name", "description", "V", "n") if k in d}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(keep, stations=[{k: s[k] for k in ("y", "chord", "twist", "x", "z", "section")} for s in d["stations"]]),
                  fh, indent=1)


# ------------------------------------------------------------------------------------------- sections
def strip_loops(w, coords, names, labels=None):
    """each strip's unit-chord section as the field decoder gets it (research strip_shapes / wing_rans3d_case.Geo.loop): the
    station section between stations of the same section (a NACA designation drawn by the AirfRANS generator, family 'naca'),
    else the linear blend of the two stations' coordinates on common stations (predict_field.section_polygon: 221 nodes per
    surface, cosine in arc length), family 'naca' when both stations are NACA designations (a 4415/2412 blend lies within
    6e-5 c of the 4-digit section of interpolated camber and thickness; the decoder's shape-code check still guards unlike
    shapes), else None. Returns (loops, index per strip, family per loop, label per loop); labels: {section: display name}.
    Deviation from the research matrix: a coordinate-file or UIUC section goes to the decoder as given (predict_field
    spline-resamples it and closes an open trailing edge, as the 2D application does), not through wing_rans3d_case.raw_loop
    (linear on 100 cosine-x intervals, closed at the upper trailing-edge point). For the Clark Y wing that raw loop gave a
    30-65% lower SST C_D0 with a negative pressure drag at 0 and 4 deg; this handling is closer to the 3D RANS at 60 m/s
    (wing C_D 0.0140/0.0304/0.0545 against RANS 0.0166/0.0307/0.0584 and raw loop 0.0112/0.0254/0.0492 at 2/6/10 deg)."""
    import predict_field as PF
    labels = labels or {}
    label = lambda s: labels.get(s) or (f"NACA {s}" if is_naca(s) else os.path.splitext(os.path.basename(s.replace("uiuc:", "")))[0])
    poly = {}

    def resampled(j):                                       # station j's section on the decoder's common stations
        if j not in poly:
            g, _, _, _ = PF.field_frame(coords[j])
            poly[j] = PF.section_polygon(np.r_[g, g[:1]] if np.allclose(coords[j][0], coords[j][-1]) else g)
        return poly[j]
    k, t = w.blend(w.y)
    loops, fam, lab, index, seen = [], [], [], [], {}
    for kk, tt in zip(k, t):
        a, b = names[kk], names[kk + 1]
        if a == b or tt <= 0 or tt >= 1:
            j = kk + 1 if (a != b and tt >= 1) else kk
            key = names[j]
            if key not in seen:
                seen[key] = len(loops); loops.append(coords[j]); fam.append("naca" if is_naca(key) else None); lab.append(label(key))
        else:
            key = (kk, round(float(tt), 9))
            if key not in seen:
                seen[key] = len(loops); loops.append((1 - tt) * resampled(kk) + tt * resampled(kk + 1))
                fam.append("naca" if is_naca(a) and is_naca(b) else None)
                lab.append(f"{label(a)} to {label(b)} blend {tt:.2f}")
        index.append(seen[key])
    return loops, np.array(index), fam, lab


def _hash(*parts):
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:24]


def _shape_key(*arrays):
    return hashlib.sha1(b"".join(np.round(np.asarray(a, np.float64), 9).tobytes() for a in arrays)).hexdigest()[:16]


def _plan(keys, Re, grid):
    """per strip [(table Re, weight)]: the strip's own Re (4 significant digits), or, per shape where that needs fewer
    tables, the two nodes of the log10(Re) grid around it with linear weights in log10(Re)"""
    sig = lambda x: float(f"{x:.4g}")
    plan = {i: [(sig(Re[i]), 1.0)] for i in range(len(keys))}
    if grid:
        for s in dict.fromkeys(keys):
            idx = [i for i, k in enumerate(keys) if k == s]
            g = {}
            for i in idx:
                u = np.log10(Re[i]) / grid; k0 = int(np.floor(u + 1e-9)); a, b = sig(10 ** (k0 * grid)), sig(10 ** ((k0 + 1) * grid))
                f = (np.log10(Re[i]) - np.log10(a)) / (np.log10(b) - np.log10(a))
                g[i] = [(a, 1.0)] if f < 1e-9 else [(a, 1.0 - f), (b, f)]
            if len({R for i in idx for R, _ in g[i]}) < len({R for i in idx for R, _ in plan[i]}):
                plan.update(g)
    return [plan[i] for i in range(len(keys))]


def section_tables(state, keys, payload, Re, dec, tick, grid=None):
    """per strip (n, m) C_l, C_d, C_m (+ the SST drag split) over the state's incidence grid. One table per shape and Re
    (strips sharing both share it), kept for the process and, in the RANS state, on disk (CACHE); xfoil: one VortFoil sweep
    per table; sst: predict_field.surface per incidence (no process pools: the application runs under pythonw).
    payload(i) -> (cst, rotation) or (loop, family) of strip i; tick(text) before each unit of work (progress, cancel)."""
    import predict_field as PF
    sst = state == "sst"
    grid_deg = TABLE_S if sst else TABLE_X
    plan = _plan(keys, Re, grid if sst else None)
    tkey = lambda i, R: _hash(_VERSION, state, keys[i], f"{R:.4g}", grid_deg.tobytes().hex(), os.path.basename(dec) if sst else "")
    need = {tkey(i, R): (i, R) for i in range(len(keys)) for R, _ in plan[i]}
    todo = [k for k in need if k not in _MEM and not (sst and os.path.exists(os.path.join(CACHE, k + ".npz")))]
    tick.add(len(todo) * (len(grid_deg) if sst else 1))
    built = 0
    for k, (i, R) in need.items():
        if k in _MEM:
            continue
        path = os.path.join(CACHE, k + ".npz")
        if sst and os.path.exists(path):
            try:
                with np.load(path) as z:
                    _MEM[k] = {q: z[q] for q in SST_COLS}
                continue
            except Exception:                                                    # a damaged file is rebuilt
                tick.add(len(grid_deg))
        built += 1
        if not sst:
            cst, rot = payload(i)
            tick(f"XFOIL-state section tables: {built} of {len(todo)}")
            r = _model().sweep(cst, grid_deg - rot, Re=R)
            _MEM[k] = {q: np.ravel(getattr(r, q)).astype(np.float64) for q in ("cl", "cd", "cm")}
            continue
        loop, family = payload(i); rows = []
        for a in grid_deg:
            tick(f"RANS-state section tables: {built} of {len(todo)} (field decoder at {len(grid_deg)} incidences)")
            r = PF.surface(loop, float(a), R, decoder=dec, family=family)
            rows.append((r["cl"], r["cd"], r["cm"], r["cd_pressure"], r["cd_friction"]))
        _MEM[k] = dict(zip(SST_COLS, np.array(rows).T))
        try:
            os.makedirs(CACHE, exist_ok=True)
            with open(path + ".part", "wb") as fh:
                np.savez(fh, **_MEM[k])
            os.replace(path + ".part", path)
        except OSError:                                                          # a read-only or full disk: memory only
            pass
    cols = SST_COLS if sst else ("cl", "cd", "cm")
    tab = {q: np.array([sum(f * _MEM[tkey(i, R)][q] for R, f in plan[i]) for i in range(len(keys))]) for q in cols}
    return dict(tab, alpha=np.radians(grid_deg), built=built, reused=len(need) - built)


def _model():
    import vortfoil
    return vortfoil.load_model()           # cached per process (shared with view_airfoil)


# ------------------------------------------------------------------------------------------- lifting line
def solve(w, V, alpha_deg, polar, tol=1e-10, it_max=60, v=None):
    """Newton on the vortex lifting law; polar(i, alpha_rad) -> (C_l, dC_l/dalpha, C_d, C_m) for all segments at once.
    v: the induced velocity at the control points of each unit-strength horseshoe (n, n, 3, body frame); the straight
    horseshoes along the freestream when None."""
    a = np.radians(alpha_deg); u = np.array([np.cos(a), 0.0, np.sin(a)])
    if v is None:
        v = horseshoe_v(w.P, w.A, w.B, u)                                             # (n, n, 3)
    dA = w.c * w.ds
    al0 = np.arctan2(w.un @ u, w.ua @ u)
    G = 0.5 * V * w.c * polar(al0)[0] * 0.7
    for k in range(it_max):
        Vi = V * u + np.einsum("ijk,j->ik", v, G)
        wv = np.cross(Vi, w.dl); nw = np.linalg.norm(wv, axis=1)
        va, vn = np.sum(Vi * w.ua, 1), np.sum(Vi * w.un, 1)
        ai = np.arctan2(vn, va)
        cl, dcl, cd, cm = polar(ai)
        F = nw * G - 0.5 * V ** 2 * dA * cl
        if np.max(np.abs(F)) / (0.5 * V ** 2 * np.max(dA)) < tol:
            break
        dw = np.cross(v, w.dl[:, None, :])                                              # d(w_i)/dG_j = v_ij x dl_i
        J = np.einsum("ik,ijk->ij", wv, dw) * (G / nw)[:, None] + np.diag(nw)
        dai = (va[:, None] * np.einsum("ijk,ik->ij", v, w.un) - vn[:, None] * np.einsum("ijk,ik->ij", v, w.ua)) / (va ** 2 + vn ** 2)[:, None]
        J -= (0.5 * V ** 2 * dA * dcl)[:, None] * dai
        step = np.linalg.solve(J, -F); f0 = np.max(np.abs(F))
        for om in (1.0, 0.5, 0.25, 0.125, 0.0625):                                       # backtracking: accept a decrease
            Gt = G + om * step
            Vt = V * u + np.einsum("ijk,j->ik", v, Gt)
            Ft = np.linalg.norm(np.cross(Vt, w.dl), axis=1) * Gt - 0.5 * V ** 2 * dA * polar(np.arctan2(np.sum(Vt * w.un, 1), np.sum(Vt * w.ua, 1)))[0]
            if np.max(np.abs(Ft)) < f0:
                break
        G = Gt
    Vi = V * u + np.einsum("ijk,j->ik", v, G); ai = np.arctan2(np.sum(Vi * w.un, 1), np.sum(Vi * w.ua, 1))
    cl, _, cd, cm = polar(ai)
    Fs = G[:, None] * np.cross(Vi, w.dl)                                                # strip forces, rho = 1
    Ft = np.sum(Fs, axis=0)
    q = 0.5 * V ** 2 * w.S; uL = np.array([-np.sin(a), 0.0, np.cos(a)])
    CL = Ft @ uL / q
    # induced drag in the Trefftz plane: trailing vortices at the nodes carry the jumps of Gamma; their far-wake velocity normal
    # to the wake trace (the lifting line projected along the freestream) gives D_i = -1/2 rho sum Gamma_i w_i ds_i
    Pn = np.r_[w.A, w.B[-1:]]; yn, zn = Pn[:, 1], -Pn[:, 0] * np.sin(a) + Pn[:, 2] * np.cos(a)
    dy = w.P[:, 1][:, None] - yn[None]; dz = (-w.P[:, 0] * np.sin(a) + w.P[:, 2] * np.cos(a))[:, None] - zn[None]
    ty, tz = np.diff(yn), np.diff(zn); dsT = np.hypot(ty, tz)
    Gp = np.r_[0.0, G, 0.0]; gam = Gp[:-1] - Gp[1:]
    wT = np.sum(gam[None, :] * (dy * ty[:, None] + dz * tz[:, None]) / (dy ** 2 + dz ** 2), axis=1) / (2 * np.pi * dsT)
    CDi = -0.5 * np.sum(G * wT * dsT) / q
    CD0 = float(np.sum(cd * dA) / w.S)
    # pitching moment about the root quarter chord (y component, nose up > 0): the strip forces (vortex force and profile drag
    # along the freestream) at the strips' bound-vortex points, plus the section moments c_m q c^2 about the strips' axes
    Fs = Fs + (0.5 * V ** 2 * dA * cd)[:, None] * u
    r = w.P - w.ref
    CM = float((np.sum(r[:, 2] * Fs[:, 0] - r[:, 0] * Fs[:, 2]) + 0.5 * V ** 2 * np.sum(cm * w.c ** 2 * np.diff(w.yn))) / (q * w.mac))
    stall = int(np.sum(ai > polar.alpha_stall)) if hasattr(polar, "alpha_stall") else 0
    if stall:
        warnings.warn(f"alpha {alpha_deg:g} deg: {stall} of {w.n} strips beyond their section's stall angle; the lifting line "
                      "is unreliable there", WingRangeWarning, stacklevel=2)
    return dict(strips_past_stall=stall, alpha=alpha_deg, CL=float(CL), CDi=float(CDi), CD0=CD0, CD=float(CDi + CD0), CM=CM,
                e=float(CL ** 2 / (np.pi * w.AR * CDi)) if CDi > 0 else None,
                iterations=k + 1, residual=float(np.max(np.abs(F)) / (0.5 * V ** 2 * np.max(dA))), y=w.y.tolist(), gamma=G.tolist(),
                cl=cl.tolist(), alpha_eff=np.degrees(ai).tolist(), cd=cd.tolist(), cm=cm.tolist())


def table_polar(tab, deg=12):
    """per-section C_l(alpha) as a Chebyshev fit (continuous slope for Newton); C_d, C_m interpolated in the table"""
    A = tab["alpha"]; C = np.polynomial.Chebyshev
    fits = [C.fit(A, row, deg) for row in tab["cl"]]; d = [f.deriv() for f in fits]
    tab["cl_fit_max_err"] = float(max(np.max(np.abs(f(A) - row)) for f, row in zip(fits, tab["cl"])))
    def f(ai):
        ai = np.clip(ai, A[0], A[-1])
        cl = np.array([fits[i](x) for i, x in enumerate(ai)]); dcl = np.array([d[i](x) for i, x in enumerate(ai)])
        cd = np.array([np.interp(x, A, tab["cd"][i]) for i, x in enumerate(ai)])
        cm = np.array([np.interp(x, A, tab["cm"][i]) for i, x in enumerate(ai)])
        return cl, dcl, cd, cm
    f.alpha_stall = A[np.argmax(tab["cl"], axis=1)]                                     # incidence of each section's largest C_l
    return f


# ------------------------------------------------------------------------------------------- the application's wing
SST_KINDS = (("the field decoder was trained on NACA", "are not NACA 4/5-digit sections (the decoder's training family): read "
              "their values as indicative"), ("the coordinates were not in the chord frame", "were normalized to the chord frame"),
             ("the 16-parameter form deviates", "deviate from their 16-parameter form (the operator state is the fitted shape's)"),
             ("trailing-edge gap", "have a trailing-edge gap, closed for the decoder"),
             ("the section's shape code", "are unlike the decoder's NACA training sections (shape code)"))


def _e(x):
    return f"{x:.0e}".replace("e+0", "e").replace("e+", "e")          # 2e+06 -> 2e6


def _sst_warnings(msgs, Re, re_lo, re_hi):
    """the field decoder's messages at the strips' states, grouped by kind with strip counts"""
    import predict_field as PF
    kinds = {}
    for ms, R in zip(msgs, Re):
        for m in dict.fromkeys(ms):
            if m.startswith("Reynolds number"):
                k = f"at Re {'below' if R < re_lo else 'above'} its training range {_e(re_lo)}-{_e(re_hi)}"
            elif m.startswith("incidence"):
                k = f"at incidences outside its training range {PF.FIELD_ALPHA[0]:g} to {PF.FIELD_ALPHA[1]:g} deg"
            else:
                k = next((t for p, t in SST_KINDS if m.startswith(p)), m)
            kinds[k] = kinds.get(k, 0) + 1
    return [f"RANS decoder: {k}" if k.startswith("experimental") else f"RANS decoder: {c} of {len(msgs)} strips {k}"
            for k, c in kinds.items()]


class _Ticker:
    """progress(fraction, text) before each unit of work, cancel() between them"""

    def __init__(self, progress, cancel):
        self.progress, self.cancel, self.done, self.total = progress, cancel, 0, 0

    def add(self, k):
        self.total += k

    def __call__(self, text):
        if self.cancel is not None and self.cancel():
            raise RuntimeError("cancelled")
        if self.progress is not None:
            self.progress(min(self.done / max(self.total, 1), 1.0), text)
        self.done += 1


def _side(x, xs, vs):
    """one surface's values (nodes in any order) at the chordwise points x"""
    o = np.argsort(xs)
    return np.interp(x, np.asarray(xs)[o], np.asarray(vs)[o])


def compute(defn, V, alpha, state="xfoil", sections=None, progress=None, cancel=None, decoder=None, labels=None):
    """The wing of definition `defn` (validate) at speed V (m/s) and incidence alpha (deg, of the root chord) with its sections
    in `state` ('xfoil' or 'sst'). sections: {'current' | 'uiuc:<name>': unit-chord coordinates} resolved by the application
    (arrays are coordinates: family None for the field decoder; put a NACA section's designation in its station instead).
    labels: {section: display name} (e.g. {'current': 'clarky'}) for the loops' labels and the returned defn's stations.
    progress(fraction, text) is called from the calling thread; cancel() returning True stops between units of work with
    RuntimeError('cancelled'). decoder: the field decoder's weights file (RANS state; default predict_field.DECODER).
    Returns a plain dict of numbers and numpy arrays (n strips, m tabulated incidences, NX chordwise points):
      defn (validated; its sections as labelled), V, alpha, state, nu, decoder (file name, sst) | sst_range (sst, else
      None): n, below, above (strips with Re outside the decoder's training range lo-hi), lo, hi, message (None unless more
      than half the strips lie outside: then the sentence to lead the figure text and the status line with) | wing: name n b
      S AR MAC taper c_root c_tip Re_root Re_tip x_ref z_ref | coef: CL CDi CD0 CD LD e CM strips_past_stall in_range (+
      CD0_pressure CD0_friction, sst) | strips (n,): y eta chord twist_deg x_c4 z_c4 Re alpha_geo_deg alpha_eff_deg alpha_ind_deg
      gamma cl cd cm cl_c_over_cbar (+ cd_pressure cd_friction, sst) | polar (k,): alpha (1 deg steps over the state's training
      incidences, xfoil -10..18, sst -5..15, widened to alpha) CL CDi CD0 CD CM LD e in_range; in_range (coef and each polar
      point): every strip's incidence (as its model reads it: less the chord rotation it measures) and Re inside the state's
      training ranges and no strip past its section's stall | tables: alpha_deg
      (m,), cl cd cm (n, m) (+ cd_pressure cd_friction), cl_fit_max_err | surface: x_c (NX,), upper/lower: X Y Z (wind
      frame, m) and C_p, plus theta/c delta*/c H (xfoil) or C_f (sst), each (n, NX) | strip_detail: xfoil x_cp cp (n, 240;
      panels 0-119 upper TE->LE, 120-239 lower LE->TE), BL_X (121,), theta dstar (n, 2, 121; upper, lower); sst lists of n
      wall-node arrays x y cp cf upper | loops: coords (list of unit-chord loops), index (n,) into it, family ('naca' or
      None), label per loop | trailing (n+1,): y x_c4 z_c4 x_te z_te (wind frame) strength (shed circulation, m^2/s) |
      warnings [str] | newton: iterations residual | timing: tables solve strips total (s), built reused (tables)."""
    t0 = time.perf_counter()
    d = validate(defn)
    V = _num("V", V, lambda v: v > 0, "a positive speed (m/s)")
    alpha = _num("alpha", alpha, lambda a: abs(a) <= 30, "between -30 and 30 deg")
    if state not in ("xfoil", "sst"):
        raise ValueError("state: 'xfoil' (surface model) or 'sst' (field decoder)")
    import vortfoil
    import predict_field as PF
    from vortfoil import geometry
    sst, dec = state == "sst", decoder or PF.DECODER
    names = [s["section"] for s in d["stations"]]
    coords = [section_coords(s, sections) for s in names]
    tick = _Ticker(progress, cancel)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        w = Wing(stations=d["stations"], n=d["n"], coords=coords)
        with warnings.catch_warnings():
            if sst:
                warnings.simplefilter("ignore")         # the decoder reads the coordinates (it warns about its own fit)
            cst, rot = w.cst(), w.chord_rotation()
        n, Re = w.n, V * w.c / NU_AIR
        loops, li, fam, lab = strip_loops(w, coords, names, labels)
        if sst:                     # table keys and inputs; the state's training ranges, and the incidence offset of each strip's model
            keys = [_shape_key(loops[j]) + str(fam[j]) for j in li]
            payload = lambda i: (loops[li[i]], fam[li[i]])
            alim, relim = PF.FIELD_ALPHA, PF.FIELD_RE_OF.get(dec, PF.FIELD_RE)
            off = np.array([PF.field_frame(L)[1] for L in loops])[li]
        else:
            keys = [_shape_key(cst[i], rot[i]) for i in range(n)]
            payload = lambda i: (cst[i], rot[i])
            from vortfoil.api import TRAIN_ALPHA_DEG as alim, TRAIN_RE as relim
            off = rot
        tick.add(len({(k, f"{R:.4g}") for k, R in zip(keys, Re)}) if sst else 1)      # the strips' states (estimate)
        t1 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")             # the tables span the training ranges on purpose, and pass them by the
            tab = section_tables(state, keys, payload, Re, dec, tick, RE_GRID)   # fits' rotation; polar points may pass stall
            t2 = time.perf_counter()
            pol = table_polar(tab)
            al = np.arange(min(alim[0], np.floor(alpha)), max(alim[1], np.ceil(alpha)) + 0.5, 1.0)
            curve = [solve(w, V, a, pol) for a in al]
        s = solve(w, V, alpha, pol)
        t3 = time.perf_counter()
        ae = np.array(s["alpha_eff"])
        if sst:                                         # each strip's wall values at its state (mirror strips share one)
            got, det = {}, []
            for i in range(n):
                k = (keys[i], f"{Re[i]:.4g}", round(float(ae[i]), 6))
                if k not in got:
                    tick(f"RANS-state strips at their effective incidence: {len(got) + 1}")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")  # the messages are in meta['warnings'], grouped below
                        got[k] = PF.surface(loops[li[i]], float(ae[i]), float(Re[i]), decoder=dec, family=fam[li[i]])
                det.append(got[k])
            detail = {q: [np.asarray(r[q]) for r in det] for q in ("x", "y", "cp", "cf", "upper")}
        else:
            tick("XFOIL-state strips at their effective incidence")
            r = _model().predict(cst=cst, alpha=ae - rot, Re=Re)         # its range warnings are the strips' states'
            detail = dict(x_cp=r.x_cp, cp=r.cp, BL_X=vortfoil.BL_X, theta=r.theta, dstar=r.dstar)
    warn = list(dict.fromkeys(("" if issubclass(m.category, WingRangeWarning) else "strip sections: ") + str(m.message)
                              for m in ws if issubclass(m.category, (WingRangeWarning, vortfoil.RangeWarning))))
    sst_range = None
    if sst:
        lo, hi = PF.FIELD_RE_OF.get(dec, PF.FIELD_RE)
        warn += _sst_warnings([r["meta"]["warnings"] for r in det], Re, lo, hi)
        below, above = int(np.sum(Re < lo)), int(np.sum(Re > hi))      # more than half outside: say so first (app, figure)
        sst_range = dict(n=n, below=below, above=above, lo=lo, hi=hi, message=None if 2 * (below + above) <= n else
                         f"RANS state outside the field decoder's Reynolds range ({_e(lo)}-{_e(hi)}) on {below + above} of {n} "
                         "strips: the RANS-state numbers are not meaningful here; use the XFOIL state")

    def in_range(p):                                    # a solution's strips inside the state's training ranges, none past stall
        ae = np.asarray(p["alpha_eff"]) - off
        return bool(np.all((ae >= alim[0]) & (ae <= alim[1]) & (Re >= relim[0]) & (Re <= relim[1])) and p["strips_past_stall"] == 0)
    if s["residual"] > 1e-6:                            # e.g. far past stall: no lifting-line solution
        warn.append(f"the lifting line did not converge at {alpha:g} deg (residual {s['residual']:.0e} after {s['iterations']} Newton steps)")

    # strip values at alpha: the solution's (tables), the SST drag split interpolated alike
    A = tab["alpha"]; aec = np.clip(np.radians(ae), A[0], A[-1])
    rows = lambda q: np.array([np.interp(x, A, row) for x, row in zip(aec, tab[q])])
    G, cl, cd, cm = (np.array(s[q]) for q in ("gamma", "cl", "cd", "cm"))
    strips = dict(y=w.y, eta=w.y / (0.5 * w.b), chord=w.c, twist_deg=np.degrees(w.twist), x_c4=w.P[:, 0], z_c4=w.P[:, 2], Re=Re,
                  alpha_geo_deg=alpha + np.degrees(w.twist), alpha_eff_deg=ae, alpha_ind_deg=alpha + np.degrees(w.twist) - ae,
                  gamma=G, cl=cl, cd=cd, cm=cm, cl_c_over_cbar=w.c * cl / (w.S / w.b))
    coef = {q: s[q] for q in ("CL", "CDi", "CD0", "CD", "CM", "strips_past_stall")}
    coef.update(LD=s["CL"] / s["CD"], e=s["e"] if s["e"] is not None else float("nan"), in_range=in_range(s))
    if sst:
        strips.update(cd_pressure=rows("cd_pressure"), cd_friction=rows("cd_friction"))
        dA = w.c * w.ds
        coef.update(CD0_pressure=float(np.sum(strips["cd_pressure"] * dA) / w.S), CD0_friction=float(np.sum(strips["cd_friction"] * dA) / w.S))

    # 3D surfaces (research wing_viz3d): each strip's fitted section scaled by its chord, twisted about its quarter chord,
    # placed on the quarter-chord line and pitched by alpha (wind frame), with its 2D values at its effective incidence.
    # ponytail: drawn in each fit's chord frame, up to 0.5 deg nose down of the section's own chord (visual only)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        x, yu, yl = geometry.surface(cst, NX)
    a, th = np.radians(alpha), w.twist[:, None]
    surface = {"x_c": x}
    for side, ys, j in (("upper", yu, 0), ("lower", yl, 1)):
        xl, zl = (x - 0.25) * w.c[:, None], ys * w.c[:, None]
        xb = xl * np.cos(th) + zl * np.sin(th) + w.P[:, :1]; zb = -xl * np.sin(th) + zl * np.cos(th) + w.P[:, 2:]
        S = dict(X=xb * np.cos(a) + zb * np.sin(a), Y=np.repeat(w.y[:, None], NX, 1), Z=-xb * np.sin(a) + zb * np.cos(a))
        if sst:
            m = [u == (j == 0) for u in detail["upper"]]
            S["C_p"] = np.array([_side(x, xs[k], v[k]) for xs, v, k in zip(detail["x"], detail["cp"], m)])
            S["C_f"] = np.array([_side(x, xs[k], v[k]) for xs, v, k in zip(detail["x"], detail["cf"], m)])
        else:
            sl = slice(0, 120) if j == 0 else slice(120, 240)               # sheet panels: upper TE->LE, lower LE->TE
            S["C_p"] = np.array([_side(x, xs[sl], v[sl]) for xs, v in zip(r.x_cp, r.cp)])
            th_, ds_ = (np.array([np.interp(x, vortfoil.BL_X, v[j]) for v in arr]) for arr in (r.theta, r.dstar))
            S["theta/c"], S["delta*/c"] = th_, ds_
            ok = (th_ > 0) & (ds_ > 0)
            S["H"] = np.where(ok, ds_ / np.where(ok, th_, 1.0), np.nan)      # NaN where theta or delta* <= 0 (near the LE)
        surface[side] = S
    Gp = np.r_[0.0, G, 0.0]; Pn = np.r_[w.A, w.B[-1:]]
    xte, zte = surface["upper"]["X"][:, -1], surface["upper"]["Z"][:, -1]
    mid = lambda v: np.r_[v[0], 0.5 * (v[:-1] + v[1:]), v[-1]]                  # strip values -> nodes
    trailing = dict(y=w.yn, x_c4=Pn[:, 0] * np.cos(a) + Pn[:, 2] * np.sin(a), z_c4=-Pn[:, 0] * np.sin(a) + Pn[:, 2] * np.cos(a),
                    x_te=mid(xte), z_te=mid(zte), strength=Gp[:-1] - Gp[1:])

    c0, c1 = d["stations"][0]["chord"], d["stations"][-1]["chord"]
    wing = dict(name=d["name"], n=n, b=w.b, S=w.S, AR=w.AR, MAC=w.mac, taper=c1 / c0, c_root=c0, c_tip=c1,
                Re_root=V * c0 / NU_AIR, Re_tip=V * c1 / NU_AIR, x_ref=float(w.ref[0]), z_ref=float(w.ref[2]))
    nan = lambda v: float("nan") if v is None else v
    polar = dict(alpha=al, **{q: np.array([p[q] for p in curve]) for q in ("CL", "CDi", "CD0", "CD", "CM")})
    polar.update(LD=polar["CL"] / polar["CD"], e=np.array([nan(p["e"]) for p in curve]), in_range=np.array([in_range(p) for p in curve]))
    tables = dict(alpha_deg=np.degrees(tab["alpha"]), cl_fit_max_err=tab["cl_fit_max_err"],
                  **{q: tab[q] for q in (SST_COLS if sst else ("cl", "cd", "cm"))})
    t4 = time.perf_counter()
    if progress is not None:
        progress(1.0, "done")
    if labels:                                          # what the figure and the CSV show: 'clarky', not the token 'current'
        d = dict(d, stations=[dict(t, section=labels.get(t["section"], t["section"])) for t in d["stations"]])
    return dict(defn=d, V=V, alpha=alpha, state=state, nu=NU_AIR, decoder=os.path.basename(dec) if sst else None, wing=wing,
                sst_range=sst_range, coef=coef, strips=strips, polar=polar, tables=tables, surface=surface, strip_detail=detail,
                loops=dict(coords=loops, index=li, family=fam, label=lab), trailing=trailing, warnings=warn,
                newton=dict(iterations=s["iterations"], residual=s["residual"]),
                timing=dict(tables=t2 - t1, solve=t3 - t2, strips=t4 - t3, total=t4 - t0, built=tab["built"], reused=tab["reused"]))


# ------------------------------------------------------------------------------------------- checks
REGRESSION = (      # research results (wing_viz3d free wake; matrix 60 m/s: exact planform area), C_L at 2/6/10 deg, C_D or None
    ("Full-scale wing (span 10 m)", 45.0, "xfoil", (0.2817, 0.6265, 1.0001), None, 3e-3),
    ("Full-scale wing (span 10 m)", 45.0, "sst", (0.2743, 0.6242, 0.9588), None, 4e-3),
    ("Full-scale wing (span 10 m)", 60.0, "xfoil", (0.2850, 0.6218, 0.9938), (0.00877, 0.02060, 0.04505), None),
    ("Full-scale wing (span 10 m)", 60.0, "sst", (0.2777, 0.6255, 0.9643), (0.01247, 0.02455, 0.04558), None),
    ("Kinked wing (4415 to 2412, dihedral)", 60.0, "xfoil", (0.4068, 0.7330, 1.0890), None, None),
    ("Kinked wing (4415 to 2412, dihedral)", 60.0, "sst", (0.3821, 0.7236, 1.0558), None, None),
)   # not listed: the Clark Y wing's RANS state, which departs from the research on purpose (+32-65% C_D0; see strip_loops)


def check(regression=False):
    """analytic sections C_l = 2 pi alpha: an elliptic wing must give CL = 2 pi alpha/(1 + 2/AR) and e = 1; an extra collinear
    station changes nothing; dihedral lowers C_L by about cos^2; the wing C_M of a flat unswept wing is its sections' c_m and
    falls with sweep back; the XFOIL-state tables equal VortFoil's own coordinate path. regression=True adds the research
    results of both states (REGRESSION: C_L within the given absolute tolerance or 1%, C_D within 3%)."""
    lin = lambda ai: (2 * np.pi * ai, np.full_like(ai, 2 * np.pi), np.zeros_like(ai), np.zeros_like(ai))
    for n in (20, 40, 80):
        w = Wing(span=8.0, c_root=4 / np.pi * 1.0, n=n, elliptic=True)                 # AR = b^2/S with S = pi b c0 / 4
        s = solve(w, 10.0, 5.0, lin); th = 2 * np.pi * np.radians(5) / (1 + 2 / w.AR)
        print(f"elliptic n={n}: AR {w.AR:.3f}  CL {s['CL']:.5f} (theory {th:.5f})  e {s['e']:.4f}  Newton {s['iterations']}")
        assert abs(s["CL"] / th - 1) < 5e-3 and abs(s["e"] - 1) < 5e-3
    st = lambda y, c, t: dict(y=y, chord=c, twist=t, x=0.0, z=0.0, section="2412")
    w2, w3 = Wing(span=8.0, c_root=1.2, taper=0.5, twist_tip=-3.0, n=60), Wing(stations=[st(0, 1.2, 0), st(1.6, 0.96, -1.2), st(4.0, 0.6, -3.0)], n=60)
    s2, s3 = solve(w2, 10.0, 5.0, lin), solve(w3, 10.0, 5.0, lin)
    assert abs(s3["CL"] - s2["CL"]) < 1e-12 and abs(s3["CDi"] - s2["CDi"]) < 1e-12      # an extra collinear station: the same wing
    flat = solve(Wing(span=8.0, c_root=1.0, n=60), 10.0, 5.0, lin)
    for gd in (5.0, 10.0):          # dihedral: each panel sees alpha cos(G) and tilts its lift by G, so C_L falls by about cos^2(G)
        sd = solve(Wing(stations=[st(0, 1.0, 0), dict(st(4.0, 1.0, 0), z=4.0 * np.tan(np.radians(gd)))], n=60), 10.0, 5.0, lin)
        assert np.cos(np.radians(gd)) ** 2 - 0.02 < sd["CL"] / flat["CL"] < 1.0 and abs(sd["e"] - flat["e"]) < 0.02
    cm0 = lambda ai: lin(ai)[:3] + (np.full_like(ai, -0.05),)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", WingRangeWarning)                              # the sweep warning is expected
        sw = solve(Wing(span=8.0, c_root=1.2, taper=0.5, twist_tip=-3.0, sweep=20.0, n=60), 10.0, 5.0, cm0)
    sm = solve(w2, 10.0, 5.0, cm0)
    print(f"C_M with section c_m -0.05: flat unswept {sm['CM']:.6f}, swept back 20 deg {sw['CM']:.4f}")
    assert abs(sm["CM"] + 0.05) < 1e-12 and sw["CM"] < -0.05 - 0.1 * sw["CL"]
    wc, al = Wing(span=8.0, c_root=1.0, root="23018", tip="23018", n=4), np.array([-2.0, 0.0, 4.0])   # tables = VortFoil's
    cst, rot = wc.cst(), wc.chord_rotation()                                                          # own coordinate path
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = section_tables("xfoil", [_shape_key(c, r) for c, r in zip(cst, rot)], lambda i: (cst[i], rot[i]), np.full(4, 1e6), None, _Ticker(None, None))
    cl = t["cl"][0, [int(np.flatnonzero(TABLE_X == a)[0]) for a in al]]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):                           # Accelerate's spurious flags
        ref = np.asarray(_model().predict(coords=section_coords("23018"), alpha=al, Re=1e6).cl).ravel()
    print(f"NACA 23018 tables vs VortFoil from coordinates: C_l {np.round(cl, 4)} vs {np.round(ref, 4)} (fit rotation {rot[0]:.3f} deg)")
    assert np.allclose(cl, ref, atol=1e-4)
    if regression:
        bad = []
        print(f"{'wing':38s} {'V':>4s} {'state':5s} {'alpha':>5s} {'C_L ref':>8s} {'C_L':>8s} {'diff':>8s} {'C_D ref':>8s} {'C_D':>8s} {'diff':>7s}")
        for name, V, state, CL, CD, tol in REGRESSION:
            t = time.perf_counter()
            res = compute(PRESETS[name], V, 6.0, state)                                # the polar holds 2, 6 and 10 deg
            for k, a in enumerate((2.0, 6.0, 10.0)):
                j = int(np.flatnonzero(res["polar"]["alpha"] == a)[0])
                cl, cd = res["polar"]["CL"][j], res["polar"]["CD"][j]
                ok = abs(cl - CL[k]) < (tol or 0.01 * abs(CL[k])) and (CD is None or abs(cd / CD[k] - 1) < 0.03)
                bad += [] if ok else [f"{name} {V:g} m/s {state} {a:g} deg"]
                print(f"{name:38s} {V:4.0f} {state:5s} {a:5.0f} {CL[k]:8.4f} {cl:8.4f} {cl - CL[k]:+8.4f} "
                      + (f"{CD[k]:8.5f} {cd:8.5f} {cd / CD[k] - 1:+7.2%}" if CD else f"{'':8s} {cd:8.5f}") + ("" if ok else "  FAIL"))
            print(f"  ({time.perf_counter() - t:.1f} s; tables built {res['timing']['built']}, reused {res['timing']['reused']})")
        assert not bad, f"regression failed: {bad}"
    print("checks passed")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check(regression="--regression" in sys.argv)
