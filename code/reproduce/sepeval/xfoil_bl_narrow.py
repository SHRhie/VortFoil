"""
NeuroFoil — XFoil viscous BL extractor (rewrite for Drela source build).

Calls xfoil in viscous mode and uses DUMP to extract surface BL state.
The DUMP output (14 columns) is:
    s, x, y, Ue/Vinf, Dstar, Theta, Cf, H, H*, P, m, K   (12 numeric values;
    header lists 14 names but P,m,K replace the trailing tau,Di for surface
    panels — wake adds the trailing 4 columns).

Surface ordering produced by XFOIL panel numbering: TE upper → LE →
TE lower (so the rows traverse the airfoil contour once).  We split into
upper/lower based on the LE node (minimum x).

Transition x_tr_top / x_tr_bot are parsed from XFOIL's stdout (the
"Side 1 free transition at x/c = ..." lines after a viscous solve).
Cl, Cd, Cm parsed from the same stdout block.

Cached on disk under .xfoil_bl_cache/.
"""
from __future__ import annotations

import os, re, json, shutil, subprocess, tempfile, hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import re as _re
_MRCH_RE = _re.compile(r"MRCH(?:DU|UE): Convergence failed")


def _default_xfoil_path() -> str:
    """OS-aware default for the XFOIL binary.  Override via env XFOIL_EXE.

    Prefers a custom Drela-source build under ~/xfoil_src/bin/xfoil over
    the Debian apt package — the apt build is an older binary that
    happens to use the same DUMP format but ships with extra patches.
    """
    if os.name == "nt":
        return r"C:\Users\KHU\Desktop\XFOIL6.99\xfoil.exe"
    home_build = Path.home() / "xfoil_src/bin/xfoil"
    if home_build.is_file():
        return str(home_build)
    p = shutil.which("xfoil")
    return p if p else "/usr/bin/xfoil"


_DEFAULT_XFOIL = _default_xfoil_path()
XFOIL_EXE = Path(os.environ.get("XFOIL_EXE", _DEFAULT_XFOIL))
CACHE_DIR = Path(__file__).resolve().parent / ".xfoil_bl_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_key(naca, alpha, Re, n_panels, n_crit):
    raw = f"{naca}|a{alpha:.4f}|Re{Re:.6e}|n{n_panels}|nc{n_crit:.2f}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def run_xfoil_visc_bl(naca: str, alpha_deg: float, Re: float,
                       n_panels: int = 200, max_iter: int = 200,
                       n_crit: float = 9.0,
                       coords=None,
                       cache_label: Optional[str] = None,
                       timeout: float = 120.0) -> Optional[dict]:
    """Run XFoil viscous + DUMP.  Return per-surface BL state + polar
    scalars, or None on failure.

    Geometry source:
      - naca: 4-digit string (e.g. "0012"), uses XFOIL's analytic NACA.
      - coords: tuple (xu, yu, xl, yl) of upper/lower surface arrays
        (LE→TE order per branch).  Written to a temp .dat in Selig
        format and loaded via XFOIL `LOAD`.  Use `cache_label` to
        distinguish CST-perturbed shapes from NACA names.

    Keys returned:
        cl, cd, cm, x_tr_top, x_tr_bot
        x_upper, ue_upper, dstar_upper, theta_upper, cf_upper, H_upper
        x_lower, ue_lower, dstar_lower, theta_lower, cf_lower, H_lower
        s_upper, s_lower
        converged (bool)
    """
    cache_id = cache_label if cache_label is not None else naca
    key = _cache_key(cache_id, float(alpha_deg), float(Re), n_panels, n_crit)
    cf_npz = CACHE_DIR / f"{key}.npz"
    if cf_npz.exists():
        try:
            return _load_cache(cf_npz)
        except Exception:
            cf_npz.unlink(missing_ok=True)

    if not XFOIL_EXE.is_file():
        print(f"XFOIL_EXE not found: {XFOIL_EXE}")
        return None

    work = Path(tempfile.mkdtemp(prefix="xfoil_bl_"))
    dump_file = work / "bl.dat"

    # Geometry source — NACA string or coordinate array → temp .dat
    if coords is not None:
        xu, yu, xl, yl = coords
        local_dat = work / "shape_selig.dat"
        with open(local_dat, "w") as f:
            f.write("shape\n")
            for x, y in zip(xu[::-1], yu[::-1]):     # upper TE→LE
                f.write(f"  {float(x):.7f}  {float(y):.7f}\n")
            for x, y in zip(xl[1:], yl[1:]):         # lower LE→TE (skip dup)
                f.write(f"  {float(x):.7f}  {float(y):.7f}\n")
        airfoil_cmds = [f"LOAD {local_dat.name}", ""]
    else:
        airfoil_cmds = [f"NACA {naca}"]

    # NOTE on command order: DUMP must be issued BEFORE PACC is enabled —
    # when PACC is active the DUMP command line gets swallowed (XFOIL
    # interprets it as a polar-dump filename argument).  We don't need
    # PACC at all; Cl/Cd/transition come from stdout parsing.
    cmds = [
        "PLOP", "G F", "",
        *airfoil_cmds,
        "PPAR", f"N {n_panels}", "", "",
        "OPER",
        f"VISC {Re:g}",
        f"ITER {max_iter}",
        "VPAR", f"N {n_crit:g}", "",
        f"ALFA {alpha_deg:g}",
        f"DUMP {dump_file}",
        "QUIT", "",
    ]
    stdin = "\n".join(cmds) + "\n"

    env = os.environ.copy()
    env.pop("DISPLAY", None)
    try:
        proc = subprocess.run([str(XFOIL_EXE)], input=stdin,
                                capture_output=True, text=True,
                                cwd=str(work), timeout=timeout, env=env)
    except (subprocess.TimeoutExpired, OSError):
        shutil.rmtree(work, ignore_errors=True)
        return None

    out = _parse_outputs(dump_file, proc.stdout)
    shutil.rmtree(work, ignore_errors=True)
    if out is None:
        return None

    try:
        _save_cache(cf_npz, out)
    except Exception:
        pass
    return out


def _parse_outputs(dump_file: Path, stdout: str) -> Optional[dict]:
    if not dump_file.exists() or dump_file.stat().st_size < 100:
        return None

    # ---- Parse DUMP file (14-col header, 12 numeric per surface row) ----
    rows = []
    with open(dump_file) as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            try:
                vals = [float(x) for x in s.split()]
            except ValueError:
                continue
            if len(vals) < 8:
                continue
            rows.append(vals)
    if len(rows) < 20:
        return None

    # Different number of columns on surface (12) vs wake (8) — keep only
    # surface rows where we have all the BL state we need.
    surface_rows = [r for r in rows if len(r) >= 8]
    arr = np.array([r[:8] for r in surface_rows], dtype=np.float64)
    s_vals = arr[:, 0]
    x_vals = arr[:, 1]
    y_vals = arr[:, 2]
    ue     = arr[:, 3]
    dstar  = arr[:, 4]
    theta  = arr[:, 5]
    cf     = arr[:, 6]
    H      = arr[:, 7]

    # Separate wake from airfoil surface.  XFOIL surface panel ordering is
    # TE_upper → LE → TE_lower, then wake panels with x > 1.  Save wake
    # state separately for far-wake Cd estimation.
    on_surf = x_vals <= 1.0 + 1e-3
    in_wake = ~on_surf
    x_wake   = x_vals[in_wake]; y_wake = y_vals[in_wake]
    ue_wake  = ue[in_wake];     ds_wake = dstar[in_wake]
    th_wake  = theta[in_wake];  H_wake  = H[in_wake]
    s_vals = s_vals[on_surf]; x_vals = x_vals[on_surf]; y_vals = y_vals[on_surf]
    ue     = ue[on_surf];     dstar = dstar[on_surf]; theta  = theta[on_surf]
    cf     = cf[on_surf];     H     = H[on_surf]

    # Find LE (minimum x) — split index.  XFOIL surface ordering is
    # TE_upper → LE → TE_lower, so rows [0..le_idx] are upper (in
    # reverse direction).
    le_idx = int(np.argmin(x_vals))

    # Upper: rows [0..le_idx] reversed so the array runs LE → TE_upper
    def _rev(a): return a[le_idx::-1]
    s_u   = _rev(s_vals)
    x_u   = _rev(x_vals)
    y_u   = _rev(y_vals)
    ue_u  = _rev(ue)
    ds_u  = _rev(dstar)
    th_u  = _rev(theta)
    cf_u  = _rev(cf)
    H_u   = _rev(H)

    # Lower: rows [le_idx..] — already LE → TE_lower order
    sel_l = slice(le_idx, None)
    s_l   = s_vals[sel_l]
    x_l   = x_vals[sel_l]
    y_l   = y_vals[sel_l]
    ue_l  = ue[sel_l]
    ds_l  = dstar[sel_l]
    th_l  = theta[sel_l]
    cf_l  = cf[sel_l]
    H_l   = H[sel_l]

    # ---- Parse stdout for Cl, Cd, Cm, x_tr ----
    cl, cd, cm = float("nan"), float("nan"), float("nan")
    x_tr_top, x_tr_bot = 1.0, 1.0
    # Capture the *last* occurrence per pattern (final converged iter).
    for ln in stdout.splitlines():
        m = re.search(r"a\s*=\s*([\-\d.]+)\s+CL\s*=\s*([\-\d.]+)", ln)
        if m:
            cl = float(m.group(2))
        m = re.search(r"Cm\s*=\s*([\-\d.]+)\s+CD\s*=\s*([\-\d.]+)", ln)
        if m:
            cm = float(m.group(1)); cd = float(m.group(2))
        m = re.search(r"Side\s+1\s+free\s+transition\s+at\s+x/c\s*=\s*([\d.]+)", ln)
        if m: x_tr_top = float(m.group(1))
        m = re.search(r"Side\s+2\s+free\s+transition\s+at\s+x/c\s*=\s*([\d.]+)", ln)
        if m: x_tr_bot = float(m.group(1))

    # STRICT convergence (2026-07-02 theory-audit fix): XFOIL prints the LAST
    # Newton iterate's CL/CD even when it hits the ITER limit, together with a
    # "VISCAL:  Convergence failed" message.  Parsing success alone therefore
    # does NOT imply convergence — check the failure message explicitly.
    # NARROWED (2026-09-18).  The substring test also matched the per-station
    # "MRCHDU/MRCHUE: Convergence failed at <i> side <j>" messages, which XFOIL
    # emits during a healthy BL march -- MEASURED at alpha=0 on some shapes and
    # 12..729 times at alpha=15..18.  That discarded precisely the separated
    # cases.  The original intent (comment above) is the GLOBAL Newton failure.
    viscal_failed = ("VISCAL:  Convergence failed" in stdout)
    station_march_failures = len(_MRCH_RE.findall(stdout))
    converged = (not (np.isnan(cl) or np.isnan(cd))) and (not viscal_failed)

    return {
        "cl": cl, "cd": cd, "cm": cm,
        "x_tr_top": x_tr_top, "x_tr_bot": x_tr_bot,
        "converged": converged, "viscal_failed": viscal_failed,
        "station_march_failures": station_march_failures,
        # Upper (LE → TE)
        "s_upper": s_u, "x_upper": x_u, "y_upper": y_u,
        "ue_upper": ue_u, "dstar_upper": ds_u, "theta_upper": th_u,
        "cf_upper": cf_u, "H_upper": H_u,
        # Lower (LE → TE)
        "s_lower": s_l, "x_lower": x_l, "y_lower": y_l,
        "ue_lower": ue_l, "dstar_lower": ds_l, "theta_lower": th_l,
        "cf_lower": cf_l, "H_lower": H_l,
        # Wake (TE → downstream).  Last entry ≈ far-wake state used by
        # XFOIL for Squire-Young Cd extrapolation.
        "x_wake":  x_wake,  "y_wake":  y_wake,
        "ue_wake": ue_wake, "dstar_wake": ds_wake,
        "theta_wake": th_wake, "H_wake": H_wake,
    }


def _save_cache(path: Path, data: dict):
    flat = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            flat[k] = v.astype(np.float64)
        elif isinstance(v, bool):
            flat[k] = np.array(int(v))
        else:
            flat[k] = np.array(v)
    np.savez(str(path), **flat)


def _load_cache(path: Path) -> dict:
    z = np.load(str(path))
    out = {}
    for k in z.files:
        v = z[k]
        if v.ndim == 0:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = bool(v)
        else:
            out[k] = v
    return out
