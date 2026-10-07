# -*- coding: utf-8 -*-
"""supervised_field/airfrans_p1_labels.py — MARIO comparison P1: generate the
FORCED-TRANSITION (XTR 0.05/0.05 = fully-turbulent, matching AirfRANS RANS truth to
1-3 cts per P0) XFOIL label set over the AirfRANS box.

- shapes : the 200 scarce_train case shapes (NACA 4/5-digit params from case names,
           geometry via airfrans.naca_generator — analytic, XFOIL-safe)
- grid   : alpha in [-5..15] step 1 (21) x Re in {2,3,4,5,6}e6 (5) -> 105 runs/shape
- runner : ONE process per run (this XFOIL build SIGILLs on a 2nd ALFA), headless
           (PLOP G F, captured stdout), md5 json cache -> fully resumable
- output : supervised_field/_airfrans_xtr_labels.npz
           (params[4] padded, n_params, alpha, Re, cl, cd, converged)
"""
import os, json, hashlib, subprocess, tempfile, re, time
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XFOIL_EXE", os.path.expanduser("~/src/xfoil/bin/xfoil"))
import numpy as np
from pathlib import Path
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AIRFRANS_ROOT",   # external dataset, not shipped with this release
                     os.path.join(os.path.dirname(HERE), "airfrans_data", "Dataset"))
CACHE = Path(HERE) / "_airfrans_xtr_cache"; CACHE.mkdir(exist_ok=True)
OUT = os.path.join(HERE, "_airfrans_xtr_labels.npz")
ALPHAS = [float(a) for a in range(-5, 16)]            # -5..15 deg
RES = [2e6, 3e6, 4e6, 5e6, 6e6]
NW = 6                                                 # parallel workers (P-cores)
BLOCK_RE = re.compile(r"a =\s*([-\d.]+)\s+CL =\s*([-\d.]+)\s*\n\s*Cm =\s*[-\d.]+"
                      r"\s+CD =\s*([-\d.]+)")


def train_shapes():
    with open(os.path.join(ROOT, "manifest.json")) as f:
        man = json.load(f)
    shapes = []
    seen = set()
    for name in man["scarce_train"]:
        prm = tuple(float(x) for x in name.split("_")[4:])
        if prm not in seen:
            seen.add(prm); shapes.append(prm)
    return shapes


def run_one(task):
    prm, alpha, Re = task
    key = hashlib.md5(f"XTR|{prm}|{alpha:.1f}|{Re:.3g}".encode()).hexdigest()[:16]
    cf = CACHE / f"{key}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    from airfrans.naca_generator import naca_generator
    s = np.asarray(naca_generator(np.array(prm), nb_samples=220,
                                  cosine_spacing=True, verbose=False, CTE=True))
    out = dict(params=list(prm), alpha=alpha, Re=Re, cl=None, cd=None)
    try:
        with tempfile.TemporaryDirectory() as td:
            with open(Path(td) / "foil.dat", "w") as f:
                f.write("af\n")
                for x, y in s:
                    f.write(f" {x:.6f} {y:.6f}\n")
            stdin = "\n".join(["PLOP", "G F", "", "LOAD foil.dat", "PANE", "OPER",
                               f"VISC {Re:.0f}", "ITER 400", "VPAR",
                               "XTR 0.05 0.05", "", f"ALFA {alpha:g}", "",
                               "QUIT"]) + "\n"
            p = subprocess.run([os.environ["XFOIL_EXE"]], input=stdin,
                               capture_output=True, text=True, cwd=td, timeout=150,
                               env={**os.environ, "DISPLAY": ""})  # no X11 plot window
            b = BLOCK_RE.findall(p.stdout)
            if b:
                a_, cl_, cd_ = (float(v) for v in b[-1])
                if abs(a_ - alpha) < 0.3 and 0.0 < cd_ < 0.5:
                    out["cl"], out["cd"] = cl_, cd_
    except Exception:
        pass
    cf.write_text(json.dumps(out))
    return out


def main():
    shapes = train_shapes()
    tasks = [(p, a, r) for p in shapes for a in ALPHAS for r in RES]
    print(f"[p1] {len(shapes)} shapes x {len(ALPHAS)} alpha x {len(RES)} Re "
          f"= {len(tasks)} runs ({NW} workers, headless, resumable)", flush=True)
    t0 = time.time(); rows = []
    with Pool(NW) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, tasks, chunksize=8)):
            rows.append(r)
            if (i + 1) % 500 == 0:
                ok = sum(1 for x in rows if x["cl"] is not None)
                print(f"  {i+1}/{len(tasks)}  converged {ok}/{len(rows)} "
                      f"({100*ok/len(rows):.1f}%)  [{time.time()-t0:.0f}s]", flush=True)
    ok = [r for r in rows if r["cl"] is not None]
    print(f"[p1] DONE {len(ok)}/{len(rows)} converged ({100*len(ok)/len(rows):.1f}%) "
          f"in {time.time()-t0:.0f}s", flush=True)
    P = np.zeros((len(rows), 4), np.float32)
    npar = np.zeros(len(rows), np.int8)
    for i, r in enumerate(rows):
        npar[i] = len(r["params"]); P[i, :npar[i]] = r["params"]
    np.savez(OUT, params=P, n_params=npar,
             alpha=np.array([r["alpha"] for r in rows], np.float32),
             Re=np.array([r["Re"] for r in rows], np.float32),
             cl=np.array([r["cl"] if r["cl"] is not None else np.nan for r in rows], np.float32),
             cd=np.array([r["cd"] if r["cd"] is not None else np.nan for r in rows], np.float32))
    print(f"[p1] saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
