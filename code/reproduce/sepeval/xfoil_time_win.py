# -*- coding: utf-8 -*-
"""Per-case XFOIL wall time on the Xeon (2026-09-20), for the manuscript's inference-cost comparison.

Same solver path as the evaluation decks (gen_targets_win.run_one -> xfoil_bl_narrow.run_xfoil_visc_bl:
viscous VISCAL, N_PANELS=200, MAX_ITER=200, n_crit as the deck, DUMP of the boundary layer), ONE process,
sequential, cache bypassed with a fresh cache_label prefix, so every call is a real XFOIL run.
  conda run -n base python xfoil_time_win.py --n_shapes 2 --out xfoil_time.json
"""
import sys, time, json, argparse, statistics, hashlib, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_targets_win as GT

ap = argparse.ArgumentParser()
ap.add_argument("--n_shapes", type=int, default=2)
ap.add_argument("--alphas", default="-4,0,4,8,12")
ap.add_argument("--Res", default="2e5,5e5,1e6,3e6")
ap.add_argument("--out", default="xfoil_time.json")
a = ap.parse_args()
alphas = [float(x) for x in a.alphas.split(",")]; Res = [float(x) for x in a.Res.split(",")]
prefix = "time" + time.strftime("%H%M%S")            # unique -> cache miss -> real solve
rows = []; j = 0
for i in range(a.n_shapes):
    for al in alphas:
        for re in Res:
            t0 = time.perf_counter(); r = GT.run_one((j, i, al, re, prefix)); dt = time.perf_counter() - t0
            rows.append(dict(shape=int(i), alpha=al, Re=re, secs=dt, converged=r[1] is not None, viscal_failed=int(r[3]))); j += 1
            print(f"  shape {i} a={al:+5.1f} Re={re:.0e}  {dt:6.2f}s  {'ok' if r[1] is not None else 'FAIL'}", flush=True)
secs = [x["secs"] for x in rows]; ok = [x["secs"] for x in rows if x["converged"]]
out = dict(schema="XFOIL_PER_CASE_TIME_V1", host="Xeon Gold 6530, Windows, single process", n=len(rows), n_converged=len(ok),
           settings=dict(n_panels=GT.N_PANELS, max_iter=GT.MAX_ITER, n_crit=float(GT.N_CRIT), timeout=GT.TIMEOUT, cache_bypassed=True),
           secs_all=dict(median=statistics.median(secs), mean=statistics.mean(secs), min=min(secs), max=max(secs)),
           secs_converged=dict(median=statistics.median(ok) if ok else None, mean=statistics.mean(ok) if ok else None), rows=rows)
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("n", "n_converged", "secs_all", "secs_converged")}))
