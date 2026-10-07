# -*- coding: utf-8 -*-
"""Smoke test of the confirmed VortFoil surface chain (OP1 + OP2 + drag closure).

Runs code/reproduce/eval_on_targets.py -- the exact code path behind the manuscript's
surface and drag numbers -- on data/eval/holdout_targets.npz with the weights shipped in
weights/, and checks two sealed numbers to 1e-6:

    circulation Cl R^2   results/surface_contract_cst16.json  ["cl_circ_r2"]
    Cd R^2 (fitted_poly2) results/final_fitted_poly2.json     ["cd_raw_r2"]

Run with the TF 2.10 environment:
    /opt/anaconda3/envs/tf210/bin/python code/smoke_confirmed_chain.py

The field-decoder check is a separate script, code/smoke_field_decoder.py, because
supervised_field imports the `airfrans` package, which lives in the TF 2.18 environment.
"""
import json
import os
import runpy
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vortfoil_paths as VP                                                # noqa: E402

TOL = 1e-6


def r2(p, t):
    p, t = np.asarray(p, np.float64), np.asarray(t, np.float64)
    return float(1 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2))


def main():
    VP.add_paths()
    tgt = os.path.join(VP.DATA_EVAL, "holdout_targets.npz")
    script = os.path.join(VP.CODE, "reproduce", "eval_on_targets.py")
    for p in (tgt, script,
              os.path.join(VP.WEIGHTS, "op1_supervised_eps0.50.scr_bd128.npz"),
              os.path.join(VP.WEIGHTS, "op2_m325_BT.npz"),
              os.path.join(VP.WEIGHTS, "cd_closure_poly2.npz")):
        if not os.path.exists(p):
            print(f"FAIL  missing input: {p}")
            return 1

    out = os.path.join(tempfile.mkdtemp(prefix="vortfoil_smoke_"), "holdout_preds.npz")
    sys.argv = [script, tgt, out]
    runpy.run_path(script, run_name="__main__")          # same chain, same gate, same weights
    d = np.load(out)

    ref_cl = json.load(open(os.path.join(VP.RESULTS, "surface_contract_cst16.json")))["cl_circ_r2"]
    ref_cd = json.load(open(os.path.join(VP.RESULTS, "final_fitted_poly2.json")))["cd_raw_r2"]
    checks = [
        ("circulation Cl R^2   ", r2(d["clp"], d["cl_t"]), ref_cl,
         "results/surface_contract_cst16.json['cl_circ_r2']"),
        ("Cd R^2 fitted_poly2  ", r2(d["cd_ode"], d["cd_t"]), ref_cd,
         "results/final_fitted_poly2.json['cd_raw_r2']"),
    ]
    bad = 0
    print(f"\nrows {len(d['cd_t'])}")
    for name, got, ref, key in checks:
        ok = abs(got - ref) <= TOL
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name} {got:.12f}  sealed {ref:.12f}  "
              f"|d| {abs(got - ref):.3e}  <- {key}")
    print("\nSMOKE " + ("PASS" if not bad else f"FAIL ({bad} check(s))"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
