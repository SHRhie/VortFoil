# -*- coding: utf-8 -*-
"""NeuralFoil xlarge per-case inference cost (2026-09-20), same rows as the attached-set comparison and the same call
the paper's NeuralFoil numbers came from (run_nf_full.py: get_aero_from_coordinates, n_crit 7, free transition, one call
per shape with all its (alpha, Re) pairs = NeuralFoil's natural batched sweep).  Also the single-query latency.
  VortFoil_v2/.venvs/nf_benchmark_033_20260912T230825Z/bin/python VortFoil/code/vf_speed_nf.py
"""
import os, sys, json, time, datetime, statistics, platform
import numpy as np
import neuralfoil
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
R = os.path.join(REPO, "VortFoil_v2/artifacts/op2_cdloss_20260917T044259Z")
z = np.load(os.path.join(R, "raw/nf_in_attached_v2.npz"), allow_pickle=False)
XY, al, Re = z["coords"], z["alpha"].reshape(-1), z["Re"].reshape(-1)
uniq = {}
for i in range(len(al)):
    uniq.setdefault(XY[i].tobytes(), []).append(i)
load0 = os.getloadavg()
def sweep():
    for _k, idx in uniq.items():
        idx = np.array(idx)
        neuralfoil.get_aero_from_coordinates(coordinates=XY[idx[0]].astype(np.float64), alpha=al[idx].astype(np.float64), Re=Re[idx].astype(np.float64),
                                             n_crit=7.0, xtr_upper=1.0, xtr_lower=1.0, model_size="xlarge")
sweep()                                                                   # warm-up
ts = []
for _ in range(3):
    t0 = time.perf_counter(); sweep(); ts.append(time.perf_counter() - t0)
t_sweep = statistics.median(ts)
def single():
    neuralfoil.get_aero_from_coordinates(coordinates=XY[0].astype(np.float64), alpha=float(al[0]), Re=float(Re[0]), n_crit=7.0, xtr_upper=1.0, xtr_lower=1.0, model_size="xlarge")
single(); ts1 = []
for _ in range(30):
    t0 = time.perf_counter(); single(); ts1.append(time.perf_counter() - t0)
out = dict(schema="VF_SPEED_NEURALFOIL_V1", utc=datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"), host=platform.platform(), cpu_count=os.cpu_count(),
           neuralfoil=getattr(neuralfoil, "__version__", "0.3.3"), model_size="xlarge", n_rows=int(len(al)), n_shapes=len(uniq), calls_per_sweep=len(uniq),
           sweep_wall_s=t_sweep, us_per_case_batched=1e6 * t_sweep / len(al), single_query_ms_median=1e3 * statistics.median(ts1), single_query_ms_min=1e3 * min(ts1),
           loadavg_start=load0, loadavg_end=os.getloadavg())
od = os.path.join(REPO, "VortFoil/tables/speed"); os.makedirs(od, exist_ok=True)
fp = os.path.join(od, f"_speed_confirmed_neuralfoil_{out['utc']}.json"); json.dump(out, open(fp, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("n_rows", "n_shapes", "sweep_wall_s", "us_per_case_batched", "single_query_ms_median")})); print("wrote", fp)
