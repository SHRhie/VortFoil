# -*- coding: utf-8 -*-
"""Validation of predict_field.surface() against the RANS wall values of AirfRANS full_test (200 cases).

For every case, surface() runs from the section's coordinates alone (the case's 441-node NACA polyline as a coordinate loop,
the incidence and the Reynolds number), as a user would call it. Its wall C_p is compared with the RANS wall pressure at the
mesh's wall nodes (each wall node projected on the nearest segment of the section, C_p interpolated along it), and its pressure lift and pressure drag with
the AirfRANS force routine on the RANS fields (Simulation.force_coefficient(reference=True): total lift and its pressure
part, pressure drag). The decoder's training standard deviation of p normalizes the surface-pressure MSE, as for the field
errors in the manuscript.
Writes results/field_surface_validation.json.

  AIRFRANS_ROOT=/path/to/AirfRANS/Dataset python code/field/validate_field_surface.py [--workers 6] [--limit N]
"""
import os, sys, json, time, datetime, hashlib
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import vortfoil_paths as VP                         # noqa: E402
VP.add_paths()
sys.path.insert(0, HERE)

RECORD = os.path.join(VP.RESULTS, "f0cm_final_rescore_20261002T051450Z", "per_case.npz")
BANDS = [(-6, 0), (0, 8), (8, 12), (12, 16)]
_PF = None


def _init():
    global _PF
    import predict_field as PF
    _PF = PF


def _work(nm):
    import warnings
    import airfrans as af
    from supervised_field.nf_eval200 import ROOT
    from supervised_field.airfrans_p3_fields_v2 import case_params, analytic_polyline, NU_AIR
    t0 = time.time()
    U, a, prm = case_params(nm); Re = U / NU_AIR
    sim = af.Simulation(root=ROOT, name=nm)
    sm = np.asarray(sim.surface, bool); Pw = np.asarray(sim.position, np.float64)[sm, :2]
    cp_true = np.asarray(sim.pressure, np.float64).ravel()[sm] / (0.5 * U ** 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _PF.surface(analytic_polyline(prm), a, Re, family="naca")
    A = np.c_[r["x"], r["y"]]; B = np.roll(A, -1, axis=0); ca, cb = r["cp"], np.roll(r["cp"], -1)   # closed section, segment by segment
    AB = B - A; tt = np.clip(np.einsum("wsk,sk->ws", Pw[:, None] - A[None], AB) / np.sum(AB ** 2, 1)[None], 0, 1)
    k = np.argmin(np.sum((Pw[:, None] - (A[None] + tt[..., None] * AB[None])) ** 2, axis=2), axis=1)
    tk = tt[np.arange(len(Pw)), k]
    cp = ca[k] + tk * (cb[k] - ca[k])                                                # projected on the nearest segment
    (cd, cdp, _), (cl, clp, _) = sim.force_coefficient(reference=True)
    mid = (Pw[:, 0] >= 0.05) & (Pw[:, 0] <= 0.95)
    std_p = float(_PF.load_decoder()["std"][2])
    return dict(case=nm, alpha=a, Re=Re, cl=r["cl"], cd_pressure=r["cd_pressure"], cl_rans=float(cl), clp_rans=float(clp), cdp_rans=float(cdp),
                cd_rans=float(cd), cp_rmse=float(np.sqrt(np.mean((cp - cp_true) ** 2))), cp_rmse_mid=float(np.sqrt(np.mean((cp[mid] - cp_true[mid]) ** 2))),
                p_std_mse=float(np.mean(((cp - cp_true) * 0.5 * U ** 2 / std_p) ** 2)), secs=time.time() - t0)


def _rank(v):
    r = np.empty(len(v)); r[np.argsort(v)] = np.arange(len(v)); return r


def _stats(p, t):
    p, t = np.asarray(p), np.asarray(t)
    return dict(r2=float(1 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2)), mean_rel=float(np.mean(np.abs(p - t) / np.abs(t))),
                median_rel=float(np.median(np.abs(p - t) / np.abs(t))), spearman=float(np.corrcoef(_rank(p), _rank(t))[0, 1]),
                mae=float(np.mean(np.abs(p - t))), bias=float(np.mean(p - t)))


def main():
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    nw, lim = int(arg("--workers", "6")), arg("--limit")
    cases = [str(c) for c in np.load(RECORD, allow_pickle=True)["case"]]
    cases = cases[: int(lim)] if lim else cases
    t0 = time.time()
    with mp.get_context("spawn").Pool(nw, initializer=_init) as pool:
        rows = pool.map(_work, cases, chunksize=1)
    T = lambda k: np.array([r[k] for r in rows])
    alpha = T("alpha")
    res = dict(schema="FIELD_SURFACE_VALIDATION_V1", utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
               script=os.path.relpath(__file__, VP.RELEASE_ROOT), script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
               predict_field_sha256=hashlib.sha256(open(os.path.join(HERE, "predict_field.py"), "rb").read()).hexdigest(),
               n_cases=len(rows), seconds=round(time.time() - t0, 1),
               wall_cp=dict(rmse=float(T("cp_rmse").mean()), rmse_5_95_chord=float(T("cp_rmse_mid").mean()), p_std_mse=float(T("p_std_mse").mean()),
                            rmse_by_band={f"[{lo},{hi})": float(T("cp_rmse")[(alpha >= lo) & (alpha < hi)].mean()) for lo, hi in BANDS}),
               lift_vs_rans_total=_stats(T("cl"), T("cl_rans")), lift_vs_rans_pressure_part=_stats(T("cl"), T("clp_rans")),
               pressure_drag_vs_rans=_stats(T("cd_pressure"), T("cdp_rans")),
               rows=rows)
    os.makedirs(VP.RESULTS, exist_ok=True)
    json.dump(res, open(os.path.join(VP.RESULTS, "field_surface_validation.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
