# -*- coding: utf-8 -*-
"""Validation of predict_field.boundary_layer() (the RANS-state boundary layer) against the AirfRANS RANS fields (full_test).

For every case the decoder's layer comes from the section's coordinates alone (the case's 441-node NACA polyline, the incidence
and the Reynolds number, family 'naca'), as the application computes it (prepare, _wall, _boundary_layer = boundary_layer()).
On the same wall-normal rays (predict_field.bl_rays) the RANS velocity and pressure are sampled from the case's own mesh (VTK
probe of its cells) and given to the same predict_field.bl_states: one extractor, two fields. Relative errors (decoder / RANS
- 1) of theta, delta* and H are split by surface, by region (leading edge x/c < 0.1, aft 0.1-0.95, trailing edge > 0.95) and by
the RANS state of the station (separated: near-wall flow against the outer flow; else attached). Criterion (set before the
run): median |relative error| <= 0.20 for theta and delta* on the attached aft stations. Also measured on the RANS fields: the
research gate's reading (the edge criterion on u_t itself instead of u_t / U_I) at the same stations, and the research audit's
U4 on both fields: the relative spread of theta and delta* as the edge fraction moves over 0.98, 0.99, 0.995, 0.999 (registered:
median <= 10%).
Writes results/field_boundary_layer_validation.json.

  AIRFRANS_ROOT=/path/to/AirfRANS/Dataset python code/field/validate_field_boundary_layer.py [--workers 6] [--limit N]
(about 3 s per case and worker; needs airfrans and pyvista, as the dataset does)
"""
import os, sys, json, time, datetime, hashlib
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import vortfoil_paths as VP                         # noqa: E402
VP.add_paths()
sys.path.insert(0, HERE)

RECORD = os.path.join(VP.RESULTS, "f0cm_final_rescore_20261002T051450Z", "per_case.npz")   # the 200 full_test cases
REGIONS = {"leading_edge": (0.0, 0.1), "aft": (0.1, 0.95), "trailing_edge": (0.95, 1.0)}
CRITERION = 0.20
LADDER = (0.98, 0.99, 0.995, 0.999)                                    # research audit U4
_PF = None


def _init():
    global _PF
    os.environ.setdefault("VORTFOIL_FIELD_THREADS", "1")              # one thread per worker
    import predict_field as PF
    import vf_field_fast as F
    F.install()                                                        # as boundary_layer(fast=True), the application's call
    _PF = PF


def _work(nm):
    import warnings
    import airfrans as af
    import pyvista as pv
    from supervised_field.nf_eval200 import ROOT
    from supervised_field.airfrans_p3_fields_v2 import case_params, analytic_polyline, NU_AIR
    PF = _PF
    t0 = time.time()
    U, a, prm = case_params(nm); Re = U / NU_AIR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = PF.prepare(analytic_polyline(prm), a, Re, family="naca")    # boundary_layer(), step by step, to keep its rays
        w = PF._wall(c, Re)
        dec = PF._boundary_layer(c, Re, w)
    i, k, tn, nn, X = PF.bl_rays(c["poly"], w["upper"])
    sim = af.Simulation(root=ROOT, name=nm)
    probe = pv.PolyData(np.c_[X.reshape(-1, 2), np.full(X.size // 2, sim.internal.points[0, 2])]).sample(sim.internal)
    ok = np.asarray(probe.point_data["vtkValidPointMask"]).astype(bool)
    V = np.asarray(probe.point_data["U"], np.float64)[:, :2]; p = np.asarray(probe.point_data["p"], np.float64).ravel()
    V[~ok], p[~ok] = np.nan, np.nan
    V, p = V.reshape(X.shape), p.reshape(X.shape[:2])
    ut, un = np.einsum("kmd,kd->km", V, tn), np.einsum("kmd,kd->km", V, nn)
    tru = PF.bl_states(PF.BL_N, ut, un, p, U, k)
    raw = PF.bl_states(PF.BL_N, ut, 0.0 * un, 0.0 * p, U)               # U_I = U: the gate's criterion on u_t itself
    edge, ladder = PF.layer_edge, {"rans": [], "dec": []}                # U4: the edge fraction ladder, on both fields
    try:
        for frac in LADDER:
            PF.layer_edge = lambda n, u, frac=frac: edge(n, u, frac)
            r = PF.bl_states(PF.BL_N, ut, un, p, U)
            ladder["rans"].append([r["theta"].tolist(), r["dstar"].tolist()])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = PF._boundary_layer(c, Re, w)
            ladder["dec"].append([np.r_[r["upper"][q], r["lower"][q]].tolist() for q in ("theta", "dstar")])
    finally:
        PF.layer_edge = edge
    d = {q: np.r_[dec["upper"][q], dec["lower"][q]] for q in dec["upper"]}
    return dict(case=nm, alpha=a, Re=Re, five_digit=len(prm) == 4, x=d["x_over_c"].tolist(), upper=(np.arange(len(i)) < k).tolist(),
                **{f"dec_{q}": np.asarray(d[q], float).tolist() for q in ("theta", "dstar", "H")}, dec_separated=d["separated"].tolist(),
                **{f"rans_{q}": tru[q].tolist() for q in ("theta", "dstar", "H", "delta")}, rans_separated=tru["separated"].tolist(),
                **{f"gate_{q}": raw[q].tolist() for q in ("theta", "dstar")}, ladder=ladder, probe_valid=float(ok.mean()),
                secs=time.time() - t0)


def _stats(e):
    e = np.abs(np.asarray(e, float))
    return dict(n=int(e.size), median_abs_rel=float(np.median(e)) if e.size else None,
                p90_abs_rel=float(np.percentile(e, 90)) if e.size else None)


def main():
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    nw, lim = int(arg("--workers", "6")), arg("--limit")
    cases = [str(c) for c in np.load(RECORD, allow_pickle=True)["case"]]
    cases = cases[:: max(1, len(cases) // int(lim))][: int(lim)] if lim else cases      # --limit: spread over the sorted list
    t0 = time.time()
    with mp.get_context("spawn").Pool(nw, initializer=_init) as pool:
        rows = pool.map(_work, cases, chunksize=1)
    T = lambda k: np.concatenate([np.asarray(r[k], float) for r in rows])
    x, upper, sep = T("x"), T("upper").astype(bool), T("rans_separated").astype(bool)
    both = np.isfinite(T("rans_theta")) & np.isfinite(T("dec_theta"))
    err = {q: T(f"dec_{q}") / T(f"rans_{q}") - 1 for q in ("theta", "dstar", "H")}
    table = {}
    for surf, sm in (("upper", upper), ("lower", ~upper), ("both", np.ones_like(upper))):
        for reg, (lo, hi) in REGIONS.items():
            rm = (x >= lo) & (x < hi) if hi < 1 else (x >= lo)
            for st, stm in (("attached", ~sep), ("separated", sep)):
                m = sm & rm & stm & both
                table[f"{surf}/{reg}/{st}"] = {q: _stats(err[q][m]) for q in err}
    verdict = {}
    for surf in ("upper", "lower", "both"):
        cell = table[f"{surf}/aft/attached"]
        verdict[surf] = dict(theta=cell["theta"]["median_abs_rel"], dstar=cell["dstar"]["median_abs_rel"],
                             passed=bool(cell["theta"]["median_abs_rel"] <= CRITERION and cell["dstar"]["median_abs_rel"] <= CRITERION))
    rt, dt = np.isfinite(T("rans_theta")), np.isfinite(T("dec_theta"))
    gate = {}
    for reg, (lo, hi) in REGIONS.items():
        rm = (x >= lo) & (x < hi) if hi < 1 else (x >= lo)
        g = np.isfinite(T("gate_theta"))
        gb = rm & g & rt
        gate[reg] = dict(stations=int(rm.sum()), no_layer_read_gate_criterion=float(np.mean(~g[rm])),
                         no_layer_read_this_reading=float(np.mean(~rt[rm])),
                         median_abs_rel_gate_vs_this_theta=float(np.median(np.abs(T("gate_theta")[gb] / T("rans_theta")[gb] - 1))) if gb.any() else None,
                         median_abs_rel_gate_vs_this_dstar=float(np.median(np.abs(T("gate_dstar")[gb] / T("rans_dstar")[gb] - 1))) if gb.any() else None)
    u4 = {}
    for fld in ("rans", "dec"):
        L = np.concatenate([np.asarray(r["ladder"][fld], float) for r in rows], axis=2)        # (ladder, quantity, stations)
        for j, q in enumerate(("theta", "dstar")):
            M = L[:, j]; okm = np.isfinite(M).all(0)
            for reg, (lo, hi) in REGIONS.items():
                m = okm & (x >= lo) & ((x < hi) if hi < 1 else True)
                sp = (M.max(0) - M.min(0))[m] / np.median(M, axis=0)[m]
                u4[f"{fld}/{q}/{reg}"] = dict(n=int(m.sum()), median_rel_spread=float(np.median(sp)), p90_rel_spread=float(np.percentile(sp, 90)))
    alpha, Re = np.array([r["alpha"] for r in rows]), np.array([r["Re"] for r in rows])
    per_case = []                                                       # one line per case (the stations are re-made by a rerun)
    for r in rows:
        xc, sp, up = np.asarray(r["x"]), np.asarray(r["rans_separated"]), np.asarray(r["upper"], bool)
        m = (xc >= REGIONS["aft"][0]) & (xc < REGIONS["aft"][1]) & ~sp
        med = {}                                                        # both surfaces, then each
        for tag, sm in (("", m), ("_upper", m & up), ("_lower", m & ~up)):
            for q in ("theta", "dstar"):
                v = np.abs(np.asarray(r[f"dec_{q}"], float)[sm] / np.asarray(r[f"rans_{q}"], float)[sm] - 1)
                med[f"aft_attached_median_abs_rel_{q}{tag}"] = float(np.nanmedian(v)) if np.isfinite(v).any() else None
        per_case.append(dict(case=r["case"], alpha=r["alpha"], Re=r["Re"], five_digit=r["five_digit"], stations=int(xc.size),
                             rans_read=int(np.isfinite(r["rans_theta"]).sum()), decoder_read=int(np.isfinite(r["dec_theta"]).sum()),
                             rans_separated=int(sp.sum()), probe_valid=r["probe_valid"], secs=round(r["secs"], 2), **med))
    res = dict(schema="FIELD_BOUNDARY_LAYER_VALIDATION_V1", utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
               script=os.path.relpath(__file__, VP.RELEASE_ROOT), script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
               predict_field_sha256=hashlib.sha256(open(os.path.join(HERE, "predict_field.py"), "rb").read()).hexdigest(),
               n_cases=len(rows), five_digit_cases=int(sum(r["five_digit"] for r in rows)), alpha_range=[float(alpha.min()), float(alpha.max())],
               Re_range=[float(Re.min()), float(Re.max())], cases_with_separated_rans_stations=int(sum(any(r["rans_separated"]) for r in rows)),
               seconds=round(time.time() - t0, 1), criterion=f"median |relative error| <= {CRITERION} for theta and delta* on the attached aft "
               f"stations (x/c {REGIONS['aft'][0]}-{REGIONS['aft'][1]})", verdict=verdict,
               stations=dict(total=int(x.size), rans_read=int(rt.sum()), decoder_read=int(dt.sum()), both_read=int(both.sum()),
                             rans_read_decoder_not=int((rt & ~dt).sum()), rans_separated=int(sep.sum())),
               errors=table, gate_reading_on_rans=gate, edge_fraction_ladder_U4=dict(ladder=list(LADDER), registered="median <= 0.10 on the RANS "
               "fields", passed_on_rans=bool(all(v["median_rel_spread"] <= 0.10 for k, v in u4.items() if k.startswith("rans/") and "/aft" in k)),
               spread=u4), cases=per_case)
    os.makedirs(VP.RESULTS, exist_ok=True)
    json.dump(res, open(os.path.join(VP.RESULTS, "field_boundary_layer_validation.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("cases", "errors", "edge_fraction_ladder_U4")}, indent=1))
    for key, v in u4.items():
        print(f"U4 {key:30s} n {v['n']:5d}  median spread {v['median_rel_spread']:.3f}  p90 {v['p90_rel_spread']:.3f}")
    for key, cell in table.items():
        if cell["theta"]["n"]:
            print(f"{key:34s} n {cell['theta']['n']:5d}  " + "  ".join(
                f"{q} {cell[q]['median_abs_rel']:.3f}/{cell[q]['p90_abs_rel']:.3f}" for q in ("theta", "dstar", "H")))


if __name__ == "__main__":
    main()
