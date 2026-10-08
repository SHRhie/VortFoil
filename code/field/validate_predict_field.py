# -*- coding: utf-8 -*-
"""Validation of code/field/predict_field.py against the deployed evaluation on AirfRANS full_test (200 cases).

Every case is re-assembled twice and scored with the deployed metric (state_provider.case_metrics: standardized MSE
per channel over the case's mesh nodes, prediction zeroed on the surface nodes as in training):
  E  same inputs as the deployed evaluation -- the case's NACA polyline, its mesh surface nodes for the shape code,
     the provider's state by case parameters -- through predict_field.field_inputs / decode. It must reproduce the
     recorded per-case errors (results/f0cm_final_rescore_20261002T051450Z/per_case.npz, CM_all) up to float64
     re-association (predict_field installs the fast paint kernels).
  U  only what a user has: the section's coordinates (the case's 441-node NACA polyline, given as a coordinate loop),
     the incidence and the Reynolds number, through predict_field.prepare (frame, section, operator state from the
     section, shape code from the section's nodes).
  C  the same from a sparse .dat-like section: the NACA generator at 81 stations per surface (161 nodes), which
     predict_field resamples on the training stations (section_polygon). Mesh nodes that this resampled wall puts inside
     the section (it moves the wall by ~1e-5 chord, inside the first cells of a RANS mesh) are scored at wall values,
     and are counted. Scores are reported on all nodes, near the wall (0 < d < 0.05) and off the body (d > 0.05).
     On a user-style grid (0.01 chord), the C prediction is also compared with the U prediction, as a fraction of the
     decoder's own test error (most RANS-mesh nodes sit inside the boundary layer; few grid points do).
Writes results/predict_field_validation.json and results/predict_field_validation_per_case.npz.

  AIRFRANS_ROOT=/path/to/AirfRANS/Dataset python code/field/validate_predict_field.py [--workers 6] [--limit N]
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
FIELDS = ("ux", "uy", "p", "nut")
BANDS = [(-6, 0), (0, 8), (8, 12), (12, 16)]
_PF = _W = _PROV = None


def _init():
    global _PF, _W, _PROV
    import predict_field as PF
    import vf_field_fast as F
    F.install()
    _PF, _W, _PROV = PF, PF.load_decoder(), PF.provider()


def _metrics(pred, TR, std, masks):
    return {mk: [float(np.mean(((pred[mm, c] - TR[mm, c]) / std[c]) ** 2)) for c in range(4)] for mk, mm in masks.items()}


def _work(nm):
    t0 = time.time()
    import airfrans as af
    from airfrans.naca_generator import naca_generator
    from supervised_field.nf_eval200 import ROOT
    from supervised_field.airfrans_p3_fields_v2 import case_params, analytic_polyline, NU_AIR
    from supervised_field.nf_extract_shape import resample_le, M
    from supervised_field.nf_eval_shape import encode_np
    from sdf2d.geometry import sdf_np
    PF, W, PROV = _PF, _W, _PROV
    U, a, prm = case_params(nm); Re = U / NU_AIR
    sim = af.Simulation(root=ROOT, name=nm)
    P = np.asarray(sim.position, np.float64)[:, :2]
    TR = np.column_stack([np.asarray(sim.velocity, np.float64), np.asarray(sim.pressure, np.float64).reshape(-1, 1),
                          np.asarray(sim.nu_t, np.float64).reshape(-1, 1)])
    sm = np.asarray(sim.surface, bool)
    poly = analytic_polyline(prm)
    d = sdf_np(P, poly)
    masks = dict(all=np.ones(len(P), bool), near_wall=(d > 0) & (d < 0.05), off_body=d > 0.05)
    std = W["std"]
    out = dict(case=nm, alpha=a, Re=Re)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        stE = PROV.state_for(prm, a, Re)                      # E: the deployed evaluation's inputs
        zE = encode_np(resample_le(P[sm], M, seed=0).astype(np.float64), W)
        predE = PF.decode(PF.field_inputs(P, poly, a, Re, stE["cl"], stE["s"], surf=sm), zE, W)
        out.update(cl_E=stE["cl"], s_E=stE["s"], E=_metrics(predE, TR, std, masks))
        coarse = np.asarray(naca_generator(np.array(prm), nb_samples=80, cosine_spacing=True, verbose=False, CTE=True), np.float64)
        prep = {}
        for tag, loop in (("U", poly), ("C", coarse)):
            c = prep[tag] = PF.prepare(loop, a, Re, family="naca")    # AirfRANS sections are NACA 4/5-digit
            PU = c["to_frame"](P)
            inp = PF.field_inputs(PU, c["poly"], c["alpha"], Re, c["cl"], c["s"], surf=sm)
            inside = (inp["d"] <= 0.0) & ~sm
            inp["sm"] = sm | inside                           # nodes the resampled wall swallowed: wall values, as surface nodes
            pr = PF.decode(inp, c["z"], W)
            ang = np.radians(c["rot"]); Rm = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
            pr[:, :2] = pr[:, :2] @ Rm                        # back to the AirfRANS frame (identity for these sections)
            out.update({f"cl_{tag}": c["cl"], f"s_{tag}": c["s"], f"rot_{tag}": c["rot"], f"inside_{tag}": int(inside.sum()),
                        f"fit_{tag}": c["meta"]["cst16_fit_max_error"], f"dz_{tag}": c["meta"]["shape_code_distance"],
                        f"dref": c["meta"]["shape_code_reference"], tag: _metrics(pr, TR, std, masks),
                        f"warn_{tag}": len(c["meta"]["warnings"])})
        # the sparse input on a user grid (0.01 chord over the default window, body frame), against the dense one:
        # the difference a user sees, where few points sit inside the first cells of the boundary layer
        Gx, Gy = PF.grid(PF.WINDOW, 0.01); Q = np.c_[Gx.ravel(), Gy.ravel()]
        gp = {}
        for tag in ("U", "C"):
            c = prep[tag]
            inp = PF.field_inputs(Q, c["poly"], c["alpha"], Re, c["cl"], c["s"])
            gp[tag] = (PF.decode(inp, c["z"], W), inp["d"] > 0)
        m = gp["U"][1] & gp["C"][1]
        out["grid_C_vs_U"] = [float(np.mean(((gp["C"][0][m, k] - gp["U"][0][m, k]) / std[k]) ** 2)) for k in range(4)]
    out["secs"] = time.time() - t0
    return out


def main():
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    nw, lim = int(arg("--workers", "6")), arg("--limit")
    rec = np.load(RECORD, allow_pickle=True)
    cases = [str(c) for c in rec["case"]]
    idx = list(range(len(cases)))[: int(lim)] if lim else list(range(len(cases)))
    t0 = time.time()
    with mp.get_context("spawn").Pool(nw, initializer=_init) as pool:
        rows = pool.map(_work, [cases[i] for i in idx], chunksize=1)
    D = rec["CM_all"][idx]
    A = {t: np.array([r[t]["all"] for r in rows]) for t in ("E", "U", "C")}
    alpha = np.array([r["alpha"] for r in rows])
    rel = lambda X, Y: float(np.max(np.abs(X - Y) / np.maximum(np.abs(Y), 1e-30)))
    ch = lambda v: dict(zip(FIELDS, [float(x) for x in v]))
    ratio = lambda t, m: ch(np.array([r[t][m] for r in rows]).mean(0) / np.array([r["E"][m] for r in rows]).mean(0))
    res = dict(schema="PREDICT_FIELD_VALIDATION_V1", utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
               script=os.path.relpath(__file__, VP.RELEASE_ROOT), script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
               predict_field_sha256=hashlib.sha256(open(os.path.join(HERE, "predict_field.py"), "rb").read()).hexdigest(),
               record=os.path.relpath(RECORD, VP.RELEASE_ROOT), n_cases=len(rows), seconds=round(time.time() - t0, 1),
               E_reproduces_record=dict(max_rel_diff_per_case_mse=rel(A["E"], D),
                                        max_abs_diff_cl=float(np.max(np.abs(np.array([r["cl_E"] for r in rows]) - rec["cl_provider"][idx]))),
                                        max_abs_diff_s=float(np.max(np.abs(np.array([r["s_E"] for r in rows]) - rec["s_max"][idx])))),
               mean_mse_all=dict(deployed_record=ch(D.mean(0)), E=ch(A["E"].mean(0)), U=ch(A["U"].mean(0)), C=ch(A["C"].mean(0))),
               U_over_deployed=ch(A["U"].mean(0) / D.mean(0)), C_over_deployed=ch(A["C"].mean(0) / D.mean(0)),
               by_mask_over_E={t: {m: ratio(t, m) for m in ("all", "near_wall", "off_body")} for t in ("U", "C")},
               per_case_ratio_over_deployed={t: dict(median=ch(np.median(A[t] / D, 0)), p90=ch(np.percentile(A[t] / D, 90, 0)),
                                                     max=ch((A[t] / D).max(0))) for t in ("U", "C")},
               by_alpha_band={}, routes={t: dict(max_abs_diff_cl_vs_E=float(max(abs(r[f"cl_{t}"] - r["cl_E"]) for r in rows)),
                                                 max_abs_diff_s_vs_E=float(max(abs(r[f"s_{t}"] - r["s_E"]) for r in rows)),
                                                 max_abs_rot_deg=float(max(abs(r[f"rot_{t}"]) for r in rows)),
                                                 max_cst16_fit_error=float(max(r[f"fit_{t}"] for r in rows)),
                                                 mesh_nodes_inside_resampled_wall=dict(cases=int(sum(r[f"inside_{t}"] > 0 for r in rows)),
                                                                                       max_nodes=int(max(r[f"inside_{t}"] for r in rows))),
                                                 shape_check_flagged=int(sum(r[f"dz_{t}"] > r["dref"] for r in rows)),
                                                 cases_with_warnings=int(sum(r[f"warn_{t}"] > 0 for r in rows)))
                                         for t in ("U", "C")},
               shape_check_reference=float(rows[0]["dref"]),
               user_grid_C_vs_U=dict(note="mean over cases of the standardized MSE between the sparse-input (C) and dense-input (U) "
                                          "predictions on a 0.01-chord grid over x -0.5..1.5, y -0.5..0.5 (chord frame), outside both sections; "
                                          "compare with the decoder's own mean test error on the RANS mesh (mean_mse_all.deployed_record)",
                                     mean_std_mse=ch(np.array([r["grid_C_vs_U"] for r in rows]).mean(0)),
                                     over_deployed_error=ch(np.array([r["grid_C_vs_U"] for r in rows]).mean(0) / D.mean(0))))
    for lo, hi in BANDS:
        m = (alpha >= lo) & (alpha < hi)
        if m.any():
            res["by_alpha_band"][f"[{lo},{hi})"] = dict(n=int(m.sum()), U_over_deployed=ch(A["U"][m].mean(0) / D[m].mean(0)),
                                                        C_over_deployed=ch(A["C"][m].mean(0) / D[m].mean(0)))
    out = os.path.join(VP.RESULTS, "predict_field_validation.json") if not lim else os.path.join(VP.RESULTS, f"_predict_field_validation_limit{lim}.json")
    json.dump(res, open(out, "w"), indent=1)
    if not lim:
        np.savez(os.path.join(VP.RESULTS, "predict_field_validation_per_case.npz"), case=np.array([r["case"] for r in rows]),
                 alpha=alpha, channels=np.array(FIELDS), deployed_all=D, E_all=A["E"], U_all=A["U"], C_all=A["C"],
                 **{f"{k}_{t}": np.array([r[f"{k}_{t}"] for r in rows]) for t in ("E", "U", "C") for k in ("cl", "s")},
                 **{f"{k}_{t}": np.array([r[f"{k}_{t}"] for r in rows]) for t in ("U", "C") for k in ("inside", "dz")})
    print(json.dumps({k: res[k] for k in ("n_cases", "seconds", "E_reproduces_record", "U_over_deployed", "C_over_deployed",
                                          "by_mask_over_E", "by_alpha_band", "routes", "user_grid_C_vs_U")}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
