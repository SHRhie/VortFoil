# -*- coding: utf-8 -*-
"""Inference cost of the CONFIRMED field pathway per case (2026-09-20): provider state (OP1 C_l + OP2 H_TE), physics
paint at the query points, feature assembly, shape-latent encoder, decoder forward.  As deployed: the decoder runs in
numpy (nf_eval_shape.fwd_np), the provider in eager TF on CPU.  Query points = the AirfRANS full_test case's own
mesh nodes (~179k), so the number is "one RANS field at CFD-mesh resolution"; per-1k-point cost is also given.
  python code/reproduce/vf_speed_field.py [--cases 3]
"""
import os, sys, json, time, datetime, statistics, platform
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # <release>/code
import vortfoil_paths as VP; REPO = VP.add_paths()
ncase = int(sys.argv[sys.argv.index("--cases") + 1]) if "--cases" in sys.argv else 3
sys.argv = [sys.argv[0]]
import state_provider as G
from supervised_field.nf_eval200 import ROOT
from supervised_field.nf_hybrid_field import _case_inputs
from supervised_field.airfrans_p3_fields_v2 import paint_points_v2, analytic_polyline, nearest_idx
from supervised_field.nf_extract import panel_mids_normals
from supervised_field.nf_eval_shape import encode_np, fwd_np
from supervised_field.nf_extract_shape import resample_le, M
from sdf2d.geometry import sdf_np
import airfrans as af


def med(fn, reps):
    fn(); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    load0 = os.getloadavg()
    G._init({0: os.path.join(VP.WEIGHTS, "field_decoder_f0cm_s1.npz")}, None); prov = G.ConfirmedStateProvider(s_rule="teH"); W = G._W[0]
    cases = sorted(json.load(open(os.path.join(ROOT, "manifest.json")))["full_test"])[:ncase]
    rows = []
    for nm in cases:
        ci = _case_inputs(nm, prov, feature_mode="legacy_paint4")          # also loads the mesh (I/O, not timed)
        P, poly, a, Re, U, prm = ci["P"], ci["poly"], float(ci["a"]), float(ci["Re"]), float(ci["U"]), ci["prm"]
        sim = af.Simulation(root=ROOT, name=nm); xy = np.asarray(sim.position, np.float64)[np.asarray(sim.surface, bool)][:, :2]
        t_state = med(lambda: prov.state_for(prm, a, Re), 10)                                  # OP1 C_l + OP2 H_TE (TF eager)
        cl = ci["cl_used"]
        t_paint = med(lambda: paint_points_v2(P[:, :2].astype(np.float64), poly, a, cl, Re), 3)  # physics paint at all nodes
        mid, nrm = panel_mids_normals(poly)
        t_feat = med(lambda: (nearest_idx(P, mid), sdf_np(P[:, :2], poly)), 3)                  # nearest panel + signed distance
        tok = resample_le(xy, M, seed=0).astype(np.float64)
        t_enc = med(lambda: encode_np(tok, W), 20)
        z = encode_np(tok, W); X = ci["X"].astype(np.float64); s_row = ci["s_row"].astype(np.float64); PA = ci["PA"].astype(np.float64)
        t_dec = med(lambda: fwd_np(X, s_row, PA, z, W), 5)
        n = len(P); tot = t_state + t_paint + t_feat + t_enc + t_dec
        rows.append(dict(case=nm, n_points=int(n), provider_state_s=t_state, paint_s=t_paint, features_s=t_feat, encoder_s=t_enc, decoder_s=t_dec, total_s=tot,
                         decoder_ms_per_1k_points=1e3 * t_dec / n * 1e3, total_ms_per_1k_points=1e3 * tot / n * 1e3))
        print(f"{nm[:40]:40s} n={n:7d}  state {t_state*1e3:7.1f} ms  paint {t_paint:6.2f} s  feat {t_feat:6.2f} s  enc {t_enc*1e3:6.1f} ms  dec {t_dec:6.2f} s  total {tot:6.2f} s", flush=True)
    agg = {k: statistics.median([r[k] for r in rows]) for k in ("provider_state_s", "paint_s", "features_s", "encoder_s", "decoder_s", "total_s", "decoder_ms_per_1k_points", "total_ms_per_1k_points")}
    out = dict(schema="VF_SPEED_FIELD_V1", utc=datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"), host=platform.platform(), cpu_count=os.cpu_count(),
               decoder="numpy fwd_np (as evaluated), weights field_decoder_f0cm_s1.npz", provider="TF eager CPU, OP1 scr_bd128 + OP2 m325_BT", loadavg_start=load0, loadavg_end=os.getloadavg(),
               n_cases=len(rows), median=agg, rows=rows)
    od = VP.RESULTS; os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, f"_speed_confirmed_field_f0cm_{out['utc']}.json"); json.dump(out, open(fp, "w"), indent=1); print("wrote", fp); print(json.dumps(agg))


if __name__ == "__main__":
    main()
