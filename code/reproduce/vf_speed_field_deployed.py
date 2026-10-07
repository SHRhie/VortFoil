# -*- coding: utf-8 -*-
"""Cost of the field pathway as it is actually deployed, with and without the faster kernels.

Two things this measures that the earlier timing did not:

1. The deployed pathway computes the signed distance ONCE, inside the paint, and reuses it for the
   query features (`_case_inputs` takes `r["d"]` straight from the paint's return value). The
   earlier script timed a second, independent signed distance in its "features" stage, so its
   total counted work the deployed path never does. The nearest-panel index, by contrast, really
   is computed twice: once inside the paint and once in `_case_inputs`.

2. The same pathway with the kernels of `vf_field_fast`, which produce the same numbers (verified
   there against the originals) but form no three-dimensional temporaries and push every reduction
   over panels into a matrix product.

The decoder's own predictions are compared between the two configurations, because that, not the
intermediate field, is what the reported errors are computed from.

  python code/reproduce/vf_speed_field_deployed.py [--cases 3]
"""
import datetime
import json
import os
import platform
import statistics
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # <release>/code
import vortfoil_paths as VP; REPO = VP.add_paths()
ncase = int(sys.argv[sys.argv.index("--cases") + 1]) if "--cases" in sys.argv else 3
sys.argv = [sys.argv[0]]

import state_provider as G
import vf_field_fast as FAST
import supervised_field.airfrans_p3_fields_v2 as PAINT
from supervised_field.nf_eval200 import ROOT
from supervised_field.nf_hybrid_field import _case_inputs
from supervised_field.nf_extract import panel_mids_normals
from supervised_field.nf_eval_shape import encode_np, fwd_np
from supervised_field.nf_extract_shape import resample_le, M as N_TOK
import airfrans as af

ORIGINAL = dict(field_uv=PAINT.field_uv, sdf_chunked=PAINT.sdf_chunked, nearest_idx=PAINT.nearest_idx)


def use(which):
    if which == "original":
        for k, f in ORIGINAL.items():
            setattr(PAINT, k, f)
    else:
        FAST.install()


def med(fn, reps=3):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    load0 = os.getloadavg()
    G._init({0: os.path.join(VP.WEIGHTS, "field_decoder_f0cm_s1.npz")}, None)
    prov = G.ConfirmedStateProvider(s_rule="teH")
    W = G._W[0]
    cases = sorted(json.load(open(os.path.join(ROOT, "manifest.json")))["full_test"])[:ncase]
    rows = []
    for nm in cases:
        use("original")
        ci = _case_inputs(nm, prov, feature_mode="legacy_paint4")           # mesh I/O, not timed
        P = np.asarray(ci["P"], np.float64)
        poly, a, Re, cl = ci["poly"], float(ci["a"]), float(ci["Re"]), ci["cl_used"]
        prm = ci["prm"]
        sim = af.Simulation(root=ROOT, name=nm)
        xy = np.asarray(sim.position, np.float64)[np.asarray(sim.surface, bool)][:, :2]
        tok = resample_le(xy, N_TOK, seed=0).astype(np.float64)
        mid, nrm = panel_mids_normals(poly)
        X = ci["X"].astype(np.float64); s_row = ci["s_row"].astype(np.float64); PA = ci["PA"].astype(np.float64)
        z = encode_np(tok, W)
        t_state = med(lambda: prov.state_for(prm, a, Re), 10)
        t_enc = med(lambda: encode_np(tok, W), 20)
        pred = {}
        stage = {}
        for which in ("original", "fast"):
            use(which)
            decoder = (lambda: fwd_np(X, s_row, PA, z, W)) if which == "original" else \
                      (lambda: FAST.fwd_parallel(X, s_row, PA, z, W))
            t_dec = med(decoder, 3)
            t_paint = med(lambda: PAINT.paint_points_v2(P[:, :2], poly, a, cl, Re))
            t_near = med(lambda: PAINT.nearest_idx(P, mid))
            stage[which] = dict(provider_state_s=t_state, paint_s=t_paint, nearest_panel_s=t_near,
                                encoder_s=t_enc, decoder_s=t_dec,
                                total_s=t_state + t_paint + t_near + t_enc + t_dec)
            r = PAINT.paint_points_v2(P[:, :2], poly, a, cl, Re)
            pa = np.column_stack([r["u"], r["v"], r["cp"], r["nut"]]).astype(np.float64)
            pred[which] = np.asarray(fwd_np(X, s_row, pa, z, W) if which == "original"
                                     else FAST.fwd_parallel(X, s_row, pa, z, W), np.float64)
        d = np.abs(pred["fast"] - pred["original"])
        scale = np.maximum(np.abs(pred["original"]).max(axis=0), 1e-30)
        rows.append(dict(case=nm, n_points=int(len(P)), original=stage["original"], fast=stage["fast"],
                         speedup=stage["original"]["total_s"] / stage["fast"]["total_s"],
                         prediction_max_rel_diff=[float(x) for x in (d.max(axis=0) / scale)]))
        o, f = stage["original"], stage["fast"]
        print(f"{nm[:42]:42s} n={len(P):7d}")
        print(f"   paint   {o['paint_s']:6.2f} -> {f['paint_s']:6.2f} s      nearest {o['nearest_panel_s']:5.2f} -> {f['nearest_panel_s']:5.2f} s"
              f"      decoder {o['decoder_s']:5.2f} -> {f['decoder_s']:5.2f} s      state {1e3*o['provider_state_s']:5.1f} ms")
        print(f"   TOTAL   {o['total_s']:6.2f} -> {f['total_s']:6.2f} s   ({o['total_s']/f['total_s']:.2f}x)"
              f"   predictions agree to {max(rows[-1]['prediction_max_rel_diff']):.1e} relative")
    agg = {w: {k: statistics.median([r[w][k] for r in rows]) for k in rows[0]["original"]} for w in ("original", "fast")}
    out = dict(schema="VF_SPEED_FIELD_DEPLOYED_V1", utc=datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
               host=platform.platform(), cpu_count=os.cpu_count(), loadavg_start=load0, loadavg_end=os.getloadavg(),
               n_cases=len(rows), note=__doc__.strip().splitlines()[0], median=agg,
               speedup_total=agg["original"]["total_s"] / agg["fast"]["total_s"],
               prediction_agreement="the decoder's own outputs, max relative difference per channel: "
                                    + str([max(r["prediction_max_rel_diff"]) for r in rows]),
               rows=rows)
    od = VP.RESULTS; os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, f"_speed_field_deployed_f0cm_{out['utc']}.json")
    json.dump(out, open(fp, "w"), indent=1)
    print(f"\nmedian over {len(rows)} cases: {agg['original']['total_s']:.2f} s -> {agg['fast']['total_s']:.2f} s "
          f"({out['speedup_total']:.2f}x)")
    print("wrote", fp)


if __name__ == "__main__":
    main()
