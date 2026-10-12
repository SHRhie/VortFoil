# -*- coding: utf-8 -*-
"""Store full-field predictions of an operator-conditioned decoder for the figure cases (2026-09-19).
Writes npz files (prediction, label, query_xy, case_index, case_name, checkpoint_sha256, arm) that
code/figures/vf_field_f2_contour.py renders with --preds/--tag.  Same inputs as the evaluation worker
(state_provider._work, via nf_hybrid_field._case_inputs), provider with the maximum-side s rule the decoder was trained with
(state_provider.ConfirmedStateProvider(s_rule="teH")); nothing is recomputed differently.
Fig. 6 is a case named outside the selection file (--name: eval_<name>.npz, case_index -1), deployed run (run 1):
  python code/field/step2_store_preds.py --weights-tag f0cm --seed 1 --name airFoil2D_SST_39.31_8.883_4.707_0.0_12.552
  -> results/preds_f0cm_s1/eval_airFoil2D_SST_39.31_8.883_4.707_0.0_12.552.npz
"""
import os, sys, json, hashlib, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # <release>/code
import vortfoil_paths as VP; REPO = VP.add_paths()
sys.path.insert(0, HERE)
import state_provider as G

def main():
    tag = sys.argv[sys.argv.index("--weights-tag") + 1]; sd = int(sys.argv[sys.argv.index("--seed") + 1])
    # cases are named by results/fig_case_selection.json, whose case_index is NOT the sorted full_test position;
    # map by case_name and keep that index in the file name so vf_field_f2_contour.py finds eval_{index:04d}.npz
    SEL_JSON = os.environ.get("VORTFOIL_FIG_CASE_SELECTION", os.path.join(VP.RESULTS, "fig_case_selection.json"))
    sel = json.load(open(SEL_JSON))["selected_full_test_cases"]
    want = {int(ci): name for ci, name, _ in sel}
    if "--cases" in sys.argv:
        keep = [int(x) for x in sys.argv[sys.argv.index("--cases") + 1].split(",")]; want = {k: v for k, v in want.items() if k in keep}
    if "--name" in sys.argv:   # a case by name, in the selection file or not; it has no index there, so case_index -1 and the name in the file name
        want = {-1: sys.argv[sys.argv.index("--name") + 1]}
    from supervised_field.nf_eval200 import ROOT
    cases = sorted(json.load(open(os.path.join(ROOT, "manifest.json")))["full_test"])
    arm = f"{tag}_s{sd}"
    wf = os.path.join(VP.WEIGHTS, f"field_decoder_{tag}_s{sd}.npz")   # release names
    wsha = hashlib.sha256(open(wf, "rb").read()).hexdigest()
    G.SEEDS = {sd: os.path.basename(wf)}; G._init({sd: wf}, None); G._PROV = G.ConfirmedStateProvider(s_rule="teH"); G._SX = {}
    out_dir = os.path.join(VP.RESULTS, f"preds_{arm}"); os.makedirs(out_dir, exist_ok=True)
    import airfrans as af
    from supervised_field.nf_hybrid_field import _case_inputs
    from supervised_field.nf_extract_shape import resample_le, M
    from supervised_field.nf_eval_shape import encode_np, fwd_np
    for ci, nm in want.items():
        assert nm in cases, nm
        c = _case_inputs(nm, G._PROV, feature_mode="legacy_paint4"); P, TR, sm, PA = c["P"], c["TR"], c["sm"], c["PA"].astype(np.float64)
        sim = af.Simulation(root=ROOT, name=nm); xy = np.asarray(sim.position, np.float64)[np.asarray(sim.surface, bool)][:, :2]
        z = encode_np(resample_le(xy, M, seed=0).astype(np.float64), G._W[sd])
        pred = fwd_np(c["X"].astype(np.float64), c["s_row"].astype(np.float64), PA, z, G._W[sd]); pred[sm, 0] = 0; pred[sm, 1] = 0; pred[sm, 3] = 0
        assert not np.allclose(pred, TR)
        np.savez(os.path.join(out_dir, f"eval_{ci:04d}.npz" if ci >= 0 else f"eval_{nm}.npz"), prediction=pred.astype(np.float32), label=TR.astype(np.float64), query_xy=P[:, :2].astype(np.float32),
                 case_index=np.int32(ci), case_name=nm, checkpoint_sha256=wsha, arm=arm, cl_provider=np.float64(c["cl_used"]), s_provider=np.float64(c["s"]))
        print("stored", ci, nm, "cl %.3f s %.3f" % (c["cl_used"], c["s"]), flush=True)
    print("dir", out_dir)

if __name__ == "__main__":
    main()
