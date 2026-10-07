# -*- coding: utf-8 -*-
"""Step 2b (2026-09-18): evaluate operator-conditioned field decoders (trained on the operator-input corpus) on
AirfRANS full_test n=200 with the SAME provider at inference (train = test provider, XFOIL-free),
paired case-by-case against the sealed XFOIL-conditioned runs (XFOIL-input decoders, XFOIL-input evaluation).

Reuses state_provider verbatim (worker, metric kernel, pairing); only the decoder weights, the provider rule
(the maximum-side rule "teH" of the decoders' training rows, s = clip((max(H_TE,u, H_TE,l) - 2)/8, 0, 1)) and
the arm set (A only: the provider is the deployment condition) differ.  Nothing here touches the sealed runs.
results/step2_eval_f0cm_20260925T213247Z is this script's record for the three f0cm runs.

Pre-registration: primary = case-equal all-mesh standardized MSE per channel, R = provider decoder /
sealed XFOIL-conditioned run, paired case bootstrap B=10000 seed 0; the borrowed guardrail margins of Gate 0
(ux 1.15, p 1.20, nut 1.25; uy reported without a label).  MARIO's published numbers are positioning only.

  python code/field/step2_eval_f0c.py [--workers 6] [--limit N] [--weights-tag f0cm] [--seeds 0,1,2]
"""
import os, sys, json, time, datetime
import numpy as np
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # <release>/code
import vortfoil_paths as VP; REPO = VP.add_paths()
sys.path.insert(0, HERE)
import state_provider as G

MARIO = {"ux": 1.52e-3, "uy": 1.13e-3, "p": 2.40e-3, "nut": 9.60e-3}


def _init(seed_files, s_rule):
    G._init(seed_files, None)
    G._PROV = G.ConfirmedStateProvider(s_rule=s_rule)     # the training-row rule of the decoders; W = G.TEH_W (8.0)
    G._SX = {}                                             # arm D off: A only


def main():
    nw = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    tag = sys.argv[sys.argv.index("--weights-tag") + 1] if "--weights-tag" in sys.argv else "f0cm"
    rule = "teH"                                      # maximum side, the decoders' training-row rule
    if "--seeds" in sys.argv:                         # early read on a subset of seeds (e.g. --seeds 0 while 1,2 still train)
        keep = [int(x) for x in sys.argv[sys.argv.index("--seeds") + 1].split(",")]
        G.SEEDS = {sd: G.SEEDS[sd] for sd in keep}
    utc = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    OUT = os.path.join(VP.RESULTS, f"step2_eval_{tag}_{utc}"); os.makedirs(OUT, exist_ok=True)
    from supervised_field.nf_eval200 import ROOT
    sealed = {sd: np.load(os.path.join(G.EVAL_RUNS, G.SEALED[sd], "per_case_metrics.npz"), allow_pickle=True) for sd in G.SEEDS}
    cases = sorted(json.load(open(os.path.join(ROOT, "manifest.json")))["full_test"])
    assert list(sealed[0]["case"]) == cases
    if lim:
        cases = cases[:lim]
    seed_files = {sd: os.path.join(VP.WEIGHTS, f"field_decoder_{tag}_s{sd}.npz") for sd in G.SEEDS}   # release names
    for f in seed_files.values():
        assert os.path.exists(f), f
    print(f"[step2-eval] {len(cases)} cases, decoders {tag} s0/s1/s2, provider {rule} W={G.TEH_W}, workers={nw}", flush=True)
    t0 = time.time()
    with mp.Pool(nw, initializer=_init, initargs=(seed_files, rule)) as pool:
        rows = pool.map(G._work, cases, chunksize=2)
    ok = [r for r in rows if r["status"] == "ok"]; bad = [r for r in rows if r["status"] != "ok"]
    print(f"[step2-eval] done {len(ok)} ok / {len(bad)} error in {time.time()-t0:.0f}s", flush=True)
    for r in bad[:3]:
        print("  ERROR", r["case"], r["error"]); print(r.get("tb", "")[-600:])
    if not ok:
        sys.exit(1)
    # run_config recorded by the trainer inside each checkpoint (recipe proof)
    rc = {}
    for sd, f in seed_files.items():
        z = np.load(f, allow_pickle=True)
        rc[str(sd)] = json.loads(str(z["run_config"])) if "run_config" in z.files else {"run_config": "absent"}
    paired = {mk: G.paired(ok, sealed, mk, arm="A_op1cl_op2s", label=(mk == "all")) for mk in ("all", "near_wall", "off_body")}
    absolute = {mk: {ch: float(np.mean([np.mean([r["arms"]["A_op1cl_op2s"][str(sd)][mk][ch] for sd in G.SEEDS]) for r in ok])) for ch in G.FIELDS} for mk in ("all", "near_wall", "off_body")}
    per_seed_abs = {str(sd): {ch: float(np.mean([r["arms"]["A_op1cl_op2s"][str(sd)]["all"][ch] for r in ok])) for ch in G.FIELDS} for sd in G.SEEDS}
    # incidence-band breakdown (user request 2026-09-18): MARIO publishes no per-band numbers, so the bands carry the
    # provider decoder vs the sealed XFOIL-conditioned runs only, with MARIO's aggregate as the single reference line.
    BANDS = ((-6.0, 0.0), (0.0, 8.0), (8.0, 12.0), (12.0, 16.0))
    by_band = {}
    for lo, hi in BANDS:
        sub = [r for r in ok if lo <= r["alpha"] < hi]
        if len(sub) < 5:
            continue
        pb = G.paired(sub, sealed, "all", B=4000, arm="A_op1cl_op2s", label=False)
        by_band[f"[{lo:g},{hi:g})"] = dict(n=len(sub), **{ch: dict(f0a_xfoil=pb[ch]["mean_sealed_xfoil_input"], f0c_provider=pb[ch]["mean_confirmed_provider"],
                                                                  R=pb[ch]["R"], R_ci95=pb[ch]["R_ci95"], f0c_over_MARIO_aggregate=pb[ch]["mean_confirmed_provider"] / MARIO[ch]) for ch in G.FIELDS})
    result = dict(schema="STEP2_F0C_EVAL_V1", by_alpha_band=by_band, alpha_bands_note="MARIO publishes aggregate full_test numbers only; per-band ratios use that aggregate as the reference", utc=utc, script=os.path.relpath(__file__, REPO), script_sha256=G.sha(__file__), provider_script_sha256=G.sha(G.__file__),
                  decoders={str(sd): dict(path=os.path.relpath(f, REPO), sha256=G.sha(f), run_config=rc[str(sd)]) for sd, f in seed_files.items()},
                  op1=dict(path=os.path.relpath(G.OP1_W, REPO), sha256=G.sha(G.OP1_W)), op2=dict(path=os.path.relpath(G.OP2_W, REPO), sha256=G.sha(G.OP2_W)),
                  provider=dict(s_rule=rule, H0=G.TEH_H0, W=G.TEH_W, cl="OP1 readout 2*sum(gamma_panels*ds)"),
                  sealed_reference={str(sd): G.SEALED[sd] for sd in G.SEEDS}, split="full_test", n_cases=len(ok), n_error=len(bad),
                  preregistration=dict(primary="case-equal all-mesh standardized MSE; R = provider decoder / sealed XFOIL-conditioned run (XFOIL input); paired case bootstrap B=10000 seed 0",
                                       margins=G.MARGINS, margins_borrowed_from="f0_freeze P1 adjudication guardrails", label_on="all-mesh only"),
                  absolute_std_MSE_provider=absolute, per_seed_all=per_seed_abs,
                  ratio_to_MARIO_published={ch: absolute["all"][ch] / MARIO[ch] for ch in G.FIELDS}, mario_published=MARIO,
                  paired_vs_sealed_f0a=paired,
                  provider_vs_xfoil_diagnostic=dict(cl_R2=float(1 - np.sum((np.array([r["cl_used"] for r in ok]) - sealed[0]["cl_used"][[int(np.where(sealed[0]["case"] == r["case"])[0][0]) for r in ok]]) ** 2)
                                                             / np.sum((sealed[0]["cl_used"][[int(np.where(sealed[0]["case"] == r["case"])[0][0]) for r in ok]] - np.mean(sealed[0]["cl_used"])) ** 2)),
                                                    s_frac_nonzero=float(np.mean([r["s_val"] > 0 for r in ok])), s_frac_saturated=float(np.mean([r["s_val"] >= 1 for r in ok]))))
    json.dump(result, open(os.path.join(OUT, "result.json"), "w"), indent=1)
    np.savez(os.path.join(OUT, "per_case.npz"), case=np.array([r["case"] for r in ok]), cl_provider=np.array([r["cl_used"] for r in ok]), s_provider=np.array([r["s_val"] for r in ok]),
             **{f"mse_{mk}_{ch}_s{sd}": np.array([r["arms"]["A_op1cl_op2s"][str(sd)][mk][ch] for r in ok]) for mk in ("all", "near_wall", "off_body") for ch in G.FIELDS for sd in G.SEEDS})
    json.dump(rows, open(os.path.join(OUT, "rows.json"), "w"), default=str)
    print(f"[step2-eval] artifact {OUT}")
    print(f"{'mask':10s}{'ch':5s}{'XFOIL-cond. sealed':>19s}{'provider dec.':>15s}{'R':>8s}  {'CI95':>16s}  {'label':13s}{'vs MARIO':>9s}")
    for mk in ("all", "near_wall"):
        for ch in G.FIELDS:
            p = paired[mk][ch]
            print(f"{mk:10s}{ch:5s}{p['mean_sealed_xfoil_input']:19.3e}{p['mean_confirmed_provider']:15.3e}{p['R']:8.3f}  [{p['R_ci95'][0]:.3f},{p['R_ci95'][1]:.3f}]  {str(p['label']):13s}"
                  + (f"{absolute['all'][ch]/MARIO[ch]:8.2f}x" if mk == "all" else ""))
    print("per-seed all-mesh:", json.dumps(per_seed_abs))
    print(f"{'alpha band':12s}{'n':>4s}  " + "  ".join(f"{ch+' XFOIL':>10s}{ch+' prov':>10s}{'R':>7s}{'/MARIO':>8s}" for ch in G.FIELDS))
    for b, v in by_band.items():
        print(f"{b:12s}{v['n']:4d}  " + "  ".join(f"{v[ch]['f0a_xfoil']:10.2e}{v[ch]['f0c_provider']:10.2e}{v[ch]['R']:7.3f}{v[ch]['f0c_over_MARIO_aggregate']:7.2f}x" for ch in G.FIELDS))


if __name__ == "__main__":
    main()
