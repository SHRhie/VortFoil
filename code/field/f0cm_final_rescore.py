# -*- coding: utf-8 -*-
"""Operator-conditioned field decoder (2026-10-02): every manuscript field number that uses it, recomputed for the deployed
weight file weights/field_decoder_f0cm_s1.npz, which is trained AND scored with
s = clip((max(H_TE,upper, H_TE,lower) - 2)/8, 0, 1) = state_provider.ConfirmedStateProvider(s_rule="teH").

Arms (AirfRANS full_test, 200 cases, sorted manifest order; harness and scorer of state_provider, bootstrap R.paired):
  CM   f0cm_s1, provider cl + max-side s                         (must reproduce step2_eval_f0cm_20260925T213247Z s1, <=1e-6 rel)
  DB   drop-in "both" on the XFOIL-conditioned run 2 (f0a_s2): provider cl + max-side s
  DL   drop-in "lift only" on run 2: provider cl + XFOIL s
  CX   f0cm_s1, XFOIL cl + max-side s                            (C_l substitution, diagnostic)
  CB   f0cm_s1, (provider cl - 0.0294) + max-side s              (bias-corrected, diagnostic)
  CMm  f0cm_s1, provider cl + mean-side s                        (evaluation-rule sensitivity)
Comparator: the sealed per-case array of the XFOIL-conditioned run 2 (evaluation_runs/, mse_all_<channel>).
Every array is also checked against the shipped record results/f0cm_final_rescore_20261002T051450Z/per_case.npz (<=1e-6 rel).
--limit N scores the first N cases, runs the checks on rows 0..N-1 and writes nothing.
  python code/field/f0cm_final_rescore.py [--workers 8] [--limit N]          (needs AIRFRANS_ROOT)
"""
import os, sys, json, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s_rule_meanside_rescore as R          # imports the provider module first (pins BLAS / CUDA before numpy)
import numpy as np
import multiprocessing as mp

VP, G, DG = R.VP, R.G, R.DG
REPO = VP.RELEASE_ROOT
CM_W, F0A2_W = os.path.join(VP.WEIGHTS, "field_decoder_f0cm_s1.npz"), os.path.join(VP.WEIGHTS, "field_decoder_f0a_s2.npz")
CM_SHA = "5e72da181372c9b966f2b306c32f9a92aaa069704979646ed713726718568c33"
STEP2 = os.path.join(VP.RESULTS, "step2_eval_f0cm_20260925T213247Z", "per_case.npz")
RUN2 = os.path.join(G.EVAL_RUNS, G.SEALED[2], "per_case_metrics.npz")
SHIPPED = os.path.join(VP.RESULTS, "f0cm_final_rescore_20261002T051450Z", "per_case.npz")
S_MAX = "clip((max(H_TE,upper, H_TE,lower) - 2)/8, 0, 1) = state_provider.ConfirmedStateProvider(s_rule='teH')"


def _init(files, sx, mode, clx, side):
    G._init(files, sx)
    DG._CLX = clx
    base = G.ConfirmedStateProvider(s_rule="teH_mean" if side == "mean" else "teH")
    G._PROV = base if mode == "none" else DG.ClOverrideProvider(base, mode)


def run(cases, files, sx, mode, clx, side, nw):
    with mp.Pool(nw, initializer=_init, initargs=(files, sx, mode, clx, side)) as pool:
        rows = pool.map(G._work, cases, chunksize=2)
    bad = [r for r in rows if r["status"] != "ok"]
    for r in bad[:3]:
        print("  ERROR", r["case"], r["error"], r.get("tb", "")[-600:])
    assert not bad and [r["case"] for r in rows] == cases
    return rows


def maxrel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64); d = np.abs(a - b)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.max(np.where(d == 0, 0.0, d / np.abs(b))))


def main():
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    nw = int(arg("--workers", "8")); lim = int(arg("--limit", "0"))
    assert G.sha(CM_W) == CM_SHA
    from supervised_field.nf_eval200 import ROOT
    cases = sorted(json.load(open(os.path.join(ROOT, "manifest.json")))["full_test"])[: lim or None]
    sealed = np.load(os.path.join(G.EVAL_RUNS, G.SEALED[0], "per_case_metrics.npz"), allow_pickle=True)
    r2 = np.load(RUN2, allow_pickle=True)
    assert np.array_equal(r2["s_val"], sealed["s_val"])                        # XFOIL s: the same for runs 0 and 2
    sx = {str(c): float(v) for c, v in zip(sealed["case"], sealed["s_val"])}
    clx = {}
    for i in range(len(sealed["case"])):
        prm = sealed["prm4"][i][:int(sealed["n_prm"][i])]
        clx[(tuple(np.round(np.asarray(prm, np.float64), 4)), round(float(sealed["alpha"][i]), 3), round(float(sealed["Re"][i]) / 1e3))] = float(sealed["cl_used"][i])
    i2 = {str(c): i for i, c in enumerate(r2["case"])}
    X2 = np.column_stack([r2[f"mse_all_{c}"][[i2[n] for n in cases]] for c in G.FIELDS])   # sealed run 2, sorted case order
    st = np.load(STEP2, allow_pickle=True); assert list(st["case"])[:len(cases)] == cases
    CM_ref = np.column_stack([st[f"mse_all_{c}_s1"] for c in G.FIELDS])[:len(cases)]

    P0 = run(cases, {0: F0A2_W, 1: CM_W}, sx, "none", clx, "max", nw); print("[P0 max-side: CM, DB, DL] ok", flush=True)
    P1 = run(cases, {0: CM_W}, {}, "xfoil", clx, "max", nw); print("[P1 CX] ok", flush=True)
    P2 = run(cases, {0: CM_W}, {}, "bias", clx, "max", nw); print("[P2 CB] ok", flush=True)
    P3 = run(cases, {0: CM_W}, {}, "none", clx, "mean", nw); print("[P3 CMm] ok", flush=True)
    A = dict(CM=R.arr(P0, "A_op1cl_op2s", 1, "all"), DB=R.arr(P0, "A_op1cl_op2s", 0, "all"), DL=R.arr(P0, "D_op1cl_xfoils", 0, "all"),
             CX=R.arr(P1, "A_op1cl_op2s", 0, "all"), CB=R.arr(P2, "A_op1cl_op2s", 0, "all"), CMm=R.arr(P3, "A_op1cl_op2s", 0, "all"))
    alpha = np.array([r["alpha"] for r in P0])
    arrays = dict(case=np.array(cases), alpha=alpha, channels=np.array(G.FIELDS), xfoil_conditioned_run2_sealed_all=X2,
                  cl_provider=np.array([r["cl_used"] for r in P0]), s_max=np.array([r["s_val"] for r in P0]), s_mean=np.array([r["s_val"] for r in P3]),
                  cl_xfoil=np.array([r["cl_used"] for r in P1]), s_xfoil=np.array([sx[c] for c in cases]), **{f"{k}_all": v for k, v in A.items()})
    n = len(cases); sh = np.load(SHIPPED, allow_pickle=True)
    assert [str(c) for c in sh["case"][:n]] == cases and [str(c) for c in sh["channels"]] == list(G.FIELDS)
    vs_shipped = {k: maxrel(v, sh[k][:n]) for k, v in arrays.items() if k not in ("case", "channels")}
    checks = dict(CM_vs_step2_eval_f0cm_s1_maxrel=maxrel(A["CM"], CM_ref), vs_shipped_record_maxrel=vs_shipped,
                  CMm_s_is_mean_side_of_P0=bool(np.allclose([r["s_val"] for r in P3], [np.clip((r["diag"]["H_TE"] - 2) / 8, 0, 1) for r in P0])))
    print(f"[checks] rows 0..{n - 1}: CM vs step2_eval_f0cm s1 {checks['CM_vs_step2_eval_f0cm_s1_maxrel']:.1e}; vs the shipped record "
          + ", ".join(f"{k} {v:.1e}" for k, v in vs_shipped.items()), flush=True)
    assert checks["CM_vs_step2_eval_f0cm_s1_maxrel"] <= 1e-6 and max(vs_shipped.values()) <= 1e-6 and checks["CMm_s_is_mean_side_of_P0"], checks
    if lim:
        print(f"[limit {n}] rows 0..{n - 1} of {os.path.relpath(SHIPPED, REPO)} reproduced, max relative difference "
              f"{max(vs_shipped.values()):.2e}; nothing written", flush=True)
        return

    mean = lambda v: dict(zip(G.FIELDS, v.mean(0).tolist()))
    vsM = lambda v: {c: float(v[:, j].mean() / R.MARIO[c]) for j, c in enumerate(G.FIELDS)}
    utc = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    res = dict(schema="F0CM_FINAL_RESCORE_V1", utc=utc, script=os.path.relpath(__file__, REPO), script_sha256=G.sha(__file__),
               provider_module_sha256=G.sha(G.__file__), n_cases=len(cases), split="AirfRANS full_test, sorted manifest order",
               provider_rule=S_MAX, cl_rule="OP1 readout C_l^gamma = 2*sum(gamma*ds)", cl_bias_CB=DG.CL_BIAS,
               weights=dict(f0cm_s1=dict(file=os.path.basename(CM_W), sha256=G.sha(CM_W)), f0a_s2=dict(file=os.path.basename(F0A2_W), sha256=G.sha(F0A2_W))),
               comparator=dict(name="XFOIL-conditioned decoder run 2, sealed", file=os.path.relpath(RUN2, REPO), key="mse_all_<channel>"),
               mario_published=R.MARIO, bootstrap="R.paired: case bootstrap B=10000, rng default_rng(0), channels in order", checks=checks,
               arms=dict(CM="f0cm_s1, provider cl + max-side s", DB="run 2, provider cl + max-side s", DL="run 2, provider cl + XFOIL s",
                         CX="f0cm_s1, XFOIL cl + max-side s", CB="f0cm_s1, provider cl - 0.0294 + max-side s", CMm="f0cm_s1, provider cl + mean-side s"),
               mean={k: mean(v) for k, v in A.items()} | dict(XF2=mean(X2)), vs_MARIO={k: vsM(v) for k, v in A.items()} | dict(XF2=vsM(X2)),
               paired_vs_xfoil_conditioned_run2={k: R.paired(A[k], X2) for k in ("CM", "DB", "DL")},
               cl_substitution_vs_CM=dict(xfoil_cl=R.paired(A["CX"], A["CM"]), bias_corrected_cl=R.paired(A["CB"], A["CM"])),
               eval_rule_sensitivity_CMm_over_CM=R.paired(A["CMm"], A["CM"]), by_alpha_band={})
    for lo, hi in R.BANDS:
        m = (alpha >= lo) & (alpha < hi)
        res["by_alpha_band"][f"[{lo},{hi})"] = dict(
            n=int(m.sum()), abs={k: mean(v[m]) for k, v in A.items()} | dict(XF2=mean(X2[m])),
            over_xfoil_conditioned_run2={k: dict(zip(G.FIELDS, (v[m].mean(0) / X2[m].mean(0)).tolist())) for k, v in A.items()},
            over_MARIO_aggregate={k: vsM(v[m]) for k, v in A.items()} | dict(XF2=vsM(X2[m])))
    OUT = os.path.join(VP.RESULTS, f"f0cm_final_rescore_{utc}"); assert not os.path.exists(OUT); os.makedirs(OUT)
    json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1)
    np.savez(os.path.join(OUT, "per_case.npz"), **arrays)
    f = lambda d: " ".join(f"{c} {d[c]['R']:.2f}[{d[c]['ci95'][0]:.2f},{d[c]['ci95'][1]:.2f}] n{d[c]['n_improved']}" for c in G.FIELDS)
    for k in A:
        print(f"[{k}] mean " + " ".join(f"{c} {v:.3e}" for c, v in res["mean"][k].items()) + " | /MARIO " + " ".join(f"{v:.2f}" for v in res["vs_MARIO"][k].values()), flush=True)
    for k, d in res["paired_vs_xfoil_conditioned_run2"].items():
        print(f"[{k}/XF2] {f(d)}", flush=True)
    for k, d in res["cl_substitution_vs_CM"].items():
        print(f"[{k}/CM] {f(d)}", flush=True)
    print(f"[CMm/CM] {f(res['eval_rule_sensitivity_CMm_over_CM'])}", flush=True)
    print(f"[rescore] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
