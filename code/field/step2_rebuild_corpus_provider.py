# -*- coding: utf-8 -*-
"""Step 2a (2026-09-18): rebuild the full-800 AirfRANS field training corpus with the operator state provider.

A decoder trained on XFOIL inputs does not take the operators' (cl, s) as a drop-in.  The fix is train/test
consistency: the SAME provider that is used at inference supplies the corpus inputs.  This script takes
supervised_field/_nf_train_v5_full800.npz (the XFOIL-input corpus) and replaces ONLY the two inference-time
inputs that came from XFOIL:
    PAINT  <- paint_points_v2 at the SAME stored rows, with cl = OP1 sheet readout (2*sum(gamma_panels*ds))
    s      <- OP2 boundary-layer operator, rule "teH": s = clip((max(H_TE,upper, H_TE,lower) - 2.0)/8.0, 0, 1)
Everything else is copied verbatim: X (geometry/condition features), TRUTH, case_id, g_band, names, U,
alpha, Re, params4, n_params, w_case / near_stall (LOSS WEIGHTS -- train-time only, allowed to keep their
XFOIL-era derivation; recorded), tokens file untouched.  The XFOIL columns cl_xtr / f_sep_xfoil are kept
under their names for reference and are NOT read by the trainer (it reads PAINT and s).

Verification built in: with the ORIGINAL cl_xtr the row-level recomputation reproduces the stored PAINT
bit-for-bit (checked on 3 cases before any rewrite; the 2026-09-18 spot check gave max|d| = 0 on all
four channels), so the only change in the corpus is the one intended.

Neither corpus is shipped (VORTFOIL_CORPUS_DIR points at them).  The operator-conditioned decoder was trained on
_nf_train_v5_full800_provider_maxs.npz, sha256 a387d918e81e2bbe035fbdac2d40eb678422be05089307a7dbc836b3cb51a7c7
(the corpus named in the run_config of weights/field_decoder_f0cm_s*.npz).

Provider = code/field/state_provider.ConfirmedStateProvider(s_rule="teH"):
OP1 op1_supervised_eps0.50.scr_bd128 + OP2 CANDIDATE_op2_m325_BT, CST16 fit of the NACA polyline.

  python code/field/step2_rebuild_corpus_provider.py [--workers 6]
"""
import os, sys, json, time, hashlib, datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # <release>/code
import vortfoil_paths as VP; REPO = VP.add_paths()
sys.path.insert(0, HERE)
# Training corpora live beside the supervised_field package in the research tree and are
# NOT shipped in this release (see README); VORTFOIL_CORPUS_DIR points at them.
SF = os.environ.get("VORTFOIL_CORPUS_DIR", os.path.join(VP.FIELD, "supervised_field"))
SRC = os.path.join(SF, "_nf_train_v5_full800.npz")
OUT = os.path.join(SF, "_nf_train_v5_full800_provider_maxs.npz")
S_RULE = "teH"           # maximum side, the operator-conditioned decoder's rule


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def paint_rows(args):
    """recompute the 4 paint channels at the stored rows of one case (numpy only; identical to nf_extract_v5)"""
    i, X, prm, a, Re, U, cl = args
    from supervised_field.airfrans_p3_fields_v2 import analytic_polyline, paint_points_v2
    poly = analytic_polyline(np.asarray(prm, np.float64))
    r = paint_points_v2(X[:, 0:2].astype(np.float64), poly, float(a), float(cl), float(Re))
    is_s = X[:, 5] > 0.5
    up = r["u"] * U; vp = r["v"] * U; pp = r["cp"] * 0.5 * U ** 2; nt = r["nut"] * U
    up[is_s] = 0.0; vp[is_s] = 0.0; nt[is_s] = 0.0
    return i, np.column_stack([up, vp, pp, nt]).astype(np.float32), float(r["vn_leak"]), float(np.abs(r["d"] - X[:, 2]).max())


def main():
    nw = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
    utc = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    ART = os.path.join(VP.RESULTS, f"step2_provider_corpus_{utc}"); os.makedirs(ART, exist_ok=True)
    z = np.load(SRC, mmap_mode="r", allow_pickle=True)
    N = len(z["names"]); cid = np.asarray(z["case_id"])
    prm4, npar, al, Re, U = np.asarray(z["params4"]), np.asarray(z["n_params"]), np.asarray(z["alpha"]), np.asarray(z["Re"]), np.asarray(z["U"])
    print(f"[step2] corpus {os.path.basename(SRC)}: {N} cases, {len(cid)} rows", flush=True)

    # ---- confirmed provider: cl and s for all 800 cases (parent process, batched) ----
    sys.argv = [sys.argv[0]]
    import state_provider as G
    prov = G.ConfirmedStateProvider(s_rule=S_RULE)
    cl_new = np.zeros(N); s_new = np.zeros(N); H_te = np.zeros(N); fit_err = np.zeros(N); rot = np.zeros(N)
    t0 = time.time()
    for i in range(N):
        st = prov.state_for(prm4[i, :npar[i]], al[i], Re[i])
        cl_new[i], s_new[i] = st["cl"], st["s"]; H_te[i] = prov.last["H_TE"]   # two-side mean (stored as H_te_max_op2, a historical name)
        fit_err[i] = prov.last["cst_fit_err"]; rot[i] = prov.last["frame_rot_deg"]
    print(f"[step2] provider states for {N} cases in {time.time()-t0:.0f}s | cl vs cl_xtr R2 "
          f"{1 - np.sum((cl_new - z['cl_xtr'])**2) / np.sum((z['cl_xtr'] - np.mean(z['cl_xtr']))**2):.3f} | "
          f"s quantiles {np.quantile(s_new, [0, .25, .5, .75, .9, 1]).round(3).tolist()} frac>0 {np.mean(s_new > 0):.3f} | "
          f"XFOIL s frac>0 {np.mean(np.asarray(z['s']) > 0):.3f}", flush=True)

    # ---- verification: original cl must reproduce the stored PAINT bit-for-bit on 3 cases ----
    chk = [0, N // 2, N - 1]
    for i in chk:
        rows = np.flatnonzero(cid == i)
        _, rec, _, dd = paint_rows((i, np.asarray(z["X"][rows]), prm4[i, :npar[i]], al[i], Re[i], U[i], float(z["cl_used"][i])))
        assert np.array_equal(rec, np.asarray(z["PAINT"][rows])), f"row-level recomputation != stored PAINT on case {i}"
        assert dd < 1e-5, dd
    print("[step2] verification: original cl reproduces stored PAINT bit-for-bit on cases", chk, flush=True)

    # ---- rebuild PAINT with the provider cl ----
    tasks = []
    for i in range(N):
        rows = np.flatnonzero(cid == i)
        tasks.append((i, np.asarray(z["X"][rows]), prm4[i, :npar[i]], al[i], Re[i], U[i], cl_new[i]))
    PA = np.array(z["PAINT"]).copy(); leak = np.zeros(N); t0 = time.time()
    with mp.Pool(nw) as pool:
        for k, (i, rec, vnl, dd) in enumerate(pool.imap_unordered(paint_rows, tasks, chunksize=4)):
            rows = np.flatnonzero(cid == i); PA[rows] = rec; leak[i] = vnl
            if (k + 1) % 100 == 0:
                print(f"  [{k+1}/{N}] {time.time()-t0:.0f}s", flush=True)
    changed = float(np.mean(np.any(PA != np.asarray(z["PAINT"]), axis=1)))
    print(f"[step2] PAINT rebuilt in {time.time()-t0:.0f}s; rows changed {changed:.3f}", flush=True)

    out = {k: np.asarray(z[k]) for k in z.files}
    out["PAINT"] = PA; out["s"] = s_new.astype(np.float32)
    out["cl_used"] = cl_new; out["xfoil_cl_ok"] = np.ones(N, bool)   # the trainer never reads these; kept coherent
    out["cl_op1"] = cl_new; out["s_teH"] = s_new.astype(np.float32); out["H_te_max_op2"] = H_te.astype(np.float32)   # holds the two-side MEAN despite its name
    out["vn_leak"] = leak.astype(np.float32); out["provider"] = np.array(json.dumps(dict(
        cl="OP1 readout 2*sum(gamma_panels*ds), op1_supervised_eps0.50.scr_bd128", s=f"OP2 {S_RULE}: clip((max-side H_TE-{G.TEH_H0})/{G.TEH_W},0,1), CANDIDATE_op2_m325_BT",
        kept_from_xfoil_era=["w_case", "near_stall (loss weights, train-time only)", "cl_xtr, f_sep_xfoil (reference columns, unread)"], source_corpus=os.path.basename(SRC))))
    np.savez_compressed(OUT, **out)
    prov_rec = dict(schema="STEP2_PROVIDER_CORPUS_V1", utc=utc, source=dict(path=os.path.relpath(SRC, REPO), sha256=sha(SRC)), output=dict(path=os.path.relpath(OUT, REPO), sha256=sha(OUT)),
                    script=os.path.relpath(__file__, REPO), script_sha256=sha(__file__), provider_script_sha256=sha(G.__file__),
                    op1=dict(path=os.path.relpath(G.OP1_W, REPO), sha256=sha(G.OP1_W)), op2=dict(path=os.path.relpath(G.OP2_W, REPO), sha256=sha(G.OP2_W)),
                    s_rule=dict(name=S_RULE, H0=G.TEH_H0, W=G.TEH_W), n_cases=int(N), n_rows=int(len(cid)), rows_changed_frac=changed,
                    cl_op1_vs_cl_xtr=dict(R2=float(1 - np.sum((cl_new - z["cl_xtr"])**2) / np.sum((z["cl_xtr"] - np.mean(z["cl_xtr"]))**2)), bias=float(np.mean(cl_new - z["cl_xtr"])),
                                          median_abs=float(np.median(np.abs(cl_new - z["cl_xtr"])))),
                    s_stats=dict(quantiles=np.quantile(s_new, [0, .25, .5, .75, .9, 1]).tolist(), frac_nonzero=float(np.mean(s_new > 0)), xfoil_frac_nonzero=float(np.mean(np.asarray(z["s"]) > 0)),
                                 corr_with_xfoil_s=float(np.corrcoef(s_new, np.asarray(z["s"], np.float64))[0, 1])),
                    cst_fit_err_max=float(fit_err.max()), frame_rot_deg_max=float(np.abs(rot).max()), verification="original cl reproduces stored PAINT bit-for-bit on cases " + str(chk),
                    unchanged_keys=[k for k in z.files if k not in ("PAINT", "s", "cl_used", "xfoil_cl_ok", "vn_leak")])
    json.dump(prov_rec, open(os.path.join(ART, "provenance.json"), "w"), indent=1)
    np.savez(os.path.join(ART, "case_states.npz"), names=out["names"], cl_op1=cl_new, cl_xtr=np.asarray(z["cl_xtr"]), s_teH=s_new, s_xfoil=np.asarray(z["s"]), H_te_max=H_te)
    print(f"[step2] wrote {OUT} ({os.path.getsize(OUT)/1e6:.0f} MB); provenance {ART}/provenance.json")


if __name__ == "__main__":
    main()
