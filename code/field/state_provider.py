# -*- coding: utf-8 -*-
"""Gate 0 (2026-09-18): the first contract-compliant field state provider, built from the CONFIRMED
surface model, evaluated on the sealed f0a field decoders WITHOUT retraining.

Question. The field decoder's two XFOIL-at-inference inputs are the paint circulation scale `cl`
and the separation scalar `s`.  Since the 2026-09-05 refactor (31cd352) nf_hybrid_field._case_inputs
requires a non-XFOIL `state_provider` and "a compliant state_provider does not exist in this
repository".  This script supplies one from the single model the paper describes:
    cl  = 2 * sum(gamma_panels * ds)      -- readout of the OP1 sheet (op1_supervised_eps0.50.scr_bd128)
    s   = clip(tail(H > 4.0)/0.10, 0, 1)   -- from the OP2 boundary-layer operator (CANDIDATE m325_BT): the
          contiguous H>4 run back from the TE, mean of the two surfaces (rule calibrated on TRAIN, see below)
and re-evaluates the three sealed f0a decoders (seeds 0,1,2; AirfRANS full_test n=200) with every
other input bit-identical to the sealed runs (legacy_paint4 feature route, same weights loader,
same metric kernel expression, same masks, same tok_seed=0).

Pre-registration (written before the run):
  * primary metric = per-case all-mesh standardized MSE per channel, case-equal mean over the 200
    cases, ratio R = mean(confirmed-provider) / mean(sealed XFOIL-input run), per seed and pooled
    over seeds; paired case bootstrap (B=10000, seed 0) for the CI.
  * guardrail margins are BORROWED from the frozen P1 adjudication (f0_freeze: ux 1.15, p 1.20,
    nut 1.25; uy carried no non-inferiority margin there and is reported without a label).
    label NON_INFERIOR if R_ci95_upper < margin; DEGRADED if R_ci95_lower > margin; else INCONCLUSIVE.
  * s rule: the 2-case smoke test showed frac(H>3) over all nodes saturating s=1 on attached cases
    (OP2's laminar/transitional H of 3.5-4 is not separation).  The rule was therefore CALIBRATED ON
    THE 800 TRAIN CASES ONLY (gate0_calibrate_s_proxy.py; XFOIL s is stored per train case in
    _nf_train_v5_full800.npz): selected = contiguous H>4.0 run back from the TE ("tail_h4.0"),
    MAE 0.121 against an all-zero baseline of 0.126, i.e. the OP2 shape factor is a WEAK surrogate
    for XFOIL's Cf<0 fraction and the gate is expected to pay for it.  Two diagnostic arms isolate
    that: B = s=0 (no separation input) and D = s from the sealed run's XFOIL value (cl-only
    substitution; NOT contract-compliant, diagnostic only).  Labels are attached to arm A only.
  * the sealed runs' XFOIL cl is a FORCED-TRANSITION (XTR 0.05/0.05, fully turbulent) lift matching
    the RANS state; OP1's readout is a free-transition lift, so a bias is expected (train: +0.029).
  * the OP1 cl and OP2 s are also compared with the sealed runs' recorded XFOIL cl_used / s_val on
    the same 200 cases; that is a diagnostic, not a selection criterion.
  * AirfRANS Re = U/nu spans ~0.7e6-4.2e6; the surface model's deck tops out at 3e6, so the upper
    part is a mild Re extrapolation for the provider.  Reported, not corrected.

Run (tf218 has airfrans + TF 2.18; the clone imports under it):
  python code/field/state_provider.py [--workers 6] [--limit N]   (needs TF + the airfrans package)
"""
import os, sys, json, time, hashlib, datetime, glob
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"          # pinned to 1 as in the sealed runs (BLAS reduction order)
import numpy as np
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # <release>/code, for vortfoil_paths
import vortfoil_paths as VP                                                   # noqa: E402
REPO = VP.add_paths()                              # release root; puts code/model + code/field on sys.path
sys.path.insert(0, HERE)                           # cst16_fit.py sits beside this file

OP1_W = os.path.join(VP.WEIGHTS, "op1_supervised_eps0.50.scr_bd128.npz")
OP2_W = os.path.join(VP.WEIGHTS, "op2_m325_BT.npz")   # CANDIDATE_op2_m325_BT.npz in the research tree
# SEEDS / SEALED name the three f0a field decoders (XFOIL-conditioned; run 2 is the deployed one) in weights/
# and their sealed evaluation runs in evaluation_runs/ (VORTFOIL_WEIGHTS / VORTFOIL_EVAL_RUNS override both).
SEEDS = {0: "field_decoder_f0a_s0.npz", 1: "field_decoder_f0a_s1.npz", 2: "field_decoder_f0a_s2.npz"}
SEALED = {0: "_nf_shape_f0a_s0__full_test__n200__verify__20260903T032946Z",
          1: "_nf_shape_f0a_s1__full_test__n200__verify__20260903T034939Z",
          2: "_nf_shape_f0a_s2__full_test__n200__verify__20260903T040834Z"}
EVAL_RUNS = os.environ.get("VORTFOIL_EVAL_RUNS", os.path.join(REPO, "evaluation_runs"))
H_SEP = 4.0          # tail rule threshold, SELECTED ON TRAIN (gate0_calibrate_s_proxy.py -> tail_h4.0, MAE 0.121 vs all-zero 0.126)
H_SENS = (3.5, 3.0)  # sensitivity only
S_RULE = "tail"      # s = clip(mean over sides of (contiguous H>H_SEP run back from the TE)/120 / 0.10, 0, 1)
CALIB = "results/gate0_calibration.json"   # VortFoil/artifacts/gate0_calibration/calibration.json in the research tree
TEH_H0, TEH_W = 2.0, 8.0   # s = clip((H_TE - 2.0)/8.0, 0, 1) -- H_TE 2 -> 0, 10 -> 1; W chosen on the 800 AirfRANS training cases only
                           # "teH": H_TE = max of the two per-surface values (manuscript Eq. 15): the operator-conditioned
                           #        decoder's training rows and every evaluation of it
                           # "teH_mean": H_TE = mean of the two values: only the evaluation-rule sensitivity of f0cm_final_rescore.py
MARGINS = {"ux": 1.15, "p": 1.20, "nut": 1.25}        # borrowed from the frozen P1 adjudication
FIELDS = ("ux", "uy", "p", "nut")
NU_AIR = 1.56e-5   # AirfRANS air kinematic viscosity, as nf_hybrid_field (Re = U / NU_AIR)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ----------------------------------------------------------------------------- provider
class ConfirmedStateProvider:
    """cl from the OP1 sheet readout, s from the OP2 boundary-layer operator.  Fails closed."""

    def __init__(self, h_sep=H_SEP, s_rule=S_RULE):
        import tensorflow as tf
        try:          # possible only before TensorFlow initializes (e.g. not after vortfoil.load_model); then its settings stay
            tf.config.set_visible_devices([], "GPU")
            tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:
            pass
        from staged.op1_inviscid import build_op1
        from staged.op2_bl_surrogate import build_op2_surrogate
        from staged.common.geometry import build_single_loop_contour_cst_tf
        from cst16_fit import basis
        self.tf, self.basis, self.loop = tf, basis, build_single_loop_contour_cst_tf
        DT = tf.float32
        z2 = np.zeros((2, 16), np.float32); a2 = np.zeros((2, 1), np.float32); r2 = np.full((2, 1), 1e6, np.float32)
        self.op1 = build_op1(n_surface_pts=121, branch_dim=128)
        self.op1.compute_losses(tf.constant(z2), tf.constant(a2), tf.constant(r2), training=False)
        ck = np.load(OP1_W); k = sorted(x for x in ck.files if x.startswith("v_"))
        self.op1.field_decoder.build((None, int(ck[k[-6]].shape[0])))
        self.op1.load_weights(OP1_W)
        self.op2 = build_op2_surrogate(n_surface_pts=121, enable_wake=False, branch_dim=128, branch_depth=3, trunk_depth=2, N_m=16)
        self.op2.predict_bl(tf.constant(z2), tf.constant(a2), tf.constant(r2), training=False)
        self.op2.load_weights(OP2_W, strict=False)
        self.h_sep = float(h_sep)
        self.s_rule = s_rule          # "tail" (Gate 0, calibrated on train), "teH_mean" or "teH" (continuous TE shape factor, see TEH_W)
        self.last = None
        from supervised_field.airfrans_p3_fields_v2 import analytic_polyline
        self.polyline = analytic_polyline

    def _cst(self, poly):
        g = np.asarray(poly, np.float64)
        if np.allclose(g[0], g[-1]):
            g = g[:-1]
        te = 0.5 * (g[0] + g[-1]); d = np.hypot(*(g - te).T); li = int(np.argmax(d)); le = g[li]
        c = te - le; ch = np.hypot(*c); ang = np.arctan2(c[1], c[0])
        Rm = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
        gn = ((g - le) @ Rm.T) / ch
        xu, yu = gn[li::-1, 0], gn[li::-1, 1]; xl, yl = gn[li:, 0], gn[li:, 1]
        if yu.mean() < yl.mean():                       # first leg must be the upper surface
            xu, yu, xl, yl = xl, yl, xu, yu
        Pu, Pl = self.basis(xu), self.basis(xl)
        au = np.linalg.lstsq(Pu, yu, rcond=None)[0]; al = np.linalg.lstsq(Pl, -yl, rcond=None)[0]
        err = float(max(np.abs(Pu @ au - yu).max(), np.abs(-(Pl @ al) - yl).max()))
        return np.r_[au, al].astype(np.float32), float(np.degrees(ang)), err

    def state_for(self, prm, alpha_deg, Re):
        return self.state_for_polygon(self.polyline(np.asarray(prm, np.float64)), alpha_deg, Re, source="NACA polyline")

    def state_for_polygon(self, poly, alpha_deg, Re, source="polygon"):
        """The same state for any closed section polygon (code/field/predict_field.py); state_for passes the NACA one."""
        tf = self.tf
        cst, rot, err = self._cst(poly)
        a_eff = float(alpha_deg) - rot
        C = tf.constant(cst[None]); A = tf.constant(np.array([[a_eff]], np.float32)); R = tf.constant(np.array([[float(Re)]], np.float32))
        g = self.op1.predict_gamma(C, A, Re=R, training=False)
        geom = self.loop(C, n_surface_pts=121)
        cl = float(2.0 * tf.reduce_sum(g["gamma_panels"] * geom["ds"], axis=1).numpy()[0])
        bl = self.op2.predict_bl(C, A, R, training=False)
        th = np.r_[bl["theta_u"].numpy()[0], bl["theta_l"].numpy()[0]]
        ds = np.r_[bl["delta_u"].numpy()[0], bl["delta_l"].numpy()[0]]
        H = ds / np.maximum(th, 1e-9)
        def tail(Hs, h):                                 # contiguous separated run back from the TE, fraction of the surface
            k = 0
            while k < 121 and Hs[120 - k] > h: k += 1
            return k / 120.0
        fsep = {h: 0.5 * (tail(H[:121], h) + tail(H[121:], h)) for h in (self.h_sep,) + H_SENS}
        H_te_max = float(max(H[120], H[-1]))
        H_te_mean = float(H[120] + H[-1]) / 2.0
        if self.s_rule == "teH_mean":
            s = float(np.clip((H_te_mean - TEH_H0) / TEH_W, 0.0, 1.0))     # mean side (evaluation-rule sensitivity only)
        elif self.s_rule == "teH":
            s = float(np.clip((H_te_max - TEH_H0) / TEH_W, 0.0, 1.0))      # max side, manuscript Eq. 15
        else:
            s = float(np.clip(fsep[self.h_sep] / 0.10, 0.0, 1.0))
        self.last = dict(cst_fit_err=err, frame_rot_deg=rot, a_eff=a_eff, H_TE=H_te_mean,
                         fsep={str(k): v for k, v in fsep.items()},
                         s_sens={str(h): float(np.clip(fsep[h] / 0.10, 0, 1)) for h in H_SENS})
        return {"cl": cl, "s": s,
                "provenance": {"cl_source": "gamma_eff_circulation_functional",
                               "s_source": "learned_viscous_state",
                               "external_solver_calls": 0,
                               "detail": f"cl = 2*sum(gamma_panels*ds) of OP1 {os.path.basename(OP1_W)}; "
                                         f"s rule {self.s_rule} (tail: clip(tail(H>{self.h_sep})/0.10,0,1); teH_mean: clip((H_TE,mean-{TEH_H0})/{TEH_W},0,1); teH: clip((H_TE,max-{TEH_H0})/{TEH_W},0,1)) of OP2 {os.path.basename(OP2_W)}; "
                                         f"CST16 fit of the {source} (max err {err:.2e}), alpha_eff = alpha - frame rot"}}


# ----------------------------------------------------------------------------- worker
_W = {}; _STD = None; _PROV = None; _SX = {}


def _init(seed_files, sx=None):
    global _W, _STD, _PROV, _SX
    _SX = sx or {}
    import airfrans  # noqa
    for sd, f in seed_files.items():
        z = np.load(f, allow_pickle=True)
        _W[sd] = {k: np.asarray(v, np.float64) for k, v in z.items() if np.asarray(v).dtype.kind in "fiu"}
    _STD = next(iter(_W.values()))["std"]          # any decoder: a single seed other than 0 must not fail
    _PROV = ConfirmedStateProvider()


def case_metrics(pred, TR, std, masks):
    return {mk: {k: float(np.mean(((pred[mm, c] - TR[mm, c]) / std[c]) ** 2))
                 for c, k in enumerate(FIELDS)} for mk, mm in masks.items()}


def _work(nm):
    t0 = time.time()
    try:
        import airfrans as af
        from supervised_field.nf_hybrid_field import _case_inputs
        from supervised_field.nf_eval200 import ROOT
        from supervised_field.nf_extract_shape import resample_le, M
        from supervised_field.nf_eval_shape import encode_np, fwd_np
        from sdf2d.geometry import sdf_np
        ci = _case_inputs(nm, _PROV, feature_mode="legacy_paint4")
        P = ci["P"]; TR = ci["TR"]; sm = ci["sm"]; PA = ci["PA"].astype(np.float64)
        sim = af.Simulation(root=ROOT, name=nm)
        xy = np.asarray(sim.position, np.float64)[np.asarray(sim.surface, bool)][:, :2]
        tok = resample_le(xy, M, seed=0)
        d = sdf_np(P[:, :2], ci["poly"])
        masks = dict(all=np.ones(len(P), bool), near_wall=(d > 0) & (d < 0.05), off_body=d > 0.05)
        out = dict(case=nm, status="ok", alpha=float(ci["a"]), U_inf=float(ci["U"]), Re=float(ci["Re"]),
                   cl_used=float(ci["cl_used"]), s_val=float(ci["s"]), diag=dict(_PROV.last), metrics={}, arms={}, secs=0.0)
        arms = {"A_op1cl_op2s": float(ci["s"]), "B_op1cl_s0": 0.0}
        if nm in _SX:
            arms["D_op1cl_xfoils"] = float(_SX[nm])
        for sd, W in _W.items():
            z = encode_np(tok.astype(np.float64), W)
            for arm, sv in arms.items():
                pred = fwd_np(ci["X"].astype(np.float64), np.full(len(P), sv, np.float64), PA, z, W)
                pred[sm, 0] = 0.0; pred[sm, 1] = 0.0; pred[sm, 3] = 0.0
                out["arms"].setdefault(arm, {})[str(sd)] = case_metrics(pred, TR, W["std"], masks)
                out.setdefault("nonfinite_pred", {})[f"{arm}_s{sd}"] = int((~np.isfinite(pred)).sum())
            out["metrics"][str(sd)] = out["arms"]["A_op1cl_op2s"][str(sd)]
        out["secs"] = time.time() - t0
        return out
    except Exception as e:
        import traceback
        return dict(case=nm, status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:], secs=time.time() - t0)


# ----------------------------------------------------------------------------- pairing
def paired(new_rows, sealed, mask="all", B=10000, seed=0, arm="A_op1cl_op2s", label=True):
    rng = np.random.default_rng(seed)
    res = {}
    cases = [r["case"] for r in new_rows]
    for ch in FIELDS:
        per_seed = {}
        pooled_new, pooled_old = [], []
        for sd in SEEDS:
            old = sealed[sd]
            idx = [int(np.where(old["case"] == c)[0][0]) for c in cases]
            o = old[f"mse_{mask}_{ch}"][idx]
            n = np.array([r["arms"][arm][str(sd)][mask][ch] for r in new_rows])
            per_seed[str(sd)] = float(n.mean() / o.mean())
            pooled_new.append(n); pooled_old.append(o)
        N = np.mean(pooled_new, 0); O = np.mean(pooled_old, 0)        # seed-mean per case
        R = float(N.mean() / O.mean())
        bs = np.empty(B)
        for b in range(B):
            ii = rng.integers(0, len(N), len(N))
            bs[b] = N[ii].mean() / O[ii].mean()
        lo, hi = np.percentile(bs, [2.5, 97.5])
        lab = None
        if label and ch in MARGINS:
            lab = "NON_INFERIOR" if hi < MARGINS[ch] else ("DEGRADED" if lo > MARGINS[ch] else "INCONCLUSIVE")
        res[ch] = dict(mean_confirmed_provider=float(N.mean()), mean_sealed_xfoil_input=float(O.mean()), R=R,
                       R_ci95=[float(lo), float(hi)], R_per_seed=per_seed, margin=MARGINS.get(ch), label=lab,
                       n_cases_improved=int((N < O).sum()), n_cases=len(N))
    return res


def main():
    nw = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    utc = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    OUT = os.path.join(VP.RESULTS, f"gate0_confirmed_provider_{utc}"); os.makedirs(OUT, exist_ok=True)
    from supervised_field.nf_eval200 import ROOT
    sealed = {sd: np.load(os.path.join(EVAL_RUNS, SEALED[sd], "per_case_metrics.npz"), allow_pickle=True) for sd in SEEDS}
    cases = sorted(json.load(open(os.path.join(ROOT, "manifest.json")))["full_test"])
    assert list(sealed[0]["case"]) == cases, "sealed row order != sorted full_test"
    if lim:
        cases = cases[:lim]
    seed_files = {sd: os.path.join(VP.WEIGHTS, f) for sd, f in SEEDS.items()}
    print(f"[gate0] {len(cases)} cases, {len(seed_files)} sealed decoders, workers={nw}, H_SEP={H_SEP}", flush=True)
    t0 = time.time()
    sx = {str(c): float(v) for c, v in zip(sealed[0]["case"], sealed[0]["s_val"])}
    with mp.Pool(nw, initializer=_init, initargs=(seed_files, sx)) as pool:
        rows = pool.map(_work, cases, chunksize=2)
    ok = [r for r in rows if r["status"] == "ok"]; bad = [r for r in rows if r["status"] != "ok"]
    print(f"[gate0] done {len(ok)} ok / {len(bad)} error in {time.time()-t0:.0f}s", flush=True)
    for r in bad[:3]:
        print("  ERROR", r["case"], r["error"]); print(r.get("tb", "")[-600:])
    if not ok:
        sys.exit(1)
    result = dict(schema="GATE0_CONFIRMED_STATE_PROVIDER_V1", utc=utc, script=os.path.relpath(__file__, REPO), script_sha256=sha(__file__),
                  op1=dict(path=os.path.relpath(OP1_W, REPO), sha256=sha(OP1_W)), op2=dict(path=os.path.relpath(OP2_W, REPO), sha256=sha(OP2_W)),
                  decoders={str(sd): dict(path=os.path.relpath(f, REPO), sha256=sha(f)) for sd, f in seed_files.items()},
                  sealed_runs={str(sd): SEALED[sd] for sd in SEEDS}, split="full_test", n_cases=len(ok), n_error=len(bad),
                  preregistration=dict(H_SEP=H_SEP, H_SENS=list(H_SENS), margins_borrowed_from="f0_freeze P1 adjudication guardrails", margins=MARGINS,
                                       primary="all-mesh per-case standardized MSE, case-equal mean, ratio to the sealed XFOIL-input run, paired case bootstrap B=10000 seed 0"),
                  paired={mk: paired(ok, sealed, mk) for mk in ("all", "near_wall", "off_body")},
                  diagnostic_arms={arm: {mk: paired(ok, sealed, mk, arm=arm, label=False) for mk in ("all", "near_wall")}
                                   for arm in ("B_op1cl_s0", "D_op1cl_xfoils")},
                  s_rule=dict(rule=S_RULE, H_SEP=H_SEP, calibration=CALIB))
    # provider-vs-XFOIL diagnostics on the same cases
    idx = [int(np.where(sealed[0]["case"] == r["case"])[0][0]) for r in ok]
    cl_x = sealed[0]["cl_used"][idx]; s_x = sealed[0]["s_val"][idx]
    cl_o = np.array([r["cl_used"] for r in ok]); s_o = np.array([r["s_val"] for r in ok])
    result["provider_vs_xfoil_diagnostic"] = dict(
        cl=dict(R2=float(1 - np.sum((cl_o - cl_x) ** 2) / np.sum((cl_x - cl_x.mean()) ** 2)), median_abs_err=float(np.median(np.abs(cl_o - cl_x))),
                median_rel_err=float(np.median(np.abs(cl_o - cl_x) / np.maximum(np.abs(cl_x), 1e-3))), n_xfoil_fallback_cases=int((sealed[0]["cl_src"][idx] != "xfoil_xtr").sum())),
        s=dict(MAE=float(np.mean(np.abs(s_o - s_x))), frac_both_zero=float(np.mean((s_o == 0) & (s_x == 0))), frac_sign_agree=float(np.mean((s_o > 0) == (s_x > 0))),
               xfoil_frac_nonzero=float(np.mean(s_x > 0)), provider_frac_nonzero=float(np.mean(s_o > 0)),
               sensitivity_MAE={h: float(np.mean(np.abs(np.array([r["diag"]["s_sens"][h] for r in ok]) - s_x))) for h in map(str, H_SENS)}),
        re_range=[float(min(r["Re"] for r in ok)), float(max(r["Re"] for r in ok))], cst_fit_err_max=float(max(r["diag"]["cst_fit_err"] for r in ok)))
    json.dump(result, open(os.path.join(OUT, "result.json"), "w"), indent=1)
    np.savez(os.path.join(OUT, "per_case.npz"), case=np.array([r["case"] for r in ok]), cl_provider=cl_o, s_provider=s_o, cl_xfoil=cl_x, s_xfoil=s_x,
             **{f"{arm}_mse_{mk}_{ch}_s{sd}": np.array([r["arms"][arm][str(sd)][mk][ch] for r in ok]) for arm in ok[0]["arms"] for mk in ("all", "near_wall", "off_body") for ch in FIELDS for sd in SEEDS})
    json.dump(rows, open(os.path.join(OUT, "rows.json"), "w"), default=str)
    print(f"[gate0] artifact {OUT}")
    print(f"{'mask':10s}{'ch':5s}{'sealed(XFOIL in)':>17s}{'confirmed OP1/OP2':>19s}{'R':>8s}  {'CI95':>16s}  label")
    for mk in ("all", "near_wall"):
        for ch in FIELDS:
            p = result["paired"][mk][ch]
            print(f"{mk:10s}{ch:5s}{p['mean_sealed_xfoil_input']:17.3e}{p['mean_confirmed_provider']:19.3e}{p['R']:8.3f}  [{p['R_ci95'][0]:.3f},{p['R_ci95'][1]:.3f}]  {p['label']}")
    for arm in ("B_op1cl_s0", "D_op1cl_xfoils"):
        for ch in FIELDS:
            p = result["diagnostic_arms"][arm]["all"][ch]
            print(f"{arm:16s} all {ch:5s} R {p['R']:.3f} [{p['R_ci95'][0]:.3f},{p['R_ci95'][1]:.3f}]  improved {p['n_cases_improved']}/{p['n_cases']}")
    print("provider vs XFOIL:", json.dumps(result["provider_vs_xfoil_diagnostic"]))


if __name__ == "__main__":
    main()
