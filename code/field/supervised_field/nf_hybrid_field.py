# -*- coding: utf-8 -*-
"""supervised_field/nf_hybrid_field.py — per-channel HYBRID field model.

STATUS (2026-09-05).  The XFOIL-free input boundary is IMPLEMENTED: this module makes no
XFOIL call and no direct-coefficient-head call.  What the import graph guarantees is
narrower than "never touches XFOIL_EXE": the historical modules DO still execute their own
module-level `os.environ.setdefault("XFOIL_EXE", ...)`, and every such import here is
wrapped in `no_solver_env()`, which RESTORES the variable to its exact prior state -- absent
stays absent, a pre-existing value is returned unchanged.  The env is touched and then put
back, not left untouched.  End-to-end field inference is nevertheless **BLOCKED**, because `cl` and `s`
now require an injected provider and no compliant provider exists in this repository yet.
The word "DEPLOYABLE" was in this header before that change; it is removed because it is
no longer true of the end-to-end route.

Tonight's Pareto-dominant result: a per-CHANNEL hybrid that takes
  * velocity (ux, uy) + nut  from the full_train-800 DENSE model
        weights = _nf_scaling_n800.npz  (cond=concat, readout=dense, --fourier
        multiscale; 298k — beats trunk-branch on ux/uy/nut, NO capacity confound)
        [CORRECTED 2026-06-24: was _nf_scaling_n800_tb.npz (basisremix 412k);
         adversarial review found plain-dense N=800 is strictly better on all
         velocity/nut channels AND smaller → swapped.]
  * pressure (p) + p_surf    from the 200-case production model
        weights = _nf_concat_v5_ms.npz  (cond=concat, readout=dense, md5 889da1c1)

LEGACY-PROTOCOL NUMBERS — NOT VALID FOR THE CURRENT ROUTE.  The figures below were measured
under the LEGACY input protocol, i.e. with `cl` from XFOIL XTR and `s` from XFOIL Cf.  The
current route sources both from an injected provider, so these numbers are NOT
protocol-matched to it and carry over to nothing.  They are retained only as the historical
record of the checkpoints this module loads.
  hybrid full_test-200 mean std-MSE (velocity from dense N=800):
  ux 0.0272  uy 0.0461  p 0.0858  nut 0.0802  p_surf 0.1696
vs the 200-case model:
  ux 0.0473  uy 0.0675  p 0.0858  nut 0.1180  p_surf 0.1696
(p / p_surf are IDENTICAL to the 200-case model by construction; ux/uy/nut from dense N=800.)

NEW file.  Frozen files import-only:
  * forward + helpers  : nf_eval200_v5.{nf_forward_v5, gband}
  * per-case constants  : nf_eval200.{NU_AIR, FIELDS, ROOT, STATS_NPZ, HERE}
  * paint / geometry    : airfrans_p3_fields_v2.{case_params, analytic_polyline,
                          paint_points_v2}, nf_extract.{panel_mids_normals,
                          nearest_idx, PSCALE, A_NORM, RE_LO, RE_HI}
  [2026-09-05: `sep_scalar_xfoil`, `ENS_PATHS`, `head_feats`, `ens_cl` and
   `airfrans_p1_labels.run_one` are NO LONGER imported -- see the removal note below.]

The per-case assembly here MIRRORS nf_eval200_v5.main()'s per-case loop
(lines ~228-282): cl -> v2 paint PA (surface zeroing) -> 13-dim feature matrix X
-> per-case separation scalar s_row + geometric g_band -> destandardized net for
each model -> §4.4 reconstruction:
    pred[:,0:3] = g_band*PA[:,0:3] + (net[:,0:3]*std[:3] + t_mean[:3])
    pred[:,3]   =                    net[:,3]  *std[3]  + t_mean[3]   # nut, NO paint
Both models share the SAME X / s_row / PA / g_band; only net/std/t_mean differ.
The HYBRID then takes columns [ux,uy,nut] from the dense-N800 reconstruction and
column [p] from the 200-case reconstruction.

2026-09-05 -- XFOIL INPUT DEPENDENCY REMOVED FROM THE AUTHORITATIVE ROUTE.
Both XFOIL-sourced model inputs are gone from this module:
  * `cl_used` no longer comes from `airfrans_p1_labels.run_one` (XFOIL XTR), and the
    `head_feats`/`ens_cl` direct-coefficient-head fallback is gone with it;
  * `s` no longer comes from `nf_eval200_v5.sep_scalar_xfoil` (XFOIL Cf);
  * `XFOIL_EXE` is no longer set here.
Both now arrive from an INJECTED state provider validated by
`supervised_field.field_state_contract`: `cl` must be a `gamma_eff` circulation readout and
`s` must be derived from the learned viscous state, with `external_solver_calls == 0`.
There is NO default provider and NO legacy fallback -- a missing or non-compliant provider
raises.  See `field_state_contract` for the reconciliation result: no compliant provider
exists in this repository yet, so end-to-end field inference is BLOCKED on that upstream
component even though the XFOIL removal here is complete.

CAVEATS THAT ARE NOT HIDDEN
  * The existing field weights (`_nf_scaling_n800.npz`, `_nf_concat_v5_ms.npz`) were TRAINED
    on the legacy XFOIL-sourced `s` and XFOIL-cl paint distribution.  Under a new provider
    the input distribution changes, so those checkpoints are NOT protocol-matched to this
    route and NO accuracy claim -- improvement or retention -- is made here.
  * `paint_points_v2` still closes the paint strength with TWO CONVENTIONAL PANEL SOLVES
    (`airfrans_p3_fields_v2.decamber_gamma` -> `sdf2d.panel_inviscid.panel_solve` at
    alpha=0 and alpha=90, with `Gam_t = 0.5 * cl_visc`).  That is a separate forbidden
    dependency and is left UNTOUCHED by this XFOIL-only change; it is reported as an
    explicit unresolved gap rather than quietly altered.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np

from supervised_field.field_state_contract import (
    resolve_case_state, validate_provider, diagnostics_block, no_solver_env)

# ---- frozen imports (NO edits to those files) ----------------------------
# nf_eval200:29 and nf_eval200_v5:35 both run os.environ.setdefault("XFOIL_EXE", ...) at
# MODULE level.  Only solver-free constants and the pure-numpy forward are taken from
# them.  That setdefault still RUNS; `no_solver_env` then restores XFOIL_EXE to its exact
# prior state, so the observable post-import state is unchanged (absent stays absent, a
# pre-existing value is preserved) -- which is the guarantee, not "never touched".
# ENS_PATHS is deliberately
# NOT imported any more: it existed only for the removed ens_cl direct-head fallback.
with no_solver_env():
    from supervised_field.nf_eval200 import (
        NU_AIR, FIELDS, ROOT, STATS_NPZ, HERE)
    from supervised_field.nf_eval200_v5 import nf_forward_v5, gband

# velocity (ux,uy) + nut from this model; pressure from the 200-case model
W_VEL = os.path.join(HERE, "_nf_scaling_n800.npz")        # full_train-800 dense (298k; beats tb on ux/uy/nut, no capacity confound)
W_PRES = os.path.join(HERE, "_nf_concat_v5_ms.npz")       # 200-case production
# hybrid channel routing into the 4-col [ux,uy,p,nut] reconstruction
VEL_COLS = (0, 1, 3)      # ux, uy, nut  <- dense N=800
PRES_COLS = (2,)          # p            <- 200-case

_CACHE = {}               # lazy singleton: weights + stats ONLY (keys in _LOAD_KEYS)
_P3 = {}                  # lazy singleton: guarded p3 geometry helpers ONLY

#: every key `_load()` must have produced before it may fast-return.  Guarding on the
#: full key set rather than on `if _CACHE` keeps `_load()` correct no matter what else
#: is memoised, and no matter the call order.
_LOAD_KEYS = ("std", "wv", "wp", "sz")


def _p3_geometry():
    """GUARDED lazy import of the p3 geometry helpers.

    `airfrans_p3_fields_v2.py:46` runs `os.environ.setdefault("XFOIL_EXE", ...)` at MODULE
    level, and that line sits BEFORE its own `import airfrans` at :50 -- so a *failing* p3
    import still leaks the variable.  `_case_inputs` used to import p3 directly, outside
    the guard, which meant a cold process reached XFOIL_EXE through the real per-case path
    even though the top-level `nf_hybrid_field` import was clean.

    Routing every p3 access through this one helper closes that hole: the import happens
    inside `no_solver_env()`, so XFOIL_EXE is restored to its exact prior state (absent
    stays absent) whether the import succeeds or raises.  Because Python caches the module
    in `sys.modules`, a LATER unguarded import of the same module -- e.g. the one inside
    `field_features.legacy_paint4` -- reuses this already-guarded module object and
    therefore re-runs no module-level code and leaks nothing.

    That ordering is the contract: p3 must be first imported HERE.
    """
    if "fns" not in _P3:
        with no_solver_env():
            from supervised_field.airfrans_p3_fields_v2 import (
                case_params, analytic_polyline)
        _P3["fns"] = (case_params, analytic_polyline)
    return _P3["fns"]


def _load():
    """Load both weight dicts + stats ONCE (singleton).

    Fast-returns only when EVERY key in `_LOAD_KEYS` is present, so a partially populated
    cache can never be handed back.  (`if _CACHE:` was not safe: another memoised entry
    would make the dict truthy and short-circuit the load.)"""
    if all(k in _CACHE for k in _LOAD_KEYS):
        return _CACHE
    sz = np.load(STATS_NPZ, allow_pickle=True)
    std = np.array([float(sz[k][1]) for k in FIELDS], np.float64)
    wv = np.load(W_VEL, allow_pickle=True)
    wp = np.load(W_PRES, allow_pickle=True)
    # stats consistency (same assert nf_eval200_v5.main does)
    for tag, wz in (("vel/tb-800", wv), ("pres/200-case", wp)):
        assert np.allclose(std, np.asarray(wz["std"], np.float64)), \
            f"{tag} std mismatch vs {os.path.basename(STATS_NPZ)}"
    _CACHE.update(std=std, wv=wv, wp=wp, sz=sz)
    return _CACHE


def _case_inputs(case_name, state_provider, feature_mode="legacy_paint4",
                 model_state=None):
    """MINIMAL per-case assembly mirrored from nf_eval200_v5.main() lines
    ~228-268.  Returns the shared (P, sm, X, s_row, g_band, PA, U, a, prm, Re,
    cl_used, cl_src, truth).

    state_provider is REQUIRED and positional -- it supplies `cl` and `s`, which used to
    come from XFOIL.  There is no default: see field_state_contract.validate_provider for
    the migration contract carried in the error message.

    feature_mode is the EXPLICIT feature route and goes through the one canonical
    assembler, field_features.assemble_field_features -- the same implementation the
    training corpus builders use.  "legacy_paint4" (default) is the P0 / P1-E0 feature and
    is bit-identical to the inline paint block this function used before 2026-09-03;
    "predicted_profile_prof4" is the preregistered P1 feature (P1-E1).  An unknown mode
    RAISES in the assembler: no silent legacy fallback, because evaluating P1 weights on
    the legacy feature by default is exactly the defect this argument exists to fix."""
    import airfrans as af
    case_params, analytic_polyline = _p3_geometry()   # GUARDED -- see _p3_geometry
    from supervised_field.field_features import assemble_field_features
    from supervised_field.nf_extract import (
        panel_mids_normals, nearest_idx, PSCALE, A_NORM, RE_LO, RE_HI)

    # (no _load() here: this function stopped needing the weight/stats singleton when the
    # ens_cl direct-head fallback -- its only consumer, via c["ez"] -- was removed.
    # hybrid_field() still calls _load() for the reconstruction.)
    U, a, prm = case_params(case_name)
    Re = U / NU_AIR

    # cl (paint strength) and s (model input) from the INJECTED provider.
    # WAS: cl <- airfrans_p1_labels.run_one (XFOIL XTR) with an ens_cl direct-head
    # fallback, and s <- nf_eval200_v5.sep_scalar_xfoil (XFOIL Cf).  Both are gone.
    # resolve_case_state raises on a missing provider, malformed output, or any forbidden
    # provenance -- there is no default and no legacy fallback.
    st = resolve_case_state(state_provider, prm, a, Re)
    cl_used = st["cl"]
    s_val = st["s"]
    prov = st["provenance"]
    cl_src = prov["cl_source"]
    s_src = prov["s_source"]

    # AirfRANS RANS truth (L238-243)
    sim = af.Simulation(root=ROOT, name=case_name)
    P = np.asarray(sim.position, np.float64)
    TR = np.column_stack([np.asarray(sim.velocity, np.float64),
                          np.asarray(sim.pressure, np.float64).reshape(-1, 1),
                          np.asarray(sim.nu_t, np.float64).reshape(-1, 1)])
    sm = np.asarray(sim.surface, bool)
    n_pt = len(P)

    # feature assembly, ONE canonical route (was: an inline unconditional v2 paint block)
    poly = analytic_polyline(prm)
    ff = assemble_field_features(
        dict(P=P, poly=poly, alpha=a, Re=Re, U=U, cl_used=cl_used, surf=sm, name=case_name),
        model_state=model_state, feature_mode=feature_mode)
    PA = ff["feat"]
    r = ff["paint_raw"]                     # d / wake_band for X and g_band, as before

    # 13-dim feature matrix X (L252-261)
    mid, nrm = panel_mids_normals(poly)
    nn = nearest_idx(P, mid)
    prm4 = np.zeros(4); prm4[:len(prm)] = prm
    X = np.column_stack([
        P[:, 0], P[:, 1], r["d"], nrm[nn, 0], nrm[nn, 1],
        sm.astype(np.float64),
        np.full(n_pt, a / A_NORM),
        np.full(n_pt, 2.0 * (np.log(Re) - RE_LO) / (RE_HI - RE_LO) - 1.0),
        np.tile(prm4 / PSCALE, (n_pt, 1)),
        np.full(n_pt, float(len(prm) == 4))]).astype(np.float32)

    # per-case s comes from the provider (above), NOT from sep_scalar_xfoil;
    # geometric g_band is unchanged (L264-268)
    s_row = np.full(n_pt, s_val, np.float32)
    g_band = gband(r["d"], r["wake_band"], P[:, 0], Re)

    return dict(P=P, sm=sm, X=X, s_row=s_row, g_band=g_band, PA=PA,
                TR=TR, U=U, a=a, prm=prm, Re=Re, poly=poly,
                cl_used=cl_used, cl_src=cl_src, s=s_val, s_src=s_src,
                state_provenance=prov, state_diagnostics=diagnostics_block(st),
                n_pt=n_pt, n_surf=int(sm.sum()),
                feature_mode=feature_mode, feature_meta=ff["meta"])


def _reconstruct(wz, X, s_row, g_band, PA, std):
    """§4.4 reconstruction for ONE model (mirrors eval200_v5 L279-285).
    Both our models are nut_mode=scratch, so nut = destd(net), NO paint."""
    t_mean = np.asarray(wz["t_mean"], np.float64)
    net = nf_forward_v5(wz, X, s_row) * std[None, :] + t_mean[None, :]
    pred = net.copy()
    pred[:, :3] += g_band[:, None] * PA[:, :3]              # g_band paint for ux,uy,p
    if ("nut_mode" in wz.files) and (str(wz["nut_mode"]) == "residual"):
        pred[:, 3] += PA[:, 3]                             # (neither of ours; guarded)
    return pred


def hybrid_field(case_name, state_provider=None, return_components=False):
    """Per-channel hybrid for one full_test case.

    NOT DEPLOYABLE END-TO-END TODAY.  The XFOIL-free input boundary is implemented, but a
    compliant `state_provider` does not exist in this repository, so every call currently
    raises `ProviderMissing`.  That is the intended fail-closed behaviour, not a defect.

    SIGNATURE CHANGED 2026-09-05: `state_provider` is now REQUIRED and is the second
    positional parameter.  It replaces the two XFOIL-sourced inputs (`cl` from XFOIL XTR,
    `s` from XFOIL Cf).  It defaults to None only so that omitting it raises the
    contract's migration message instead of a bare TypeError; None is NOT a usable value.

    MIGRATION:  hybrid_field(name)  ->  hybrid_field(name, state_provider=<provider>)
    where the provider implements
        state_for(prm, alpha_deg, Re) -> {"cl": float, "s": float, "provenance": {...}}
    See supervised_field.field_state_contract for the provenance requirements.

    Returns dict:
        nodes   (N,2)  xy coordinates
        pred    (N,4)  hybrid [ux, uy, p, nut]
        truth   (N,4)  AirfRANS RANS truth (or None if unavailable)
        n_surf  int    number of surface nodes (p_surf = pred[surf,2])
        surf    (N,)   bool surface mask
        + meta : U, alpha, Re, prm, poly, cl_used, cl_source, s, s_source,
                 state_provenance, state_diagnostics (carries external_solver_calls=0),
                 g_band_frac
    If return_components, also includes pred_vel / pred_pres (the two full
    single-model reconstructions) for adversarial cross-checks.

    NO accuracy claim is made for this route: the loaded checkpoints were trained on the
    legacy XFOIL-sourced s and XFOIL-cl paint distribution and are NOT protocol-matched to
    a new provider's input distribution.
    """
    validate_provider(state_provider)          # fail closed BEFORE any weight is touched
    c = _load()
    ci = _case_inputs(case_name, state_provider)
    std = c["std"]

    pred_vel = _reconstruct(c["wv"], ci["X"], ci["s_row"], ci["g_band"],
                            ci["PA"], std)        # tb-800: source of ux,uy,nut
    pred_pres = _reconstruct(c["wp"], ci["X"], ci["s_row"], ci["g_band"],
                             ci["PA"], std)       # 200-case: source of p

    pred = np.empty_like(pred_vel)
    pred[:, list(VEL_COLS)] = pred_vel[:, list(VEL_COLS)]   # ux,uy,nut <- tb
    pred[:, list(PRES_COLS)] = pred_pres[:, list(PRES_COLS)]  # p <- 200-case

    out = dict(nodes=ci["P"][:, :2].copy(), pred=pred, truth=ci["TR"],
               surf=ci["sm"], n_surf=ci["n_surf"],
               U=ci["U"], alpha=ci["a"], Re=ci["Re"], prm=ci["prm"],
               poly=ci["poly"], cl_used=ci["cl_used"], cl_source=ci["cl_src"],
               s=ci["s"], s_source=ci["s_src"],
               state_provenance=ci["state_provenance"],
               state_diagnostics=ci["state_diagnostics"],
               g_band_frac=float(ci["g_band"].mean()))
    if return_components:
        out["pred_vel"] = pred_vel      # full tb-800 reconstruction (all 4 cols)
        out["pred_pres"] = pred_pres    # full 200-case reconstruction (all 4 cols)
        out["std"] = std
    return out


def case_mse(out):
    """Standardized full-volume MSE + surface-p MSE for a hybrid_field() dict
    (mirrors eval200_v5's nested mses())."""
    std = _load()["std"]
    TR, pred, sm = out["truth"], out["pred"], out["surf"]
    e = ((pred - TR) / std[None, :]) ** 2
    d = {k: float(v) for k, v in zip(FIELDS, e.mean(0))}
    d["p_surf"] = float(np.mean(((pred[sm, 2] - TR[sm, 2]) / std[2]) ** 2))
    return d


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # The old self-check ran two real full_test cases.  It cannot run any more, and NOT
    # because of a defect: `cl` and `s` now require an injected provider, and no compliant
    # provider exists in this repository yet.  Running a case would also be an evaluation,
    # which this change is not permitted to perform.  So this entry point reports the
    # contract state and the blocker instead of fabricating a run.
    from supervised_field.field_state_contract import (
        CL_SOURCE_REQUIRED, S_SOURCE_REQUIRED, ProviderMissing)
    print(f"[hybrid] vel/nut <- {os.path.basename(W_VEL)}")
    print(f"[hybrid] pressure <- {os.path.basename(W_PRES)}")
    print("[contract] cl_source must be:", CL_SOURCE_REQUIRED)
    print("[contract] s_source  must be:", S_SOURCE_REQUIRED)
    print("[contract] external_solver_calls must be: 0")
    print("[removed]  airfrans_p1_labels.run_one (XFOIL XTR) cl")
    print("[removed]  head_feats/ens_cl direct-coefficient-head cl fallback")
    print("[removed]  nf_eval200_v5.sep_scalar_xfoil (XFOIL Cf) s")
    print("[removed]  module-level XFOIL_EXE default")
    print("[guarded]  historical imports wrapped in no_solver_env(): their module-level "
          "setdefault still runs, and XFOIL_EXE is then RESTORED to its exact prior state "
          "(absent stays absent; a pre-existing value is preserved)")
    try:
        hybrid_field("any_case_name")
    except ProviderMissing as exc:
        print("\n[fail-closed OK] hybrid_field without a provider raises "
              f"ProviderMissing:\n{exc}")
    else:
        raise SystemExit("CONTRACT BREACH: hybrid_field ran without a state provider")
    print("\n[status] XFOIL removal in this module: COMPLETE")
    print("[status] deployable end-to-end field inference: BLOCKED -- no compliant "
          "gamma_eff-cl / learned-viscous-s provider exists upstream")
    print("[unresolved] paint_points_v2 still closes paint strength with two conventional "
          "panel solves (decamber_gamma -> sdf2d.panel_inviscid.panel_solve); out of "
          "scope for this XFOIL-only change")
    print("[no claim] the loaded checkpoints were trained on the legacy XFOIL-sourced s / "
          "XFOIL-cl paint distribution; they are NOT protocol-matched to this route and "
          "no accuracy improvement or retention is claimed")
