# -*- coding: utf-8 -*-
"""supervised_field/field_features.py — THE canonical field-feature assembler.

ROOT CAUSE this file exists to remove: the P1 training corpus builder
(p1_prof_feature.py) and the evaluator (nf_hybrid_field._case_inputs -> nf_eval_percase)
each implemented feature construction independently, so they diverged silently.  P1 was
trained on prof4 and evaluated on legacy paint4, and nothing in either code path could
notice.  From here on BOTH routes go through assemble_field_features() and the mode is an
explicit, recorded argument.

    assemble_field_features(case, model_state, feature_mode, profile_contract) -> dict

    feature_mode = "legacy_paint4"           -> P0 and P1-E0
                     airfrans_p3_fields_v2.paint_points_v2, U-scaled, v2 surface zeroing.
                     The SAME function the P0 corpus was generated with (nf_extract.py:120)
                     and the same 4 lines nf_hybrid_field._case_inputs used inline.
    feature_mode = "predicted_profile_prof4" -> P1-E1
                     p1_prof_feature.prof4_case, CALLED, not re-implemented.  That is the
                     function that produced _nf_train_v5_full800_p1prof_20260903T072000Z.npz
                     (build_report code_sha256 7095b0c8...).  Its base is the legacy_paint4
                     array above, so the registered fallback stays bit-identical.

ONE implementation, two drivers.  p1_prof_feature.build() (the batched 800-case corpus
driver) is deliberately NOT rewired to call this module: its file sha256 is pinned in the
P1 corpus build_report and asserted below, so editing it would break the provenance of the
corpus the existing checkpoints were trained on, for no gain.  The shared implementation is
prof4_case itself, and the sharing is verified numerically rather than asserted -- see
--check C2, which reproduces the stored corpus rows through this module.

FAIL CLOSED.  An unknown or unsupported mode raises.  There is NO silent fallback to
legacy: that fallback is exactly the defect.  Two further raise-not-fallback conditions:
the profile contract sha256 must match, and a case for which the predicted-state chain
cannot produce a branch raises Prof4StateUnavailable (the caller must STOP and report it,
never quietly paint it with paint4 -- that would recreate P1-E0 case by case).

Self-check (controls that can fail, incl. the fail-closed control):
    conda run --no-capture-output -n tf218 python -m supervised_field.field_features --check
"""
from __future__ import annotations

import os
import sys
import json
import hashlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

FEATURE_MODES = ("legacy_paint4", "predicted_profile_prof4")
CHANNEL_ORDER = ("ux", "uy", "p", "nut")
SURFACE_ZEROED_CHANNELS = (0, 1, 3)                     # v2 convention: p is NOT zeroed
PHYSICAL_SCALING = ("u*U", "v*U", "cp*0.5*U**2", "nut*U")
FEATURE_DTYPE = "float64"                               # corpus STORES float32; see dtype_note
DTYPE_NOTE = ("the assembler returns float64; both corpora store PAINT as float32, so the "
              "trained-on feature is the float32 rounding of this array.  This is identical "
              "in kind for legacy_paint4 and predicted_profile_prof4 -- the sealed P0/P1-E0 "
              "runs already evaluate float64 paint against float32-trained weights -- so it "
              "is not a P0-vs-P1 asymmetry.")

CONTRACT = os.path.join(REPO, "VortFoil", "artifacts", "f0b_p1_feature_contract.json")
CONTRACT_SHA256 = "81bcf53759d73e2ad0528b4a841d1c718a534d40b8bd71d69252d23813c76216"
PROF4_GENERATOR = os.path.join(HERE, "p1_prof_feature.py")
PROF4_GENERATOR_SHA256 = "7095b0c8349bf4b62de2c7995f191fbb4e52ff2fa44a09155eae1cb2ee8f2e29"
STATE_OPERATOR = os.path.join(HERE, "_adafol_lpa_xfoil_diverse.npz")
STATE_OPERATOR_SHA256 = "2dccbeae2424bd242677a324b5f81b721611df1f7dc080af5353288663e39ef1"
P1_CORPUS = os.path.join(HERE, "_nf_train_v5_full800_p1prof_20260903T072000Z.npz")
P0_CORPUS = os.path.join(HERE, "_nf_train_v5_full800.npz")

ACCEPTANCE_RULE = ("station accepted iff fields_v8_adafdisp._adaf_at_mids gate "
                   "(H_cl in [H_LO,H_HI], theta>0, d99 finite>0, theta<=3*theta_lam, "
                   "branch_ok: finite/non-degenerate/H_te in [1.0,3.5]/x_tr<=0.6) AND the "
                   "Coles fit needs delta*>0, ue finite>0, fit residual<1e-3, delta>0; "
                   "p1_prof_feature.state_at_mids + prof4_case, verbatim")
FALLBACK_RULE = ("node keeps its legacy_paint4 row BIT-IDENTICALLY (same array element) "
                 "unless station is live AND 0 < d < delta(station); prof_nut is ALWAYS the "
                 "legacy paint nut (registered measured NO-GO); a case whose predicted-state "
                 "chain yields no turbulent branch RAISES Prof4StateUnavailable -- it is "
                 "never quietly painted with paint4")
GRID = dict(field_nodes="the case's own query points, P[:,0:2] (corpus X[:,0:2] at train)",
            wall_distance="paint_points_v2 sdf_chunked d, or the stored corpus X[:,2]",
            surface_stations="440 panel mids of analytic_polyline(params4) (220 cosine, CTE)",
            bl_collocation="121-node turbulent-leg xi (NT_SEG) -> 81-node assembled xi grid",
            branches=2)


class Prof4StateUnavailable(RuntimeError):
    """the predicted-state chain could not produce a state for this case -- STOP, do not
    fall back to legacy paint (that is the P1-E0 defect, per case)"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prewarm_platform():
    """Call ONCE before installing a subprocess block, if TensorFlow will be imported later.

    predicted_profile_prof4 needs TF (the frozen BL operator).  TF -> h5py -> platform
    reads platform.processor(), which shells out to `uname -p`; under the evaluator's
    XFOIL subprocess block that raises and the TF import dies.  platform caches the uname
    result, so one call before the block removes the only shell-out on the import path.
    It runs `uname -p` and nothing else -- the XFOIL guarantee is unaffected."""
    import platform
    return platform.processor()


ASSEMBLER_SHA256 = None                                  # filled on first use (self-hash)


def _self_sha():
    global ASSEMBLER_SHA256
    if ASSEMBLER_SHA256 is None:
        ASSEMBLER_SHA256 = sha256(os.path.abspath(__file__))
    return ASSEMBLER_SHA256


# ------------------------------------------------------------------ legacy_paint4 (P0)
def legacy_paint4(case):
    """The P0 feature.  Same call and same 4 lines as nf_extract.py:120-127 (which built
    the corpus PAINT) and as nf_hybrid_field._case_inputs used inline."""
    from supervised_field.airfrans_p3_fields_v2 import paint_points_v2
    P = np.asarray(case["P"], np.float64)
    U = float(case["U"])
    r = paint_points_v2(P, case["poly"], float(case["alpha"]), float(case["cl_used"]),
                        float(case["Re"]))
    PA = np.column_stack([r["u"] * U, r["v"] * U, r["cp"] * 0.5 * U ** 2, r["nut"] * U])
    sm = np.asarray(case["surf"], bool)
    for c in SURFACE_ZEROED_CHANNELS:
        PA[sm, c] = 0.0
    return PA, r


# -------------------------------------------------- predicted-state provider (P1 prof4)
class Prof4StateProvider:
    """(params4|poly, alpha, Re) -> per-branch predicted (theta, delta*, ue) state.

    This is the ONLY piece the batched training builder did not expose as a per-case
    function: p1_prof_feature.build() forwards the whole 800-case corpus in one operator
    call.  The chain here is the same one, reduced to a single case exactly as the archived
    AdafPainter.paint() already does it (build_case_branches_forced -> build_branch_samples
    -> frozen operator forward -> assemble_branch), with the ue attachment that prof4_case
    requires (p1_prof_feature.build line: st[side]["ue"] = cb[side]["ue"]).

    NOTE it is a case OUTSIDE the training corpus that matters here: the 200 full_test cases
    share zero names with the 800 train cases, so prof4 must be computable from
    (params4, alpha, Re) alone.  It is -- nothing in this chain reads the corpus.
    """

    def __init__(self, contract=CONTRACT):
        got = sha256(contract)
        if got != CONTRACT_SHA256:
            raise RuntimeError(f"profile contract sha256 {got} != registered {CONTRACT_SHA256}")
        self.contract_sha256 = got
        # p1_prof_feature sets XFOIL_EXE=/nonexistent at import (it is XFOIL-free by
        # construction).  Restore the caller's value so the legacy arm's cl_used lookup is
        # bit-identical to the sealed runs; no-window safety is the subprocess block's job.
        keep = os.environ.get("XFOIL_EXE")
        try:
            # PPF FIRST and deliberately so: its module-level os.environ.setdefault gives
            # NF_CPU_THREADS/OMP_NUM_THREADS = 6, which is the TF intra-op width the P1
            # corpus was actually built at (build.log: "intra=6").  TF reduction order is
            # thread-count dependent, so importing steady51/TF before it would silently
            # evaluate the feature at a different width than training used.
            from supervised_field import p1_prof_feature as PPF
            import tensorflow as tf
            import steady51 as nf
            from ibl_marcher import _Consts
            from supervised_field.adafol_problem import NT_SEG, DT
            from supervised_field.adafol_train import build_branch_samples
        finally:
            if keep is None:
                os.environ.pop("XFOIL_EXE", None)
            else:
                os.environ["XFOIL_EXE"] = keep
        gen_sha = sha256(PROF4_GENERATOR)
        if gen_sha != PROF4_GENERATOR_SHA256:
            raise RuntimeError(f"prof4 generator sha256 {gen_sha} != the sha the P1 corpus "
                               f"was built with ({PROF4_GENERATOR_SHA256})")
        op_sha = sha256(STATE_OPERATOR)
        if op_sha != STATE_OPERATOR_SHA256:
            raise RuntimeError(f"state operator sha256 {op_sha} != {STATE_OPERATOR_SHA256}")
        self._tf, self._nt, self._dt = tf, NT_SEG, DT
        self._bbs = build_branch_samples
        self.PPF = PPF
        self.generator_sha256 = gen_sha
        self.operator_sha256 = op_sha
        self.prob, self.op, self.pack = PPF.load_operator()
        self.model = nf.SingleLoopContinuousStagADAF(n_surface_pts=81)
        self.consts = _Consts(self.model)

    def perturb_operator(self, rel):
        """state-sensitivity control: mutate the frozen weights (p1_prof_feature --smoke C2)"""
        for v in self.op.net.trainable_variables:
            v.assign(v * (1.0 + rel))

    def state(self, poly, alpha, Re, name=""):
        tf = self._tf
        cb = self.PPF.build_case_branches_forced(self.model, self.consts, poly,
                                                 float(alpha), float(Re), N=81)
        cb["meta"] = dict(case_id=0, name=str(name), alpha=float(alpha), Re=float(Re),
                          params=None)
        pack = dict(self.pack)
        pack["cases"] = np.array([cb], dtype=object)
        pack["n_cases"] = 1
        samp = self._bbs(pack, self.prob, self._nt)
        bmap = {(int(samp["case_id"][m]), int(samp["side_flag"][m])): m
                for m in range(len(samp["case_id"]))}
        missing = [s for si, s in ((0, "upper"), (1, "lower")) if (0, si) not in bmap]
        if missing:
            raise Prof4StateUnavailable(
                f"{name}: no turbulent branch for {missing} (has_tr=0) -> the predicted-state "
                f"chain has no state for that side; every training case had both")
        Xop = tf.constant(samp["X"], self._dt)
        U0 = tf.constant(samp["U0"], self._dt)
        th = Xop[:, self.prob.state_dim:self.prob.state_dim + self.prob.param_dim]
        Up = self.op.forward(Xop, th, U0, training=False)["x"].numpy()
        if not np.isfinite(Up).all():
            raise Prof4StateUnavailable(f"{name}: operator forward produced non-finite state")
        xi81 = np.asarray(cb["xi"], np.float64)
        st = {}
        for si, side in ((0, "upper"), (1, "lower")):
            m = bmap[(0, si)]
            st[side] = self.PPF.assemble_branch(cb[side], xi81, Up[m], samp["xi_c"][m])
            st[side]["ue"] = np.asarray(cb[side]["ue"], np.float64)
        return st


_PROVIDER = None


def profile_state_provider(profile_contract=CONTRACT):
    """lazy per-process singleton (mp.Pool workers each build their own)"""
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = Prof4StateProvider(profile_contract)
    return _PROVIDER


# --------------------------------------------------------------------------- the entry
def assemble_field_features(case, model_state=None, feature_mode="legacy_paint4",
                            profile_contract=CONTRACT):
    """THE feature route.  Training corpus builder and evaluator both call this.

    case: dict with P (n,>=2), poly, alpha (deg), Re, U, cl_used, surf (n,) bool,
          optional name, optional paint4 (n,4) and d (n,) -- supply those two to assemble
          on STORED corpus rows instead of recomputing them (that is what the training
          corpus does: p1_prof_feature.build passes PAINT[rm] and X[rm][:,2]).
    model_state: a Prof4StateProvider, or None to use the lazy singleton.
    returns dict(feat, paint4, paint_raw, feature_mode, meta) -- meta is the manifest block.
    """
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"unsupported feature_mode {feature_mode!r}; supported: "
                         f"{FEATURE_MODES}.  FAIL CLOSED -- no silent legacy fallback "
                         f"(a silent legacy fallback is the P1-E0 defect being fixed)")
    sm = np.asarray(case["surf"], bool)
    if case.get("paint4") is None:
        PA, r = legacy_paint4(case)
        d = np.asarray(r["d"], np.float64)
        base_src, d_src = "paint_points_v2 (recomputed)", "paint_points_v2 sdf_chunked"
    else:
        PA = np.asarray(case["paint4"], np.float64).copy()
        r = None
        d = np.asarray(case["d"], np.float64)
        base_src, d_src = "supplied stored corpus PAINT", "supplied stored corpus X[:,2]"

    meta = dict(feature_mode=feature_mode,
                feature_assembler="supervised_field/field_features.py",
                feature_assembler_sha256=_self_sha(),
                channel_order=list(CHANNEL_ORDER),
                physical_scaling=list(PHYSICAL_SCALING),
                surface_zeroed_channels=list(SURFACE_ZEROED_CHANNELS),
                dtype=FEATURE_DTYPE, dtype_note=DTYPE_NOTE,
                feature_shape=list(PA.shape), n_rows=int(len(PA)),
                n_surface_rows=int(sm.sum()),
                legacy_base_source=base_src, wall_distance_source=d_src,
                grid=dict(GRID), case_name=str(case.get("name", "")))

    if feature_mode == "legacy_paint4":
        meta.update(profile_generator_sha256=None, profile_contract_sha256=None,
                    state_provider=None, acceptance_rule="n/a (no predicted state)",
                    fallback_rule="n/a (no override; legacy IS the feature)",
                    active_mask="none -- every row is legacy paint4",
                    n_override=0, override_row_frac=0.0)
        return dict(feat=PA, paint4=PA, paint_raw=r, feature_mode=feature_mode, meta=meta)

    ps = model_state or profile_state_provider(profile_contract)
    st = ps.state(case["poly"], float(case["alpha"]), float(case["Re"]),
                  name=str(case.get("name", "")))
    prof4, info = ps.PPF.prof4_case(np.asarray(case["P"], np.float64)[:, 0:2], d,
                                    case["poly"], float(case["alpha"]),
                                    float(case["cl_used"]), float(case["Re"]),
                                    float(case["U"]), PA, st)
    fb = np.asarray(prof4) != PA
    meta.update(profile_generator="supervised_field/p1_prof_feature.py :: prof4_case",
                profile_generator_sha256=ps.generator_sha256,
                profile_contract="VortFoil/artifacts/f0b_p1_feature_contract.json",
                profile_contract_sha256=ps.contract_sha256,
                state_provider=dict(
                    kind="PREDICTED (frozen panel solve + frozen ADAF-OL BL operator); "
                         "XFOIL-free, truth-free",
                    checkpoint="supervised_field/_adafol_lpa_xfoil_diverse.npz",
                    checkpoint_sha256=ps.operator_sha256,
                    operator_class="adafol_train_precision.PrecisionOperator basis lpa "
                                   "N_p=14 hidden256 x3 NT_SEG=121",
                    pod_pack="supervised_field/_archive_session_2026-07/"
                             "_adafol_data_diverse.npz",
                    frozen=True,
                    tf_intra_threads=os.environ.get("NF_CPU_THREADS"),
                    tf_intra_at_training="6 (corpus build.log)"),
                acceptance_rule=ACCEPTANCE_RULE, fallback_rule=FALLBACK_RULE,
                active_mask="live station AND 0 < d < delta(station); prof_nut never "
                            "overridden",
                n_override=int(info["n_override"]),
                override_row_frac=float(info["n_override"]) / max(len(PA), 1),
                n_changed_rows=int(fb.any(1).sum()),
                per_channel_changed=[int(fb[:, c].sum()) for c in range(4)],
                fallback_rows_bit_identical=bool(
                    np.array_equal(np.asarray(prof4)[~fb.any(1)], PA[~fb.any(1)])),
                station_bookkeeping={k: info[k] for k in
                                     ("n_mid", "n_accept_v8", "n_accept", "n_live",
                                      "n_fit_fail", "fit_resid_max", "delta_min",
                                      "delta_max", "branch_ok")})
    return dict(feat=np.asarray(prof4, np.float64), paint4=PA, paint_raw=r,
                feature_mode=feature_mode, meta=meta)


# ------------------------------------------------------------------------- self-check
def _corpus_case(z, ci):
    rm = z["case_id"] == ci
    prm = tuple(np.asarray(z["params4"][ci], np.float64)[:int(z["n_params"][ci])])
    from supervised_field.airfrans_p3_fields_v2 import analytic_polyline
    X = z["X"][rm]
    return dict(P=X[:, 0:2].astype(np.float64), d=X[:, 2].astype(np.float64),
                surf=X[:, 5].astype(bool), poly=analytic_polyline(prm),
                alpha=float(z["alpha"][ci]), Re=float(z["Re"][ci]), U=float(z["U"][ci]),
                cl_used=float(z["cl_used"][ci]), name=str(z["names"][ci])), rm


def check():
    """controls that can fail, and are shown failing"""
    import subprocess
    prewarm_platform()                                  # see prewarm_platform.__doc__
    os.environ["DISPLAY"] = ""                          # no XFOIL window on the Mac, ever
    def _blocked(*a, **k):
        raise RuntimeError("subprocess blocked by field_features.check (XFOIL must not run)")
    subprocess.run = subprocess.Popen = subprocess.call = _blocked

    rep = {}
    # ---- C0 FAIL-CLOSED: nonsense / unsupported / None modes must RAISE ---------------
    c0 = []
    for bad in ("paint4", "legacy_paint", "prof4", "adafpaint", "", None, 0):
        try:
            assemble_field_features(dict(P=np.zeros((1, 2)), surf=np.zeros(1, bool)),
                                    feature_mode=bad)
            c0.append(dict(mode=repr(bad), raised=False))
        except ValueError as e:
            c0.append(dict(mode=repr(bad), raised=True, err=str(e)[:60]))
        except Exception as e:                          # anything else = not fail-closed
            c0.append(dict(mode=repr(bad), raised=False, wrong=f"{type(e).__name__}: {e}"))
    rep["C0_fail_closed"] = dict(cases=c0, PASS=all(c["raised"] for c in c0))
    print(f"[C0] fail-closed on {len(c0)} bad modes: PASS={rep['C0_fail_closed']['PASS']}")

    z0 = np.load(P0_CORPUS, allow_pickle=True)
    z1 = np.load(P1_CORPUS, allow_pickle=True)
    case, rm = _corpus_case(z0, 0)

    # ---- C1 legacy route == the inline block nf_hybrid_field used before, bit-for-bit -
    # (the refactor regression test.  It cannot be run against the stored corpus PAINT:
    #  the corpus keeps node coordinates as float32 while the paint that produced it was
    #  computed on the float64 mesh, so a corpus comparison is quantization-limited, not
    #  implementation-limited.  The corpus-consistency number is reported below as a
    #  diagnostic, and the exact test is done on a real float64 case.)
    from supervised_field.airfrans_p3_fields_v2 import paint_points_v2
    from supervised_field.nf_eval200 import ROOT, NU_AIR
    from supervised_field.airfrans_p3_fields_v2 import case_params, analytic_polyline
    import airfrans as af
    with open(os.path.join(ROOT, "manifest.json")) as f:
        man = json.load(f)
    nm = sorted(man["full_test"])[0]
    U, a, prm = case_params(nm)
    Re = U / NU_AIR
    sim = af.Simulation(root=ROOT, name=nm)
    Pf = np.asarray(sim.position, np.float64)
    smf = np.asarray(sim.surface, bool)
    poly = analytic_polyline(prm)
    # the pre-2026-09-03 nf_hybrid_field._case_inputs paint block, verbatim:
    _r = paint_points_v2(Pf, poly, a, 0.4321, Re)
    _PA = np.column_stack([_r["u"] * U, _r["v"] * U, _r["cp"] * 0.5 * U ** 2, _r["nut"] * U])
    _PA[smf, 0] = 0.0; _PA[smf, 1] = 0.0; _PA[smf, 3] = 0.0
    fcase = dict(P=Pf, poly=poly, alpha=a, Re=Re, U=U, cl_used=0.4321, surf=smf, name=nm)
    got = assemble_field_features(fcase, feature_mode="legacy_paint4")["feat"]
    dev = float(np.abs(got - _PA).max())
    rep["C1_legacy_route_bit_identical_to_old_inline"] = dict(
        case=nm, n=int(len(_PA)), max_abs_dev=dev, bit_identical=bool(np.array_equal(got, _PA)),
        PASS=bool(np.array_equal(got, _PA)))
    print(f"[C1] legacy route vs the old inline block: max|dev|={dev:.3e} "
          f"bit_identical={rep['C1_legacy_route_bit_identical_to_old_inline']['bit_identical']}")

    # ---- C1n CONTROL that must FAIL: same route with cl_used shifted ------------------
    fbad = dict(fcase); fbad["cl_used"] = 0.9321
    gotb = assemble_field_features(fbad, feature_mode="legacy_paint4")["feat"]
    devb = float(np.abs(gotb - _PA).max())
    rep["C1n_control_clbias"] = dict(max_abs_dev=devb, differs=devb > 0.0, PASS=devb > 0.0)
    print(f"[C1n] CONTROL cl_used+0.5 must differ: max|dev|={devb:.3e} "
          f"{'RED as required' if devb > 0 else 'BROKEN CHECK'}")

    # diagnostic only (quantization-limited, NOT a gate)
    gc = assemble_field_features(case, feature_mode="legacy_paint4")["feat"]
    refc = z0["PAINT"][rm]
    sc = np.maximum(np.abs(refc).max(0), 1e-12)
    rep["D1_legacy_vs_stored_P0_corpus_diagnostic"] = dict(
        case=case["name"], max_abs_dev_float32=float(np.abs(gc.astype(np.float32) - refc).max()),
        max_rel_dev=float((np.abs(gc.astype(np.float32) - refc) / sc[None, :]).max()),
        note="corpus stores node coords as float32; the stored paint was computed on the "
             "float64 mesh -- a nonzero value here is coordinate quantization")
    print(f"[D1] diagnostic legacy vs stored P0 corpus: max_rel_dev="
          f"{rep['D1_legacy_vs_stored_P0_corpus_diagnostic']['max_rel_dev']:.3e}")

    # ---- C2 prof4 route == the P1 corpus the checkpoints were TRAINED on --------------
    # inputs are the STORED corpus rows, exactly what p1_prof_feature.build() passed:
    # PAINT[rm] as the base, X[rm][:,2] as the wall distance, analytic_polyline(params4).
    # Three cases on purpose: a typical one, the corpus's heaviest override (91), and a
    # whole-case fallback (9, branch_ok False/False, n_live 0) where equality must be EXACT.
    # TOLERANCE, declared here and before any P1-E1 prediction exists: this is a code-
    # equality check, not a study threshold.  Station bookkeeping must be EXACT and the
    # route deterministic; the continuous residual is allowed <= 1e-5 of the corpus channel
    # std because the batched-vs-single TF forward cannot be made bit-identical (train-time
    # batch = 1600 branches; a single test case has 2).  Raw numbers are reported so the
    # author can overrule.
    TOL_REL_STD = 1e-5
    chstd = [27.31719937528972, 19.220414828238127, 32111.789696474763, 0.0025499401761537627]
    br = json.load(open(os.path.join(REPO, "VortFoil", "artifacts",
                                     "f0b_p1_corpus_20260903T072000Z", "build_report.json")))
    keys_exact = ("n_mid", "n_accept_v8", "n_accept", "n_live", "n_fit_fail", "n_rows",
                  "n_override")
    ps = profile_state_provider()
    percase = {}
    for ci in (0, 91, 9):
        cse, rmi = _corpus_case(z0, ci)
        base = z0["PAINT"][rmi]
        cpi = dict(cse); cpi["paint4"] = base.astype(np.float64)
        oa = assemble_field_features(cpi, model_state=ps,
                                     feature_mode="predicted_profile_prof4")
        ob = assemble_field_features(cpi, model_state=ps,
                                     feature_mode="predicted_profile_prof4")
        det = bool(np.array_equal(oa["feat"], ob["feat"]))
        r1 = z1["PAINT"][rmi]
        g1 = oa["feat"].astype(np.float32)
        dd = np.abs(g1 - r1)
        sb = oa["meta"]["station_bookkeeping"]
        pcr = next(p for p in br["per_case"] if p["case_id"] == ci)
        book = {k: dict(mine=(int(oa["meta"]["n_override"]) if k == "n_override" else
                              int(oa["meta"]["n_rows"]) if k == "n_rows" else int(sb[k])),
                        corpus=int(pcr[k])) for k in keys_exact}
        for k in book:
            book[k]["exact"] = book[k]["mine"] == book[k]["corpus"]
        bex = all(v["exact"] for v in book.values())
        relstd = [float(dd[:, c].max() / chstd[c]) for c in range(4)]
        percase[ci] = dict(
            case=cse["name"], n=int(len(r1)), max_abs_dev_float32=float(dd.max()),
            n_rows_differing=int((dd > 0).any(1).sum()),
            per_channel_max_abs=[float(dd[:, c].max()) for c in range(4)],
            per_channel_max_rel_to_corpus_std=relstd,
            max_rel_to_corpus_std=float(max(relstd)),
            bit_identical=bool(float(dd.max()) == 0.0),
            route_deterministic_within_config=det,
            station_bookkeeping_vs_build_report=book, station_bookkeeping_exact=bex,
            continuous_station_diagnostics={
                k: dict(mine=float(sb[k]), corpus=float(pcr[k]),
                        rel=abs(float(sb[k]) - float(pcr[k])) / max(abs(float(pcr[k])), 1e-30))
                for k in ("fit_resid_max", "delta_min", "delta_max")},
            meta=oa["meta"],
            case_PASS=bool(bex and det and max(relstd) <= TOL_REL_STD))
        p = percase[ci]
        print(f"[C2/case{ci}] {cse['name']}\n"
              f"     max|dev|={dd.max():.3e} rows_differing={p['n_rows_differing']} "
              f"bit_identical={p['bit_identical']} deterministic={det}\n"
              f"     per-channel max|dev|/corpus-std={['%.2e' % v for v in relstd]} "
              f"override={oa['meta']['n_override']} "
              f"fallback_bit_identical={oa['meta']['fallback_rows_bit_identical']}\n"
              f"     station bookkeeping exact={bex} "
              f"{ {k: v['mine'] for k, v in book.items()} } case_PASS={p['case_PASS']}")
    rep["C2_prof4_reproduces_P1_corpus"] = dict(
        tolerance_rel_corpus_std=TOL_REL_STD, per_case=percase,
        all_station_bookkeeping_exact=all(v["station_bookkeeping_exact"]
                                          for v in percase.values()),
        worst_rel_to_corpus_std=max(v["max_rel_to_corpus_std"] for v in percase.values()),
        degenerate_case_9_bit_identical=percase[9]["bit_identical"],
        residual_cause="batched-vs-single-case TF operator forward (XLA kernel / reduction "
                       "order depend on batch shape: 1600 branches at train time, 2 for a "
                       "single case).  Same weights, same input vector, same code, same "
                       "intra=6 thread width.  p and nut channels are EXACT; only prof_ux / "
                       "prof_uy, which read u_tau/Pi fitted to the operator's theta/delta*, "
                       "move.  Not reachable by any code change: a test case has 2 branches.",
        PASS=bool(all(v["case_PASS"] for v in percase.values())))
    print(f"[C2] verdict PASS={rep['C2_prof4_reproduces_P1_corpus']['PASS']} "
          f"worst_rel_to_std={rep['C2_prof4_reproduces_P1_corpus']['worst_rel_to_corpus_std']:.2e}")

    # ---- C2n CONTROL that must FAIL: perturb the frozen state operator ---------------
    case0, rm0 = _corpus_case(z0, 0)
    cp0 = dict(case0); cp0["paint4"] = z0["PAINT"][rm0].astype(np.float64)
    ps2 = Prof4StateProvider(); ps2.perturb_operator(1e-3)
    outp = assemble_field_features(cp0, model_state=ps2,
                                   feature_mode="predicted_profile_prof4")
    d2 = float(np.abs(outp["feat"].astype(np.float32) - z1["PAINT"][rm0]).max())
    rep["C2n_control_operator_perturbed"] = dict(
        max_abs_dev_float32=d2, differs=d2 > 0.0,
        vs_honest_route_max_dev=percase[0]["max_abs_dev_float32"],
        PASS=bool(d2 > 1e3 * max(percase[0]["max_abs_dev_float32"], 1e-30)))
    print(f"[C2n] CONTROL operator +0.1% must differ: max|dev|={d2:.3e} vs honest route "
          f"{percase[0]['max_abs_dev_float32']:.3e} "
          f"{'RED as required' if d2 > 0 else 'BROKEN CHECK'}")

    # ---- C3 FBR-(-1): prof4 for a case OUTSIDE the training corpus -------------------
    from supervised_field.nf_hybrid_field import _case_inputs
    assert nm not in set(str(s) for s in z0["names"]), "test case leaked into the corpus"
    ci = _case_inputs(nm, feature_mode="predicted_profile_prof4", model_state=ps)
    lg = _case_inputs(nm, feature_mode="legacy_paint4")
    diff = ci["PA"] != lg["PA"]
    rep["C3_out_of_corpus_case"] = dict(
        case=nm, in_train_corpus=False, shape=list(ci["PA"].shape),
        n_changed_rows=int(diff.any(1).sum()),
        frac_changed_rows=float(diff.any(1).mean()),
        fallback_bit_identical=bool(np.array_equal(ci["PA"][~diff.any(1)],
                                                   lg["PA"][~diff.any(1)])),
        nut_channel_untouched=bool(int(diff[:, 3].sum()) == 0),
        finite=bool(np.isfinite(ci["PA"]).all()),
        cl_src=ci["cl_src"], meta=ci["feature_meta"],
        PASS=bool(diff.any(1).sum() > 0 and np.isfinite(ci["PA"]).all()
                  and int(diff[:, 3].sum()) == 0))
    r3 = rep["C3_out_of_corpus_case"]
    print(f"[C3] out-of-corpus {nm}\n     changed_rows={r3['n_changed_rows']} "
          f"({100 * r3['frac_changed_rows']:.2f}%) nut_untouched={r3['nut_channel_untouched']} "
          f"fallback_bit_identical={r3['fallback_bit_identical']} PASS={r3['PASS']}")

    rep["ALL_PASS"] = all(v["PASS"] for k, v in rep.items()
                          if isinstance(v, dict) and "PASS" in v)
    rep["gates"] = {k: v["PASS"] for k, v in rep.items()
                    if isinstance(v, dict) and "PASS" in v}
    print(f"\n[field_features] ALL_PASS={rep['ALL_PASS']}")
    return rep


if __name__ == "__main__":
    if "--check" in sys.argv:
        out = check()
        if "--report" in sys.argv:
            p = sys.argv[sys.argv.index("--report") + 1]
            os.makedirs(os.path.dirname(p), exist_ok=False)
            json.dump(out, open(p, "w"), indent=1, default=float)
            print(f"[artifact] {p}")
        sys.exit(0 if out["ALL_PASS"] else 1)
    print(__doc__)
