# -*- coding: utf-8 -*-
"""nf_eval_shape.py — d-binned (all / near-wall / off-body) std-MSE eval for the
geometry-conditioned shape-encoder field model (_nf_shape_*.npz).

Numpy forward of the PointNet encoder + concat_ms non-res trunk.  The shape token is
built from the REAL surface (sim.surface for AirfRANS) via the SAME resample as
nf_extract_shape — NOT analytic_polyline.

  python -m supervised_field.nf_eval_shape --weights _nf_shape_s1.npz

ARTIFACT NAMING (fixed 2026-08-29).  This script used to dump to a single fixed filename,
`_nf_shape_eval.json`, so every run silently destroyed the previous model's record.  That is
how the *with-paint* arm of the 2026-07-21 paint ablation lost its stored evaluation: the
no-paint arm was evaluated second and overwrote it, leaving the paper's Table `tab:field`
row 2 with no primary artifact while row 3 kept one.  Output is now named after the weights
file, and an existing record is never clobbered -- a re-run writes a timestamped superseding
record beside it.  The legacy `_nf_shape_eval.json` is the 2026-07-21 *no-paint* arm
(full_test 200) and is left in place as that record.
"""
import os, sys, json, time, hashlib, datetime
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XFOIL_EXE", os.path.expanduser("~/src/xfoil/bin/xfoil"))
import numpy as np
# airfrans / pyvista are imported inside the functions that read AirfRANS files (2026-10-08), so the paint and the
# decoder forward run without them (code/field/predict_field.py).
from sdf2d.geometry import sdf_np
from supervised_field.nf_eval200 import ROOT, FIELDS, MARIO_PUBLISHED, HERE
from supervised_field.nf_hybrid_field import _case_inputs, _load
from supervised_field.nf_extract_shape import resample_le, M


def _swish(h):
    return h * (1.0 / (1.0 + np.exp(-h)))


def encode_np(tok, w):
    """tok (M,2) -> z (16). PointNet: Dense+swish x2, maxpool, Dense+swish, Dense."""
    h = _swish(tok @ w["enc_W0"] + w["enc_b0"])
    h = _swish(h @ w["enc_W1"] + w["enc_b1"])
    h = h.max(axis=0)                                   # global max-pool over points
    h = _swish(h @ w["enc_W2"] + w["enc_b2"])
    return h @ w["enc_W3"] + w["enc_b3"]                # (16,)


def fwd_np(X, s_row, PA, z, w):
    """X(n,13), s_row(n), PA raw paint(n,4), z(16) -> pred(n,4) physical units."""
    B = w["fourier_B"]; ob = float(w["ob"])
    # paint robust-std (saved clip/mean/std)
    clip = w["paint_clip"]; pm = w["paint_mean"]; ps = w["paint_std"]
    Pc = np.clip(PA, clip[:, 0][None, :], clip[:, 1][None, :])
    pf = (Pc - pm[None, :]) / ps[None, :]
    if int(w.get("no_paint", 0)):                    # paint-ablation weights: match zeroed training input
        pf = np.zeros_like(pf)
    ph = X[:, :2] @ B
    spatial = np.concatenate([np.sin(ph), np.cos(ph), X[:, 2:6]], axis=1)
    n = len(X)
    zt = np.tile(z[None, :], (n, 1))
    c = np.concatenate([X[:, 6:8], zt, s_row[:, None], pf], axis=1)
    h = np.concatenate([spatial, c], axis=1).astype(np.float64)
    for i in range(5):
        h = _swish(h @ w[f"trunk_W{i}"] + w[f"trunk_b{i}"])
    vp = h @ w["vp_W0"] + w["vp_b0"]
    nh = _swish(h @ w["nut_W0"] + w["nut_b0"])
    nh = nh @ w["nut_W1"] + w["nut_b1"]
    raw = np.concatenate([vp, nh], axis=1)
    net = ob * np.tanh(raw / ob)
    return net * w["std"][None, :] + w["t_mean"][None, :]


def main():
    import airfrans as af
    from supervised_field import WEIGHTS as _W
    wf = os.path.join(HERE, sys.argv[sys.argv.index("--weights") + 1]) if "--weights" in sys.argv \
        else os.path.join(_W, "field_decoder_f0cm_s1.npz")   # release name of the deployed operator-conditioned decoder
    w = {k: np.asarray(v, np.float64) for k, v in np.load(wf, allow_pickle=True).items()
         if np.asarray(v).dtype.kind in "fiu"}
    std = w["std"]
    with open(os.path.join(ROOT, "manifest.json")) as f:
        test = sorted(json.load(f)["full_test"])
    clbias = float(sys.argv[sys.argv.index("--clbias") + 1]) if "--clbias" in sys.argv else 0.0
    if clbias:  # M1 probe: recompute paint with cl_used+bias (match biased-trained weights)
        from supervised_field.airfrans_p3_fields_v2 import analytic_polyline, paint_points_v2
        print(f"[eval] --clbias +{clbias}: paint recomputed with cl_used+{clbias}", flush=True)
    # --adafpaint is GONE (2026-09-03).  It was a dangling import
    # (supervised_field.adaf_paint_extract lives only in _archive_session_2026-07/ and is
    # untracked, so this raised ImportError), it painted paint_points_v8 -- NOT the
    # registered prof4 feature -- and it was the nearest thing to a predicted-state
    # evaluation route, i.e. the thing most likely to be mistaken for P1-E1.  The one
    # canonical route is field_features.assemble_field_features via
    # nf_eval_percase --feature-mode predicted_profile_prof4.
    if "--adafpaint" in sys.argv:
        raise SystemExit("--adafpaint is removed: it was a dangling import and it painted "
                         "paint_points_v8, not the registered prof4 feature.  Use "
                         "`python -m supervised_field.nf_eval_percase --feature-mode "
                         "predicted_profile_prof4` (field_features.assemble_field_features).")
    rows = []; t0 = time.time()
    for i, nm in enumerate(test):
        ci = _case_inputs(nm); P = ci["P"]; TR = ci["TR"]; sm = ci["sm"]
        PA = ci["PA"].astype(np.float64)
        if clbias:
            U = ci["U"]; rs = paint_points_v2(P, ci["poly"], ci["a"], ci["cl_used"] + clbias, ci["Re"])
            PA = np.column_stack([rs["u"] * U, rs["v"] * U,
                                  rs["cp"] * 0.5 * U**2, rs["nut"] * U]).astype(np.float64)
            PA[sm, 0] = 0.0; PA[sm, 1] = 0.0; PA[sm, 3] = 0.0
        sim = af.Simulation(root=ROOT, name=nm)
        xy = np.asarray(sim.position, np.float64)[np.asarray(sim.surface, bool)][:, :2]
        tok = resample_le(xy, M, seed=0)
        z = encode_np(tok.astype(np.float64), w)
        pred = fwd_np(ci["X"].astype(np.float64), ci["s_row"].astype(np.float64),
                      PA, z, w)
        pred[sm, 0] = 0.0; pred[sm, 1] = 0.0; pred[sm, 3] = 0.0
        d = sdf_np(P[:, :2], ci["poly"])
        masks = dict(all=np.ones(len(P), bool), near=(d > 0) & (d < 0.05), off=d > 0.05)
        r = {mk: {k: float(np.mean(((pred[mm, c] - TR[mm, c]) / std[c]) ** 2))
                  for c, k in enumerate(FIELDS)} for mk, mm in masks.items()}
        rows.append(r)
        if (i + 1) % 50 == 0 or i + 1 == len(test):
            print(f"  [{i+1}/{len(test)}] ({time.time()-t0:.0f}s)", flush=True)

    def agg(mk):
        return {k: float(np.mean([row[mk][k] for row in rows])) for k in FIELDS}
    _h = hashlib.md5()
    with open(wf, "rb") as _f:
        for _ch in iter(lambda: _f.read(1 << 20), b""):
            _h.update(_ch)
    out = dict(weights=os.path.basename(wf), weights_md5=_h.hexdigest(),
               no_paint=int(w.get("no_paint", 0)),
               generated_utc=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
               n=len(rows),
               all=agg("all"), near_wall=agg("near"), off_body=agg("off"),
               mario=MARIO_PUBLISHED)
    # one record per weights file; never clobber an existing one (see module docstring)
    stem = os.path.splitext(os.path.basename(wf))[0].lstrip("_")
    outp = os.path.join(HERE, f"_nf_shape_eval_{stem}.json")
    if os.path.exists(outp):
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        outp = os.path.join(HERE, f"_nf_shape_eval_{stem}.superseding_{ts}.json")
        print(f"[artifact] prior record kept; writing superseding record -> {os.path.basename(outp)}",
              flush=True)
    json.dump(out, open(outp, "w"), indent=1, default=float)
    print(f"[artifact] {outp}", flush=True)
    print(f"\n=== {os.path.basename(wf)} d-binned std-MSE (full_test {len(rows)}) ===")
    for lab in ("all", "near_wall", "off_body"):
        dd = out[lab]; print(f"  {lab:10s} " + "  ".join(f"{k}={dd[k]:.4e}" for k in FIELDS), flush=True)
    print("  MARIO      " + "  ".join(f"{k}={MARIO_PUBLISHED[k]:.4e}" for k in FIELDS))
    print("  geomfree-B0 all ux7.3e-3 uy3.4e-2 p2.8e-1 nut5.6e-2 | near p4.5e-1 uy5.5e-2")
    print("  production(params) all ux1.8e-3 uy2.6e-3 p4.4e-3 nut1.0e-2")


if __name__ == "__main__":
    main()
