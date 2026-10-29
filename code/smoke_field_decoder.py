# -*- coding: utf-8 -*-
"""Smoke test of the AirfRANS-only operator-conditioned field decoder (weights/field_decoder_f0cm_s1.npz; deployed in 2.0.0).

Runs supervised_field.nf_eval_shape.encode_np + fwd_np on a synthetic 1,000-point query
with a synthetic shape token, and asserts the four output channels are finite and
correctly shaped.  No AirfRANS data is touched.

Run with the TF 2.18 environment (supervised_field imports the `airfrans` package):
    python code/smoke_field_decoder.py

The surface chain is checked separately by code/smoke_confirmed_chain.py, which must run
under TF 2.10 to reproduce the sealed numbers bit-for-bit.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vortfoil_paths as VP                                                # noqa: E402

VP.add_paths()
from supervised_field.nf_eval_shape import encode_np, fwd_np               # noqa: E402
from supervised_field.nf_extract_shape import M                            # noqa: E402

W = os.path.join(VP.WEIGHTS, "field_decoder_f0cm_s1.npz")
FIELDS = ("ux", "uy", "p", "nut")
N = 1000


def main():
    if not os.path.exists(W):
        print(f"FAIL  missing weights: {W}")
        return 1
    w = {k: v for k, v in np.load(W, allow_pickle=True).items()}
    zdim = int(w["z_dim"])
    rng = np.random.default_rng(0)

    # synthetic shape token: a NACA-0012-like closed loop resampled to M points
    t = np.linspace(0.0, 2.0 * np.pi, M, endpoint=False)
    x = 0.5 * (1.0 + np.cos(t))
    y = np.sign(np.sin(t)) * 0.6 * (0.2969 * np.sqrt(np.maximum(x, 0)) - 0.1260 * x
                                    - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)
    # NumPy's float32 matmul loop on Apple Accelerate raises spurious divide/overflow/invalid
    # RuntimeWarnings here; the finiteness assertions below are the real check.
    with np.errstate(all="ignore"):
        z = encode_np(np.stack([x, y], 1).astype(np.float64), w)

    # synthetic query, 13 columns in nf_hybrid_field._case_inputs' layout:
    # x, y, d, nx, ny, surface_flag, alpha/A_NORM, Re_norm, prm4/PSCALE (4), is_4param
    X = np.zeros((N, 13))
    X[:, 0] = rng.uniform(-2.0, 4.0, N)                       # x
    X[:, 1] = rng.uniform(-1.5, 1.5, N)                       # y
    X[:, 2] = np.abs(rng.normal(0.0, 0.5, N))                 # wall distance
    th = rng.uniform(0.0, 2.0 * np.pi, N)
    X[:, 3], X[:, 4] = np.cos(th), np.sin(th)                 # nearest-panel normal
    X[:, 5] = 0.0                                             # off-surface query
    X[:, 6] = rng.uniform(-1.0, 1.0, N)                       # alpha / A_NORM
    X[:, 7] = rng.uniform(-1.0, 1.0, N)                       # Re, log-normalised
    X[:, 8:12] = rng.uniform(-1.0, 1.0, (N, 4))               # NACA params / PSCALE
    X[:, 12] = 1.0                                            # 4-parameter shape
    H_te_u, H_te_l = 2.4, 4.4                                 # synthetic per-surface trailing-edge shape factors
    s_row = np.full(N, np.clip((max(H_te_u, H_te_l) - 2.0) / 8.0, 0.0, 1.0))   # separation scalar, rule "teH" (= 0.3)
    PA = rng.uniform(-1.0, 1.0, (N, 4))                       # raw paint channels

    with np.errstate(all="ignore"):
        pred = fwd_np(X, s_row, PA, z, w)

    ok = True
    for name, got, want in (("z shape   ", z.shape, (zdim,)),
                            ("pred shape", pred.shape, (N, len(FIELDS)))):
        good = got == want
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {name}  {got}  expected {want}")
    fin = bool(np.all(np.isfinite(z))) and bool(np.all(np.isfinite(pred)))
    ok &= fin
    print(f"{'PASS' if fin else 'FAIL'}  all finite   z {np.isfinite(z).all()}  "
          f"pred {np.isfinite(pred).all()}")
    for i, f in enumerate(FIELDS):
        print(f"      {f:4s} min {pred[:, i].min():+.4e}  max {pred[:, i].max():+.4e}  "
              f"mean {pred[:, i].mean():+.4e}")
    print("\nSMOKE " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
