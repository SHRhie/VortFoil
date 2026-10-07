"""Least-squares CST16 fit of a G241 contour, in the exact form staged/common/geometry.py evaluates.

G241 contract: node 0 upper TE, node 120 LE, node 240 lower TE (CCW).  OP1 with n_surface_pts=121
stitches its own loop in the same order, so a fitted CST16 lets OP1 be evaluated on a G241 shape.
Form (cst_geometry_batch_tf, 16-d coeffs -> no LE-weight, no TE-gap term):
    y_u(x) =  C(x) * sum_i a_u,i B_i(x),   y_l(x) = -C(x) * sum_i a_l,i B_i(x),
    C(x) = x^0.5 (1-x)^1.0,   B_i = Bernstein degree 7 (8 per surface).
A blunt TE cannot be represented (C(1)=0); the gate reports the error with and without the TE node.
"""
import numpy as np
from math import comb

N_BERN, N1, N2 = 8, 0.5, 1.0


def basis(x):
    x = np.asarray(x, np.float64); n = N_BERN - 1
    C = np.clip(x, 0, 1) ** N1 * (1.0 - np.clip(x, 0, 1)) ** N2
    B = np.stack([comb(n, i) * x**i * (1 - x)**(n - i) for i in range(N_BERN)], 1)
    return C[:, None] * B                                   # (P, 8)


def fit_g241(g):
    """g: (241,2) -> (cst16 float32, err dict).  Upper = nodes 120..0 (LE->TE), lower = nodes 120..240."""
    xu, yu = g[120::-1, 0], g[120::-1, 1]
    xl, yl = g[120:, 0], g[120:, 1]
    Pu, Pl = basis(xu), basis(xl)
    au = np.linalg.lstsq(Pu, yu, rcond=None)[0]
    al = np.linalg.lstsq(Pl, -yl, rcond=None)[0]
    ru = np.abs(Pu @ au - yu); rl = np.abs(-(Pl @ al) - yl)
    err = {"max": float(max(ru.max(), rl.max())),
           "max_excl_te": float(max(ru[:-1].max(), rl[:-1].max())),
           "rms": float(np.sqrt(np.mean(np.r_[ru, rl] ** 2))),
           "te_gap": float(abs(g[0, 1] - g[-1, 1]))}
    return np.r_[au, al].astype(np.float32), err


if __name__ == "__main__":
    # self-check: a CST16 shape rendered on the cosine grid must fit itself to fp precision
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # <release>/code
    import vortfoil_paths as VP; VP.add_paths()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf; tf.config.set_visible_devices([], "GPU")
    from staged.common.geometry import build_single_loop_contour_cst_tf
    # uiuc_cst.npz is the CST-16 training deck; it is NOT shipped in this release
    # (see README).  Point UIUC_CST_NPZ at it to run this self-check.
    UIUC_CST_NPZ = os.environ.get("UIUC_CST_NPZ", os.path.join(VP.DATA_EVAL, "uiuc_cst.npz"))
    c = np.load(UIUC_CST_NPZ, allow_pickle=True)["cst_test"][:5].astype(np.float32)
    geom = build_single_loop_contour_cst_tf(tf.constant(c), n_surface_pts=121)
    G = np.stack([geom["x_nodes"].numpy(), geom["y_nodes"].numpy()], -1)
    for i in range(5):
        cf, e = fit_g241(G[i])
        assert np.abs(cf - c[i]).max() < 1e-4, (i, np.abs(cf - c[i]).max())
        assert e["max"] < 1e-6, e
    print("self-check OK: CST16 -> loop -> fit recovers coefficients to <1e-4, recon <1e-6")
