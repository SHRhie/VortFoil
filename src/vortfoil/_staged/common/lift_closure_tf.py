# -*- coding: utf-8 -*-
"""Deployed lift closure (2026-09-20): viscous decambering correction to the sheet's circulation lift, a degree-two
polynomial in the predicted trailing-edge state alone -- the lift analogue of the fitted drag closure.
    C_l = C_l^gamma + w . phi(C_l^gamma, theta_u,TE, theta_l,TE, delta*_u,TE, delta*_l,TE, u_e,TE)
phi = standardized [1, x_i, x_i x_j (i<=j)] (28 terms).  Coefficients are ridge-fitted once on the training deck
(VortFoil/code/vf_cl_decamber_closure.py --ntrain all) and frozen; loaded from an npz with mu, sd, w.
"""
import numpy as np
import tensorflow as tf

FEATURES = ("cl_circ", "theta_u_te", "theta_l_te", "dstar_u_te", "dstar_l_te", "ue_te")


def poly2_features_tf(F):
    """F: (N,6) -> (N,28): 1, x_i, x_i x_j (i<=j)"""
    F = tf.convert_to_tensor(F); n = F.shape[1]
    cols = [tf.ones_like(F[:, 0])] + [F[:, i] for i in range(n)] + [F[:, i] * F[:, j] for i in range(n) for j in range(i, n)]
    return tf.stack(cols, 1)


def load_lift_closure(path, dtype=tf.float32):
    z = np.load(path)
    return dict(mu=tf.constant(z["mu"], dtype), sd=tf.constant(z["sd"], dtype), w=tf.constant(z["w"], dtype))


def cl_closed_tf(cl_circ, th_u, th_l, ds_u, ds_l, ue, coef):
    """all (N,) tensors of one dtype; returns the closed lift (N,)"""
    X = poly2_features_tf(tf.stack([cl_circ, th_u, th_l, ds_u, ds_l, ue], 1))
    return cl_circ + tf.linalg.matvec((X - coef["mu"]) / coef["sd"], coef["w"])


if __name__ == "__main__":   # self-check: the feature layout matches the numpy fitter's and the closure is exactly the ridge model
    F = np.random.default_rng(0).normal(size=(5, 6)); X = poly2_features_tf(tf.constant(F, tf.float64)).numpy()
    ref = np.stack([np.ones(5)] + [F[:, i] for i in range(6)] + [F[:, i] * F[:, j] for i in range(6) for j in range(i, 6)], 1)
    assert X.shape == (5, 28) and np.allclose(X, ref); print("ok")
