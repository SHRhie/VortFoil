# -*- coding: utf-8 -*-
"""Airfoil geometry for the deployed model: the 16-parameter Kulfan/CST form it consumes, and a
least-squares fit from an arbitrary coordinate polyline onto that form.

The operators are defined on a fixed discretization: 16 CST coefficients (eight Bernstein weights per
surface, class exponents N1 = 0.5, N2 = 1, sharp trailing edge) rendered on a 241-node closed loop.
A shape given as coordinates is therefore *fitted* first, and the fit residual is reported so the
caller can see how much of the shape the representation could not carry.
"""
from math import comb

import numpy as np

N_BERN, N1, N2 = 8, 0.5, 1.0


def bernstein_basis(x):
    """Class/shape basis of the CST form, evaluated at x/c: (P, 8)."""
    x = np.asarray(x, np.float64)
    xc = np.clip(x, 0.0, 1.0)
    C = xc ** N1 * (1.0 - xc) ** N2
    B = np.stack([comb(N_BERN - 1, i) * xc ** i * (1.0 - xc) ** (N_BERN - 1 - i) for i in range(N_BERN)], 1)
    return C[:, None] * B


def fit_cst16(coords, normalize=True):
    """Fit a coordinate polyline to the 16-parameter CST form the operators consume.

    Parameters
    ----------
    coords : (P, 2) array
        Closed or open polyline in any frame. The leading edge is taken as the point farthest from
        the trailing edge, and the two legs are assigned so that the first is the upper surface.
    normalize : bool
        Rotate and scale onto the unit chord first (recommended for measured or rotated sections).
        The rotation applied is returned, because the incidence the model is asked for must be
        measured from the same chord line: use ``alpha - info["rotation_deg"]``.

    Returns
    -------
    cst16 : (16,) float32
    info : dict with ``max_abs_error``, ``max_abs_error_excluding_te``, ``rms_error``,
        ``rotation_deg``, ``chord``, ``te_gap``.

    Notes
    -----
    A blunt trailing edge cannot be represented, since the class function vanishes at x/c = 1;
    ``max_abs_error_excluding_te`` reports the fit away from that node so a blunt section is not
    judged by a residual the form cannot remove.
    """
    g = np.asarray(coords, np.float64)
    if g.ndim != 2 or g.shape[1] != 2:
        raise ValueError(f"coords must be (P,2), got {g.shape}")
    if len(g) < 2 * N_BERN:
        raise ValueError(f"need at least {2 * N_BERN} points, got {len(g)}")
    closed = np.allclose(g[0], g[-1])
    if closed:
        g = g[:-1]
    te = g[0] if closed else 0.5 * (g[0] + g[-1])          # closed: g[-1] is the last lower node, not the trailing edge
    d = np.hypot(*(g - te).T)
    li = int(np.argmax(d))
    le = g[li]
    chord_vec = te - le
    chord = float(np.hypot(*chord_vec))
    ang = float(np.arctan2(chord_vec[1], chord_vec[0]))
    if normalize:
        rot = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
        gn = ((g - le) @ rot.T) / chord
    else:
        gn = g
        ang = 0.0
    xu, yu = gn[li::-1, 0], gn[li::-1, 1]
    xl, yl = gn[li:, 0], gn[li:, 1]
    if yu.mean() < yl.mean():                       # the first leg must be the upper surface
        xu, yu, xl, yl = xl, yl, xu, yu
    Pu, Pl = bernstein_basis(xu), bernstein_basis(xl)
    au = np.linalg.lstsq(Pu, yu, rcond=None)[0]
    al = np.linalg.lstsq(Pl, -yl, rcond=None)[0]
    ru = np.abs(Pu @ au - yu)
    rl = np.abs(-(Pl @ al) - yl)
    info = {
        "max_abs_error": float(max(ru.max(), rl.max())),
        "max_abs_error_excluding_te": float(max(ru[:-1].max(), rl[:-1].max())),
        "rms_error": float(np.sqrt(np.mean(np.r_[ru, rl] ** 2))),
        "rotation_deg": float(np.degrees(ang)),
        "chord": chord,
        "te_gap": 0.0 if closed else float(abs(gn[0, 1] - gn[-1, 1])),
    }
    return np.r_[au, al].astype(np.float32), info


def surface(cst16, n_points=121):
    """Render the CST form as upper and lower surface coordinates on the model's own cosine grid."""
    cst16 = np.asarray(cst16, np.float64).reshape(-1, 16)
    beta = np.linspace(0.0, np.pi, n_points)
    x = 0.5 * (1.0 - np.cos(beta))
    P = bernstein_basis(x)
    upper = P @ cst16[:, :N_BERN].T
    lower = -(P @ cst16[:, N_BERN:].T)
    return x, upper.T.squeeze(), lower.T.squeeze()


def _self_check():
    """A CST shape rendered on the cosine grid must fit itself back to fp precision."""
    rng = np.random.default_rng(0)
    c = np.r_[rng.uniform(0.05, 0.25, 8), rng.uniform(0.02, 0.15, 8)].astype(np.float32)
    x, yu, yl = surface(c, 121)
    loop = np.r_[np.c_[x[::-1], yu[::-1]], np.c_[x[1:], yl[1:]]]
    fit, info = fit_cst16(loop, normalize=False)
    assert info["max_abs_error_excluding_te"] < 1e-6, info
    assert np.allclose(fit, c, atol=1e-5), np.abs(fit - c).max()
    print("geometry self-check PASS  max fit error %.2e" % info["max_abs_error_excluding_te"])


if __name__ == "__main__":
    _self_check()
