# -*- coding: utf-8 -*-
"""RANS coefficient correction: the surface branch's coefficients mapped to the RANS level (manuscript Section 2.2).

Every operator and closure stays frozen. Five networks (5-32-32-3, tanh), fitted on the 780 AirfRANS training cases
(NACA 4- and 5-digit sections, k-omega SST, alpha -5 to 15 deg, Re 2e6 to 6e6), read
chi = (C_l, ln C_d, C_m, alpha, log10 Re) and return the lift and moment residuals and the log drag ratio against RANS;
their mean is applied as

    C_l,RANS = C_l + f_l,    C_d,RANS = C_d exp(f_d),    C_m,RANS = C_m + f_m.

There is no shape or boundary-layer-state input. On the AirfRANS test split (200 cases) the mean relative lift and drag
errors fall from 7.7% and 25.9% to 2.9% and 4.3%, and the moment MAE from 0.0082 to 0.0028; on 107 unseen non-NACA
SST sections the drag error falls from 30.2% to 8.9% and the lift MAE from 0.037 to 0.028.

    r = vf.predict(coords=xy, alpha=4.0, Re=3e6, rans=True)
    r.cl_rans, r.cd_rans, r.cm_rans
"""
import os
import warnings

import numpy as np

FIT_ALPHA_DEG = (-5.0, 15.0)   # AirfRANS full_train incidences
FIT_RE = (2e6, 6e6)            # AirfRANS full_train Reynolds numbers
_W = {}


def _weights(weights_dir=None):
    from .api import _weights_dir
    path = os.path.join(weights_dir or _weights_dir(), "rans_correction_nn.npz")
    if path not in _W:
        with np.load(path) as z:
            _W[path] = {k: z[k] for k in z.files if k != "meta"}
    return _W[path]


def correct(cl, cd, cm, alpha, Re, weights_dir=None):
    """RANS-level (C_l, C_d, C_m) from the surface branch's deployed coefficients at incidence ``alpha`` (deg, the flow
    angle in the frame of the given shape, as passed to ``predict``) and Reynolds number ``Re``. Arrays broadcast.
    Warns (vortfoil.RangeWarning) where the condition lies outside the range the correction was fitted on."""
    from .api import RangeWarning
    cl, cd, cm, alpha, Re = np.broadcast_arrays(*(np.asarray(v, np.float64) for v in (cl, cd, cm, alpha, Re)))
    out = (alpha < FIT_ALPHA_DEG[0]) | (alpha > FIT_ALPHA_DEG[1]) | (Re < FIT_RE[0]) | (Re > FIT_RE[1])
    if out.any():
        warnings.warn(f"{int(out.sum())} of {out.size} conditions lie outside the range the RANS correction was fitted on "
                      f"(AirfRANS: {FIT_ALPHA_DEG[0]:g} to {FIT_ALPHA_DEG[1]:g} deg, Re 2e6 to 6e6); the corrected values "
                      "there are extrapolations", RangeWarning, stacklevel=2)
    W = _weights(weights_dir)
    z = (np.stack([cl, np.log(cd), cm, alpha, np.log10(Re)], -1) - W["x_mean"]) / W["x_std"]
    o = np.mean([np.tanh(np.tanh(z @ W["W1"][s] + W["b1"][s]) @ W["W2"][s] + W["b2"][s]) @ W["W3"][s] + W["b3"][s]
                 for s in range(len(W["W1"]))], 0) * W["y_std"] + W["y_mean"]
    return cl + o[..., 0], cd * np.exp(o[..., 1]), cm + o[..., 2]
