# -*- coding: utf-8 -*-
"""The deployed VortFoil surface model as one object.

One conditioning vector (16 CST coefficients, incidence, Reynolds number) feeds two operators in
parallel. The sheet operator returns an effective vortex sheet whose panel readout gives surface
pressure, circulation lift, moment, the stagnation point and the trailing-edge edge velocity; the
boundary-layer operator returns momentum and displacement thickness on each surface. They meet at
the trailing edge, where two frozen polynomial closures in the predicted state return profile drag
and a viscous-decambering correction to the circulation lift.

    import vortfoil
    vf = vortfoil.load_model()
    r = vf.predict(cst=cst16, alpha=4.0, Re=1e6)
    r.cl, r.cd, r.cm, r.cp, r.stagnation

Execution is a choice, not a property of the model: ``mode="graph"`` compiles the chain with
``tf.function`` and is roughly twice as fast as eager on a CPU and nine times faster at a single
query. ``device="gpu"`` runs the compiled chain on a GPU. See ``vortfoil.about()`` for the measured
costs and for the sealed scores this build reproduces.
"""
import json
import os
import warnings
from dataclasses import dataclass, field as _field

import numpy as np

from . import geometry

_DEFAULT_MAX_BATCH = {"cpu": 4096, "gpu": 512}   # the panel kernel materializes (B, 240, 241)

# Input domain of the training data: the XFOIL decks of both operators span these incidences and Reynolds
# numbers (449,732 sheet rows, 325,370 boundary-layer rows). The evaluation populations cover -4..12 deg and
# 2e5..3e6. Outside the training range every output is an extrapolation.
TRAIN_ALPHA_DEG = (-10.0, 18.0)
TRAIN_RE = (5e4, 1e7)
FIT_ERROR_WARN = 5e-3      # chord fraction, CST16 fit away from the TE node (about the 96th percentile of 1,664 UIUC sections)
TE_GAP_WARN = 1e-2         # chord fraction (about the 95th percentile of the same sections)


class RangeWarning(UserWarning):
    """An input lies outside the range the model was trained on, or the 16-parameter shape form misrepresents it."""


def _range_check(alpha, Re, fit_infos):
    """Warn (RangeWarning) about inputs outside the training domain; return the per-row flag."""
    a, r = np.asarray(alpha, np.float64).ravel(), np.asarray(Re, np.float64).ravel()
    bad_a = (a < TRAIN_ALPHA_DEG[0]) | (a > TRAIN_ALPHA_DEG[1])
    bad_r = (r < TRAIN_RE[0]) | (r > TRAIN_RE[1])
    if bad_a.any():
        warnings.warn(f"{int(bad_a.sum())} of {a.size} incidences lie outside the training range "
                      f"{TRAIN_ALPHA_DEG[0]:g} to {TRAIN_ALPHA_DEG[1]:g} deg (first: {a[bad_a][0]:g} deg); "
                      "the outputs there are extrapolations", RangeWarning, stacklevel=4)
    if bad_r.any():
        warnings.warn(f"{int(bad_r.sum())} of {r.size} Reynolds numbers lie outside the training range "
                      f"{TRAIN_RE[0]:.0e} to {TRAIN_RE[1]:.0e} (first: {r[bad_r][0]:.3g}); "
                      "the outputs there are extrapolations", RangeWarning, stacklevel=4)
    for i, f in enumerate(fit_infos or ()):
        if f["max_abs_error_excluding_te"] > FIT_ERROR_WARN:
            warnings.warn(f"shape {i}: the 16-parameter form deviates from the given coordinates by "
                          f"{100 * f['max_abs_error_excluding_te']:.2f}% chord; the outputs are for the fitted shape",
                          RangeWarning, stacklevel=4)
        if f["te_gap"] > TE_GAP_WARN:
            warnings.warn(f"shape {i}: trailing-edge gap {100 * f['te_gap']:.2f}% chord; the 16-parameter form "
                          "closes it (blunt trailing edges are out of scope)", RangeWarning, stacklevel=4)
    return bad_a | bad_r


def _weights_dir():
    override = os.environ.get("VORTFOIL_WEIGHTS")
    if override:
        return override
    try:                                          # normal installs
        from importlib.resources import files
        return str(files("vortfoil") / "weights")
    except Exception:                             # pragma: no cover - very old importlib
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


def manifest():
    """The deployed build's record: weight hashes, training recipe, evaluation populations, scores."""
    with open(os.path.join(_weights_dir(), "MANIFEST.json")) as fh:
        return json.load(fh)


@dataclass
class SurfacePrediction:
    """Every surface output of one evaluation. Arrays are (N,) unless stated."""
    cl: np.ndarray                 # deployed lift: circulation lift + trailing-edge-state closure
    cl_gamma: np.ndarray           # circulation lift of the sheet alone, 2 * oint gamma ds
    cd: np.ndarray                 # profile drag from the fitted closure of the trailing-edge state
    cm: np.ndarray                 # pressure-integral moment about the quarter chord
    cp: np.ndarray                 # (N, 240) surface pressure on the panel midpoints
    stagnation: np.ndarray         # arc-length fraction of the tangential-velocity sign change
    n_crossings: np.ndarray        # sign changes found; 1 is the well-posed case
    ue_te: np.ndarray              # edge velocity at the trailing edge, per unit freestream
    theta_te: np.ndarray           # summed momentum thickness at the trailing edge
    dstar_te: np.ndarray           # summed displacement thickness at the trailing edge
    info: dict = _field(default_factory=dict)

    def __len__(self):
        return len(self.cl)

    def as_dict(self):
        return {k: getattr(self, k) for k in
                ("cl", "cl_gamma", "cd", "cm", "cp", "stagnation", "n_crossings", "ue_te", "theta_te", "dstar_te")}


class VortFoil:
    """The deployed sheet operator, boundary-layer operator and the two frozen closures."""

    def __init__(self, mode="graph", device="cpu", max_batch=None, weights_dir=None):
        if mode not in ("graph", "eager"):
            raise ValueError("mode must be 'graph' or 'eager'")
        if device not in ("cpu", "gpu", "auto"):
            raise ValueError("device must be 'cpu', 'gpu' or 'auto'")
        if device == "cpu":
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        import tensorflow as tf
        self._tf = tf
        # The device must be fixed before anything builds a tensor: importing the operator modules
        # initializes TensorFlow's context, after which the visible-device list is frozen.
        gpus = tf.config.list_physical_devices("GPU")
        if device == "cpu" or (device == "auto" and not gpus):
            if gpus:
                tf.config.set_visible_devices([], "GPU")
            self.device, self.device_kind = "/CPU:0", "cpu"
        else:
            if not gpus:
                raise RuntimeError("device='gpu' requested but TensorFlow sees no GPU")
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            self.device = tf.config.list_logical_devices("GPU")[0].name
            self.device_kind = "gpu"
        from ._staged.op1_inviscid import build_op1
        from ._staged.op2_bl_surrogate import build_op2_surrogate
        from ._staged.common.geometry import build_single_loop_contour_cst_tf
        from ._staged.common.panel import compute_inviscid_surface_velocity
        from ._staged.common.wake_ode_tf import wake_ode_cd_tf
        from ._staged.common.lift_closure_tf import cl_closed_tf, load_lift_closure
        self.mode = mode
        self.max_batch = int(max_batch or _DEFAULT_MAX_BATCH[self.device_kind])
        w = weights_dir or _weights_dir()
        self.weights = {"sheet": os.path.join(w, "op1_supervised_eps0.50.scr_bd128.npz"),
                        "boundary_layer": os.path.join(w, "op2_m325_BT.npz"),
                        "lift_closure": os.path.join(w, "lift_closure_poly2.npz")}
        for k, p in self.weights.items():
            if not os.path.exists(p):
                raise FileNotFoundError(f"{k} weights not found: {p}")
        self._op1 = build_op1(n_surface_pts=121, branch_dim=128)
        self._op2 = build_op2_surrogate(n_surface_pts=121, enable_wake=False, branch_dim=128,
                                        branch_depth=3, trunk_depth=2, N_m=16)
        z2 = np.zeros((2, 16), np.float32)
        a2 = np.zeros((2, 1), np.float32)
        r2 = np.full((2, 1), 1e6, np.float32)
        self._op1.compute_losses(tf.constant(z2), tf.constant(a2), tf.constant(r2), training=False)
        self._op2.predict_bl(tf.constant(z2), tf.constant(a2), tf.constant(r2), training=False)
        ck = np.load(self.weights["sheet"])
        keys = sorted(x for x in ck.files if x.startswith("v_"))
        self._op1.field_decoder.build((None, int(ck[keys[-6]].shape[0])))
        self._op1.load_weights(self.weights["sheet"])
        self._op2.load_weights(self.weights["boundary_layer"], strict=False)
        self._lift = load_lift_closure(self.weights["lift_closure"])
        pi = np.pi

        def raw(c, a, r):
            geom = build_single_loop_contour_cst_tf(c, n_surface_pts=121)
            g = self._op1.predict_gamma(c, a, Re=r, training=False)
            ut = compute_inviscid_surface_velocity(geom, g["gamma_nodes"], a)["Ut"]
            cp = 1.0 - ut ** 2
            ca, sa = tf.cos(a * pi / 180.0), tf.sin(a * pi / 180.0)
            xc, yc, ds, nx, ny = geom["xc"], geom["yc"], geom["ds"], geom["nx"], geom["ny"]
            cl_gamma = 2.0 * tf.reduce_sum(g["gamma_panels"] * ds, axis=1)
            ue_te = 0.5 * (tf.abs(ut[:, 0]) + tf.abs(ut[:, -1]))
            cm = tf.reduce_sum((-(xc - 0.25) * (-cp * ny) + yc * (-cp * nx)) * ds, axis=1)
            sgn = tf.sign(ut)
            n_cross = tf.reduce_sum(tf.cast(sgn[:, 1:] * sgn[:, :-1] < 0, tf.int32), axis=1)
            bl = self._op2.predict_bl(c, a, r, training=False)
            th_u, th_l = bl["theta_u"][:, -1], bl["theta_l"][:, -1]
            d_u, d_l = bl["delta_u"][:, -1], bl["delta_l"][:, -1]
            theta_te, dstar_te = th_u + th_l, d_u + d_l
            return dict(cl=cl_closed_tf(cl_gamma, th_u, th_l, d_u, d_l, ue_te, self._lift),
                        cl_gamma=cl_gamma,
                        cd=wake_ode_cd_tf(theta_te, dstar_te, ue_te, closure="fitted_poly2"),
                        cm=cm, cp=cp, ut=ut, ue_te=ue_te, theta_te=theta_te, dstar_te=dstar_te,
                        n_crossings=n_cross)

        def grad(c, a, r, which):
            with tf.GradientTape() as tape:
                tape.watch(c)
                out = raw(c, a, r)
                gr = tape.gradient(tf.reduce_sum(out[which]), c)
            if gr is None:      # TF 2.10 can return None when the tape is read outside its context
                raise RuntimeError(f"gradient of {which} is not connected")
            return gr

        wrap = (lambda f: tf.function(f, reduce_retracing=True)) if mode == "graph" else (lambda f: f)
        self._raw = wrap(raw)
        self._grads = {k: wrap(lambda c, a, r, _k=k: grad(c, a, r, _k)) for k in ("cl", "cl_gamma", "cd")}

    # ---------------------------------------------------------------- inputs
    def _prep(self, cst, coords, alpha, Re):
        info = {}
        if (cst is None) == (coords is None):
            raise ValueError("give exactly one of cst= or coords=")
        if coords is not None:
            coords = np.asarray(coords, np.float64)
            if coords.ndim == 2:
                coords = coords[None, ...]
            fits, infos = zip(*(geometry.fit_cst16(c) for c in coords))
            cst = np.stack(fits)
            info["fit"] = list(infos)
            rot = np.array([f["rotation_deg"] for f in infos])
            if np.any(np.abs(rot) > 1e-9):
                info["chord_rotation_deg"] = rot
        cst = np.asarray(cst, np.float32)
        if cst.ndim == 1:
            cst = cst[None, :]
        if cst.shape[1] != 16:
            raise ValueError(f"cst must have 16 columns, got {cst.shape}")
        n = len(cst)
        alpha = np.broadcast_to(np.asarray(alpha, np.float32).reshape(-1, 1), (n, 1)).copy()
        Re = np.broadcast_to(np.asarray(Re, np.float32).reshape(-1, 1), (n, 1)).copy()
        if "chord_rotation_deg" in info:
            alpha = alpha - info["chord_rotation_deg"].reshape(-1, 1).astype(np.float32)
            info["alpha_used"] = alpha.ravel().copy()
        info["outside_training_range"] = _range_check(alpha, Re, info.get("fit"))
        return cst, alpha, Re, info

    def _chunks(self, n):
        for i in range(0, n, self.max_batch):
            yield i, min(i + self.max_batch, n)

    # ---------------------------------------------------------------- public
    def predict(self, cst=None, coords=None, alpha=0.0, Re=1e6):
        """Evaluate the chain. Give either ``cst`` (N,16) or ``coords`` ((N,)P,2); scalars broadcast.

        When ``coords`` is given the shape is fitted to the CST form first and the incidence is
        measured from the fitted chord line, so a rotated section is handled correctly; the fit
        residual and the rotation are returned in ``result.info``.
        """
        tf = self._tf
        cst, a, r, info = self._prep(cst, coords, alpha, Re)
        acc = {}
        with tf.device(self.device):
            for i, j in self._chunks(len(cst)):
                out = self._raw(tf.constant(cst[i:j]), tf.constant(a[i:j]), tf.constant(r[i:j]))
                for k, v in out.items():
                    acc.setdefault(k, []).append(v.numpy())
        res = {k: np.concatenate(v) for k, v in acc.items()}
        ut = res.pop("ut")
        return SurfacePrediction(stagnation=_stagnation(ut), info=info, **res)

    def sweep(self, cst, alphas, Re=1e6):
        """One shape over many incidences, as a single batch."""
        alphas = np.asarray(alphas, np.float32).ravel()
        cst = np.asarray(cst, np.float32).reshape(1, 16).repeat(len(alphas), 0)
        return self.predict(cst=cst, alpha=alphas, Re=Re)

    def gradients(self, cst=None, coords=None, alpha=0.0, Re=1e6, of="cl"):
        """Reverse-mode d(of)/d(CST16) through the whole chain; ``of`` in {'cl','cl_gamma','cd'}."""
        if of not in self._grads:
            raise ValueError("of must be 'cl', 'cl_gamma' or 'cd'")
        tf = self._tf
        cst, a, r, _ = self._prep(cst, coords, alpha, Re)
        out = []
        with tf.device(self.device):
            for i, j in self._chunks(len(cst)):
                out.append(self._grads[of](tf.constant(cst[i:j]), tf.constant(a[i:j]), tf.constant(r[i:j])).numpy())
        return np.concatenate(out)

    def __repr__(self):
        return f"VortFoil(mode={self.mode!r}, device={self.device_kind!r}, max_batch={self.max_batch})"


def _stagnation(ut):
    """Arc-length fraction of the tangential-velocity sign change nearest mid-arc."""
    sp = (np.arange(ut.shape[1]) + 0.5) / ut.shape[1]
    s = np.full(len(ut), np.nan)
    for i, row in enumerate(ut):
        idx = np.flatnonzero(np.diff(np.sign(row)) != 0)
        if idx.size == 0:
            continue
        j = idx[np.argmin(np.abs(sp[idx] - 0.5))]
        u0, u1 = row[j], row[j + 1]
        t = 0.0 if u1 == u0 else -u0 / (u1 - u0)
        s[i] = sp[j] + t * (sp[j + 1] - sp[j])
    return s
