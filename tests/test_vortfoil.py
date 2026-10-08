# -*- coding: utf-8 -*-
"""Tests for the installed package. The heavy ones need the evaluation grid; point at it with

    VORTFOIL_EVAL_DATA=data/eval/holdout_targets.npz pytest -q

Without it they are skipped and the fast checks still run.
"""
import os
import warnings

import numpy as np
import pytest

import vortfoil

EVAL = os.environ.get("VORTFOIL_EVAL_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                        "data", "eval", "holdout_targets.npz"))
# Agreement tolerances. Reductions over the 240 panels are done in float32, and their order
# depends on the batch size and on the TensorFlow build, so the same case evaluated alone, in a
# batch, or in a chunked batch agrees only to the last few digits. Measured spread across the
# supported builds: ~4e-6 on the lift and moment, ~4e-8 on the drag. These bounds sit an order of
# magnitude above that and well below the precision anything is reported to. They are absolute on
# purpose: a relative tolerance vanishes at the incidence where the lift crosses zero.
ATOL_CL = 1e-4
ATOL_CM = 1e-4
ATOL_CD = 1e-6

NACA2412_CST = np.array([0.1721, 0.1595, 0.2032, 0.1581, 0.2017, 0.1668, 0.1830, 0.1752,
                         0.1101, 0.0913, 0.1129, 0.0777, 0.1069, 0.0847, 0.0932, 0.0967], np.float32)


@pytest.fixture(scope="session")
def model():
    return vortfoil.load_model(mode="graph", device="cpu")


def r2(p, t):
    return float(1 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2))


# ---------------------------------------------------------------- geometry
def test_cst_fit_roundtrip():
    """A shape rendered from CST coefficients must fit back to them."""
    x, yu, yl = vortfoil.geometry.surface(NACA2412_CST, 121)
    loop = np.r_[np.c_[x[::-1], yu[::-1]], np.c_[x[1:], yl[1:]]]
    fit, info = vortfoil.geometry.fit_cst16(loop, normalize=False)
    assert info["max_abs_error_excluding_te"] < 1e-6
    assert np.allclose(fit, NACA2412_CST, atol=1e-5)


def test_fit_rejects_bad_shapes():
    with pytest.raises(ValueError):
        vortfoil.geometry.fit_cst16(np.zeros((5, 2)))
    with pytest.raises(ValueError):
        vortfoil.geometry.fit_cst16(np.zeros((50, 3)))


# ---------------------------------------------------------------- api shape and invariants
def test_predict_shapes(model):
    r = model.predict(cst=NACA2412_CST, alpha=4.0, Re=1e6)
    assert len(r) == 1 and r.cp.shape == (1, 240)
    for k, v in r.as_dict().items():
        assert np.all(np.isfinite(v)), k


def test_batch_matches_single(model):
    alphas = np.array([-4.0, 0.0, 4.0, 8.0])
    batch = model.sweep(NACA2412_CST, alphas)
    for i, a in enumerate(alphas):
        one = model.predict(cst=NACA2412_CST, alpha=float(a))
        assert abs(one.cl[0] - batch.cl[i]) < ATOL_CL
        assert abs(one.cd[0] - batch.cd[i]) < ATOL_CD


def test_chunking_is_transparent(model):
    """max_batch must not change any value."""
    small = vortfoil.VortFoil(mode="graph", device="cpu", max_batch=3)
    alphas = np.linspace(-4, 8, 7)
    a = model.sweep(NACA2412_CST, alphas)
    b = small.sweep(NACA2412_CST, alphas)
    assert np.allclose(a.cl, b.cl, rtol=0, atol=ATOL_CL)
    assert np.allclose(a.cd, b.cd, rtol=0, atol=ATOL_CD)
    assert np.allclose(a.cm, b.cm, rtol=0, atol=ATOL_CM)


def test_lift_closure_is_a_correction(model):
    """The deployed lift is the circulation lift plus a small state-dependent correction."""
    r = model.sweep(NACA2412_CST, np.linspace(-4, 10, 8))
    d = np.abs(r.cl - r.cl_gamma)
    assert np.all(d > 0) and np.all(d < 0.2)


def test_coords_route_and_frame_normalization(model):
    """A rotated, scaled section queried at the matching incidence gives the same lift."""
    x, yu, yl = vortfoil.geometry.surface(NACA2412_CST, 121)
    loop = np.r_[np.c_[x[::-1], yu[::-1]], np.c_[x[1:], yl[1:]]]
    base = model.predict(coords=loop, alpha=4.0)
    th = np.deg2rad(7.0)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    turned = model.predict(coords=(loop @ rot.T) * 1.5, alpha=11.0)
    assert abs(turned.info["fit"][0]["rotation_deg"] - 7.0) < 1e-3
    assert abs(turned.cl[0] - base.cl[0]) < 1e-4


def test_range_warnings(model):
    """Inputs outside the training range, or sections the 16-parameter form misrepresents, raise RangeWarning."""
    three = np.repeat(NACA2412_CST[None], 3, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", vortfoil.RangeWarning)          # the range ends are inside: silent
        r = model.predict(cst=three, alpha=[-10.0, 4.0, 18.0], Re=[5e4, 1e6, 1e7])
    assert not r.info["outside_training_range"].any()
    with pytest.warns(vortfoil.RangeWarning, match="incidences"):
        r = model.predict(cst=three[:2], alpha=[4.0, 25.0], Re=1e6)
    assert r.info["outside_training_range"].tolist() == [False, True]
    with pytest.warns(vortfoil.RangeWarning, match="Reynolds"):
        model.predict(cst=NACA2412_CST, alpha=4.0, Re=1e8)
    x, yu, yl = vortfoil.geometry.surface(NACA2412_CST, 121)
    blunt = np.r_[np.c_[x[::-1], yu[::-1] + 0.01 * x[::-1]], np.c_[x[1:], yl[1:] - 0.01 * x[1:]]]   # 2% chord gap
    with pytest.warns(vortfoil.RangeWarning) as rec:
        model.predict(coords=blunt, alpha=4.0)
    assert any("trailing-edge gap" in str(w.message) for w in rec)


def test_surface_distributions(model):
    """cp comes with its panel abscissae; the boundary-layer distributions end at the trailing-edge state."""
    r = model.predict(cst=np.repeat(NACA2412_CST[None], 2, 0), alpha=[0.0, 6.0], Re=1e6)
    assert r.x_cp.shape == r.cp.shape == (2, 240) and 0.0 <= r.x_cp.min() and r.x_cp.max() <= 1.0
    assert r.theta.shape == r.dstar.shape == (2, 2, len(vortfoil.BL_X))
    assert np.allclose(r.theta[:, :, -1].sum(1), r.theta_te, rtol=1e-5)
    assert np.allclose(r.dstar[:, :, -1].sum(1), r.dstar_te, rtol=1e-5)
    assert (r.dstar[:, :, 1:] > r.theta[:, :, 1:]).all()              # shape factor above one


def test_stagnation_moves_with_incidence(model):
    r = model.sweep(NACA2412_CST, np.array([-4.0, 0.0, 4.0, 8.0]))
    assert np.all(np.diff(r.stagnation) > 0)
    assert np.all(r.n_crossings == 1)


# ---------------------------------------------------------------- gradients
def test_gradients_match_finite_differences(model):
    """The reverse-mode shape gradient is the derivative of the function actually evaluated."""
    g = model.gradients(cst=NACA2412_CST, alpha=4.0, of="cl_gamma")[0]
    h = 5e-3
    fd = np.empty(16)
    for j in range(16):
        cp, cm = NACA2412_CST.copy(), NACA2412_CST.copy()
        cp[j] += h
        cm[j] -= h
        fd[j] = (model.predict(cst=cp, alpha=4.0).cl_gamma[0] - model.predict(cst=cm, alpha=4.0).cl_gamma[0]) / (2 * h)
    rel = np.linalg.norm(g - fd) / np.linalg.norm(fd)
    assert rel < 5e-2, rel


def test_gradient_targets(model):
    for of in ("cl", "cl_gamma", "cd"):
        g = model.gradients(cst=NACA2412_CST, alpha=4.0, of=of)
        assert g.shape == (1, 16) and np.all(np.isfinite(g)) and np.linalg.norm(g) > 0
    with pytest.raises(ValueError):
        model.gradients(cst=NACA2412_CST, of="nonsense")


# ---------------------------------------------------------------- build record
def test_manifest_matches_shipped_weights():
    import hashlib
    from vortfoil.api import _weights_dir
    m = vortfoil.manifest()
    for entry in m["weights"].values():
        p = os.path.join(_weights_dir(), entry["file"])
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == entry["sha256"], entry["file"]


def test_eager_and_graph_agree(model):
    eager = vortfoil.VortFoil(mode="eager", device="cpu")
    a = model.sweep(NACA2412_CST, np.linspace(-4, 8, 5))
    b = eager.sweep(NACA2412_CST, np.linspace(-4, 8, 5))
    assert np.allclose(a.cl, b.cl, rtol=0, atol=ATOL_CL)
    assert np.allclose(a.cd, b.cd, rtol=0, atol=ATOL_CD)


# ---------------------------------------------------------------- recorded scores
@pytest.mark.skipif(not os.path.exists(EVAL), reason="evaluation grid not available")
def test_reproduces_recorded_scores(model):
    z = np.load(EVAL, allow_pickle=True)
    keep = z["valid"] & (np.abs(z["cl_visc"][:, 0]) <= 3) & (z["cd_visc"][:, 0] > 1e-4) & (z["cd_visc"][:, 0] <= 0.1)
    for k in ("theta_u_te", "theta_l_te", "dstar_u_te", "dstar_l_te", "ue_te"):
        keep &= np.isfinite(z[k][:, 0]) & ((z[k][:, 0] != 0) if "theta" in k else True)
    ref = vortfoil.manifest()["evaluation"]["all_validity_filtered_rows"]
    assert int(keep.sum()) == ref["n"]
    r = model.predict(cst=z["cst"][keep], alpha=z["alpha"][keep].ravel(), Re=z["Re"][keep].ravel())
    cl_t, cd_t = z["cl_visc"][keep, 0], z["cd_visc"][keep, 0]
    assert abs(r2(r.cl, cl_t) - float(ref["scores"]["cl"])) < 2e-4
    assert abs(r2(r.cl_gamma, cl_t) - float(ref["scores"]["cl_gamma"])) < 2e-4
    assert abs(r2(np.clip(r.cd, 1e-4, 0.1), cd_t) - float(ref["scores"]["cd"])) < 2e-4
