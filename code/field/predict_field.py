# -*- coding: utf-8 -*-
"""RANS field of the operator-conditioned decoder for any airfoil, incidence and Reynolds number.

    python code/field/predict_field.py --coords section.dat --alpha 4 --Re 3e6 --out field.npz --png field.png
    python code/field/predict_field.py --naca 2412 --alpha 4 --Re 3e6 --out field.npz

The decoder's inputs are built from the shape alone, the way they were built for training; no AirfRANS file and no
XFOIL run is needed:
  1. shape: the coordinates (Selig or Lednicer .dat; leading edge at the origin and unit chord along +x, the usual
     convention, else they are normalized). With at least 441 nodes (the training sections' count) they are used as
     given; sparser ones are resampled to 441 nodes through every given point (field_frame, section_polygon). Dense
     coordinates reproduce the deployed decoder; see the validation below.
  2. operator state: the sheet operator's circulation lift C_l^gamma and the separation scalar
     s = clip((max(H_TE^u, H_TE^l) - 2)/8, 0, 1) of the boundary-layer operator, from the section's 16-parameter fit
     (state_provider.ConfirmedStateProvider, max-side rule), as in training.
  3. physics paint: the panel solve on the section at the incidence with circulation C_l^gamma (paint_points_v2), the
     wall distance and the nearest panel's normal at every query point.
  4. shape code: the PointNet latent of 256 leading-edge-weighted points of the section (nf_extract_shape.resample_le).
  5. decoder: weights/field_decoder_f0cm_s1.npz (the deployed run) at the query points.
Units follow AirfRANS: chord 1 m, air with nu = 1.56e-5 m^2/s, so U = Re * nu; u, v in m/s, p as p/rho in m^2/s^2,
nu_t in m^2/s. Normalized copies are stored too: u/U, v/U, Cp = p/(U^2/2), nu_t/(U c).

Domain. The decoder was trained on 780 AirfRANS k-omega SST cases only: NACA 4- and 5-digit sections at incidences of
-5 to 15 deg and Reynolds numbers of 2e6 to 6e6. Outside those ranges, or for a section whose shape code lies farther from
every training section than the training sections lie from each other, a FieldRangeWarning is issued and recorded.
Families. Ten UIUC sections of other families, run with the AirfRANS protocol (k-omega SST, inside the ranges above),
gave mean errors 26 times the decoder's NACA test level for u_x (1.5 to 80 times per section: the near-NACA rae2822,
naca642415 and du84132v stay in the test tail, cambered high-lift and thin low-Re sections are far off), largest near the
wall; the shape-code distance did not flag any of them. So every section that is not given as a NACA designation
(family="naca") gets a FieldRangeWarning saying so.
The validation of this pathway against the deployed evaluation is code/field/validate_predict_field.py.
"""
import os, sys, json, hashlib, warnings
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # <release>/code
import vortfoil_paths as VP                         # noqa: E402
VP.add_paths()
sys.path.insert(0, HERE)

DECODER = os.path.join(VP.WEIGHTS, "field_decoder_f0cm_s1.npz")    # the deployed operator-conditioned decoder (run 1)
# Experimental, for personal checks (local branch exp/rext-decoder, not released): the same recipe retrained with 89 more
# NACA cases at Re 1e6-2e6 and 6e6-1e7 (2026-10-10). It did not pass its pre-registered test: better than the deployed run
# outside 2e6-6e6 on all four channels, but p and nu_t were not shown non-inferior inside. Used only when asked for.
DECODER_REXT = os.path.join(VP.WEIGHTS, "field_decoder_rext_s2.npz")
FIELD_RE_OF = {DECODER_REXT: (1e6, 1e7)}     # a decoder's training Reynolds range when it differs from FIELD_RE
TOKENS = os.path.join(VP.WEIGHTS, "airfrans_surface_tokens.npz")   # shape tokens of its training cases (shape check)
FIELD_ALPHA = (-5.0, 15.0)     # AirfRANS full_train incidences: -4.93 .. 14.93 deg
FIELD_RE = (2e6, 6e6)          # AirfRANS full_train: U 31.3 .. 93.6 m/s, chord 1 m, nu 1.56e-5 m^2/s
FIT_WARN = 5e-3                # chord fraction, as the surface package
WINDOW = (-0.5, 1.5, -0.5, 0.5)
H = 0.005
N_STATION = 220                # cosine intervals per surface of the training sections (airfrans_p3_fields_v2.analytic_polyline)


class FieldRangeWarning(UserWarning):
    """The request lies outside the decoder's training domain."""


# ------------------------------------------------------------------------------------------- shape
def read_coords(path):
    """Selig (one loop, TE -> upper -> LE -> lower -> TE) or Lednicer (point counts, then upper and lower from the LE)."""
    rows = []
    for line in open(path, errors="ignore"):
        p = line.replace(",", " ").split()
        if len(p) >= 2:
            try:
                rows.append((float(p[0]), float(p[1])))
            except ValueError:
                pass                                       # title line
    xy = np.asarray(rows, np.float64)
    if len(xy) and xy[0, 0] > 1.5 and xy[0, 1] > 1.5:     # Lednicer: "nu. nl." counts, both surfaces start at the LE
        nu, nl = int(xy[0, 0]), int(xy[0, 1])
        up, lo = xy[1:1 + nu], xy[1 + nu:1 + nu + nl]
        xy = np.r_[up[::-1], lo[1:]]
    if len(xy) < 10:
        raise ValueError(f"{path}: fewer than 10 coordinate pairs")
    return xy


def naca_coords(code):
    """The NACA 4- or 5-digit section exactly as the AirfRANS cases were generated (airfrans' generator, vendored)."""
    from naca_generator import naca_generator
    c = str(code).strip()
    if len(c) == 4:
        prm = (float(c[0]), float(c[1]), float(c[2:]))
    elif len(c) == 5:
        prm = (float(c[0]), float(c[1]), float(c[2]), float(c[3:]))     # design C_l = 0.15 L, max camber at P/20, Q reflex
    else:
        raise ValueError("--naca takes a 4- or 5-digit designation")
    s = np.asarray(naca_generator(np.array(prm), nb_samples=N_STATION, cosine_spacing=True, verbose=False, CTE=True))
    return s


def field_frame(loop, tol=0.01):
    """The decoder's frame is the AirfRANS one: leading edge at the origin, chord along +x, unit chord (the usual .dat
    convention, in which the incidence is given). Coordinates already in it are used as given. Otherwise they are
    translated, rotated and scaled with the provider's chord definition (LE = node farthest from the trailing-edge
    midpoint) and the rotation is returned so that the incidence can be measured from that chord.
    Returns (open loop in the frame, rotation deg, map for other points, LE index)."""
    g = np.asarray(loop, np.float64)
    closed = np.allclose(g[0], g[-1])
    if closed:
        g = g[:-1]
    te = g[0] if closed else 0.5 * (g[0] + g[-1]); li = int(np.argmin(g[:, 0]))   # closed: g[-1] is the last lower node
    if abs(g[li, 0]) < tol and abs(te[0] - 1.0) < tol and abs(te[1]) < tol:   # the nose of a cambered section may sit above y = 0
        return g, 0.0, (lambda P: np.asarray(P, np.float64)), li
    li = int(np.argmax(np.hypot(*(g - te).T))); le = g[li]
    c = te - le; ang = np.arctan2(c[1], c[0]); ch = np.hypot(*c)
    Rm = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
    to_frame = lambda P: ((np.asarray(P, np.float64) - le) @ Rm.T) / ch
    return to_frame(g), float(np.degrees(ang)), to_frame, li


def _natural_spline(t, y):
    """Natural cubic spline through (t, y), t strictly increasing; returns an evaluator (numpy only)."""
    n = len(t); h = np.diff(t)
    lo, di, up, r = np.zeros(n), np.ones(n), np.zeros(n), np.zeros(n)
    lo[1:-1], di[1:-1], up[1:-1] = h[:-1], 2.0 * (h[:-1] + h[1:]), h[1:]
    r[1:-1] = 3.0 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])
    for i in range(1, n):                                         # Thomas algorithm (first and last rows are c = 0)
        m = lo[i] / di[i - 1]; di[i] -= m * up[i - 1]; r[i] -= m * r[i - 1]
    c = np.zeros(n); c[-1] = r[-1] / di[-1]
    for i in range(n - 2, -1, -1):
        c[i] = (r[i] - up[i] * c[i + 1]) / di[i]
    b = (y[1:] - y[:-1]) / h - h * (2.0 * c[:-1] + c[1:]) / 3.0
    d = (c[1:] - c[:-1]) / (3.0 * h)

    def ev(q):
        k = np.clip(np.searchsorted(t, q) - 1, 0, n - 2); dq = q - t[k]
        return y[k] + dq * (b[k] + dq * (c[k] + dq * d[k]))
    return ev


def section_polygon(loop):
    """A sparse section on the training node count: a closed loop TE -> upper -> LE -> lower -> TE with 221 nodes per
    surface, cosine-spaced in arc length from the leading edge (the left-most node) to the trailing edge, on a natural
    cubic spline in arc length through every given point. The geometry therefore stays the user's; only its sampling
    changes. (The decoder reads the wall distance and the nearest panel of points inside the boundary layer, so the wall
    must not be smoothed, and the nose must stay resolved: on AirfRANS, abscissa-cosine stations or the training
    sections' own arc-length fractions resolved the nose of cambered sections worse and raised the near-wall errors;
    see code/field/validate_predict_field.py.) `loop` is in the field frame (field_frame)."""
    g = np.asarray(loop, np.float64)
    closed = np.allclose(g[0], g[-1])
    if not closed and np.hypot(*(g[0] - g[-1])) < 1e-12:
        closed = True
    li = int(np.argmin(g[:, 0]))
    if g[:li + 1, 1].mean() < g[li:, 1].mean():                   # first leg must be the upper surface
        g = g[::-1]; li = int(np.argmin(g[:, 0]))
    keep = np.r_[True, np.hypot(*np.diff(g, axis=0).T) > 1e-12]
    g = g[keep]; li = int(np.argmin(g[:, 0]))
    sarc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(g, axis=0).T))]
    X, Y = _natural_spline(sarc, g[:, 0]), _natural_spline(sarc, g[:, 1])
    beta = np.pi * np.linspace(1.0, 0.0, N_STATION + 1)             # 221 nodes per surface, TE -> LE
    legs = []
    for s0, s1 in ((sarc[li], sarc[0]), (sarc[li], sarc[-1])):        # LE -> TE along each surface
        sk = s0 + (s1 - s0) * 0.5 * (1.0 - np.cos(beta))              # cosine in arc length, TE -> LE: the nose stays resolved
        legs.append(np.c_[X(sk), Y(sk)])
    up, lo = legs                                                 # both TE -> LE
    poly = np.r_[up, lo[::-1][1:]]
    if closed:
        poly[-1] = poly[0]
    return poly


# ------------------------------------------------------------------------------------------- model
_PROV = None


def provider():
    global _PROV
    if _PROV is None:
        import state_provider as G
        _PROV = G.ConfirmedStateProvider(s_rule="teH")
    return _PROV


def load_decoder(path=DECODER):
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(v, np.float64) for k, v in z.items() if np.asarray(v).dtype.kind in "fiu"}


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def shape_code(nodes, w):
    from supervised_field.nf_extract_shape import resample_le, M
    from supervised_field.nf_eval_shape import encode_np
    g = np.asarray(nodes, np.float64)
    if np.allclose(g[0], g[-1]):
        g = g[:-1]
    return encode_np(resample_le(g, M, seed=0).astype(np.float64), w)


def field_inputs(P, poly, alpha, Re, cl, s, surf=None):
    """The decoder's per-point inputs, exactly as nf_hybrid_field._case_inputs assembles them (legacy_paint4), from a
    section polygon instead of an AirfRANS case. The NACA-parameter columns 8-12 of X are not read by the decoder
    (nf_eval_shape.fwd_np uses X[:, :8]) and are zero here."""
    from supervised_field.field_features import assemble_field_features
    from supervised_field.nf_extract import panel_mids_normals, A_NORM, RE_LO, RE_HI
    from supervised_field.airfrans_p3_fields_v2 import nearest_idx, NU_AIR
    P = np.asarray(P, np.float64)
    sm = np.zeros(len(P), bool) if surf is None else np.asarray(surf, bool)
    U = Re * NU_AIR
    ff = assemble_field_features(dict(P=P, poly=poly, alpha=alpha, Re=Re, U=U, cl_used=cl, surf=sm, name="predict_field"),
                                 feature_mode="legacy_paint4")
    r = ff["paint_raw"]
    mid, nrm = panel_mids_normals(poly)
    nn = nearest_idx(P, mid)
    n = len(P)
    X = np.zeros((n, 13), np.float64)
    X[:, 0], X[:, 1], X[:, 2] = P[:, 0], P[:, 1], r["d"]
    X[:, 3], X[:, 4], X[:, 5] = nrm[nn, 0], nrm[nn, 1], sm
    X[:, 6] = alpha / A_NORM
    X[:, 7] = 2.0 * (np.log(Re) - RE_LO) / (RE_HI - RE_LO) - 1.0
    return dict(X=X.astype(np.float32), s_row=np.full(n, s, np.float32), PA=ff["feat"], d=np.asarray(r["d"]), U=U, sm=sm)


def decode(inp, z, w):
    import vf_field_fast as F                    # nf_eval_shape.fwd_np over row blocks: memory bounded, rows independent
    pred = F.fwd_parallel(inp["X"].astype(np.float64), inp["s_row"].astype(np.float64), inp["PA"].astype(np.float64), z, w)
    pred[inp["sm"], 0] = 0.0; pred[inp["sm"], 1] = 0.0; pred[inp["sm"], 3] = 0.0
    return pred


_ZTRAIN = {}                       # per decoder: each encodes the training sections into its own code space


def shape_distance(z, w, key=DECODER):
    """Distance of a shape code to the nearest training shape code, and the 99th percentile of the training sections'
    own nearest-neighbour distances (the scale of 'near'); key names the decoder w (its training codes are cached)."""
    if key not in _ZTRAIN:
        from supervised_field.nf_eval_shape import encode_np
        with np.load(TOKENS, allow_pickle=True) as f:
            t = f["surf_tokens"].astype(np.float64)
        Z = np.stack([encode_np(k, w) for k in t])
        D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
        _ZTRAIN[key] = (Z, float(np.percentile(D.min(1), 99)))
    Z, ref = _ZTRAIN[key]
    return float(np.sqrt(((Z - z[None]) ** 2).sum(-1)).min()), ref


def grid(window=WINDOW, h=H):
    x0, x1, y0, y1 = window
    return np.meshgrid(np.arange(x0, x1 + 1e-9, h), np.arange(y0, y1 + 1e-9, h))


def wind_to_body(x, y, alpha_deg):
    """Wind frame (freestream along +x, the section at its incidence) -> the section's chord frame, about the quarter chord."""
    a = np.radians(alpha_deg); ca, sa = np.cos(a), np.sin(a)
    return 0.25 + ca * (x - 0.25) - sa * y, sa * (x - 0.25) + ca * y


def body_to_wind(x, y, alpha_deg):
    a = np.radians(alpha_deg); ca, sa = np.cos(a), np.sin(a)
    return 0.25 + ca * (x - 0.25) + sa * y, -sa * (x - 0.25) + ca * y


def close_trailing_edge(g):
    """Open loop (TE -> ... -> TE) with a trailing-edge gap -> the same loop with the two TE nodes meeting at their
    midpoint, the gap taken out linearly in x along each surface (as XFOIL's TGAP does). The training sections had a
    closed trailing edge; an open one adds a short panel across the gap that disturbs the paint at the trailing edge.
    Returns (closed loop, largest node displacement)."""
    g = np.asarray(g, np.float64).copy()
    if np.allclose(g[0], g[-1]):
        return g, 0.0
    li = int(np.argmin(g[:, 0])); mid = 0.5 * (g[0] + g[-1])
    d0 = g[0] - mid; d1 = g[-1] - mid
    w0 = np.clip((g[:li + 1, 0] - g[li, 0]) / max(g[0, 0] - g[li, 0], 1e-12), 0, 1)
    w1 = np.clip((g[li:, 0] - g[li, 0]) / max(g[-1, 0] - g[li, 0], 1e-12), 0, 1)
    g[:li + 1] -= w0[:, None] * d0[None, :]
    g[li:] -= w1[:, None] * d1[None, :]
    g[0] = g[-1] = mid
    return g, float(max(np.hypot(*d0), np.hypot(*d1)))


def _closed_upper_first(g):
    """Open or closed loop -> closed loop TE -> upper -> LE -> lower -> TE (the paint's node order)."""
    g = np.asarray(g, np.float64)
    if np.allclose(g[0], g[-1]):
        g = g[:-1]
    li = int(np.argmin(g[:, 0]))
    if g[:li + 1, 1].mean() < g[li:, 1].mean():
        g = g[::-1]
    return np.r_[g, g[:1]]


FAMILY_NOTE = ("the field decoder was trained on NACA 4- and 5-digit sections only; on ten UIUC sections of other "
               "families its errors were 1.5 to 80 times its NACA test level (26 times on average for u_x), largest near "
               "the wall: read this field as indicative")


def prepare(loop, alpha, Re, decoder=DECODER, family=None):
    """Everything the decoder needs about the section and the flow condition; the query points come later.
    Coordinates with at least 2 x 220 nodes are used as given (the training sections had 441); sparser ones are
    resampled to 441 nodes through every given point (section_polygon). An open trailing edge is closed first."""
    msgs = [] if family == "naca" else [FAMILY_NOTE]
    w = load_decoder(decoder)
    g, rot, to_frame, li = field_frame(loop)
    closed = np.allclose(np.asarray(loop, np.float64)[0], np.asarray(loop, np.float64)[-1])
    te_gap = 0.0 if closed else float(abs(g[0, 1] - g[-1, 1]))
    if rot:
        msgs.append(f"the coordinates were not in the chord frame (LE at the origin, chord 1 along +x); they were normalized "
                    f"and the incidence is measured from the chord through the node farthest from the trailing edge "
                    f"(rotation {rot:.2f} deg)")
    a = float(alpha) - rot
    te_shift = 0.0
    if not closed:
        g, te_shift = close_trailing_edge(g)                      # g now ends on its first node
        g = g[:-1]
    dense = len(g) >= 2 * N_STATION
    poly = _closed_upper_first(g) if dense else section_polygon(np.r_[g, g[:1]])
    prov = provider()
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):   # Accelerate BLAS raises spurious flags in matmul
        st = prov.state_for_polygon(poly, a, Re, source="section")  # the provider fits CST16 and measures its own chord
        _, _, fit_err = prov._cst(poly)
    a_ok = FIELD_ALPHA[0] <= a <= FIELD_ALPHA[1]
    if not a_ok:
        msgs.append(f"incidence {a:.2f} deg lies outside the decoder's training range {FIELD_ALPHA[0]:g} to {FIELD_ALPHA[1]:g} deg")
    re_lo, re_hi = FIELD_RE_OF.get(decoder, FIELD_RE)
    re_ok = re_lo <= Re <= re_hi
    if not re_ok:
        msgs.append(f"Reynolds number {Re:.3g} lies outside the decoder's training range {re_lo:.0e} to {re_hi:.0e}")
    if decoder == DECODER_REXT:
        msgs.append("experimental Re-extended decoder (trained to Re 1e6-1e7; not shown non-inferior in p and nu_t at Re "
                    "2e6-6e6): for checks, not the released result")
    if fit_err > FIT_WARN:
        msgs.append(f"the 16-parameter form deviates from the section by {100 * fit_err:.2f}% chord; the operator state "
                    "(C_l^gamma, s) is that of the fitted shape")
    if te_gap > 1e-2:
        msgs.append(f"trailing-edge gap {100 * te_gap:.2f}% chord, closed linearly along the chord (training sections had a "
                    "closed trailing edge); the field is for the closed section")
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        z = shape_code(poly, w)
        dz, dref = shape_distance(z, w, decoder)
    if dz > dref:
        msgs.append(f"the section's shape code lies {dz:.3g} from the nearest training section (training sections lie within "
                    f"{dref:.3g} of each other, 99th percentile): the section is unlike the NACA 4/5-digit training set")
    return dict(w=w, poly=poly, alpha=a, Re=float(Re), cl=st["cl"], s=st["s"], z=z, to_frame=to_frame, rot=rot,
                meta=dict(alpha_given=float(alpha), alpha_used=a, chord_rotation_deg=rot, Re=float(Re), cl_gamma=st["cl"],
                          s_sep=st["s"], section_nodes_given=int(len(g)), section_resampled=bool(not dense),
                          cst16_fit_max_error=fit_err, te_gap=te_gap, trailing_edge_closed_by=te_shift, shape_code_distance=dz, shape_code_reference=dref,
                          in_training_range=bool(a_ok and re_ok),       # incidence and Re (family and shape notes do not count)
                          warnings=msgs, family=family or "coordinates", decoder=os.path.basename(decoder),
                          decoder_sha256=_sha256(decoder)))


def predict(loop, alpha, Re, points=None, window=WINDOW, h=H, frame="wind", decoder=DECODER, fast=True, family=None, wall=False,
            bl=False):
    """Field for a section given as a coordinate loop, at `points` (n,2) or on a grid over `window`, both in `frame`:
    'wind' (freestream along +x, the section at its incidence; default) or 'body' (the chord frame: LE at the origin,
    chord along +x). Velocity components are returned in the same frame; x_body, y_body give the points in the chord frame.
    wall=True adds meta["wall"]: the decoder's wall values in its RANS state (see surface()); bl=True adds
    meta["boundary_layer"]: its boundary layer (see boundary_layer(); lists, NaN where no layer is read)."""
    if frame not in ("wind", "body"):
        raise ValueError("frame is 'wind' or 'body'")
    if fast:
        import vf_field_fast as F
        F.install()
    c = prepare(loop, alpha, Re, decoder, family)            # family="naca": a NACA 4/5-digit designation (no family warning)
    for m in c["meta"]["warnings"]:
        warnings.warn(m, FieldRangeWarning, stacklevel=2)
    a, poly = c["alpha"], c["poly"]
    if points is None:
        GX, GY = grid(window, h)
        Q = np.c_[GX.ravel(), GY.ravel()]
    else:
        Q = np.asarray(points, np.float64); GX = None
    P = np.c_[wind_to_body(Q[:, 0], Q[:, 1], a)] if frame == "wind" else Q
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):   # Accelerate BLAS raises spurious flags in matmul
        inp = field_inputs(P, poly, a, Re, c["cl"], c["s"])
        pred = decode(inp, c["z"], c["w"])
    inside = inp["d"] <= 0.0
    if not (np.isfinite(pred[~inside]).all() and np.isfinite(inp["PA"][~inside]).all()):
        raise FloatingPointError("non-finite paint or decoder output outside the section")
    pred[inside] = np.nan
    u, v = pred[:, 0], pred[:, 1]
    if frame == "wind":
        ca, sa = np.cos(np.radians(a)), np.sin(np.radians(a))
        u, v = u * ca + v * sa, -u * sa + v * ca
    U = inp["U"]
    out = dict(x=Q[:, 0], y=Q[:, 1], x_body=P[:, 0], y_body=P[:, 1], u=u, v=v, p=pred[:, 2], nut=pred[:, 3], inside=inside,
               u_over_U=u / U, v_over_U=v / U, cp=pred[:, 2] / (0.5 * U ** 2), nut_over_Uc=pred[:, 3] / U)
    if GX is not None:
        out = {k: val.reshape(GX.shape) for k, val in out.items()}
    sec = np.c_[body_to_wind(poly[:, 0], poly[:, 1], a)] if frame == "wind" else poly
    meta = dict(c["meta"], frame=frame, U=U, chord=1.0, nu=1.56e-5, section=sec.tolist())
    if wall or bl:
        w = _wall(c, Re)
        if wall:
            meta["wall"] = w
        if bl:
            meta["boundary_layer"] = _boundary_layer(c, Re, w)
    return out, meta




WALL_D = 2e-4                  # chords from the wall where the wall function reads the decoder's velocity (log layer)
KAPPA, B_LOG = 0.41, 5.0       # Spalding's law of the wall


def spalding_utau(ut, d, nu):
    """Friction velocity from Spalding's law, y+ = u+ + exp(-kB)(exp(k u+) - 1 - k u+ - (k u+)^2/2 - (k u+)^3/6), for the
    speed ut (> 0) at wall distance d; bisection, vectorized."""
    ut = np.abs(np.asarray(ut, np.float64)); lo, hi = np.full_like(ut, 1e-9), np.maximum(ut, 1e-6)
    for _ in range(80):
        mid = 0.5 * (lo + hi); up = ut / mid; ku = KAPPA * up
        f = d * mid / nu - (up + np.exp(-KAPPA * B_LOG) * (np.exp(np.minimum(ku, 50.0)) - 1 - ku - ku ** 2 / 2 - ku ** 3 / 6))
        lo, hi = np.where(f < 0, mid, lo), np.where(f < 0, hi, mid)
    return 0.5 * (lo + hi)


def _node_frame(P):
    """Nodes P (open loop) of a closed section -> (node tangent along the loop, node outward normal, segment lengths, ccw)."""
    Q = np.r_[P, P[:1]]
    t = np.diff(Q, axis=0)
    ccw = np.sum(Q[:-1, 0] * Q[1:, 1] - Q[1:, 0] * Q[:-1, 1]) > 0
    L = np.linalg.norm(t, axis=1); tu = t / L[:, None]
    tn = tu + np.roll(tu, 1, axis=0); tn /= np.linalg.norm(tn, axis=1, keepdims=True)              # node tangent, loop direction
    nn = np.c_[tn[:, 1], -tn[:, 0]] * (1.0 if ccw else -1.0)                                     # node outward normal
    return tn, nn, L, ccw


def _wall(c, Re):
    """Wall values from a prepared state (see surface()); JSON-safe."""
    poly, a = np.asarray(c["poly"], np.float64), c["alpha"]
    P = poly[:-1] if np.allclose(poly[0], poly[-1]) else poly
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):   # Accelerate BLAS raises spurious flags in matmul
        inp = field_inputs(P, poly, a, Re, c["cl"], c["s"], surf=np.ones(len(P), bool))
        pred = decode(inp, c["z"], c["w"])
    U = inp["U"]; cp = pred[:, 2] / (0.5 * U ** 2)
    Q, cq = np.r_[P, P[:1]], np.r_[cp, cp[:1]]
    t = np.diff(Q, axis=0)
    tn, nn, L, ccw = _node_frame(P)
    n_out = np.c_[t[:, 1], -t[:, 0]] * (1.0 if ccw else -1.0)               # outward normal times segment length
    F = -np.sum(0.5 * (cq[1:] + cq[:-1])[:, None] * n_out, axis=0)          # pressure force / (q c)
    # friction by a wall function: the decoder's tangential velocity WALL_D off the wall (log layer), Spalding's law for
    # u_tau, tau_w = u_tau^2 along the near-wall flow; the field's own wall gradient is not resolved (first nodes at y+ ~ 1)
    from supervised_field.airfrans_p3_fields_v2 import NU_AIR
    dsn = 0.5 * (L + np.roll(L, 1))
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        pv = decode(field_inputs(P + WALL_D * nn, poly, a, Re, c["cl"], c["s"]), c["z"], c["w"])
    ut = np.sum(pv[:, :2] * tn, axis=1)
    utau = spalding_utau(ut, WALL_D, NU_AIR)
    Ff = np.sum((np.sign(ut) * utau ** 2 * dsn)[:, None] * tn, axis=0) / (0.5 * U ** 2)        # friction force / (q c)
    cf = np.sign(np.sign(tn[:, 0]) * ut) * utau ** 2 / (0.5 * U ** 2)                          # > 0: near-wall flow toward the TE
    ca, sa = np.cos(np.radians(a)), np.sin(np.radians(a))
    le = int(np.argmin(P[:, 0]))
    upper = np.arange(len(P)) <= le
    if P[upper, 1].mean() < P[~upper, 1].mean():
        upper = ~upper
    cl_p, cd_p = float(-F[0] * sa + F[1] * ca), float(F[0] * ca + F[1] * sa)
    cl_f, cd_f = float(-Ff[0] * sa + Ff[1] * ca), float(Ff[0] * ca + Ff[1] * sa)
    w = dict(x=P[:, 0].tolist(), y=P[:, 1].tolist(), cp=cp.tolist(), cf=cf.tolist(), upper=upper.tolist(),
             cl=cl_p + cl_f, cd=cd_p + cd_f, cl_pressure=cl_p, cd_pressure=cd_p, cd_friction=cd_f,
             yplus_median=float(np.median(WALL_D * utau / NU_AIR)), wall_function_distance=WALL_D,
             reference_state="k-omega SST (the decoder's training fields)")
    w["cm"] = wall_cm(w)
    return w


def wall_cm(r):
    """c_m about the quarter chord (nose up positive, as the surface model's cm) of wall values x, y, cp, cf (surface()):
    the wall pressure and wall-function friction integrated as _wall integrates the forces (pressure on the segments,
    friction at the nodes along the wall toward the trailing edge; the same integration rebuilds surface()'s C_l and C_d
    exactly). Against the XFOIL state's cm, NACA 2412 at 0-8 deg: within 0.0095 at Re 3e6 and 0.0092 at Re 4.5e6
    (tests/test_wing.py holds it to 0.02)."""
    P = np.c_[r["x"], r["y"]]; Q = np.r_[P, P[:1]]; t = np.diff(Q, axis=0)
    tn, _, L, ccw = _node_frame(P)
    cq = np.r_[r["cp"], r["cp"][:1]]
    fp = -0.5 * (cq[1:] + cq[:-1])[:, None] * np.c_[t[:, 1], -t[:, 0]] * (1.0 if ccw else -1.0)    # pressure force / (q c)
    ff = (np.asarray(r["cf"]) * np.sign(tn[:, 0]) * 0.5 * (L + np.roll(L, 1)))[:, None] * tn       # friction force / (q c)
    arm = lambda p, f: np.sum((p[:, 0] - 0.25) * f[:, 1] - p[:, 1] * f[:, 0])                     # counterclockwise: nose down
    return float(-(arm(0.5 * (Q[1:] + Q[:-1]), fp) + arm(P, ff)))


def surface(loop, alpha, Re, decoder=DECODER, fast=True, family=None):
    """Wall values of the decoder, in its own reference state (k-omega SST, as its training fields).

    The pressure the decoder predicts on the section's nodes, flagged as wall nodes as in training (velocity and eddy
    viscosity are zero there), as C_p = p / (U^2/2), integrated around the closed section. Friction comes from a wall
    function, because the decoder does not resolve the wall-normal velocity gradient (MODEL_CARD.md, Field decoder): its
    tangential velocity WALL_D chords off the wall (log layer) gives the friction velocity by Spalding's law, and
    C_f = u_tau^2 / (U^2/2), positive where the near-wall flow runs toward the trailing edge. Forces are resolved normal and
    parallel to the freestream (chord 1). On the 200 AirfRANS test cases these values are compared with the RANS wall
    pressure and forces by code/field/validate_field_surface.py.
    Returns x, y (chord frame), cp, cf, upper (bool per node) as arrays; cl, cd (pressure + friction), cl_pressure,
    cd_pressure, cd_friction, cm (quarter chord, wall_cm); and the meta of predict()."""
    if fast:
        import vf_field_fast as F
        F.install()
    c = prepare(loop, alpha, Re, decoder, family)
    for m in c["meta"]["warnings"]:
        warnings.warn(m, FieldRangeWarning, stacklevel=2)
    w = _wall(c, Re)
    return dict({k: np.asarray(w[k]) for k in ("x", "y", "cp", "cf", "upper")},
                **{k: w[k] for k in ("cl", "cd", "cl_pressure", "cd_pressure", "cd_friction", "cm", "yplus_median")},
                meta=dict(c["meta"], frame="body", reference_state=w["reference_state"]))


# ------------------------------------------------------------------------------------------- boundary layer (RANS state)
# The layer is read off the decoder's field on wall-normal rays with the research gate's extraction (airfoil_operator,
# VortFoil/code/field_profile_gate.py: layer_edge and truth_state, audited by field_profile_extraction_audit.py), with one
# change: the profile given to the edge criterion is u_t / U_I, U_I the local inviscid speed from the field's own pressure
# (Bernoulli with the freestream total pressure), where the gate gave u_t itself. Over the curved front the inviscid speed
# falls off with the wall distance, and the vorticity integral of u_t then puts the edge out in the inviscid flow: on the RANS
# fields of the 200 AirfRANS test cases the gate's reading gave no layer (mostly theta < 0) at 74% of the stations at x/c < 0.1
# and 12% at 0.1-0.95, this one at 8% and 1.8%; where both read a layer at 0.1-0.95 they agree within 0.4% (median).
BL_BAND = 0.1          # chords: length of the wall-normal rays (the gate's band was 0.05, which cut off thick aft layers)
BL_EDGE_FRAC = 0.995   # layer edge: fraction of the corrected vorticity integral (the gate's EDGE_FRAC)
BL_X = (0.01, 0.99)    # x/c range of the stations: every second section node in it, per surface
BL_KEYS = ("theta", "dstar", "H", "Ue", "delta")


def _bl_distances(first=1e-6, ratio=1.15, steps=100):
    """Sample distances along a ray (chords): geometric from `first` until the step reaches BL_BAND / steps, then that step up
    to BL_BAND. The outer fifth of the samples, where layer_edge reads the outer flow, is the outer 31% of the ray (the last 32
    of 157, from 0.069 chord); bl_states drops an edge beyond 0.8 of the ray, so at least 20 of them lie outside the layer."""
    d = BL_BAND / steps
    g = first * ratio ** np.arange(int(np.ceil(np.log(d / ((ratio - 1) * first)) / np.log(ratio))) + 1)
    return np.r_[g, np.arange(g[-1] + d, BL_BAND + 0.5 * d, d)]


BL_N = _bl_distances()         # 157 distances


def layer_edge(n, u, frac=BL_EDGE_FRAC, outer=0.2):
    """Index of the edge of the layer in the profile u(n), or None (the gate's corrected criterion, verbatim): the integral of
    |du/dn| from the wall saturates at the edge of a rotational layer; the outer flow's small gradient (median over the outer
    `outer` fraction of the samples) is taken out linearly, and the edge is the first sample (not one of the first three)
    where the corrected integral reaches `frac` of its final value. A threshold relative to the wall gradient instead follows
    the Reynolds number, not the layer (tests/test_field_predict.py)."""
    g = np.abs(np.gradient(u, n))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(n))])
    k0 = int((1.0 - outer) * len(n))
    corr = cum - float(np.median(g[k0:])) * n
    if corr[-1] <= 0:
        return None
    k = np.where(corr >= frac * corr[-1])[0]
    k = k[k >= 3]
    return int(k[0]) if len(k) else None


def _trapz(f, x):
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(x)))       # np.trapezoid needs numpy 2


def layer_state(n, u, ke):
    """ue, theta, dstar, H, delta of the profile u(n) up to index ke with the wall (n = 0, u = 0) prepended (the gate's
    truth_state, verbatim); None when the edge speed is zero."""
    nn = np.concatenate([[0.0], n[:ke + 1]])
    uu = np.concatenate([[0.0], u[:ke + 1]])
    ue = float(uu[-1])
    if abs(ue) < 1e-9:
        return None
    r = uu / ue
    th = _trapz(r * (1 - r), nn)
    ds = _trapz(1 - r, nn)
    return dict(ue=ue, theta=th, dstar=ds, H=ds / max(th, 1e-12), delta=float(nn[-1]))


def bl_states(n, ut, un, p, U, n_upper=None):
    """The boundary layer on k rays from the field along them, each (k, m) at the distances n (m,) from the wall: tangential and
    normal velocity ut, un and kinematic pressure p (p / rho, zero in the freestream); U the freestream speed. Each ray is
    oriented so that its outer flow is positive, and its profile u_t / U_I, U_I = sqrt(U^2 - 2 p - u_n^2) the local inviscid
    speed, goes to layer_edge and layer_state. Returns (k,) arrays theta, dstar (chords), H, Ue (edge speed / U), delta (edge
    distance, chords), NaN where no layer is read (a non-finite sample, no edge, fewer than 7 samples in the layer, theta or
    delta* <= 0, or the edge beyond 0.8 of the ray, in the outer flow's window of layer_edge, _bl_distances), and separated
    (bool: the flow WALL_D off the wall runs against the outer flow). With n_upper, the rays are bl_rays' stations (the first
    n_upper on the upper surface, each surface from the leading edge), and those ahead of the stagnation point are not flagged:
    on the surface with the highest near-wall pressure, the run of flagged stations from the leading edge, whose near-wall
    flow runs toward the leading edge under an outer flow toward the trailing edge."""
    k = len(ut)
    out = {q: np.full(k, np.nan) for q in BL_KEYS}
    out["separated"] = np.zeros(k, bool)
    ui = np.sqrt(np.maximum(U ** 2 - 2.0 * np.asarray(p) - np.asarray(un) ** 2, 1e-12))
    for i in range(k):
        if not (np.all(np.isfinite(ut[i])) and np.all(np.isfinite(ui[i]))):
            continue
        r = ut[i] / ui[i]
        r = r if r[-1] >= 0 else -r
        out["separated"][i] = np.interp(WALL_D, n, r) < 0
        ke = layer_edge(n, r)
        st = layer_state(n, r, ke) if ke is not None and ke >= 6 else None
        if st is None or st["theta"] <= 0 or st["dstar"] <= 0 or st["delta"] > 0.8 * n[-1]:
            continue
        out["theta"][i], out["dstar"][i], out["H"][i], out["delta"][i] = st["theta"], st["dstar"], st["H"], st["delta"]
        out["Ue"][i] = abs(ut[i][ke]) / U
    if n_upper is not None:
        p0 = np.asarray(p, np.float64)[:, 0]
        if np.isfinite(p0).any():                                   # the stagnation point's surface: the highest near-wall pressure
            s = out["separated"][:n_upper] if np.nanargmax(p0) < n_upper else out["separated"][n_upper:]     # a view
            run = np.flatnonzero(~s)
            s[:run[0] if len(run) else len(s)] = False                # the flagged run from the leading edge
    return out


def bl_rays(poly, upper):
    """The stations of a closed section polygon (TE -> upper -> LE -> lower -> TE) with its nodes' upper flags (_wall): every
    second node with x/c in BL_X, upper surface then lower, each from the leading edge to the trailing edge. Returns the open
    node list's indices of the stations, the number of them on the upper surface, their node tangents and outward normals
    (k, 2), and the ray points (k, m, 2): each station plus its normal times BL_N."""
    P = poly[:-1] if np.allclose(poly[0], poly[-1]) else poly
    tn, nn, _, _ = _node_frame(P)
    up = np.asarray(upper, bool)
    sides = []
    for m in (up, ~up):
        i = np.flatnonzero(m & (P[:, 0] >= BL_X[0]) & (P[:, 0] <= BL_X[1]))
        sides.append(i[np.argsort(P[i, 0], kind="stable")][::2])
    i = np.r_[sides[0], sides[1]]
    return i, len(sides[0]), tn[i], nn[i], P[i, None, :] + BL_N[None, :, None] * nn[i, None, :]


def _boundary_layer(c, Re, w):
    """The boundary layer from a prepared state and its wall values w (_wall); JSON-safe (see boundary_layer())."""
    poly, a = np.asarray(c["poly"], np.float64), c["alpha"]
    i, k, tn, nn, X = bl_rays(poly, w["upper"])
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):   # Accelerate BLAS raises spurious flags in matmul
        inp = field_inputs(X.reshape(-1, 2), poly, a, Re, c["cl"], c["s"])
        pred = decode(inp, c["z"], c["w"])
    pred[inp["d"] <= 0] = np.nan                                          # a ray that crosses the section (thin trailing edge)
    pred = pred.reshape(len(i), len(BL_N), 4)
    ut = pred[..., 0] * tn[:, None, 0] + pred[..., 1] * tn[:, None, 1]
    un = pred[..., 0] * nn[:, None, 0] + pred[..., 1] * nn[:, None, 1]
    st = bl_states(BL_N, ut, un, pred[..., 2], inp["U"], k)
    x, cf = np.asarray(w["x"])[i], np.asarray(w["cf"])[i]
    out = {}
    for name, sl in (("upper", slice(0, k)), ("lower", slice(k, None))):
        out[name] = dict(x_over_c=x[sl].tolist(), Cf=cf[sl].tolist(), separated=st["separated"][sl].tolist(),
                         **{q: st[q][sl].tolist() for q in BL_KEYS})
    return out


def boundary_layer(loop, alpha, Re, decoder=DECODER, fast=True, family=None):
    """Boundary layer of the decoder's field, in its RANS state (k-omega SST, fully turbulent, as its training fields).

    Stations: every second node of the section with x/c in BL_X, on each surface. At each, the decoder's velocity and pressure
    on the wall-normal ray (BL_N, up to BL_BAND chords) give the layer edge and the momentum and displacement thicknesses
    (bl_states); C_f is the wall function's at that node, exactly surface()'s. Against the same extraction from the RANS
    fields of the 200 AirfRANS test cases (code/field/validate_field_boundary_layer.py), on attached stations at x/c 0.1-0.95:
    median relative error of theta and delta* 3.6% and 3.8% on the upper surface (90th percentile 19%), 6.4% and 5.8% on
    the lower (46% and 36%), H 1.1%; trailing edge (x/c > 0.95) 1-3%; leading edge (x/c < 0.1) 10% upper, 48% lower: not
    validated there.
    Returns upper, lower: dicts of (k,) arrays from the leading edge to the trailing edge: x_over_c, theta, dstar (chords),
    H, Ue (edge speed / U), delta (chords), Cf, separated (bool); NaN where no layer is read. And the meta of predict()."""
    if fast:
        import vf_field_fast as F
        F.install()
    c = prepare(loop, alpha, Re, decoder, family)
    for m in c["meta"]["warnings"]:
        warnings.warn(m, FieldRangeWarning, stacklevel=2)
    b = _boundary_layer(c, Re, _wall(c, Re))
    return dict({s: {k: np.asarray(v) for k, v in b[s].items()} for s in ("upper", "lower")}, meta=dict(c["meta"], frame="body"))

# ------------------------------------------------------------------------------------------- output
STYLE = {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
         "font.size": 10, "xtick.direction": "in", "ytick.direction": "in", "savefig.dpi": 400}


def _fill_inside(f):
    """Grid values inside the section (NaN) take the mean of their valid neighbours, layer by layer inwards, so that
    Gouraud shading blends the wall cells with nearby fluid values instead of a hole; the section is drawn over them."""
    f = np.array(f, np.float64)
    bad = ~np.isfinite(f)
    while bad.any():
        G = np.pad(~bad, 1); V = np.pad(np.where(bad, 0.0, f), 1)
        acc = V[:-2, 1:-1] + V[2:, 1:-1] + V[1:-1, :-2] + V[1:-1, 2:]
        cnt = G[:-2, 1:-1].astype(int) + G[2:, 1:-1] + G[1:-1, :-2] + G[1:-1, 2:]
        new = bad & (cnt > 0)
        if not new.any():
            break
        f[new] = acc[new] / cnt[new]
        bad &= ~new
    return f


def draw_fields(fig, axes, out, meta):
    """The four field panels (u along / across the freestream or u, v; C_p; nu_t) of a grid result on four axes."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    sec = np.asarray(meta["section"])
    wind = meta["frame"] == "wind"
    panels = [("u_over_U", r"$u_\parallel/U_\infty$" if wind else r"$u/U_\infty$", "viridis", None),
              ("v_over_U", r"$u_\perp/U_\infty$" if wind else r"$v/U_\infty$", "coolwarm", "sym"),
              ("cp", r"$C_p$", "coolwarm", "sym"), ("nut_over_Uc", r"$\nu_t/(U_\infty c)$", "YlGnBu", "log")]
    for ax, (k, lab, cmap, kind) in zip(axes, panels):
        cm = plt.get_cmap(cmap)
        kw = dict(cmap=cm, shading="gouraud", rasterized=True)          # values interpolated between the grid points
        if kind == "log":
            f = np.clip(out[k], 1e-9, None)
            cf = ax.pcolormesh(out["x"], out["y"], _fill_inside(f), norm=LogNorm(1e-7, max(float(np.nanmax(f)), 1e-6)), **kw)
        elif kind == "sym":
            m = float(np.nanpercentile(np.abs(out[k]), 99.5))
            cf = ax.pcolormesh(out["x"], out["y"], _fill_inside(out[k]), vmin=-m, vmax=m, **kw)
        else:
            cf = ax.pcolormesh(out["x"], out["y"], _fill_inside(out[k]), vmin=np.nanmin(out[k]), vmax=np.nanmax(out[k]), **kw)
        ax.fill(sec[:, 0], sec[:, 1], color="0.25", lw=0)
        ax.set_aspect("equal"); ax.set_title(lab)
        ax.set_xlim(out["x"].min(), out["x"].max()); ax.set_ylim(out["y"].min(), out["y"].max())
        fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.01)


def plot(out, meta, path):
    """Quick look of a grid result, in the frame it was computed in."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True, constrained_layout=True)
    draw_fields(fig, axes.ravel(), out, meta)
    fig.suptitle(rf"$\alpha$ = {meta['alpha_used']:.2f}$^\circ$, Re = {meta['Re']:.3g}   ($C_l^\gamma$ = {meta['cl_gamma']:.3f}, $s_{{sep}}$ = {meta['s_sep']:.2f})", fontsize=10)
    fig.savefig(path, dpi=400)
    plt.close(fig)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--coords", help="airfoil coordinates, Selig or Lednicer .dat")
    g.add_argument("--naca", help="NACA 4- or 5-digit designation, e.g. 2412 or 23012")
    ap.add_argument("--alpha", type=float, required=True, help="incidence, deg")
    ap.add_argument("--Re", type=float, required=True, help="Reynolds number (chord 1 m, nu 1.56e-5 m^2/s)")
    ap.add_argument("--window", default=",".join(f"{v:g}" for v in WINDOW), help="x0,x1,y0,y1 in chords (default %(default)s)")
    ap.add_argument("--h", type=float, default=H, help="grid spacing in chords (default %(default)s)")
    ap.add_argument("--points", help="optional .npy (n,2) query points instead of the grid")
    ap.add_argument("--frame", default="wind", choices=("wind", "body"),
                    help="frame of the grid, the points and the velocity components: wind (freestream along +x, default) or body (chord frame)")
    ap.add_argument("--out", default="field.npz")
    ap.add_argument("--png", help="optional quick-look figure")
    a = ap.parse_args(argv)
    loop = read_coords(a.coords) if a.coords else naca_coords(a.naca)
    pts = np.load(a.points) if a.points else None
    out, meta = predict(loop, a.alpha, a.Re, points=pts, window=tuple(float(v) for v in a.window.split(",")), h=a.h, frame=a.frame,
                        family="naca" if a.naca else None, wall=True)
    wl = meta["wall"]
    np.savez_compressed(a.out, meta=json.dumps(meta), wall_x=np.asarray(wl["x"]), wall_y=np.asarray(wl["y"]), wall_cp=np.asarray(wl["cp"]),
                        wall_cf=np.asarray(wl["cf"]), wall_upper=np.asarray(wl["upper"]), **out)
    print(f"wrote {a.out}: {np.size(out['u'])} points; C_l^gamma {meta['cl_gamma']:.4f}, s {meta['s_sep']:.3f}, "
          f"alpha used {meta['alpha_used']:.3f} deg, U {meta['U']:.2f} m/s; warnings: {len(meta['warnings'])}")
    print(f"wall values (RANS state): C_l {wl['cl']:.4f}, C_d {wl['cd']:.5f} (pressure {wl['cd_pressure']:.5f}, friction "
          f"{wl['cd_friction']:.5f} by the wall function)")
    if a.png and np.ndim(out["x"]) == 2:
        plot(out, meta, a.png)
        print(f"wrote {a.png}")
    elif a.png:
        print("--png draws the grid output only; skipped for --points")


if __name__ == "__main__":
    main()
