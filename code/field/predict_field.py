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
    from supervised_field.nf_eval_shape import fwd_np
    pred = fwd_np(inp["X"].astype(np.float64), inp["s_row"].astype(np.float64), inp["PA"].astype(np.float64), z, w)
    pred[inp["sm"], 0] = 0.0; pred[inp["sm"], 1] = 0.0; pred[inp["sm"], 3] = 0.0
    return pred


_ZTRAIN = None


def shape_distance(z, w):
    """Distance of a shape code to the nearest training shape code, and the 99th percentile of the training sections'
    own nearest-neighbour distances (the scale of 'near')."""
    global _ZTRAIN
    if _ZTRAIN is None:
        from supervised_field.nf_eval_shape import encode_np
        with np.load(TOKENS, allow_pickle=True) as f:
            t = f["surf_tokens"].astype(np.float64)
        Z = np.stack([encode_np(k, w) for k in t])
        D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
        _ZTRAIN = (Z, float(np.percentile(D.min(1), 99)))
    Z, ref = _ZTRAIN
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


def prepare(loop, alpha, Re, decoder=DECODER):
    """Everything the decoder needs about the section and the flow condition; the query points come later.
    Coordinates with at least 2 x 220 nodes are used as given (the training sections had 441); sparser ones are
    resampled to 441 nodes through every given point (section_polygon). An open trailing edge is closed first."""
    msgs = []
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
    if not FIELD_ALPHA[0] <= a <= FIELD_ALPHA[1]:
        msgs.append(f"incidence {a:.2f} deg lies outside the decoder's training range {FIELD_ALPHA[0]:g} to {FIELD_ALPHA[1]:g} deg")
    if not FIELD_RE[0] <= Re <= FIELD_RE[1]:
        msgs.append(f"Reynolds number {Re:.3g} lies outside the decoder's training range {FIELD_RE[0]:.0e} to {FIELD_RE[1]:.0e}")
    if fit_err > FIT_WARN:
        msgs.append(f"the 16-parameter form deviates from the section by {100 * fit_err:.2f}% chord; the operator state "
                    "(C_l^gamma, s) is that of the fitted shape")
    if te_gap > 1e-2:
        msgs.append(f"trailing-edge gap {100 * te_gap:.2f}% chord, closed linearly along the chord (training sections had a "
                    "closed trailing edge); the field is for the closed section")
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        z = shape_code(poly, w)
        dz, dref = shape_distance(z, w)
    if dz > dref:
        msgs.append(f"the section's shape code lies {dz:.3g} from the nearest training section (training sections lie within "
                    f"{dref:.3g} of each other, 99th percentile): the section is unlike the NACA 4/5-digit training set")
    return dict(w=w, poly=poly, alpha=a, Re=float(Re), cl=st["cl"], s=st["s"], z=z, to_frame=to_frame, rot=rot,
                meta=dict(alpha_given=float(alpha), alpha_used=a, chord_rotation_deg=rot, Re=float(Re), cl_gamma=st["cl"],
                          s_sep=st["s"], section_nodes_given=int(len(g)), section_resampled=bool(not dense),
                          cst16_fit_max_error=fit_err, te_gap=te_gap, trailing_edge_closed_by=te_shift, shape_code_distance=dz, shape_code_reference=dref,
                          warnings=msgs, decoder=os.path.basename(decoder),
                          decoder_sha256=_sha256(decoder)))


def predict(loop, alpha, Re, points=None, window=WINDOW, h=H, frame="wind", decoder=DECODER, fast=True):
    """Field for a section given as a coordinate loop, at `points` (n,2) or on a grid over `window`, both in `frame`:
    'wind' (freestream along +x, the section at its incidence; default) or 'body' (the chord frame: LE at the origin,
    chord along +x). Velocity components are returned in the same frame; x_body, y_body give the points in the chord frame."""
    if frame not in ("wind", "body"):
        raise ValueError("frame is 'wind' or 'body'")
    if fast:
        import vf_field_fast as F
        F.install()
    c = prepare(loop, alpha, Re, decoder)
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
    return out, meta


# ------------------------------------------------------------------------------------------- output
STYLE = {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
         "font.size": 10, "xtick.direction": "in", "ytick.direction": "in"}


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
        cm = plt.get_cmap(cmap).copy(); cm.set_bad("0.25")                  # cells inside the section take its colour
        f = np.ma.masked_invalid(out[k])
        if kind == "log":
            f = np.ma.masked_invalid(np.clip(out[k], 1e-9, None))
            cf = ax.pcolormesh(out["x"], out["y"], f, cmap=cm, norm=LogNorm(1e-7, max(float(f.max()), 1e-6)), shading="auto", rasterized=True)
        elif kind == "sym":
            m = float(np.nanpercentile(np.abs(out[k]), 99.5))
            cf = ax.pcolormesh(out["x"], out["y"], f, cmap=cm, vmin=-m, vmax=m, shading="auto", rasterized=True)
        else:
            cf = ax.pcolormesh(out["x"], out["y"], f, cmap=cm, shading="auto", rasterized=True)
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
    fig.savefig(path, dpi=200)
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
    out, meta = predict(loop, a.alpha, a.Re, points=pts, window=tuple(float(v) for v in a.window.split(",")), h=a.h, frame=a.frame)
    np.savez_compressed(a.out, meta=json.dumps(meta), **out)
    print(f"wrote {a.out}: {np.size(out['u'])} points; C_l^gamma {meta['cl_gamma']:.4f}, s {meta['s_sep']:.3f}, "
          f"alpha used {meta['alpha_used']:.3f} deg, U {meta['U']:.2f} m/s; warnings: {len(meta['warnings'])}")
    if a.png and np.ndim(out["x"]) == 2:
        plot(out, meta, a.png)
        print(f"wrote {a.png}")
    elif a.png:
        print("--png draws the grid output only; skipped for --points")


if __name__ == "__main__":
    main()
