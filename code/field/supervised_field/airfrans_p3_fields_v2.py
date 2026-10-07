# -*- coding: utf-8 -*-
"""supervised_field/airfrans_p3_fields_v2.py — field-construction v2 on the SAME 20
held-out AirfRANS full_test cases as airfrans_p3_fields.py (v1, untouched).  Three
upgrades over the v1 paint:

UPGRADE 4 — circulatory-only decambering (replaces v1's multiplicative gamma scale,
  the diagnosed flaw corr(gamma_scale, relL2_p) = -0.967):
  the panel RHS is linear in the freestream, so solve TWICE with unit freestreams
  U=(1,0) (alpha=0) and U=(0,1) (alpha=90).  gamma(a) = cos(a) gamma_c + sin(a) gamma_s
  is an EXACT panel solution at incidence a; pick the closed-form effective angle a_e
  with R sin(a_e + phi) = Gamma_target = cl_XTR/2 (unit chord), R = hypot(Gc, Gs),
  phi = atan2(Gc, Gs).  Induced field uses gamma(a_e), sigma(a_e); the uniform term
  keeps the TRUE freestream direction alpha.  Thickness (non-circulatory) solution is
  preserved exactly; the small no-penetration leak is O(alpha - a_e) ~ 1 deg, measured
  per case (max |V.n| at the control points), and the near-wall paint overrides the
  wall band anyway.

UPGRADE 2 — viscous-region pressure (BL theory dp/dn = 0):
  inside the near-wall band the SURFACE pressure (Nyquist-filtered readout) is
  impressed along the wall normal: Cp(node) = Cp_surf(nearest station), smoothstep-
  blended to the inviscid Bernoulli Cp at the band edge (same w(d) as the velocity
  paint).  Wake keeps the inviscid p (thin-wake impressment).

UPGRADE 1 — nu_t closure paint (NEW output):
  BL band (d < delta(s)): inner mixing length + van Driest,
    l = 0.41 d (1 - exp(-d+/26)),  d+ = d u_tau / nu_nd,  nu_nd = 1/Re,
    u_tau = ue sqrt(Cf/2),  Cf = 0.0576 Re_x^-0.2 (consistent with the delta closure),
    du_t/dn = ue/(N delta) (d/delta)^(1/N - 1)  (power-law profile, analytic),
    nut_inner = l^2 |du_t/dn|;  outer cap (Cebeci-Smith) nut_outer = 0.0168 ue delta*
    with delta* = delta/(N+1);  nut_BL = min(inner, outer).
  WAKE (|eta| < 3 sigma): constant-eddy-viscosity plane-wake closure
    nut_wake = u_def_centerline(xi) * sigma(xi) / 12.5   (Townsend R_T ~ 12.5),
    smoothstep-blended from the TE BL (outer-cap) value over xi in [0, 0.2].
  Elsewhere nut = 0.  Dimensionalized nut_phys = nut_nd * U * chord(=1).

EVAL (same 20 cases / masks as v1) -> supervised_field/_airfrans_p3_fields_v2.json:
  (a) relL2 (d>0.05c) ux, uy, p + surface-p relL2 + nut relL2 on BOTH the full fluid
      mask and a viscous-region mask (d<0.05c | wake band) where nut actually lives;
  (b) AirfRANS-protocol STANDARDIZED MSE (per-field mean/std from a 50-sim strided
      sample of scarce_train), FULL fluid volume (no d>0.05 mask), ux, uy, p, nut +
      surface p (volume-p stats);
  (c) v1 baseline means carried alongside.
"""
import os, json, time
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XFOIL_EXE", os.path.expanduser("~/src/xfoil/bin/xfoil"))
import numpy as np
from multiprocessing import Pool

import airfrans as af
import pyvista as pv
from airfrans.naca_generator import naca_generator
from sdf2d.panel_inviscid import panel_solve, _induced_at
from sdf2d.geometry import sdf_np
from supervised_field.airfrans_p1_labels import run_one          # resumable XTR cache

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AIRFRANS_ROOT",   # external dataset, not shipped with this release
                     os.path.join(os.path.dirname(HERE), "airfrans_data", "Dataset"))
OUT = os.path.join(HERE, "_airfrans_p3_fields_v2.json")
from supervised_field import WEIGHTS as _W   # <release>/weights
STATS_NPZ = os.path.join(_W, "airfrans_train_stats.npz")
NU_AIR = 1.56e-5
CELL = 0.0092          # nominal cell of the validated multishape paint grid (240x180)
CHUNK = 8000
N_CASES = 20
N_STATS = 50           # scarce_train sims used for the standardization stats

# v1 baseline means (supervised_field/_airfrans_p3_fields.json, 20/20 evaluated)
V1_BASE = dict(relL2_ux=0.0696, relL2_uy=0.1518, relL2_p=0.1906, relL2_p_surf=0.3522)


def case_params(name):
    p = name.split("_")
    return float(p[2]), float(p[3]), tuple(float(x) for x in p[4:])


def analytic_polyline(prm):
    s = np.asarray(naca_generator(np.array(prm), nb_samples=220, cosine_spacing=True,
                                  verbose=False, CTE=True), np.float64)
    if not np.allclose(s[0], s[-1]):
        s = np.vstack([s, s[:1]])
    return s


def field_uv(P, geom, gamma, sigma, Uinf, chunk=CHUNK):
    u = np.empty(len(P)); v = np.empty(len(P))
    for i0 in range(0, len(P), chunk):
        sl = slice(i0, i0 + chunk)
        usx, usy, uvx, uvy = _induced_at(P[sl], geom["p1"], geom["t"], geom["L"])
        u[sl] = Uinf[0] + (uvx * gamma[None, :]).sum(1) + sigma * usx.sum(1)
        v[sl] = Uinf[1] + (uvy * gamma[None, :]).sum(1) + sigma * usy.sum(1)
    return u, v


def sdf_chunked(P, nodes, chunk=CHUNK):
    d = np.empty(len(P))
    for i0 in range(0, len(P), chunk):
        d[i0:i0 + chunk] = sdf_np(P[i0:i0 + chunk], nodes)
    return d


def nearest_idx(P, Q, chunk=CHUNK):
    nn = np.empty(len(P), int)
    for i0 in range(0, len(P), chunk):
        sl = slice(i0, i0 + chunk)
        nn[sl] = np.argmin(((P[sl, None, :] - Q[None, :, :])**2).sum(-1), axis=1)
    return nn


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def decamber_gamma(nodes, alpha_deg, cl_visc):
    """UPGRADE 4.  Two unit-freestream panel solves -> closed-form effective angle
    a_e whose EXACT panel solution carries the viscous circulation cl_visc/2
    (unit chord).  Returns gamma(a_e), sigma(a_e), geom, plus diagnostics."""
    sol_c = panel_solve(nodes, 0.0)        # U = (1, 0)
    sol_s = panel_solve(nodes, 90.0)       # U = (0, 1)
    geom = sol_c["geom"]; L = geom["L"]
    g_c = np.asarray(sol_c["gamma"]); s_c = float(sol_c["sigma"])
    g_s = np.asarray(sol_s["gamma"]); s_s = float(sol_s["sigma"])
    Gc = float(np.sum(g_c * L)); Gs = float(np.sum(g_s * L))

    a_true = np.radians(alpha_deg)
    cl_panel = 2.0 * (np.cos(a_true) * Gc + np.sin(a_true) * Gs)   # = v1 cl_panel

    Gam_t = 0.5 * cl_visc                  # Cl = 2 Gamma / chord, chord = 1
    R = float(np.hypot(Gc, Gs)); phi = float(np.arctan2(Gc, Gs))
    r = float(np.clip(Gam_t / max(R, 1e-12), -1.0, 1.0))
    clipped = abs(Gam_t / max(R, 1e-12)) > 1.0
    cands = [_wrap(np.arcsin(r) - phi), _wrap(np.pi - np.arcsin(r) - phi)]
    a_e = min(cands, key=lambda c: abs(_wrap(c - a_true)))

    gamma = np.cos(a_e) * g_c + np.sin(a_e) * g_s
    sigma = np.cos(a_e) * s_c + np.sin(a_e) * s_s
    cl_eff = 2.0 * (np.cos(a_e) * Gc + np.sin(a_e) * Gs)           # == cl_visc
    Uinf_true = np.array([np.cos(a_true), np.sin(a_true)])
    return dict(gamma=gamma, sigma=sigma, geom=geom, Uinf=Uinf_true,
                a_e_deg=float(np.degrees(a_e)),
                dalpha_deg=float(np.degrees(_wrap(a_e - a_true))),
                cl_panel=float(cl_panel), cl_eff=float(cl_eff),
                asin_clipped=bool(clipped))


def paint_points_v2(P, surf, alpha, cl_visc, Re):
    """v1 paint_points + the three upgrades.  Returns painted (u,v), painted Cp,
    nondim nu_t, wall distance d, wake-band mask, surface Cp, diagnostics."""
    nodes = surf if np.allclose(surf[0], surf[-1]) else np.vstack([surf, surf[:1]])
    dec = decamber_gamma(nodes, alpha, cl_visc)
    geom = dec["geom"]; Uinf = dec["Uinf"]
    gamma = dec["gamma"]; sigma = dec["sigma"]

    # ---- no-penetration leak of the mixed (true-freestream + a_e-induced) field ----
    mids = geom["mid"]
    lu, lv = field_uv(mids, geom, gamma, sigma, Uinf)
    vn_leak = float(np.max(np.abs(lu * geom["n"][:, 0] + lv * geom["n"][:, 1])))

    # ---- flat-plate fully-turbulent BL state at panel mids (same closure as v1) ----
    xq = np.maximum(mids[:, 0], 0.02)
    delta_s = 0.37 * xq / (Re * xq) ** 0.2
    N_s = np.full_like(delta_s, 7.0)
    off = 1.5 * CELL
    pn = geom["n"].copy()
    ctr = mids.mean(0)
    pn *= np.sign(np.sum((mids - ctr) * pn, axis=1, keepdims=True))   # outward
    ue_u, ue_v = field_uv(mids + off * pn, geom, gamma, sigma, Uinf)
    ue_s = np.clip(np.hypot(ue_u, ue_v), 0.2, 1.8)

    # ---- panel field at the sample points ----
    u, v = field_uv(P, geom, gamma, sigma, Uinf)
    d = sdf_chunked(P, nodes)
    nn = nearest_idx(P, mids)
    delt = delta_s[nn]; Nex = N_s[nn]

    Vm = np.sqrt(u**2 + v**2); big = Vm > 1.8
    u[big] *= 1.8 / Vm[big]; v[big] *= 1.8 / Vm[big]
    u_inv, v_inv = u.copy(), v.copy()        # unpainted panel field (Bernoulli Cp)

    # ---- near-wall power-law velocity reconstruction (same as v1) ----
    tpan = geom["t"]
    sgn = np.sign(tpan @ Uinf); sgn[sgn == 0] = 1.0
    tdir = tpan * sgn[:, None]
    f = np.clip(d / np.maximum(delt, 1e-9), 0, 1) ** (1.0 / Nex)
    u_nw = ue_s[nn] * f * tdir[nn, 0]; v_nw = ue_s[nn] * f * tdir[nn, 1]
    Drec = np.maximum(delt, 1.5 * CELL)
    w = np.clip((d - 0.7 * Drec) / (0.3 * Drec), 0, 1); w = w * w * (3 - 2 * w)
    near = (d > 0) & (d < Drec)
    u[near] = (1 - w[near]) * u_nw[near] + w[near] * u[near]
    v[near] = (1 - w[near]) * v_nw[near] + w[near] * v[near]
    u[d <= 0] = 0.0; v[d <= 0] = 0.0         # on/inside the wall: no-slip

    # ---- wake: streamline from TE + self-similar Gaussian momentum deficit ----
    th = delta_s * N_s / ((N_s + 1) * (N_s + 2))
    th_u, th_l = float(th[0]), float(th[-1]); th_tot = th_u + th_l
    s0 = nodes[:-1]
    te = s0[np.argmax(s0[:, 0])]
    xmax = float(P[:, 0].max())

    def vel_at(pts):
        return field_uv(np.atleast_2d(pts), geom, gamma, sigma, Uinf)

    ds_ = 0.02; C = [te + 0.01 * Uinf]
    for _ in range(400):
        p0 = C[-1][None, :]
        u0, v0 = vel_at(p0); V0 = np.hypot(u0, v0) + 1e-9
        ph = p0 + 0.5 * ds_ * np.column_stack([u0 / V0, v0 / V0])
        u1, v1 = vel_at(ph); V1 = np.hypot(u1, v1) + 1e-9
        nxt = (p0 + ds_ * np.column_stack([u1 / V1, v1 / V1]))[0]
        C.append(nxt)
        if nxt[0] > xmax + 0.1:
            break
    C = np.asarray(C)
    T = np.gradient(C, axis=0); T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    xi_c = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
    wkm = (d > 0) & (P[:, 0] > te[0] - 0.02)
    Pw = P[wkm]
    j = nearest_idx(Pw, C, chunk=20000)
    eta = T[j, 0] * (Pw[:, 1] - C[j, 1]) - T[j, 1] * (Pw[:, 0] - C[j, 0])
    xi = xi_c[j]
    sig_u = float(delta_s[0]) / 4 + 0.3 * np.sqrt(np.maximum(th_u * xi, 0))
    sig_l = float(delta_s[-1]) / 4 + 0.3 * np.sqrt(np.maximum(th_l * xi, 0))
    sig_side = np.where(eta >= 0, sig_u, sig_l); sbar = 0.5 * (sig_u + sig_l)
    s2pi = sbar * np.sqrt(2 * np.pi); spi = sbar * np.sqrt(np.pi)
    disc = np.maximum(s2pi**2 - 4 * spi * th_tot, 0)
    D_mom = np.clip((s2pi - np.sqrt(disc)) / (2 * spi), 0, 0.9)
    D = np.clip(D_mom + (0.6 - D_mom) * np.exp(-xi / 0.15), 0, 0.95)
    fac = 1 - D * np.exp(-eta**2 / (2 * sig_side**2))
    u[wkm] *= fac; v[wkm] *= fac

    # ---- surface Cp readout (Nyquist-filtered gamma, 2e-3 offset — same as v1) ----
    gs = 0.5 * gamma + 0.25 * (np.roll(gamma, 1) + np.roll(gamma, -1))
    su, sv = field_uv(mids + 2e-3 * pn, geom, gs, sigma, Uinf)
    Vt = su * tpan[:, 0] + sv * tpan[:, 1]
    cp_surf = 1.0 - Vt**2

    # ---- UPGRADE 2: viscous-region pressure (impress surface p along the normal) ----
    cp = 1.0 - (u_inv**2 + v_inv**2)         # inviscid Bernoulli Cp (clipped field)
    band_p = d < Drec                        # includes d<=0 (wall nodes -> surface Cp)
    cp[band_p] = (1 - w[band_p]) * cp_surf[nn[band_p]] + w[band_p] * cp[band_p]
    # wake: inviscid p kept (thin-wake impressment) — no change.

    # ---- UPGRADE 1: nu_t closure paint (nondim, units of U*chord) ----
    nut = np.zeros(len(P))
    bl = (d > 0) & (d < delt)                # BL band: d < delta(s)
    if bl.any():
        db = d[bl]; ueb = ue_s[nn[bl]]; deb = delt[bl]; Nb = Nex[bl]
        Rex = Re * xq[nn[bl]]
        Cf = 0.0576 * Rex ** -0.2            # flat-plate, consistent with delta closure
        utau = ueb * np.sqrt(0.5 * Cf)
        nu_nd = 1.0 / Re
        dplus = db * utau / nu_nd
        l_mix = 0.41 * db * (1.0 - np.exp(-dplus / 26.0))
        q = np.clip(db / deb, 1e-9, 1.0)
        dutdn = ueb / (Nb * deb) * q ** (1.0 / Nb - 1.0)
        nut_in = l_mix**2 * np.abs(dutdn)
        nut_out = 0.0168 * ueb * deb / (Nb + 1.0)     # Cebeci-Smith outer cap
        nut[bl] = np.minimum(nut_in, nut_out)
    # wake closure, blended from the TE outer-cap value
    band_w = np.abs(eta) < 3.0 * sig_side
    nut_te = 0.0168 * 0.5 * (ue_s[0] * delta_s[0] / (N_s[0] + 1.0)
                             + ue_s[-1] * delta_s[-1] / (N_s[-1] + 1.0))
    wb = np.clip(xi / 0.2, 0, 1); wb = wb * wb * (3 - 2 * wb)
    nut_wake = (1 - wb) * nut_te + wb * (D * sig_side / 12.5)      # Townsend R_T~12.5
    idx_w = np.flatnonzero(wkm)[band_w]
    nut[idx_w] = np.maximum(nut[idx_w], nut_wake[band_w])
    nut[d <= 0] = 0.0

    wake_band = np.zeros(len(P), bool); wake_band[idx_w] = True

    return dict(u=u, v=v, cp=cp, nut=nut, d=d, mids=mids, cp_surf=cp_surf,
                wake_band=wake_band, cl_panel=dec["cl_panel"], cl_eff=dec["cl_eff"],
                a_e_deg=dec["a_e_deg"], dalpha_deg=dec["dalpha_deg"],
                vn_leak=vn_leak, asin_clipped=dec["asin_clipped"])


def rel_l2(pred, truth, mask):
    num = float(np.sqrt(np.sum((pred[mask] - truth[mask])**2)))
    den = float(np.sqrt(np.sum(truth[mask]**2)))
    return num / (den + 1e-12)


def train_stats():
    """Per-field mean/std over a deterministic 50-sim strided sample of scarce_train
    (every 4th of the 200 sorted names — spans the full U range; approximation to
    the official full-train-set normalization, stated in the JSON)."""
    if os.path.exists(STATS_NPZ):
        z = np.load(STATS_NPZ, allow_pickle=True)
        return {k: (float(z[k][0]), float(z[k][1])) for k in ("ux", "uy", "p", "nut")}
    with open(os.path.join(ROOT, "manifest.json")) as f:
        names = sorted(json.load(f)["scarce_train"])[::4][:N_STATS]
    acc = {k: [0.0, 0.0, 0] for k in ("ux", "uy", "p", "nut")}
    t0 = time.time()
    for i, nm in enumerate(names):
        m = pv.read(os.path.join(ROOT, nm, nm + "_internal.vtu"))
        U = np.asarray(m.point_data["U"][:, :2], np.float64)
        arrs = dict(ux=U[:, 0], uy=U[:, 1],
                    p=np.asarray(m.point_data["p"], np.float64).ravel(),
                    nut=np.asarray(m.point_data["nut"], np.float64).ravel())
        for k, a in arrs.items():
            acc[k][0] += float(a.sum()); acc[k][1] += float((a**2).sum()); acc[k][2] += a.size
        if (i + 1) % 10 == 0:
            print(f"  [stats] {i+1}/{len(names)} sims  ({time.time()-t0:.0f}s)", flush=True)
    out = {}
    for k, (s, ss, n) in acc.items():
        mu = s / n; sd = float(np.sqrt(max(ss / n - mu * mu, 1e-30)))
        out[k] = (float(mu), sd)
    np.savez(STATS_NPZ, **{k: np.array(v) for k, v in out.items()})
    return out


def main():
    with open(os.path.join(ROOT, "manifest.json")) as f:
        names = sorted(json.load(f)["full_test"])[:N_CASES]

    print(f"[v2] standardization stats over {N_STATS} scarce_train sims...", flush=True)
    stats = train_stats()
    for k, (mu, sd) in stats.items():
        print(f"  [stats] {k:4s} mean={mu:+.4e} std={sd:.4e}", flush=True)

    tasks = []
    for nm in names:
        U, a, prm = case_params(nm)
        tasks.append((prm, a, U * 1.0 / NU_AIR))
    print(f"[v2] XFOIL XTR labels for {len(tasks)} cases (cached/parallel)...", flush=True)
    with Pool(6) as pool:
        labels = pool.map(run_one, tasks)

    rows = []
    for nm, lab in zip(names, labels):
        U, a, prm = case_params(nm)
        Re = U * 1.0 / NU_AIR
        if lab["cl"] is None:
            print(f"  SKIP {nm}  (XFOIL XTR failed, a={a:+.2f})", flush=True)
            rows.append(dict(name=nm, U=U, alpha=a, Re=Re, skipped=True,
                             reason="xfoil_xtr_failed"))
            continue
        cl_visc = float(lab["cl"])
        t0 = time.time()
        sim = af.Simulation(root=ROOT, name=nm)
        P = np.asarray(sim.position, np.float64)
        ut = np.asarray(sim.velocity, np.float64)
        pt = np.asarray(sim.pressure, np.float64).ravel()
        nt = np.asarray(sim.nu_t, np.float64).ravel()
        surf_mask = np.asarray(sim.surface, bool)

        surf = analytic_polyline(prm)
        r = paint_points_v2(P, surf, a, cl_visc, Re)

        ux_pred = r["u"] * U; uy_pred = r["v"] * U
        p_pred = r["cp"] * 0.5 * U**2                 # painted (UPGRADE 2)
        nut_pred = r["nut"] * U                       # * chord(=1) — kinematic m^2/s
        # AirfRANS surface nodes are exactly on the wall: no-slip / zero nut there
        ux_pred[surf_mask] = 0.0; uy_pred[surf_mask] = 0.0; nut_pred[surf_mask] = 0.0

        d = r["d"]
        mask = d > 0.05
        visc = (d < 0.05) | r["wake_band"]            # where nut actually lives
        full = np.ones(len(P), bool)

        e_ux = rel_l2(ux_pred, ut[:, 0], mask)
        e_uy = rel_l2(uy_pred, ut[:, 1], mask)
        e_p = rel_l2(p_pred, pt, mask)
        e_nut_full = rel_l2(nut_pred, nt, full)
        e_nut_visc = rel_l2(nut_pred, nt, visc)

        Ps = P[surf_mask]
        js = nearest_idx(Ps, r["mids"])
        p_surf_pred = r["cp_surf"][js] * 0.5 * U**2
        p_surf_true = pt[surf_mask]
        e_ps = float(np.sqrt(np.sum((p_surf_pred - p_surf_true)**2))
                     / (np.sqrt(np.sum(p_surf_true**2)) + 1e-12))

        # ---- (b) AirfRANS-protocol standardized MSE, FULL fluid volume ----
        mse = {}
        for k, pr, tr in (("ux", ux_pred, ut[:, 0]), ("uy", uy_pred, ut[:, 1]),
                          ("p", p_pred, pt), ("nut", nut_pred, nt)):
            sd = stats[k][1]
            mse[k] = float(np.mean(((pr - tr) / sd) ** 2))
        mse["p_surf"] = float(np.mean(((p_surf_pred - p_surf_true) / stats["p"][1]) ** 2))

        rows.append(dict(name=nm, U=U, alpha=a, Re=Re, skipped=False,
                         cl_xfoil_xtr=cl_visc, cl_panel=r["cl_panel"],
                         cl_eff=r["cl_eff"], a_e_deg=r["a_e_deg"],
                         dalpha_deg=r["dalpha_deg"], vn_leak=r["vn_leak"],
                         asin_clipped=r["asin_clipped"],
                         n_nodes=int(len(P)), n_mask=int(mask.sum()),
                         n_visc=int(visc.sum()), n_surf=int(surf_mask.sum()),
                         relL2_ux=e_ux, relL2_uy=e_uy, relL2_p=e_p,
                         relL2_p_surf=e_ps, relL2_nut_full=e_nut_full,
                         relL2_nut_visc=e_nut_visc,
                         nut_pred_max=float(nut_pred.max()),
                         nut_true_max=float(nt.max()),
                         mse_std=mse))
        print(f"  {nm[:48]:48s} a={a:+6.2f} da={r['dalpha_deg']:+5.2f} "
              f"leak={r['vn_leak']:.3f}  ux={e_ux:.4f} uy={e_uy:.4f} p={e_p:.4f} "
              f"ps={e_ps:.4f} nut(v/f)={e_nut_visc:.3f}/{e_nut_full:.3f}  "
              f"nutmax {nut_pred.max():.4f}/{nt.max():.4f}  "
              f"mse ux={mse['ux']:.3e} p={mse['p']:.3e} nut={mse['nut']:.3e}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    ok = [r for r in rows if not r["skipped"]]
    rel_keys = ("relL2_ux", "relL2_uy", "relL2_p", "relL2_p_surf",
                "relL2_nut_full", "relL2_nut_visc")
    means = {k: float(np.mean([r[k] for r in ok])) for k in rel_keys}
    mse_means = {k: float(np.mean([r["mse_std"][k] for r in ok]))
                 for k in ("ux", "uy", "p", "nut", "p_surf")}
    out = dict(n_requested=len(names), n_evaluated=len(ok),
               n_skipped=len(names) - len(ok),
               means=means, mse_std_means=mse_means, baseline_v1_means=V1_BASE,
               stats={k: dict(mean=v[0], std=v[1]) for k, v in stats.items()},
               stats_note=f"mean/std from {N_STATS} scarce_train sims (sorted[::4] "
                          "stride — spans the full U range; approximation to the "
                          "official full-train normalization)",
               cases=rows,
               note="v2 = circulatory-only decambering (exact panel solution at "
                    "effective angle a_e, true-freestream uniform term; leak=max|V.n| "
                    "at control points) + BL-impressed surface pressure in the "
                    "near-wall band (dp/dn=0, smoothstep to Bernoulli) + nu_t paint "
                    "(van-Driest mixing length / Cebeci-Smith cap in BL, Townsend "
                    "plane-wake closure in wake, dimensionalized *U*chord). "
                    "relL2_nut_full looks large because the prediction is exactly 0 "
                    "outside BL+wake while the truth carries small nonzero nut "
                    "there; relL2_nut_visc (d<0.05c | wake band) is the meaningful "
                    "number. mse_std_* = AirfRANS-protocol MSE on standardized "
                    "fields over the FULL volume (surface nodes included, velocity/"
                    "nut zeroed at the wall).")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\n[v2] evaluated {len(ok)}/{len(names)}")
    print("[v2] relL2 means : " + "  ".join(f"{k}={v:.4f}" for k, v in means.items()))
    print("[v2] mse_std means: " + "  ".join(f"{k}={v:.4e}" for k, v in mse_means.items()))
    print("[v2] v1 baseline : " + "  ".join(f"{k}={v:.4f}" for k, v in V1_BASE.items()))
    print(f"[v2] -> {OUT}")
    return out


if __name__ == "__main__":
    main()
