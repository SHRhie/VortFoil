# -*- coding: utf-8 -*-
"""Standalone 2D inviscid panel method for NACA airfoils.  PURE PHYSICS
(potential flow) — NO data, NO Fluent.

Formulation
-----------
Constant-strength VORTEX panel per panel (one unknown vortex strength gamma_j on
each of the N panels) PLUS a single uniform-strength SOURCE sigma over the whole
body.  Unknowns = (gamma_0, ..., gamma_{N-1}, sigma)  ->  N+1 unknowns.
Equations:
  * N no-penetration (Neumann) conditions  V . n = 0  at the N panel control points
  * 1 Kutta condition  gamma_first + gamma_last = 0   (vortex sheet -> 0 at the
    sharp trailing edge, the standard constant-vortex Kutta condition).

Why this works (and the previous single-uniform-vortex Hess-Smith did not):
  - A *single* uniform vortex of strength gamma can only represent a circulation
    of gamma * perimeter; coupled with the source distribution it produced a far
    too weak (and ill-defined) circulation, so Cl ~ 0 for every alpha.
  - A *per-panel* vortex distribution carries the real bound-circulation profile;
    the Kutta condition pins it; the lift then comes out correct
    (Cl(0)=0, Cl(4)=0.48, Cl(8)=0.97 for NACA0012, slightly above 2*pi*sin(alpha)
    as expected for a finite-thickness airfoil).
  - The pure constant-vortex Neumann system is rank-(N-1) (closed-body identity),
    so replacing one collocation row by the Kutta row leaves a small V.n leak at
    that panel (~1e-4).  Adding ONE extra uniform-source unknown makes the system
    square AND lets every one of the N no-penetration equations be enforced
    exactly -> max|V.n| at the control points is at machine precision (~1e-14).
    The recovered sigma is ~0 (it only removes the rank deficiency).

Lift is reported from the BOUND CIRCULATION,  Cl = 2*Gamma/chord with
Gamma = sum_j gamma_j * L_j, which is exact and free of the near-singular surface
tangential-velocity evaluation.  Cp returned by surface_cp_cl is a small-offset
field evaluation intended for plotting only (it is unreliable right at the sharp
trailing edge); the Cl in that dict is the circulation value.

API (unchanged):
  panel_solve(nodes, alpha_deg) -> dict(gamma, sigma, geom, ...)        # solve
  inviscid_velocity(xy, nodes, alpha_deg, sol=None) -> (N,2)            # (u,v)/U_inf
  surface_cp_cl(nodes, alpha_deg) -> dict(x_mid, Cp, Cl, max_un, ...)   # validation
"""
import numpy as np

_2PI = 2.0 * np.pi


def _panel_geometry(nodes):
    """nodes (Np1,2) closed (last==first or not). Returns per-panel mid, normal,
    tangent, length, node pairs. Panel j: node[j] -> node[j+1] (wrap)."""
    nodes = np.asarray(nodes, float)
    # Drop consecutive near-duplicate nodes.  naca_surface has a sharp TE with
    # several coincident TE points (e.g. node0, node[-2], node[-1] all ~ (1,0)).
    # A zero-length wrap panel would blow up the influence coefficients.
    keep = [nodes[0]]
    for p in nodes[1:]:
        if np.hypot(*(p - keep[-1])) > 1e-9:
            keep.append(p)
    nodes = np.asarray(keep, float)
    # also drop trailing nodes coincident with node[0] (the TE), so the wrap panel
    # node[N-1]->node[0] is the proper (non-degenerate) lower-TE panel.
    while len(nodes) > 3 and np.hypot(*(nodes[-1] - nodes[0])) < 1e-9:
        nodes = nodes[:-1]
    N = len(nodes)
    p1 = nodes
    p2 = np.roll(nodes, -1, axis=0)
    mid = 0.5 * (p1 + p2)
    d = p2 - p1
    L = np.hypot(d[:, 0], d[:, 1])
    t = d / L[:, None]                          # tangent (panel direction)
    # normal = tangent rotated -90deg: (ty, -tx).  For this CCW-ordered NACA loop
    # this points OUTWARD, but the no-penetration condition V.n=0 is invariant to
    # the normal's sign, so orientation is not critical for the solve.
    n = np.stack([t[:, 1], -t[:, 0]], axis=1)
    return dict(p1=p1, p2=p2, mid=mid, t=t, n=n, L=L, N=N)


def _induced_at(points, p1, t, L):
    """Induced velocity (global frame) at `points` (M,2) from each unit-strength
    constant SOURCE panel and unit-strength constant VORTEX panel.
    Returns usx, usy (source) and uvx, uvy (vortex), each (M,N).
    Standard constant-strength panel formulas, evaluated in each panel's local
    frame (xl along tangent, yl along normal) then rotated to the global frame."""
    M = len(points); N = len(p1)
    tx, ty = t[:, 0], t[:, 1]
    dx = points[:, None, 0] - p1[None, :, 0]
    dy = points[:, None, 1] - p1[None, :, 1]
    # local coords: xl along tangent t, yl along (rotated +90deg) (-ty, tx)
    xl = dx * tx[None, :] + dy * ty[None, :]
    yl = -dx * ty[None, :] + dy * tx[None, :]
    Ln = L[None, :]
    eps = 1e-30
    r1sq = xl**2 + yl**2
    r2sq = (xl - Ln)**2 + yl**2
    lnr = 0.5 * np.log((r1sq + eps) / (r2sq + eps))         # ln(r1/r2)
    dtheta = np.arctan2(yl, xl - Ln) - np.arctan2(yl, xl)   # theta2 - theta1
    # constant-strength SOURCE (local frame): u_s = ln(r1/r2)/2pi, w_s = dtheta/2pi
    us_l = lnr / _2PI
    ws_l = dtheta / _2PI
    # constant-strength VORTEX (source rotated 90deg): u_v = dtheta/2pi, w_v = -ln(r1/r2)/2pi
    uv_l = dtheta / _2PI
    wv_l = -lnr / _2PI

    # rotate local (xl-along-t, yl-along (-ty,tx)) back to global:
    #   vx = ul*tx + wl*(-ty);  vy = ul*ty + wl*(tx)
    def to_global(ul, wl):
        vx = ul * tx[None, :] - wl * ty[None, :]
        vy = ul * ty[None, :] + wl * tx[None, :]
        return vx, vy

    usx, usy = to_global(us_l, ws_l)
    uvx, uvy = to_global(uv_l, wv_l)
    return usx, usy, uvx, uvy


def panel_solve(nodes, alpha_deg):
    """Solve for the per-panel vortex strengths gamma_j and the single uniform
    source strength sigma.  Returns a dict consumed by inviscid_velocity."""
    g = _panel_geometry(nodes)
    N = g["N"]; mid = g["mid"]; n = g["n"]; t = g["t"]
    a = np.radians(alpha_deg)
    Uinf = np.array([np.cos(a), np.sin(a)])

    # unit source/vortex panel influence at the control points (panel midpoints)
    usx, usy, uvx, uvy = _induced_at(mid, g["p1"], g["t"], g["L"])

    # normal-velocity influence
    AnV = uvx * n[:, 0][:, None] + uvy * n[:, 1][:, None]          # (N,N) per-panel vortex
    AnS_uni = (usx * n[:, 0][:, None] + usy * n[:, 1][:, None]).sum(axis=1)  # (N,) uniform source col

    # (N+1)x(N+1) system: N no-penetration + 1 Kutta
    A = np.zeros((N + 1, N + 1)); b = np.zeros(N + 1)
    A[:N, :N] = AnV
    A[:N, N] = AnS_uni
    b[:N] = -(Uinf[0] * n[:, 0] + Uinf[1] * n[:, 1])
    # Kutta: vortex sheet strength -> 0 at the sharp TE:  gamma_first + gamma_last = 0
    A[N, 0] = 1.0
    A[N, N - 1] = 1.0
    b[N] = 0.0

    x = np.linalg.solve(A, b)
    gamma = x[:N]      # per-panel vortex strengths
    sigma = x[N]       # single uniform source strength
    return dict(gamma=gamma, sigma=sigma, geom=g, alpha=alpha_deg, Uinf=Uinf)


def inviscid_velocity(xy, nodes, alpha_deg, sol=None):
    """(u,v)/U_inf at arbitrary field points xy (M,2)."""
    if sol is None:
        sol = panel_solve(nodes, alpha_deg)
    g = sol["geom"]; gamma = sol["gamma"]; sigma = sol["sigma"]; Uinf = sol["Uinf"]
    xy = np.asarray(xy, float).reshape(-1, 2)
    usx, usy, uvx, uvy = _induced_at(xy, g["p1"], g["t"], g["L"])
    # per-panel vortices + single uniform source over all panels
    u = Uinf[0] + (uvx * gamma[None, :]).sum(1) + sigma * usx.sum(1)
    v = Uinf[1] + (uvy * gamma[None, :]).sum(1) + sigma * usy.sum(1)
    return np.stack([u, v], axis=1)


def bound_circulation(sol):
    """Total bound circulation Gamma = sum_j gamma_j * L_j (vortex-sheet integral)."""
    g = sol["geom"]
    return float(np.sum(sol["gamma"] * g["L"]))


def surface_cp_cl(nodes, alpha_deg, cp_offset=2e-3):
    """Surface Cp (for plotting) and a RELIABLE Cl.

    Cl is computed from the bound circulation,  Cl = 2*Gamma/chord with
    Gamma = sum_j gamma_j*L_j  (exact; free of the near-singular surface velocity).
    Cp is a small-offset off-surface field evaluation (Vt at mid + cp_offset*n),
    intended for visualization — it is approximate right at the sharp TE.
    """
    sol = panel_solve(nodes, alpha_deg)
    g = sol["geom"]; t = g["t"]; n = g["n"]; mid = g["mid"]
    chord = float(mid[:, 0].max() - mid[:, 0].min())

    # --- reliable lift from circulation ---
    Gamma = bound_circulation(sol)
    Cl = 2.0 * Gamma / chord

    # --- Cp from a slightly off-surface field eval (plotting / diagnostics) ---
    vel = inviscid_velocity(mid + cp_offset * n, nodes, alpha_deg, sol)
    Vt = vel[:, 0] * t[:, 0] + vel[:, 1] * t[:, 1]
    Cp = 1.0 - Vt**2

    # --- no-penetration check at the actual control points (must be ~machine eps) ---
    vel_on = inviscid_velocity(mid, nodes, alpha_deg, sol)
    Vn = vel_on[:, 0] * n[:, 0] + vel_on[:, 1] * n[:, 1]

    return dict(x_mid=mid[:, 0], Cp=Cp, Cl=float(Cl), Cl_circ=float(Cl),
                Gamma=float(Gamma), max_un=float(np.max(np.abs(Vn))),
                chord=chord, gamma=sol["gamma"], sigma=float(sol["sigma"]))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from sdf2d.geometry import naca_surface
    surf = naca_surface("0012", n=240)
    print("=== panel_inviscid smoke (pure potential flow, no data) ===")
    print("  per-panel constant vortex + single uniform source; Kutta: g_first+g_last=0")
    for al in [0.0, 4.0, 8.0]:
        r = surface_cp_cl(surf, al)
        cl_thin = 2 * np.pi * np.sin(np.radians(al))
        print(f"  a={al:+.1f}:  Cl(circ)={r['Cl_circ']:+.4f}  (2pi.sin={cl_thin:+.4f})"
              f"   Gamma={r['Gamma']:+.4f}   max|Vn|(ctrl pts)={r['max_un']:.2e}")
    # field sanity: upper faster than lower for alpha>0, far field -> freestream
    al = 4.0
    a = np.radians(al)
    fp = inviscid_velocity(np.array([[0.5, 0.3], [0.5, -0.3], [1.5, 0.2], [-5.0, 0.0]]), surf, al)
    sp = np.hypot(fp[:, 0], fp[:, 1])
    print(f"  a=+4.0 field:")
    print(f"    (0.5,+0.3) speed={sp[0]:.4f}  (0.5,-0.3) speed={sp[1]:.4f}  "
          f"-> upper faster: {sp[0] > sp[1]}")
    print(f"    (1.5, 0.2) u,v={fp[2]}   (1.5,0.2) finite: {np.all(np.isfinite(fp[2]))}")
    print(f"    far (-5,0) u,v={fp[3]}  (expect ~[cos4,sin4]=[{np.cos(a):.4f},{np.sin(a):.4f}])")
    print("  Cl(0)~0 + Cl~2pi.sina + max|Vn|~0 + upper faster + far->freestream => panel OK")
