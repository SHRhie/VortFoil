# -*- coding: utf-8 -*-
"""The 3D view of a wing result (wing_nll.compute): the data of a standalone WebGL page and the page itself
(code/wing_view.html, opened in the browser). Ported from the research code (airfoil_operator VortFoil/code
wing_streamlines.py, wing_freewake.py, wing_viscous.py and wing_viewer/build.py as of a31607b) for the release wing (its
stations, lofted sections and both section states); serial (no process pools), numpy only, no prints.

At the result's incidence only (the application is where alpha changes) the page shows the wing's surfaces coloured by each
strip's C_p, the trailing vortices (straight legs; or, free_wake=True, the relaxed free wake, which rolls up), the polar and
the spanwise loading, the coefficients and every warning, and a flow field:
  field='none'      no flow field;
  field='inviscid'  streamlines of the lifting line's composite field: outer = the horseshoe vortices (or the relaxed wake)
                    with a core, inner = each strip's 2D Hess-Smith panel flow matched to the strip's circulation, blended
                    in span (wing_streamlines.py);
  field='viscous'   the same with each strip's inner flow from the RANS field decoder (k-omega SST, AirfRANS-trained) at the
                    strip's effective incidence and Reynolds number: streamlines, three chordwise and two wake-survey planes
                    and the wake sheet (wing_viscous.py). Colour on lines and planes: q = 1 - |V|^2/V_inf^2 (C_p where the
                    flow is inviscid, positive in the boundary layers and the wake).
Frames as wing_nll: the wind frame (X along the freestream, Y spanwise, Z up; the wing pitched by alpha), metres.
"""
import os, re, json, gzip, base64, warnings
import numpy as np
import wing_nll as W

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "wing_view.html")
FIELDS = ("none", "inviscid", "viscous")

# ------------------------------------------------------------------------------------------- composite field (wing_streamlines)
RC, H, X0, X_END, STEPS = 0.02, 0.0025, -0.35, 1.5, 1400      # core radius, step, upstream and downstream ends, step limit;
                                                               # lengths per metre of half-span (the 2 m demo wing as given)
WIN = (1.0, 2.0)                                               # inner correction: full within 1 chord of mid-chord, none beyond 2
EX = np.array([1.0, 0.0, 0.0])
TRACE_TICK = 50                                                # trace steps between progress reports (and cancel checks)


def naca4(code, n=161):
    """closed-trailing-edge NACA 4-digit loop, Selig order (upper TE -> LE -> lower TE), cosine spacing (research wing_nll)"""
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    if m > 0:
        yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x ** 2))
        dy = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    else:
        yc = dy = np.zeros_like(x)
    th = np.arctan(dy)
    up = np.c_[x - yt * np.sin(th), yc + yt * np.cos(th)]; lo = np.c_[x + yt * np.sin(th), yc - yt * np.cos(th)]
    return np.r_[up[::-1], lo[1:]]


def panel_vel(P, X0, X1):
    """velocity at points P (m,2) of unit-strength source and unit counterclockwise-vortex panels X0 -> X1: four (m,n) arrays"""
    t = X1 - X0; L = np.hypot(t[:, 0], t[:, 1]); c, s = t[:, 0] / L, t[:, 1] / L
    dx = P[:, None, 0] - X0[None, :, 0]; dy = P[:, None, 1] - X0[None, :, 1]
    xl = dx * c + dy * s; yl = -dx * s + dy * c
    lnr = 0.5 * np.log((xl ** 2 + yl ** 2) / ((xl - L) ** 2 + yl ** 2))
    beta = np.arctan2(yl, xl - L) - np.arctan2(yl, xl)
    k = 1 / (2 * np.pi)
    return k * (lnr * c - beta * s), k * (lnr * s + beta * c), k * (-beta * c - lnr * s), k * (-beta * s + lnr * c)


class Section:
    """Hess-Smith solution of a section (chord 1, leading edge at the origin), for unit freestream along x and y: a NACA 4-digit
    designation, or a closed loop xy (counterclockwise: upper trailing edge -> leading edge -> lower trailing edge)"""

    def __init__(self, code="2412", n=161, xy=None):
        if xy is None:
            self.m, self.p, self.t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
            xy = naca4(code, n)                                                 # counterclockwise: upper TE -> LE -> lower TE
        else:
            xy = np.asarray(xy, np.float64); le = int(np.argmin(xy[:, 0])); self.m = None
            self.up, self.lo = xy[:le + 1][::-1], xy[le:]                       # both from the leading edge
        self.X0, self.X1 = xy[:-1], xy[1:]
        d = self.X1 - self.X0; L = np.hypot(d[:, 0], d[:, 1]); tv = d / L[:, None]
        nv = np.c_[tv[:, 1], -tv[:, 0]]                                         # outward normal of a counterclockwise loop
        self.cpts = 0.5 * (self.X0 + self.X1) + 1e-9 * nv                       # control points just outside each panel
        us, vs, uv, vv = panel_vel(self.cpts, self.X0, self.X1)
        N = len(L); A = np.zeros((N + 1, N + 1))
        A[:N, :N] = us * nv[:, :1] + vs * nv[:, 1:]
        A[:N, N] = np.sum(uv * nv[:, :1] + vv * nv[:, 1:], 1)
        tf, tl = tv[0], tv[-1]                                                  # Kutta: V.t on the two trailing-edge panels cancel
        A[N, :N] = us[0] * tf[0] + vs[0] * tf[1] + us[-1] * tl[0] + vs[-1] * tl[1]
        A[N, N] = np.sum(uv[0] * tf[0] + vv[0] * tf[1] + uv[-1] * tl[0] + vv[-1] * tl[1])
        self.Ainv = np.linalg.inv(A)
        with np.errstate(all="ignore"):                                        # Accelerate BLAS raises spurious FP flags
            self.sol = np.array([self.Ainv @ np.r_[-(nv @ V), -(V @ tf + V @ tl)] for V in (np.array([1.0, 0]), np.array([0, 1.0]))])
        self.cl_xy = -2 * self.sol[:, N] * L.sum()                              # C_l = -2 Gamma_ccw for each unit freestream
        self.nv, self.tf, self.tl, self.Lsum = nv, tf, tl, L.sum()

    def angle_for(self, cl):
        """inviscid incidence giving lift coefficient cl"""
        R, ph = np.hypot(*self.cl_xy), np.arctan2(self.cl_xy[1], self.cl_xy[0])
        return ph - np.arccos(np.clip(cl / R, -1, 1))

    def vel(self, Q, s, a=None):
        """velocity at section points Q (m,2): panel strengths s (m, N+1), plus unit freestream at incidences a (m,) if given"""
        us, vs, uv, vv = panel_vel(Q, self.X0, self.X1)
        fx, fy = (np.cos(a), np.sin(a)) if a is not None else (0.0, 0.0)
        return (fx + np.sum(s[:, :-1] * us, 1) + s[:, -1] * np.sum(uv, 1),
                fy + np.sum(s[:, :-1] * vs, 1) + s[:, -1] * np.sum(vv, 1))

    def camber_thickness(self, x):
        if self.m is None:
            yu, yl = np.interp(x, self.up[:, 0], self.up[:, 1]), np.interp(x, self.lo[:, 0], self.lo[:, 1])
            return 0.5 * (yu + yl), 0.5 * (yu - yl)
        m, p, t = self.m, self.p, self.t
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
        yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x ** 2)) if m > 0 else 0 * x
        return yc, yt

    def inside(self, xi, ze):
        yc, yt = self.camber_thickness(np.clip(xi, 0, 1))
        return (xi > 0) & (xi < 1) & (np.abs(ze - yc) < yt)


def seg_v(P, A, B, rc):
    """unit filaments A -> B (k,3) at points P (m,3), core radius rc: (m,k,3). The research expression per component (no
    size-3 axis in the temporaries): the same operations in the same order, bitwise equal and twice as fast"""
    r1 = [P[:, None, i] - A[None, :, i] for i in range(3)]; r2 = [P[:, None, i] - B[None, :, i] for i in range(3)]; r0 = B - A
    c = [r1[1] * r2[2] - r1[2] * r2[1], r1[2] * r2[0] - r1[0] * r2[2], r1[0] * r2[1] - r1[1] * r2[0]]
    n1 = np.maximum(np.sqrt(r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2), 1e-12); n2 = np.maximum(np.sqrt(r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2), 1e-12)
    f = ((r0[:, 0] * (r1[0] / n1 - r2[0] / n2) + r0[:, 1] * (r1[1] / n1 - r2[1] / n2) + r0[:, 2] * (r1[2] / n1 - r2[2] / n2))
         / (4 * np.pi * (c[0] ** 2 + c[1] ** 2 + c[2] ** 2 + rc ** 2 * np.sum(r0 * r0, -1))))
    return np.stack([f * c[0], f * c[1], f * c[2]], -1)


def leg_v(P, A, rc):
    """unit semi-infinite filaments from A (k,3) to +x infinity, at points P (m,3), core radius rc: (m,k,3)"""
    r = P[:, None] - A[None]; n = np.maximum(np.linalg.norm(r, axis=-1), 1e-12); c = np.cross(EX, r)
    return ((1 + r[..., 0] / n) / (4 * np.pi * (np.sum(c * c, -1) + rc ** 2)))[..., None] * c


def strip_sections(res):
    """the strips' inviscid sections and each strip's index into them: one NACA Section when every station carries that 4-digit
    designation (as the research), otherwise one per distinct strip loop of the result (res['loops'], the shapes its state
    used) through its CST16 fit on 161 points per surface, turned back by the fit's chord rotation (research strip_sections)"""
    secs = {s["section"] for s in res["defn"]["stations"]}
    if len(secs) == 1 and re.fullmatch(r"\d{4}", next(iter(secs))):
        return [Section(next(iter(secs)))], np.zeros(res["wing"]["n"], int)
    from vortfoil import geometry
    turn = lambda p, r: p @ np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
    out = []
    for loop in res["loops"]["coords"]:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):                   # Accelerate BLAS raises spurious FP flags
            cst, info = geometry.fit_cst16(loop)
            x, yu, yl = geometry.surface(cst, 161)
        out.append(Section(xy=turn(np.r_[np.c_[x, yu][::-1], np.c_[x, yl][1:]], np.radians(info["rotation_deg"]))))
    return out, np.asarray(res["loops"]["index"])


class Flow:
    """composite velocity field (per unit freestream, wind frame: x downstream, z up; the wing pitched by alpha about the root
    quarter chord) of the wing w at freestream V for case: alpha, y, chord, gamma, cl, alpha_eff of the strips [, wake]; secs,
    sid: the strips' inviscid sections (strip_sections)"""

    def __init__(self, w, V, case, secs, sid):
        a, b2 = np.radians(case["alpha"]), 0.5 * w.b
        chord = w.chord
        geo = lambda y: a + np.radians(w.at(y, "twist"))                                    # section pitch in the wind frame
        qc = lambda y: wind(np.c_[w.at(y, "x"), y, w.at(y, "z")], case["alpha"])           # quarter-chord point, wind frame
        self.y, self.c, self.ag, self.b2 = np.array(case["y"]), np.array(case["chord"]), geo(np.array(case["y"])), b2
        self.chord, self.geo, self.qc, self.L, self.rc = chord, geo, qc, b2, RC * b2
        self.Qs = qc(self.y)                                                                  # the strips' quarter chords
        self.secs, self.sid = secs, np.asarray(sid)
        self.G = np.array(case["gamma"]) / V                                                  # Gamma / V_inf (m)
        Q, T, _ = nodes(w, case["alpha"])
        if "wake" in case:                                                                    # the relaxed wake (solve_free)
            Wk = np.array(case["wake"]); g = np.r_[0.0, self.G, 0.0]; gam = g[:-1] - g[1:]   # net node strengths, away from the wing
            self.sA = np.r_[Q[:-1], Q, Wk[:, :-1].reshape(-1, 3)]; self.sB = np.r_[Q[1:], T, Wk[:, 1:].reshape(-1, 3)]
            self.sG = np.r_[self.G, gam, np.repeat(gam, Wk.shape[1] - 1)]
            self.lA, self.lG = Wk[:, -1], gam
        else:
            self.sA, self.sB, self.sG = np.r_[T[:-1], Q[:-1], Q[1:]], np.r_[Q[:-1], Q[1:], T[1:]], np.r_[self.G, self.G, self.G]
            self.lA, self.lG = np.r_[T[:-1], T[1:]], np.r_[-self.G, self.G]                   # horseshoe i: inf -> T_i -> Q_i -> Q_i+1 -> T_i+1 -> inf
        self.Vc = self.outer(self.Qs)                                                         # outer field on the lifting line
        n, ca, sa = len(self.y), np.cos(self.ag)[:, None], np.sin(self.ag)[:, None]
        CP = np.stack([secs[k].cpts for k in self.sid])                                       # every strip's panel control points
        xi, ze = CP[..., 0] - 0.25, CP[..., 1]
        X, Z = self.c[:, None] * (xi * ca + ze * sa), self.c[:, None] * (-xi * sa + ze * ca)   # wind frame, from the strip's quarter chord
        E = self.outer(np.stack([X + self.Qs[:, :1], np.repeat(self.y[:, None], X.shape[1], 1), Z + self.Qs[:, 2:]], -1).reshape(-1, 3)).reshape(n, -1, 3)
        E -= self.Vc[:, None] + (self.G[:, None] / (2 * np.pi * (X ** 2 + Z ** 2 + self.rc ** 2)))[..., None] * np.stack([Z, 0 * Z, -X], -1)
        Ex, Ez = E[..., 0] * ca - E[..., 2] * sa, E[..., 0] * sa + E[..., 2] * ca              # outer minus common, section frame
        corr = np.empty((n, CP.shape[1] + 1)); self.astar = np.empty(n); self.coef = np.empty_like(corr)
        cl = np.array(case["cl"])
        for j in range(n):
            sec = secs[self.sid[j]]
            Vn = Ex[j] * sec.nv[:, 0] + Ez[j] * sec.nv[:, 1]
            Vk = Ex[j, 0] * sec.tf[0] + Ez[j, 0] * sec.tf[1] + Ex[j, -1] * sec.tl[0] + Ez[j, -1] * sec.tl[1]
            with np.errstate(all="ignore"):
                corr[j] = sec.Ainv @ np.r_[-Vn, -Vk]                                           # transpiration correction, with Kutta
            self.astar[j] = sec.angle_for(cl[j] + 2 * corr[j, -1] * sec.Lsum)                 # total circulation = Gamma
            self.coef[j] = np.cos(self.astar[j]) * sec.sol[0] + np.sin(self.astar[j]) * sec.sol[1] + corr[j]
        self.corr, self.dcl = corr, -2 * corr[:, -1] * np.array([secs[k].Lsum for k in self.sid])

    def _by_strip(self, j, f):
        """f(section, rows) over the points' strips j, a call per distinct section: (u, v) arrays"""
        if len(self.secs) == 1:
            return f(self.secs[0], slice(None))
        s = self.sid[j]; u, v = np.empty(len(j)), np.empty(len(j))
        for k in np.unique(s):
            m = s == k; u[m], v[m] = f(self.secs[k], m)
        return u, v

    def inner_vel(self, Q, j):
        """strip j's inner 2D velocity at section points Q (chord frame, per unit freestream)"""
        return self._by_strip(j, lambda sec, m: sec.vel(Q[m], self.coef[j[m]], self.astar[j[m]]))

    def window(self, xi, ze):
        """weight of the inner correction: full within WIN[0] chords of mid-chord, none beyond WIN[1]"""
        w = np.clip((WIN[1] - np.hypot(xi - 0.5, ze)) / (WIN[1] - WIN[0]), 0, 1)
        return w * w * (3 - 2 * w)

    def outer(self, P, chunk=512):
        v = np.empty((len(P), 3))
        for i in range(0, len(P), chunk):                                                     # bounded memory with a relaxed wake
            p = P[i:i + chunk]
            v[i:i + chunk] = np.einsum("mkd,k->md", seg_v(p, self.sA, self.sB, self.rc), self.sG) + np.einsum("mkd,k->md", leg_v(p, self.lA, self.rc), self.lG)
        v[:, 0] += 1.0
        return v

    def _frame(self, P, j, q=None):
        """section coordinates of points P in strip j's frame, placed at the quarter chord of the section through each point's
        span (the wing's local position between strip centres: sweep and dihedral move it)"""
        ca, sa, c = np.cos(self.ag[j]), np.sin(self.ag[j]), self.c[j]
        q = self.qc(P[:, 1]) if q is None else q
        X, Z = P[:, 0] - q[:, 0], P[:, 2] - q[:, 2]
        return 0.25 + (X * ca - Z * sa) / c, (X * sa + Z * ca) / c, ca, sa

    def _strips(self, y):
        n = len(self.y); i = np.clip(np.searchsorted(self.y, y) - 1, 0, n - 2)
        wR = np.clip((y - self.y[i]) / (self.y[i + 1] - self.y[i]), 0, 1)
        fade = np.clip((self.b2 - np.abs(y)) / (self.b2 - np.abs(self.y[0])), 0, 1)          # inner part vanishes at the tips
        return ((i, (1 - wR) * fade), (i + 1, wR * fade))

    def vel(self, P):
        V = self.outer(P); q = self.qc(P[:, 1])
        for j, wt in self._strips(P[:, 1]):
            xi, ze, ca, sa = self._frame(P, j, q)
            u2, v2 = self.inner_vel(np.c_[xi, ze], j)
            X, Z = P[:, 0] - q[:, 0], P[:, 2] - q[:, 2]
            Vin = np.c_[u2 * ca + v2 * sa, 0 * u2, -u2 * sa + v2 * ca]
            Vcom = self.Vc[j] + (self.G[j] / (2 * np.pi * (X ** 2 + Z ** 2 + self.rc ** 2)))[:, None] * np.c_[Z, 0 * Z, -X]
            V += (wt * self.window(xi, ze))[:, None] * (Vin - Vcom)
        return V

    def inside(self, P):
        (i, wl), (k, wr) = self._strips(P[:, 1])
        j = np.where(wl >= wr, i, k)
        xi, ze, _, _ = self._frame(P, j)
        s = self.sid[j]; ins = np.zeros(len(P), bool)
        for q in np.unique(s):
            m = s == q; ins[m] = self.secs[q].inside(xi[m], ze[m])
        return (np.abs(P[:, 1]) < self.b2) & ins


def section_point(fl, xi, ze, y):
    """section point (chords, leading edge at 0) of the section at span y -> wind frame"""
    c, ag, q = fl.chord(y), fl.geo(y), fl.qc(np.asarray(y, np.float64))
    return np.c_[c * ((xi - 0.25) * np.cos(ag) + ze * np.sin(ag)) + q[:, 0], y, c * (-(xi - 0.25) * np.sin(ag) + ze * np.cos(ag)) + q[:, 2]]


def seeds(fl):
    """near the wing: at 21 spanwise stations, 15% chord, 1.2% and 4.5% chord above the upper and below the lower surface;
    and a ring of 8 points around each tip just behind its trailing edge. Every seed is traced both ways."""
    ys = 0.97 * fl.b2 * np.sin(np.linspace(-np.pi / 2, np.pi / 2, 21))
    ct = np.array([fl.secs[fl.sid[int(np.argmin(np.abs(fl.y - y)))]].camber_thickness(0.15) for y in ys]); yc, yt = ct[:, 0], ct[:, 1]
    P = [section_point(fl, np.full_like(ys, 0.15), np.full_like(ys, ze), ys) for ze in (yc + yt + 0.012, yc + yt + 0.045, yc - yt - 0.012, yc - yt - 0.045)]
    for sg in (-1, 1):
        te = section_point(fl, np.array([1.0]), np.array([0.0]), np.array([sg * fl.b2]))[0]
        ph = np.linspace(0, 2 * np.pi, 9)[:-1]
        P.append(np.c_[np.full(8, te[0] + 0.03 * fl.L), sg * (fl.b2 - 0.01 * fl.L) + 0.035 * fl.L * np.cos(ph), te[2] + 0.035 * fl.L * np.sin(ph)])
    return np.concatenate(P)


def trace(flow, P, sgn=1, tick=None):
    """RK4 on the unit direction field (sgn -1: upstream), fixed arc-length step; stop outside [X0, X_END] or inside the wing"""
    P = P.copy(); alive = np.ones(len(P), bool); paths = [[p.copy()] for p in P]; hit = np.zeros(len(P), bool)
    unit = lambda q: (lambda v: sgn * v / np.linalg.norm(v, axis=1, keepdims=True))(flow.vel(q))
    for step in range(STEPS):
        idx = np.where(alive)[0]
        if not len(idx):
            break
        if tick is not None and step % TRACE_TICK == 0:
            tick(f"streamlines {'downstream' if sgn > 0 else 'upstream'}: step {step} of at most {STEPS}")
        p = P[idx]
        h, L = H * flow.L, flow.L
        k1 = unit(p); k2 = unit(p + 0.5 * h * k1); k3 = unit(p + 0.5 * h * k2); k4 = unit(p + h * k3)
        pn = p + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ins = flow.inside(pn); stop = (pn[:, 0] > X_END * L) | (pn[:, 0] < X0 * L) | ins | (np.abs(pn[:, 1]) > 2 * L)
        for m, q in zip(idx, pn):
            paths[m].append(q.copy())
        P[idx] = pn; alive[idx[stop]] = False; hit[idx[ins]] = True
    return [np.array(p) for p in paths], hit


def thin(path, L=1.0, turn=np.radians(3), lmax=0.03):
    """keep the points where the line has turned 3 degrees or run 3 cm (per metre of half-span) since the last kept point"""
    keep, last, dlast, acc = [0], path[0], None, 0.0
    for k in range(1, len(path) - 1):
        d = path[k + 1] - path[k]; d = d / (np.linalg.norm(d) + 1e-15); acc += np.linalg.norm(path[k] - path[k - 1])
        if dlast is None:
            dlast = d
        if acc >= lmax * L or np.arccos(np.clip(d @ dlast, -1, 1)) >= turn:
            keep.append(k); dlast, acc = d, 0.0
    keep.append(len(path) - 1)
    return path[keep]


def streamlines(fl, tick=None):
    """the seeds traced both ways, thinned, coloured by q = 1 - |V|^2/V_inf^2: [{x, y, z, cp}] (research run_case)"""
    S = seeds(fl)
    (fw, _), (bw, _) = trace(fl, S, 1, tick), trace(fl, S, -1, tick)
    paths = [thin(np.r_[b[::-1], f[1:]], fl.L) for b, f in zip(bw, fw)]
    if tick is not None:
        tick("streamline colours")
    allp = np.concatenate(paths); q = _q(fl, allp)
    lines, k = [], 0
    for p in paths:
        lines.append(dict(x=np.round(p[:, 0], 4).tolist(), y=np.round(p[:, 1], 4).tolist(), z=np.round(p[:, 2], 4).tolist(),
                          cp=np.round(q[k:k + len(p)], 3).tolist()))
        k += len(p)
    return lines


# ------------------------------------------------------------------------------------------- free wake (wing_freewake)
# The trailing vortex sheet follows the local flow and rolls up. At node k the net trailing filament (strength
# gamma_k = Gamma_{k-1} - Gamma_k) runs along the chord from the quarter chord to the trailing edge, then along a polyline
# of K wake segments and a semi-infinite segment along +x. Relaxation: the wake nodes are marched from the trailing edge
# along the local velocity (core radius RC per metre of half-span), under-relaxed by OMEGA, until the RMS move is below TOL
# of the half-span; then the circulation is re-solved with the relaxed wake's influence (wing_nll.solve with it), OUTER
# times. Induced drag stays the Trefftz-plane value of the shed circulation, which roll-up does not change.
K, DTAU0, GROWTH = 50, 0.01, 1.07          # wake segments; first step and growth per metre of half-span (about 2 spans long)
OMEGA, TOL, ITERS, OUTER = 0.3, 2e-4, 40, 1                # TOL: RMS node move per metre of half-span


def wind(P, alpha_deg):
    """body-frame points (wing at zero incidence) -> wind frame (the body pitched up by alpha)"""
    a = np.radians(alpha_deg)
    return np.c_[P[:, 0] * np.cos(a) + P[:, 2] * np.sin(a), P[:, 1], -P[:, 0] * np.sin(a) + P[:, 2] * np.cos(a)]


def nodes(w, alpha_deg):
    """quarter-chord nodes Q, trailing-edge points T (wind frame) and node y"""
    Q = np.r_[w.A, w.B[-1:]]
    T = Q + 0.75 * w.cn[:, None] * np.c_[np.cos(w.twn), 0 * w.twn, -np.sin(w.twn)]
    return wind(Q, alpha_deg), wind(T, alpha_deg), Q[:, 1]


def wake_steps(b2):
    return DTAU0 * b2 * GROWTH ** np.arange(K)


def straight_wake(w, alpha_deg):
    _, T, _ = nodes(w, alpha_deg)
    s = np.r_[0.0, np.cumsum(wake_steps(0.5 * w.b))]
    return T[:, None, :] + s[None, :, None] * np.array([1.0, 0, 0])[None, None, :]     # (nodes, K+1, 3)


def elements(w, alpha_deg, G, wake):
    """segments (A, B, strength) and semi-infinite tails (start, strength) of the whole vortex system; G = Gamma / V_inf"""
    Q, T, _ = nodes(w, alpha_deg)
    g = np.r_[0.0, G, 0.0]; gam = g[:-1] - g[1:]                                         # net node strengths, away from the wing
    A = [Q[:-1], Q, wake[:, :-1].reshape(-1, 3)]; B = [Q[1:], T, wake[:, 1:].reshape(-1, 3)]
    S = [G, gam, np.repeat(gam, wake.shape[1] - 1)]
    return np.concatenate(A), np.concatenate(B), np.concatenate(S), wake[:, -1], gam


def velocity(P, el, rc, chunk=256):
    A, B, S, TA, TG = el
    out = np.empty((len(P), 3))
    for i in range(0, len(P), chunk):
        p = P[i:i + chunk]
        out[i:i + chunk] = np.einsum("mkd,k->md", seg_v(p, A, B, rc), S) + np.einsum("mkd,k->md", leg_v(p, TA, rc), TG)
    out[:, 0] += 1.0
    return out


def relax(w, alpha_deg, G, wake=None, tick=None):
    """relaxed wake polylines (nodes, K+1, 3) for the circulation G (per unit freestream) and the iteration count"""
    b2 = 0.5 * w.b; rc = RC * b2; dt = wake_steps(b2)
    X = straight_wake(w, alpha_deg) if wake is None else wake.copy()
    for it in range(ITERS):
        if tick is not None:
            tick(f"relaxing the free wake: iteration {it + 1}")
        V = velocity(X.reshape(-1, 3), elements(w, alpha_deg, G, X), rc).reshape(X.shape)
        step = 0.5 * (V[:, :-1] + V[:, 1:]) * dt[None, :, None]
        Xm = np.concatenate([X[:, :1], X[:, :1] + np.cumsum(step, axis=1)], axis=1)
        move = np.sqrt(np.mean(np.sum((Xm - X) ** 2, axis=2)))
        X = X + OMEGA * (Xm - X)
        if move < TOL * b2:
            break
    return X, it + 1


def influence_body(w, alpha_deg, wake, rc_frac=1e-9):
    """induced velocity at the control points of each unit-strength horseshoe with this wake, body frame (n, n, 3)"""
    Q, T, _ = nodes(w, alpha_deg)
    Pc = wind(w.P, alpha_deg); rc = rc_frac * 0.5 * w.b
    vb = seg_v(Pc, Q[:-1], Q[1:], rc)                                                    # (n, strips, 3)
    vb[np.arange(len(Pc)), np.arange(len(Pc))] = 0.0      # each control point lies on its own bound segment, which induces nothing
                                                          # there; with sweep or dihedral, rounding (1e-16) over the 1e-9 core
                                                          # would turn that zero into an O(1) velocity
    nn = len(Q); vp = np.zeros((len(Pc), nn, 3))
    for k in range(nn):                                                                   # node paths, unit strength away from the wing
        A = np.r_[Q[k:k + 1], wake[k, :-1]]; B = np.r_[T[k:k + 1], wake[k, 1:]]
        vp[:, k] = seg_v(Pc, A, B, rc).sum(1) + leg_v(Pc, wake[k, -1:], rc)[:, 0]
    v = vb + vp[:, 1:] - vp[:, :-1]                                                       # strip j: bound + path j+1 - path j
    a = np.radians(alpha_deg)
    return np.stack([v[..., 0] * np.cos(a) - v[..., 2] * np.sin(a), v[..., 1], v[..., 0] * np.sin(a) + v[..., 2] * np.cos(a)], -1)


def solve_free(w, V, alpha_deg, polar, tick=None):
    """wing_nll.solve with a relaxed wake: alternate relaxation (fixed circulation) and the circulation solve (fixed wake);
    the solution with its wake polylines under 'wake'"""
    s = W.solve(w, V, alpha_deg, polar)
    wake = None
    for _ in range(OUTER):
        wake, _ = relax(w, alpha_deg, np.array(s["gamma"]) / V, wake, tick)
        s = W.solve(w, V, alpha_deg, polar, v=influence_body(w, alpha_deg, wake))
    s["wake"], _ = relax(w, alpha_deg, np.array(s["gamma"]) / V, wake, tick)
    return s


# ------------------------------------------------------------------------------------------- viscous field (wing_viscous)
# Each strip's inner flow is the decoder's (u, v)/U for the strip's section at its effective incidence and Reynolds number on
# a stretched grid in the chord frame inside the decoder's box [-2, 4] x [-1.5, 1.5] chords, plus the transpiration correction
# of the potential version; the window absorbs the decoder's own circulation's mismatch with the lifting line's Gamma. With a
# relaxed wake the decoder's 2D wake is moved down onto the 3D sheet behind each trailing edge (zero at the trailing edge, full
# half a chord behind it, faded where the strip's edge filaments drift spanwise by a fifth of its chord: near the tips).
XI = np.r_[np.arange(-1.5, -0.2, 0.05), np.arange(-0.2, 1.3, 0.005), np.arange(1.3, 4.0001, 0.02)]
ZE = np.r_[np.arange(-1.5, -0.3, 0.05), np.arange(-0.3, 0.3, 0.005), np.arange(0.3, 1.5001, 0.05)]


def sample(loop, family, alpha, Re, decoder):
    """decoder field (u, v)/U in the chord frame on XI x ZE (float32; inside the section 0) and the decoder's messages"""
    import predict_field as PF
    GX, GZ = np.meshgrid(XI, ZE, indexing="ij")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                       # its messages are in meta['warnings'], grouped by the caller
        out, meta = PF.predict(loop, float(alpha), float(Re), points=np.c_[GX.ravel(), GZ.ravel()], frame="body", family=family,
                               decoder=decoder)
    u, v = (np.nan_to_num(out[k].reshape(GX.shape)).astype(np.float32) for k in ("u_over_U", "v_over_U"))
    return u, v, meta["warnings"]


def bilinear(F, j, xi, ze):
    """F (strips, len(XI), len(ZE)) for strip j (per point) at chord-frame points; clamped to the grid"""
    i = np.clip(np.searchsorted(XI, xi) - 1, 0, len(XI) - 2); k = np.clip(np.searchsorted(ZE, ze) - 1, 0, len(ZE) - 2)
    t = np.clip((xi - XI[i]) / (XI[i + 1] - XI[i]), 0, 1); s = np.clip((ze - ZE[k]) / (ZE[k + 1] - ZE[k]), 0, 1)
    return (1 - t) * (1 - s) * F[j, i, k] + t * (1 - s) * F[j, i + 1, k] + (1 - t) * s * F[j, i, k + 1] + t * s * F[j, i + 1, k + 1]


def wake_centre(U, V, j):
    """the strip's 2D wake centre: the zeta of the lowest decoder speed at each chord station XI >= 1 (-0.6 < zeta < 0.3)"""
    aft, zm = XI >= 1.0, (ZE > -0.6) & (ZE < 0.3)
    return ZE[zm][np.argmin(np.hypot(U[j], V[j])[aft][:, zm], axis=1)]


class ViscousFlow(Flow):
    def __init__(self, w, Vinf, case, secs, sid, U, V):
        super().__init__(w, Vinf, case, secs, sid)
        self.U, self.V = U, V
        self.dz = self._wake_shift(case) if "wake" in case else None

    def _wake_shift(self, case):
        """per strip, the height (wind frame) of the relaxed 3D wake sheet above the strip's 2D wake centre, at XI >= 1"""
        Wk = np.array(case["wake"]); aft = XI >= 1.0; D = np.zeros((len(self.y), len(XI)))
        for j in range(len(self.y)):
            zc = wake_centre(self.U, self.V, j); ca, sa, c = np.cos(self.ag[j]), np.sin(self.ag[j]), self.c[j]
            Xc, Zc = c * ((XI[aft] - 0.25) * ca + zc * sa) + self.Qs[j, 0], c * (-(XI[aft] - 0.25) * sa + zc * ca) + self.Qs[j, 2]
            f = 0.5 * (Wk[j] + Wk[j + 1]); o = np.argsort(f[:, 0])                  # the strip's two edge filaments
            dy = np.abs(np.interp(Xc, f[o, 0], f[o, 1]) - f[0, 1])                   # spanwise drift since the trailing edge
            D[j, aft] = np.minimum(np.interp(Xc, f[o, 0], f[o, 2]) - Zc, 0.0) * np.clip(1 - dy / (0.2 * c), 0, 1)
        return D

    def shift(self, Q, j):
        """section-frame offset that moves the 2D wake onto the 3D sheet (zero ahead of the trailing edge)"""
        if self.dz is None:
            return np.zeros_like(Q)
        i = np.clip(np.searchsorted(XI, Q[:, 0]) - 1, 0, len(XI) - 2); t = np.clip((Q[:, 0] - XI[i]) / (XI[i + 1] - XI[i]), 0, 1)
        r = np.clip((Q[:, 0] - 1.0) / 0.5, 0, 1)
        dz = ((1 - t) * self.dz[j, i] + t * self.dz[j, i + 1]) * r * r * (3 - 2 * r)
        return np.c_[-dz * np.sin(self.ag[j]), dz * np.cos(self.ag[j])] / self.c[j][:, None]

    def inner_vel(self, Q, j):
        cu, cv = self._by_strip(j, lambda sec, m: sec.vel(Q[m], self.corr[j[m]]))  # transpiration correction, no freestream
        Qs = Q - self.shift(Q, j)                                                     # the RANS part follows the 3D wake
        return bilinear(self.U, j, Qs[:, 0], Qs[:, 1]) + cu, bilinear(self.V, j, Qs[:, 0], Qs[:, 1]) + cv

    def window(self, xi, ze):
        sm = lambda w: (lambda c: c * c * (3 - 2 * c))(np.clip(w, 0, 1))
        return sm((xi + 1.5) / 0.5) * sm((4.0 - xi) / 0.8) * sm((1.5 - np.abs(ze)) / 0.5)


def enc(q):
    """q in [-2, 1] as one byte per point (0..254), 255 for no data; base64"""
    b = np.where(np.isfinite(q), np.round((np.clip(np.nan_to_num(q), -2, 1) + 2) / 3 * 254), 255).astype(np.uint8)
    return base64.b64encode(b.tobytes()).decode()


def _q(fl, P, chunk=4096):
    """q = 1 - |V|^2 at points P, in chunks (bounded memory: the panel velocities are points x panels)"""
    return np.concatenate([1 - np.sum(fl.vel(P[i:i + chunk]) ** 2, 1) for i in range(0, len(P), chunk)])


def planes(fl, tick=None):
    """chordwise slices (x-z) at 2y/b 0, 0.5, 0.85 about the section's quarter chord, and wake-survey planes (y-z) 0.25 and 1
    root chord behind the root trailing edge; q as enc() bytes (255 inside the section)"""
    b2, cr = fl.b2, float(fl.chord(0.0))
    out = []
    for eta in (0.0, 0.5, 0.85):
        if tick is not None:
            tick(f"flow planes: {len(out) + 1} of 5")
        y = eta * b2; c = float(fl.chord(y)); q = fl.qc(np.array([y]))[0]
        xs = q[0] + np.linspace(-0.6 * c, 2.2 * c, 141); zs = q[2] + np.linspace(-0.55 * c, 0.45 * c, 51)
        X, Z = np.meshgrid(xs, zs, indexing="ij"); P = np.c_[X.ravel(), np.full(X.size, y), Z.ravel()]
        q = _q(fl, P); q[fl.inside(P)] = np.nan
        out.append(dict(kind="xz", at=y, u0=xs[0], u1=xs[-1], v0=zs[0], v1=zs[-1], nu=len(xs), nv=len(zs), q=enc(q)))
    for dx in (0.25, 1.0):
        if tick is not None:
            tick(f"flow planes: {len(out) + 1} of 5")
        q0 = fl.qc(np.array([0.0]))[0]; x = q0[0] + 0.75 * cr * np.cos(fl.geo(0.0)) + dx * cr        # root trailing edge + dx
        zq = fl.Qs[:, 2]
        ys = np.linspace(-b2 - 0.3 * cr, b2 + 0.3 * cr, 221); zs = np.linspace(min(zq.min(), 0.0) - 0.75 * cr, max(zq.max(), 0.0) + 0.3 * cr, 71)
        Y, Z = np.meshgrid(ys, zs, indexing="ij"); P = np.c_[np.full(Y.size, x), Y.ravel(), Z.ravel()]
        out.append(dict(kind="yz", at=x, u0=ys[0], u1=ys[-1], v0=zs[0], v1=zs[-1], nu=len(ys), nv=len(zs), q=enc(_q(fl, P))))
    return out


def sheet(fl, U, V):
    """wake centre behind each strip: at chord stations xi = 1..3.2, the zeta of the lowest decoder speed (-0.6 < zeta < 0.3,
    refined by a parabola through the minimum), in the wind frame; q of the composite field there"""
    xs = np.arange(1.0, 3.2001, 0.05); n = len(fl.y); P = np.zeros((n, len(xs), 3))
    zmask = (ZE > -0.6) & (ZE < 0.3); zz = ZE[zmask]
    for j in range(n):
        sp = np.hypot(U[j], V[j])[:, zmask]
        for m, x in enumerate(xs):
            i = np.clip(np.searchsorted(XI, x) - 1, 0, len(XI) - 2); t = (x - XI[i]) / (XI[i + 1] - XI[i])
            col = (1 - t) * sp[i] + t * sp[i + 1]; k = int(np.clip(np.argmin(col), 1, len(zz) - 2))
            a, b, c = col[k - 1], col[k], col[k + 1]; den = a - 2 * b + c
            ze = zz[k] + (0.5 * (a - c) / den * (zz[k + 1] - zz[k - 1]) / 2 if den > 1e-12 else 0.0)
            ze += fl.shift(np.array([[x, ze]]), np.array([j]))[0, 1] if fl.dz is not None else 0.0   # onto the 3D sheet
            ca, sa = np.cos(fl.ag[j]), np.sin(fl.ag[j])
            P[j, m] = [fl.c[j] * ((x - 0.25) * ca + ze * sa) + fl.Qs[j, 0], fl.y[j], fl.c[j] * (-(x - 0.25) * sa + ze * ca) + fl.Qs[j, 2]]
    q = _q(fl, P.reshape(-1, 3))
    return dict(nj=n, ni=len(xs), x=np.round(P[..., 0].ravel(), 4).tolist(), y=np.round(P[..., 1].ravel(), 4).tolist(),
                z=np.round(P[..., 2].ravel(), 4).tolist(), q=np.round(q, 3).tolist())


# ------------------------------------------------------------------------------------------- the page
def _r(a, d):
    """numbers -> (nested) lists rounded to d decimals, non-finite -> None (JSON null)"""
    a = np.round(np.asarray(a, np.float64), d)
    return np.where(np.isfinite(a), a, None).tolist()


def flow_field(field, w, V, case, secs, sid, loops, decoder, tick=None):
    """the page's flow-field block for `field` 'inviscid' or 'viscous' of the wing w at freestream V for case (Flow);
    secs, sid: strip_sections; loops: the strips' decoder sections (coords, index, family as wing_nll's res['loops']);
    decoder: the field decoder's weights file"""
    if field == "inviscid":
        return dict(kind="inviscid", decoder_warnings=[], cases=[dict(alpha=case["alpha"], lines=streamlines(Flow(w, V, case, secs, sid), tick))])
    import predict_field as PF
    Re, got, out, n2 = V * np.asarray(case["chord"]) / W.NU_AIR, {}, [], (len(sid) + 1) // 2
    for i, a in enumerate(case["alpha_eff"]):                           # mirror strips share one decoder call (same inputs)
        k = (loops["index"][i], f"{Re[i]:.4g}", round(float(a), 6))
        if k not in got:
            if tick is not None:
                tick(f"RANS field decoder: strip {len(got) + 1} of {n2} (mirror strips share one)")
            li = loops["index"][i]
            got[k] = sample(loops["coords"][li], loops["family"][li], a, Re[i], decoder)
        out.append(got[k])
    U, Vv = np.stack([o[0] for o in out]), np.stack([o[1] for o in out])
    fl = ViscousFlow(w, V, case, secs, sid, U, Vv)
    lines, pl = streamlines(fl, tick), planes(fl, tick)
    if tick is not None:
        tick("wake sheet")
    msgs = W._sst_warnings([o[2] for o in out], Re, *PF.FIELD_RE_OF.get(decoder, PF.FIELD_RE))   # grouped as the SST state's
    return dict(kind="viscous", decoder=os.path.basename(decoder), decoder_warnings=[m.replace("SST decoder: ", "", 1) for m in msgs],
                cases=[dict(alpha=case["alpha"], lines=lines, planes=pl, sheet=sheet(fl, U, Vv))])


def decoder_of(res):
    """the field decoder's weights file of a wing result: the SST state's own, else the released one"""
    import predict_field as PF
    return os.path.join(os.path.dirname(PF.DECODER), res["decoder"]) if res.get("decoder") else PF.DECODER


def re_floor(res):
    """the lowest strip Reynolds number the viscous flow field takes: 3/4 of its decoder's lower training Re (1.5e6 for the released
    decoder). Below it the decoder's near-wall flow turns into the section (NACA 2412, 1-5% chord above the upper surface: within
    1 deg of tangent at Re >= 1.5e6, 2-4 deg into it at 1e6, 7-16 deg at 5e5) and streamlines end inside the wing (full-scale
    wing: none at 30 m/s, 4 at 25 m/s; model wing at 20 m/s: 44 of 100). view_data refuses such a wing; the application greys
    the viscous field out for it."""
    import predict_field as PF
    return 0.75 * PF.FIELD_RE_OF.get(decoder_of(res), PF.FIELD_RE)[0]


def view_data(res, field="viscous", free_wake=False, progress=None, cancel=None):
    """The page's data for the result `res` of wing_nll.compute (either section state) at its incidence: {"wing": record,
    "field": flow-field record or None (field 'none')}; write_page puts them into the page. field: 'none', 'inviscid' or
    'viscous' (see the module doc; a ValueError, before any work, when a strip's Re is below re_floor);
    free_wake: relax the trailing vortex sheet (it rolls up), drawn and used by the field.
    progress(fraction, text) before each unit of work and cancel() returning True stops with RuntimeError('cancelled'), as
    wing_nll.compute (called from the calling thread)."""
    if field not in FIELDS:
        raise ValueError(f"field: one of {', '.join(FIELDS)}")
    import predict_field as PF
    d, s, c, alpha, V = res["defn"], res["strips"], res["coef"], float(res["alpha"]), float(res["V"])
    s = {k: np.asarray(v) for k, v in s.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", W.WingRangeWarning)              # the result's warnings are in res['warnings'] already
        w = W.Wing(stations=d["stations"], n=d["n"])
    n, L = w.n, res["loops"]
    dec, floor = decoder_of(res), re_floor(res)
    lo, hi = PF.FIELD_RE_OF.get(dec, PF.FIELD_RE)
    if field == "viscous" and s["Re"].min() < floor:
        e = lambda x: f"{x:.2g}".replace("e+0", "e").replace("e+", "e")
        raise ValueError(f"the viscous flow field needs Re >= {e(floor)} on every strip, and this wing's strips go down to Re "
                         f"{e(s['Re'].min())} (the RANS field decoder is trained at Re {e(lo)}-{e(hi)}; below {e(floor)} its "
                         f"near-wall flow crosses the section). Choose Inviscid streamlines, or a speed of at least "
                         f"{np.ceil(V * floor / s['Re'].min()):.0f} m/s.")
    tick = W._Ticker(progress, cancel)                                   # work units, estimated (the bar's fraction)
    tick.add((OUTER + 1) * ITERS * free_wake + (int(np.ceil((X_END - X0) / H / TRACE_TICK)) + 3) * (field != "none")
             + ((n + 1) // 2 + 6) * (field == "viscous"))
    case = dict(alpha=alpha, y=s["y"], chord=s["chord"], gamma=s["gamma"], cl=s["cl"], alpha_eff=s["alpha_eff_deg"])
    wake_model = "straight trailing legs along the freestream"
    if free_wake:
        tab = res["tables"]
        pol = W.table_polar(dict(alpha=np.radians(tab["alpha_deg"]), cl=tab["cl"], cd=tab["cd"], cm=tab["cm"]))
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")                              # stall notes: in res['warnings'] already
            f = solve_free(w, V, alpha, pol, tick)
        case = dict(alpha=alpha, y=f["y"], chord=w.c, gamma=f["gamma"], cl=f["cl"], alpha_eff=f["alpha_eff"], wake=f["wake"])
        wake_model = (f"relaxed free wake (rolled up); the flow field uses its solution: C_L {f['CL']:.5f} "
                      f"(straight trailing legs: {c['CL']:.5f})")
    fld = None if field == "none" else flow_field(field, w, V, case, *strip_sections(res), L, dec, tick)
    if fld is not None:                                                  # the SST state's result lists the same ones already
        fld["decoder_warnings"] = [m for m in fld["decoder_warnings"] if f"SST decoder: {m}" not in res["warnings"]]

    label = lambda t: f"NACA {t}" if W.is_naca(t) else os.path.splitext(os.path.basename(t.replace("uiuc:", "")))[0]
    names = [label(st["section"]) for st in d["stations"]]               # no local paths in the page
    S, t, wg, sst = res["surface"], res["trailing"], res["wing"], res["state"] == "sst"
    num = lambda v: float(v) if np.isfinite(v) else None                 # JSON: no NaN (e.g. e where C_Di <= 0)
    keys = ("CL", "CDi", "CD0", "CD", "CM", "LD", "e") + (("CD0_pressure", "CD0_friction") if sst else ())
    rec_case = dict(alpha=alpha, **{k: num(c[k]) for k in keys}, strips_past_stall=int(c["strips_past_stall"]),
                    iterations=int(res["newton"]["iterations"]), residual=num(res["newton"]["residual"]),
                    y=_r(s["y"], 5), chord=_r(s["chord"], 5), cl=_r(s["cl"], 5), alpha_eff=_r(s["alpha_eff_deg"], 4),
                    gamma=_r(s["gamma"], 6),
                    surface={side: dict(x=_r(S[side]["X"], 5), y=_r(S[side]["Y"], 5), z=_r(S[side]["Z"], 5), cp=_r(S[side]["C_p"], 4))
                             for side in ("upper", "lower")},
                    trailing=[dict(y=num(t["y"][k]), x0=num(t["x_c4"][k]), z0=num(t["z_c4"][k]), strength=num(t["strength"][k]))
                              for k in range(n + 1)])
    if free_wake:
        rec_case["wake"] = _r(case["wake"], 5)
    lead = (res.get("sst_range") or {}).get("message")                  # SST state mostly outside the decoder's Re range (B4)
    P, pk = res["polar"], ("CL", "CDi", "CD0", "CD", "CM", "LD", "e")
    polar = [dict(alpha=float(a), **{q: num(P[q][i]) for q in pk}) for i, a in enumerate(P["alpha"]) if a != alpha]
    polar.append(dict(alpha=alpha, **{q: rec_case[q] for q in pk}, y=rec_case["y"], cl=rec_case["cl"]))   # this incidence: the result's
    rec = dict(wing=dict(name=wg["name"], section=" to ".join(dict.fromkeys(names)), state=res["state"],
                         sections_state="SST (RANS field decoder: wall pressure, wall-function friction)" if sst else "XFOIL (VortFoil surface model)",
                         V=V, span=wg["b"], S=wg["S"], AR=wg["AR"], MAC=wg["MAC"], taper=wg["taper"], c_root=wg["c_root"], c_tip=wg["c_tip"],
                         Re_root=wg["Re_root"], Re_tip=wg["Re_tip"], n=n,
                         stations=[dict({k: st[k] for k in ("y", "chord", "twist", "x", "z")}, section=nm) for st, nm in zip(d["stations"], names)]),
               cases=[rec_case], polar=sorted(polar, key=lambda p: p["alpha"]), focus=alpha, warnings=([lead] if lead else []) + list(res["warnings"]),
               wake_model=wake_model)
    if progress is not None:
        progress(1.0, "done")
    return dict(wing=rec, field=fld)


def write_page(data, path):
    """the standalone page: TEMPLATE with each block of `data` (view_data) as gzip + base64 JSON (unpacked by the browser) in
    place of its placeholder __<KEY>_DATA__ (None: empty); written atomically (a finished page is never partial); the path"""
    pack = lambda o: "" if o is None else base64.b64encode(gzip.compress(json.dumps(o, separators=(",", ":"), allow_nan=False).encode(), 6, mtime=0)).decode()
    with open(TEMPLATE, encoding="utf-8") as fh:
        page = fh.read()
    for k, v in data.items():
        page = page.replace(f"__{k.upper()}_DATA__", pack(v))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".part", "w", encoding="utf-8") as fh:
        fh.write(page)
    os.replace(path + ".part", path)
    return path
