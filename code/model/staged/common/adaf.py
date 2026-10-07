# -*- coding: utf-8 -*-
"""
Anti-Derivative Approximator (ADAF) — Fourier basis + analytical integration.

Maps a panel-node weight vector W ∈ ℝ^N_p to a smooth function on a fixed
grid via:
    W → Fourier coefficients (a_n, b_n) → analytical integration g1, g2, g3

Properties (operator-learning relevance):
    - Spectral 1/k damping from analytical antiderivative (smoothness prior).
    - Exact derivatives — dg1/dx is recovered as a linear combination of
      W and the same basis (no autograd / FD needed for ODE residuals).
      CAVEAT (defect D1, 2026-09-02): the fixed-grid routine that claims
      to do this, `g1x_from_coeffs`, is off by a factor of 2 and is kept
      that way deliberately.  `g1x_from_coeffs_corrected` and
      `evaluate_g1_g1x_at_xi` are the correct ones.  See the comment
      block above `g1x_from_coeffs`.
    - Mode truncation N_m acts as explicit regularization.
    - Boundary conditions enforced via separate `init1, init2, init3`
      additive terms (clean separation of BC from interior solution).

Extracted verbatim from steady51.py with no physics changes.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

DTYPE = tf.float32


class BatchedADAFReusable(tf.Module):
    """
    Full-Fourier reusable ADAF on a fixed normalized grid x in [0, gamma].

    Inputs:
        W      : (B, N_p)   panel-node weights from a network
        init1  : (B, 1)     bias for g1 at x=0
        init2  : (B, 1)     bias for g2
        init3  : (B, 1)     bias for g3

    Outputs:
        g1, g2, g3 — first / second / third anti-derivatives evaluated
        on the fixed `xgrid_norm` grid, batched over B.
    """

    def __init__(self, xgrid_norm, N_p: int = 100, N_m: int = 100,
                 L: float = 1.0, gamma: float = 1.0,
                 dtype=DTYPE, name=None):
        super().__init__(name=name)
        self.dtype = dtype
        self.L = tf.constant(L, dtype)
        self.gamma = float(gamma)
        self.N_p = int(N_p)
        self.N_m = int(N_m)

        xgrid_norm = np.asarray(xgrid_norm, dtype=np.float32).reshape(-1)
        if np.any(np.diff(xgrid_norm) < 0):
            raise ValueError("xgrid_norm must be monotone increasing")
        if xgrid_norm[0] < -1e-7 or xgrid_norm[-1] > self.gamma + 1e-7:
            raise ValueError("xgrid_norm must lie in [0, gamma]")

        # Panel nodes
        if gamma == 1.0:
            x_i_np = np.linspace(0.0, L, N_p + 1, dtype=np.float32)
        else:
            x_i_np = np.concatenate(
                (np.linspace(0.0, gamma, N_p - 1, dtype=np.float32)[:-1],
                 np.linspace(gamma, 1.0, 3, dtype=np.float32)),
                axis=0,
            ).astype(np.float32)

        self.x_i = tf.constant(x_i_np, dtype=dtype)
        x1 = self.x_i[1:]
        x2 = self.x_i[:-1]

        n = tf.range(1, self.N_m + 1, dtype=dtype)
        PI_tf = tf.constant(np.pi, dtype=dtype)

        self.scale = 2.0 / (n * PI_tf)            # (N_m,)
        self.factor = self.L / (n * PI_tf)        # (N_m,)

        k = (n * PI_tf / self.L)[:, None]         # (N_m, 1)

        cos_kx1 = tf.cos(k * x1[None, :])
        cos_kx2 = tf.cos(k * x2[None, :])
        sin_kx1 = tf.sin(k * x1[None, :])
        sin_kx2 = tf.sin(k * x2[None, :])

        self.Mb = (-cos_kx1 + cos_kx2)            # (N_m, N_p)
        self.Ma = (sin_kx1 - sin_kx2)             # (N_m, N_p)

        x = tf.convert_to_tensor(xgrid_norm, dtype=dtype)   # (Nx,)
        self.x = x
        self.x2 = tf.square(x)
        self.x3 = tf.pow(x, 3)

        xf = x[None, :] / self.factor[:, None]    # (N_m, Nx)
        self.S = tf.sin(xf)
        self.C = tf.cos(xf)
        self.Cm1 = self.C - 1.0
        self.one_m_C = 1.0 - self.C

        self.f = self.factor[:, None]             # (N_m, 1)
        self.f2 = tf.square(self.f)               # (N_m, 1)

    # ---- Fourier projection of W ----
    def coeffs(self, W):
        """W: (B, N_p) → (a0, a_n, b_n)."""
        bn_num = tf.linalg.matmul(W, self.Mb, transpose_b=True)   # (B, N_m)
        an_num = tf.linalg.matmul(W, self.Ma, transpose_b=True)
        b_n = bn_num * self.scale[None, :]
        a_n = an_num * self.scale[None, :]
        a0 = tf.reduce_mean(W, axis=1, keepdims=True)              # (B, 1)
        return a0, a_n, b_n

    # ---- Anti-derivatives ----
    def g1_from_coeffs(self, a0, a_n, b_n, init1):
        term = (
            0.5 * (-self.f[None, :, :] * b_n[:, :, None]) * self.Cm1[None, :, :]
            + 0.5 * (self.f[None, :, :] * a_n[:, :, None]) * self.S[None, :, :]
        )
        return (
            0.5 * a0 * self.x[None, :]
            + tf.reduce_sum(term, axis=1)
            + init1
        )

    def g2_from_coeffs(self, a0, a_n, b_n, init1, init2):
        term1 = 0.5 * (-self.f[None, :, :] * b_n[:, :, None]) * (
            self.f[None, :, :] * self.S[None, :, :] - self.x[None, None, :]
        )
        term2 = 0.5 * (self.f2[None, :, :] * a_n[:, :, None]) * self.one_m_C[None, :, :]
        return (
            (a0 / 4.0) * self.x2[None, :]
            + tf.reduce_sum(term1 + term2, axis=1)
            + init1 * self.x[None, :]
            + init2
        )

    def g3_from_coeffs(self, a0, a_n, b_n, init1, init2, init3):
        term1 = 0.5 * (self.f[None, :, :] * b_n[:, :, None]) * (
            0.5 * self.x2[None, None, :] + self.f2[None, :, :] * self.Cm1[None, :, :]
        )
        term2 = 0.5 * (self.f2[None, :, :] * a_n[:, :, None]) * (
            self.x[None, None, :] - self.f[None, :, :] * self.S[None, :, :]
        )
        return (
            (a0 / 12.0) * self.x3[None, :]
            + tf.reduce_sum(term1 + term2, axis=1)
            + 0.5 * init1 * self.x2[None, :]
            + init2 * self.x[None, :]
            + init3
        )

    def evaluate_g123(self, W, init1, init2, init3):
        a0, a_n, b_n = self.coeffs(W)
        g1 = self.g1_from_coeffs(a0, a_n, b_n, init1)
        g2 = self.g2_from_coeffs(a0, a_n, b_n, init1, init2)
        g3 = self.g3_from_coeffs(a0, a_n, b_n, init1, init2, init3)
        return g1, g2, g3

    # ============================================================
    # Spectral first derivative on the fixed grid (no FD, no AD)
    #
    # DEFECT D1 [found 2026-09-02, kept deliberately -- read this before
    # you "fix" anything here].  `g1x_from_coeffs` below does NOT return
    # dg1/dx.  Differentiating `g1_from_coeffs` AS IMPLEMENTED gives
    #
    #   TRUE  g1'(x) = a0/2 + (1/2) Σ_n [ a_n cos(k_n x) + b_n sin(k_n x) ]
    #
    # but `g1x_from_coeffs` returns the Fourier sum with NO 1/2:
    #
    #   RETURNED     = a0/2 + (1)   Σ_n [ a_n cos(k_n x) + b_n sin(k_n x) ]
    #                = 2 * TRUE - a0/2        (exact; float32 residual 2.4e-07)
    #
    # so it is ~2x the derivative (median ratio vs central FD: 1.9846 on
    # the deployed 240-panel / N_m=24 geometry, 1.9917 on the audit probe's
    # fixture; both bracket 2 and neither is 1).  The comment that used to
    # sit here stated the no-1/2 formula, i.e. code and comment AGREED with
    # each other and both disagreed with calculus -- which is exactly why
    # this survived unnoticed.  `evaluate_g1_g1x_at_xi` (below) carries the
    # 1/2 and IS correct.
    #
    # The wrong symbol is retained, unchanged and bug-compatible, because
    # the legacy reproduction path must keep reproducing (author decision,
    # 2026-09-02).  Use `g1x_from_coeffs_corrected` for anything new.
    # NOT on the deployed gamma readout: the deployed OP1 path calls only
    # `coeffs` + `g1_from_coeffs` (op1_inviscid.py:325-326, 332-333), and no
    # file in `.neurofoil_clone/staged/` calls `g1x_from_coeffs` at all --
    # verified by live poison trace, VortFoil/code/trace_g1x_callgraph.py.
    # The only numerically live consumer found repo-wide is the professor's
    # pre-closure-redesign Op2 snapshot,
    # `neurofoil/physics_based_reserve/code/staged/op2_bl.py:104,123,146,179,197`,
    # which feeds it into the BL ODE residuals R_theta / R_theta_Hstar / R_S.
    #
    # The trig tables `self.C` (cos k_n x_j) and `self.S` (sin k_n x_j)
    # were already precomputed at __init__ for `g1_from_coeffs`, so both
    # variants are a pure matmul on the same (N_m, Nx) tables — same cost
    # as a single Fourier evaluation.
    # ============================================================
    def g1x_from_coeffs(self, a0, a_n, b_n):
        """BUG-COMPATIBLE (defect D1): returns 2*g1'(x) - a0/2, NOT g1'(x).

        Kept verbatim on purpose so the legacy reproduction path stays
        bit-identical.  See the comment block above.  For the real
        derivative use `g1x_from_coeffs_corrected`.  Returns (B, Nx).
        """
        sum_term = tf.reduce_sum(
            a_n[:, :, None] * self.C[None, :, :]
            + b_n[:, :, None] * self.S[None, :, :],
            axis=1
        )
        return 0.5 * a0 + sum_term

    def g1x_from_coeffs_legacy(self, a0, a_n, b_n):
        """Explicitly-named alias of the bug-compatible `g1x_from_coeffs`.

        Delegates rather than copying the body, so the two names can never
        drift apart.  Use this name when the 2x behaviour is INTENDED (legacy
        reproduction); it documents the intent at the call site.
        """
        return self.g1x_from_coeffs(a0, a_n, b_n)

    def g1x_from_coeffs_corrected(self, a0, a_n, b_n):
        """TRUE dg1/dx on the fixed grid: a0/2 + (1/2) Σ_n [a_n·C + b_n·S].

        Matches central FD of `g1_from_coeffs` (ratio 1.0) and the analytic
        derivative.  Same 1/2 convention as `evaluate_g1_g1x_at_xi`.  Returns
        (B, Nx).  New code should use this; existing callers are NOT
        re-pointed here -- that is the G1X0/G1X1 ablation's job.
        """
        sum_term = tf.reduce_sum(
            a_n[:, :, None] * self.C[None, :, :]
            + b_n[:, :, None] * self.S[None, :, :],
            axis=1
        )
        return 0.5 * a0 + 0.5 * sum_term

    def evaluate_g1_g1x(self, W, init1):
        """g1(x) and the BUG-COMPATIBLE g1x (defect D1) on the fixed grid.

        The second return value is `g1x_from_coeffs`, i.e. 2*g1'(x) - a0/2,
        NOT g1'(x).  Left wired to the legacy routine on purpose so this
        wrapper stays bug-compatible too; it has no caller anywhere in the
        repo (live poison trace: VortFoil/code/trace_g1x_callgraph.py).
        """
        a0, a_n, b_n = self.coeffs(W)
        g1  = self.g1_from_coeffs (a0, a_n, b_n, init1)
        g1x = self.g1x_from_coeffs(a0, a_n, b_n)
        return g1, g1x

    # ============================================================
    # v51-fix [2026-05-08] — arbitrary-ξ spectral evaluation
    #
    # The fixed `xgrid_norm` passed at __init__ pre-computes self.S and
    # self.C trig tables, which makes the standard `g1_from_coeffs` and
    # the gradient `g1_x = a0/2 + an·C + bn·S` operate ONLY on that
    # grid.  For random refinement collocation (option A) we need to
    # evaluate g1 and g1_x at arbitrary ξ each step.
    #
    # Implementation: re-build only the trig tables for the new ξ.
    # The Fourier projection W → (a0, an, bn) is grid-independent and
    # computed once via self.coeffs(W).  Cost ~ N_m × N_extra extra
    # cos/sin per call (cheap; N_m=32, N_extra typically 200-500).
    #
    # Returns g1 AND g1_x (= dg1/dx) — both spectrally exact.  This
    # replaces the central_diff_1d FD derivative used elsewhere with a
    # higher-accuracy spectral derivative, which is what ADAF was built
    # to enable.
    # ============================================================
    def evaluate_g1_g1x_at_xi(self, W, xi_arbitrary):
        """
        Evaluate g1(x) and g1'(x) at arbitrary ξ using the same Fourier
        coefficients (no FD).

        Parameters
        ----------
        W            : (B, N_p)   panel-node weights  (same as used in g1)
        xi_arbitrary : (N_extra,) ξ values in [0, gamma]; SHARED across B

        Returns
        -------
        g1   : (B, N_extra)
        g1_x : (B, N_extra)
        """
        a0, a_n, b_n = self.coeffs(W)                          # (B, N_m), ...

        n = tf.range(1, self.N_m + 1, dtype=self.dtype)
        PI = tf.constant(np.pi, dtype=self.dtype)
        factor = self.L / (n * PI)                              # (N_m,)
        f = factor[:, None]                                     # (N_m, 1)

        xf = xi_arbitrary[None, :] / f                          # (N_m, N_extra)
        S_ext = tf.sin(xf)
        C_ext = tf.cos(xf)
        Cm1 = C_ext - 1.0

        # g1 = (a0/2)·x + Σ_n [ -(L/nπ)·b_n · (cos(...)-1)/2 + (L/nπ)·a_n·sin(...)/2 ]
        term = (
            0.5 * (-f[None, :, :] * b_n[:, :, None]) * Cm1[None, :, :]
            + 0.5 * (f[None, :, :] * a_n[:, :, None]) * S_ext[None, :, :]
        )
        g1 = (
            0.5 * a0 * xi_arbitrary[None, :]
            + tf.reduce_sum(term, axis=1)
        )

        # g1_x = a0/2 + Σ_n [ a_n · cos(nπx/L) + b_n · sin(nπx/L) ]   (·1/2)
        g1_x = (
            0.5 * a0
            + 0.5 * tf.linalg.matmul(a_n, C_ext)
            + 0.5 * tf.linalg.matmul(b_n, S_ext)
        )
        return g1, g1_x
