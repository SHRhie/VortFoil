# -*- coding: utf-8 -*-
"""
Op1 -- Inviscid panel-method PINN  (Phase A.4: Kulfan CST generalization).

Stage 1 of the staged operator-learning curriculum.

Inputs:
    shape_params (B, 2 * n_bern)   Kulfan CST coefficients
                                   default n_bern = 8  ->  (B, 16)
                                   [a_u_0..7, a_l_0..7]
    alpha_deg    (B, 1)
    Re           (B, 1)            only fed to cond_stag for downstream
                                   compatibility

Trainable:
    cond_gamma   Branch-Trunk + ADAF head -> ADAF weights for gamma_inv
    cond_stag    ConditionNet              -> s_stag scalar

Forward (inviscid only -- no source, no BL):
    gamma_inv(s) = ADAF(W) with sum-Kutta linear correction
    s_stag       in [s_le +/- stag_le_half_width]
    Ut, Un       = U_inf * t_hat + vortex-induced

Loss = L_kutta + L_np + L_stag.

Migration notes (Phase A.4):
    - Geometry source: build_single_loop_contour_cst_tf (was naca4 variant).
    - Branch input grows from 3 (m, p, t) to 16 (CST) so cond_gamma's
      branch input dim is now 18 (16 + sin/cos alpha) and cond_stag's
      is 19 (+ log Re normalized).
    - NACA 4-digit weight files are NOT interchangeable -- a fresh
      train run is required.  Pre-Phase-A weights live in
      staged/weights/op1_final_98pct_backup.npz (legacy backup).

Used downstream:
    Op2 still receives the legacy NACA4 (B, 3) tensor; the Op1->Op2
    bridge converts via staged.common.naca4_to_cst_coeffs at the call
    site (until Op2 also migrates).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# ---- operator_lib import path (LPA basis lives there) ----
# Mirror adafol_train_precision.py's `_OPLIB` pattern: the sibling
# operator_lib package (two levels up from this file -> repo root's
# parent) supplies models/lpa_basis.py.  Added lazily so the default
# Fourier path never imports it.
_HERE = os.path.dirname(os.path.abspath(__file__))                # .../.neurofoil_clone/staged
_REPO = os.path.dirname(os.path.dirname(_HERE))                   # .../airfoil_operator
_OPLIB = os.path.join(os.path.dirname(_REPO), "operator_lib")     # .../Operator/operator_lib
if os.path.isdir(_OPLIB) and _OPLIB not in sys.path:
    sys.path.insert(0, _OPLIB)

from .common import (
    BatchedADAFReusable,
    BranchTrunkADAFHead,
    ConditionNet,
    CST_N_BERN_DEFAULT,
    build_single_loop_contour_cst_tf,
    compute_inviscid_surface_velocity,
    encode_flow,
    normalize_log_re,
)
from .common.profile_encoder import (
    ProfileEncoder, build_profile_from_cst,
)

DTYPE = tf.float32


# ============================================================
# Model
# ============================================================
class Op1Inviscid(tf.Module):
    """Stage-1 inviscid operator: (CST shape, alpha, Re) -> gamma_inv, s_stag."""

    def __init__(
        self,
        n_surface_pts: int = 61,
        n_bern: int = CST_N_BERN_DEFAULT,
        N_p_gamma: int = 240,
        N_m_gamma: int = 24,
        hidden: int = 128,
        branch_dim: int = 128,
        lambda_kutta: float = 1.0,
        lambda_stag: float = 10.0,
        stag_le_half_width: float = 0.15,
        use_profile_enc: bool = False,
        profile_enc_hidden: int = 64,
        profile_enc_kernel: int = 5,
        profile_enc_depth: int = 4,
        gamma_basis: str = "fourier",
        lpa_max_order: int = 23,
        mixed_lpa_order: int = 7,
        name: str = "op1_inviscid",
    ):
        super().__init__(name=name)

        self.gamma_basis = str(gamma_basis).lower()
        if self.gamma_basis not in ("fourier", "lpa", "mixed"):
            raise ValueError(
                f"gamma_basis must be 'fourier'|'lpa'|'mixed', got {gamma_basis!r}")
        self.lpa_max_order = int(lpa_max_order)
        self.mixed_lpa_order = int(mixed_lpa_order)
        self.use_profile_enc = bool(use_profile_enc)
        self.n_bern = int(n_bern)
        self.shape_dim = 2 * self.n_bern                  # 16 for default
        self.n_surface_pts = int(n_surface_pts)
        self.Nside = self.n_surface_pts - 1
        self.Np = 2 * self.Nside
        self.Nnode = self.Np + 1

        self.lambda_kutta = tf.constant(lambda_kutta, dtype=DTYPE)
        self.lambda_stag = tf.constant(lambda_stag, dtype=DTYPE)
        self.stag_le_half_width = tf.constant(stag_le_half_width, dtype=DTYPE)
        self.s_le = tf.constant(0.5, dtype=DTYPE)

        # Residual-ADAF patch: discrepancy head & CDCALC
        self.use_discrepancy = True
        self.lambda_cd = tf.constant(1.0, dtype=DTYPE)
        self.lambda_sobolev = tf.constant(0.01, dtype=DTYPE)  # for Δψ

        self.pi_over_180 = tf.constant(np.pi / 180.0, dtype=DTYPE)
        self.s_query_min = tf.constant(0.5 / self.Np, dtype=DTYPE)
        self.s_query_max = tf.constant(1.0 - 0.5 / self.Np, dtype=DTYPE)

        s_nodes_np = np.linspace(0.0, 1.0, self.Nnode, dtype=np.float32)
        self.s_nodes_1d = tf.constant(s_nodes_np, dtype=DTYPE)
        self.s_panels_1d = tf.constant(
            np.linspace(0.5 / self.Np, 1.0 - 0.5 / self.Np,
                        self.Np, dtype=np.float32), dtype=DTYPE,
        )

        # ---- γ_inv decoder (constants — not trainable) ----
        # Two interchangeable bases, both mapping network panel weights W
        # → γ̃ on the SAME uniform node grid s_nodes = linspace(0,1,Nnode):
        #   - "fourier": BatchedADAFReusable.g1_from_coeffs  (W→Fourier→∫)
        #   - "lpa"    : BatchLPABasis  (W→Legendre coeffs→antiderivative,
        #                 x(0)=x0 hard IC, x = x0 + ∫W).  Both are the
        #                 g1-equivalent; the sum-Kutta correction downstream
        #                 is basis-agnostic.
        if self.gamma_basis == "fourier":
            self.gamma_decoder = BatchedADAFReusable(
                s_nodes_np, N_p=N_p_gamma, N_m=N_m_gamma,
                gamma=1.0, name="gamma_decoder",
            )
            self._gamma_Np = int(N_p_gamma)
        elif self.gamma_basis == "mixed":
            # 3-way protocol (2026-07-02): Fourier backbone (carries init1, the
            # proven winner for the periodic gamma loop) + HOMOGENEOUS LPA
            # correction (x0=0 -> vanishes at s=0; pure low-order polynomial
            # trend atoms added on top).  W = [W_fourier | W_lpa] split below.
            try:
                from models.lpa_basis import BatchLPABasis
            except ImportError as exc:                            # not distributed with the package
                raise ImportError(
                    "gamma_basis='%s' needs the Legendre basis module (models.lpa_basis), which is "
                    "not distributed here; the deployed model uses the Fourier basis "
                    "(gamma_basis='fourier')." % gamma_basis) from exc
            self.gamma_decoder = BatchedADAFReusable(
                s_nodes_np, N_p=N_p_gamma, N_m=N_m_gamma,
                gamma=1.0, name="gamma_decoder",
            )
            lpa_Np = self.mixed_lpa_order + 1
            self.gamma_decoder_lpa = BatchLPABasis(
                T_seg=1.0, Nt=self.Nnode, gamma=1.0,
                max_order=self.mixed_lpa_order, N_p=lpa_Np, dtype=DTYPE,
                name="gamma_decoder_mixlpa",
            )
            if not np.allclose(self.gamma_decoder_lpa.t_local_np, s_nodes_np,
                               atol=1e-6):
                raise RuntimeError(
                    "mixed: BatchLPABasis grid does not match s_nodes; "
                    "sum-Kutta would be invalid.")
            self._gamma_Np_fourier = int(N_p_gamma)
            self._gamma_Np = int(N_p_gamma) + int(self.gamma_decoder_lpa.coef_dim)
        else:  # lpa
            try:
                from models.lpa_basis import BatchLPABasis
            except ImportError as exc:                            # not distributed with the package
                raise ImportError(
                    "gamma_basis='%s' needs the Legendre basis module (models.lpa_basis), which is "
                    "not distributed here; the deployed model uses the Fourier basis "
                    "(gamma_basis='fourier')." % gamma_basis) from exc
            # coef_dim = N_p (network W width), INDEPENDENT of max_order.
            # Match the Fourier path's effective DOF (N_m=24 modes) by
            # emitting N_p = lpa_max_order+1 panel weights projected onto
            # max_order Legendre modes (full-rank: 24 panels -> P_0..P_23).
            lpa_Np = self.lpa_max_order + 1
            self.gamma_decoder = BatchLPABasis(
                T_seg=1.0, Nt=self.Nnode, gamma=1.0,
                max_order=self.lpa_max_order, N_p=lpa_Np, dtype=DTYPE,
                name="gamma_decoder_lpa",
            )
            # Verify the LPA node grid coincides with the Fourier s_nodes
            # (uniform 0..1, Nnode points) — required for the sum-Kutta
            # correction + downstream panel kernel to be basis-agnostic.
            if not np.allclose(self.gamma_decoder.t_local_np, s_nodes_np,
                               atol=1e-6):
                raise RuntimeError(
                    "BatchLPABasis grid t_local does not match s_nodes "
                    "(linspace(0,1,Nnode)); sum-Kutta would be invalid.")
            # cond_gamma must emit W ∈ R^{coef_dim} = R^{N_p} for the LPA.
            self._gamma_Np = int(self.gamma_decoder.coef_dim)

        # ---- Trainable networks ----
        # Phase A.4 [2026-05-15]: shape dim grew NACA4(3) -> CST(16).
        # cond_gamma:  18-dim cond (CST[16] + sin alpha + cos alpha)
        #              -> Branch-Trunk + ADAF head -> (W in R^{N_p}, init_1).
        #              W goes through `gamma_decoder` (ADAF) to give gamma_tilde
        #              on the contour node grid.  Re is omitted from
        #              gamma-conditioning since gamma_inv is Re-independent.
        # cond_stag:   19-dim cond (CST[16] + sin alpha + cos alpha + log_Re_norm)
        #              -> scalar offset -> s_stag.  Stagnation position depends
        #              on Re only weakly (transition near-stag).
        self.cond_gamma = BranchTrunkADAFHead(
            N_p=self._gamma_Np,
            branch_dim=branch_dim,
            branch_depth=3,
            trunk_depth=2,
            activation="gelu",
            last_layer_scale=0.02,
            # Pinned OFF for the deployed path.  Three reasons, in order of
            # weight: (1) the shipped gamma weights (_opt3_gamma.npz, 13
            # tensors) were fitted with no W-projection, and applying it costs
            # Cl R2 0.983429 -> 0.961659 and Cm R2 0.855694 -> 0.783680 on the
            # 2730-row K18 holdout; (2) Kutta is ALREADY exact downstream --
            # gamma_nodes = gamma_tilde - s_nodes * (gamma_tilde[0] +
            # gamma_tilde[-1]) below, residual ~1e-7 -- so the projection is
            # redundant; (3) it constrains the wrong quantity, since W are
            # rate-panel weights feeding the ADAF antiderivative and
            # gamma(0) + gamma(L) = 0 is not W[Nside-1] + W[N-1] = 0.
            # Guard: VortFoil/code/v1_kutta_guard.py
            enforce_kutta=False,
            name="cond_gamma_bt",
        )
        self.cond_stag = ConditionNet(
            out_dim=1,
            hidden=64, depth=2,
            activation="gelu",
            use_skip=False,
        )

        # Optional 1D causal-conv profile encoder.  Captures local geometric
        # detail along the chord direction (LE→TE) that CST's 16 global
        # Bernstein coefficients smooth over.  Causal padding preserves the
        # physical direction of BL propagation: state at chord x depends on
        # upstream geometry [0, x].
        if self.use_profile_enc:
            self.profile_enc = ProfileEncoder(
                n_pts=self.n_surface_pts,
                hidden=profile_enc_hidden,
                kernel=profile_enc_kernel,
                depth=profile_enc_depth,
                pool="mean_plus_te",
                name="op1_profile_enc",
            )
        else:
            self.profile_enc = None

        # ---- Residual-ADAF patch: Discrepancy head ----
        # Outputs: ΔADAF_coeff (N_p), ΔBL_closure (6: theta, H, S per side), Δwake (3)
        # Shared trunk + task-specific heads
        self.disc_hidden = [
            tf.keras.layers.Dense(128, activation="gelu", name=f"disc_h{i}")
            for i in range(2)
        ]
        # Task-specific output heads
        self.disc_adaf_out = tf.keras.layers.Dense(
            self._gamma_Np, kernel_initializer="zeros", bias_initializer="zeros",
            name="disc_adaf_out")
        self.disc_bl_out = tf.keras.layers.Dense(
            6, kernel_initializer="zeros", bias_initializer="zeros",
            name="disc_bl_out")  # theta_u, H_u, S_u, theta_l, H_l, S_l
        self.disc_wake_out = tf.keras.layers.Dense(
            3, kernel_initializer="zeros", bias_initializer="zeros",
            name="disc_wake_out")  # theta_w, disp_w, Ue_ratio_w

        # ---- Field reconstruction: Δψ stream function correction (v2 style) ----
        # Δψ decoder on fixed query grid
        self.field_grid_pts = 200
        self.field_decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="gelu", name="field_h0"),
            tf.keras.layers.Dense(64, activation="gelu", name="field_h1"),
            tf.keras.layers.Dense(1, kernel_initializer="zeros", bias_initializer="zeros", name="field_out"),
        ], name="field_delta_psi")

        # CDCALC endpoint (port from VortFoil_v2) - differentiable primitive
        # We'll call it from the loss when BL state is available

    # --------------------------------------------------------
    # Inference building blocks
    # --------------------------------------------------------
    def predict_gamma(self, shape_params, alpha_deg, Re=None,
                       training: bool = False) -> dict:
        """Sum-Kutta gamma prediction.  gamma(0) + gamma(L) = 0 (structural).

        Conditioning is shape (CST coefficients, 16-dim) + flow
        (sin α, cos α) + log Re_norm = 19-dim.

        Re argument:
          - None (legacy):  uses placeholder Re=1e6.  Combined with
            zero-padded Re-column in cond_gamma's first kernel (after
            load_weights of a pre-Re-input checkpoint), this exactly
            reproduces the Re-independent inviscid γ behaviour.
          - given:  Re-aware γ.  Captures small but real Re effect on
            effective γ when supervised on XFOIL viscous Cp (transpiration
            σ effect varies with Re even though pure inviscid γ doesn't).
        """
        alpha_rad = alpha_deg * self.pi_over_180
        sin_a = tf.sin(alpha_rad)
        cos_a = tf.cos(alpha_rad)
        if Re is None:
            Re_in = tf.ones_like(sin_a) * 1.0e6
        else:
            Re_in = tf.cast(Re, DTYPE)
        log_re = normalize_log_re(Re_in)
        cond = tf.concat([shape_params, sin_a, cos_a, log_re], axis=1)
        # cond shape (B, shape_dim + 3) — append profile-conv features if on
        if self.profile_enc is not None:
            profile = build_profile_from_cst(
                shape_params, n_surface_pts=self.n_surface_pts,
                n_bern=self.n_bern)
            cond = tf.concat(
                [cond, self.profile_enc(profile, training=training)], axis=1)

        # Branch-Trunk → raw panel weights W + init_1
        W, init1 = self.cond_gamma(cond, training=training)    # (B, N_p), (B, 1)

        if self.gamma_basis == "fourier":
            a0, a_n, b_n = self.gamma_decoder.coeffs(W)
            gamma_tilde = self.gamma_decoder.g1_from_coeffs(a0, a_n, b_n, init1)
        elif self.gamma_basis == "mixed":
            # Fourier backbone (init1 anchor) + homogeneous LPA correction
            # (x0=0).  W = [W_fourier (N_p_gamma) | W_lpa (mixed_lpa_order+1)].
            Wf = W[:, :self._gamma_Np_fourier]
            Wl = W[:, self._gamma_Np_fourier:]
            a0, a_n, b_n = self.gamma_decoder.coeffs(Wf)
            g_f = self.gamma_decoder.g1_from_coeffs(a0, a_n, b_n, init1)
            g_l = self.gamma_decoder_lpa(
                Wl[:, None, :], tf.zeros_like(init1))["x"][:, :, 0]
            gamma_tilde = g_f + g_l
        else:  # lpa — γ̃ = x0 + ∫W on the SAME uniform node grid.
            # BatchLPABasis(W_batch (B,J,N_p), x0 (B,J)) → dict; 'x' has
            # shape (B, Nt, J) (transposed internally).  Single channel
            # J=1: W→(B,1,N_p), x0=init1 (B,1), take [:, :, 0] → (B, Nnode).
            gamma_tilde = self.gamma_decoder(
                W[:, None, :], init1)["x"][:, :, 0]

        s_nodes = self.s_nodes_1d[None, :]
        end_sum = gamma_tilde[:, :1] + gamma_tilde[:, -1:]
        gamma_nodes = gamma_tilde - s_nodes * end_sum

        gamma_panels = 0.5 * (gamma_nodes[:, :-1] + gamma_nodes[:, 1:])

        return {
            "gamma_nodes": gamma_nodes,
            "gamma_panels": gamma_panels,
            "gamma_kutta_sum": gamma_nodes[:, :1] + gamma_nodes[:, -1:],
        }

    def predict_s_stag(self, shape_params, alpha_deg, Re,
                        training: bool = False):
        """Stagnation s_stag in [s_le +/- stag_le_half_width].

        Conditioning is shape (CST, 16-dim) + flow (sin a, cos a, log_Re_norm).

        WARNING -- this is an auxiliary regressed scalar, not the flow feature.
        `cond_stag` is built lazily, so a checkpoint trained WITHOUT a stagnation
        term (lambda_stag = 0, or a Cp/Cl-only backbone such as _opt3_gamma.npz)
        silently yields a fresh untrained head whose output is ~constant at s_le.
        Check that the checkpoint actually carries these weights before using it.
        The stagnation point reported in the paper is not this scalar: it is the
        zero crossing of the tangential velocity, `stagnation_from_ut(...)`.
        """
        alpha_rad = alpha_deg * self.pi_over_180
        sin_a = tf.sin(alpha_rad)
        cos_a = tf.cos(alpha_rad)
        # v51-fix [2026-05-08] -- Re input z-scored via normalize_log_re
        # (raw tf.math.log(Re) ~ 12-14 saturated tanh activations).
        cond = tf.concat([shape_params, sin_a, cos_a,
                          normalize_log_re(Re)], axis=1)            # (B, shape_dim+3)
        if self.profile_enc is not None:
            profile = build_profile_from_cst(
                shape_params, n_surface_pts=self.n_surface_pts,
                n_bern=self.n_bern)
            cond = tf.concat(
                [cond, self.profile_enc(profile, training=training)], axis=1)
        raw = self.cond_stag(cond, training=training)
        delta = self.stag_le_half_width * tf.tanh(raw)
        s_stag = self.s_le + delta
        return tf.clip_by_value(s_stag, self.s_query_min, self.s_query_max)

    # --------------------------------------------------------
    # Residual-ADAF patch: Discrepancy head
    # --------------------------------------------------------
    def predict_discrepancy(self, shape_params, alpha_deg, Re,
                            training: bool = False) -> dict:
        """Single discrepancy head: ΔADAF_coeff + ΔBL + Δwake."""
        alpha_rad = alpha_deg * self.pi_over_180
        sin_a = tf.sin(alpha_rad)
        cos_a = tf.cos(alpha_rad)
        Re_in = tf.cast(Re, DTYPE)
        log_re = normalize_log_re(Re_in)
        cond = tf.concat([shape_params, sin_a, cos_a, log_re], axis=1)
        if self.profile_enc is not None:
            profile = build_profile_from_cst(
                shape_params, n_surface_pts=self.n_surface_pts,
                n_bern=self.n_bern)
            cond = tf.concat(
                [cond, self.profile_enc(profile, training=training)], axis=1)

        h = cond
        for layer in self.disc_hidden:
            h = layer(h, training=training)
        
        d_adaf = self.disc_adaf_out(h)      # (B, N_p)
        d_bl = self.disc_bl_out(h)          # (B, 6)
        d_wake = self.disc_wake_out(h)      # (B, 3)
        
        return {
            "delta_adaf_coeff": d_adaf,
            "delta_bl": d_bl,          # [theta_u, H_u, S_u, theta_l, H_l, S_l]
            "delta_wake": d_wake,      # [theta_w, disp_w, Ue_ratio_w]
        }

    # --------------------------------------------------------
    # Residual-ADAF patch: Field reconstruction via Δψ (v2 style)
    # --------------------------------------------------------
    def predict_field_delta_psi(self, x_query, y_query, base_features,
                                training: bool = False) -> tf.Tensor:
        """Predict Δψ at arbitrary query points.
        
        Args:
            x_query: (B, N_query)
            y_query: (B, N_query)
            base_features: (B, N_query, F) - base physics features
            
        Returns:
            delta_psi: (B, N_query) - stream function correction
        """
        # Concatenate query coordinates with base features
        query_feats = tf.concat([
            x_query[..., None], y_query[..., None], base_features
        ], axis=-1)
        
        delta_psi = self.field_decoder(query_feats, training=training)
        return delta_psi[..., 0]

    def compute_field_velocity_correction(self, x_query, y_query, delta_psi):
        """Compute velocity correction from Δψ using ∇⟂.
        
        u_corr = -∂Δψ/∂y, v_corr = ∂Δψ/∂x
        Uses central differences on the query grid.
        """
        # For now, return zero - in practice would use gradient tape or FD
        # This is a placeholder for the actual gradient computation
        batch_size = tf.shape(x_query)[0]
        n_query = tf.shape(x_query)[1]
        return tf.zeros((batch_size, n_query), dtype=DTYPE), \
               tf.zeros((batch_size, n_query), dtype=DTYPE)

    # --------------------------------------------------------
    # Residual-ADAF patch: CDCALC endpoint (from VortFoil_v2)
    # --------------------------------------------------------
    @staticmethod
    def cdcalc_endpoint(theta_w, disp_w, ue_ratio_w, chord=1.0):
        """XFOIL 6.996 incompressible CDCALC endpoint.
        
        Args:
            theta_w: momentum thickness at wake endpoint
            disp_w: displacement thickness at wake endpoint
            ue_ratio_w: Ue/Uinf at wake endpoint
            chord: chord length
            
        Returns:
            Cd: profile drag coefficient
        """
        H_w = disp_w / theta_w
        Cd_raw = 2.0 * theta_w * tf.pow(ue_ratio_w, (5.0 + H_w) / 2.0)
        Cd = Cd_raw / chord
        return Cd

    def compute_losses(self, shape_params, alpha_deg, Re,
                        training: bool = True,
                        return_details: bool = False,
                        target_Cp=None, target_Cl=None, target_Cd=None,
                        target_BL=None, target_wake=None) -> dict:
        base_geom = build_single_loop_contour_cst_tf(
            shape_params, n_surface_pts=self.n_surface_pts,
            n_bern=self.n_bern)

        gamma_out = self.predict_gamma(shape_params, alpha_deg, Re=Re,
                                          training=training)
        gamma_nodes = gamma_out["gamma_nodes"]
        gamma_panels = gamma_out["gamma_panels"]
        R_kutta = gamma_out["gamma_kutta_sum"]

        s_stag = self.predict_s_stag(shape_params, alpha_deg, Re, training=training)

        inv_out = compute_inviscid_surface_velocity(
            base_geom, gamma_nodes, alpha_deg)
        Ut = inv_out["Ut"]
        Un = inv_out["Un"]

        # Stagnation residual: Ut interpolated at s_stag must vanish
        Ut_stag = self._interp_panel_to_stag(Ut, s_stag)

        # ---- Discrepancy prediction ----
        disc = self.predict_discrepancy(shape_params, alpha_deg, Re, training=training)
        d_adaf = disc["delta_adaf_coeff"]
        d_bl = disc["delta_bl"]
        d_wake = disc["delta_wake"]

        # ---- Losses ----
        L_kutta = self.lambda_kutta * tf.reduce_mean(R_kutta ** 2)
        L_np = tf.reduce_mean(Un ** 2)
        L_stag = self.lambda_stag * tf.reduce_mean(Ut_stag ** 2)

        # Discrepancy losses (supervised if targets provided)
        L_disc = tf.constant(0.0, dtype=DTYPE)
        if target_BL is not None:
            L_disc += tf.reduce_mean((d_bl - target_BL) ** 2)
        if target_wake is not None:
            L_disc += tf.reduce_mean((d_wake - target_wake) ** 2)
        # ADAF coeff discrepancy is implicit through Cp/Cl loss

        # CDCALC endpoint for Cd (if BL/wake targets available)
        L_cd = tf.constant(0.0, dtype=DTYPE)
        if target_Cd is not None and target_wake is not None:
            theta_w = target_wake[:, 0:1]
            disp_w = target_wake[:, 1:2]
            ue_ratio_w = target_wake[:, 2:3]
            Cd_pred = self.cdcalc_endpoint(theta_w, disp_w, ue_ratio_w)
            L_cd = self.lambda_cd * tf.reduce_mean((Cd_pred - target_Cd) ** 2)

        # Sobolev regularization for Δψ (field smoothness)
        L_sobolev = tf.constant(0.0, dtype=DTYPE)
        # Note: actual Sobolev would need field query points; skipped for now

        L_total = L_kutta + L_np + L_stag + L_disc + L_cd + self.lambda_sobolev * L_sobolev

        # ---- Cl from Kutta–Joukowski ----
        ds = base_geom["ds"]
        Gamma = tf.reduce_sum(gamma_panels * ds, axis=1, keepdims=True)
        Cl = 2.0 * Gamma

        out = {
            "total": L_total,
            "loss_kutta_inv": L_kutta,
            "loss_np_inv": L_np,
            "loss_stag": L_stag,
            "loss_disc": L_disc,
            "loss_cd": L_cd,
            "loss_sobolev": L_sobolev,
            "Cl": Cl,
            "s_stag": s_stag,
            "Ut_stag": Ut_stag,
        }
        if return_details:
            out.update({
                "gamma_nodes": gamma_nodes,
                "gamma_panels": gamma_panels,
                "Ut_inv": Ut,
                "Un_inv": Un,
                "xc": base_geom["xc"],
                "yc": base_geom["yc"],
                "ds": ds,
                "disc": disc,
            })
        return out

    # --------------------------------------------------------
    # helpers
    # --------------------------------------------------------
    def _interp_panel_to_stag(self, Ut_panels, s_stag):
        """Linearly interpolate Ut (defined at panel midpoints) to s_stag."""
        N = tf.shape(Ut_panels)[1]
        s_grid = self.s_panels_1d                                # (N,)
        idx_hi = tf.searchsorted(s_grid, tf.reshape(s_stag, [-1]),
                                 side="left")
        idx_hi = tf.clip_by_value(idx_hi, 1, N - 1)
        idx_lo = idx_hi - 1
        x0 = tf.gather(s_grid, idx_lo)
        x1 = tf.gather(s_grid, idx_hi)
        y0 = tf.gather(Ut_panels, idx_lo[:, None], batch_dims=1)
        y1 = tf.gather(Ut_panels, idx_hi[:, None], batch_dims=1)
        s_stag_flat = tf.reshape(s_stag, [-1, 1])
        x0 = tf.reshape(x0, [-1, 1])
        x1 = tf.reshape(x1, [-1, 1])
        w = (s_stag_flat - x0) / tf.maximum(x1 - x0, 1e-8)
        return y0 * (1.0 - w) + y1 * w

    # --------------------------------------------------------
    # weight save / load
    # --------------------------------------------------------
    def trainable_variables_list(self):
        vs = (
            list(self.cond_gamma.trainable_variables)
            + list(self.cond_stag.trainable_variables)
            + list(self.disc_hidden[0].trainable_variables)
            + list(self.disc_hidden[1].trainable_variables)
            + list(self.disc_adaf_out.trainable_variables)
            + list(self.disc_bl_out.trainable_variables)
            + list(self.disc_wake_out.trainable_variables)
            + list(self.field_decoder.trainable_variables)
        )
        if self.profile_enc is not None:
            vs += list(self.profile_enc.trainable_variables)
        return vs

    def save_weights(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {f"v_{i:04d}": v.numpy()
             for i, v in enumerate(self.trainable_variables_list())}
        np.savez(str(path), **d)

    def load_weights(self, path, strict: bool = True):
        """Load weights with backward-compat zero-padding for Re-aware
        cond_gamma upgrade (18-dim → 19-dim conditioning).

        When a pre-Re checkpoint is loaded into the new 19-dim model,
        cond_gamma's first dense kernel has shape (18, hidden) on disk
        but (19, hidden) in memory.  We pad row 18 (the Re weight) with
        zeros so the network's output exactly reproduces legacy behavior
        until trained otherwise.
        """
        data = np.load(str(path))
        loaded = 0
        skipped = 0
        padded = 0
        for i, v in enumerate(self.trainable_variables_list()):
            key = f"v_{i:04d}"
            if key not in data:
                skipped += 1
                continue
            saved = data[key]
            if v.shape != saved.shape:
                # Detect 18→19 cond_dim upgrade on a dense kernel
                # (2-D, last dim equal, first dim 18 → 19).
                if (saved.ndim == 2 and v.shape.ndims == 2
                        and v.shape[0] == 19 and saved.shape[0] == 18
                        and v.shape[1] == saved.shape[1]):
                    padded_kernel = np.zeros(tuple(v.shape),
                                              dtype=saved.dtype)
                    padded_kernel[:18, :] = saved
                    v.assign(padded_kernel)
                    loaded += 1
                    padded += 1
                    continue
                # Detect cond_dim expansion via profile-encoder concat:
                # first kernel has shape (19+enc_dim, hidden) but saved is
                # (19, hidden) or (18, hidden) — pad the new tail rows
                # (profile features) with zeros.
                if (saved.ndim == 2 and v.shape.ndims == 2
                        and v.shape[0] > saved.shape[0]
                        and saved.shape[0] in (18, 19)
                        and v.shape[1] == saved.shape[1]):
                    padded_kernel = np.zeros(tuple(v.shape),
                                              dtype=saved.dtype)
                    padded_kernel[:saved.shape[0], :] = saved
                    v.assign(padded_kernel)
                    loaded += 1
                    padded += 1
                    continue
                if strict:
                    raise RuntimeError(
                        f"shape mismatch on var {i} ({v.name}): "
                        f"{v.shape} vs saved {saved.shape}")
                skipped += 1
                continue
            v.assign(saved)
            loaded += 1
        return {"loaded": loaded, "skipped": skipped, "padded": padded}

    def num_trainable_parameters(self) -> int:
        return int(sum(np.prod(v.shape) for v in self.trainable_variables_list()))


# ============================================================
# Standalone build helper (for validation scripts)
# ============================================================
def build_op1(n_surface_pts: int = 61, **kwargs) -> Op1Inviscid:
    return Op1Inviscid(n_surface_pts=n_surface_pts, **kwargs)


def stagnation_from_ut(Ut, s_le: float = 0.5):
    """Stagnation point as the zero crossing of the tangential velocity.

    This is the definition used in the paper: no auxiliary head, no search
    heuristic. `Ut` is (B, N_p) on the single closed surface loop. Returns the
    crossing nearest the mesh leading edge, in arc-length fraction, linearly
    interpolated between the bracketing panels, plus the number of sign changes
    per case (1 in every held-out case we measured -- anything else means the
    sheet is not well behaved for that geometry).

    Returns (s_stag, n_crossings) as numpy arrays of shape (B,).
    """
    Ut = np.asarray(Ut)
    n_p = Ut.shape[1]
    sf = (np.arange(n_p) + 0.5) / n_p
    sgn = np.sign(Ut)
    cross = np.diff(sgn, axis=1) != 0
    s_out = np.full(len(Ut), np.nan)
    n_out = cross.sum(axis=1)
    for b in range(len(Ut)):
        idx = np.flatnonzero(cross[b])
        if idx.size == 0:
            continue
        k = idx[np.argmin(np.abs(sf[idx] - s_le))]
        u0, u1 = Ut[b, k], Ut[b, k + 1]
        frac = 0.0 if u1 == u0 else -u0 / (u1 - u0)
        s_out[b] = sf[k] + frac * (sf[k + 1] - sf[k])
    return s_out, n_out


    # --------------------------------------------------------
    # Loss / forward
    # --------------------------------------------------------
