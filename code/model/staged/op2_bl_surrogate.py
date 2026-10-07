# -*- coding: utf-8 -*-
"""Op2 BL Surrogate — pure supervised, predicts (θ, δ*) per branch.

Architecture (lightweight, OP1-mirror):
  cond (CST, sin α, cos α, log Re) → 4 × BranchTrunkADAFHead
    → (W per head, init_1 discarded)
  ADAF integration with init₁ = 0  →  raw distribution
    g₁(ξ) = 0 + ∫₀^ξ [a₀ + Σ harmonics] dξ  ← starts at 0 at ξ=0
  Output: scale factor only (no softplus/tanh)
    θ(ξ)  = θ_raw · θ_scale
    δ*(ξ) = δ_raw · δ_scale
  Derived:  H = δ*/θ  (computed downstream, NOT predicted)

Physical priors baked in:
  - θ(ξ=0) = 0  exactly (LE stagnation, BL has zero momentum thickness)
  - δ*(ξ=0) = 0 exactly (same reason)
  - Both grow downstream via ADAF antiderivative integration

Loss (scaled MSE):
  L_θ = MSE((θ_pred - θ_target) / θ_scale)
  L_δ = MSE((δ*_pred - δ*_target) / δ_scale)
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf

from staged.common import (
    BatchedADAFReusable,
    normalize_log_re,
)
from staged.common.branch_trunk import BranchTrunkADAFHead
from staged.common.profile_encoder import (
    ProfileEncoder, build_profile_from_cst,
)

DTYPE = tf.float32
CST_N_BERN_DEFAULT = 8

# Scale factors — surface (θ, δ*) + wake (θ_wake, δ_wake, ue_wake).
# std(θ) ≈ 5e-3, std(δ*) ≈ 1e-2 for surface; wake θ slightly bigger.
THETA_SCALE = 5.0e-3
DELTA_SCALE = 1.0e-2
THETA_WAKE_SCALE = 7.0e-3
DELTA_WAKE_SCALE = 1.2e-2
UE_WAKE_SCALE = 0.05      # ue_wake ∈ [~0.9, ~1.0], variation ~ 0.1, std ~ 0.05

# Wake-ξ grid points (matches dataset's xi_wake_grid: 25 uniform points)
N_WAKE_DEFAULT = 25


class Op2BLSurrogate(tf.keras.Model):
    """Pure supervised BL surrogate — predicts θ and δ* per branch."""

    def __init__(
        self,
        n_surface_pts: int = 121,
        n_wake: int = N_WAKE_DEFAULT,
        n_bern: int = CST_N_BERN_DEFAULT,
        N_m: int = 16,
        N_m_wake: int = 8,
        branch_dim: int = 128,
        branch_depth: int = 3,
        trunk_depth: int = 2,
        last_layer_scale: float = 0.02,
        theta_scale: float = THETA_SCALE,
        delta_scale: float = DELTA_SCALE,
        theta_wake_scale: float = THETA_WAKE_SCALE,
        delta_wake_scale: float = DELTA_WAKE_SCALE,
        ue_wake_scale:    float = UE_WAKE_SCALE,
        enable_wake: bool = True,
        use_profile_enc: bool = False,
        profile_enc_hidden: int = 64,
        profile_enc_kernel: int = 5,
        profile_enc_depth: int = 4,
        name: str = "op2_bl_surrogate",
    ):
        super().__init__(name=name)
        self.use_profile_enc = bool(use_profile_enc)
        self.n_surface_pts = int(n_surface_pts)
        self.n_wake = int(n_wake)
        self.n_bern = int(n_bern)
        self.shape_dim = 2 * self.n_bern
        self.N_p = int(n_surface_pts)
        self.N_m = int(N_m)
        self.N_m_wake = int(N_m_wake)
        self.theta_scale = float(theta_scale)
        self.delta_scale = float(delta_scale)
        self.theta_wake_scale = float(theta_wake_scale)
        self.delta_wake_scale = float(delta_wake_scale)
        self.ue_wake_scale = float(ue_wake_scale)
        self.enable_wake = bool(enable_wake)
        self.pi_over_180 = tf.constant(np.pi / 180.0, dtype=DTYPE)

        # ADAF decoder on ξ ∈ [0, 1] — shared across all heads.
        xi_nodes = np.linspace(0.0, 1.0, self.N_p, dtype=np.float32)
        self.adaf = BatchedADAFReusable(
            xi_nodes, N_p=self.N_p, N_m=self.N_m,
            gamma=1.0, name="bl_adaf",
        )

        # 4 independent Branch-Trunk heads — θ and δ* per branch.
        def _mk_head(tag):
            return BranchTrunkADAFHead(
                N_p=self.N_p,
                branch_dim=branch_dim,
                branch_depth=branch_depth,
                trunk_depth=trunk_depth,
                activation="gelu",
                last_layer_scale=last_layer_scale,
                name=f"bt_{tag}",
            )
        self.head_theta_u = _mk_head("theta_u")
        self.head_theta_l = _mk_head("theta_l")
        self.head_delta_u = _mk_head("delta_u")
        self.head_delta_l = _mk_head("delta_l")

        # ---- Wake heads (3 heads, single combined wake stream) ----
        if self.enable_wake:
            # Separate ADAF for wake grid (25 points)
            xi_wake_nodes = np.linspace(0.0, 1.0, self.n_wake,
                                           dtype=np.float32)
            self.adaf_wake = BatchedADAFReusable(
                xi_wake_nodes, N_p=self.n_wake, N_m=self.N_m_wake,
                gamma=1.0, name="bl_adaf_wake",
            )
            def _mk_wake_head(tag):
                return BranchTrunkADAFHead(
                    N_p=self.n_wake,
                    branch_dim=branch_dim,
                    branch_depth=branch_depth,
                    trunk_depth=trunk_depth,
                    activation="gelu",
                    last_layer_scale=last_layer_scale,
                    name=f"bt_wake_{tag}",
                )
            self.head_theta_wake = _mk_wake_head("theta")
            self.head_delta_wake = _mk_wake_head("delta")
            self.head_ue_wake    = _mk_wake_head("ue")

        # 1D causal conv profile encoder — captures upstream→downstream
        # geometry detail along chord (LE→TE). Causal padding respects the
        # physical direction of BL propagation.
        if self.use_profile_enc:
            self.profile_enc = ProfileEncoder(
                n_pts=self.n_surface_pts,
                hidden=profile_enc_hidden,
                kernel=profile_enc_kernel,
                depth=profile_enc_depth,
                pool="mean_plus_te",
                name="op2_profile_enc",
            )
        else:
            self.profile_enc = None

    # ---------------------------------------------------------------
    def _cond(self, shape_params, alpha_deg, Re, training: bool = False):
        """Shared conditioning vector.  19-dim base (CST + sin/cos α + log Re)
        optionally extended with profile-conv features (2·enc_hidden)."""
        alpha_rad = alpha_deg * self.pi_over_180
        sin_a = tf.sin(alpha_rad)
        cos_a = tf.cos(alpha_rad)
        log_re = normalize_log_re(tf.cast(Re, DTYPE))
        cond = tf.concat([shape_params, sin_a, cos_a, log_re], axis=1)
        if self.profile_enc is not None:
            profile = build_profile_from_cst(
                shape_params, n_surface_pts=self.n_surface_pts,
                n_bern=self.n_bern)
            cond = tf.concat(
                [cond, self.profile_enc(profile, training=training)], axis=1)
        return cond

    def _decode(self, head, cond, training):
        """ADAF g₁ integration with NN-learned init₁ (Re-aware Hiemenz prior).

        Head outputs both W (per-panel ADAF coefficients) and init_1
        (LE value of the antiderivative).  init_1 lets the network learn
        the small but nonzero LE value of θ / δ* that depends on Re and
        local pressure gradient (Falkner-Skan β=1 / Hiemenz flow).
        """
        W, init1 = head(cond, training=training)
        a0, a_n, b_n = self.adaf.coeffs(W)
        return self.adaf.g1_from_coeffs(a0, a_n, b_n, init1)

    def _decode_wake(self, head, cond, init1, training):
        """ADAF decode for wake distribution, with externally-provided
        init₁ (physical TE-combined initial condition)."""
        W, _ = head(cond, training=training)   # discard head's init1
        a0, a_n, b_n = self.adaf_wake.coeffs(W)
        return self.adaf_wake.g1_from_coeffs(a0, a_n, b_n, init1)

    def predict_bl(self, shape_params, alpha_deg, Re,
                    training: bool = False) -> dict:
        cond = self._cond(shape_params, alpha_deg, Re, training=training)

        theta_u_raw = self._decode(self.head_theta_u, cond, training)
        theta_l_raw = self._decode(self.head_theta_l, cond, training)
        delta_u_raw = self._decode(self.head_delta_u, cond, training)
        delta_l_raw = self._decode(self.head_delta_l, cond, training)

        # Output = raw · scale  for θ, δ* (no nonlinearity).
        theta_u = theta_u_raw * self.theta_scale
        theta_l = theta_l_raw * self.theta_scale
        delta_u = delta_u_raw * self.delta_scale
        delta_l = delta_l_raw * self.delta_scale

        eps = tf.constant(1.0e-8, dtype=DTYPE)
        H_u = delta_u / tf.maximum(theta_u, eps)
        H_l = delta_l / tf.maximum(theta_l, eps)

        result = {
            "theta_u": theta_u, "theta_l": theta_l,
            "delta_u": delta_u, "delta_l": delta_l,
            "H_u":     H_u,     "H_l":     H_l,
        }

        if self.enable_wake:
            # Wake initial conditions = surface TE combined (structural).
            #   θ_wake(ξ_w=0) = θ_u_TE + θ_l_TE  (BL momentum merge)
            #   δ*_wake(ξ_w=0) = δ*_u_TE + δ*_l_TE
            # ADAF integrates from these physical anchors.  NN learns the
            # downstream wake evolution (decay of H toward 1, ue recovery).
            theta_TE_combined = theta_u[:, -1:] + theta_l[:, -1:]
            delta_TE_combined = delta_u[:, -1:] + delta_l[:, -1:]

            # ue_wake init: Kutta gives common ue at TE — approximated by
            # NN here (head_ue_wake's init1 from network output).  Could
            # be tied to OP1's panel Ut at TE but kept independent for
            # operator independence.
            W_ue, init_ue = self.head_ue_wake(cond, training=training)

            θ_wake_raw = self._decode_wake(
                self.head_theta_wake, cond, theta_TE_combined, training)
            δ_wake_raw = self._decode_wake(
                self.head_delta_wake, cond, delta_TE_combined, training)
            # ue_wake uses NN init (variable across cases)
            a0_ue, an_ue, bn_ue = self.adaf_wake.coeffs(W_ue)
            ue_wake_raw = self.adaf_wake.g1_from_coeffs(
                a0_ue, an_ue, bn_ue, init_ue)

            # Scale: θ_wake and δ_wake have physical anchors at ξ_w=0;
            # the NN-learned deviation is scaled separately.  Here we
            # treat the full ADAF output as the prediction (anchor +
            # learned).  Scale factor handled via TE values (in physical
            # units) and learned deviation in same units.
            theta_wake = θ_wake_raw   # already in physical units (init1 is)
            delta_wake = δ_wake_raw
            ue_wake = ue_wake_raw * self.ue_wake_scale + 1.0  # center at 1.0

            result.update({
                "theta_wake": theta_wake,
                "delta_wake": delta_wake,
                "ue_wake":    ue_wake,
            })
        return result

    # ---------------------------------------------------------------
    def trainable_variables_list(self):
        vars_ = (list(self.head_theta_u.trainable_variables)
                  + list(self.head_theta_l.trainable_variables)
                  + list(self.head_delta_u.trainable_variables)
                  + list(self.head_delta_l.trainable_variables))
        if self.enable_wake:
            vars_ += (list(self.head_theta_wake.trainable_variables)
                       + list(self.head_delta_wake.trainable_variables)
                       + list(self.head_ue_wake.trainable_variables))
        if self.profile_enc is not None:
            vars_ += list(self.profile_enc.trainable_variables)
        return vars_

    def num_trainable_parameters(self) -> int:
        return int(sum(int(np.prod(v.shape))
                        for v in self.trainable_variables_list()))

    def save_weights(self, path):
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        vars_ = self.trainable_variables_list()
        d = {f"v_{i:04d}": v.numpy() for i, v in enumerate(vars_)}
        np.savez(str(path), **d)

    def load_weights(self, path, strict: bool = True) -> dict:
        data = np.load(str(path))
        loaded = 0; skipped = 0; padded = 0
        for i, v in enumerate(self.trainable_variables_list()):
            key = f"v_{i:04d}"
            if key not in data:
                skipped += 1; continue
            saved = data[key]
            if v.shape != saved.shape:
                # Cond-dim expansion (19 → 19 + enc_dim) when profile_enc
                # is added.  Pad the new tail rows with zeros so the model
                # initially reproduces the no-encoder behaviour.
                if (saved.ndim == 2 and v.shape.ndims == 2
                        and v.shape[0] > saved.shape[0]
                        and saved.shape[0] in (18, 19)
                        and v.shape[1] == saved.shape[1]):
                    new = np.zeros(tuple(v.shape), dtype=saved.dtype)
                    new[:saved.shape[0], :] = saved
                    v.assign(new)
                    loaded += 1; padded += 1
                    continue
                if strict:
                    raise RuntimeError(
                        f"shape mismatch v_{i}: {v.shape} vs {saved.shape}")
                skipped += 1; continue
            v.assign(saved)
            loaded += 1
        return {"loaded": loaded, "skipped": skipped, "padded": padded}


def build_op2_surrogate(**kw) -> Op2BLSurrogate:
    return Op2BLSurrogate(**kw)
