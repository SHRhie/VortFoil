# -*- coding: utf-8 -*-
"""
Backbone neural-network building blocks for the staged operator pipeline.

All classes are pure NN architectures (no physics knowledge — they only map
tensors to tensors).  Physics happens in staged/common/{panel,bl_closures,...}.

Extracted verbatim from steady51.py.  Numerical equivalence with steady51
is enforced in staged/common/_smoke_test_networks.py.

Used by:
    Op1 (op1_inviscid.py)  →  ConditionNet × 2 (cond_gamma, cond_stag)
    Op2 (op2_bl.py)        →  BLOperatorNet  (UeEncoder + ResidualBlock trunk)
    Op3 (op3_correction.py) →  ConditionNet (cond_gamma_bl) [+ source net TBD]
"""
from __future__ import annotations

import tensorflow as tf

DTYPE = tf.float32


# ============================================================
# ConditionNet — feed-forward Dense stack with optional skip connections
# ============================================================
class ConditionNet(tf.keras.Model):
    """
    Maps a small condition vector (shape, alpha, optional log_Re) → output.

    Architecture:
        Dense(hidden)  → input projection
        depth × Dense(hidden, activation)   [optional residual + LayerNorm]
        Dense(out_dim)  → final, scaled init for stable start

    `last_layer_scale` lets us start with near-zero output (e.g. for
    γ_bl correction where we want the network to begin at zero).
    """

    def __init__(self, out_dim: int, hidden: int = 128, depth: int = 3,
                 activation: str = "tanh", last_layer_scale: float = 1.0,
                 use_skip: bool = False):
        super().__init__()
        self.use_skip = use_skip
        self.hidden = hidden

        self.input_proj = tf.keras.layers.Dense(hidden, activation=None)

        self.blocks = []
        for _ in range(depth):
            self.blocks.append(tf.keras.layers.Dense(hidden, activation=activation))

        if use_skip:
            self.layer_norms = [
                tf.keras.layers.LayerNormalization(epsilon=1e-6)
                for _ in range(depth)
            ]

        self.output_layer = tf.keras.layers.Dense(
            out_dim, activation=None,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=0.02 * last_layer_scale
            ),
            bias_initializer="zeros",
        )

    def call(self, x, training=False):
        h = self.input_proj(x)
        if self.use_skip:
            for block, ln in zip(self.blocks, self.layer_norms):
                residual = h
                h = ln(h, training=training)
                h = block(h)
                h = h + residual
        else:
            for block in self.blocks:
                h = block(h)
        return self.output_layer(h)


# ============================================================
# ResidualBlock — pre-LayerNorm Dense block (used by BL trunks)
# ============================================================
class ResidualBlock(tf.keras.layers.Layer):
    """Pre-LayerNorm residual block: LN → Dense(expand) → GELU → Dense(project) → Add."""

    def __init__(self, dim: int, expand_ratio: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.fc1 = tf.keras.layers.Dense(dim * expand_ratio, activation="gelu")
        self.fc2 = tf.keras.layers.Dense(
            dim, activation=None,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        )

    def call(self, x, training=False):
        h = self.ln(x, training=training)
        h = self.fc1(h)
        h = self.fc2(h)
        return x + h


# ============================================================
# UeEncoder — compact encoder for ue_inv profile
# ============================================================
class UeEncoder(tf.keras.Model):
    """
    Maps ue_inv(ξ) profile (Nside+1 points) to a compact feature vector.

    Architecture: Dense(64, GELU) → Dense(ue_feat_dim, tanh)
    The tanh output keeps features bounded — aids trunk stability.
    """

    def __init__(self, ue_feat_dim: int = 32):
        super().__init__()
        self.enc = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="gelu"),
            tf.keras.layers.Dense(ue_feat_dim, activation="tanh"),
        ], name="ue_encoder")

    def call(self, ue_inv, training=False):
        return self.enc(ue_inv, training=training)


# ============================================================
# BLOperatorNet — single trunk + 4 ADAF-weight heads (θ_u, θ_l, δ_u, δ_l)
# ============================================================
class BLOperatorNet(tf.keras.Model):
    """
    Predicts ADAF weights for θ_turb, H_turb (shape factor), and
    S_turb (=√Cτ for lag), upper / lower.

    v51-fix [2026-05-10] D-revert — ξ_tr heads REMOVED.
    Earlier attempt to learn transition position via ξ_tr led to two
    degenerate trivial solutions (ξ_tr → 1 fully laminar, ξ_tr → 0
    fully turbulent).  Reverted to physics-based n-factor (Drela CL2)
    for transition determination — operator learns BL state, transition
    emerges from the state.

    Input  : cond(d_in) + ue_feat_u + ue_feat_l
    Trunk  : Dense → N×ResidualBlock → LN → h
    Heads  : 6 × Dense_head_dim → Dense_bl_out_dim
              theta_{u,l}, H_{u,l}, ctau_{u,l}

    Output : dict with keys theta_*, H_*, ctau_* — each (B, bl_out_dim)
    """

    def __init__(self, trunk_dim: int = 256, trunk_depth: int = 4,
                 head_dim: int = 128, head_depth: int = 2,
                 bl_out_dim: int = 121, last_layer_scale: float = 0.1,
                 dropout_rate: float = 0.0, ue_feat_dim: int = 32):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.ue_encoder = UeEncoder(ue_feat_dim=ue_feat_dim)

        self.input_proj = tf.keras.layers.Dense(trunk_dim, activation=None)
        self.trunk_blocks = [
            ResidualBlock(trunk_dim, expand_ratio=2)
            for _ in range(trunk_depth)
        ]
        self.trunk_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.trunk_dropout = (
            tf.keras.layers.Dropout(dropout_rate) if dropout_rate > 0 else None
        )

        # v51-fix [2026-05-08] OPTION B — added ctau heads for 3-eq turb
        # BL formulation (S = √Cτ trainable via ADAF, drives Cd lag-
        # entrainment in xfoil_closures.cd_turb_xfoil).
        self.heads = {}
        for name in ["theta_u", "theta_l", "H_u", "H_l",
                     "ctau_u", "ctau_l"]:
            layers = []
            for _ in range(head_depth):
                layers.append(tf.keras.layers.Dense(head_dim, activation="gelu"))
                if dropout_rate > 0:
                    layers.append(tf.keras.layers.Dropout(dropout_rate))
            layers.append(tf.keras.layers.Dense(
                bl_out_dim, activation=None,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                    stddev=0.02 * last_layer_scale),
                bias_initializer="zeros",
            ))
            self.heads[name] = tf.keras.Sequential(layers, name=f"head_{name}")
        # v51-fix [2026-05-10] D-revert — ξ_tr heads removed.
        # Transition is determined by physics-derived n-factor (Drela CL2)
        # in solve_laminar_bl_full_2eq, not by a learned head.
        self._head_list = list(self.heads.values())

    def call(self, x, ue_inv_u, ue_inv_l, training=False):
        ue_feat_u = self.ue_encoder(ue_inv_u, training=training)
        ue_feat_l = self.ue_encoder(ue_inv_l, training=training)
        x_cat = tf.concat([x, ue_feat_u, ue_feat_l], axis=1)

        h = self.input_proj(x_cat)
        for block in self.trunk_blocks:
            h = block(h, training=training)
        h = self.trunk_ln(h, training=training)
        if self.trunk_dropout is not None:
            h = self.trunk_dropout(h, training=training)

        return {k: self.heads[k](h, training=training) for k in self.heads}
