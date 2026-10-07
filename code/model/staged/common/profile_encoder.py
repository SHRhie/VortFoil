# -*- coding: utf-8 -*-
"""ProfileEncoder — 1D causal dilated conv on airfoil (x, y) profile.

Encodes airfoil shape from raw upper/lower (x, y) coordinates via stacked
dilated causal 1D convolutions, with LE→TE chord direction. Causal padding
respects the physical direction of BL propagation: state at chord position
x depends on upstream geometry [0, x].

Input  : (B, P, 4) — channels (x_u, y_u_scaled, x_l, y_l_scaled)
Output : (B, hidden) global feature  (default pool="mean")
         or (B, P, hidden) per-point features  (return_sequence=True)
"""
from __future__ import annotations
import tensorflow as tf

DTYPE = tf.float32


class ProfileEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        n_pts: int = 121,
        hidden: int = 64,
        kernel: int = 5,
        depth: int = 4,
        dilation_base: int = 2,
        pool: str = "mean_plus_te",
        name: str = "profile_enc",
    ):
        super().__init__(name=name)
        self.n_pts = int(n_pts)
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.pool = pool

        self.proj_in = tf.keras.layers.Dense(
            hidden, activation=None, name="proj_in")
        self.convs = []
        for i in range(depth):
            self.convs.append(tf.keras.layers.Conv1D(
                filters=hidden, kernel_size=kernel,
                padding="causal",
                dilation_rate=dilation_base ** i,
                activation="gelu",
                name=f"conv{i}_d{dilation_base**i}",
            ))

    def output_dim(self) -> int:
        if self.pool == "mean_plus_te":
            return 2 * self.hidden
        return self.hidden

    def call(self, profile, training: bool = False,
              return_sequence: bool = False):
        # profile: (B, P, C)
        h = self.proj_in(profile)
        for conv in self.convs:
            h = h + conv(h, training=training)   # residual
        if return_sequence:
            return h
        if self.pool == "mean":
            return tf.reduce_mean(h, axis=1)
        if self.pool == "max":
            return tf.reduce_max(h, axis=1)
        # "mean_plus_te": global avg + TE-point (last) features
        avg = tf.reduce_mean(h, axis=1)
        te = h[:, -1, :]
        return tf.concat([avg, te], axis=1)


def build_profile_from_cst(
    shape_params,
    n_surface_pts: int = 121,
    n_bern: int = 8,
    y_scale: float = 10.0,
) -> tf.Tensor:
    """Compute (x_u, y_u, x_l, y_l) at cosine-spaced points from CST coeffs.

    y is scaled by `y_scale` (default 10) since y ∈ [-0.1, 0.1] for typical
    airfoils — brings y into ~[-1, 1] for stable conv training.

    Returns
    -------
    profile : (B, P, 4)  channels: x_u, y_u*scale, x_l, y_l*scale
    """
    from staged.common.geometry import cst_geometry_batch_tf
    xu, yu, xl, yl = cst_geometry_batch_tf(
        shape_params, n_surface_pts=n_surface_pts, n_bern=n_bern)
    return tf.stack([xu, yu * y_scale, xl, yl * y_scale], axis=-1)
