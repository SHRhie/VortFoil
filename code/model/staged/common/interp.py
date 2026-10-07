# -*- coding: utf-8 -*-
"""
Numerical primitives:  1-D interpolation and finite differences.

Extracted verbatim from steady51.py.
"""
from __future__ import annotations

import tensorflow as tf

DTYPE = tf.float32


# ============================================================
# Batched 1-D linear interpolation on a common monotone grid
# ============================================================
def interp1d_common_grid(grid_1d, values, query):
    """
    Linear interpolation on a shared 1-D monotone grid.

    Parameters
    ----------
    grid_1d : (N,)        monotone increasing
    values  : (B, N)
    query   : (B, M) or (B, 1)

    Returns
    -------
    (B, M)
    """
    grid_1d = tf.convert_to_tensor(grid_1d, dtype=DTYPE)
    values = tf.convert_to_tensor(values, dtype=DTYPE)
    query = tf.convert_to_tensor(query, dtype=DTYPE)

    query_shape = tf.shape(query)
    query_flat = tf.reshape(query, [-1])
    N = tf.shape(grid_1d)[0]

    idx_hi = tf.searchsorted(grid_1d, query_flat, side="left")
    idx_hi = tf.clip_by_value(idx_hi, 1, N - 1)
    idx_lo = idx_hi - 1

    x0_flat = tf.gather(grid_1d, idx_lo)
    x1_flat = tf.gather(grid_1d, idx_hi)

    idx_lo_2d = tf.reshape(idx_lo, query_shape)
    idx_hi_2d = tf.reshape(idx_hi, query_shape)

    y0 = tf.gather(values, idx_lo_2d, batch_dims=1)
    y1 = tf.gather(values, idx_hi_2d, batch_dims=1)

    x0 = tf.reshape(x0_flat, query_shape)
    x1 = tf.reshape(x1_flat, query_shape)

    w = (query - x0) / tf.maximum(x1 - x0, 1e-8)
    return y0 * (1.0 - w) + y1 * w


# ============================================================
# Central differences
# ============================================================
def central_diff_1d(f, dx):
    """
    2nd-order central difference on a uniform grid.
    Boundaries use 2nd-order one-sided stencils.

    f  : (B, N)
    dx : scalar
    """
    left = (-3.0 * f[:, :1] + 4.0 * f[:, 1:2] - f[:, 2:3]) / (2.0 * dx)
    mid = (f[:, 2:] - f[:, :-2]) / (2.0 * dx)
    right = (3.0 * f[:, -1:] - 4.0 * f[:, -2:-1] + f[:, -3:-2]) / (2.0 * dx)
    return tf.concat([left, mid, right], axis=1)


def central_diff_1d_nonuniform(f, ds):
    """
    Non-uniform central difference: df/ds at node positions.

    Parameters
    ----------
    f  : (B, N)
    ds : (B, N-1)   panel arc-lengths between consecutive nodes

    Returns
    -------
    df_ds : (B, N)
    """
    ds_L = ds[:, :-1]
    ds_R = ds[:, 1:]
    f_L = f[:, :-2]
    f_C = f[:, 1:-1]
    f_R = f[:, 2:]
    denom = ds_L * ds_R * (ds_L + ds_R) + 1e-30
    mid = (ds_L ** 2 * f_R - ds_R ** 2 * f_L
           + (ds_R ** 2 - ds_L ** 2) * f_C) / denom

    left = (f[:, 1:2] - f[:, :1]) / (ds[:, :1] + 1e-30)
    right = (f[:, -1:] - f[:, -2:-1]) / (ds[:, -1:] + 1e-30)

    return tf.concat([left, mid, right], axis=1)
