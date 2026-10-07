# -*- coding: utf-8 -*-
"""
Input-scaling helpers for the staged operator pipeline.

All ConditionNet / BLOperatorNet inputs that come from physical units
(Re, etc.) go through these helpers so the network sees roughly z-scored
values regardless of the underlying physical scale.

Single source of truth: change the constants here and every stage
(Op1, Op2, Op3, Stage 4) sees the same normalization.
"""
from __future__ import annotations

import tensorflow as tf

DTYPE = tf.float32


# ============================================================
# v51-fix [2026-05-08] — Re input normalization
#
# Training Re distribution (datasets.sample_random_shapes):
#     Re_range = (5e4, 2e6)
#     log10(Re) ~ U(4.7, 6.3)
#     mean   = 5.5
#     std_U  ≈ 0.462    (uniform stdev = (b-a)/√12)
#
# We use std = 0.5 below — slightly larger than the true uniform stdev,
# which keeps the resulting normalized value within roughly [-1.6, +1.6]
# while leaving small headroom outside the training range.
#
# Why this matters:
# Previous code passed `tf.math.log(Re)` directly, ranging [11.5, 14.5]
# in natural log — a 10–14× larger scale than every other network input
# (shape ∈ [0, 0.6], alpha_rad ∈ [-0.26, 0.26]).  ConditionNet uses tanh
# activations which saturate hard for |x| > 3, so the Re input was
# essentially always saturated and contributed almost no gradient
# signal.  This is the suspected cause of:
#   - Op1.cond_stag's weak Re sensitivity (s_stag barely varies with Re)
#   - Op2 BL state having uniform H_L1 ≈ 0.8 across iterations
#     regardless of ADAF capacity, lag-entrainment, or refinement
#     collocation — none of those changes can override a saturated
#     input feature.
#
# Affected sites (replaced with `normalize_log_re(Re)`):
#   - op1_inviscid.py    cond_stag input
#   - op2_bl.py          cond_bl  (BLOperatorNet) input
#   - op3_correction.py  cond_gamma_bl input
#   - stage4_picard.py   cond_bl in the inner Op2 forward
#
# All Op1 → Op2 → Op3 → Stage 4 weights must be retrained from scratch
# after this change — the input distribution to every conditioning net
# has shifted by ~12 units, so old weights are meaningless.
# ============================================================

LOG10_RE_MEAN = 5.5
LOG10_RE_STD = 0.5

# precomputed constant — log_e(10), used for log -> log10 conversion
_LN10 = tf.constant(2.302585092994046, dtype=DTYPE)


def normalize_log_re(Re):
    """
    Map raw Re (Pa·s units) to a near-zero-mean, ~unit-std scalar.

    Returns
    -------
    (log10(Re) - LOG10_RE_MEAN) / LOG10_RE_STD

    Safe against Re < 1 (clamped to 1.0 first).
    """
    log10_re = tf.math.log(tf.maximum(Re, 1.0)) / _LN10
    return (log10_re - LOG10_RE_MEAN) / LOG10_RE_STD
