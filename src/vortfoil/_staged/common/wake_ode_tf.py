# -*- coding: utf-8 -*-
"""TF (batched, differentiable) wake-BL ODE integrator.

Forward Euler integration of Drela's wake momentum equation from TE state
(θ_TE, δ*_TE, ue_TE) to far-wake.  Identical numerics to
`staged.scripts.wake_ode_cd.integrate_wake_ode` but operates on TF tensors
of shape (B,) so the entire batched computation can be JIT-compiled with
the rest of the OP2 forward pass and is fully differentiable for Cd-loss
training.

ODE:
    dθ/dx = -(H + 2) · θ / u_e · du_e/dx              (no Cf)
    H(x)  = 1 + (H_TE - 1) · exp(-x · k_H)            (decay to 1)
    u_e(x)= u_e_TE + (u_e_inf - u_e_TE) · (x / L)     (linear recovery)

Cd at far-wake (H ≈ 1):
    Cd = 2 · θ_far · u_e_far ^ ((H_far + 5) / 2)
"""
from __future__ import annotations
import tensorflow as tf

DTYPE = tf.float32

# fitted_poly2 coefficients -- deg-2 ridge on (log theta_TE, H_TE, ue_TE) -> log10 Cd,
# fit on 325,370 rows of staged/data/xfoil_nf_scale_merged_clean.npz (train split only).
# Term order: 1, f0, f1, f2, f0f0, f0f1, f0f2, f1f1, f1f2, f2f2.
_POLY2_MU = tf.constant([0.0, -4.7113519736264049, 3.807192249037616, 0.92140549246766656,
                         22.429542839095042, -17.578735319443268, -4.3247078874306313,
                         25.94551997164406, 3.6590099481620362, 0.85485808421894594], DTYPE)
_POLY2_SD = tf.constant([1.0, 0.48239550132866726, 3.3839041284456242, 0.076615942657247318,
                         4.4624471531167114, 17.599720141017237, 0.43523387129337082,
                         671.78195589493419, 4.029415104563042, 0.14477139267621314], DTYPE)
_POLY2_W  = tf.constant([-1.8228202673594334, 0.096716363239070652, 0.059135609886142221,
                         0.12075388550697497, -0.072562908512465052, 0.0091874264215751917,
                         0.033358912547668339, -0.02722121314899658, 0.0098493711451736604,
                         0.040567130687654676], DTYPE)


def wake_ode_cd_tf(theta_TE, delta_TE, ue_TE,
                    n_steps: int = 100,
                    L_wake: float = 1.0,
                    k_H: float = 3.5,
                    ue_inf: float = 1.0,
                    closure: str = "fitted_poly2"):
    """Batched wake-BL ODE → Cd.

    Parameters
    ----------
    theta_TE, delta_TE, ue_TE : tensors of shape (B,) — wake-combined
                                  TE state.  θ_TE = θ_u_TE + θ_l_TE,
                                  δ_TE = δ*_u_TE + δ*_l_TE.
    n_steps : Euler integration steps along the wake.
    L_wake  : wake length (chord units).
    k_H     : H decay rate.
    ue_inf  : far-field velocity ratio.
    closure : "fitted_poly2" (default), "squire_young", or "legacy_ode".

    DEFECT AND FIX [2026-09-18]
    ---------------------------
    "legacy_ode" is the behaviour this module shipped with, kept so that sealed
    artifacts stay reproducible.  It has two measured problems:

    1. The Squire-Young exponent never applies.  ue_far is set to ue_inf = 1.0,
       so `ue_far ** ((H_far + 5) / 2)` is identically 1 and the effective
       formula is `Cd = 2 * theta_far` -- the docstring above describes an
       exponent the code does not use.
    2. The transported quantity uses the momentum-equation coefficient (H + 2)
       where Squire-Young uses (H + 5) / 2.  The difference, (H - 1) / 2, grows
       with H, so theta is over-decayed exactly where the wake is thickest.

    MEASURED on a 1,930-row separated-flow set (H_TE up to ~15), same predicted
    BL state fed to each closure, Cd R^2:

                                all      H<6    6..10    H>=10
        legacy_ode           0.6849   0.8281   0.4327  -0.2764
        squire_young         0.8452   0.8281   0.7074   0.5004

    The legacy closure is worse than predicting the mean in deep separation.

    "squire_young" is the textbook relation with no free parameters:

        Cd = 2 * theta_TE * ue_TE ** ((H_TE + 5) / 2)

    It repairs the collapse but carries its own problem here: with ue_TE < 1 the
    exponent makes Cd DECREASE as H grows, so d(Cd)/d(H) has the sign the data
    shows for only 53.2% of rows (coin flip).  That sign is not wrong inside
    Squire-Young's own derivation -- it is a far-wake momentum statement for a
    thin attached wake -- but it is outside its validity at a separated TE.

    "fitted_poly2" (the default) is a degree-2 polynomial in the SAME three
    physical inputs, log10 Cd = poly2(log theta_TE, H_TE, ue_TE), with the 10
    coefficients below fit on 325,370 rows of the merged-clean training deck and
    validated on two independent evaluation sets.  Same character as XFOIL's own
    Cf / H* / CD closures: empirical relations among BL state variables, no
    geometry, alpha or Re input.  MEASURED, same predicted state throughout, with
    the evaluation gate's clip to [1e-4, 0.1] applied to every closure (only
    "fitted_poly2" clips internally; the other two are clipped by the caller, as
    the eval scripts have always done -- comparing them unclipped is meaningless
    because both diverge in deep separation):

                       original holdout (n=2737)   separated set (n=1930)
        legacy_ode                     0.8507                    0.6849
        squire_young                   0.8390                    0.8452
        fitted_poly2                   0.8904                    0.9338

        d(Cd)/d(theta) > 0     d(Cd)/d(H) > 0      monotone in H on extrapolation
        squire_young   98.9%              53.2%                    no
        fitted_poly2   99.2%              99.2%                   yes

    Training range is wide (H in [1.09, 490], ue in [0.692, 1.862]); both
    evaluation sets fall inside it.  Output is clipped to [1e-4, 0.1], the same
    band the evaluation gate uses.

    n_steps, L_wake and k_H are ignored by "squire_young" and "fitted_poly2"
    (neither needs a march).

    Returns
    -------
    Cd : tensor of shape (B,)
    """
    eps = tf.constant(1.0e-12, dtype=DTYPE)
    theta_TE = tf.cast(theta_TE, DTYPE)
    delta_TE = tf.cast(delta_TE, DTYPE)
    ue_TE = tf.cast(ue_TE, DTYPE)

    H_TE = delta_TE / tf.maximum(theta_TE, eps)

    if closure == "fitted_poly2":
        f = [tf.math.log(tf.maximum(theta_TE, eps)), H_TE, ue_TE]
        cols = [tf.ones_like(theta_TE)] + f + [f[i] * f[j]
                for i in range(3) for j in range(i, 3)]
        X = tf.stack(cols, axis=1)
        z = tf.reduce_sum((X - _POLY2_MU) / _POLY2_SD * _POLY2_W, axis=1)
        return tf.clip_by_value(tf.pow(tf.constant(10.0, dtype=DTYPE), z), 1.0e-4, 0.1)

    if closure == "squire_young":
        return tf.maximum(
            2.0 * theta_TE * tf.pow(tf.maximum(ue_TE, eps), (H_TE + 5.0) / 2.0),
            tf.constant(0.0, dtype=DTYPE))
    if closure != "legacy_ode":
        raise ValueError("closure must be 'fitted_poly2', 'squire_young' or "
                          "'legacy_ode', got %r" % closure)

    dx = L_wake / float(n_steps)
    due_dx = (ue_inf - ue_TE) / L_wake

    theta = theta_TE
    # Unrolled loop — small n_steps, JIT folds it tightly.
    for i in range(n_steps):
        x = (i + 1) * dx
        ue = ue_TE + (ue_inf - ue_TE) * (x / L_wake)
        H  = 1.0 + (H_TE - 1.0) * tf.exp(-x * k_H)
        dtheta = -(H + 2.0) * theta / tf.maximum(ue, eps) * due_dx
        theta = theta + dtheta * dx

    H_far  = 1.0 + (H_TE - 1.0) * tf.exp(-L_wake * k_H)
    ue_far = tf.constant(ue_inf, dtype=DTYPE)
    Cd = 2.0 * theta * ue_far ** ((H_far + 5.0) / 2.0)
    # Clamp pathological negatives (θ shouldn't go negative under sane state)
    return tf.maximum(Cd, tf.constant(0.0, dtype=DTYPE))
