# -*- coding: utf-8 -*-
"""OP2 — pure supervised training against XFOIL viscous (θ, H) targets.

  L = MSE(θ_pred, θ_xfoil) per branch (upper + lower)
    + MSE(H_pred, H_xfoil) per branch
    + λ_cons · MSE(δ*_pred - θ_pred·H_pred, 0)
    + λ_dstar · MSE(δ*_pred, δ*_xfoil)    [optional, redundant if cons high]
    + λ_xtr · MSE(x_tr_pred, x_tr_xfoil)  [optional transition supervision]

OP1 frozen (loaded from supervised v2 weight if available, else physics).
OP2 architecture unchanged from current direct_blend mode.

Usage:
    python -m staged.train_op2_supervised \\
        --data staged/data/xfoil_all_targets.npz \\
        --epochs 1000 --batch 128 --lr 5e-4 --out_tag v2
"""
from __future__ import annotations
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# cp949 (Korean Windows) consoles raise UnicodeEncodeError on unicode in log
# prints (theta/delta/alpha glyphs, em-dash). Degrade to '?' instead of crashing.
import sys
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

import argparse, time
from pathlib import Path
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from staged.common import naca4_to_cst_coeffs
from staged.common.wake_ode_tf import wake_ode_cd_tf

_CLOSURE = None
def _fit_closure_cd(theta, dstar, ue):
    """Fitted Squire-Young-class closure: Cd = f(theta_TE, H_TE, ue_TE).
    Same three inputs as wake_ode_cd_tf; differentiable; coefficients frozen
    from a deg-2 ridge fit on the training deck (no eval data seen)."""
    mu, sd, w = _CLOSURE
    H = dstar / tf.maximum(theta, 1e-8)
    f = [tf.math.log(tf.maximum(theta, 1e-12)), H, ue]
    cols = [tf.ones_like(theta)] + f + [f[i] * f[j]
            for i in range(3) for j in range(i, 3)]
    X = tf.stack(cols, axis=1)
    return tf.clip_by_value(tf.pow(10.0, tf.reduce_sum((X - mu) / sd * w, 1)),
                            1e-4, 0.1)
from op2_dense_surrogate import (
    build_op2_surrogate,
    THETA_SCALE, DELTA_SCALE,
    THETA_WAKE_SCALE, DELTA_WAKE_SCALE, UE_WAKE_SCALE,
)

DTYPE = tf.float32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int,
                    default=int(os.environ.get("OP2_SUP_EPOCHS", 1000)))
    ap.add_argument("--batch",  type=int,
                    default=int(os.environ.get("BATCH_SIZE", 128)))
    ap.add_argument("--lr",     type=float,
                    default=float(os.environ.get("OP2_SUP_LR", 5e-4)))
    ap.add_argument("--lambda_theta", type=float, default=1.0)
    ap.add_argument("--lambda_dstar", type=float, default=1.0)
    ap.add_argument("--lambda_te",    type=float, default=10.0,
                    help="Extra weight on TE-point (last ξ) values of θ, δ*. "
                         "TE state drives Cd via Squire-Young, but a uniform "
                         "MSE over 121 ξ points gives the TE only ~1% weight.")
    ap.add_argument("--lambda_cd",    type=float, default=0.0,
                    help="Direct Cd loss via wake-ODE in forward chain. "
                         "Inputs: θ_w_pred, δ_w_pred from OP2 + ue_TE from XFOIL "
                         "(supervised). Trains θ/δ jointly with Cd sensitivity.")
    ap.add_argument("--cd_scale",     type=float, default=1.0e-2,
                    help="Normalization for L_Cd: MSE((Cd_pred - Cd_xf)/cd_scale)².")
    ap.add_argument("--cd_n_steps",   type=int, default=25,
                    help="Wake-ODE Euler steps inside the Cd loss. 100 (eval "
                         "default) is overkill for a training signal and the "
                         "unrolled backward blows up XLA memory; 25 is plenty.")
    ap.add_argument("--lambda_wake",  type=float, default=0.0,
                    help="Wake supervision (θ_wake + δ_wake + ue_wake). DEFAULT 0 = wake heads are NOT supervised; the wake state used for drag then comes from integrating the trailing-edge state through the Squire-Young wake ODE, not from these heads"
                         "Set 1.0 to enable wake learning.")
    # GATE 2-W (2026-08-27) — two additions, both backward-compatible:
    #   --enable_wake  the wake heads were previously UNBUILDABLE from this
    #                  trainer (build_op2_surrogate was called with a hardcoded
    #                  enable_wake=False), so --lambda_wake>0 could only ever
    #                  crash on out["theta_wake"].  -1 = auto (build the heads
    #                  iff lambda_wake>0 and the dataset carries wake fields),
    #                  which reproduces the old behaviour exactly at the old
    #                  default lambda_wake=0.
    #   --seed         the run was previously unseeded (numpy shuffle + TF
    #                  init).  Seed control is required to tell a real gain
    #                  from seed spread.  Default 0 (was: nondeterministic).
    ap.add_argument("--enable_wake", type=int, default=-1,
                    help="Build the 3 wake heads. -1 = auto (lambda_wake>0 and "
                         "wake fields present). 0 forces the deployed "
                         "no-wake-head model; 1 forces the heads to exist.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for numpy batch shuffling and TF weight init.")
    # GATE 2-W RE-RUN (2026-08-27) — two more additions, both DEFAULT-OFF so
    # the 2026-08-27 first run reproduces bit-for-bit.
    #
    #   --loss_winsor P   Upper-winsorise the TRAINING targets at the P-th
    #                     percentile of the FIT (train) rows, per channel,
    #                     BEFORE the scaled MSE.  Measured motivation
    #                     (VortFoil/code/gate2w_scale_audit.py, fit rows only,
    #                     no test data read): with the shipped scales the top
    #                     0.1% of rows carry 98.3% of the delta_wake term,
    #                     85.6% of delta_l and 62.5% of delta_u, because the
    #                     delta* channels have a tail that is DISCONTINUOUS
    #                     with their own distribution (delta_wake p99.9 =
    #                     0.183 chord, p100 = 83.5 chord -> 456x).  theta is
    #                     not affected (p100/p99.9 = 3.6x) because momentum
    #                     thickness is bounded and delta* is not.
    #                     A per-channel RESCALE cannot fix this: the share of a
    #                     term held by its worst rows is invariant under
    #                     division by a scalar (measured: 98.3% -> 98.3%).
    #                     Huber at 3 sigma only reaches 24.6%.  The clip
    #                     reaches 5.2% while touching 0.54% of rows.
    #                     EVALUATION TARGETS ARE NEVER WINSORISED — this is a
    #                     training-loss change only; relL2 and Cd are always
    #                     scored against the raw truth.
    #   --select_metric   Which validation composite the best-val checkpoint
    #                     minimises.  'total' (default, = the first run) is
    #                     each arm's OWN objective and therefore includes
    #                     lambda_wake*L_w for wake arms, so arms are selected
    #                     on different metrics.  'surface' uses
    #                     lambda_theta*L_th + lambda_dstar*L_d + lambda_te*L_te
    #                     ONLY -- an identical formula in every arm, and the
    #                     baseline's objective, so it is the choice that is
    #                     LEAST favourable to the wake treatment.
    ap.add_argument("--loss_winsor", type=float, default=0.0,
                    help="Percentile P for upper-winsorising TRAINING targets "
                         "(0 = off = shipped behaviour). 99.9 = the GATE 2-W "
                         "RE-RUN declared value. ue_wake is clipped two-sided "
                         "at [100-P, P] since both its tails are pathological.")
    ap.add_argument("--select_metric", default="total",
                    choices=["total", "surface"],
                    help="Validation composite for best-val checkpointing. "
                         "'total' = own objective (first-run behaviour). "
                         "'surface' = common surface-only composite.")
    ap.add_argument("--out_tag", default=os.environ.get("OP2_SUP_OUT_TAG", "sup_v2"))
    ap.add_argument("--print_every", type=int, default=25)
    ap.add_argument("--use_jit", type=int, default=1)
    ap.add_argument("--init_weights", default=os.environ.get("OP2_SUP_INIT", ""),
        help="Pre-trained OP2 weights to warm-start (zero-pads new cond cols "
             "if loading a no-encoder checkpoint into a profile_enc model).")
    ap.add_argument("--use_profile_enc", type=int,
        default=int(os.environ.get("OP2_USE_PROFILE_ENC", 0)),
        help="Enable 1D causal-conv airfoil profile encoder (concat with CST cond).")
    ap.add_argument("--profile_enc_hidden", type=int, default=64)
    ap.add_argument("--profile_enc_kernel", type=int, default=5)
    ap.add_argument("--profile_enc_depth",  type=int, default=4)
    # Capacity sweep (task 2): Branch-Trunk rank/depth + ADAF mode count.
    ap.add_argument("--branch_dim",   type=int, default=128)
    ap.add_argument("--branch_depth", type=int, default=3)
    ap.add_argument("--trunk_depth",  type=int, default=2)
    ap.add_argument("--cd_closure", default="ode", choices=("ode", "fit"),
                    help="Closure used by the Cd loss: deployed wake-ODE, or "
                         "the fitted deg-2 polynomial in (theta,H,ue).")
    ap.add_argument("--closure_npz", default="")
    ap.add_argument("--head_mode", default="dense",
                    choices=("bt", "dense"))
    ap.add_argument("--N_m",          type=int, default=16,
                    help="ADAF Fourier modes (spectral resolution of theta/dstar).")
    args = ap.parse_args()

    # `tf.random.set_seed` alone does NOT reproduce Keras weight init on
    # TF 2.10 (Keras keeps its own seed generator) — MEASURED on the RTX host:
    # two runs at the same seed gave different initial weights (maxdiff 0.40,
    # i.e. as different as two different seeds).  `set_random_seed` seeds
    # python/numpy/tf AND resets the Keras generator -> bit-identical init.
    if args.cd_closure == "fit":
        _c = np.load(args.closure_npz)
        globals()["_CLOSURE"] = tuple(
            tf.constant(_c[k], DTYPE) for k in ("mu", "sd", "w"))
        print(f"Cd loss closure = fitted poly2 ({args.closure_npz})")

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    try:
        tf.keras.utils.set_random_seed(args.seed)
    except AttributeError:
        print("WARN: tf.keras.utils.set_random_seed missing → init NOT seeded")

    # ---- Load data ----
    z = np.load(args.data, allow_pickle=False)
    def take(split, k): return z[f"{split}_{k}"]
    tr_v = z["train_valid"]; va_v = z["val_valid"]
    tr_idx = np.where(tr_v)[0]; va_idx = np.where(va_v)[0]

    def pack(split, idx):
        d = {
            "cst":     take(split, "cst")[idx],
            "alpha":   take(split, "alpha")[idx],
            "Re":      take(split, "Re")[idx],
            "theta_u": take(split, "theta_u")[idx],
            "theta_l": take(split, "theta_l")[idx],
            "delta_u": take(split, "delta_u")[idx],
            "delta_l": take(split, "delta_l")[idx],
            "H_u":     take(split, "H_u")[idx],
            "H_l":     take(split, "H_l")[idx],
        }
        # ue_TE (XFOIL viscous) + Cd target — used by wake-ODE Cd loss
        if f"{split}_ue_u" in z.files and f"{split}_ue_l" in z.files:
            ue_u = z[f"{split}_ue_u"][idx]; ue_l = z[f"{split}_ue_l"][idx]
            d["ue_TE"] = (0.5 * (np.abs(ue_u[:, -1])
                                  + np.abs(ue_l[:, -1])))[:, None].astype(np.float32)
        if f"{split}_cd_visc" in z.files:
            d["cd"] = z[f"{split}_cd_visc"][idx].astype(np.float32)
        # Wake fields — present only in wake-enabled dataset.
        for k in ("theta_wake", "delta_wake", "ue_wake"):
            key = f"{split}_{k}"
            if key in z.files:
                d[k] = z[key][idx]
        return d
    # alias-friendly tensor names
    _ = None
    tr = pack("train", tr_idx); va = pack("val", va_idx)
    N_tr = len(tr_idx); N_va = len(va_idx)
    print(f"Data: train={N_tr}, val={N_va}")
    print(f"  θ_u range: [{tr['theta_u'].min():.3e}, {tr['theta_u'].max():.3e}]")
    print(f"  H_u range: [{tr['H_u'].min():.3f},   {tr['H_u'].max():.3f}]")
    print(f"  δ*_u range:[{tr['delta_u'].min():.3e}, {tr['delta_u'].max():.3e}]")

    # ---- Wake availability / wake-head decision (must precede the build) ----
    wake_in_data = ("theta_wake" in tr) and ("delta_wake" in tr) and ("ue_wake" in tr)
    wake_enabled = (args.lambda_wake > 0) and wake_in_data
    if args.enable_wake < 0:
        build_wake = wake_enabled
    else:
        build_wake = bool(args.enable_wake)
    if wake_enabled and not build_wake:
        print("WARN: --lambda_wake>0 but --enable_wake 0 → wake loss disabled")
        wake_enabled = False
    if args.lambda_wake > 0 and not wake_in_data:
        print("WARN: --lambda_wake>0 but dataset has no theta/delta/ue_wake → disabled")

    # ---- GATE 2-W RE-RUN: training-target winsorisation (default OFF) ----
    # Bounds come from the FIT (train) split ONLY, pooled over nodes, so they
    # are byte-identical for every arm reading this data file and no
    # validation/test information enters them.  They are printed and stored in
    # the history npz for audit.  Applied to BOTH train and val targets, so
    # val_total stays a like-for-like selection metric across arms.
    winsor_bounds = {}
    if args.loss_winsor > 0:
        P = float(args.loss_winsor)
        chans = ["theta_u", "theta_l", "delta_u", "delta_l"]
        if wake_in_data:
            chans += ["theta_wake", "delta_wake", "ue_wake"]
        for ch in chans:
            pooled = tr[ch].astype(np.float64).ravel()
            hi = float(np.percentile(pooled, P))
            lo = (float(np.percentile(pooled, 100.0 - P))
                  if ch == "ue_wake" else -np.inf)
            winsor_bounds[ch] = (lo, hi)
            n_tr_touch = float(((tr[ch] < lo) | (tr[ch] > hi)).any(axis=1).mean())
            tr[ch] = np.clip(tr[ch], lo, hi).astype(np.float32)
            va[ch] = np.clip(va[ch], lo, hi).astype(np.float32)
            print(f"  winsor[{ch:11s}] p{P} -> "
                  f"[{lo:.6g}, {hi:.6g}]  train rows touched "
                  f"{100.0 * n_tr_touch:.3f}%")
        print(f"  winsor: TRAINING+VAL targets only. Evaluation is always "
              f"against RAW truth (op2_wake_eval.py never winsorises).")
    else:
        print("  winsor: OFF (shipped behaviour; reproduces the first run)")

    # ---- Build pure-supervised OP2 surrogate (no OP1 dependency) ----
    n_sp = int(z["n_surface_pts"][0])
    op2 = build_op2_surrogate(
        n_surface_pts=n_sp,
        enable_wake=build_wake,
        branch_dim=args.branch_dim,
        branch_depth=args.branch_depth,
        trunk_depth=args.trunk_depth,
        N_m=args.N_m,
        head_mode=args.head_mode,
        use_profile_enc=bool(args.use_profile_enc),
        profile_enc_hidden=args.profile_enc_hidden,
        profile_enc_kernel=args.profile_enc_kernel,
        profile_enc_depth=args.profile_enc_depth,
    )
    cst_init = tf.constant(tr["cst"][:2], dtype=DTYPE)
    a_init   = tf.constant(tr["alpha"][:2], dtype=DTYPE)
    Re_init  = tf.constant(tr["Re"][:2], dtype=DTYPE)
    _ = op2.predict_bl(cst_init, a_init, Re_init, training=False)
    op2_vars = op2.trainable_variables_list()
    print(f"OP2 surrogate trainable params: "
          f"{sum(int(np.prod(v.shape)) for v in op2_vars)} "
          f"({len(op2_vars)} variables); use_profile_enc={bool(args.use_profile_enc)}")

    if args.init_weights:
        from pathlib import Path as _P
        if _P(args.init_weights).exists():
            info = op2.load_weights(args.init_weights, strict=False)
            print(f"OP2 init from {args.init_weights}: {info}")

    # ---- Optimizer ----
    n_batches = int(np.ceil(N_tr / args.batch))
    decay_steps = n_batches * args.epochs
    lr_sched = tf.keras.optimizers.schedules.CosineDecay(
        args.lr, decay_steps, alpha=0.1)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_sched)

    lam_θ  = tf.constant(args.lambda_theta, dtype=DTYPE)
    lam_d  = tf.constant(args.lambda_dstar, dtype=DTYPE)
    lam_te = tf.constant(args.lambda_te,    dtype=DTYPE)
    lam_cd = tf.constant(args.lambda_cd,    dtype=DTYPE)
    cd_scale = tf.constant(args.cd_scale,   dtype=DTYPE)
    lam_w  = tf.constant(args.lambda_wake,  dtype=DTYPE)
    print(f"wake: in_data={wake_in_data} heads_built={build_wake} "
          f"loss_active={wake_enabled} lambda_wake={args.lambda_wake} "
          f"seed={args.seed}")
    cd_enabled = (args.lambda_cd > 0) and ("ue_TE" in tr) and ("cd" in tr)
    if args.lambda_cd > 0 and not cd_enabled:
        print(f"WARN: --lambda_cd>0 but dataset missing ue_u/ue_l/cd_visc → disabled")

    @tf.function(jit_compile=bool(args.use_jit))
    def step(cst, alpha, Re, θu_t, θl_t, du_t, dl_t,
              θw_t, dw_t, uw_t, ue_TE_t, cd_t, training=True):
        with tf.GradientTape() as tape:
            out = op2.predict_bl(cst, alpha, Re, training=training)
            θu = out["theta_u"]; θl = out["theta_l"]
            du = out["delta_u"]; dl = out["delta_l"]
            θu_e = (θu - θu_t) / THETA_SCALE
            θl_e = (θl - θl_t) / THETA_SCALE
            du_e = (du - du_t) / DELTA_SCALE
            dl_e = (dl - dl_t) / DELTA_SCALE
            L_θ = (tf.reduce_mean(θu_e ** 2)
                    + tf.reduce_mean(θl_e ** 2))
            L_d = (tf.reduce_mean(du_e ** 2)
                    + tf.reduce_mean(dl_e ** 2))
            # TE-point loss
            L_te = (tf.reduce_mean(θu_e[:, -1] ** 2)
                     + tf.reduce_mean(θl_e[:, -1] ** 2)
                     + tf.reduce_mean(du_e[:, -1] ** 2)
                     + tf.reduce_mean(dl_e[:, -1] ** 2))
            # Wake loss (active only when --lambda_wake > 0)
            if wake_enabled:
                θw = out["theta_wake"]; dw = out["delta_wake"]
                uw = out["ue_wake"]
                θw_e = (θw - θw_t) / THETA_WAKE_SCALE
                dw_e = (dw - dw_t) / DELTA_WAKE_SCALE
                uw_e = (uw - uw_t) / UE_WAKE_SCALE
                L_w = (tf.reduce_mean(θw_e ** 2)
                        + tf.reduce_mean(dw_e ** 2)
                        + tf.reduce_mean(uw_e ** 2))
            else:
                L_w = tf.constant(0.0, dtype=DTYPE)
            # Cd loss via wake-ODE forward in graph.  ue_TE comes from XFOIL
            # (supervised, decouples from OP1 prediction during training).
            # Chain rule propagates back through θ_w, δ_w → SY's 4×
            # amplification, training the model with Cd sensitivity.
            if cd_enabled:
                # Floor theta/dstar to physical positives: random-init
                # predictions can be ~0 or negative -> H_TE=dstar/theta blows
                # up -> wake-ODE Cd NaN, which (via the global grad guard)
                # zeros the WHOLE batch's gradient and discards the good
                # theta/dstar signal.  Clamp at the source; the maximum's
                # gradient is 0 in the clamped region (no 1/theta^2 blowup).
                θ_w = tf.maximum(θu[:, -1] + θl[:, -1], 1e-5)
                δ_w = tf.maximum(du[:, -1] + dl[:, -1], 1e-6)
                Cd_pred = (_fit_closure_cd(θ_w, δ_w, ue_TE_t[:, 0])
                            if args.cd_closure == "fit" else
                            wake_ode_cd_tf(θ_w, δ_w, ue_TE_t[:, 0],
                                            n_steps=args.cd_n_steps))
                cd_err = (Cd_pred - cd_t[:, 0]) / cd_scale
                cd_err = tf.where(tf.math.is_finite(cd_err),
                                  cd_err, tf.zeros_like(cd_err))   # belt+suspenders
                # Huber (not MSE): separation-case Cd outliers produce huge
                # gradients that destabilize training mid-run (observed
                # from-scratch divergence ~ep240, theta/dstar -> NaN).  Huber
                # caps the per-sample gradient magnitude at 1 -> bounded Cd
                # nudge.  delta=1 == a Cd error of cd_scale (1e-2).
                abs_e = tf.abs(cd_err)
                L_cd = tf.reduce_mean(
                    tf.where(abs_e < 1.0, 0.5 * cd_err ** 2, abs_e - 0.5))
            else:
                L_cd = tf.constant(0.0, dtype=DTYPE)
            L_surf = lam_θ * L_θ + lam_d * L_d + lam_te * L_te
            total = L_surf + lam_w * L_w + lam_cd * L_cd
        if training:
            grads = tape.gradient(total, op2_vars)
            grads = [tf.zeros_like(v) if g is None else g
                     for g, v in zip(grads, op2_vars)]
            grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                     for g in grads]
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            opt.apply_gradients(zip(grads, op2_vars))
        return {"total": total, "L_θ": L_θ, "L_d": L_d,
                 "L_te": L_te, "L_cd": L_cd, "L_w": L_w,
                 # surface-only composite: identical formula in EVERY arm, so
                 # --select_metric surface selects every arm on one metric.
                 "surface": L_surf}

    @tf.function(jit_compile=bool(args.use_jit))
    def eval_step(cst, alpha, Re, θu_t, θl_t, du_t, dl_t,
                    θw_t, dw_t, uw_t, ue_TE_t, cd_t):
        return step(cst, alpha, Re, θu_t, θl_t, du_t, dl_t,
                     θw_t, dw_t, uw_t, ue_TE_t, cd_t, training=False)

    # ---- Constants ----
    def to_tf(d):
        return {k: tf.constant(v, dtype=DTYPE) for k, v in d.items()}
    tr_tf = to_tf(tr); va_tf = to_tf(va)

    # Wake tensors — placeholder zeros if wake not in dataset (lambda_wake=0).
    if wake_enabled:
        _wake_keys = ("theta_wake", "delta_wake", "ue_wake")
    else:
        # Create dummy zero tensors (matched shape) so step() trace consistent
        n_wake = op2.n_wake if hasattr(op2, "n_wake") else 25
        for k in ("theta_wake", "delta_wake", "ue_wake"):
            if k not in tr_tf:
                tr_tf[k] = tf.zeros((tr_tf["cst"].shape[0], n_wake), dtype=DTYPE)
                va_tf[k] = tf.zeros((va_tf["cst"].shape[0], n_wake), dtype=DTYPE)

    # ue_TE / Cd placeholders if Cd loss disabled
    for k, shape_tail in [("ue_TE", (1,)), ("cd", (1,))]:
        if k not in tr_tf:
            tr_tf[k] = tf.zeros((tr_tf["cst"].shape[0], *shape_tail), dtype=DTYPE)
            va_tf[k] = tf.zeros((va_tf["cst"].shape[0], *shape_tail), dtype=DTYPE)

    print("Tracing ...")
    t0 = time.time()
    _ = step(tr_tf["cst"][:args.batch], tr_tf["alpha"][:args.batch],
              tr_tf["Re"][:args.batch],
              tr_tf["theta_u"][:args.batch], tr_tf["theta_l"][:args.batch],
              tr_tf["delta_u"][:args.batch], tr_tf["delta_l"][:args.batch],
              tr_tf["theta_wake"][:args.batch],
              tr_tf["delta_wake"][:args.batch],
              tr_tf["ue_wake"][:args.batch],
              tr_tf["ue_TE"][:args.batch],
              tr_tf["cd"][:args.batch])
    _ = eval_step(va_tf["cst"], va_tf["alpha"], va_tf["Re"],
                    va_tf["theta_u"], va_tf["theta_l"],
                    va_tf["delta_u"], va_tf["delta_l"],
                    va_tf["theta_wake"], va_tf["delta_wake"], va_tf["ue_wake"],
                    va_tf["ue_TE"], va_tf["cd"])
    print(f"  Trace done in {time.time() - t0:.1f}s")

    history = {k: [] for k in ("total", "L_θ", "L_d", "L_te", "L_cd", "L_w",
                                 "surface", "val_total", "val_surface",
                                 "val_L_w", "val_select")}
    out_w = Path(f"staged/weights/op2_final_eps0.50_direct.{args.out_tag}.npz")
    out_w.parent.mkdir(parents=True, exist_ok=True)
    out_h = out_w.with_suffix(".history.npz")
    snap_every = max(1, int(os.environ.get("OP2_SUP_SNAP_EVERY",
                                              max(10, args.epochs // 20))))
    def _hist_payload(h):
        """history + run provenance, so an arm's npz is self-describing."""
        d = {k: np.array(v, dtype=np.float32) for k, v in h.items()}
        d["prov_loss_winsor"] = np.array([args.loss_winsor], np.float64)
        d["prov_select_metric"] = np.array([args.select_metric], "<U16")
        d["prov_lambda_wake"] = np.array([args.lambda_wake], np.float64)
        d["prov_enable_wake"] = np.array([int(build_wake)], np.int32)
        d["prov_wake_loss_active"] = np.array([int(wake_enabled)], np.int32)
        d["prov_seed"] = np.array([args.seed], np.int32)
        d["prov_use_jit"] = np.array([int(args.use_jit)], np.int32)
        d["prov_split_fingerprint"] = np.array(
            [str(z["split_fingerprint"][0]) if "split_fingerprint" in z.files
             else "ABSENT"], "<U24")
        if winsor_bounds:
            d["prov_winsor_channels"] = np.array(
                list(winsor_bounds.keys()), "<U16")
            d["prov_winsor_bounds"] = np.array(
                [winsor_bounds[k] for k in winsor_bounds], np.float64)
        return d

    best_val = float("inf"); best_ep = 0
    iterbar = tqdm(range(1, args.epochs + 1), desc="OP2 sup", unit="ep",
                    mininterval=2.0, dynamic_ncols=True)
    for ep in iterbar:
        perm = np.random.permutation(N_tr)
        running = {k: 0.0 for k in ("total", "L_θ", "L_d", "L_te", "L_cd",
                                      "L_w", "surface")}
        for bi in range(n_batches):
            idx = perm[bi * args.batch:(bi + 1) * args.batch]
            it = tf.constant(idx, dtype=tf.int32)
            out = step(
                tf.gather(tr_tf["cst"], it),     tf.gather(tr_tf["alpha"], it),
                tf.gather(tr_tf["Re"], it),
                tf.gather(tr_tf["theta_u"], it), tf.gather(tr_tf["theta_l"], it),
                tf.gather(tr_tf["delta_u"], it), tf.gather(tr_tf["delta_l"], it),
                tf.gather(tr_tf["theta_wake"], it),
                tf.gather(tr_tf["delta_wake"], it),
                tf.gather(tr_tf["ue_wake"],   it),
                tf.gather(tr_tf["ue_TE"],     it),
                tf.gather(tr_tf["cd"],        it))
            for k in running:
                running[k] += float(out[k].numpy())
        for k in running:
            running[k] /= n_batches
        val_out = eval_step(va_tf["cst"], va_tf["alpha"], va_tf["Re"],
                              va_tf["theta_u"], va_tf["theta_l"],
                              va_tf["delta_u"], va_tf["delta_l"],
                              va_tf["theta_wake"], va_tf["delta_wake"],
                              va_tf["ue_wake"], va_tf["ue_TE"], va_tf["cd"])
        val_tot = float(val_out["total"].numpy())
        val_surf = float(val_out["surface"].numpy())
        val_Lw = float(val_out["L_w"].numpy())
        val_sel = val_surf if args.select_metric == "surface" else val_tot
        for k in running:
            history[k].append(running[k])
        history["val_total"].append(val_tot)
        history["val_surface"].append(val_surf)
        history["val_L_w"].append(val_Lw)
        history["val_select"].append(val_sel)
        if ep == 1 or ep % args.print_every == 0 or ep == args.epochs:
            tqdm.write(
                f"[{ep:4d}/{args.epochs}] tot={running['total']:.3e}  "
                f"L_θ={running['L_θ']:.3e}  L_d={running['L_d']:.3e}  "
                f"L_te={running['L_te']:.3e}  L_cd={running['L_cd']:.3e}  "
                f"L_w={running['L_w']:.3e}  surf={running['surface']:.3e}  "
                f"val={val_tot:.3e}  val_surf={val_surf:.3e}  "
                f"val_L_w={val_Lw:.3e}")
        # Best-val checkpoint: keep the lowest-val weights, not the last
        # (guards against over-fitting on the diverse UIUC distribution).
        # --select_metric surface makes this metric identical across arms.
        if val_sel < best_val:
            best_val = val_sel
            best_ep = ep
            op2.save_weights(out_w)
        if ep % snap_every == 0 or ep == args.epochs:
            np.savez(out_h, **_hist_payload(history))

    # Final: weights already saved at best val; persist history + report
    np.savez(out_h, **{k: np.array(v, dtype=np.float32)
                         for k, v in history.items()})
    print(f"Best val[{args.select_metric}]={best_val:.3e} at epoch "
           f"{best_ep}/{args.epochs}")
    print(f"Weights (best-val) → {out_w}")
    print(f"History → {out_h}")


if __name__ == "__main__":
    main()
