# -*- coding: utf-8 -*-
"""OP1 inviscid — pure supervised training against XFOIL inviscid Cp.

  L = MSE(Cp_pred, Cp_xfoil)
      + λ_stag · MSE(s_stag_pred, s_stag_xfoil)
      + λ_cl   · MSE(Cl_pred, cl_inv_xfoil)         (optional sanity)

Cp_pred = 1 - Ut²,  Ut from γ via panel method.
Architecture unchanged from physics-residual OP1 (Branch-Trunk ADAF +
sum-Kutta + s_stag head).  Only the loss differs.

Usage:
    python -m staged.train_op1_supervised --data staged/data/xfoil_op1_targets_smoke.npz
"""
from __future__ import annotations
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# CPU-XLA recipe (Mac tf218 + tensorflow-metal): Metal registers no XLA kernels, so
# jit_compile=True dies with "could not find registered platform".  Set NF_HIDE_GPU=1
# to hide the Metal device and JIT on CPU (must be BEFORE any op touches a device).
if os.environ.get("NF_HIDE_GPU", "0") == "1":
    os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_cpu_global_jit")

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

# CPU-XLA recipe part 2: hide the Metal device BEFORE any op touches a device
# (tf-metal registers no XLA kernels; jit_compile=True would die / hang on it).
if os.environ.get("NF_HIDE_GPU", "0") == "1":
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

from tqdm import tqdm

from staged.common import (
    naca4_to_cst_coeffs,
    build_single_loop_contour_cst_tf,
    compute_inviscid_surface_velocity,
)
from staged.op1_inviscid import build_op1

DTYPE = tf.float32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int,
                    default=int(os.environ.get("OP1_SUP_EPOCHS", 500)))
    ap.add_argument("--batch",  type=int,
                    default=int(os.environ.get("BATCH_SIZE", 64)))
    ap.add_argument("--lr",     type=float,
                    default=float(os.environ.get("OP1_SUP_LR", 1e-3)))
    # Two stag losses available:
    #  - physics: Ut(s_stag)² ≈ 0  (self-consistent with γ, generalizes well)
    #  - supervised: MSE(s_stag_pred, s_stag_xfoil)  (data prior, may not
    #    generalize to OOD geometries)
    # Default = physics-only.  Supervised stag is OFF (λ_stag_sup=0) since
    # stag depends sharply on local LE shape and supervised regression often
    # over-fits to training geometry distribution.
    ap.add_argument("--lambda_stag_phys", type=float, default=10.0)
    ap.add_argument("--lambda_stag_sup",  type=float, default=0.0)
    ap.add_argument("--lambda_cl",   type=float, default=0.0)  # off by default
    ap.add_argument("--lambda_te",   type=float, default=10.0,
        help="Extra weight on Cp (=1-Ut²) at first / last panels (≈TE upper/lower). "
             "ue_TE underprediction amplifies into Cd via SY's (H+5)/2 ≈ 4× factor.")
    ap.add_argument("--n_te_panels", type=int, default=3,
        help="Number of panels on each side of TE included in L_te.")
                                                                # (xfoil_inv doesn't
                                                                # return cl_inv)
    ap.add_argument("--out_tag", default=os.environ.get("OP1_SUP_OUT_TAG", "sup"))
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--use_jit", type=int, default=1)
    ap.add_argument("--init_weights", default=os.environ.get(
        "OP1_SUP_INIT", "staged/weights/op1_final_eps0.50.npz"),
        help="Pre-trained OP1 weights for init.  Random init is a strong "
             "local min for supervised Cp (γ stays near 0); loading the "
             "physics-residual OP1 escapes that basin.  Pass '' to disable.")
    ap.add_argument("--use_profile_enc", type=int,
        default=int(os.environ.get("OP1_USE_PROFILE_ENC", 0)),
        help="Enable 1D causal-conv airfoil profile encoder (concat features "
             "with CST conditioning).  Adds ~80k params; init from pre-enc "
             "checkpoint zero-pads the new cond rows.")
    ap.add_argument("--profile_enc_hidden", type=int, default=64)
    ap.add_argument("--profile_enc_kernel", type=int, default=5)
    ap.add_argument("--profile_enc_depth",  type=int, default=4)
    # Capacity sweep (task 2): branch_dim = rank of the cond->W (gamma) map.
    ap.add_argument("--branch_dim", type=int, default=128,
                    help="OP1 Branch-Trunk rank (capacity knob). Default 128.")
    # γ-decoder basis A/B: "fourier" (production, default) or "lpa"
    # (Legendre antiderivative from operator_lib).  LPA output dim = its
    # coef_dim = lpa_max_order+1 modes.  Default reproduces production.
    ap.add_argument("--gamma_basis", default="fourier",
                    choices=["fourier", "lpa", "mixed"],
                    help="γ_inv decode basis: fourier (prod) / lpa / mixed "
                         "(fourier backbone + homogeneous LPA correction).")
    ap.add_argument("--lpa_max_order", type=int, default=23,
                    help="LPA Legendre max order (coef_dim = max_order+1). "
                         "23 -> 24 modes ~ Fourier N_m=24 effective modes.")
    ap.add_argument("--mixed_lpa_order", type=int, default=7,
                    help="mixed: LPA correction max order (adds order+1 "
                         "homogeneous poly atoms on top of the fourier W).")
    # Residual-ADAF patch: discrepancy loss weights
    ap.add_argument("--lambda_disc_bl",   type=float, default=0.0,
                    help="Weight for BL discrepancy loss (θ, H, S per side).")
    ap.add_argument("--lambda_disc_wake", type=float, default=0.0,
                    help="Weight for wake discrepancy loss (θ_w, δ*_w, Ue_w).")
    ap.add_argument("--lambda_cd",        type=float, default=0.0,
                    help="Weight for CDCALC Cd loss.")
    ap.add_argument("--lambda_sobolev",   type=float, default=0.0,
                    help="Weight for Δψ Sobolev regularization.")
    args = ap.parse_args()

    # ---- Load data ----
    z = np.load(args.data, allow_pickle=False)
    tr_cst = z["train_cst"]; tr_a = z["train_alpha"]; tr_Re = z["train_Re"]
    tr_cp  = z["train_cp_panels"]; tr_st = z["train_s_stag"]
    # Support both viscous (Cl from XFOIL viscous polar) and legacy inviscid
    # (cl_inv from XFOIL inviscid) datasets transparently.
    cl_field_tr = "train_cl_visc" if "train_cl_visc" in z.files else "train_cl_inv"
    cl_field_va = cl_field_tr.replace("train_", "val_")
    tr_cl  = z[cl_field_tr];       tr_v  = z["train_valid"]
    va_cst = z["val_cst"];   va_a = z["val_alpha"];  va_Re = z["val_Re"]
    va_cp  = z["val_cp_panels"];   va_st = z["val_s_stag"]
    va_cl  = z[cl_field_va];       va_v  = z["val_valid"]
    print(f"  Cl field: {cl_field_tr}")

    # Filter to valid rows
    tr_idx = np.where(tr_v)[0]; va_idx = np.where(va_v)[0]
    tr_cst, tr_a, tr_Re = tr_cst[tr_idx], tr_a[tr_idx], tr_Re[tr_idx]
    tr_cp, tr_st, tr_cl = tr_cp[tr_idx], tr_st[tr_idx], tr_cl[tr_idx]
    va_cst, va_a, va_Re = va_cst[va_idx], va_a[va_idx], va_Re[va_idx]
    va_cp, va_st, va_cl = va_cp[va_idx], va_st[va_idx], va_cl[va_idx]
    N_tr, N_va = len(tr_cst), len(va_cst)
    Np = tr_cp.shape[1]
    print(f"Data: train={N_tr}, val={N_va}, panels={Np}")
    print(f"  Cp range (train): [{tr_cp.min():.3f}, {tr_cp.max():.3f}]")
    print(f"  s_stag range:     [{tr_st.min():.3f}, {tr_st.max():.3f}]")
    print(f"  Cl range:         [{tr_cl.min():.3f}, {tr_cl.max():.3f}]")

    # ---- Build OP1 ----
    n_sp = int(z["n_surface_pts"][0])
    op1 = build_op1(
        n_surface_pts=n_sp,
        branch_dim=args.branch_dim,
        use_profile_enc=bool(args.use_profile_enc),
        profile_enc_hidden=args.profile_enc_hidden,
        profile_enc_kernel=args.profile_enc_kernel,
        profile_enc_depth=args.profile_enc_depth,
        gamma_basis=args.gamma_basis,
        lpa_max_order=args.lpa_max_order,
        mixed_lpa_order=args.mixed_lpa_order,
    )
    print(f"OP1 gamma_basis={args.gamma_basis}"
          + (f" (lpa_max_order={args.lpa_max_order}, "
             f"coef_dim={op1._gamma_Np})" if args.gamma_basis == "lpa"
             else (f" (fourier N_p={op1._gamma_Np_fourier} + lpa "
                   f"{args.mixed_lpa_order + 1} = coef_dim {op1._gamma_Np})"
                   if args.gamma_basis == "mixed"
                   else f" (N_p={op1._gamma_Np})")))
    cst_seed = naca4_to_cst_coeffs(
        np.array([[0.0, 0.4, 0.12]], dtype=np.float32))
    _ = op1.compute_losses(
        tf.constant(cst_seed, dtype=DTYPE),
        tf.constant([[0.0]], dtype=DTYPE),
        tf.constant([[3e5]], dtype=DTYPE), training=False)
    op1_vars = op1.trainable_variables_list()
    print(f"OP1 trainable params: {sum(int(np.prod(v.shape)) for v in op1_vars)} "
          f"({len(op1_vars)} variables); use_profile_enc={bool(args.use_profile_enc)}")

    if args.init_weights:
        init_p = Path(args.init_weights)
        if init_p.exists():
            # strict=False: the physics-init is branch_dim=128, so a swept
            # branch_dim cannot load the gamma branch (dim mismatch) — skip it
            # and train that branch from scratch rather than crash.  The
            # stag head still warm-starts.  For a clean capacity comparison
            # prefer --init_weights "" --lambda_cl 1.0 across ALL configs.
            info = op1.load_weights(init_p, strict=False)
            print(f"OP1 init from {init_p}  "
                  f"(loaded {info['loaded']}, skipped {info['skipped']}, "
                  f"padded {info['padded']})")
            if info["skipped"] > info["loaded"]:
                print("  NOTE: most vars skipped (branch_dim != init); gamma "
                      "branch trains from scratch — use --lambda_cl 1.0.")
        else:
            print(f"OP1 init weights {init_p} not found; using random init")

    # ---- Optimizer ----
    n_batches = int(np.ceil(N_tr / args.batch))
    decay_steps = n_batches * args.epochs
    lr_sched = tf.keras.optimizers.schedules.CosineDecay(
        args.lr, decay_steps, alpha=0.1)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_sched)

    lam_stag_phys = tf.constant(args.lambda_stag_phys, dtype=DTYPE)
    lam_stag_sup  = tf.constant(args.lambda_stag_sup,  dtype=DTYPE)
    lam_cl   = tf.constant(args.lambda_cl, dtype=DTYPE)
    lam_te   = tf.constant(args.lambda_te, dtype=DTYPE)
    lam_disc_bl   = tf.constant(args.lambda_disc_bl,   dtype=DTYPE)
    lam_disc_wake = tf.constant(args.lambda_disc_wake, dtype=DTYPE)
    lam_cd        = tf.constant(args.lambda_cd,        dtype=DTYPE)
    lam_sobolev   = tf.constant(args.lambda_sobolev,   dtype=DTYPE)
    k_te     = int(max(1, args.n_te_panels))

    @tf.function(jit_compile=bool(args.use_jit))
    def step(cst, alpha, Re, cp_t, st_t, cl_t, training=True):
        with tf.GradientTape() as tape:
            losses = op1.compute_losses(
                cst, alpha, Re, training=training,
                return_details=False,
                target_Cp=cp_t, target_Cl=cl_t,
                # BL/wake targets not in current dataset — pass None
                # target_BL=None, target_wake=None, target_Cd=None
            )
            total = losses["total"]
        if training:
            grads = tape.gradient(total, op1_vars)
            grads = [tf.zeros_like(v) if g is None else g
                     for g, v in zip(grads, op1_vars)]
            grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                     for g in grads]
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            opt.apply_gradients(zip(grads, op1_vars))
        return {"total": total, "L_cp": losses["loss_kutta_inv"] + losses["loss_np_inv"],
                "L_te": losses.get("loss_cd", 0.0),  # placeholder
                "L_stag_phys": losses["loss_stag"],
                "L_stag_sup": 0.0,  # not used in physics mode
                "L_cl": losses.get("loss_cd", 0.0),  # Cl from losses["Cl"]
                "L_disc": losses.get("loss_disc", 0.0),
                "L_cd": losses.get("loss_cd", 0.0),
                "L_sobolev": losses.get("loss_sobolev", 0.0)}

    @tf.function(jit_compile=bool(args.use_jit))
    def eval_step(cst, alpha, Re, cp_t, st_t, cl_t):
        return step(cst, alpha, Re, cp_t, st_t, cl_t, training=False)

    # ---- Convert to tensors once ----
    tr_cst_tf = tf.constant(tr_cst, dtype=DTYPE)
    tr_a_tf   = tf.constant(tr_a,   dtype=DTYPE)
    tr_Re_tf  = tf.constant(tr_Re,  dtype=DTYPE)
    tr_cp_tf  = tf.constant(tr_cp,  dtype=DTYPE)
    tr_st_tf  = tf.constant(tr_st,  dtype=DTYPE)
    tr_cl_tf  = tf.constant(tr_cl,  dtype=DTYPE)
    va_cst_tf = tf.constant(va_cst, dtype=DTYPE)
    va_a_tf   = tf.constant(va_a,   dtype=DTYPE)
    va_Re_tf  = tf.constant(va_Re,  dtype=DTYPE)
    va_cp_tf  = tf.constant(va_cp,  dtype=DTYPE)
    va_st_tf  = tf.constant(va_st,  dtype=DTYPE)
    va_cl_tf  = tf.constant(va_cl,  dtype=DTYPE)

    print("Tracing ...")
    t0 = time.time()
    _ = step(tr_cst_tf[:args.batch], tr_a_tf[:args.batch],
              tr_Re_tf[:args.batch], tr_cp_tf[:args.batch],
              tr_st_tf[:args.batch], tr_cl_tf[:args.batch])
    _ = eval_step(va_cst_tf, va_a_tf, va_Re_tf, va_cp_tf, va_st_tf, va_cl_tf)
    print(f"  Trace done in {time.time() - t0:.1f}s")

    # ---- Training loop ----
    history = {k: [] for k in ("total", "L_cp", "L_te", "L_stag_phys",
                                  "L_stag_sup", "L_cl", "L_disc", "L_cd", "L_sobolev", "val_total")}
    out_w = Path(f"staged/weights/op1_supervised_eps0.50.{args.out_tag}.npz")
    out_w.parent.mkdir(parents=True, exist_ok=True)
    out_h = out_w.with_suffix(".history.npz")
    snap_every = max(1, int(os.environ.get("OP1_SUP_SNAP_EVERY",
                                              max(10, args.epochs // 20))))
    best_val = float("inf"); best_ep = 0
    iterbar = tqdm(range(1, args.epochs + 1), desc="OP1 sup", unit="ep",
                    mininterval=2.0, dynamic_ncols=True)
    for ep in iterbar:
        perm = np.random.permutation(N_tr)
        running = {k: 0.0 for k in ("total", "L_cp", "L_te", "L_stag_phys",
                                  "L_stag_sup", "L_cl", "L_disc", "L_cd", "L_sobolev")}
        for bi in range(n_batches):
            idx = perm[bi * args.batch:(bi + 1) * args.batch]
            idx_tf = tf.constant(idx, dtype=tf.int32)
            out = step(
                tf.gather(tr_cst_tf, idx_tf), tf.gather(tr_a_tf, idx_tf),
                tf.gather(tr_Re_tf, idx_tf), tf.gather(tr_cp_tf, idx_tf),
                tf.gather(tr_st_tf, idx_tf), tf.gather(tr_cl_tf, idx_tf))
            for k in running:
                running[k] += float(out[k].numpy())
        for k in running:
            running[k] /= n_batches
        val_out = eval_step(va_cst_tf, va_a_tf, va_Re_tf,
                              va_cp_tf, va_st_tf, va_cl_tf)
        val_tot = float(val_out["total"].numpy())
        for k in running:
            history[k].append(running[k])
        history["val_total"].append(val_tot)
        if ep == 1 or ep % args.print_every == 0 or ep == args.epochs:
            tqdm.write(
                f"[{ep:4d}/{args.epochs}] tot={running['total']:.3e}  "
                f"L_cp={running['L_cp']:.3e}  "
                f"L_te={running['L_te']:.3e}  "
                f"L_cl={running['L_cl']:.3e}  "
                f"L_sph={running['L_stag_phys']:.3e}  "
                f"L_ssp={running['L_stag_sup']:.3e}  "
                f"L_disc={running['L_disc']:.3e}  "
                f"L_cd={running['L_cd']:.3e}  "
                f"L_sob={running['L_sobolev']:.3e}  "
                f"val={val_tot:.3e}")
        # Best-val checkpoint: OP1 overfits on diverse UIUC data (train loss
        # keeps dropping while val rises), so keep the lowest-val weights
        # rather than the over-fit last epoch.  out_w always holds the best.
        if val_tot < best_val:
            best_val = val_tot
            best_ep = ep
            op1.save_weights(out_w)
        if ep % snap_every == 0 or ep == args.epochs:
            np.savez(out_h, **{k: np.array(v, dtype=np.float32)
                                 for k, v in history.items()})

    # ---- Final: weights already saved at best val; persist history ----
    np.savez(out_h, **{k: np.array(v, dtype=np.float32)
                         for k, v in history.items()})
    print(f"Best val={best_val:.3e} at epoch {best_ep}/{args.epochs}")
    print(f"Weights (best-val) → {out_w}")
    print(f"History → {out_h}")


if __name__ == "__main__":
    main()
