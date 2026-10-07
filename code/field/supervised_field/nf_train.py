# -*- coding: utf-8 -*-
"""supervised_field/nf_train.py — MARIO-class conditional neural field on the
AirfRANS scarce_train split (192/200 cases with XTR labels, _nf_train.npz from
nf_extract.py), in TWO variants with identical architecture + budget:

  RESIDUAL (ours): target = standardized(TRUTH - PAINT), PAINT = v2 physics paint
                   (airfrans_p3_fields_v2).  Inference: pred = PAINT + destd(net).
  SCRATCH (ablation): target = standardized(TRUTH).  Inference: pred = destd(net).

Architecture: Fourier features on (x,y) — 64 freqs, B ~ N(0, sigma=6.0), numpy
seed 0 fixed BEFORE build, identical for both variants (features = [sin(xB),
cos(xB)], no 2*pi factor; sigma=6 => wavelengths ~ chord-scale) — concatenated
with the other 11 features (sdf, nx, ny, is_surf, alpha/15, re_norm, params4/
PSCALE, is5) -> Dense[256 x5] swish -> 4 outputs (ux, uy, p, nut standardized).
Standardization: per-field std from _airfrans_v2_trainstats.npz (SAME stats as
the v2 paint eval and the MARIO table conversion), per-field mean of the TARGET
computed over the training rows here; all constants stored in the weights npz.

Split: the LAST 20 of the 200 sorted scarce_train names are held out entirely
(all 20 present in the npz) -> early sanity standardized MSE on their sampled
rows (surface-enriched sample, NOT full volume — sanity only).

Training: Adam, staged lr 1e-3 x30ep, 3e-4 x30, 1e-4 x20 (80 epochs), batch
8192, full reshuffle each epoch (drop last partial batch).  ~10 s/epoch on
6 P-cores => ~14 min/variant.  Resumable: existing weights npz skips training.

EVAL: the SAME 20 full_test cases as the v2 paint eval (first 20 sorted),
FULL volume (ALL nodes, as MARIO) standardized MSE for ux, uy, p, nut + surface
p MSE (volume-p std) on surface nodes, for both variants, paint-only recomputed
per case as a cross-check against _airfrans_p3_fields_v2.json.  Inference uses
the SAVED npz weights through a numpy forward pass (artifact verified).
-> supervised_field/_nf_eval.json with per-case rows + comparison rows
(v2 paint means, MARIO's published numbers, raw).
"""
import os, json, time, sys
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XFOIL_EXE", os.path.expanduser("~/src/xfoil/bin/xfoil"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AIRFRANS_ROOT",   # external dataset, not shipped with this release
                     os.path.join(os.path.dirname(HERE), "airfrans_data", "Dataset"))
NPZ = os.path.join(HERE, "_nf_train.npz")
from supervised_field import WEIGHTS as _W   # <release>/weights
STATS_NPZ = os.path.join(_W, "airfrans_train_stats.npz")
W_RES = os.path.join(HERE, "_nf_residual.npz")
W_SCR = os.path.join(HERE, "_nf_scratch.npz")
OUT = os.path.join(HERE, "_nf_eval.json")
NU_AIR = 1.56e-5
N_TEST = 20
N_HOLD = 20

# ---- architecture / budget (identical for both variants) ----
N_FREQ = 64
FF_SIGMA = 6.0
FF_SEED = 0
HIDDEN = [256, 256, 256, 256, 256]
BATCH = 8192
STAGES = [(1e-3, 30), (3e-4, 30), (1e-4, 20)]      # 80 epochs
FIELDS = ("ux", "uy", "p", "nut")

# comparison rows for the JSON (raw standardized MSE)
V2_PAINT_ROW = dict(ux=8.0e-2, uy=2.09e-1, p=4.5e-2, nut=5.0e-1, p_surf=1.63e-1)
MARIO_PUBLISHED = dict(ux=1.52e-3, uy=1.13e-3, p=2.4e-3, nut=9.6e-3, p_surf=2.7e-2)


def load_stats():
    z = np.load(STATS_NPZ, allow_pickle=True)
    std = np.array([float(z[k][1]) for k in FIELDS], np.float64)
    mean = np.array([float(z[k][0]) for k in FIELDS], np.float64)
    return mean, std


def fourier_B():
    rng = np.random.default_rng(FF_SEED)            # fixed BEFORE build
    return (rng.standard_normal((2, N_FREQ)) * FF_SIGMA).astype(np.float32)


# ----------------------------------------------------------------------------
# training (TensorFlow, CPU)
# ----------------------------------------------------------------------------
def train_variant(variant, X, Y, hold_mask, std, t_mean, paint_h, truth_h, wfile):
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    nthr = int(os.environ.get("NF_CPU_THREADS", "6"))
    tf.config.threading.set_intra_op_parallelism_threads(nthr)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(0)               # identical init both variants

    B = tf.constant(fourier_B())
    layers = [tf.keras.layers.Dense(h, activation="swish") for h in HIDDEN]
    layers.append(tf.keras.layers.Dense(len(FIELDS)))

    def fwd(x):
        ph = tf.matmul(x[:, :2], B)
        h = tf.concat([tf.sin(ph), tf.cos(ph), x[:, 2:]], axis=1)
        for l in layers:
            h = l(h)
        return h

    opt = tf.keras.optimizers.Adam(STAGES[0][0])

    @tf.function
    def step(xb, yb):
        with tf.GradientTape() as t:
            loss = tf.reduce_mean(tf.square(fwd(xb) - yb))
        tv = sum([l.trainable_variables for l in layers], [])
        opt.apply_gradients(zip(t.gradient(loss, tv), tv))
        return loss

    Xtr = X[~hold_mask]; Ytr = Y[~hold_mask]
    Xh = X[hold_mask]
    n = len(Xtr); spe = n // BATCH
    print(f"[{variant}] train rows {n} (holdout rows {int(hold_mask.sum())}), "
          f"{spe} steps/epoch, batch {BATCH}", flush=True)

    def holdout_sanity(tag):
        # standardized MSE of the PHYSICAL prediction on held-out sampled rows
        pr = np.empty((len(Xh), len(FIELDS)), np.float32)
        for i0 in range(0, len(Xh), 65536):
            pr[i0:i0+65536] = fwd(tf.constant(Xh[i0:i0+65536])).numpy()
        pred = pr * std[None, :] + t_mean[None, :]
        if variant == "residual":
            pred = paint_h + pred
        mse = ((pred - truth_h) / std[None, :]) ** 2
        m = mse.mean(0)
        print(f"[{variant}] holdout sanity ({tag}): " +
              "  ".join(f"{k}={v:.4e}" for k, v in zip(FIELDS, m)), flush=True)
        return {k: float(v) for k, v in zip(FIELDS, m)}

    rng = np.random.default_rng(1234)
    ep = 0; t0 = time.time(); sanity = {}
    for lr_v, n_ep in STAGES:
        opt.learning_rate.assign(lr_v)
        for _ in range(n_ep):
            ep += 1
            perm = rng.permutation(n)
            tl = 0.0
            te0 = time.time()
            for i0 in range(0, spe * BATCH, BATCH):
                idx = perm[i0:i0 + BATCH]
                tl += float(step(tf.constant(Xtr[idx]), tf.constant(Ytr[idx])))
            print(f"[{variant}] ep {ep:3d}/80 lr={lr_v:.0e} "
                  f"loss={tl/spe:.5f}  {time.time()-te0:.1f}s "
                  f"(total {(time.time()-t0)/60:.1f} min)", flush=True)
        sanity = holdout_sanity(f"after lr={lr_v:.0e} stage, ep {ep}")

    wd = {}
    for i, l in enumerate(layers):
        wd[f"W{i}"] = l.kernel.numpy(); wd[f"b{i}"] = l.bias.numpy()
    np.savez(wfile, **wd, fourier_B=B.numpy(), std=std, t_mean=t_mean,
             variant=variant, n_layers=len(layers), n_freq=N_FREQ,
             ff_sigma=FF_SIGMA, ff_seed=FF_SEED, hidden=np.array(HIDDEN),
             fields=np.array(FIELDS), n_train_rows=n,
             holdout_sanity=json.dumps(sanity),
             note="pred_std = net(X); pred_phys = pred_std*std + t_mean"
                  " (+ PAINT if variant=residual). X[:,:2]=(x,y)->Fourier"
                  " [sin(xB),cos(xB)] concat X[:,2:13].")
    print(f"[{variant}] saved {wfile} ({os.path.getsize(wfile)/1e3:.0f} KB), "
          f"total {(time.time()-t0)/60:.1f} min", flush=True)
    return sanity


def np_forward(wz, X, chunk=65536):
    """Numpy forward pass from the saved npz (verifies the artifact)."""
    B = wz["fourier_B"]; nl = int(wz["n_layers"])
    out = np.empty((len(X), len(FIELDS)), np.float32)
    for i0 in range(0, len(X), chunk):
        x = X[i0:i0 + chunk]
        ph = x[:, :2] @ B
        h = np.concatenate([np.sin(ph), np.cos(ph), x[:, 2:]], axis=1)
        for i in range(nl):
            h = h @ wz[f"W{i}"] + wz[f"b{i}"]
            if i < nl - 1:
                h = h * (1.0 / (1.0 + np.exp(-h)))     # swish
        out[i0:i0 + chunk] = h
    return out


# ----------------------------------------------------------------------------
# evaluation on the 20 full_test cases (same as the v2 paint eval)
# ----------------------------------------------------------------------------
def evaluate(std, sanity_res, sanity_scr):
    import airfrans as af
    from supervised_field.airfrans_p3_fields_v2 import (
        case_params, analytic_polyline, paint_points_v2)
    from supervised_field.nf_extract import (
        panel_mids_normals, nearest_idx, PSCALE, A_NORM, RE_LO, RE_HI)
    from supervised_field.airfrans_p1_labels import run_one

    wz_r = np.load(W_RES, allow_pickle=True)
    wz_s = np.load(W_SCR, allow_pickle=True)
    mu_r = np.asarray(wz_r["t_mean"], np.float64)
    mu_s = np.asarray(wz_s["t_mean"], np.float64)

    with open(os.path.join(ROOT, "manifest.json")) as f:
        names = sorted(json.load(f)["full_test"])[:N_TEST]
    tasks = [(case_params(nm)[2], case_params(nm)[1],
              case_params(nm)[0] / NU_AIR) for nm in names]
    print(f"[eval] XFOIL XTR labels for {len(tasks)} test cases (cached)...",
          flush=True)
    with Pool(6) as pool:
        labels = pool.map(run_one, tasks)

    rows = []
    for nm, lab in zip(names, labels):
        U, a, prm = case_params(nm)
        Re = U / NU_AIR
        if lab["cl"] is None:
            rows.append(dict(name=nm, skipped=True, reason="xfoil_xtr_failed"))
            print(f"  SKIP {nm}", flush=True)
            continue
        t0 = time.time()
        sim = af.Simulation(root=ROOT, name=nm)
        P = np.asarray(sim.position, np.float64)
        TR = np.column_stack([np.asarray(sim.velocity, np.float64),
                              np.asarray(sim.pressure, np.float64).reshape(-1, 1),
                              np.asarray(sim.nu_t, np.float64).reshape(-1, 1)])
        sm = np.asarray(sim.surface, bool)

        poly = analytic_polyline(prm)
        r = paint_points_v2(P, poly, a, float(lab["cl"]), Re)
        PA = np.column_stack([r["u"] * U, r["v"] * U,
                              r["cp"] * 0.5 * U**2, r["nut"] * U])
        PA[sm, 0] = 0.0; PA[sm, 1] = 0.0; PA[sm, 3] = 0.0    # v2 convention

        mid, nrm = panel_mids_normals(poly)
        nn = nearest_idx(P, mid)
        prm4 = np.zeros(4); prm4[:len(prm)] = prm
        n_pt = len(P)
        X = np.column_stack([
            P[:, 0], P[:, 1], r["d"], nrm[nn, 0], nrm[nn, 1],
            sm.astype(np.float64),
            np.full(n_pt, a / A_NORM),
            np.full(n_pt, 2.0 * (np.log(Re) - RE_LO) / (RE_HI - RE_LO) - 1.0),
            np.tile(prm4 / PSCALE, (n_pt, 1)),
            np.full(n_pt, float(len(prm) == 4))]).astype(np.float32)

        pred_r = PA + (np_forward(wz_r, X) * std[None, :] + mu_r[None, :])
        pred_s = np_forward(wz_s, X) * std[None, :] + mu_s[None, :]

        def mses(pred):
            e = ((pred - TR) / std[None, :]) ** 2
            d = {k: float(v) for k, v in zip(FIELDS, e.mean(0))}
            d["p_surf"] = float(np.mean(((pred[sm, 2] - TR[sm, 2]) / std[2]) ** 2))
            return d

        m_pa, m_r, m_s = mses(PA), mses(pred_r), mses(pred_s)
        rows.append(dict(name=nm, U=U, alpha=a, Re=Re, n_nodes=n_pt,
                         n_surf=int(sm.sum()), skipped=False,
                         cl_xtr=float(lab["cl"]),
                         mse_paint=m_pa, mse_residual=m_r, mse_scratch=m_s))
        print(f"  {nm[:46]:46s} a={a:+6.2f} | p: paint {m_pa['p']:.3e} res "
              f"{m_r['p']:.3e} scr {m_s['p']:.3e} | ux: {m_pa['ux']:.3e} "
              f"{m_r['ux']:.3e} {m_s['ux']:.3e} | nut: {m_pa['nut']:.3e} "
              f"{m_r['nut']:.3e} {m_s['nut']:.3e} [{time.time()-t0:.0f}s]",
              flush=True)

    ok = [r for r in rows if not r["skipped"]]
    keys = list(FIELDS) + ["p_surf"]

    def agg(tag):
        return {k: float(np.mean([r[tag][k] for r in ok])) for k in keys}

    means = dict(paint=agg("mse_paint"), residual=agg("mse_residual"),
                 scratch=agg("mse_scratch"))
    out = dict(
        n_evaluated=len(ok), n_requested=len(names),
        test_cases="first 20 sorted full_test (same as v2 paint eval)",
        stats_npz=os.path.basename(STATS_NPZ),
        arch=dict(fourier=dict(n_freq=N_FREQ, sigma=FF_SIGMA, seed=FF_SEED,
                               form="[sin(xB), cos(xB)], no 2*pi"),
                  hidden=HIDDEN, act="swish", batch=BATCH,
                  stages=[list(s) for s in STAGES],
                  train_npz=os.path.basename(NPZ),
                  holdout="last 20 sorted scarce_train names (all in npz)"),
        mse_std_means=means,
        comparison=dict(
            v2_paint_published=V2_PAINT_ROW,
            mario_published=MARIO_PUBLISHED,
            note="raw standardized MSE; MARIO printed-table conversion: "
                 "x1e-2 for ux/uy/p, x1e-1 for nut/p_surf. v2_paint_published "
                 "= rounded means of _airfrans_p3_fields_v2.json; mse_paint "
                 "here recomputes them per case (cross-check)."),
        holdout_sanity=dict(residual=sanity_res, scratch=sanity_scr,
                            note="standardized MSE on the surface-enriched "
                                 "SAMPLED rows of the 20 held-out train cases "
                                 "(not full volume)"),
        cases=rows,
        eval_protocol="full volume, ALL nodes (MARIO protocol), standardized "
                      "with _airfrans_v2_trainstats stds; p_surf on surface "
                      "nodes with the volume-p std; predictions are raw model "
                      "output (no surface zeroing at eval).")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("\n[eval] means over", len(ok), "cases:")
    for tag in ("paint", "residual", "scratch"):
        print(f"  {tag:8s} " + "  ".join(f"{k}={means[tag][k]:.4e}" for k in keys),
              flush=True)
    print(f"[eval] -> {OUT}", flush=True)
    return out


def main():
    z = np.load(NPZ, allow_pickle=True)
    X = z["X"]; PAINT = z["PAINT"]; TRUTH = z["TRUTH"]; cid = z["case_id"]
    names = [str(n) for n in z["names"]]
    with open(os.path.join(ROOT, "manifest.json")) as f:
        all_train = sorted(json.load(f)["scarce_train"])
    hold_names = set(all_train[-N_HOLD:])
    hold_ids = np.array([i for i, n in enumerate(names) if n in hold_names])
    hold_mask = np.isin(cid, hold_ids)
    print(f"[nf_train] rows {len(X)}, cases {len(names)}, holdout cases "
          f"{len(hold_ids)} ({int(hold_mask.sum())} rows)", flush=True)

    _, std = load_stats()
    print("[nf_train] stds:", {k: float(s) for k, s in zip(FIELDS, std)}, flush=True)

    sanities = {}
    for variant, wfile in (("residual", W_RES), ("scratch", W_SCR)):
        if os.path.exists(wfile):
            wz = np.load(wfile, allow_pickle=True)
            sanities[variant] = json.loads(str(wz["holdout_sanity"]))
            print(f"[{variant}] weights exist ({wfile}) — skip training. "
                  f"sanity={sanities[variant]}", flush=True)
            continue
        if variant == "residual":
            R = (TRUTH - PAINT).astype(np.float64)
        else:
            R = TRUTH.astype(np.float64)
        t_mean = R[~hold_mask].mean(0)
        Y = ((R - t_mean[None, :]) / std[None, :]).astype(np.float32)
        print(f"[{variant}] target mean={t_mean}, "
              f"std-target var={Y[~hold_mask].var(0)}", flush=True)
        sanities[variant] = train_variant(
            variant, X, Y, hold_mask, std, t_mean,
            PAINT[hold_mask].astype(np.float64),
            TRUTH[hold_mask].astype(np.float64), wfile)
        del Y, R

    del X, PAINT, TRUTH
    evaluate(std, sanities["residual"], sanities["scratch"])


if __name__ == "__main__":
    main()
