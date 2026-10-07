# -*- coding: utf-8 -*-
"""nf_train_shape.py — geometry-conditioned field operator (DESIGN_geomcond.md winner).

Production non-residual concat_ms field MLP, but the NACA params4+is5 slot of the
condition vector is REPLACED by a LEARNED shape latent z_s(16) from a tiny PointNet
encoder over the REAL surface point cloud.  The encoder trains END-TO-END inside the
TF step (per batch: run encoder on ALL case tokens -> z_all, gather to rows by case_id)
— the mandatory repair vs a frozen offline encoder.

Self-contained fork (concat / non-residual / multiscale only) so the production
nf_train_v5 trainer is untouched.

  python -m supervised_field.nf_train_shape --npz _nf_train_v5_full800.npz \
      --tokens _nf_shape_tokens_airfrans.npz --out-tag s0 [--seed 0] [--gpu] [--dry N] [--init W.npz]
"""
import os, sys, json, time, hashlib
os.environ.setdefault("MPLBACKEND", "Agg")
if "--gpu" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
from supervised_field.nf_train import load_stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AIRFRANS_ROOT",   # external dataset, not shipped with this release
                     os.path.join(os.path.dirname(HERE), "airfrans_data", "Dataset"))
N_HOLD = 20
N_FREQ = 64
FF_SEED = 0
FF_BANDS = [1.5, 3.0, 6.0, 12.0]
TRUNK_WIDTH = 248
N_TRUNK = 5
NUT_HID = 64
OB = 50.0                  # non-residual bound (v5 PROBE)
WEIGHT_DECAY = 1e-5
BATCH = 8192
STAGES = [(1e-3, 30), (3e-4, 30), (1e-4, 20)]
# env override for warm-start / big-data runs, e.g. NF_STAGES="5e-4:10,2e-4:10,1e-4:8"
if os.environ.get("NF_STAGES"):
    STAGES = [(float(p.split(":")[0]), int(p.split(":")[1]))
              for p in os.environ["NF_STAGES"].split(",")]
W_NUT = 0.5
CURRICULUM_EPOCHS = 10
Z_DIM = 16
M_TOK = 256
FIELDS = ("ux", "uy", "p", "nut")


def multiscale_fourier_B():
    rng = np.random.default_rng(FF_SEED)
    per = N_FREQ // len(FF_BANDS)
    blocks = [rng.standard_normal((2, per)) * s for s in FF_BANDS]
    return np.concatenate(blocks, axis=1).astype(np.float32)   # (2,64)


def main():
    def arg(flag, default=None):
        return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default
    npz_path = os.path.join(HERE, arg("--npz", "_nf_train_v5_full800.npz"))
    from supervised_field import WEIGHTS as _W
    tok_path = (os.path.join(HERE, arg("--tokens")) if "--tokens" in sys.argv
                else os.path.join(_W, "airfrans_surface_tokens.npz"))   # release name
    out_tag = "_" + str(arg("--out-tag", "s0")).lstrip("_")
    dry = int(arg("--dry", "0"))
    init_path = arg("--init", None)
    src_nw_dw = float(arg("--src-nw-dw", "1.0"))   # graft1: down-weight 2nd-source near-wall rows (two-corpus runs)
    npz2_path = os.path.join(HERE, arg("--npz2")) if "--npz2" in sys.argv else None  # optional second corpus
    tok2_path = os.path.join(HERE, arg("--tokens2")) if "--tokens2" in sys.argv else None
    uni_holdout = int(arg("--uni-holdout", "0"))   # drop last K cases of npz2 (kept for non-NACA gate)
    w_uy = float(arg("--w-uy", "1.0"))             # uy lever: loss weight on uy channel
    zdim = int(arg("--z-dim", "16"))               # capacity lever: shape latent dim
    encw = int(arg("--enc-width", "32"))           # capacity lever: encoder hidden width
    no_paint = "--no-paint" in sys.argv            # ablation: zero the physics-paint feature
    WFILE = os.path.join(HERE, "_nf_shape" + out_tag + ".npz")
    t_all = time.time()

    # ---- data loader (1 or 2 datasets, in-loader merge to avoid 1.25GB transfer) ----
    def load_one(npzp, tokp, drop_last=0):
        z = np.load(npzp, allow_pickle=True)
        X = z["X"]; PA = z["PAINT"]; TR = z["TRUTH"]; cid = z["case_id"].astype(np.int64)
        gb = np.asarray(z["g_band"], np.float32); nms = [str(x) for x in z["names"]]
        sc = np.asarray(z["s"], np.float32); wc = np.asarray(z["w_case"], np.float32)
        if drop_last > 0:                            # keep first (ncase-drop_last) cases
            keepc = len(nms) - drop_last
            rowm = cid < keepc
            X, PA, TR, cid, gb = X[rowm], PA[rowm], TR[rowm], cid[rowm], gb[rowm]
            nms, sc, wc = nms[:keepc], sc[:keepc], wc[:keepc]
        tz = np.load(tokp, allow_pickle=True)
        tok = tz["surf_tokens"].astype(np.float32); tnm = [str(x) for x in tz["names"]]
        tmap = {nm: i for i, nm in enumerate(tnm)}
        st = np.stack([tok[tmap[nm]] for nm in nms]).astype(np.float32)
        return X, PA, TR, cid, gb, nms, sc, wc, st

    X, PAINT, TRUTH, cid, g_band, names, s_case, wcase_case, surf_tok = load_one(npz_path, tok_path)
    src_case = np.zeros(len(names), np.int32)
    if npz2_path is not None:
        X2, PA2, TR2, cid2, gb2, nm2, sc2, wc2, st2 = load_one(npz2_path, tok2_path, uni_holdout)
        off = len(names)
        X = np.concatenate([X, X2]); PAINT = np.concatenate([PAINT, PA2])
        TRUTH = np.concatenate([TRUTH, TR2]); g_band = np.concatenate([g_band, gb2])
        cid = np.concatenate([cid, cid2 + off])
        names = names + nm2; s_case = np.concatenate([s_case, sc2])
        wcase_case = np.concatenate([wcase_case, wc2])
        src_case = np.concatenate([src_case, np.ones(len(nm2), np.int32)])
        surf_tok = np.concatenate([surf_tok, st2])
        print(f"[shape] TWO CORPORA: ds1={off} cases + ds2={len(nm2)} cases "
              f"(dropped last {uni_holdout} of ds2 for non-NACA gate)", flush=True)
    assert surf_tok.shape == (len(names), M_TOK, 2), surf_tok.shape
    print(f"[shape] npz={os.path.basename(npz_path)} rows={len(X)} cases={len(names)} "
          f"src1={int((src_case==1).sum())} tokens={surf_tok.shape}", flush=True)

    # holdout = last N_HOLD scarce_train names (same as v5); absent -> none
    try:
        with open(os.path.join(ROOT, "manifest.json")) as f:
            all_train = sorted(json.load(f)["scarce_train"])
        hold_names = set(all_train[-N_HOLD:])
    except Exception:
        hold_names = set()
    hold_ids = np.array([i for i, nm in enumerate(names) if nm in hold_names])
    hold_mask = np.isin(cid, hold_ids) if len(hold_ids) else np.zeros(len(X), bool)
    train_idx = np.where(~hold_mask)[0].astype(np.int64)
    n = len(train_idx)
    _, std = load_stats()
    print(f"[shape] holdout cases {len(hold_ids)} ({int(hold_mask.sum())} rows), "
          f"train rows {n}; stds={dict(zip(FIELDS, np.round(std,5)))}", flush=True)

    s_row = s_case[cid].astype(np.float32)
    nw_weight = float(arg("--nw-weight", "1.0"))   # probe: up-weight near-wall rows in the loss
    # per-row weight: w_case * (src near-wall down-weight, graft1) * (near-wall up-weight)
    near_wall = (X[:, 5] <= 0.5) & (X[:, 2] > 0.0) & (X[:, 2] < 0.05)
    dw = np.where((src_case[cid] == 1) & near_wall, np.float32(src_nw_dw), np.float32(1.0))
    uw = np.where(near_wall, np.float32(nw_weight), np.float32(1.0))
    wrow = (wcase_case[cid] * dw * uw).astype(np.float32)
    if nw_weight != 1.0:
        print(f"[shape] --nw-weight {nw_weight}: near-wall rows up-weighted ({int(near_wall.sum())} rows)", flush=True)

    # ---- paint robust-std (== v5 non-res) ----
    Praw = PAINT.astype(np.float64)
    tr_sel = (~hold_mask) if n else np.ones(len(PAINT), bool)
    med = np.median(Praw[tr_sel], 0); q1 = np.percentile(Praw[tr_sel], 25, 0)
    q3 = np.percentile(Praw[tr_sel], 75, 0); iqr = q3 - q1
    clo = med - 5.0 * iqr; chi = med + 5.0 * iqr
    paint_clip = np.stack([clo, chi], 1)
    Pclip = np.clip(Praw, clo[None, :], chi[None, :])
    paint_mean = Pclip[tr_sel].mean(0); paint_std = Pclip[tr_sel].std(0) + 1e-9
    PAINT_FEAT = ((Pclip - paint_mean[None, :]) / paint_std[None, :]).astype(np.float32)
    if no_paint:                                     # ablation: paint carries no info (train & eval both zero)
        PAINT_FEAT = np.zeros_like(PAINT_FEAT)
        print("[shape] --no-paint: physics-paint feature ZEROED (paint-ablation run)", flush=True)

    # ---- target (non-res full field) ----
    R = TRUTH.astype(np.float64).copy()
    t_mean = R[~hold_mask].mean(0) if n else R.mean(0)
    Y = ((R - t_mean[None, :]) / std[None, :]).astype(np.float32)
    del R
    print(f"[shape] t_mean={np.round(t_mean,4)}  Y var={np.round(Y[tr_sel].var(0),3)}", flush=True)

    # ---- TF model ----
    import tensorflow as tf
    if "--gpu" in sys.argv:
        gpus = tf.config.list_physical_devices("GPU")
        assert gpus, "no GPU"
        for gp in gpus:
            tf.config.experimental.set_memory_growth(gp, True)
        print(f"[shape] GPU on: {len(gpus)} dev", flush=True)
    # --seed S : robustness re-seed of the Keras variable initialiser -- the ONLY previously
    # unpinned source of run-to-run variation here (FF_SEED stays 0 so the Fourier embedding
    # is identical; the batch-shuffle RandomState(0) and the offline resample_le(seed=0)
    # tokens were already fixed).  Default 0.  Must be set BEFORE the Dense layers below.
    TRAIN_SEED = int(arg("--seed", "0"))
    tf.keras.utils.set_random_seed(TRAIN_SEED)
    print(f"[shape] --seed {TRAIN_SEED} (initialiser); FF_SEED={FF_SEED}, shuffle RandomState(0)", flush=True)
    B = tf.constant(multiscale_fourier_B())
    TOK = tf.constant(surf_tok)                      # (n_cases, M, 2) captured

    enc = [tf.keras.layers.Dense(encw, activation="swish"),
           tf.keras.layers.Dense(encw, activation="swish"),
           tf.keras.layers.Dense(encw, activation="swish"),
           tf.keras.layers.Dense(zdim, activation=None)]
    trunk = [tf.keras.layers.Dense(TRUNK_WIDTH, activation=None) for _ in range(N_TRUNK)]
    vp_head = [tf.keras.layers.Dense(3, activation=None)]
    nut_head = [tf.keras.layers.Dense(NUT_HID, activation="swish"),
                tf.keras.layers.Dense(1, activation=None)]
    all_layers = enc + trunk + vp_head + nut_head

    def encode(tokens):                              # (K,M,2) -> (K,Z_DIM)
        h = enc[0](tokens); h = enc[1](h)
        h = tf.reduce_max(h, axis=1)                 # global max-pool over points
        h = enc[2](h); z_s = enc[3](h)
        return z_s

    def fwd(xb, z_row):
        """xb = [X(13), s(1), paint4(4)] (18).  z_row = (B,Z_DIM)."""
        ph = tf.matmul(xb[:, :2], B)
        spatial = tf.concat([tf.sin(ph), tf.cos(ph), xb[:, 2:6]], axis=1)   # 128+4
        c = tf.concat([xb[:, 6:8], z_row, xb[:, 13:14], xb[:, 14:18]], axis=1)  # a/re, z16, s, paint4
        h = tf.concat([spatial, c], axis=1)
        for l in trunk:
            h = tf.nn.swish(l(h))
        vp = vp_head[0](h)
        nh = h
        for l in nut_head:
            nh = l(nh)
        return OB * tf.tanh(tf.concat([vp, nh], axis=1) / OB)

    def assemble(idx):
        return np.concatenate([X[idx], s_row[idx, None], PAINT_FEAT[idx]], axis=1).astype(np.float32)

    # build vars
    xprobe = assemble(train_idx[:min(BATCH, n)])
    zr = encode(tf.constant(surf_tok[cid[train_idx[:min(BATCH, n)]]]))
    _ = fwd(tf.constant(xprobe), zr)
    tv = sum([l.trainable_variables for l in all_layers], [])
    nparm = int(sum(int(np.prod(v.shape)) for v in tv))
    enc_parm = int(sum(int(np.prod(v.shape)) for l in enc for v in l.trainable_variables))
    print(f"[shape] PARAMS total={nparm} (encoder={enc_parm}, trunk x{N_TRUNK}@{TRUNK_WIDTH}, "
          f"c=[a,re,z{zdim},s,paint4]={2+zdim+1+4}d w_uy={w_uy:g} enc_w={encw}, OB={OB:g})", flush=True)

    if init_path is not None:
        iz = np.load(os.path.join(HERE, init_path), allow_pickle=True)
        nok = 0
        for pfx, layers in [("enc", enc), ("trunk", trunk), ("vp", vp_head), ("nut", nut_head)]:
            for i, l in enumerate(layers):
                if f"{pfx}_W{i}" in iz:
                    l.kernel.assign(iz[f"{pfx}_W{i}"]); l.bias.assign(iz[f"{pfx}_b{i}"]); nok += 1
        print(f"[shape] --init {init_path}: {nok} layers", flush=True)

    try:
        opt = tf.keras.optimizers.AdamW(learning_rate=STAGES[0][0], weight_decay=WEIGHT_DECAY)
    except (AttributeError, TypeError):
        opt = tf.keras.optimizers.experimental.AdamW(
            learning_rate=STAGES[0][0], weight_decay=WEIGHT_DECAY, jit_compile=False)
    w_nut_t = tf.constant(W_NUT, tf.float32)
    w_uy_t = tf.constant(w_uy, tf.float32)

    @tf.function
    def step(xb, yb, wb, cidb):
        with tf.GradientTape() as t:
            z_all = encode(TOK)                      # (n_cases, Z_DIM) — END-TO-END
            z_row = tf.gather(z_all, cidb)
            pred = fwd(xb, z_row)
            sq = tf.square(pred - yb)
            per = sq[:, 0] + w_uy_t * sq[:, 1] + sq[:, 2] + w_nut_t * sq[:, 3]
            loss = tf.reduce_mean(wb * per)
        grads = t.gradient(loss, tv)
        opt.apply_gradients(zip(grads, tv))
        # encoder grad norm (GATE 0: must be > 0 => end-to-end learning)
        enc_tv = sum([l.trainable_variables for l in enc], [])
        eg = tf.linalg.global_norm([g for g, v in zip(grads, tv)
                                    if any(v is e for e in enc_tv)])
        return loss, eg

    # ---- GATE 0 / dry ----
    if dry > 0:
        idx = train_idx[:BATCH]
        xb = tf.constant(assemble(idx)); yb = tf.constant(Y[idx])
        wb = tf.constant(wrow[idx]); cidb = tf.constant(cid[idx])
        lv, eg = step(xb, yb, wb, cidb)
        t0 = time.time()
        for _ in range(dry):
            lv, eg = step(xb, yb, wb, cidb)
        ms = (time.time() - t0) / dry * 1000
        print(f"[shape] DRY {dry}-step: {ms:.0f} ms/step  loss={float(lv):.5f}  "
              f"ENCODER grad-norm={float(eg):.4e}  (must be >0 => end-to-end)", flush=True)
        print(f"[shape] GATE0: {'PASS' if np.isfinite(float(lv)) and float(eg) > 0 else 'FAIL'}", flush=True)
        return

    # ---- full train ----
    is_surf = X[:, 5] > 0.5
    nwall = (~is_surf) & (X[:, 2] > 0.0) & (X[:, 2] < 0.05)
    curric_train = train_idx[(is_surf | nwall)[train_idx]]
    print(f"[shape] curriculum ep1-{CURRICULUM_EPOCHS}: {len(curric_train)}/{n} rows", flush=True)

    Xh = X[hold_mask]; sh = s_row[hold_mask]; pfh = PAINT_FEAT[hold_mask]
    th = TRUTH[hold_mask].astype(np.float64); hcid = cid[hold_mask]

    def holdout(tag):
        if not len(hcid):
            return
        pr = np.empty((len(Xh), 4), np.float32)
        z_all = encode(TOK).numpy()
        for i0 in range(0, len(Xh), 65536):
            sl = slice(i0, i0 + 65536)
            xb = np.concatenate([Xh[sl], sh[sl, None], pfh[sl]], 1).astype(np.float32)
            zr = z_all[hcid[sl]]
            pr[sl] = fwd(tf.constant(xb), tf.constant(zr)).numpy()
        pred = pr * std[None, :] + t_mean[None, :]
        e = ((pred - th) / std[None, :]) ** 2
        nexp = sum(1 for ci in np.unique(hcid)
                   if e[hcid == ci, 0].mean() > 1 or e[hcid == ci, 1].mean() > 1)
        print(f"[shape] holdout ({tag}): " + "  ".join(f"{k}={v:.4e}" for k, v in zip(FIELDS, e.mean(0)))
              + f"  | explosions {nexp}/{len(np.unique(hcid))}", flush=True)

    def sha256(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()

    manifest = os.path.join(ROOT, "manifest.json")
    # the recipe itself, stored IN the checkpoint so a run is attributable after the fact
    # (epochs / LR schedule / loss weights / corpus path used to live only in the invocation).
    RUN_CFG = dict(
        train_seed=TRAIN_SEED,
        seed_covers="tf.keras.utils.set_random_seed -> python/numpy-global/TF RNG = Dense kernel+bias init",
        seed_excludes=f"FF_SEED={FF_SEED} Fourier B (np.random.default_rng, independent); "
                      "batch order np.random.RandomState(0); surface tokens resample_le(seed=0), "
                      "resampled offline into the tokens npz",
        argv=sys.argv[1:],
        code_sha256=sha256(os.path.abspath(__file__)),
        corpus=npz_path, corpus_sha256=sha256(npz_path),
        tokens=tok_path, tokens_sha256=sha256(tok_path),
        corpus2=npz2_path, corpus2_sha256=sha256(npz2_path) if npz2_path else None,
        tokens2=tok2_path, tokens2_sha256=sha256(tok2_path) if tok2_path else None,
        uni_holdout=uni_holdout,
        split=f"train = all rows minus the last {N_HOLD} sorted scarce_train case names",
        manifest=manifest, manifest_sha256=sha256(manifest) if os.path.exists(manifest) else None,
        holdout_names=sorted(names[i] for i in hold_ids), holdout_cases=len(hold_ids),
        holdout_rows=int(hold_mask.sum()), rows=len(X), cases=len(names), train_rows=int(n),
        stages=[[lr, ne] for lr, ne in STAGES], nf_stages_env=os.environ.get("NF_STAGES"),
        epochs_planned=int(sum(ne for _, ne in STAGES)), batch=BATCH,
        weight_decay=WEIGHT_DECAY, optimizer="AdamW", w_nut=W_NUT, w_uy=w_uy,
        nw_weight=nw_weight, src_nw_dw=src_nw_dw, curriculum_epochs=CURRICULUM_EPOCHS,
        z_dim=zdim, enc_width=encw, m_tok=M_TOK, ob=OB, no_paint=bool(no_paint),
        trunk_width=TRUNK_WIDTH, n_trunk=N_TRUNK, nut_hid=NUT_HID,
        n_freq=N_FREQ, ff_seed=FF_SEED, ff_bands=FF_BANDS, init=init_path,
        params_total=nparm, params_encoder=enc_parm,
        tf_version=tf.__version__, gpu_flag="--gpu" in sys.argv,   # the one that decides the device
        gpu_visible=bool(tf.config.list_physical_devices("GPU")),  # Mac lists Metal even at CVD=-1
        devices=[d.name for d in tf.config.list_physical_devices()],
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        nf_cpu_threads=os.environ.get("NF_CPU_THREADS"),
    )

    def dump_weights(path, **extra):
        """Save full weight set. Atomic (tmp+rename) so a power cut mid-write
        never leaves a corrupt file. Checkpoints load via --init unchanged."""
        wd = {}
        for pfx, layers in [("enc", enc), ("trunk", trunk), ("vp", vp_head), ("nut", nut_head)]:
            for i, l in enumerate(layers):
                wd[f"{pfx}_W{i}"] = l.kernel.numpy(); wd[f"{pfx}_b{i}"] = l.bias.numpy()
        tmp = path + ".tmp.npz"
        np.savez(tmp, **wd, fourier_B=B.numpy(), std=std, t_mean=t_mean,
                 paint_mean=paint_mean, paint_std=paint_std, paint_clip=paint_clip,
                 z_dim=zdim, m_tok=M_TOK, ob=OB, no_paint=np.int32(1 if no_paint else 0),
                 c_layout="alpha/15,re_norm,z_s(16),s,paint_ux,paint_uy,paint_p,paint_nut(robust-std)",
                 variant="shape_encoder_concat_ms_nonres",
                 train_seed=np.int64(TRAIN_SEED),
                 run_config=json.dumps({**RUN_CFG, "wall_s": round(time.time() - t_all, 1)}),
                 **extra)
        os.replace(tmp, path)

    CKPT = WFILE.replace(".npz", "_ckpt.npz")
    plan = [lr for lr, ne in STAGES for _ in range(ne)]
    rng = np.random.RandomState(0)
    for ep, lr in enumerate(plan, 1):
        opt.learning_rate.assign(lr)
        pool = curric_train if ep <= CURRICULUM_EPOCHS else train_idx
        order = rng.permutation(pool)
        tot = 0.0; nb = 0; t0 = time.time()
        for i0 in range(0, len(order), BATCH):
            idx = order[i0:i0 + BATCH]
            lv, _ = step(tf.constant(assemble(idx)), tf.constant(Y[idx]),
                         tf.constant(wrow[idx]), tf.constant(cid[idx]))
            tot += float(lv); nb += 1
        print(f"[shape] ep {ep:3d}/{len(plan)} lr={lr:.0e} loss={tot/nb:.5f} "
              f"({len(order)} rows, {nb} steps) {time.time()-t0:.1f}s", flush=True)
        if ep in (1, CURRICULUM_EPOCHS, 20, 40, len(plan)):
            holdout(f"ep{ep}")
        dump_weights(CKPT, ckpt_epoch=np.int64(ep))   # per-epoch resumable checkpoint

    # ---- save ----
    dump_weights(WFILE, ckpt_epoch=np.int64(len(plan)))   # epochs COMPLETED, in the final file too
    print(f"[shape] wrote {WFILE} ({time.time()-t_all:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
