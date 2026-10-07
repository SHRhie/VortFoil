# -*- coding: utf-8 -*-
"""supervised_field/nf_eval200_v5.py — CITABLE full_test evaluation of the NF
v5-FiLM RANS-supervised geometry-conditioned residual neural field (and its
§9.2 capacity-matched concat-control) on ALL 200 AirfRANS full_test cases
(full volume, ALL mesh nodes — MARIO protocol).  NEW file; nf_eval200_v0.py
untouched, nf_eval200.py helpers imported.

Per case:
  1. cl for the v2 paint: XFOIL XTR label via airfrans_p1_labels.run_one
     (md5-cached); ensemble-head cl fallback (best_weights/airfrans_head_ens_v2.npz)
     when XFOIL fails — documented per case (same as v0/v1 harness).
  2. v2 paint at all nodes (paint_points_v2), v2 surface convention.
  3. v5 net numpy forward from the saved npz:
       - filmforward_v5  : Fourier(x,y) + geom concat -> trunk Dense208 x5 with FiLM
         (g=1+tanh, beta) on layers {1,2,3} from the hypernet (c 8-dim, last col=s),
         plain swish on {0,4}; vp-head Dense(3) OR v4 basis-remix branch (readout
         flag in the npz); SEPARATE nut head Dense(64,1); out=8*tanh(z/8) both heads.
       - concatforward_v5 : trunk Dense248 x5 swish with c(8-dim) concatenated to
         the 140-dim trunk input (no hypernet), same heads.
  4. §4.4 reconstruction (per-case):
       pred[:,0:3] = g_band*PA[:,0:3] + (net[:,0:3]*std[:3] + t_mean[:3])
       pred[:,3]   =                    net[:,3]  *std[3]  + t_mean[3]   # nut, NO paint
     g_band (alpha-independent paint Drec/wake_band) and s (XFOIL-Cf<0 fraction,
     alpha fallback on XFOIL fail) recomputed per case with the TRAIN recipe.
  5. Standardized MSE (stds from _airfrans_v2_trainstats.npz; wz['std'] allclose
     assert) per field over the full volume + surface-p MSE.  Paint-only recorded.

Aggregates: all-200 means, alpha<10 / alpha>=10 splits, medians, explosion count
(case ux or uy standardized MSE > 1).  4-way side-by-side in the JSON:
v0 (reference line), concat-control (FiLM isolation line), FiLM, MARIO.
-> supervised_field/_nf_eval200_v5.json
"""
import os, json, time, sys
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XFOIL_EXE", os.path.expanduser("~/src/xfoil/bin/xfoil"))
import numpy as np
from multiprocessing import Pool

from supervised_field.nf_eval200 import (
    head_feats, ens_cl, HERE, ROOT, STATS_NPZ, ENS_PATHS,
    NU_AIR, FIELDS, KEYS, MARIO_PUBLISHED)

W_FILM = os.path.join(HERE, "_nf_film_v5.npz")
W_CONCAT = os.path.join(HERE, "_nf_concat_v5.npz")
W_V0_JSON = os.path.join(HERE, "_nf_eval200_v0.json")
OUT = os.path.join(HERE, "_nf_eval200_v5.json")

# g_band geometric gate constants (§4.3; identical to extractor/trainer)
CELL = 0.0092
RE_DELTA_X = 0.02


# ===========================================================================
# numpy forwards (artifact-verified) — mirror nf_train_v5.fwd EXACTLY
# ===========================================================================
def _swish(h):
    return h * (1.0 / (1.0 + np.exp(-h)))


def _trunk_film(wz, feat, c, tw, film_layers, n_trunk):
    """FiLM-modulated trunk forward.  feat = [sin,cos,geom4] (132-dim); c = 8-dim
    condition.  Returns h_4 (tw-dim)."""
    # hypernet: c -> Dense64 swish -> Dense64 swish -> Dense(hyper_out) linear
    hh = c
    n_hyp = sum(1 for k in wz.files if k.startswith("hyper_W"))
    for j in range(n_hyp):
        hh = hh @ wz[f"hyper_W{j}"] + wz[f"hyper_b{j}"]
        if j < n_hyp - 1:
            hh = _swish(hh)
    gb = hh.reshape(-1, len(film_layers), tw, 2)
    h = feat
    for li in range(n_trunk):
        pre = h @ wz[f"trunk_W{li}"] + wz[f"trunk_b{li}"]
        if li in film_layers:
            fi = list(film_layers).index(li)
            g = 1.0 + np.tanh(gb[:, fi, :, 0])
            beta = gb[:, fi, :, 1]
            h = _swish(g * pre + beta)
        else:
            h = _swish(pre)
    return h


def _trunk_concat(wz, feat, c, n_trunk):
    """Concat trunk forward.  feat = [sin,cos,geom4] (132); concat c (8) -> 140."""
    h = np.concatenate([feat, c], axis=1)
    for li in range(n_trunk):
        h = _swish(h @ wz[f"trunk_W{li}"] + wz[f"trunk_b{li}"])
    return h


def _heads(wz, h4, c, readout):
    """vp-head (dense or basis-remix) + SEPARATE nut head; OB=8 tanh on both."""
    OB = float(wz["out_bound"])
    if readout == "dense":
        vp = h4 @ wz["vp_W0"] + wz["vp_b0"]                     # (N,3)
    else:  # basisremix graft: branch(c) -> coeffs C (NB x 3) + bias
        NB = int(wz["basis"])
        cc = c
        n_br = sum(1 for k in wz.files if k.startswith("branch_W"))
        for j in range(n_br):
            cc = cc @ wz[f"branch_W{j}"] + wz[f"branch_b{j}"]
            if j < n_br - 1:
                cc = _swish(cc)
        C = cc[:, :NB * 3].reshape(-1, NB, 3)
        bias = cc[:, NB * 3:]
        vp = np.einsum("ni,nif->nf", h4, C) + bias
    nh = h4
    n_nut = sum(1 for k in wz.files if k.startswith("nut_W"))
    for j in range(n_nut):
        nh = nh @ wz[f"nut_W{j}"] + wz[f"nut_b{j}"]
        if j < n_nut - 1:
            nh = _swish(nh)
    raw = np.concatenate([vp, nh], axis=1)                      # (N,4)
    return OB * np.tanh(raw / OB)


def nf_forward_v5(wz, X, s_row, chunk=65536):
    """Full v5 numpy forward (film or concat).  X = (N,13); s_row = (N,) per-row
    separation scalar (the case-level s broadcast).  Returns standardized (N,4)
    in (ux,uy,p,nut), already OB*tanh-bounded."""
    B = wz["fourier_B"]
    cond = str(wz["cond"]); readout = str(wz["readout"])
    tw = int(wz["trunk_width"]); n_trunk = int(wz["n_trunk"])
    film_layers = [int(x) for x in np.asarray(wz["film_layers"])] \
        if "film_layers" in wz.files else [1, 2, 3]
    out = np.empty((len(X), len(FIELDS)), np.float32)
    with np.errstate(all="ignore"):     # suppress spurious small-float32 BLAS warns
        for i0 in range(0, len(X), chunk):
            x = X[i0:i0 + chunk]
            ph = x[:, :2] @ B
            feat = np.concatenate([np.sin(ph), np.cos(ph), x[:, 2:6]], axis=1)  # 132
            # c = [alpha/15, re_norm, p0..p3, is5, s]  (X[:,6:13] + s)
            c = np.concatenate([x[:, 6:13], s_row[i0:i0 + chunk, None]], axis=1)
            if cond == "film":
                h4 = _trunk_film(wz, feat, c, tw, film_layers, n_trunk)
            else:
                h4 = _trunk_concat(wz, feat, c, n_trunk)
            out[i0:i0 + chunk] = _heads(wz, h4, c, readout)
    return out


# ===========================================================================
# per-case s (XFOIL-Cf, alpha fallback) + g_band (paint Drec/wake_band)
# ===========================================================================
def sep_scalar_xfoil(prm, alpha, Re):
    """s = clip(frac(Cf<0)/0.10, 0, 1) from blband_xfoil.bl_dump (md5-cached);
    XFOIL fail -> monotone alpha fallback (§3.3).  SAME source train+eval."""
    from supervised_field.blband_xfoil import bl_dump
    try:
        bl = bl_dump(prm, alpha, Re)
        cf = np.asarray(bl["cf"], np.float64)
        f_sep = float(np.mean(cf < 0.0)) if len(cf) else 0.0
        return float(np.clip(f_sep / 0.10, 0.0, 1.0)), True
    except Exception:
        return float(np.clip((abs(alpha) - 6.0) / 8.0, 0.0, 1.0)), False


def gband(d, wake_band, x, Re):
    """alpha-independent geometric g_band (§4.3): 1 inside 0<d<Drec and NOT wake."""
    xq = np.maximum(x, RE_DELTA_X)
    delta = 0.37 * xq / (Re * xq) ** 0.2
    Drec = np.maximum(delta, 1.5 * CELL)
    return ((d > 0.0) & (d < Drec) & (~wake_band)).astype(np.float32)


def main():
    import airfrans as af
    from supervised_field.airfrans_p3_fields_v2 import (
        case_params, analytic_polyline, paint_points_v2)
    from supervised_field.nf_extract import (
        panel_mids_normals, nearest_idx, PSCALE, A_NORM, RE_LO, RE_HI)
    from supervised_field.airfrans_p1_labels import run_one

    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]   # 'film' | 'concat'

    t_all = time.time()
    sz = np.load(STATS_NPZ, allow_pickle=True)
    std = np.array([float(sz[k][1]) for k in FIELDS], np.float64)

    models = {}
    for tag, wf in (("concat", W_CONCAT), ("film", W_FILM),
                    ("concat_nutres", os.path.join(HERE, "_nf_concat_v5_nutres.npz")),
                    ("film_nutres", os.path.join(HERE, "_nf_film_v5_nutres.npz")),
                    ("concat_ms", os.path.join(HERE, "_nf_concat_v5_ms.npz")),
                    ("film_ms", os.path.join(HERE, "_nf_film_v5_ms.npz")),
                    ("concat_ms_long", os.path.join(HERE, "_nf_concat_v5_ms_long.npz"))):
        if only and tag != only:
            continue
        if not os.path.exists(wf):
            print(f"[eval200v5] {tag} weights missing ({wf}) — skip", flush=True)
            continue
        wz = np.load(wf, allow_pickle=True)
        std_chk = np.asarray(wz["std"], np.float64)
        assert np.allclose(std, std_chk), (std, std_chk)        # stats consistency
        models[tag] = wz
        print(f"[eval200v5] {tag}: {os.path.basename(wf)} variant="
              f"{str(wz['variant'])} cond={str(wz['cond'])} readout="
              f"{str(wz['readout'])} n_params={int(wz['n_params'])} "
              f"epochs_run={int(wz['epochs_run'])} aborted={bool(wz['aborted'])}",
              flush=True)
    if not models:
        print("[eval200v5] no v5 weights found — train first.", flush=True)
        return

    ens_path = next(p for p in ENS_PATHS if os.path.exists(p))
    ez = np.load(ens_path)

    with open(os.path.join(ROOT, "manifest.json")) as f:
        names = sorted(json.load(f)["full_test"])
    assert len(names) == 200, len(names)

    tasks = [(case_params(nm)[2], case_params(nm)[1],
              case_params(nm)[0] / NU_AIR) for nm in names]
    print(f"[eval200v5] XFOIL XTR cl labels for {len(tasks)} full_test cases "
          f"(cached, Pool(6))...", flush=True)
    t0 = time.time()
    with Pool(6) as pool:
        labels = pool.map(run_one, tasks)
    n_ok = sum(1 for l in labels if l["cl"] is not None)
    print(f"[eval200v5] cl labels: {n_ok}/200 converged ({time.time()-t0:.0f}s); "
          f"{200-n_ok} ensemble-head cl fallback", flush=True)

    rows, fallbacks, n_sfail = [], [], 0
    t0 = time.time()
    for i, (nm, lab) in enumerate(zip(names, labels)):
        U, a, prm = case_params(nm)
        Re = U / NU_AIR
        if lab["cl"] is not None:
            cl_used, cl_src = float(lab["cl"]), "xfoil_xtr"
        else:
            x9 = head_feats(prm, a, Re, PSCALE, A_NORM, RE_LO, RE_HI)
            cl_used, cl_src = ens_cl(ez, x9), "ens_head_fallback"
            fallbacks.append(dict(name=nm, alpha=a, cl_ens=cl_used))

        sim = af.Simulation(root=ROOT, name=nm)
        P = np.asarray(sim.position, np.float64)
        TR = np.column_stack([np.asarray(sim.velocity, np.float64),
                              np.asarray(sim.pressure, np.float64).reshape(-1, 1),
                              np.asarray(sim.nu_t, np.float64).reshape(-1, 1)])
        sm = np.asarray(sim.surface, bool)
        n_pt = len(P)

        poly = analytic_polyline(prm)
        r = paint_points_v2(P, poly, a, cl_used, Re)
        PA = np.column_stack([r["u"] * U, r["v"] * U,
                              r["cp"] * 0.5 * U**2, r["nut"] * U])
        PA[sm, 0] = 0.0; PA[sm, 1] = 0.0; PA[sm, 3] = 0.0       # v2 convention

        mid, nrm = panel_mids_normals(poly)
        nn = nearest_idx(P, mid)
        prm4 = np.zeros(4); prm4[:len(prm)] = prm
        X = np.column_stack([
            P[:, 0], P[:, 1], r["d"], nrm[nn, 0], nrm[nn, 1],
            sm.astype(np.float64),
            np.full(n_pt, a / A_NORM),
            np.full(n_pt, 2.0 * (np.log(Re) - RE_LO) / (RE_HI - RE_LO) - 1.0),
            np.tile(prm4 / PSCALE, (n_pt, 1)),
            np.full(n_pt, float(len(prm) == 4))]).astype(np.float32)

        # ---- per-case s (XFOIL-Cf) + g_band (paint Drec/wake_band) ----
        s_val, s_ok = sep_scalar_xfoil(prm, a, Re)
        if not s_ok:
            n_sfail += 1
        s_row = np.full(n_pt, s_val, np.float32)
        g_band = gband(r["d"], r["wake_band"], P[:, 0], Re)

        def mses(pr):
            e = ((pr - TR) / std[None, :]) ** 2
            d = {k: float(v) for k, v in zip(FIELDS, e.mean(0))}
            d["p_surf"] = float(np.mean(((pr[sm, 2] - TR[sm, 2]) / std[2]) ** 2))
            return d

        m_pa = mses(PA)
        per_model = {}
        for tag, wz in models.items():
            t_mean = np.asarray(wz["t_mean"], np.float64)
            net = nf_forward_v5(wz, X, s_row) * std[None, :] + t_mean[None, :]
            pred = net.copy()
            pred[:, :3] += g_band[:, None] * PA[:, :3]          # §4.4 g_band recon
            # nut (col 3): scratch = net only; residual (F1) = + paint nut (global)
            if ("nut_mode" in wz.files) and (str(wz["nut_mode"]) == "residual"):
                pred[:, 3] += PA[:, 3]
            m = mses(pred)
            per_model[tag] = dict(
                mse=m, explode=bool(m["ux"] > 1.0 or m["uy"] > 1.0))

        rows.append(dict(name=nm, U=U, alpha=a, Re=Re, n_nodes=n_pt,
                         n_surf=int(sm.sum()), cl_used=cl_used, cl_source=cl_src,
                         s=s_val, s_xfoil_ok=bool(s_ok),
                         g_band_frac=float(g_band.mean()),
                         vn_leak=float(r["vn_leak"]),
                         mse_paint=m_pa,
                         **{f"mse_{t}": per_model[t]["mse"] for t in per_model},
                         **{f"explode_{t}": per_model[t]["explode"]
                            for t in per_model}))
        if (i + 1) % 10 == 0 or (i + 1) == len(names):
            el = time.time() - t0
            tagrep = next(iter(per_model))
            md = per_model[tagrep]["mse"]
            print(f"  [{i+1:3d}/200] {nm[:40]:40s} a={a:+6.2f} s={s_val:.2f} | "
                  f"{tagrep} ux {md['ux']:.3e} uy {md['uy']:.3e} p {md['p']:.3e} "
                  f"nut {md['nut']:.3e} ps {md['p_surf']:.3e} | {el:.0f}s ETA "
                  f"{el/(i+1)*(200-i-1):.0f}s", flush=True)

    # ---- aggregates ----
    al = np.array([r["alpha"] for r in rows])
    lo, hi = al < 10.0, al >= 10.0

    def agg(tag, mask):
        sel = [r for r, m in zip(rows, mask) if m]
        return {k: float(np.mean([r[tag][k] for r in sel])) for k in KEYS}

    def med(tag):
        return {k: float(np.median([r[tag][k] for r in rows])) for k in KEYS}

    summary = {}
    for tag in models:
        mk = f"mse_{tag}"
        exp_cases = [dict(name=r["name"], alpha=r["alpha"], s=r["s"],
                          ux=r[mk]["ux"], uy=r[mk]["uy"])
                     for r in rows if r[f"explode_{tag}"]]
        summary[tag] = dict(
            all200=agg(mk, np.ones(len(rows), bool)),
            alpha_lt10=agg(mk, lo), alpha_ge10=agg(mk, hi),
            median200=med(mk),
            n_explode=len(exp_cases), explode_cases=exp_cases)
    full_pa = agg("mse_paint", np.ones(len(rows), bool))

    # v0 reference line (from _nf_eval200_v0.json if present)
    v0_ref = None
    if os.path.exists(W_V0_JSON):
        v0j = json.load(open(W_V0_JSON))
        v0_ref = v0j.get("mse_v0")

    out = dict(
        n_evaluated=len(rows), n_alpha_lt10=int(lo.sum()),
        n_alpha_ge10=int(hi.sum()),
        n_xfoil_ok=n_ok, n_cl_fallback=len(fallbacks), fallback_cases=fallbacks,
        n_s_xfoil_fail=n_sfail,
        explode_rule="case counted if ux or uy standardized MSE > 1",
        stats_npz=os.path.basename(STATS_NPZ),
        ens_fallback=os.path.relpath(ens_path, HERE),
        models={t: dict(weights=os.path.basename(W_FILM if t == "film"
                                                  else W_CONCAT),
                        variant=str(models[t]["variant"]),
                        n_params=int(models[t]["n_params"]),
                        epochs_run=int(models[t]["epochs_run"]),
                        aborted=bool(models[t]["aborted"])) for t in models},
        mse_v5=summary,
        mse_paint=dict(all200=full_pa),
        comparison=dict(
            v0_reference=v0_ref,
            mario_published=MARIO_PUBLISHED,
            note="4-way: v0 = prior-art REFERENCE line (concat+uniform+no-gate+"
                 "residual-nut, _nf_eval200_v0.json); concat-control = FiLM "
                 "ISOLATION line (§9.2, bit-identical sampling/g_band/nut/capacity, "
                 "conditioning=concat); FiLM = v5 (per-layer FiLM hypernet); MARIO "
                 "= published raw (x1e-2 ux/uy/p, x1e-1 nut/p_surf). FiLM "
                 "vs concat-control isolates the FiLM mechanism; v0 vs v5 is a "
                 "bundle comparison (§9.1)."),
        cases=rows,
        eval_protocol="full volume, ALL mesh nodes (MARIO protocol); standardized "
                      "with _airfrans_v2_trainstats stds; p_surf on surface nodes "
                      "with the volume-p std; pred[ux,uy,p] = g_band*PAINT + "
                      "destd(net) (alpha-independent geometric g_band), pred[nut] = "
                      "destd(net) (scratch, NO paint); s = XFOIL-Cf<0 fraction /0.10 "
                      "clipped (alpha fallback on XFOIL fail), SAME source as train; "
                      "paint uses XFOIL-XTR cl, ensemble-head cl where XFOIL fails.")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, default=float)

    print(f"\n[eval200v5] {len(rows)} cases ({int(lo.sum())} alpha<10, "
          f"{int(hi.sum())} alpha>=10), {len(fallbacks)} cl fallbacks, "
          f"{n_sfail} s-XFOIL fails, {(time.time()-t_all)/60:.1f} min total",
          flush=True)
    for tag in models:
        s = summary[tag]
        print(f"  --- {tag} (n_explode {s['n_explode']}) ---", flush=True)
        for lab, d in (("all200", s["all200"]), ("a<10", s["alpha_lt10"]),
                       ("a>=10", s["alpha_ge10"]), ("median", s["median200"])):
            print(f"   {tag} {lab:8s} " +
                  "  ".join(f"{k}={d[k]:.4e}" for k in KEYS), flush=True)
    if v0_ref:
        print("  --- v0 reference ---", flush=True)
        for lab in ("all200", "alpha_lt10", "alpha_ge10", "median200"):
            if lab in v0_ref:
                print(f"   v0 {lab:10s} " +
                      "  ".join(f"{k}={v0_ref[lab][k]:.4e}" for k in KEYS),
                      flush=True)
    print("  mario-raw    " + "  ".join(f"{k}={MARIO_PUBLISHED[k]:.4e}"
                                        for k in KEYS), flush=True)
    print(f"[eval200v5] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
