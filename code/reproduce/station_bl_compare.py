"""Station-matched boundary-layer comparison on the attached grid (2026-09-18).

NeuralFoil's BL arrays live on 32 fixed x/c stations (0.0156..0.984, LE->TE); its last station is
NOT the trailing edge, and XFOIL's theta at x/c=0.984 is ~10% below theta_TE (delta* ~18%), so the
earlier TE-vs-station-31 comparison overstated NeuralFoil's error. Here all three are read at the
same 32 stations: XFOIL truth (bl_at_nf_x, interpolated on the Xeon from XFOIL's own x arrays),
OP2 (theta, delta* on the 121-node per-surface UNIFORM x/c grid, np.interp onto the stations) and
NeuralFoil as stored. Same rows, same stations, same reference.

argv: <artifact root> [targets.npz nfin.npz nf_out.npz ours_out.npz result.json]   (defaults = attached v2 set;
       pass the five raw/ file names for the separated v2 set; phase B runs when nf_out.npz exists)
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # <release>/code
import vortfoil_paths as VP; VP.add_paths()
# argv[1] kept for compatibility with the sealed artifact layout; it now defaults to the
# release root, whose data/eval/ replaces the artifact's raw/ directory.
R = sys.argv[1] if len(sys.argv) > 1 else VP.RELEASE_ROOT
RAW = VP.DATA_EVAL if os.path.abspath(R) == os.path.abspath(VP.RELEASE_ROOT) else R + "/raw"
OP2_W = os.path.join(VP.WEIGHTS, "op2_m325_BT.npz")   # = CANDIDATE_op2_m325_BT.npz

A = sys.argv[2:] or ["attached_surface_targets_v2.npz", "nf_in_attached_v2.npz", "nf_full_attached_v2.npz", "op2_bl_at_nf_x.npz", "station_bl_compare.json"]
v2 = np.load(RAW + "/" + A[0])
nfin = np.load(RAW + "/" + A[1])
row = nfin["row"]                       # rows of the 2880 grid that pass the same gate eval_on_targets uses
nfx = v2["nf_x"].astype(np.float64)     # 32 stations
xc121 = np.linspace(0.0, 1.0, 121)   # OP2 xi-node j <-> x/c (per surface): deck generator resample_to_xi = np.interp(linspace(0,1,121), x/c, val)

OURS = RAW + "/" + A[3]
if not os.path.exists(OURS):
    import tensorflow as tf; tf.config.set_visible_devices([], "GPU")
    from staged.op2_bl_surrogate import build_op2_surrogate
    DT = tf.float32
    cst = v2["cst"][row].astype(np.float32); al = v2["alpha"][row].reshape(-1, 1).astype(np.float32); re = v2["Re"][row].reshape(-1, 1).astype(np.float32)
    op2 = build_op2_surrogate(n_surface_pts=121, enable_wake=False, branch_dim=128, branch_depth=3, trunk_depth=2, N_m=16)
    op2.predict_bl(tf.constant(cst[:2], DT), tf.constant(al[:2], DT), tf.constant(re[:2], DT), training=False)
    op2.load_weights(OP2_W, strict=False)
    N = len(row); out = {k: np.zeros((N, 32)) for k in ("theta_u", "theta_l", "dstar_u", "dstar_l")}
    for s in range(0, N, 1024):
        e = min(s + 1024, N)
        bl = op2.predict_bl(tf.constant(cst[s:e], DT), tf.constant(al[s:e], DT), tf.constant(re[s:e], DT), training=False)
        for k, kk in (("theta_u", "theta_u"), ("theta_l", "theta_l"), ("dstar_u", "delta_u"), ("dstar_l", "delta_l")):
            arr = bl[kk].numpy()
            out[k][s:e] = np.stack([np.interp(nfx, xc121, a) for a in arr])
    np.savez(OURS, **out, row=row, nf_x=nfx, op2_weights=os.path.basename(OP2_W))
    print("phase A written", OURS, N)

NF = RAW + "/" + A[2]
if not os.path.exists(NF):
    sys.exit(0)
ours = np.load(OURS); nf = np.load(NF); T = v2["bl_at_nf_x"][row].astype(np.float64)   # (N,4,32): theta_u, theta_l, dstar_u, dstar_l
truth = {"theta_u": T[:, 0], "theta_l": T[:, 1], "dstar_u": T[:, 2], "dstar_l": T[:, 3]}
nfp = {"theta_u": nf["theta_u"], "theta_l": nf["theta_l"], "dstar_u": nf["H_u"] * nf["theta_u"], "dstar_l": nf["H_l"] * nf["theta_l"]}
assert not np.allclose(ours["theta_u"], truth["theta_u"]) and not np.allclose(nfp["theta_u"], truth["theta_u"])
al = v2["alpha"][row].ravel()

def medrel(p, t, m):
    m = m & np.isfinite(p) & np.isfinite(t) & (t > 0)
    return float(np.median(np.abs(p[m] - t[m]) / t[m])), int(m.sum())

def r2(p, t, m):
    m = m & np.isfinite(p) & np.isfinite(t)
    return float(1 - np.sum((p[m] - t[m]) ** 2) / np.sum((t[m] - t[m].mean()) ** 2))

res = {"n_rows": int(len(row)), "stations": nfx.tolist(), "op2_weights": str(ours["op2_weights"]),
       "note": "all three read at NeuralFoil's 32 x/c stations; truth = XFOIL interpolated on its own x arrays; OP2 interpolated from the 121-node uniform x/c grid",
       "per_station_median_rel_err": {}, "pooled": {}, "te_most_station": {}, "by_alpha_band": {}}
allm = np.ones(len(row), bool)
for k in truth:
    res["per_station_median_rel_err"][k] = {"ours": [medrel(ours[k][:, j], truth[k][:, j], allm)[0] for j in range(32)],
                                            "nf": [medrel(nfp[k][:, j], truth[k][:, j], allm)[0] for j in range(32)]}
    band = nfx >= 0.1     # exclude the near-LE stations where theta is ~1e-5 and transition placement dominates
    po, n = medrel(ours[k][:, band].ravel(), truth[k][:, band].ravel(), np.ones(band.sum() * len(row), bool))
    pn, _ = medrel(nfp[k][:, band].ravel(), truth[k][:, band].ravel(), np.ones(band.sum() * len(row), bool))
    res["pooled"][k] = {"x_ge_0.1_median_rel_err": {"ours": po, "nf": pn}, "n_pairs": n}
    res["te_most_station"][k] = {"x": float(nfx[-1]), "median_rel_err": {"ours": medrel(ours[k][:, -1], truth[k][:, -1], allm)[0], "nf": medrel(nfp[k][:, -1], truth[k][:, -1], allm)[0]},
                                 "R2": {"ours": r2(ours[k][:, -1], truth[k][:, -1], allm), "nf": r2(nfp[k][:, -1], truth[k][:, -1], allm)}}
# summed-surface theta and H at the TE-most station (the closure's inputs, station-matched)
th_o = ours["theta_u"][:, -1] + ours["theta_l"][:, -1]; th_n = nfp["theta_u"][:, -1] + nfp["theta_l"][:, -1]; th_t = truth["theta_u"][:, -1] + truth["theta_l"][:, -1]
ds_o = ours["dstar_u"][:, -1] + ours["dstar_l"][:, -1]; ds_n = nfp["dstar_u"][:, -1] + nfp["dstar_l"][:, -1]; ds_t = truth["dstar_u"][:, -1] + truth["dstar_l"][:, -1]
res["te_most_station"]["theta_sum"] = {"median_rel_err": {"ours": medrel(th_o, th_t, allm)[0], "nf": medrel(th_n, th_t, allm)[0]}, "R2": {"ours": r2(th_o, th_t, allm), "nf": r2(th_n, th_t, allm)}}
res["te_most_station"]["dstar_sum"] = {"median_rel_err": {"ours": medrel(ds_o, ds_t, allm)[0], "nf": medrel(ds_n, ds_t, allm)[0]}, "R2": {"ours": r2(ds_o, ds_t, allm), "nf": r2(ds_n, ds_t, allm)}}
res["te_most_station"]["H_sum"] = {"median_rel_err": {"ours": medrel(ds_o / th_o, ds_t / th_t, allm)[0], "nf": medrel(ds_n / th_n, ds_t / th_t, allm)[0]}}
for lo, hi in ((-11, 0), (0, 8), (8, 19)):
    m = (al >= lo) & (al < hi)
    res["by_alpha_band"][f"[{lo},{hi})"] = {"n": int(m.sum()), **{k: {"ours": medrel(ours[k][:, -1], truth[k][:, -1], m)[0], "nf": medrel(nfp[k][:, -1], truth[k][:, -1], m)[0]} for k in truth}}
json.dump(res, open(RAW + "/" + A[4], "w"), indent=1)
print(json.dumps({"pooled": res["pooled"], "te_most": {k: v["median_rel_err"] for k, v in res["te_most_station"].items()}}, indent=1))
