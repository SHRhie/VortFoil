"""NeuralFoil xlarge with FULL outputs (CL, CD, CM, Xtr, and the trailing-edge BL state per side) on a coords dump.
Usage: run_nf_full.py <coords_npz with coords/alpha/Re> <out.npz>
TE state: NeuralFoil returns theta, H, ue/vinf at 32 points per side; index 31 is the last (TE-most) point.
theta_TE = theta_u[31]+theta_l[31]; dstar_TE = H*theta summed; ue_TE = mean of the two sides' ue[31]."""
import sys, time
import numpy as np, neuralfoil
z = np.load(sys.argv[1], allow_pickle=False); XY, al, Re = z["coords"], z["alpha"].reshape(-1), z["Re"].reshape(-1)
N = len(al); keys = ["CL", "CD", "CM", "Top_Xtr", "Bot_Xtr", "analysis_confidence"]
out = {k: np.full(N, np.nan) for k in keys}
for k in ("theta_u", "theta_l", "H_u", "H_l", "ue_u", "ue_l"): out[k] = np.full((N, 32), np.nan)
uniq = {}
for i in range(N): uniq.setdefault(XY[i].tobytes(), []).append(i)
t0 = time.time()
for _k, idx in uniq.items():
    idx = np.array(idx)
    o = neuralfoil.get_aero_from_coordinates(coordinates=XY[idx[0]].astype(np.float64), alpha=al[idx].astype(np.float64),
                                             Re=Re[idx].astype(np.float64), n_crit=7.0, xtr_upper=1.0, xtr_lower=1.0, model_size="xlarge")
    for k in keys: out[k][idx] = np.asarray(o[k]).ravel()
    for j in range(32):
        out["theta_u"][idx, j] = np.asarray(o[f"upper_bl_theta_{j}"]).ravel(); out["theta_l"][idx, j] = np.asarray(o[f"lower_bl_theta_{j}"]).ravel()
        out["H_u"][idx, j] = np.asarray(o[f"upper_bl_H_{j}"]).ravel();         out["H_l"][idx, j] = np.asarray(o[f"lower_bl_H_{j}"]).ravel()
        out["ue_u"][idx, j] = np.asarray(o[f"upper_bl_ue/vinf_{j}"]).ravel();  out["ue_l"][idx, j] = np.asarray(o[f"lower_bl_ue/vinf_{j}"]).ravel()
np.savez(sys.argv[2], **out, n_shapes=len(uniq))
print(f"{len(uniq)} shapes, {N} rows, {time.time()-t0:.1f}s; CM finite {np.isfinite(out['CM']).mean():.3f}")
