"""Fit a Squire-Young-class closure Cd = f(theta_TE, H_TE, ue_TE) on the TRAINING deck,
evaluate on the published holdout. Measurement only; nothing deployed."""
import json, os, sys
import numpy as np
from itertools import combinations_with_replacement as cwr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # <release>/code
import vortfoil_paths as VP; VP.add_paths()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # op2_dense_surrogate.py
os.environ.setdefault("CUDA_VISIBLE_DEVICES","-1")
import tensorflow as tf; tf.config.set_visible_devices([],"GPU")
from staged.common.geometry import build_single_loop_contour_cst_tf
from staged.common.panel import compute_inviscid_surface_velocity
from staged.common.wake_ode_tf import wake_ode_cd_tf
from staged.op1_inviscid import build_op1
from op2_dense_surrogate import build_op2_surrogate
# D = the XFOIL training deck directory. xfoil_nf_scale_merged_clean.npz (325,370 rows) is
# NOT shipped in this release (see README); set VORTFOIL_DECKS to point at it.
D=os.environ.get("VORTFOIL_DECKS", VP.DATA_EVAL)+os.sep; W=VP.WEIGHTS+os.sep; DT=tf.float32; NSP=121
OP2_W, HEAD = sys.argv[1], sys.argv[2]
r2=lambda p,t: float(1-np.sum((p-t)**2)/np.sum((t-t.mean())**2))

# ---- training deck TE state ----
z=np.load(D+"xfoil_nf_scale_merged_clean.npz",allow_pickle=False)
m=np.asarray(z["train_valid"])
th=(np.asarray(z["train_theta_u"])[m,-1]+np.asarray(z["train_theta_l"])[m,-1]).astype(np.float64)
ds=(np.asarray(z["train_delta_u"])[m,-1]+np.asarray(z["train_delta_l"])[m,-1]).astype(np.float64)
ue=(0.5*(np.abs(np.asarray(z["train_ue_u"])[m,-1])+np.abs(np.asarray(z["train_ue_l"])[m,-1]))).astype(np.float64)
cd=np.asarray(z["train_cd_visc"])[m,0].astype(np.float64)
def feats(th,H,ue,deg=2):
    F=np.c_[np.log(np.maximum(th,1e-12)),H,ue]
    return np.c_[tuple([np.ones(len(F))]+[F[:,list(c)].prod(1)
                 for d in range(1,deg+1) for c in cwr(range(3),d)])]
X=feats(th,ds/th,ue); y=np.log10(cd)
mu,sd=X.mean(0),np.maximum(X.std(0),1e-9); Xs=(X-mu)/sd; Xs[:,0]=1
w=np.linalg.solve(Xs.T@Xs+1e-2*len(Xs)*np.eye(X.shape[1]),Xs.T@y)
apply_=lambda th,H,ue: np.clip(10**(((feats(th,H,ue)-mu)/sd*np.r_[0,np.ones(X.shape[1]-1)]
                                     +np.r_[1,np.zeros(X.shape[1]-1)])@w),1e-4,0.1)
print(f"fit on {len(y)} train rows, {X.shape[1]} terms, train R2(log)={r2(Xs@w,y):.4f}")

# ---- holdout ----
h=np.load(os.path.join(VP.DATA_EVAL,"holdout_targets.npz"),allow_pickle=True)
v=h["valid"]&(np.abs(h["cl_visc"][:,0])<=3)&(h["cd_visc"][:,0]>1e-4)&(h["cd_visc"][:,0]<=0.1)
for k in ["theta_u_te","theta_l_te","dstar_u_te","dstar_l_te","ue_te"]:
    v&=np.isfinite(h[k][:,0])&((h[k][:,0]!=0) if "theta" in k else True)
cst=h["cst"][v].astype(np.float32); al=h["alpha"][v]; re=h["Re"][v]
cdt=h["cd_visc"][v,0].astype(np.float64); N=len(cst)
tht=(h["theta_u_te"][v,0]+h["theta_l_te"][v,0]).astype(np.float64)
dst=(h["dstar_u_te"][v,0]+h["dstar_l_te"][v,0]).astype(np.float64)
uet=h["ue_te"][v,0].astype(np.float64)

# ---- predicted state from the given OP2 ----
op1=build_op1(n_surface_pts=NSP,branch_dim=128)
op2=build_op2_surrogate(n_surface_pts=NSP,enable_wake=False,branch_dim=128,
                        branch_depth=3,trunk_depth=2,N_m=16,head_mode=HEAD)
op1.compute_losses(tf.constant(cst[:2],DT),tf.constant(al[:2],DT),tf.constant(re[:2],DT),training=False)
op2.predict_bl(tf.constant(cst[:2],DT),tf.constant(al[:2],DT),tf.constant(re[:2],DT),training=False)
_ck=np.load(W+"op1_supervised_eps0.50.scr_bd128.npz")
_k=sorted(k for k in _ck.files if k.startswith('v_'))
op1.field_decoder.build((None,int(_ck[_k[-6]].shape[0])))
op1.load_weights(W+"op1_supervised_eps0.50.scr_bd128.npz")
op2.load_weights(W+OP2_W,strict=False)   # e.g. op2_m325_BT.npz
thp=np.zeros(N); dsp=np.zeros(N); uep=np.zeros(N)
for s in range(0,N,1024):
    e=min(s+1024,N); c=tf.constant(cst[s:e],DT); a=tf.constant(al[s:e],DT); r=tf.constant(re[s:e],DT)
    geom=build_single_loop_contour_cst_tf(c,n_surface_pts=NSP)
    g=op1.predict_gamma(c,a,Re=r,training=False)
    Ut=compute_inviscid_surface_velocity(geom,g["gamma_nodes"],a)["Ut"]
    uep[s:e]=0.5*(tf.abs(Ut[:,0])+tf.abs(Ut[:,-1])).numpy()
    bl=op2.predict_bl(c,a,r,training=False)
    thp[s:e]=(bl["theta_u"][:,-1]+bl["theta_l"][:,-1]).numpy()
    dsp[s:e]=(bl["delta_u"][:,-1]+bl["delta_l"][:,-1]).numpy()
ode=lambda t,d,u: np.clip(wake_ode_cd_tf(tf.constant(t,DT),tf.constant(d,DT),tf.constant(u,DT)).numpy().astype(np.float64),1e-4,0.1)
hi=(dst/tht)>4
out={}
print(f"\n{'':44s} {'Cd R²':>8s} {'MedAPE%':>8s} {'H>4 ratio':>9s}")
for nm,(t,d,u,hh) in {
  "TRUTH 상태 + 배포 wake-ODE":(tht,dst,uet,dst/tht),
  "TRUTH 상태 + 적합 폐합식":(tht,dst,uet,dst/tht),
  f"예측 상태({HEAD}) + 배포 wake-ODE":(thp,dsp,uep,dsp/thp),
  f"예측 상태({HEAD}) + 적합 폐합식":(thp,dsp,uep,dsp/thp)}.items():
    p = apply_(t,hh,u) if "적합" in nm else ode(t,d,u)
    out[nm]={"cd_raw_r2":r2(p,cdt),"medape":float(100*np.median(abs(p/cdt-1))),
             "hi_ratio":float(np.median(p[hi]/cdt[hi]))}
    print(f"{nm:44s} {out[nm]['cd_raw_r2']:8.4f} {out[nm]['medape']:8.2f} {out[nm]['hi_ratio']:9.4f}")
json.dump({"op2":OP2_W,"head":HEAD,"n":int(N),"coef":w.tolist(),"results":out},
          open(sys.argv[3],"w"),indent=1)
