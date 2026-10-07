"""Figures for CMAME_v2 §Profile drag. Generator + saved numbers, per the figure-provenance rule.

Inputs (measured, sealed in VortFoil_v2/artifacts/op2_cdloss_20260917T044259Z/raw):
  sep_eval_preds.npz          predicted TE state + truth on the 1,930-row separated set (m325_BT, OP1 scr_bd128)
  sep_eval_preds_nfout.npz    NeuralFoil xlarge on the same rows
  ../weights/cd_closure_poly2.npz   the deployed fitted_poly2 coefficients
Outputs: VortFoil/figures/vf_drag_closure_sep.{pdf,png}, vf_drag_decomp_sep.{pdf,png},
         VortFoil/figures/vf_drag_fig_numbers.json (every plotted number)
"""
import json, os, sys
import numpy as np
from itertools import combinations_with_replacement as cwr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # <release>/code
import vortfoil_paths as VP; VP.add_paths()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # plot_style.py sits beside this file
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_style import apply_ncfe_style
import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
from staged.common.wake_ode_tf import wake_ode_cd_tf
apply_ncfe_style()
plt.rcParams['savefig.bbox'] = None   # figures below are exactly 5x4 in; plot_style sets tight, which would shrink the canvas
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = VP.FIGURES
# sep_eval_preds*.npz are the per-row prediction dumps eval_on_targets.py writes for the
# separated set; they are NOT shipped in this release (see README). Regenerate them with
#   eval_on_targets.py data/eval/sep_eval_targets_v2.npz <RAW>/sep_eval_preds.npz
RAW = os.environ.get('VORTFOIL_RAW', VP.DATA_EVAL)
d = np.load(os.path.join(RAW, 'sep_eval_preds.npz')); nf = np.load(os.path.join(RAW, 'sep_eval_preds_nfout.npz'))
thp, dsp, uep, tht, dst, uet, cd = d['thp'], d['dsp'], d['uep'], d['th_t'], d['ds_t'], d['ue_t'], d['cd_t']
H = dst / tht; N = len(cd); cdnf = np.clip(nf['cd_nf'], 1e-4, 0.1)
c = np.load(os.path.join(VP.WEIGHTS, 'cd_closure_poly2.npz')); mu, sd, w = c['mu'], c['sd'], c['w']
def poly2(t, Hh, u):
    F = np.c_[np.log(np.maximum(t, 1e-12)), Hh, u]
    X = np.c_[tuple([np.ones(len(F))] + [F[:, list(k)].prod(1) for dd in (1, 2) for k in cwr(range(3), dd)])]
    return np.clip(10 ** (((X - mu) / sd) @ w), 1e-4, 0.1)
def clo(name, t, dd, u):
    if name == 'fitted_poly2': return poly2(t, dd / t, u)
    return np.clip(wake_ode_cd_tf(tf.constant(t, tf.float32), tf.constant(dd, tf.float32), tf.constant(u, tf.float32), closure=name).numpy().astype(np.float64), 1e-4, 0.1)
r2 = lambda p, k: float(1 - np.sum((p[k] - cd[k]) ** 2) / np.sum((cd[k] - cd[k].mean()) ** 2))
bins = [('all', np.ones(N, bool)), ('H<6', H < 6), ('6-10', (H >= 6) & (H < 10)), ('H>=10', H >= 10)]
OUT = {'n_rows': int(N), 'bins': {k: int(m.sum()) for k, m in bins}}
# ---- Fig 1: parity, three closures + NeuralFoil, separated set ----
preds = {'legacy_ode': clo('legacy_ode', thp, dsp, uep), 'squire_young': clo('squire_young', thp, dsp, uep),
         'fitted_poly2': clo('fitted_poly2', thp, dsp, uep), 'NeuralFoil': cdnf}
for k, p in preds.items():
    assert not np.allclose(p, cd), k          # a prediction, not the truth re-plotted
    OUT[k] = {b: r2(p, m) for b, m in bins}
fig, ax = plt.subplots(2, 2, figsize=(5, 4), sharex=True, sharey=True, constrained_layout=True); ax = ax.ravel()
titles = {'legacy_ode': 'marched wake (baseline)', 'squire_young': 'Squire–Young, no march', 'fitted_poly2': 'fitted closure (deployed)', 'NeuralFoil': 'NeuralFoil (extra-large)'}
for a, (k, p) in zip(ax, preds.items()):
    sc = a.scatter(cd, p, c=np.clip(H, 1, 14), s=3, cmap='viridis', linewidths=0, rasterized=True)
    a.plot([1e-4, 0.1], [1e-4, 0.1], 'k-', lw=0.6); a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(cd.min(), cd.max()); a.set_ylim(4e-3, 0.11)  # x: exact plotted XFOIL Cd range, no margins (2026-09-19); 1:1 line [1e-4,0.1] still spans it
    a.set_title(titles[k], fontsize=8); a.text(0.05, 0.92, f"$R^2$ {OUT[k]['all']:.3f} | $H_{{TE}}\\!\\geq\\!10$: {OUT[k]['H>=10']:.3f}", transform=a.transAxes, fontsize=6.5, va='top')
for j in (2, 3): ax[j].set_xlabel('XFOIL $C_d$')
for j in (0, 2): ax[j].set_ylabel('predicted $C_d$')
cb = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.03); cb.set_label('$H_{TE}$ (truth)')
for ext in ('pdf', 'png'): fig.savefig(os.path.join(FIG, f'vf_drag_closure_sep.{ext}'), dpi=400)
plt.close(fig)
# ---- Fig 2: truth-replacement decomposition, fitted_poly2, separated set ----
combos = {'deployed': (thp, dsp, uep), r'$\theta_{TE}$ truth': (tht, dsp, uep), r'$\delta^*_{TE}$ truth': (thp, dst, uep),
          r'$u_{e,TE}$ truth': (thp, dsp, uet), r'$\theta,\delta^*$ truth': (tht, dst, uep), 'all truth': (tht, dst, uet)}
DEC = {k: {b: r2(poly2(t, dd / t, u), m) for b, m in bins} for k, (t, dd, u) in combos.items()}
OUT['decomposition_fitted_poly2'] = DEC
fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True); x = np.arange(len(combos)); wd = 0.26
for i, (b, _) in enumerate([bins[1], bins[2], bins[3]]):
    ax.bar(x + (i - 1) * wd, [DEC[k][b] for k in combos], wd, label={'H<6': r'$H_{TE}<6$', '6-10': r'$6\leq H_{TE}<10$', 'H>=10': r'$H_{TE}\geq10$'}[b])
ax.set_xticks(x); ax.set_xticklabels(list(combos), rotation=20, ha='right', fontsize=7); ax.set_ylabel('$C_d$ $R^2$ (separated set)'); ax.set_ylim(0.8, 1.0)
ax.legend(frameon=False, fontsize=7, ncol=3, loc='upper left'); ax.grid(axis='y', lw=0.3, alpha=0.5)
for ext in ('pdf', 'png'): fig.savefig(os.path.join(FIG, f'vf_drag_decomp_sep.{ext}'), dpi=400)
json.dump(OUT, open(os.path.join(FIG, 'vf_drag_fig_numbers.json'), 'w'), indent=1)
print(json.dumps({k: {b: round(v, 4) for b, v in OUT[k].items()} for k in preds}, indent=1))
print('decomp H>=10:', {k: round(v['H>=10'], 4) for k, v in DEC.items()})
