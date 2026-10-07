# -*- coding: utf-8 -*-
"""Scoring helpers of the field re-score, imported as R by code/field/f0cm_final_rescore.py.

  arr(rows, arm, k, mk)  per-case (n, 4) array of one arm / decoder / mask from state_provider._work rows
  paired(N, O)           paired case-bootstrap ratio mean(N)/mean(O) per channel: B=10000, one rng seeded 0,
                         channels in order (the convention of results/field_deployed_weights.json)
  MARIO, BANDS           MARIO's published aggregate (ux, uy, p, nut) and the incidence bands of the band table
The state provider (G) and the lift-override provider (DG) are reached through this module, as in the research tree.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # <release>/code
import vortfoil_paths as VP; REPO = VP.add_paths()  # release root; puts code/model + code/field on sys.path
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(VP.CODE, "reproduce"))   # step2_diag_f0c_cl
import state_provider as G          # first: pins BLAS / CUDA before numpy
import step2_diag_f0c_cl as DG
import numpy as np

MARIO = dict(ux=1.52e-3, uy=1.13e-3, p=2.40e-3, nut=9.60e-3)
BANDS = ((-6, 0), (0, 8), (8, 12), (12, 16))


def arr(rows, arm, k, mk):
    return np.array([[r["arms"][arm][str(k)][mk][c] for c in G.FIELDS] for r in rows])


def paired(N, O, B=10000):
    """same convention as the deployed-weights table (one rng, channels in order, case bootstrap)."""
    rng = np.random.default_rng(0); out = {}
    for j, ch in enumerate(G.FIELDS):
        n, o = N[:, j], O[:, j]; bs = np.empty(B)
        for b in range(B):
            ii = rng.integers(0, len(n), len(n)); bs[b] = n[ii].mean() / o[ii].mean()
        out[ch] = dict(R=float(n.mean() / o.mean()), ci95=[float(v) for v in np.percentile(bs, [2.5, 97.5])], n_improved=int((n < o).sum()))
    return out
