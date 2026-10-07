# -*- coding: utf-8 -*-
"""Lift-input override for the operator-conditioned decoder's C_l substitution diagnostics (2026-09-19).

ClOverrideProvider wraps the state provider and replaces the paint circulation scale `cl` per case, leaving s
untouched.  code/field/f0cm_final_rescore.py uses it for two inference-only arms on the same weights:
    xfoil  cl = XFOIL's forced-transition lift of the sealed XFOIL-conditioned evaluation (cl_used), looked up
              per case in _CLX by (NACA parameters, alpha, Re)
    bias   cl = provider cl - CL_BIAS, the training-set bias of the sheet's circulation lift against that lift
Both are DIAGNOSTIC: they test the trained decoder's sensitivity to its lift input and are not a deployable route.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # <release>/code
import vortfoil_paths as VP; REPO = VP.add_paths()
import state_provider as G

CL_BIAS = 0.0294   # mean sheet C_l^gamma minus XFOIL forced-transition C_l over the 779 training cases with an XFOIL lift, results/gate0_calibration.json (cl_op1_vs_xfoil_train.bias)
_CLX = {}


class ClOverrideProvider:
    """wraps the confirmed provider; replaces cl per case (keyed on (prm, alpha, Re)); s untouched."""
    def __init__(self, base, mode):
        self.base, self.mode = base, mode
        self.last = None

    def state_for(self, prm, alpha_deg, Re):
        st = self.base.state_for(prm, alpha_deg, Re); self.last = self.base.last
        if self.mode == "xfoil":
            key = (tuple(np.round(np.asarray(prm, np.float64), 4)), round(float(alpha_deg), 3), round(float(Re) / 1e3))
            st["cl"] = float(_CLX[key])
        elif self.mode == "bias":
            st["cl"] = float(st["cl"] - CL_BIAS)
        st["provenance"] = dict(st["provenance"]); st["provenance"]["detail"] += f" | DIAGNOSTIC cl override mode={self.mode}"
        return st
