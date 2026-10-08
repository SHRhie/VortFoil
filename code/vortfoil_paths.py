# -*- coding: utf-8 -*-
"""Release-root path resolution for the VortFoil single-model chain.

Every script in this repository resolves its weights, evaluation data and sealed
result JSONs through this module instead of an absolute path, so the tree can be
cloned anywhere.  Set VORTFOIL_ROOT to override the auto-detected release root
and VORTFOIL_WEIGHTS to point the model loaders at a different weights directory.

    import vortfoil_paths; vortfoil_paths.add_paths()
    from staged.op1_inviscid import build_op1          # code/model/staged
    import supervised_field.nf_eval_shape              # code/field/supervised_field
    import state_provider                              # code/field
"""
import os
import sys

RELEASE_ROOT = os.environ.get(
    "VORTFOIL_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
CODE = os.path.join(RELEASE_ROOT, "code")
SRC = os.path.join(RELEASE_ROOT, "src")
MODEL = os.path.join(CODE, "model")
FIELD = os.path.join(CODE, "field")
WEIGHTS = os.environ.get("VORTFOIL_WEIGHTS", os.path.join(RELEASE_ROOT, "weights"))
DATA_EVAL = os.path.join(RELEASE_ROOT, "data", "eval")
RESULTS = os.path.join(RELEASE_ROOT, "results")
FIGURES = os.path.join(RELEASE_ROOT, "figures")


def add_paths():
    """Put src, code/model and code/field on sys.path (idempotent). src goes ahead of site-packages, so the scripts
    use this checkout's vortfoil package rather than another installed copy of a different version."""
    for p in (SRC, MODEL, FIELD):
        if p not in sys.path:
            sys.path.insert(0, p)
    return RELEASE_ROOT


if __name__ == "__main__":
    add_paths()
    for k in ("RELEASE_ROOT", "MODEL", "FIELD", "WEIGHTS", "DATA_EVAL", "RESULTS"):
        print(f"{k:12s} {globals()[k]}")
