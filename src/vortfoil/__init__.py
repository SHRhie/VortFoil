# -*- coding: utf-8 -*-
"""VortFoil: a vortex-panel-informed neural operator for steady airfoil flow.

    import vortfoil
    vf = vortfoil.load_model()
    r = vf.predict(cst=cst16, alpha=4.0, Re=1e6)
    print(r.cl, r.cd, r.cm)

``vortfoil.about()`` prints what this build is: the weights it carries, the recipe they were trained
with, the populations they were evaluated on and the scores it reproduces.
"""
from .api import VortFoil, SurfacePrediction, manifest
from . import geometry

__version__ = "2.0.0"
__all__ = ["VortFoil", "SurfacePrediction", "load_model", "about", "manifest", "geometry", "__version__"]

_CACHE = {}


def load_model(mode="graph", device="cpu", max_batch=None, weights_dir=None, cache=True):
    """Load the deployed model. ``mode`` is 'graph' (compiled) or 'eager'; ``device`` is 'cpu',
    'gpu' or 'auto'. Repeated calls with the same settings reuse the loaded operators."""
    key = (mode, device, max_batch, weights_dir)
    if cache and key in _CACHE:
        return _CACHE[key]
    model = VortFoil(mode=mode, device=device, max_batch=max_batch, weights_dir=weights_dir)
    if cache:
        _CACHE[key] = model
    return model


def about(stream=None):
    """Print the build record: weights, recipe, evaluation populations, scores, citation."""
    import json
    import sys
    m = manifest()
    out = stream or sys.stdout
    w = out.write
    w(f"VortFoil {__version__} -- {m['model']['description']}\n\n")
    w("Components\n")
    for c in m["model"]["components"]:
        w(f"  - {c}\n")
    w("\nWeights\n")
    for k, v in m["weights"].items():
        w(f"  {k:16s} {v['file']:42s} sha256 {v['sha256'][:12]}\n")
    w("\nScores this build reproduces (R^2 against XFOIL)\n")
    for pop, d in m["evaluation"].items():
        w(f"  {pop} (n={d['n']}): " + ", ".join(f"{k} {v}" for k, v in d["scores"].items()) + "\n")
    w(f"\n  {m['evaluation_note']}\n")
    w("\nInference cost (measured; absolutes are host dependent)\n")
    for k, v in m["inference_cost"].items():
        w(f"  {k:34s} {v}\n")
    w("\nLimits\n")
    for lim in m["limits"]:
        w(f"  - {lim}\n")
    w(f"\nCitation\n  {m['citation']}\n")
    return None
