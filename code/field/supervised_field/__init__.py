# supervised_field — supervised 2D-field CFD surrogate (SDF + masking foundation)

# --- release packaging (VortFoil public release) --------------------------------------
# Trained weights live in <release>/weights/ under release names, not beside these modules.
# Everything else these modules read or write by a leading-underscore name (training
# corpora _nf_train*.npz, token caches, intermediate JSON) is a research-tree artifact that
# is NOT shipped with this release, and the corpora are not byte-for-byte regenerable -- see the README.
import os as _os

WEIGHTS = _os.environ.get(
    "VORTFOIL_WEIGHTS",
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__)))), "..", "weights"),
)
WEIGHTS = _os.path.normpath(WEIGHTS)
