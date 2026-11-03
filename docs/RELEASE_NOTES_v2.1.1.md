# VortFoil 2.1.1

Release date: 2026-11-03.

The application and `vortfoil view` now show the RANS coefficient correction that 2.1.0 added to the Python API and the
command line. The models, weights and recorded scores are those of 2.1.0, the version the revised manuscript uses.

## What changed

- The RANS state's box gives the surface model's C_l, C_d and C_m through the RANS coefficient correction
  (`vortfoil.rans`): the values of `predict(..., rans=True)` and `vortfoil predict --rans`. Outside the range the
  correction was fitted on (incidence −5° to 15°, Reynolds number 2×10⁶ to 6×10⁶) they are pale under a heading that
  says so. They are shown with the RANS state; the XFOIL state alone does not show them.
- The exported `_coefficients.csv` appends `Cl_RANS_corrected`, `Cd_RANS_corrected`, `Cm_RANS_corrected` and
  `RANS_correction_in_fitted_range`. The earlier columns keep their places, and the warnings column also lists the
  correction's range warning.

## Install or upgrade

```bash
uv tool install --python 3.12 --managed-python vortfoil      # or: uv tool upgrade vortfoil
```

Inside an environment with a pinned TensorFlow, upgrade without touching it:

```bash
pip install --no-deps --upgrade vortfoil
```

Full history is in [`CHANGELOG.md`](../CHANGELOG.md).
