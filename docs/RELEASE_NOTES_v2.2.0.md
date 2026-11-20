# VortFoil 2.2.0

Release date: 2026-11-20.

This version accompanies the revised manuscript *VortFoil: Unified Prediction of Surface States, Aerodynamic Performance, and Flow
Fields across Diverse Airfoils*. The models and weights are those of 2.1.0.

## What changed

- **Boundary-layer output rule.** At every station of both surfaces, `predict` returns θ = max(θ, 10⁻⁶) and δ* = max(δ*, θ)
  (chord units), and the trailing-edge state, the lift and drag closures, the exported profiles and the shape gradients use
  these values. The raw heads were slightly negative at about 1–2% of the stations of the 80-airfoil evaluation sets. The field
  decoder's conditioning is unchanged, and no reported metric moves beyond its last printed digit.
- **Coefficient names.** The application and `vortfoil view` name the three coefficient sets as the manuscript does:
  RANS-calibrated (listed first), panel-based and field-integrated. The `_coefficients.csv` columns keep their names; its column
  notes give the correspondence.
- **Held-out.** The documentation and the field decoder's family warning call the evaluation sections held-out instead of
  unseen, since some of them lie close to training sections.
- **Figures and metadata.** `figures/` holds the revised manuscript's figures with their generators in `code/figures/`, including
  the new field figures (`vf_field_naca`, `vf_field_nonnaca`) and the application figure (`vf_app_window`); each generator
  reproduces its PNG pixel for pixel. `paper/abstract.md` and the citation metadata follow the revised manuscript.

## Attribution and citation

The anti-derivative approximator and the staged-operator base are Prof. Jeongsu Lee's; this
package packages the VortFoil build of that line. Authors: Sang-Hyun Rhie, Gyuwon Kim, Suhyeong Lim and
Jeongsu Lee, Department of Mechanical Engineering, Kyung Hee University. See [`CITATION.cff`](../CITATION.cff),
[`NOTICE`](../NOTICE) and [`DATA_LICENSES.md`](../DATA_LICENSES.md).

## Install or upgrade

```bash
uv tool install --python 3.12 --managed-python vortfoil      # or: uv tool upgrade vortfoil
```

Inside an environment with a pinned TensorFlow, upgrade without touching it:

```bash
pip install --no-deps --upgrade vortfoil
```

Full history is in [`CHANGELOG.md`](../CHANGELOG.md).
