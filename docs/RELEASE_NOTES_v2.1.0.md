# VortFoil 2.1.0

Release date: unreleased.

Accompanies the revised manuscript. The deployed two-dimensional field decoder is now trained on
AirfRANS together with 16,923 non-NACA UniFoil airfoils, and a frozen correction maps the surface
branch's coefficients to the RANS state. The surface operators and their lift and drag closures
are unchanged.

## Install

The application and the `vortfoil` command, on Windows, macOS and Linux, with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install --python 3.12 --managed-python vortfoil
vortfoil                       # opens the application window
```

As a library, in your own environment:

```bash
pip install vortfoil
```

Inside an environment with a pinned TensorFlow, install without touching it:

```bash
pip install --no-deps vortfoil
```

Python 3.10 or newer. The tests and the self-check run on TensorFlow 2.10, 2.15 and 2.18 on CPU, and the model was
timed on an NVIDIA GPU; the uv installation is tested on Windows, macOS and Linux with the TensorFlow it resolves
(2.21 when the 2.0.0 release was prepared). The README lists the steps to uninstall.

## Use

```python
import vortfoil, numpy as np

vf = vortfoil.load_model()                       # compiled, CPU
r  = vf.predict(cst=cst16, alpha=4.0, Re=3e6, rans=True)
r.cl, r.cd, r.cm                                 # surface branch, XFOIL state
r.cl_rans, r.cd_rans, r.cm_rans                   # frozen correction, RANS state
r.cp, r.stagnation

vf.sweep(cst16, np.linspace(-4, 12, 33))         # one shape, many incidences, one batch
vf.gradients(cst=cst16, alpha=4.0, of="cl")      # reverse-mode d/dshape through the surface chain
vortfoil.about()                                 # weights, recipe, populations, scores, limits
```

From the command line:

```bash
vortfoil predict --coords section.dat --alpha 4 --Re 3e6 --rans
vortfoil view --airfoil 2412 --alpha 4 --Re 3e6                        # field, C_p, boundary layer, coefficients
vortfoil view --airfoil section.dat --alpha 4 --Re 3e6 --csv results/   # and every number as CSV files
```

`vortfoil predict --rans` and `vortfoil sweep --rans` print the corrected coefficients.
The correction is also available through `vortfoil.rans`. Each command takes `--help`.

A shape given as coordinates is fitted to the 16-parameter form first. `vortfoil sweep --coords`
now measures the incidence from the fitted chord line, as `predict` does: before this fix, a
section rotated by 2° gave C_l off by 0.22. The fit residual and the rotation come back with the
prediction result.

## What is in the model

The sheet operator, its panel readouts, the boundary-layer operator and the frozen lift and drag
closures are the surface chain of 2.0.0. The field decoder keeps its architecture and its two
conditioning inputs: the sheet's circulation lift and a boundary-layer-derived separation scalar.
No XFOIL or CFD process is launched at inference.

The added RANS coefficient correction, `weights/rans_correction_nn.npz` (also packaged), is the
mean of five 5-32-32-3 tanh networks on the surface branch's (C_l, ln C_d, C_m, α, log₁₀ Re).
With the operators and closures frozen, it returns C_l + f_l, C_d·exp(f_d) and C_m + f_m. It has no
shape or boundary-layer-state input. The fit uses the 780 AirfRANS training cases under a protocol
fixed before fitting (`results/rans_correction_nn_20261015_PRESPEC.md`). This is a fitted
coefficient correction, separate from integrating the decoder's field.

## Flow field and the application

`predict_field`, `vortfoil view` and the two-dimensional application now use
`weights/field_decoder_cmx_s0.npz`. It is trained on the 780 AirfRANS k–ω SST cases together with
16,923 non-NACA UniFoil airfoils: fully turbulent Spalart–Allmaras fields at M ≤ 0.2, using every
third stored row, with 150 airfoils held out. There is no turbulence-model input.

Run 0 of three is selected on the 20 AirfRANS validation cases. Its mean standardized channel MSE
is 0.00257; runs 1 and 2 give 0.00458 and 0.00330
(`results/cotrain_score_cmx_20261015T023434Z.json`, `selection`).

The experimental three-dimensional wing's RANS state keeps the AirfRANS-only decoder of 2.0.0,
`field_decoder_f0cm_s1.npz` (`wing_nll.WING_DECODER`). This change of the two-dimensional default
does not change that wing's decoder.

## Recorded scores

The deployed field errors are standardized MSEs over all mesh nodes, with equal weight per case,
in the channel order u_x, u_y, p and ν_t
(`results/cotrain_score_cmx_20261015T023434Z.json`).

| set | u_x | u_y | p | ν_t |
|---|---|---|---|---|
| AirfRANS test, 200 cases (k–ω SST) | 6.52e-4 | 9.69e-4 | 1.52e-3 | 3.52e-3 |
| 16 unseen non-NACA UIUC sections (k–ω SST) | 9.13e-3 | 1.50e-2 | 7.04e-2 | 1.34e-1 |
| 150 held-out UniFoil airfoils (Spalart–Allmaras; stored row subsample) | 3.00e-4 | 9.27e-5 | 2.20e-4 | 5.45e-5 |

The UniFoil row is not comparable with the SST rows. The metric's σ comes from all nodes of a
50-case sample of the 200-case scarce-task training split, rather than the full training split
previously described in the README. The published convention takes every node of the task's
training split; the values here are therefore conservative against published ones.

Against an AirfRANS-only decoder trained for as many optimizer steps, the paired error ratios
(deployed / control) are 1.00, 1.36, 0.57 and 0.51 on AirfRANS and 0.39, 0.30, 0.44 and 0.50 on
the 16 non-NACA sections (`results/stepmatch_score_20261015T070933Z.json`). The NACA velocity
gains come from training length; the non-NACA gains come from the UniFoil airfoils.

The RANS correction gives the following coefficient errors against SST values
(`results/rans_correction_nn_20261015.json`):

| set | metric | surface branch | RANS-corrected |
|---|---|---|---|
| AirfRANS test, 200 cases | lift mean relative error | 7.7% | 2.9% |
| | drag mean relative error | 25.9% | 4.3% |
| | moment MAE | 0.0082 | 0.0028 |
| 107 unseen non-NACA SST sections | lift MAE | 0.037 | 0.028 |
| | drag mean relative error | 30.2% | 8.9% |
| | moment MAE | 0.0096 | 0.0071 |

The field's own wall readouts use the same routine as before. On the 200 AirfRANS test cases
(`results/table4_row_cmx_s0.json`), wall-pressure standardized MSE is 4.5e-3, lift mean relative
error is 5.4% (Spearman 0.9995), and moment MAE is 0.0023. Drag mean relative error is 45.5%
from the wall gradient and 12.8% with wall-function friction (Spearman 0.641 and 0.743).

The corrected coefficients and the field decoder's integrated ones are not constrained to agree.
They differ by 6.8% in lift and 14.1% in drag on the test split
(`results/consistency_field_vs_corrected_20261029.json`).

## Limits

- The surface branch retains its 2.0.0 limits: steady, incompressible, two-dimensional flow;
  fixed discretization; a closed trailing edge in the 16-parameter shape form; XFOIL-state
  predictions with empirical lift and drag closures. Its training range is incidence −10° to
  18° and Reynolds number 5×10⁴ to 10⁷.
- The RANS correction issues `vortfoil.RangeWarning` outside incidence −5° to 15° or Reynolds
  number 2×10⁶ to 6×10⁶. It predicts coefficients, not a corrected surface pressure or field.
- Adding UniFoil reduces the field's errors on other airfoil families but does not remove the
  family gap. On the 16 unseen non-NACA SST sections its errors are 14, 15, 46 and 38 times its
  NACA test level; the AirfRANS-only decoder gives 27, 35, 68 and 40 times. The non-NACA family
  warning now reports the deployed decoder's own errors. Read such a field as indicative.
- The field's RANS-state boundary layer and wall values depend on its near-wall accuracy.
  The RANS coefficient correction is a separate route and does not make those readouts agree
  with it.

## Relation to the earlier package

The surface model of 2.0.0 is retained. The repository adds the deployed decoder's other runs,
`field_decoder_cmx_s{1,2}`, the step-matched AirfRANS-only controls `field_decoder_sml2_s*`, and
the decoders trained with 1,346 UniFoil airfoils, `field_decoder_cmd_s*`. The new records are
`results/cotrain_score_*`, `stepmatch_score_*`, `rans_correction_nn_*`,
`table4_row_cmx_s0.json` and `consistency_field_vs_corrected_20261029.json`.

The field validation scripts pin the decoder their shipped records were written with. The
AirfRANS-only scores, validation records and field figure from 2.0.0 remain records of that
decoder, rather than measurements of the new default.

Full history is in [`CHANGELOG.md`](../CHANGELOG.md); intended use and limits are in
[`MODEL_CARD.md`](../MODEL_CARD.md).

## Attribution and citation

The anti-derivative approximator and the staged-operator base are Prof. Jeongsu Lee's; this
package packages the VortFoil build of that line. Authors: Sang-Hyun Rhie, Gyuwon Kim, Suhyeong Lim and
Jeongsu Lee, Department of Mechanical Engineering, Kyung Hee University.

See [`CITATION.cff`](../CITATION.cff). The code is licensed under the GNU GPL v3 or later, because
the boundary-layer closures are translated from XFOIL; see [`NOTICE`](../NOTICE).
The UniFoil-trained weights (`field_decoder_cmx_s*`, `field_decoder_cmd_s*`) are made available
under CC BY-SA 4.0 with attribution to UniFoil (Kanchi et al., 2025), and carry the AirfRANS
notice. No UniFoil data are shipped. See [`DATA_LICENSES.md`](../DATA_LICENSES.md) for the
attributions and data licenses.
