# VortFoil model card

## What this is

This is the deployed surface model. One conditioning vector — 16 Kulfan/CST shape
coefficients, the incidence angle, and the Reynolds number — feeds a sheet operator
(125,634 parameters) and a boundary-layer operator (402,948 parameters) in parallel.
The sheet's panel readout gives surface pressure, circulation lift, quarter-chord
moment, a stagnation point, and the trailing-edge edge velocity. Two frozen
polynomial closures then act on the predicted trailing-edge state: a drag closure
(10 coefficients) gives profile drag, and a lift closure (28 coefficients) gives a
viscous-decambering correction to the circulation lift. So the deployed lift and
drag are not pure sheet readouts; the pressure, moment, stagnation and edge-velocity
outputs are.

## Intended use

- Batched analysis sweeps.
- Gradient-based shape-optimization inner loops. The chain is differentiable end to
  end, and its gradients are verified against finite differences. No constrained
  shape-design example has been validated: independent validation of the resulting
  geometry remains necessary before optimization use (manuscript, Limitations).
- Teaching or inspection of a physically structured surrogate.

This is an XFOIL-state model for the surface branch. It predicts what XFOIL would
return, not what a RANS solver or an experiment would return.

Inputs are expected inside the training range: incidence −10° to 18° and Reynolds number 5×10⁴ to 10⁷ (the evaluation
sets cover −4° to 12° and 2×10⁵ to 3×10⁶). Outside it the package issues `vortfoil.RangeWarning`.

## Out of scope

- Compressible flow.
- Blunt trailing edges. The 16-parameter shape form cannot represent them.
- Shapes far from the training distribution.
- Unsteady flow.
- Three-dimensional flow.
- Any use where being behind the comparator matters. See Limits below.

## Training data

The sheet operator is trained on 449,732 filtered XFOIL rows. The loss combines a
surface-pressure MSE term, a trailing-edge pressure term weighted 10x, a label-free
stagnation self-consistency term weighted 10x, and a direct circulation-lift term
weighted 1x, over 400 epochs.

The boundary-layer operator is trained on 325,370 filtered XFOIL boundary-layer
rows.

Both closures are ridge-fitted once, on the same 325,370 rows used for the
boundary-layer operator, and then frozen. No evaluation row enters either fit.

XFOIL was run at Ncrit 7, incompressible, with 200 panels. Shapes are drawn from
UIUC sections, NACA four-digit sections, and perturbed Kulfan shapes.

## Evaluation

| Population | n | Cl R² | Cl^γ R² (circulation only) | Cd R² | Cm R² |
|---|---|---|---|---|---|
| Converged | 2,678 | 0.9887 | 0.9871 | 0.9205 | 0.9529 |
| Converged, NeuralFoil xlarge | 2,678 | 0.9951 | — | 0.9550 | 0.9963 |
| All validity-filtered | 2,737 | 0.987952 | 0.986230 | 0.890353 | not reported |
| Separated | 1,930 | 0.9941 | — | 0.9338 | 0.8428 |
| Separated, NeuralFoil xlarge | 1,930 | 0.9993 | — | 0.9831 | 0.9944 |

The sixth decimal of the 2,737-row Cd depends on the TensorFlow build: 0.890353 under
TF 2.10 (the value in `MANIFEST.json`), 0.890354 under TF 2.18.

"Converged" is 80 held-out UIUC shapes, incidence up to 12 degrees, restricted to
rows that pass the validity filter and whose XFOIL solution converged. "All
validity-filtered" is the same grid without the convergence test. "Separated" is
the same 80 shapes with the incidence range extended to reach a trailing-edge shape
factor of 6 or more, with no unconverged rows.

The convergence test removes 59 rows that carry no XFOIL surface solution at all,
so no surface metric can be formed on them. This filter is a property of the truth
label, not of any model, and it is applied identically to every model being
compared. Scores from the converged population and the all-validity-filtered
population are never compared against each other.

The paper reports the converged population. The shipped self-check reports scores
on the all-validity-filtered population instead, because that is the population
the shipped per-row reference files were written against; it is a weight-integrity
check, not a reproduction command for the paper's 2,678-row table. Two further
populations exist for the station-matched boundary-layer comparison — 2,688
attached and 2,166 separated rows, read at NeuralFoil's 32 x/c stations — and are
recorded separately in `results/station_bl_compare.json` and
`results/station_bl_compare_sep.json`.

## Comparison

The model is behind NeuralFoil's extra-large model on lift, drag, and moment, on
both the converged and the separated populations. The one measured advantage is
profile drag at negative incidence.

## Performance

The timed unit is the manuscript's **coefficient chain**: the sheet operator, its panel
readouts, the boundary-layer operator, and both frozen closures. It excludes the quarter-chord
moment and the stagnation-location readout, so it is not the cost of `predict()`, which forms
both of those and copies every output to the host.

| Condition | Cost |
|---|---|
| Coefficient chain, compiled, CPU, batch 1024 | 163 µs/case |
| Coefficient chain, eager, CPU, batch 1024 | 317 µs/case |
| Coefficient chain, compiled, RTX 4060 Ti, batch 512 | 126 µs/case |
| Single query, compiled, CPU | 1.30 ms |
| Single query, eager, CPU | 11.3 ms |

Shape gradients cost 2.34x a sheet forward pass for dCl^γ, and 3.13x (dCl) and 3.12x (dCd) the
coefficient chain, verified against central finite differences.

Absolute times are host dependent. The ordering between conditions — compiled
faster than eager, batched cheaper per case than a single query — is the stable
statement.

## Verified environments

- Python 3.10 + TensorFlow 2.10 (CPU, eager and compiled)
- Python 3.10 + TensorFlow 2.10 (NVIDIA RTX 4060 Ti, compiled)
- Python 3.10 + TensorFlow 2.18 with Keras 3.12 (CPU, compiled)

## Limits

- The sheet is read on a fixed 241-node loop and the boundary layer on a fixed
  121-node grid per surface. No resolution independence is claimed.
- The surface branch is trained and evaluated against XFOIL, not against a RANS
  solver or an experiment.
- Both closures are empirical fits over their training range, validated only on
  the two evaluation populations above.
- The model is behind NeuralFoil's extra-large model on all three coefficients on
  both evaluation sets; the only measured advantage is profile drag at negative
  incidence.
- A blunt trailing edge cannot be represented by the 16-parameter shape form.

## Field decoder

A separately trained RANS field decoder also ships, in the package and in the repository (the installed `vortfoil`
command and `code/field/predict_field.py` run it for any section from its coordinates): `weights/field_decoder_f0cm_s1.npz`, conditioned on the sheet's uncorrected circulation
lift and on a boundary-layer-derived separation scalar, s_sep = clip((max(H_TE^u, H_TE^l) − 2)/8,
0, 1). It is trained and tested on k–ω SST AirfRANS fields only (780 training cases, 200 test
cases), a different reference state from the XFOIL surface branch; the two are never compared in
absolute terms.

- Against an XFOIL-conditioned decoder trained on the same cases, its paired error-ratio 95%
  confidence intervals lie below unity on u_x, u_y and p and include unity on ν_t.
- It has lower aggregate errors than MARIO's published values on u_x, p and ν_t, but not on u_y.
  Above 8 degrees incidence its u_x and u_y errors exceed MARIO's published aggregate; in the
  [12, 16) degree band all four channels do.
- Its deployed weight is one of three training runs, selected by validation score; the
  case-bootstrap intervals do not measure variation across training seeds.
- Its fixed spatial-feature and point-cloud budgets leave discretization independence untested.
- Run for a new section by `code/field/predict_field.py`: from dense coordinates alone it matches the deployed evaluation
  (mean errors 0.99, 0.99, 1.07, 1.03 times on the 200 test cases); a sparse 161-point section raises near-wall errors
  on a RANS mesh (u_y 3.8 times) but changes a 0.01-chord grid by under 6% of the decoder's error. Outside AirfRANS's
  NACA 4/5-digit sections, −5° to 15° and Re 2×10⁶ to 6×10⁶ it warns.
- **Other airfoil families:** on ten UIUC sections computed with the AirfRANS protocol inside those ranges, its errors were
  1.5 to 80 times its NACA test level (26 times on average for u_x), largest near the wall. Only near-NACA shapes (RAE 2822,
  NACA 64₂-415, DU 84-132V) stayed within the test tail. The shape-code distance did not flag them, so every non-NACA input
  carries a family warning.

The numbers are in the README's results summary.

## Attribution and license

The anti-derivative approximator and the NeuroFoil staged-operator base are
Prof. Jeongsu Lee's; this package packages the VortFoil build of that line.

The package is licensed under the GNU GPL v3 or later; see [`LICENSE`](LICENSE). Its
boundary-layer closures (`xfoil_closures.py`) are translated from XFOIL 6.99 (Drela and
Youngren, GPL v2 or later); see [`NOTICE`](NOTICE). Data attributions are in
[`DATA_LICENSES.md`](DATA_LICENSES.md).

Authors: Sang-Hyun Rhie (Kyung Hee University), Gyuwon Kim (Kyung Hee
University), Suhyeong Lim (Kyung Hee University), Jeongsu Lee (Kyung Hee University).

## Citation

Rhie, S.-H., Kim, G., Lee, J. VortFoil: A Vortex-Panel-Informed Neural Operator
for Steady Airfoil Flow Prediction (submitted).
