# VortFoil 2.0.0

First published release of `vortfoil`: a vortex-panel-informed neural operator for steady airfoil
flow. Surface pressure, circulation lift, the moment and the stagnation point are readouts of one
predicted vortex sheet; drag and the deployed lift correction are frozen closures on the state of a
parallel boundary-layer operator. The coefficient and surface chain is differentiable with respect to
the CST shape coefficients.

## Install

```bash
pip install vortfoil
```

Inside an environment with a pinned TensorFlow, install without touching it:

```bash
pip install --no-deps vortfoil
```

Python 3.10 or newer. TensorFlow 2.10 through 2.18 are tested, on CPU and on an NVIDIA GPU.

## Use

```python
import vortfoil, numpy as np

vf = vortfoil.load_model()                       # compiled, CPU
r  = vf.predict(cst=cst16, alpha=4.0, Re=1e6)    # or coords=(P,2) airfoil points
r.cl, r.cd, r.cm, r.cp, r.stagnation

vf.sweep(cst16, np.linspace(-4, 12, 33))         # one shape, many incidences, one batch
vf.gradients(cst=cst16, alpha=4.0, of="cl")      # reverse-mode d/dshape through the whole chain
vortfoil.about()                                 # weights, recipe, populations, scores, limits
```

A shape given as coordinates is fitted to the 16-parameter form first, and the incidence is
measured from the fitted chord line, so a rotated or scaled section is handled correctly. The fit
residual and the rotation come back with the result.

## What is in the model

A network predicts a rate function whose closed-form anti-derivative is the surface vortex sheet,
and a linear projection imposes the discrete Kutta condition exactly. Surface pressure, the
circulation lift, the quarter-chord moment, the stagnation point and the trailing-edge edge
velocity are panel-method readouts of that sheet, with no fitted constants of their own. A second
bank of heads predicts the integral boundary layer. The two deployed force coefficients are not
pure readouts: profile drag is a frozen ridge-fitted closure on the predicted trailing-edge state,
and the deployed lift is the circulation readout plus a frozen ridge-fitted viscous-decambering
correction read from that same state.

No XFOIL or CFD process is launched at inference. That is a statement about external solver calls,
not about fitting: both force closures are empirical fits frozen at deployment.

## Field decoder (repository only)

The repository, not the pip package, also ships the separately trained RANS field decoder of the
manuscript: `weights/field_decoder_f0cm_s1.npz`, conditioned on the sheet's circulation lift and a
boundary-layer-derived separation scalar, and trained and tested on k–ω SST AirfRANS fields only.
Against an XFOIL-conditioned decoder trained on the same 780 cases, its paired error-ratio 95%
confidence intervals lie below unity on u_x, u_y and p and include unity on ν_t. It has lower
aggregate errors than MARIO's published values on u_x, p and ν_t, but not on u_y, and its velocity
errors above 8° incidence exceed MARIO's published aggregate. The README gives the numbers and the
scripts that reproduce them.

## Recorded scores

Eighty held-out UIUC sections against XFOIL at Ncrit 7. The attached set is the rows whose XFOIL
solution converged; the separated set extends the incidence range until the trailing-edge shape
factor reaches six.

| | n | Cl | Cd | Cm |
|---|---|---|---|---|
| attached | 2,678 | 0.989 | 0.920 | 0.953 |
| attached, NeuralFoil extra-large | 2,678 | 0.995 | 0.955 | 0.996 |
| separated | 1,930 | 0.994 | 0.934 | 0.843 |
| separated, NeuralFoil extra-large | 1,930 | 0.999 | 0.983 | 0.994 |

Median surface-pressure relative L2 error is 0.100 and median stagnation-location error is 0.77
panels, both on the attached set. `vortfoil selfcheck` reproduces the package's own recorded
scores; those are written on the wider validity-filtered population of 2,737 rows, which is not
the population in the table above.

## Honest comparison

This model is behind NeuralFoil's extra-large model on lift, drag and moment, on both evaluation
sets. The one measured advantage is profile drag at negative incidence. What the package offers
instead is that every number is a readout of an inspectable predicted state, and that shape
gradients are available by reverse mode and are checked against finite differences of the same
forward map. The readouts are not forced to agree with one another: the training objective does
not enforce equality of the circulation and pressure-force lift readouts, or of the two moment
readouts, and their remaining disagreement is measured and reported.

## Cost

The coefficient chain is the sheet pass, its panel readouts, the boundary-layer pass and both
frozen closures. It excludes the moment and the stagnation-location readout, so it is not the cost
of `predict()`.

| condition | cost per case |
|---|---|
| compiled, CPU, batch 1024 | 163 µs |
| eager, CPU, batch 1024 | 317 µs |
| compiled, RTX 4060 Ti, batch 512 | 126 µs |
| single query, compiled, CPU | 1.30 ms |

Absolute times are host dependent; the ordering between conditions is the stable statement.

## Limits

- Fixed discretization: the sheet is read on 241 nodes and the boundary layer on 121 nodes per
  surface. No resolution independence is claimed.
- The surface branch predicts what XFOIL would return, not a RANS solution or an experiment.
- Both closures are empirical fits over their training range.
- A blunt trailing edge cannot be represented by the 16-parameter shape form.
- Incompressible, steady, two-dimensional flow only.

## Relation to the earlier package

The July 2026 package on this repository shipped a separately trained coefficient regressor
alongside the physics route. That regressor is gone; this release ships one confirmed model, and
every coefficient is tied to a predicted state. The 2.0.0 number matches the manuscript's model
generation and does not follow a prior 1.x package release, since there was none.

Full history is in [`CHANGELOG.md`](../CHANGELOG.md); intended use and limits are in
[`MODEL_CARD.md`](../MODEL_CARD.md).

## Attribution and citation

The anti-derivative approximator and the staged-operator base are Prof. Jeongsu Lee's; this
package packages the VortFoil build of that line. Authors: Sang-Hyun Rhie, Gyuwon Kim, Suhyeong Lim and
Jeongsu Lee, Department of Mechanical Engineering, Kyung Hee University.

See [`CITATION.cff`](../CITATION.cff). Licensed under the GNU GPL v3 or later, because the
boundary-layer closures are translated from XFOIL; see [`NOTICE`](../NOTICE) and, for the data,
[`DATA_LICENSES.md`](../DATA_LICENSES.md).
