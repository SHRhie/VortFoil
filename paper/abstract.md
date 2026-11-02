# VortFoil: A Vortex-Panel-Informed Neural Operator for Steady Airfoil Flow Prediction

Sang-Hyun Rhie, Gyuwon Kim, Suhyeong Lim, Jeongsu Lee — Department of Mechanical Engineering, Kyung Hee University

## Abstract

VortFoil is a vortex-panel-informed neural operator framework that connects airfoil
aerodynamic performance prediction with RANS flow reconstruction through continuous
aerodynamic-state representations. Vortex-sheet strength and boundary-layer thickness
distributions are represented as continuous profiles along the surface by analytic integration
of network-predicted spatial rates. A discrete Kutta projection and deterministic panel readouts
give surface pressure, circulation lift, moment and stagnation location, and frozen empirical
closures map the trailing-edge states to lift and drag. Selected readouts condition a RANS decoder
trained jointly on AirfRANS and 16,923 non-NACA UniFoil airfoils without a turbulence-model input
or inference-time XFOIL calls. Against XFOIL on 80 unseen UIUC airfoils (−10°–18°), attached-set
(−4°–12°, n = 2678) median pressure-coefficient relative L2 error is 0.100 and label-free
stagnation error 0.77 panels; separated-set drag R² is 0.934 versus 0.685 for the marched wake.
In AirfRANS-only ablations, operators' states lower velocity and pressure errors relative to
XFOIL conditioning, most strongly at high incidence. The field decoder has the lowest
surface-pressure error among the compared models and field-integrated lift R² 0.9990 against
RANS; its 5.4% mean relative lift error exceeds those of forced-transition NeuralFoil and XFOIL.
With surface weights frozen, a neural RANS correction lowers mean relative NACA lift/drag errors
from 7.7%/25.9% to 2.9%/4.3%, and drag error from 30.2% to 8.9% on 107 unseen non-NACA SST
sections. Against equally long AirfRANS-only training, UniFoil airfoils lower all four errors on
unseen non-NACA SST sections (ratios 0.30–0.50), but not AirfRANS errors; Spalart–Allmaras
standardized MSEs are 4.58e-5–2.94e-4 on 150 held-out UniFoil airfoils. Coupled through a
nonlinear lifting line, the sectional predictions reproduce three-dimensional RANS wing lift
within a mean absolute error of 5.1% at moderate incidence.

---

Transcribed from the manuscript source; the compiled manuscript is the authoritative version.
The deployed build's own record — weight hashes, training recipe, evaluation populations and the
scores this package reproduces — is in `src/vortfoil/weights/MANIFEST.json` and is printed by
`vortfoil.about()`.
