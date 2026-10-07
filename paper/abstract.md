# VortFoil: A Vortex-Panel-Informed Neural Operator for Steady Airfoil Flow Prediction

Sang-Hyun Rhie, Gyuwon Kim, Jeongsu Lee — Department of Mechanical Engineering, Kyung Hee University

## Abstract

VortFoil links aerodynamic coefficients to predicted states through a fixed-resolution neural
operator. Network-predicted panelwise rates are projected onto a fixed Fourier basis and
analytically integrated to form an effective vortex sheet. A projection enforces the discrete
Kutta condition, and panel readouts provide surface pressure, circulation lift, moment, and
stagnation location without stagnation-position labels. A boundary-layer operator predicts
thickness profiles in parallel from the same geometry and operating conditions. Drag and lift
follow from state-conditioned empirical closures fitted by ridge regression and frozen at
deployment. Selected readouts condition a separately trained Reynolds-averaged Navier–Stokes
(RANS) field decoder. On the attached XFOIL holdout of 80 unseen airfoils, median
pressure-coefficient relative L2 and stagnation-location errors are 0.100 and 0.77 panels. Drag R²
is 0.920 (attached set) and 0.934 (separated set); NeuralFoil achieves higher lift, drag, and
moment R² on both sets. Trained on the same AirfRANS cases as an XFOIL-conditioned decoder, the
operator-conditioned field decoder has paired error-ratio 95% confidence intervals, relative to that
decoder, below unity on u_x, u_y and p; the ν_t interval includes unity. It has lower aggregate
errors than a published field-operator baseline on u_x, p and ν_t, but not on u_y, and its velocity
errors above 8° incidence exceed that baseline's published aggregate. Surface lift and drag shape
gradients are checked against finite differences of the same single-precision forward map.

---

Transcribed from the manuscript source; the compiled manuscript is the authoritative version.
The deployed build's own record — weight hashes, training recipe, evaluation populations and the
scores this package reproduces — is in `src/vortfoil/weights/MANIFEST.json` and is printed by
`vortfoil.about()`.
