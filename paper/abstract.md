# VortFoil: Unified Prediction of Surface States, Aerodynamic Performance, and Flow Fields across Diverse Airfoils

Sang-Hyun Rhie, Gyuwon Kim, Suhyeong Lim, Jeongsu Lee — Department of Mechanical Engineering, Kyung Hee University

## Abstract

Computational fluid dynamics (CFD) provides detailed aerodynamic information, but its
computational cost limits repeated airfoil evaluations in design and analysis. Existing
surrogate models offer faster alternatives, yet typically address aerodynamic coefficients,
surface states, and spatial flow fields through separate predictive approaches. Here, we present
VortFoil, a unified framework that connects these levels of aerodynamic prediction across
diverse airfoil geometries and operating conditions. The framework represents effective
vortex-sheet strength and boundary-layer thickness as continuous functions using neural
anti-derivative approximators, combining learned spatial rates with analytical integration. A
discrete Kutta projection and deterministic panel readouts recover surface pressure,
circulation, and stagnation location, while boundary-layer representations provide trailing-edge
states for aerodynamic closures. Surface-state-derived quantities support RANS-calibrated
coefficient prediction and condition a neural decoder for spatial flow reconstruction, linking
surface-level aerodynamic information to performance estimates and surrounding flow fields. The
framework achieves mean relative lift and drag errors of 2.9% and 4.3% against RANS on NACA
airfoils, while training with a broader non-NACA corpus reduces non-NACA flow-field errors to
30–50% of those obtained with AirfRANS-only training under matched optimizer steps. VortFoil
provides an accessible computational framework for multilevel aerodynamic analysis, with
intermediate physical states available alongside predicted coefficients and flow fields.

---

Transcribed from the manuscript source; the compiled manuscript is the authoritative version.
The deployed build's own record — weight hashes, training recipe, evaluation populations and the
scores this package reproduces — is in `src/vortfoil/weights/MANIFEST.json` and is printed by
`vortfoil.about()`.
