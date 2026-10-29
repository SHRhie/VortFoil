# RANS coefficient correction by a supervised neural network on surface coefficients and operating conditions: pre-specification
(2026-10-15, after the professor meeting, before any fit)

## Direction (professor, 10-15 meeting; confirmed by the user)
Correct the surface branch's coefficients with a neural network supervised on RANS, conditioned on the operating condition:
Coeff* = NN(C_l, C_d, C_m from the surface, alpha, Re). No shape (CST) inputs and no boundary-layer states: on 10-14 the ridge
correction with CST inputs collapsed on non-NACA UIUC sections, while the version without shape inputs transferred
(VortFoil/artifacts/rans_correction_20261014/uiuc_check). This replaces the ridge correction of the v3 manuscript.

## Data (unchanged from 10-14)
`VortFoil/artifacts/rans_correction_20261014/`: AirfRANS train 780 (fit), val 20 (report), test 200 (report, never used to fit or
select); RANS C_l, C_d, C_m from the AirfRANS force routine. Non-NACA: `uiuc_check/` (107 UIUC SST sections; report only).
Bases: VortFoil surface branch (primary), XFOIL free/forced and NeuralFoil free/forced (same pipeline, for comparison).

## Model
- Inputs (standardized with training statistics): C_l, ln C_d, C_m, alpha [deg], log10 Re.
- Targets (standardized): C_l^RANS - C_l, ln(C_d^RANS / C_d), C_m^RANS - C_m (additive lift and moment, multiplicative drag).
- MLP 5 -> 32 -> 32 -> 3, tanh, full-batch Adam (learning rate 1e-3), mean-squared error plus L2 weight decay lambda.
- lambda in {1e-4, 1e-3, 1e-2} and epochs in {500, 1000, 2000, 4000} chosen by 10-fold cross-validation on the 780 training cases
  (folds: permutation(rng 0) mod 10, as for the ridge correction), criterion = mean standardized CV MSE of the three targets.
- Final model: the chosen setting trained on all 780 cases, an ensemble mean of 5 initialisations (seeds 0-4).

## Reporting
- C_l and C_d: mean relative error in percent; C_m: MAE; also Spearman rho_L, rho_D and the mean C_d ratio.
- NACA test (200) and the 107 non-NACA UIUC SST sections, uncorrected and corrected, per base; paired case-bootstrap differences
  (B = 10,000, seed 0) corrected minus uncorrected for VortFoil.
- For reference only: the 10-14 ridge corrections (with shape inputs; with states) from the stored results.
