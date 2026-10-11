# `code/` layout

One-line purposes below are taken from each script's own docstring.

Every script resolves the release root, weights and evaluation data through
[`vortfoil_paths.py`](vortfoil_paths.py). Overrides:

| variable | what it points at |
|---|---|
| `VORTFOIL_ROOT` | release root (default: this repository) |
| `VORTFOIL_WEIGHTS` | weights directory (default: `<root>/weights`) |
| `AIRFRANS_ROOT` | the external AirfRANS `Dataset/` directory |
| `VORTFOIL_RAW` | prediction/target dumps not shipped here (default: `<root>/data/eval`) |
| `VORTFOIL_EVAL_RUNS` | the sealed f0a evaluation runs (default: `<root>/evaluation_runs`, shipped) |
| `VORTFOIL_CORPUS_DIR`, `VORTFOIL_TRAIN_CORPUS`, `VORTFOIL_DECKS` | training corpora / XFOIL decks (not shipped) |
| `VORTFOIL_FIG_CASE_SELECTION` | the figure-case selection (default: `<root>/results/fig_case_selection.json`, shipped) |
| `ATTACHED_CP_TARGETS`, `UIUC_CST_NPZ`, `FLUENT_NACA0012_DIR` | further inputs that are not shipped; each use site carries a comment |
| `XFOIL_EXE` | XFOIL binary, needed only to regenerate labels |

Anything a script opens by a leading-underscore name (`_nf_train*.npz`, token
caches, intermediate JSON) is an artifact of the research tree and is **not**
shipped; the training corpora cannot be rebuilt byte for byte (see the repository
README and `REFACTOR_MANIFEST_v2.json`'s `not_included` list).

## Top level

| file | purpose |
|---|---|
| `vortfoil_paths.py` | Release-root path resolution for the VortFoil single-model chain. |
| `smoke_confirmed_chain.py` | Smoke test of the confirmed VortFoil surface chain (OP1 + OP2 + drag closure). Run under TF 2.10. |
| `smoke_field_decoder.py` | Smoke test of the deployed operator-conditioned field decoder (`weights/field_decoder_f0cm_s1.npz`, separation scalar by the `teH` rule). Run under TF 2.18 (needs the `airfrans` package). |

## `code/model/` — the operator definition and its trainers

| file | purpose |
|---|---|
| `op2_dense_surrogate.py` | Op2 BL Surrogate — pure supervised, predicts (θ, δ\*) per branch. Head-mode-aware variant used for `op2_m325_BT`. |
| `train_op2_dense.py` | OP2 — pure supervised training against XFOIL viscous (θ, H) targets. |
| `fit_closure.py` | Fit a Squire-Young-class closure Cd = f(theta_TE, H_TE, ue_TE) on the TRAINING deck, evaluate on the published holdout. |

### `code/model/staged/`

| file | purpose |
|---|---|
| `op1_inviscid.py` | Op1 — Inviscid panel-method PINN (Phase A.4: Kulfan CST generalization). |
| `op2_bl_surrogate.py` | Op2 BL Surrogate — pure supervised, predicts (θ, δ\*) per branch. |
| `train_op1_supervised.py` | OP1 inviscid — pure supervised training against XFOIL inviscid Cp. |

### `code/model/staged/common/`

| file | purpose |
|---|---|
| `adaf.py` | Anti-Derivative Approximator (ADAF) — Fourier basis + analytical integration. |
| `bl_closures.py` | Boundary-layer closure relations (Drela / XFOIL standard correlations). |
| `bl_residuals.py` | BL closure (Cf, f_θ, f_δ) and residual computations. |
| `branch.py` | Branch-side sampling (upper/lower) from the closed contour. |
| `branch_trunk.py` | Branch-Trunk → ADAF raw-weight head. |
| `datasets.py` | Random airfoil + alpha + Re samplers for staged training. |
| `geometry.py` | Geometry primitives for airfoils — NACA 4-digit (legacy) and Kulfan CST. |
| `interp.py` | Numerical primitives: 1-D interpolation and finite differences. |
| `networks.py` | Backbone neural-network building blocks for the staged operator pipeline. |
| `panel.py` | Linear vortex / source panel kernels with analytical influence coefficients. |
| `profile_encoder.py` | ProfileEncoder — 1D causal dilated conv on airfoil (x, y) profile. |
| `scaling.py` | Input-scaling helpers for the staged operator pipeline. |
| `source.py` | Source-related primitives for the viscous correction stage. |
| `wake_ode_tf.py` | TF (batched, differentiable) wake-BL ODE integrator; carries the `fitted_poly2`, `squire_young` and `legacy_ode` closures. |
| `xfoil_closures.py` | 1:1 Python/TensorFlow translation of XFOIL 6.99 BL closures (Drela 1989+). |

The dataset samplers and trainers keep their research-tree relative defaults
(`staged/data/…`, `staged/weights/…`); those decks are not shipped, so pass the
paths explicitly or run from a tree that has them.

## `code/field/` — the field track and its state provider

| file | purpose |
|---|---|
| `state_provider.py` | Gate 0 (2026-09-18): the contract-compliant field state provider, built from the CONFIRMED surface model. (Was `gate0_confirmed_state_provider.py`.) `s_rule="teH"` is the operator-conditioned decoder's rule, s = clip((max(H_TE,u, H_TE,l) − 2)/8, 0, 1) (manuscript Eq. 15), for its training rows and every evaluation; `"teH_mean"` (the mean of the two values) is used only by the evaluation-rule sensitivity arm of `f0cm_final_rescore.py`; `"tail"` is the Gate 0 thresholded rule (the manuscript's thresholded alternative, `results/gate0_calibration.json`). |
| `cst16_fit.py` | Least-squares CST16 fit of a G241 contour, in the exact form `staged/common/geometry.py` evaluates. |
| `f0cm_final_rescore.py` | Every manuscript field number of the operator-conditioned decoder (deployed run 1, `field_decoder_f0cm_s1.npz`) against the XFOIL-conditioned run 2: Tables 11–12, the paired ratios, the drop-in and lift-input substitution arms and the separation-rule sensitivity; writes `results/f0cm_final_rescore_<utc>/`. `--limit N` checks rows 0..N−1 of the shipped record `results/f0cm_final_rescore_20261002T051450Z/` and writes nothing. |
| `s_rule_meanside_rescore.py` | Dependency of `f0cm_final_rescore.py`: the scoring helpers it imports (`arr`, the paired case bootstrap `paired`, MARIO's published aggregate, the incidence bands), and its route to `state_provider` and `step2_diag_f0c_cl`. |
| `field_holdout_score.py` | Validation-holdout score of field-decoder weight files, computed as the trainer's `holdout()` does: the selection scores of both deployed decoders (`results/f0a_holdout_rescore_20260930.jsonl`, `results/f0cm_holdout_rescore.json`; needs the training corpora, `VORTFOIL_CORPUS_DIR`). |
| `step2_rebuild_corpus_provider.py` | Step 2a: build the operator-input field training corpus from the XFOIL-input corpus: paint recomputed at the stored rows with the sheet's C_l^γ, s from the boundary-layer operator with `S_RULE = "teH"`. Neither corpus is shipped; the operator-conditioned decoder's corpus is `_nf_train_v5_full800_provider_maxs.npz` (sha256 `a387d918…`, in its weights' `run_config`). |
| `step2_eval_f0c.py` | Step 2b: evaluate the operator-conditioned decoder runs (default `--weights-tag f0cm`, rule `teH`), seed-pooled and paired against the three sealed XFOIL-conditioned runs; `results/step2_eval_f0cm_20260925T213247Z/` is its record. |
| `step2_store_preds.py` | Store full-field predictions of an operator-conditioned decoder for the figure cases (rule `teH`); Fig. 6 uses `--weights-tag f0cm --seed 1 --cases 128`. |
| `vf_field_fast.py` | Faster kernels for the field pathway with the same arithmetic; `verify()` checks them against the originals on `field_decoder_f0cm_s1.npz`. |
| `predict_field.py` | The deployed operator-conditioned decoder for any section, incidence and Reynolds number, from coordinates alone (no AirfRANS files, no XFOIL): `--coords section.dat` or `--naca 2412`, `--alpha`, `--Re`, a grid (`--window`, `--h`) or `--points`, in the wind or chord frame; writes the field (AirfRANS units and normalized) to an npz and, with `--png`, a quick-look figure. Warns (`FieldRangeWarning`) outside the decoder's training domain (AirfRANS: NACA 4/5-digit sections, incidence −5 to 15°, Re 2×10⁶ to 6×10⁶) or for a section whose shape code lies far from every training section. `surface()` gives the decoder's RANS state at the wall (C_p, wall-function C_f, lift, drag, moment) and `boundary_layer()` its boundary layer (θ, δ*, H, edge speed on wall-normal rays through its flow). |
| `naca_generator.py` | The NACA 4/5-digit generator the AirfRANS sections were built with, copied verbatim from the airfrans package 0.1.5.1 (MIT License; see `NOTICE`); `predict_field.py --naca` uses it. |
| `validate_predict_field.py` | Checks `predict_field.py` on AirfRANS full_test (200 cases) against the deployed evaluation: the same inputs reproduce the recorded errors; the coordinates-only route, from the case's 441-node section and from a sparse 161-node one, is scored against RANS; writes `results/predict_field_validation.json` and `_per_case.npz`. Needs `AIRFRANS_ROOT`. |
| `validate_field_boundary_layer.py` | Checks `predict_field.boundary_layer()` on AirfRANS full_test (200 cases): the same extraction (wall-normal rays, edge criterion, θ, δ*, H) applied to the RANS field and to the decoder's field from the coordinates alone, scored per surface, region (leading edge, aft, trailing edge) and attached or separated station against the bound set beforehand (median relative error ≤ 0.20 on attached stations at x/c 0.1–0.95); writes `results/field_boundary_layer_validation.json`. Needs `AIRFRANS_ROOT` and pyvista; `--workers` runs cases in parallel. |

### `code/field/supervised_field/`

| file | purpose |
|---|---|
| `airfrans_p1_labels.py` | MARIO comparison P1: generate the XFOIL labels. |
| `airfrans_p3_fields_v2.py` | Field-construction v2 on the same 20 cases. |
| `field_features.py` | THE canonical field-feature assembler. |
| `field_state_contract.py` | The injected state-provider contract for the field model. |
| `nf_extract.py` | Training-tensor extraction for the MARIO-class field model. |
| `nf_extract_shape.py` | Per-case SHAPE TOKENS for the geometry-conditioned encoder. |
| `nf_train.py` | MARIO-class conditional neural field on the AirfRANS point cloud. |
| `nf_train_shape.py` | Geometry-conditioned field operator (`DESIGN_geomcond.md` winner). |
| `nf_eval200.py` | CITABLE full_test evaluation of the v1 field model. |
| `nf_eval200_v5.py` | CITABLE full_test evaluation of the NF v5 field models. |
| `nf_eval_shape.py` | d-binned (all / near-wall / off-body) std-MSE eval for the geometry-conditioned shape-encoder field model. |
| `nf_hybrid_field.py` | Per-channel HYBRID field model. |

`nf_train_shape.py` trained every shipped field decoder (sealed source `0d8ed19d`, the `code_sha256` in each
decoder's `run_config`); this copy differs from it only in its release paths, comments and printed strings.
In the other modules of this package, only paths, comments, printed strings, the name of the MARIO
constant (`MARIO_PUBLISHED`) and the JSON key it is written under (`mario_published`, in `nf_eval200.py`,
`nf_eval200_v5.py` and `nf_train.py`) differ from the research tree, and `nf_extract_shape.py` keeps only
its AirfRANS extraction.
In `airfrans_p3_fields_v2.py`, `nf_extract.py` and `nf_eval_shape.py` the `airfrans` and `pyvista` imports sit inside
the functions that read AirfRANS files (2026-10-08), so the paint and the decoder forward run without them
(`predict_field.py`). `state_provider.ConfirmedStateProvider.state_for_polygon` gives the operator state for any
closed section; `state_for` passes it the NACA polyline, as before.

### `code/field/sdf2d/`

| file | purpose |
|---|---|
| `geometry.py` | Signed-distance and mesh geometry helpers for the 2D field grid. |
| `panel_inviscid.py` | Standalone 2D inviscid panel method for NACA airfoils. PURE PHYSICS. |

## `code/reproduce/` — the measurement scripts behind the reported numbers

| file | purpose |
|---|---|
| `eval_on_targets.py` | Evaluate the physics drag route + NeuralFoil on an arbitrary holdout-schema targets npz. |
| `eval_op2_cd2.py` | Evaluate the physics drag route for a given OP2 weight file, on the published clone holdout. |
| `surface_contract_cst16.py` | Surface-representation contract for the SINGLE confirmed model (OP1 scr_bd128, CST-16, 19-input). |
| `surface_cp_stag_cst16.py` | Cp, stagnation and Cm readouts of the confirmed single model against XFOIL truth. |
| `station_bl_compare.py` | Station-matched boundary-layer comparison on the attached grid. |
| `nf_regime_search.py` | Where does the physics route beat NeuralFoil xlarge, if anywhere? |
| `run_nf_full.py` | NeuralFoil xlarge with FULL outputs (CL, CD, CM, Xtr, TE BL state per side) on a coords dump. |
| `gate0_calibrate_s_proxy.py` | Gate 0 pre-step: calibrate the OP2 → separation-scalar proxy on the 800 TRAIN cases (`results/gate0_calibration.json`: the thresholded alternative's training MAE 0.121 against 0.126 for the zero predictor). |
| `step2_diag_f0c_cl.py` | Dependency of `f0cm_final_rescore.py`: `ClOverrideProvider`, the lift-input override of its C_l substitution arms (XFOIL's forced-transition C_l, or the provider C_l less the 0.0294 training bias). |
| `vf_speed_field.py` | Component microbenchmarks of the field pathway for `field_decoder_f0cm_s1.npz` (manuscript Table 17); writes `results/_speed_confirmed_field_f0cm_<utc>.json`. |
| `vf_speed_field_deployed.py` | The deployed field pathway end to end, with the original and the fast kernels; writes `results/_speed_field_deployed_f0cm_<utc>.json`. |
| `sepeval/gen_targets_win.py` | Regenerate holdout-schema targets on the Xeon with the NARROWED convergence flag. |
| `sepeval/xfoil_bl_narrow.py` | XFoil viscous BL extractor (rewrite for Drela source build). Binary path from `XFOIL_EXE`. |

## `code/figures/`

| file | purpose |
|---|---|
| `plot_style.py` | NCFE 2026 figure style — camera-ready conference defaults. |
| `vf_surface_cst16_fig.py` | Figs. 2–3: Cp, stagnation and Cm readouts of the confirmed single model against XFOIL truth on the 2,678 converged attached rows (layout copy of the sealed generator with the converged-row mask). |
| `vf_drag_closure_fig.py` | Figures for CMAME_v2 §Profile drag. Generator + saved numbers, per the figure-provenance rule. |
| `vf_field_f2_contour.py` | Fig. 6: RANS truth against a decoder's stored prediction (`--preds`, from `code/field/step2_store_preds.py`). |

With `--print`, each generator reproduces its files in `figures/`: drawn at the manuscript's text width (5.40 in)
with 11 pt titles and labels and 10 pt ticks and legends. Without it, they draw the earlier smaller-text layout.

## `code/view_airfoil.py`

One window for one airfoil: asks for (or takes) the airfoil, incidence and Reynolds number and shows the field decoder's
four field panels next to the surface pressure, boundary-layer thicknesses and coefficients in two states, with every range
warning: the XFOIL state (the surface model) and the RANS state (the field decoder's wall values, boundary layer and
wall-function C_f). `figure(..., states=...)` draws either or both; a state outside its training range
(`meta["xfoil_in_range"]`, `meta["rans_in_range"]` from `compute`) is drawn faded and labelled. `export_csv` writes every number
(`--csv DIR`; the RANS state's boundary layer is `_boundary_layer_rans.csv`). `--save` writes the figure, `--no-show` skips
the window.
