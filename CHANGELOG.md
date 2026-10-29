# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [2.1.0] - unreleased

Accompanies the revised manuscript.

### Changed

- The deployed field decoder is `field_decoder_cmx_s0.npz`: the same architecture and conditioning, trained on the 780
  AirfRANS k-omega SST cases together with 16,923 non-NACA UniFoil airfoils (Spalart-Allmaras), with no turbulence-model
  input; run 0 of three, selected on the 20 AirfRANS validation cases. `predict_field`, `vortfoil view` and the 2D
  application use it. Its non-NACA family warning now gives its own errors on 16 unseen non-NACA SST sections (14 to 46
  times its NACA test level, against 27 to 68 for the AirfRANS-only decoder).
- The 3D wing's RANS state (`vortfoil_exp`) uses the deployed decoder, as the manuscript's wing does. Its section tables span
  the decoder's AirfRANS incidences (-5 to 15 deg), C_l continues linearly outside a table, and the lifting-line solve takes a
  root with every strip inside the table, trying the clamped form first (as 2.0.0 solved). On the research's five 45 m/s
  wings (NACA 0012, 0018, 2412, 4412, 23012) C_L agrees with the research within 0.0009 (RANS state) and 0.0015 (XFOIL
  state); a NACA 23012 tip strip no longer settles at a spurious incidence, and the default wing keeps C_L 0.6261.
- `vortfoil sweep --coords` measures the incidence from the fitted chord line, as `predict` does (a section rotated by
  2 deg gave C_l off by 0.22).
- The validation scripts of the field pathway pin the decoder their shipped records were written with.

### Added

- `vortfoil.rans`, the RANS coefficient correction (`weights/rans_correction_nn.npz`, packaged): five 5-32-32-3
  networks on the surface branch's (C_l, ln C_d, C_m, alpha, log10 Re), fitted on the 780 AirfRANS training cases.
  `predict(..., rans=True)` returns `cl_rans`, `cd_rans`, `cm_rans`; `vortfoil predict|sweep --rans` prints them; a
  RangeWarning is issued outside alpha -5 to 15 deg or Re 2e6 to 6e6.
- The decoders of the manuscript's ablations: `field_decoder_cmx_s{1,2}`, the step-matched AirfRANS-only
  `field_decoder_sml2_s*` and the 1,346-airfoil `field_decoder_cmd_s*`, with the records behind the new numbers
  (`results/cotrain_score_*`, `stepmatch_score_*`, `rans_correction_nn_*`, `table4_row_cmx_s0.json`,
  `consistency_field_vs_corrected_20261029.json`).
- `DATA_LICENSES.md`: UniFoil (CC BY-SA 4.0) and the weights trained with it.

### Fixed

- The README described the field metric's standard deviation as the full training split's; it is computed from a
  50-case sample of the scarce-task training split, which makes comparisons with published values conservative.

## [2.0.0] - 2026-10-13

First published version, accompanying the manuscript. The 2.0.0 version number matches the manuscript's model
generation; it does not follow a prior 1.x package.

### Added

- The installable `vortfoil` package: the prediction API (eager, compiled and GPU execution), the shape-geometry
  fitting module with a coordinate input route, the command-line interface, the self-check and the build manifest.
- The surface model: the effective vortex sheet with its panel readouts, the boundary-layer operator, and the frozen
  drag and lift closures (the reported lift is the closure-corrected lift; the circulation lift stays available as
  `cl_gamma`).
- The two field decoders the manuscript reports, all three training runs of each: the operator-conditioned decoder
  (`field_decoder_f0cm_s*.npz`, run 1 deployed) and the XFOIL-conditioned decoder it is compared with
  (`field_decoder_f0a_s*.npz`, run 2 deployed), with the evaluations, training logs and validation scores behind
  the field results.
- The scorers and generators that reproduce the manuscript's numbers and figures (`--print` draws the figures at
  the manuscript's text width, as shipped in `figures/`), and the records they read.
- `vortfoil.RangeWarning`: `predict()`, `sweep()` and `gradients()` warn when an incidence or Reynolds number lies
  outside the training range (-10 to 18 deg, 5e4 to 1e7), or when coordinates fit the 16-parameter form poorly or have
  a blunt trailing edge; the rows are flagged in `result.info["outside_training_range"]`. A single shape is repeated
  over arrays of incidence and Reynolds number, so `predict(coords=xy, alpha=[...])` returns a polar.
- `code/field/predict_field.py`: the operator-conditioned field decoder for any section, incidence and Reynolds
  number from coordinates alone, with warnings outside its AirfRANS training domain, and its validation
  (`code/field/validate_predict_field.py`, `results/predict_field_validation.json`).
- `code/view_airfoil.py`: one window per airfoil with the field panels, the surface pressure, the boundary-layer
  thicknesses and the coefficients; resizing the window zooms the figure instead of re-flowing its layout, and an
  upright screen gets the panels stacked. `code/vortfoil_app.py`: the same as a desktop application (NACA sections,
  the UIUC database downloaded on demand, or your own coordinate file; figures saved at 400 dpi; `--shortcut` adds a
  launcher for the environment it runs in: a desktop shortcut on Windows, a menu entry on Linux, the terminal command
  on macOS), with installation steps in the README. Three layouts (side by side, fields over a row of surface
  panels, stacked); the application uses the one that shows the figure largest.
- `vortfoil` with no arguments opens the application window; `vortfoil app` and `vortfoil view` take the options of
  the two scripts. The wheel carries the application, the field route and its four weight files under
  `vortfoil/_release/` (the repository's layout), and matplotlib is a dependency, so `uv tool install vortfoil` gives
  the whole tool on Windows, macOS and Linux; CI installs the built wheel that way on all three.
- Export data (CSV) in the application and `vortfoil view --csv DIR`: the coefficients, the surface C_p, both boundary
  layers, the field at every grid point and the section as CSV files, with a text file that explains the columns.
  `SurfacePrediction` gains `x_cp`, `theta` and `dstar` (with `vortfoil.BL_X`).
  The NACA generator of the AirfRANS sections is vendored (`code/field/naca_generator.py`, MIT).
- The viewer and the application show the surface results in two states, the XFOIL state (the surface model) and the RANS
  state (the field decoder's wall C_p, lift, drag, moment, wall-function C_f and boundary layer), and two boxes choose which
  (both by default). Each state has its own training range (XFOIL state -10 to 18 deg, Re 5e4 to 1e7; RANS state -5 to 15 deg,
  Re 2e6 to 6e6): a state outside its range is computed and drawn faded and labelled, as are the field panels when it is the
  RANS state. `predict_field.boundary_layer()` reads the RANS state's boundary layer from the decoder's flow (validated on the
  200 AirfRANS test cases, `code/field/validate_field_boundary_layer.py`, `results/field_boundary_layer_validation.json`). The CSV
  export adds `_boundary_layer_rans.csv`, and `_coefficients.csv` gains the RANS state's C_m and 1/0 flags for each state's range.
  The field panels use one colour map per channel: u_par coolwarm centred on the free stream, u_perp coolwarm, C_p
  RdYlBu_r, nu_t YlGnBu on a log scale.
- `vortfoil_exp` (from a checkout, `python code/vortfoil_app.py --experimental`): the application window with an
  experimental quasi-3D wing, a classical nonlinear lifting line that carries VortFoil sections in the XFOIL or the RANS state
  (`code/wing_nll.py`, `code/view_wing.py`, `code/wing_flow.py`). It is not part of the paper's validated claims; `vortfoil`
  keeps the paper's 2D models only (it refuses `--experimental`), and `vortfoil_exp`'s other subcommands are `vortfoil`'s. `--shortcut --experimental` adds a separate
  launcher, VortFoil (experimental). `wing_nll.input_ranges()` gives the speeds that keep every strip inside each state's
  training Reynolds range; the window shows them in a gray line below the Section model row (with the strip incidence limits, what the speed
  typed fits, and a note that the RANS state is trained on NACA 4/5-digit sections only), and the Geometry window's **Selected
  row(s)** and **All rows** put the Airfoil box's section (a NACA designation, a UIUC section or your file) on the wing's rows.
- `LICENSE` (GNU GPL v3 or later; `xfoil_closures.py` translates XFOIL, which is GPL v2 or later), `NOTICE` and
  `DATA_LICENSES.md`.

### Fixed

- `geometry.fit_cst16` and the field route's frame took the trailing edge of a closed coordinate loop as the midpoint
  of its first node and its last lower node, so a closed `.dat` file with a long last panel was rotated and rescaled
  (the 51-point UIUC NACA 64(2)-415: 0.085 deg, 2.6% chord) and reported a trailing-edge gap. The first node is now
  the trailing edge of a closed loop.
- The field route's peak memory: the fast kernels and the decoder now work in tasks of 2,048 rows, so a default
  80,601-point field needs about 0.9 GB above the loaded models instead of 2.9 GB (outputs bitwise unchanged).
- On macOS the field route printed spurious `RuntimeWarning`s from Accelerate's matrix products in worker threads.
- The field route warned only outside the decoder's incidence and Reynolds ranges. Ten UIUC sections computed with the
  AirfRANS protocol showed errors 1.5 to 80 times the NACA test level (26 times on average for u_x), and the
  shape-code check flagged none of them. Every section not given as a NACA designation now carries a family warning,
  and the README and model card state the measurement.
