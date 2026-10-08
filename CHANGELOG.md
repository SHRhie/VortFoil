# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [2.0.0] - unreleased

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
  `SurfacePrediction` gains `x_cp`, `theta` and `dstar` (with `vortfoil.BL_X`).
  The NACA generator of the AirfRANS sections is vendored (`code/field/naca_generator.py`, MIT).
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
