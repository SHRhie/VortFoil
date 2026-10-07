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
- The scorers and generators that reproduce the manuscript's numbers and figures, and the records they read.
- `LICENSE` (GNU GPL v3 or later; `xfoil_closures.py` translates XFOIL, which is GPL v2 or later), `NOTICE` and
  `DATA_LICENSES.md`.
