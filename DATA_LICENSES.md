# Data licenses and attributions

The code in this repository is licensed under the GNU GPL v3 or later (`LICENSE`, `NOTICE`). The
datasets below keep their own licenses. This file lists the shipped files that come from each
dataset, and what that means for them. None of the datasets is redistributed as such.

## AirfRANS — Open Database License (ODbL) v1.0

> F. Bonnet, J. A. Mazari, P. Cinnella, P. Gallinari. AirfRANS: High Fidelity Computational Fluid
> Dynamics Dataset for Approximating Reynolds-Averaged Navier–Stokes Solutions. Thirty-sixth
> Conference on Neural Information Processing Systems, Datasets and Benchmarks Track, 2022.
> arXiv:2212.07564. Dataset, code and license: <https://github.com/Extrality/AirfRANS>.

AirfRANS is made available under the Open Database License v1.0
(<https://opendatacommons.org/licenses/odbl/1-0/>).

| file | relation to AirfRANS |
|---|---|
| `weights/airfrans_surface_tokens.npz` | An extract: the surface points (resampled, 800 × 256 × 2) and names of 800 AirfRANS cases. This file is made available under the ODbL v1.0. |
| `weights/airfrans_train_stats.npz` | Output-normalization statistics computed from the AirfRANS training fields. |
| `weights/field_decoder_*.npz` | Trained on AirfRANS fields. |
| `figures/vf_field_f0cm_tp_*` (Fig. 6 of the manuscript and its saved numbers) | Its RANS panels show AirfRANS fields. |
| `evaluation_runs/` and the field records in `results/` | Records keyed by AirfRANS case names or computed from AirfRANS fields. |

For all of these: *Contains information from [AirfRANS](https://github.com/Extrality/AirfRANS),
which is made available here under the
[Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/).*

The field training corpora built from AirfRANS (486 MB each) are not shipped.

## UniFoil — Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)

> R. S. Kanchi, B. Melanson, N. Somasekharan, S. Pan, S. He. UniFoil: A universal dataset of airfoils in
> transitional and turbulent regimes for subsonic and transonic flows. Advances in Neural Information
> Processing Systems 38 (Datasets and Benchmarks Track), 2025. doi:10.52202/085713-2467.

UniFoil is made available under CC BY-SA 4.0 (<https://creativecommons.org/licenses/by-sa/4.0/>). No
UniFoil data are shipped.

| file | relation to UniFoil |
|---|---|
| `weights/field_decoder_cmx_s{0,1,2}.npz` | Field decoders trained on AirfRANS fields together with the Spalart–Allmaras fields of 16,923 UniFoil airfoils (every third stored row). |
| `weights/field_decoder_cmd_s{0,1,2}.npz` | The same with 1,346 UniFoil airfoils. |
| `results/cotrain_score_*.json` | Records that include errors on 150 held-out UniFoil airfoils. |

Whether trained weights are adapted material under CC BY-SA 4.0 is not settled; these weight files are
nevertheless made available under CC BY-SA 4.0, with this attribution: *contains information derived from
[UniFoil](https://doi.org/10.52202/085713-2467) (Kanchi et al., 2025), CC BY-SA 4.0*. They also carry the
AirfRANS notice above.

## UIUC Airfoil Coordinates Database

> M. S. Selig. UIUC Airfoil Coordinates Database. UIUC Applied Aerodynamics Group.
> <https://m-selig.ae.illinois.edu/ads/coord_database.html>

The site states no license for the coordinates; its pages carry "© 1994–2026 UIUC Applied
Aerodynamics Group". Ask the maintainer before redistributing beyond what is here.

| file | relation to the UIUC database |
|---|---|
| `data/eval/sep_eval_coords.npz` | 16-parameter CST fits and 121-point resampled surfaces of the 80 held-out UIUC sections. |
| `data/eval/holdout_targets.npz`, `attached_surface_targets_v2.npz`, `sep_eval_targets_v2.npz` | XFOIL results on those sections. |
| `weights/op1_*.npz`, `weights/op2_*.npz`, `weights/*closure_poly2.npz` and their packaged copies in `src/vortfoil/weights/` | Trained or fitted on XFOIL results whose shapes include UIUC sections. |
