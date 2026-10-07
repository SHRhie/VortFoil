# -*- coding: utf-8 -*-
"""
Common modules for the staged operator pipeline.

After 2026-05-09 cleanup: Fourier ADAF only (LPA archived).
Op2 viscous restored with review-aligned formulation:
  (1) spectral derivative for turbulent ADAF outputs (no central_diff
      on the turbulent contribution; lam stays grid-FD because the
      laminar BL state is an ODE result and is grid-locked)
  (2) R_H* energy-integral residual form (no dH*/dH stiffness)
  (3) R_S residual unmasked at compute time; w_tr enters as a weight.
"""
from .geometry import (
    cosine_spacing,
    naca4_geometry_batch_tf,
    cst_geometry_batch_tf,
    naca4_to_cst_coeffs,
    naca4_to_cst_coeffs_tf,
    build_single_loop_contour_tf,
    build_single_loop_contour_cst_tf,
    build_wake_geometry_tf,
    CST_N_BERN_DEFAULT,
    CST_N1_DEFAULT,
    CST_N2_DEFAULT,
)
from .adaf import BatchedADAFReusable
from .panel import (
    linear_panel_integrals,
    induced_velocity_from_gamma_linear,
    induced_velocity_from_source_linear,
    induced_velocity_from_wake_source_analytical,
    compute_inviscid_surface_velocity,
    compute_surface_velocity_coupled,
)
from .networks import (
    ConditionNet,
    ResidualBlock,
    UeEncoder,
    BLOperatorNet,
)
from .branch_trunk import (
    BranchTrunkADAFHead,
    BLOperatorBT,
    CausalUeEncoderPerPanel,
    CausalGRUBranch,
    CausalTCNBranch,
    CausalBLOperator,
    encode_flow,
)
from .datasets import (
    sample_random_shapes,
    sample_random_cst_shapes,
    CST_SIGMA_DEFAULT,
)
from .interp import (
    interp1d_common_grid,
    central_diff_1d,
    central_diff_1d_nonuniform,
)
from .scaling import (
    normalize_log_re,
    LOG10_RE_MEAN,
    LOG10_RE_STD,
)
from .bl_closures import (
    BLConfig,
    compute_thwaites,
    solve_laminar_bl_2eq,
    solve_laminar_bl_full_2eq,
    Hstar_from_H_lam,
    Hstar_from_H_turb,
    Cd_lam,
    Cd_turb,
    Cf_lam_falkner_skan,
    Cf_turb_ludwieg_tillmann,
    Re_theta_crit_drela,
    dn_dRe_dampl,
)
from .branch import (
    sample_branch_ue_from_contour,
    sample_branch_ue_arc_uniform,
    map_delta_to_contour_nodes,
)
from .bl_residuals import (
    BLPhysicsConfig,
    bl_closure,
    compute_bl_residuals,
    compute_shadow_turb_residuals,
)
from .source import (
    compute_source_strength,
    compute_wake_bl_analytical,
)

__all__ = [
    # geometry
    "cosine_spacing", "naca4_geometry_batch_tf",
    "cst_geometry_batch_tf", "naca4_to_cst_coeffs", "naca4_to_cst_coeffs_tf",
    "build_single_loop_contour_tf", "build_single_loop_contour_cst_tf",
    "build_wake_geometry_tf",
    "CST_N_BERN_DEFAULT", "CST_N1_DEFAULT", "CST_N2_DEFAULT",
    # ADAF
    "BatchedADAFReusable",
    # panel kernels
    "linear_panel_integrals",
    "induced_velocity_from_gamma_linear",
    "induced_velocity_from_source_linear",
    "induced_velocity_from_wake_source_analytical",
    "compute_inviscid_surface_velocity",
    "compute_surface_velocity_coupled",
    # networks
    "ConditionNet", "ResidualBlock", "UeEncoder", "BLOperatorNet",
    # Branch-Trunk + ADAF head (NeuralFoil-style)
    "BranchTrunkADAFHead", "BLOperatorBT", "encode_flow",
    # Causal (parabolic-aware) BL operator
    "CausalUeEncoderPerPanel", "CausalGRUBranch", "CausalBLOperator",
    # data + scaling
    "sample_random_shapes", "sample_random_cst_shapes", "CST_SIGMA_DEFAULT",
    "normalize_log_re", "LOG10_RE_MEAN", "LOG10_RE_STD",
    # numerical
    "interp1d_common_grid", "central_diff_1d", "central_diff_1d_nonuniform",
    # BL closures
    "BLConfig", "compute_thwaites",
    "solve_laminar_bl_2eq", "solve_laminar_bl_full_2eq",
    "Hstar_from_H_lam", "Hstar_from_H_turb",
    "Cd_lam", "Cd_turb",
    "Cf_lam_falkner_skan", "Cf_turb_ludwieg_tillmann",
    "Re_theta_crit_drela", "dn_dRe_dampl",
    # BL residuals
    "BLPhysicsConfig", "bl_closure",
    "compute_bl_residuals", "compute_shadow_turb_residuals",
    # branch / source / wake
    "sample_branch_ue_from_contour", "sample_branch_ue_arc_uniform",
    "map_delta_to_contour_nodes",
    "compute_source_strength", "compute_wake_bl_analytical",
]
