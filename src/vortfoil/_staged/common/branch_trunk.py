# -*- coding: utf-8 -*-
"""
Branch-Trunk → ADAF raw-weight head.

NeuralFoil-style separable mapping
    cond (∈ R^{d_cond}) → ADAF raw weights W (∈ R^{N_p}) + initial value (∈ R)
through the low-rank factorization

    Branch  : cond → b ∈ R^p           (and a scalar init_1)
    Trunk   : pos_emb(mode_idx) → T ∈ R^{N_p × p}
    Combine : W = b · T^T  ∈ R^{B × N_p}

Output (W, init_1) is fed into a `BatchedADAFReusable` decoder, which
performs the spectral antiderivative on a fixed coordinate grid — closed-
form, no AD, fp32-accurate.  The boundary-layer (Op2) ODE residuals need
this exact spectral derivative; `BatchedADAFReusable.evaluate_g1_g1x_at_xi`
delivers it from the same W.

STALE-ADVERTISEMENT CORRECTION [2026-09-02].  As shipped, the Op2 ODE
residuals in `staged/op2_bl_physics.py` and `staged/op2_bl.py` do NOT use
any spectral derivative -- they use `central_diff_1d` on the uniform xi
grid.  Nothing in `.neurofoil_clone/staged/` calls `evaluate_g1_g1x_at_xi`
or `g1x_from_coeffs` (live poison trace: VortFoil/code/trace_g1x_callgraph.py,
artifact VortFoil/tables/g1xfix/).  Read the paragraph above as design
intent, not as a description of the deployed call graph.  Note also that
the fixed-grid `g1x_from_coeffs` is off by 2x (defect D1) while
`evaluate_g1_g1x_at_xi` is correct -- see adaf.py.

Distinction from `staged/common/networks.ConditionNet`:
    ConditionNet is a single MLP `cond → (W, init_1)` with no per-mode
    structure — every output entry is an independent linear combination
    of cond features.
    `BranchTrunkADAFHead` factorizes that mapping as cond → b (rank p),
    plus a shared per-mode basis T learned across all conditions.
    Total parameter count drops from d_cond·(N_p+1) to (d_cond·p) +
    (N_p·p) when the inner Dense stack is set aside; the saving is
    biggest when N_p ≫ p, and the Trunk is encouraged to learn a
    smooth mode-correlated basis (the spectral / smoothness prior
    is implicit through the small p bottleneck).

Notes
-----
* `branch_dim` (= p) controls the rank of the cond → W mapping.  A
  starting point of `p ≈ branch_hidden_dim ≈ 128` is reasonable; tune
  if Op2 underfits.
* `trunk_depth = 0` reduces the trunk to a bare `pos_emb` lookup; for
  most use cases `trunk_depth = 2` is enough for the Trunk to learn
  a mode-correlated basis.
* The output Dense `branch_out` is intentionally NOT zero-initialised:
  for steady operators we want a non-zero initial γ field (else `L_np`
  is identically zero at start).  Use `last_layer_scale = 0.02` to keep
  it small but non-trivial; that's what `ConditionNet` defaulted to in
  the previous architecture.
"""
from __future__ import annotations

import tensorflow as tf

DTYPE = tf.float32


class BranchTrunkADAFHead(tf.keras.layers.Layer):
    """Branch · Trunk dot → ADAF raw weights `(W, init_1)`.

    Forward
    -------
    cond : (B, d_cond)
        Conditioning vector (e.g. concat([z_shape, z_flow])).
    returns
        W      : (B, N_p)   raw ADAF weights for one scalar field channel
        init_1 : (B, 1)     ADAF initial value g_1(ξ = 0) for the same field

    To learn multiple output channels (e.g. Op2's Cf_lam, Cf_turb, H_lam,
    H_turb, S, Cd) instantiate one `BranchTrunkADAFHead` per channel; the
    Trunk weights are NOT shared automatically.  If you want explicit
    Trunk sharing, override `_trunk_basis` to return a shared tensor.
    """

    def __init__(
        self,
        N_p: int,
        branch_dim: int = 128,
        branch_depth: int = 3,
        trunk_depth: int = 2,
        activation: str = "gelu",
        last_layer_scale: float = 0.02,
        enforce_kutta: bool = True,
        name: str | None = None,
        **kw,
    ):
        super().__init__(name=name, **kw)
        self.N_p = int(N_p)
        self.branch_dim = int(branch_dim)
        self.enforce_kutta = bool(enforce_kutta)

        # ---- Branch: cond → b (R^p) + init_1 (R^1) ----
        self.branch_hidden = [
            tf.keras.layers.Dense(
                self.branch_dim, activation=activation,
                name=f"branch_h{i}")
            for i in range(int(branch_depth))
        ]
        self.branch_out = tf.keras.layers.Dense(
            self.branch_dim + 1,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=float(last_layer_scale)),
            bias_initializer="zeros",
            name="branch_out",
        )

        # ---- Trunk: learnable per-mode basis (cond-independent) ----
        # When enforce_kutta=True, we inject Kutta constraint into the basis:
        # gamma(0) + gamma(L) = 0 becomes a hard linear constraint on W.
        # We do this by projecting the trunk basis onto the nullspace of
        # the Kutta row (last panel upper + last panel lower = 0).
        self.pos_emb = tf.keras.layers.Embedding(
            self.N_p, self.branch_dim, name="pos_emb")
        self.trunk_hidden = [
            tf.keras.layers.Dense(
                self.branch_dim, activation=activation,
                name=f"trunk_h{i}")
            for i in range(int(trunk_depth))
        ]
        
        # Kutta projection matrix: applied to W after trunk projection
        # For single-loop: W[-1] + W[Nside-1] = 0 (TE upper + TE lower = 0)
        # We precompute the projection operator as a non-trainable variable
        if self.enforce_kutta:
            N = self.N_p
            Nside = N // 2
            kutta_row = tf.zeros((1, N), dtype=tf.float32)
            kutta_row = tf.tensor_scatter_nd_update(
                kutta_row, [[0, Nside - 1], [0, N - 1]], [1.0, 1.0])
            # Projection: P = I - K^T (K K^T)^(-1) K
            # K is (1, N), so K K^T = 2, inverse = 0.5
            # P = I - 0.5 * K^T @ K
            self.kutta_projector = tf.eye(N, dtype=tf.float32) - \
                0.5 * tf.transpose(kutta_row) @ kutta_row
        else:
            self.kutta_projector = None

    # --------------------------------------------------------
    def _trunk_basis(self) -> tf.Tensor:
        """Compute per-mode basis T ∈ R^{N_p × branch_dim}.  cond-independent."""
        idx = tf.range(self.N_p, dtype=tf.int32)
        t = self.pos_emb(idx)                 # (N_p, p)
        for layer in self.trunk_hidden:
            t = layer(t)                       # (N_p, p)
        return t

    def call(self, cond, training=None):
        # Branch
        h = cond
        for layer in self.branch_hidden:
            h = layer(h, training=training)
        b_init = self.branch_out(h)            # (B, p + 1)
        b      = b_init[:, :-1]                # (B, p)
        init_1 = b_init[:, -1:]                # (B, 1)

        # Trunk
        T = self._trunk_basis()                # (N_p, p)

        # Dot:  W = b · T^T   ⇒   (B, p) × (p, N_p) = (B, N_p)
        W = tf.matmul(b, T, transpose_b=True)
        
        # Hard Kutta constraint: project W onto nullspace of Kutta row
        if self.enforce_kutta and self.kutta_projector is not None:
            W = tf.matmul(W, self.kutta_projector)
        
        return W, init_1


# ============================================================
# BL operator with shared Trunk + multiple Branch heads
# ============================================================
class BLOperatorBT(tf.keras.Model):
    """Branch-Trunk + ADAF operator for the BL closure outputs.

    Twelve closure fields are predicted on the ξ grid (six per branch):
        Cf_lam, Cf_turb, H_lam, H_turb, Cd, S         (× 2 branches)

    All twelve fields share the **same Trunk** (a learnable per-mode
    embedding plus a small MLP) — there is one Trunk basis
    T ∈ R^{N_p × branch_dim} computed once per forward pass, and the
    twelve Branch heads consume it independently.  Sharing the Trunk
    encourages a consistent mode-structure across fields and saves
    ~50% trainable parameters versus per-head Trunks.

    Conditioning
    ------------
    cond_base : (B, d_base)
        Base conditioning, typically [shape(3), sin α, cos α, log_Re_norm].
    ue_inv_u, ue_inv_l : (B, ue_grid)
        Inviscid edge velocity on upper / lower branches.  Encoded by a
        shared UeEncoder (Dense → tanh) into a `ue_feat_dim`-vector.
    cond_full = concat([cond_base, ue_feat_u, ue_feat_l])  → (B, d_full)

    Output
    ------
    dict mapping field name → (W, init_1)
        field name ∈ {"Cf_lam_u", "Cf_lam_l", "Cf_turb_u", "Cf_turb_l",
                      "H_lam_u",  "H_lam_l",  "H_turb_u",  "H_turb_l",
                      "Cd_u",     "Cd_l",     "S_u",       "S_l"}
        W      : (B, N_p)   ADAF raw weights for that field
        init_1 : (B, 1)     ADAF initial value at ξ=0

    The caller decodes (W, init_1) through a `BatchedADAFReusable` to
    obtain the closure on the ξ grid (61 points).

    Notes
    -----
    * UeEncoder is shared across upper/lower so the same physics
      meaning of u_e is applied symmetrically.
    * Cd does NOT have lam/turb split (per discussion: XFOIL polar
      reports a single viscous Cd, and dissipation closure does not
      change branch above the transition).  S is similarly a single
      field — the lag-equation does not split lam/turb.
    """

    FIELD_NAMES_BRANCH = (
        "Cf_lam",    "Cf_turb",
        "H_lam",     "H_turb",
        "theta_lam", "theta_turb",
        "Cd",        "S",
    )

    def __init__(
        self,
        N_p_bl: int = 120,
        ue_grid: int = 61,
        ue_feat_dim: int = 32,
        ue_hidden: int = 64,
        branch_dim: int = 192,
        branch_depth: int = 3,
        trunk_depth: int = 2,
        activation: str = "gelu",
        last_layer_scale: float = 0.02,
        name: str = "bl_operator_bt",
        field_names=None,
        **kw,
    ):
        super().__init__(name=name, **kw)
        self.N_p_bl = int(N_p_bl)
        self.ue_grid = int(ue_grid)
        self.ue_feat_dim = int(ue_feat_dim)
        self.branch_dim = int(branch_dim)

        # UeEncoder shared across upper/lower branches
        self.ue_encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(int(ue_hidden), activation=activation,
                                       name="ue_enc_h0"),
                tf.keras.layers.Dense(self.ue_feat_dim, activation="tanh",
                                       name="ue_enc_h1"),
            ],
            name="ue_encoder",
        )

        # Shared Trunk: pos_emb + small MLP
        self.pos_emb = tf.keras.layers.Embedding(
            self.N_p_bl, self.branch_dim, name="trunk_pos_emb")
        self.trunk_layers = [
            tf.keras.layers.Dense(self.branch_dim, activation=activation,
                                   name=f"trunk_h{i}")
            for i in range(int(trunk_depth))
        ]

        # Branch heads (one Sequential MLP per (field × branch))
        # Each maps cond_full → (b ∈ R^{branch_dim}, init_1 ∈ R^1)
        # `field_names` override (B3-β: pass ("H",) for single-channel
        # operator predicting H profile only; rest is closure + ODE).
        self._field_names = (tuple(field_names)
                              if field_names is not None
                              else self.FIELD_NAMES_BRANCH)
        self.branches = {}
        for side in ("u", "l"):
            for fld in self._field_names:
                key = f"{fld}_{side}"
                hidden = [
                    tf.keras.layers.Dense(self.branch_dim,
                                           activation=activation,
                                           name=f"br_{key}_h{i}")
                    for i in range(int(branch_depth))
                ]
                out_layer = tf.keras.layers.Dense(
                    self.branch_dim + 1,
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(
                        stddev=float(last_layer_scale)),
                    bias_initializer="zeros",
                    name=f"br_{key}_out",
                )
                self.branches[key] = tf.keras.Sequential(
                    hidden + [out_layer], name=f"branch_{key}")

    # --------------------------------------------------------
    def _trunk_basis(self) -> tf.Tensor:
        """Compute shared Trunk basis T ∈ R^{N_p × branch_dim}."""
        idx = tf.range(self.N_p_bl, dtype=tf.int32)
        t = self.pos_emb(idx)
        for layer in self.trunk_layers:
            t = layer(t)
        return t

    def call(self, cond_base, ue_inv_u, ue_inv_l, training=None):
        # 1. Ue encoding (shared u/l)
        ue_feat_u = self.ue_encoder(ue_inv_u, training=training)
        ue_feat_l = self.ue_encoder(ue_inv_l, training=training)

        # 2. Full conditioning
        cond_full = tf.concat([cond_base, ue_feat_u, ue_feat_l], axis=1)

        # 3. Trunk basis (one-shot, shared across all 12 heads)
        T = self._trunk_basis()                                    # (N_p, p)

        # 4. All 12 Branch heads → (W, init_1)
        out = {}
        for key, branch_mlp in self.branches.items():
            b_init = branch_mlp(cond_full, training=training)       # (B, p+1)
            b      = b_init[:, :-1]                                 # (B, p)
            init_1 = b_init[:, -1:]                                 # (B, 1)
            W = tf.matmul(b, T, transpose_b=True)                   # (B, N_p)
            out[key] = (W, init_1)
        return out


# ============================================================
# Causal (parabolic-aware) encoder + GRU branch for Op2 BL
# ============================================================
# Design rationale
# ----------------
# The integral BL equations are parabolic in ξ: state at ξ depends only
# on upstream history (ue, ∂ue/∂ξ, Re, L).  Both the Dense UeEncoder and
# the Branch-Trunk W generator in BLOperatorBT are NON-causal (every
# output position is a function of every input position), so the prior
# is left for the optimiser to discover.
#
# `CausalUeEncoderPerPanel` replaces the Dense encoder with stacked
# dilated causal 1D convs that emit a per-panel feature sequence; the
# feature at panel i is a function only of ue at panels ≤ i.
#
# `CausalGRUBranch` replaces the Branch-Trunk `cond → b ; W = b·T^T`
# factorisation with a recurrent generator: at panel i, the GRU
# advances its hidden state h_i = GRU(x_i, h_{i-1}) and emits W_i.
# h_i is interpretable as a learned latent BL state at panel i (the
# analogue of (θ_i, n_i) carried by the Heun integrator), but with
# learned dynamics.  The ADAF Fourier projection downstream is left
# untouched, so the W → field decode remains spectral and analytic.
# ============================================================
class CausalUeEncoderPerPanel(tf.keras.Model):
    """Causal dilated Conv1D encoder.

    Maps ue_inv on the ξ-node grid (length `ue_grid`) to a per-panel
    feature sequence of length `N_p` (the ADAF panel-midpoint grid).
    The mapping is causal: feature at panel i depends only on ue values
    at ξ ≤ ξ_panel[i].
    """

    def __init__(self,
                 N_p: int,
                 ue_feat_dim: int = 32,
                 kernel_size: int = 3,
                 dilations=(1, 2, 4, 8),
                 activation: str = "gelu",
                 name: str = "causal_ue_encoder"):
        super().__init__(name=name)
        self.N_p = int(N_p)
        self.ue_feat_dim = int(ue_feat_dim)
        self.convs = []
        for i, d in enumerate(dilations):
            # Last layer uses tanh to bound the feature magnitude (same
            # role as the tanh in the legacy Dense UeEncoder).
            act = activation if i < len(dilations) - 1 else "tanh"
            self.convs.append(tf.keras.layers.Conv1D(
                ue_feat_dim, kernel_size=kernel_size,
                dilation_rate=int(d), padding="causal", activation=act,
                name=f"causal_conv_d{int(d)}"))

    def call(self, ue_xi, training=None):
        """ue_xi: (B, ue_grid) on uniform ξ-grid → (B, N_p, ue_feat_dim).

        ue is interpolated from the ξ-node grid (length ue_grid) to the
        panel-midpoint grid (length N_p) via simple adjacent-node
        averaging when ue_grid == N_p + 1 (the canonical Op2 setting),
        otherwise via linear interpolation.
        """
        ue_grid = ue_xi.shape[1]
        if ue_grid == self.N_p + 1:
            # Canonical case: ue on (N_p+1) nodes → panel midpoints by
            # adjacent-node averaging (exact linear interp on uniform
            # grid).  Matches the gamma_panels reduction in op1.
            ue_panel = 0.5 * (ue_xi[:, :-1] + ue_xi[:, 1:])
        else:
            ue_panel = _linear_interp_to_panels_tf(ue_xi, self.N_p)

        h = ue_panel[..., None]                                    # (B, N_p, 1)
        for conv in self.convs:
            h = conv(h, training=training)                          # (B, N_p, feat)
        return h


def _linear_interp_to_panels_tf(ue_xi, N_p: int):
    """Linear interp from ue on (B, n_xi) uniform ξ∈[0,1] to (B, N_p)
    panel-midpoint values (midpoints at (k+0.5)/N_p, k=0..N_p-1)."""
    n_xi = ue_xi.shape[1]
    xi_panel = (tf.range(N_p, dtype=ue_xi.dtype) + 0.5) / tf.cast(N_p, ue_xi.dtype)
    idx_f = xi_panel * tf.cast(n_xi - 1, ue_xi.dtype)
    idx_lo = tf.clip_by_value(tf.cast(tf.floor(idx_f), tf.int32), 0, n_xi - 2)
    w_hi = idx_f - tf.cast(idx_lo, ue_xi.dtype)
    w_lo = 1.0 - w_hi
    ue_lo = tf.gather(ue_xi, idx_lo, axis=1)
    ue_hi = tf.gather(ue_xi, idx_lo + 1, axis=1)
    return w_lo[None, :] * ue_lo + w_hi[None, :] * ue_hi


class CausalGRUBranch(tf.keras.Model):
    """Recurrent multi-head W generator for ADAF.

    For each panel i = 0..N_p-1 the GRU advances its hidden state from
    (cond, ue_feat_i, h_{i-1}); each field f has its own (W_head_f,
    init1_head_f) reading off the shared h_seq.  This shares the
    parabolic-causal latent state across fields (θ, H, S) — the
    Drela-style closures treat them as a tightly coupled system, so a
    shared hidden trajectory is the natural inductive bias.

    Shared between upper/lower sides: pass per-side cond_side and
    per-side ue_feat; call once per side externally.
    """

    def __init__(self,
                 N_p: int,
                 hidden_dim: int = 192,
                 gru_depth: int = 2,
                 field_names=("theta", "H", "S"),
                 last_layer_scale: float = 0.02,
                 name: str = "causal_gru_branch"):
        super().__init__(name=name)
        self.N_p = int(N_p)
        self.hidden_dim = int(hidden_dim)
        self.field_names = tuple(field_names)
        # Stacked GRU.  Wrapped via RNN(GRUCell, ...) instead of the fused
        # tf.keras.layers.GRU layer so the generic op graph is used
        # (CudnnRNNV3 has no XLA kernel and would block jit_compile=True).
        # Cell name kept as default "gru_cell" so the variable path matches
        # what fused-GRU produced (gru_branch/gru_<i>/gru_cell/kernel),
        # which keeps prior checkpoints loadable.
        self.gru_layers = [
            tf.keras.layers.RNN(
                tf.keras.layers.GRUCell(self.hidden_dim,
                                          reset_after=True),
                return_sequences=True,
                unroll=False,
                name=f"gru_{i}")
            for i in range(int(gru_depth))
        ]
        kinit = tf.keras.initializers.TruncatedNormal(
            stddev=float(last_layer_scale))
        self.W_heads = {
            f: tf.keras.layers.Dense(
                1, activation=None,
                kernel_initializer=kinit,
                bias_initializer="zeros",
                name=f"W_head_{f}",
            ) for f in self.field_names
        }
        self.init1_heads = {
            f: tf.keras.layers.Dense(
                1, activation=None,
                kernel_initializer=kinit,
                bias_initializer="zeros",
                name=f"init1_head_{f}",
            ) for f in self.field_names
        }

    def call(self, cond_side, ue_feat_side, training=None):
        """
        cond_side    : (B, d_cond_side)
        ue_feat_side : (B, N_p, ue_feat_dim)
        returns      : dict[field] = (W, init_1)
                        W shape (B, N_p), init_1 shape (B, 1)
        """
        cond_seq = tf.tile(cond_side[:, None, :], [1, self.N_p, 1])
        x = tf.concat([cond_seq, ue_feat_side], axis=-1)

        h = x
        for gru in self.gru_layers:
            h = gru(h, training=training)                            # (B, N_p, hidden)

        out = {}
        for f in self.field_names:
            W = self.W_heads[f](h, training=training)[..., 0]        # (B, N_p)
            init_1 = self.init1_heads[f](h[:, 0, :], training=training)
            out[f] = (W, init_1)
        return out


class CausalTCNBranch(tf.keras.Model):
    """Stacked dilated causal Conv1D W generator (TCN) — parallel
    replacement for `CausalGRUBranch`.

    Same multi-head interface (`{field: (W, init_1)}`).  Replaces the
    sequential RNN inner loop with stacked causal 1D convolutions whose
    dilation grows geometrically (1, 2, 4, ..., 2^(L-1)) so the
    receptive field at the output is `1 + 2·(2^L - 1)` panels — large
    enough to cover N_p with a moderate depth.  All ops are pointwise /
    convolutional so the entire forward + backward maps cleanly to
    H100 tensor cores (no sequential dependency along the panel axis).

    Memory and FLOPs are comparable to a 2-layer GRU at hidden_dim=192:
        TCN(hidden=128, depth=7) ≈ 343 K params
        GRU(hidden=192, depth=2) ≈ 353 K params

    Shared across upper/lower; call once per side.
    """

    def __init__(self,
                 N_p: int,
                 hidden_dim: int = 128,
                 depth: int = 7,
                 kernel_size: int = 3,
                 field_names=("theta", "H", "S"),
                 last_layer_scale: float = 0.02,
                 activation: str = "gelu",
                 name: str = "causal_tcn_branch"):
        super().__init__(name=name)
        self.N_p = int(N_p)
        self.hidden_dim = int(hidden_dim)
        self.field_names = tuple(field_names)

        # Dilations grow geometrically: 1, 2, 4, ..., 2^(depth-1).
        # Receptive field at output ≥ N_p needed; depth=7 with k=3
        # gives RF = 1 + 2·(2^7 - 1) = 255 ≥ N_p=120.
        dilations = [2 ** i for i in range(int(depth))]
        self.tcn_layers = [
            tf.keras.layers.Conv1D(
                hidden_dim, kernel_size=kernel_size,
                dilation_rate=d, padding="causal",
                activation=activation,
                name=f"tcn_d{d}")
            for d in dilations
        ]

        kinit = tf.keras.initializers.TruncatedNormal(
            stddev=float(last_layer_scale))
        # Per-field heads
        self.W_heads = {
            f: tf.keras.layers.Conv1D(
                1, kernel_size=1, activation=None,
                kernel_initializer=kinit,
                bias_initializer="zeros",
                name=f"W_head_{f}",
            ) for f in self.field_names
        }
        self.init1_heads = {
            f: tf.keras.layers.Dense(
                1, activation=None,
                kernel_initializer=kinit,
                bias_initializer="zeros",
                name=f"init1_head_{f}",
            ) for f in self.field_names
        }

    def call(self, cond_side, ue_feat_side, training=None):
        """
        cond_side    : (B, d_cond_side)
        ue_feat_side : (B, N_p, ue_feat_dim)
        returns      : dict[field] = (W, init_1)  shapes (B, N_p), (B, 1)
        """
        # Broadcast cond_side to per-panel sequence and concat with ue_feat.
        cond_seq = tf.tile(cond_side[:, None, :], [1, self.N_p, 1])
        h = tf.concat([cond_seq, ue_feat_side], axis=-1)
        # Stacked causal Conv1D — fully parallel along the panel axis.
        for layer in self.tcn_layers:
            h = layer(h, training=training)                          # (B, N_p, hidden)

        out = {}
        for f in self.field_names:
            W = self.W_heads[f](h, training=training)[..., 0]        # (B, N_p)
            init_1 = self.init1_heads[f](h[:, 0, :], training=training)
            out[f] = (W, init_1)
        return out


class CausalBLOperator(tf.keras.Model):
    """Op2 BL NN: causal Ue encoder + per-panel W generator (TCN or GRU).

    Outputs ADAF (W, init_1) pairs for three fields per surface:
        theta_u, theta_l, H_u, H_l, S_u, S_l
    matching the legacy `BLOperatorNet` interface used by the
    Residual-Ansatz physics-only Op2.  The encoder and branch are
    SHARED across upper/lower sides — per-side asymmetry enters through
    (cond_side, ue_inv) only.

    `branch_type`: "tcn" (default, parallel, H100-friendly) or "gru"
    (sequential, kept for backward compatibility).
    """

    FIELD_NAMES = ("theta", "H", "S")

    def __init__(self,
                 N_p_bl: int = 120,
                 ue_grid: int = 121,
                 ue_feat_dim: int = 32,
                 ue_kernel_size: int = 3,
                 ue_dilations=(1, 2, 4, 8),
                 gru_hidden: int = 192,
                 gru_depth: int = 2,
                 tcn_hidden: int = 128,
                 tcn_depth: int = 7,
                 tcn_kernel_size: int = 3,
                 branch_type: str = "tcn",
                 field_names=None,
                 name: str = "causal_bl_operator"):
        super().__init__(name=name)
        self.N_p_bl = int(N_p_bl)
        self.ue_grid = int(ue_grid)
        self.field_names = (tuple(field_names) if field_names is not None
                              else self.FIELD_NAMES)
        self.ue_encoder = CausalUeEncoderPerPanel(
            N_p=N_p_bl, ue_feat_dim=ue_feat_dim,
            kernel_size=ue_kernel_size, dilations=ue_dilations,
            name="ue_encoder",
        )
        if branch_type == "tcn":
            self.gru_branch = CausalTCNBranch(
                N_p=N_p_bl, hidden_dim=tcn_hidden, depth=tcn_depth,
                kernel_size=tcn_kernel_size,
                field_names=self.field_names,
                name="tcn_branch",
            )
        else:
            self.gru_branch = CausalGRUBranch(
                N_p=N_p_bl, hidden_dim=gru_hidden, gru_depth=gru_depth,
                field_names=self.field_names,
                name="gru_branch",
            )

    def call(self, cond_side_u, cond_side_l, ue_inv_u, ue_inv_l,
              training=None):
        ue_feat_u = self.ue_encoder(ue_inv_u, training=training)
        ue_feat_l = self.ue_encoder(ue_inv_l, training=training)
        out_u = self.gru_branch(cond_side_u, ue_feat_u, training=training)
        out_l = self.gru_branch(cond_side_l, ue_feat_l, training=training)
        flat = {}
        for f in self.field_names:
            flat[f"{f}_u"] = out_u[f]
            flat[f"{f}_l"] = out_l[f]
        return flat


# ============================================================
# Optional helper: flow-vector encoder
# ============================================================
def encode_flow(alpha_deg: tf.Tensor, Re: tf.Tensor,
                 normalize_log_re_fn=None) -> tf.Tensor:
    """Build the z_flow ∈ R^3 = (sin α, cos α, normalize_log_re(Re)).

    Parameters
    ----------
    alpha_deg : (B, 1)
    Re        : (B, 1)
    normalize_log_re_fn : callable (Re → normalized log-Re), optional.
        If None, falls back to `(log10(Re) - 5.5) / 1.0` (centered at Re=3e5).

    Returns
    -------
    z_flow : (B, 3)
    """
    pi_over_180 = tf.constant(3.141592653589793 / 180.0, dtype=alpha_deg.dtype)
    alpha_rad = alpha_deg * pi_over_180
    sin_a = tf.sin(alpha_rad)
    cos_a = tf.cos(alpha_rad)
    if normalize_log_re_fn is not None:
        log_re = normalize_log_re_fn(Re)
    else:
        log_re = (tf.math.log(tf.maximum(Re, 1.0)) / tf.math.log(10.0) - 5.5)
    return tf.concat([sin_a, cos_a, log_re], axis=1)
