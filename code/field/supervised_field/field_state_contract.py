# -*- coding: utf-8 -*-
"""supervised_field/field_state_contract.py — the injected state-provider contract for
field inference.

WHY THIS FILE EXISTS
--------------------
`nf_hybrid_field` used to source two model inputs from XFOIL:

  * `cl_used`  -- the paint strength, from `airfrans_p1_labels.run_one` (XFOIL XTR),
                  with an `ens_cl` direct-coefficient-head fallback;
  * `s`        -- the separation scalar, from `nf_eval200_v5.sep_scalar_xfoil` (XFOIL Cf).

Both are removed from the authoritative route.  The authoritative architecture requires

  * `cl` derived from the ADAF-generated `gamma_eff` circulation functional, and
  * `s`  derived from the learned viscous surface/wake state,

with **no** XFOIL call, **no** direct coefficient head, **no** truth label, **no**
alpha-only heuristic and **no** silent fallback.

STATIC RECONCILIATION RESULT (2026-09-05)
-----------------------------------------
No module under `supervised_field/` provides either quantity in a compliant form:

  * the `Cl = 2 * sum(gamma * ds)` circulation functional lives under `VortFoil/code/`
    (`gamma_reach.py`, `gammaens_audit.py`); `gamma_reach.py` hard-stops on a missing
    `VortFoil_v2` input, and none of it is wired as a field-inference provider;
  * `field_features.Prof4StateProvider` does emit a learned (theta, delta*, ue) state, but
    it is the **P1 reconstructed-profile route, which is NOT SELECTED**, and it defines no
    `s` mapping.  Reviving it as the main input is explicitly out of contract.

So this module implements the *required* contract and **fails closed**.  It deliberately
does NOT ship a provider.  Until a compliant provider exists upstream, end-to-end field
inference is BLOCKED -- which is the honest state, not a regression introduced here.

WHAT THIS FILE DOES NOT DO
--------------------------
It does not compute `cl` or `s`.  It does not substitute a default.  It does not fall back
to the legacy XFOIL route.  It validates a provider's declared provenance mechanically and
refuses everything that the architecture forbids.

LIMIT OF THE PROVENANCE CHECK -- READ THIS BEFORE QUOTING IT
------------------------------------------------------------
`validate_provider` / `resolve_case_state` verify the contract a provider **declares about
itself**.  They are NOT an independent proof of the upstream computation graph.  A provider
that declares `cl_source = "gamma_eff_circulation_functional"` while actually computing
something else would pass this check.  What the check buys is: the legacy XFOIL and
direct-head routes cannot be re-attached silently, a missing provider cannot be papered
over, and every accepted case carries a machine-readable claim that can be audited later.
Proving that a real provider's `cl` truly is a `gamma_eff` circulation readout requires
inspecting that provider's own graph, which does not exist in this repository yet.

`external_solver_calls == 0` is likewise a *declared* count.  The independent evidence that
the authoritative route makes no external-process call is the subprocess/os.system poison
counter in `test_no_xfoil_contract`, not this field.
"""
import contextlib
import math
import os

__all__ = [
    "FieldStateContractError", "ProviderMissing", "ForbiddenProvenance",
    "MalformedProviderOutput",
    "CL_SOURCE_REQUIRED", "S_SOURCE_REQUIRED", "FORBIDDEN_SOURCE_TOKENS",
    "SOLVER_ENV_VARS", "no_solver_env",
    "validate_provider", "resolve_case_state", "diagnostics_block",
]

#: Environment variables that name an external solver executable.  The authoritative
#: import graph must not create or change any of them.
SOLVER_ENV_VARS = ("XFOIL_EXE",)


@contextlib.contextmanager
def no_solver_env(names=SOLVER_ENV_VARS):
    """Import-time guard: restore each solver env var to its EXACT prior state on exit.

    `nf_eval200` (line 29) and `nf_eval200_v5` (line 35) both run
    `os.environ.setdefault("XFOIL_EXE", ...)` at MODULE level.  Those two files are
    historical and must not be edited, but the authoritative route still needs
    solver-free constants and the pure-numpy forward from them.  Importing them under
    this guard means the authoritative import graph leaves `XFOIL_EXE` exactly as it
    found it: absent stays absent, and a pre-existing operator value is not overwritten.

    This isolates the *environment side effect* only.  It does not, and is not meant to,
    make those modules solver-free.
    """
    saved = {n: os.environ.get(n, KeyError) for n in names}
    try:
        yield
    finally:
        for n, prior in saved.items():
            if prior is KeyError:
                os.environ.pop(n, None)
            else:
                os.environ[n] = prior


# ---------------------------------------------------------------- error taxonomy
class FieldStateContractError(RuntimeError):
    """Base: the injected-state contract was not satisfied.  Never caught internally."""


class ProviderMissing(FieldStateContractError):
    """No provider was injected.  This is the gate that replaces the XFOIL route."""


class ForbiddenProvenance(FieldStateContractError):
    """The provider declared a source the architecture forbids."""


class MalformedProviderOutput(FieldStateContractError):
    """The provider returned something that is not a usable state block."""


# ---------------------------------------------------------------- the contract
#: `cl` must be a readout of the ADAF-generated effective vortex distribution.
CL_SOURCE_REQUIRED = "gamma_eff_circulation_functional"

#: `s` must be derived from the learned viscous surface/wake state.
S_SOURCE_REQUIRED = "learned_viscous_state"

#: Substrings that disqualify a declared source outright.  Matched case-insensitively
#: against BOTH source fields.  Each entry names a route the architecture rejects.
FORBIDDEN_SOURCE_TOKENS = (
    "xfoil",            # external solver at inference
    "xtr",              # XFOIL transition label
    "truth",            # truth-derived state
    "label",            # dataset label as a model input
    "direct_head",      # direct coefficient head
    "ens_head",         # the removed ensemble-head cl fallback
    "ens_cl",
    "coef_head",
    "panel_solve",      # conventional panel-strength solve
    "bie",              # hard boundary-integral solve
    "thwaites",         # classical BL pre-pass
    "drela",
    "cdcalc",
    "alpha_only",       # alpha-only heuristic
    "heuristic",
    "prof4",            # P1 reconstructed-profile feature: NOT SELECTED
    "reconstructed_profile",
    "fallback",         # any fallback route, silent or not
    "legacy",
)

_REQUIRED_PROVENANCE_KEYS = ("cl_source", "s_source", "external_solver_calls")
_REQUIRED_STATE_KEYS = ("cl", "s", "provenance")


def _check_tokens(field_name, value):
    low = str(value).lower()
    for tok in FORBIDDEN_SOURCE_TOKENS:
        if tok in low:
            raise ForbiddenProvenance(
                f"provenance.{field_name} = {value!r} contains the forbidden token "
                f"{tok!r}.  The authoritative field-inference route rejects XFOIL, truth "
                f"labels, direct coefficient heads, conventional panel/BIE solves, "
                f"classical BL pre-passes, alpha-only heuristics, the NOT-SELECTED P1 "
                f"profile feature, and every fallback route.")


def _validate_provenance(prov):
    if not isinstance(prov, dict):
        raise MalformedProviderOutput(
            f"provenance must be a dict, got {type(prov).__name__}")
    missing = [k for k in _REQUIRED_PROVENANCE_KEYS if k not in prov]
    if missing:
        raise MalformedProviderOutput(
            f"provenance is missing required key(s) {missing}; required: "
            f"{list(_REQUIRED_PROVENANCE_KEYS)}")

    # 1. external solver calls must be exactly zero, and must be an integer zero --
    #    not None, not False-y-by-accident, not a float that happens to compare equal.
    esc = prov["external_solver_calls"]
    if not isinstance(esc, int) or isinstance(esc, bool) or esc != 0:
        raise ForbiddenProvenance(
            f"provenance.external_solver_calls must be int 0, got {esc!r} "
            f"({type(esc).__name__}).  A non-zero or non-integer count means an external "
            f"solver was reached (or the provider did not account for it).")

    # 2. forbidden tokens in either declared source.
    _check_tokens("cl_source", prov["cl_source"])
    _check_tokens("s_source", prov["s_source"])

    # 3. positive requirement: the declared sources must be the required derivations.
    if str(prov["cl_source"]) != CL_SOURCE_REQUIRED:
        raise ForbiddenProvenance(
            f"provenance.cl_source must be {CL_SOURCE_REQUIRED!r} (cl is a readout of the "
            f"ADAF-generated gamma_eff circulation), got {prov['cl_source']!r}")
    if str(prov["s_source"]) != S_SOURCE_REQUIRED:
        raise ForbiddenProvenance(
            f"provenance.s_source must be {S_SOURCE_REQUIRED!r} (s is derived from the "
            f"learned viscous surface/wake state), got {prov['s_source']!r}")
    return prov


def validate_provider(provider):
    """Structural check on the injected provider itself, before any case is touched."""
    if provider is None:
        raise ProviderMissing(
            "no state provider was injected.  nf_hybrid_field no longer sources cl from "
            "XFOIL XTR or s from XFOIL Cf, and there is deliberately NO default: a default "
            "would be the silent fallback this change removes.\n"
            "MIGRATION: pass state_provider=<obj> with a method\n"
            "    state_for(prm, alpha_deg, Re) -> {'cl': float, 's': float, 'provenance': "
            "{'cl_source': 'gamma_eff_circulation_functional', 's_source': "
            "'learned_viscous_state', 'external_solver_calls': 0}}\n"
            "No compliant provider exists in this repository yet (see the module docstring), "
            "so end-to-end field inference is BLOCKED on that upstream component.")
    if not callable(getattr(provider, "state_for", None)):
        raise ProviderMissing(
            f"injected provider {type(provider).__name__} has no callable .state_for("
            f"prm, alpha_deg, Re); it does not implement the state-provider protocol")
    return provider


def resolve_case_state(provider, prm, alpha_deg, Re):
    """Ask the provider for this case's (cl, s) and validate it. Raises on any violation."""
    validate_provider(provider)
    try:
        st = provider.state_for(prm, alpha_deg, Re)
    except FieldStateContractError:
        raise
    except Exception as exc:                      # provider blew up -> fail closed, loudly
        raise MalformedProviderOutput(
            f"state provider raised {type(exc).__name__}: {exc}.  Failing closed rather "
            f"than substituting a value.") from exc

    if not isinstance(st, dict):
        raise MalformedProviderOutput(
            f"provider.state_for must return a dict, got {type(st).__name__}")
    missing = [k for k in _REQUIRED_STATE_KEYS if k not in st]
    if missing:
        raise MalformedProviderOutput(
            f"provider.state_for output is missing {missing}; required: "
            f"{list(_REQUIRED_STATE_KEYS)}")

    prov = _validate_provenance(st["provenance"])

    out = {}
    for k in ("cl", "s"):
        try:
            v = float(st[k])
        except (TypeError, ValueError) as exc:
            raise MalformedProviderOutput(
                f"provider {k!r} is not a real scalar: {st[k]!r}") from exc
        if not math.isfinite(v):
            raise MalformedProviderOutput(f"provider {k!r} is non-finite: {v!r}")
        out[k] = v

    # `s` keeps the range contract of the input it replaces.  The legacy XFOIL-Cf
    # separation scalar was a clipped unit-interval quantity (nf_eval200_v5's
    # sep_scalar_xfoil clipped to [0, 1], and the documented exception path used
    # clip((|alpha| - 6) / 8, 0, 1)).  The field weights were trained on that range, so a
    # provider that returns anything outside it is silently changing the input contract.
    if not (0.0 <= out["s"] <= 1.0):
        raise MalformedProviderOutput(
            f"provider s = {out['s']!r} is outside the preserved input contract "
            f"0 <= s <= 1.  The legacy separation scalar this replaces was a clipped "
            f"unit-interval quantity and the loaded field weights were trained on that "
            f"range; a provider must not widen it silently.")

    out["provenance"] = prov
    return out


def diagnostics_block(state):
    """The per-case provenance block surfaced in hybrid_field()'s output."""
    prov = state["provenance"]
    return dict(cl_source=prov["cl_source"],
                s_source=prov["s_source"],
                external_solver_calls=int(prov["external_solver_calls"]),
                xfoil_calls=0,
                cl_from_gamma_eff_circulation=bool(
                    prov["cl_source"] == CL_SOURCE_REQUIRED),
                s_from_learned_viscous_state=bool(
                    prov["s_source"] == S_SOURCE_REQUIRED),
                provider_injected=True,
                s_range_contract="0 <= s <= 1 (preserved from the legacy input)",
                provenance_is_provider_declared=True,
                provenance_limit=("these fields are the provider's OWN declaration, "
                                  "validated for contract compliance; they are NOT an "
                                  "independent proof of the upstream computation graph"))


def demo():
    """assert-based self-check.  No dataset, no weights, no solver."""
    good = {"cl_source": CL_SOURCE_REQUIRED, "s_source": S_SOURCE_REQUIRED,
            "external_solver_calls": 0}

    class P:
        def state_for(self, prm, alpha_deg, Re):
            return {"cl": 0.5, "s": 0.25, "provenance": dict(good)}

    st = resolve_case_state(P(), (0.0,) * 4, 5.0, 3e6)
    assert st["cl"] == 0.5 and st["s"] == 0.25
    d = diagnostics_block(st)
    assert d["external_solver_calls"] == 0 and d["xfoil_calls"] == 0
    assert d["cl_from_gamma_eff_circulation"] and d["s_from_learned_viscous_state"]

    def rejects(exc_type, mutate):
        pv = dict(good); st_ = {"cl": 0.5, "s": 0.25, "provenance": pv}
        mutate(pv, st_)

        class Q:
            def state_for(self, *a):
                return st_
        try:
            resolve_case_state(Q(), (0.0,) * 4, 5.0, 3e6)
        except exc_type:
            return True
        raise AssertionError(f"expected {exc_type.__name__}, none raised")

    assert rejects(ForbiddenProvenance, lambda p, s: p.update(cl_source="xfoil_xtr"))
    assert rejects(ForbiddenProvenance, lambda p, s: p.update(cl_source="ens_head_fallback"))
    assert rejects(ForbiddenProvenance, lambda p, s: p.update(s_source="xfoil_cf"))
    assert rejects(ForbiddenProvenance, lambda p, s: p.update(s_source="truth_state"))
    assert rejects(ForbiddenProvenance, lambda p, s: p.update(s_source="alpha_only_ramp"))
    assert rejects(ForbiddenProvenance, lambda p, s: p.update(cl_source="prof4_reconstructed"))
    assert rejects(ForbiddenProvenance, lambda p, s: p.update(external_solver_calls=1))
    assert rejects(ForbiddenProvenance, lambda p, s: p.update(external_solver_calls=0.0))
    assert rejects(MalformedProviderOutput, lambda p, s: p.pop("s_source"))
    assert rejects(MalformedProviderOutput, lambda p, s: s.update(cl=float("nan")))
    assert rejects(MalformedProviderOutput, lambda p, s: s.pop("s"))
    assert rejects(MalformedProviderOutput, lambda p, s: s.update(s=-0.01))
    assert rejects(MalformedProviderOutput, lambda p, s: s.update(s=1.01))
    # the closed interval itself must be ACCEPTED, or the range gate is too tight
    for edge in (0.0, 1.0):
        pv = dict(good)

        class E:
            def state_for(self, *a):
                return {"cl": 0.5, "s": edge, "provenance": pv}
        assert resolve_case_state(E(), (0.0,) * 4, 5.0, 3e6)["s"] == edge

    for bad, exc in ((None, ProviderMissing), (object(), ProviderMissing)):
        try:
            validate_provider(bad)
        except exc:
            pass
        else:
            raise AssertionError(f"validate_provider({bad!r}) did not fail closed")
    print("field_state_contract.demo: OK")


if __name__ == "__main__":
    demo()
