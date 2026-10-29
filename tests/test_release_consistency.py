# -*- coding: utf-8 -*-
"""The release carries two copies of the operators and of the weights: one inside the installable
package and one in the reproduction tree that the paper's scripts run. Two copies can drift, and a
drift would make the package and the published numbers disagree silently. These tests make that
drift an error instead.

The two code copies are not byte-identical by design: the packaged modules import each other
relatively, the reproduction tree by absolute name. Everything else must match.
"""
import hashlib
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "src", "vortfoil", "_staged")
REPRO = os.path.join(ROOT, "code", "model", "staged")
WEIGHTS_PKG = os.path.join(ROOT, "src", "vortfoil", "weights")
WEIGHTS_REPO = os.path.join(ROOT, "weights")

_IMPORT = re.compile(r"^\s*from\s+(\.|staged\.|\.\.)", re.M)


def _body(path):
    """File contents with the internal import lines removed, so the two import styles compare equal."""
    lines = [ln for ln in open(path).read().splitlines() if not _IMPORT.match(ln)]
    return "\n".join(lines)


def _shared_modules():
    out = []
    for sub in ("", "common"):
        d = os.path.join(PKG, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                rel = os.path.join(sub, f)
                if os.path.exists(os.path.join(REPRO, rel)):
                    out.append(rel)
    return out


@pytest.mark.skipif(not os.path.isdir(REPRO), reason="reproduction tree not present (wheel install)")
@pytest.mark.parametrize("rel", _shared_modules())
def test_packaged_and_reproduction_operators_agree(rel):
    a, b = _body(os.path.join(PKG, rel)), _body(os.path.join(REPRO, rel))
    assert a == b, f"{rel} differs between the package and the reproduction tree"


@pytest.mark.skipif(not os.path.isdir(WEIGHTS_REPO), reason="repository weights not present (wheel install)")
@pytest.mark.parametrize("name", ["op1_supervised_eps0.50.scr_bd128.npz", "op2_m325_BT.npz",
                                  "lift_closure_poly2.npz", "cd_closure_poly2.npz", "rans_correction_nn.npz"])
def test_packaged_and_repository_weights_agree(name):
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    assert sha(os.path.join(WEIGHTS_PKG, name)) == sha(os.path.join(WEIGHTS_REPO, name)), name


@pytest.mark.skipif(not os.path.isdir(REPRO), reason="reproduction tree not present")
def test_no_module_is_in_the_package_but_missing_from_the_reproduction_tree():
    """The package must not carry an operator module the published scripts never see."""
    extra = []
    for sub in ("", "common"):
        d = os.path.join(PKG, sub)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if f.endswith(".py") and not os.path.exists(os.path.join(REPRO, sub, f)):
                extra.append(os.path.join(sub, f))
    assert not extra, f"packaged modules with no counterpart in the reproduction tree: {extra}"
