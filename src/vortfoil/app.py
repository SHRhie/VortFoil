# -*- coding: utf-8 -*-
"""The desktop application (``vortfoil`` or ``vortfoil app``) and the one-window viewer (``vortfoil view``) of an
installed package.

The wheel carries copies of the repository modules they run on - code/vortfoil_app.py, code/view_airfoil.py,
code/vortfoil_paths.py, code/field and code/model/staged - and the four weight files of the field route, under
vortfoil/_release/ in the repository's own layout, so code/vortfoil_paths.py resolves them unchanged. Run from a source
checkout, the checkout's code/ is used instead.
"""
import os
import sys


def _code_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.path.join(here, "_release", "code"), os.path.join(os.path.dirname(os.path.dirname(here)), "code")):
        if os.path.isfile(os.path.join(d, "vortfoil_app.py")):
            return d
    raise SystemExit("the application files are missing from this installation (vortfoil/_release/code)")


def _module(name):
    d = _code_dir()
    if d not in sys.path:
        sys.path.insert(0, d)
    return __import__(name)


def app(argv=None):
    """The application window; argv as for code/vortfoil_app.py (--selftest, --shortcut)."""
    return _module("vortfoil_app").main(argv)


def view(argv=None):
    """The one-window viewer; argv as for code/view_airfoil.py (--airfoil, --alpha, --Re, --save, --no-show, ...)."""
    return _module("view_airfoil").main(argv)
