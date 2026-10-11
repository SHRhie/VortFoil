# -*- coding: utf-8 -*-
"""Command line: ``vortfoil`` (the application window), ``vortfoil app | view | predict | sweep | about | selfcheck``;
``vortfoil_exp`` is the same with the experimental 3D wing in the application window."""
import argparse
import json
import sys

import numpy as np


def _load_cst(args):
    if args.cst:
        v = np.asarray([float(x) for x in args.cst.split(",")], np.float32)
        if v.size != 16:
            raise SystemExit(f"--cst needs 16 comma-separated values, got {v.size}")
        return dict(cst=v)
    if args.coords:
        g = np.loadtxt(args.coords)
        return dict(coords=g)
    raise SystemExit("give --cst or --coords")


def main(argv=None, prog="vortfoil"):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("app", "view"):          # the windows: their own options, passed through
        from . import app
        return (app.view if argv and argv[0] == "view" else app.app)(argv[1:])
    p = argparse.ArgumentParser(prog=prog, description="Deployed VortFoil surface model",
                                epilog="vortfoil alone opens the application window (vortfoil app --help for its options); "
                                       "vortfoil view --help: the one-window viewer; vortfoil_exp: the same, with the "
                                       "experimental 3D wing in the window.")
    p.add_argument("--mode", default="graph", choices=["graph", "eager"])
    p.add_argument("--device", default="cpu", choices=["cpu", "gpu", "auto"])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("app", help="the application window (also: vortfoil with no arguments)")
    sub.add_parser("view", help="one airfoil in one window, from the command line")
    for name in ("predict", "sweep"):
        q = sub.add_parser(name)
        q.add_argument("--cst", help="16 comma-separated CST coefficients")
        q.add_argument("--coords", help="text file of x y coordinates")
        q.add_argument("--Re", type=float, default=1e6)
        q.add_argument("--json", action="store_true", help="print the full result as JSON")
        if name == "predict":
            q.add_argument("--alpha", type=float, default=0.0)
        else:
            q.add_argument("--alpha-range", default="-4,12,17", help="start,stop,count")
    sub.add_parser("about")
    sc = sub.add_parser("selfcheck")
    sc.add_argument("--data", help="holdout_targets.npz (defaults to the packaged evaluation grid if present)")
    args = p.parse_args(argv)

    import vortfoil
    if args.cmd == "about":
        vortfoil.about()
        return 0
    if args.cmd == "selfcheck":
        from .selfcheck import run
        return run(mode=args.mode, device=args.device, data=args.data)
    vf = vortfoil.load_model(mode=args.mode, device=args.device)
    inp = _load_cst(args)
    if args.cmd == "predict":
        r = vf.predict(alpha=args.alpha, Re=args.Re, **inp)
        alphas = [args.alpha]
    else:
        a0, a1, n = args.alpha_range.split(",")
        alphas = np.linspace(float(a0), float(a1), int(n))
        cst = inp.get("cst")
        if cst is None:
            cst, _ = vortfoil.geometry.fit_cst16(inp["coords"])
        r = vf.sweep(cst, alphas, Re=args.Re)
    if args.json:
        json.dump({k: np.asarray(v).tolist() for k, v in r.as_dict().items()}, sys.stdout, indent=1)
        sys.stdout.write("\n")
    else:
        print(f"{'alpha':>8} {'Cl':>9} {'Cd':>10} {'Cm':>9} {'s_stag':>8}")
        for i, a in enumerate(np.atleast_1d(alphas)):
            print(f"{float(a):8.2f} {r.cl[i]:9.4f} {r.cd[i]:10.5f} {r.cm[i]:9.4f} {r.stagnation[i]:8.4f}")
    return 0


def main_exp(argv=None):
    """``vortfoil_exp``: ``vortfoil`` with the experimental 3D wing in the application window (``vortfoil`` refuses it)."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] == "app":
        from . import app
        return app.app(argv[1:], experimental=True)
    return main(argv, "vortfoil_exp")


if __name__ == "__main__":
    raise SystemExit(main())
