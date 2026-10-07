# -*- coding: utf-8 -*-
"""Validation-holdout score of field-decoder weight files, computed as the trainer's holdout() does (sealed trainer 0d8ed19d,
code/field/supervised_field/nf_train_shape.py): per-channel standardized MSE pooled over all rows of the 20-case validation
holdout named in the file's run_config, and the mean of the four channels (the deployed-weight selection score).
Forward pass in numpy (supervised_field/nf_eval_shape.fwd_np / encode_np) instead of TF.

Source of both deployed decoders' validation scores: results/f0a_holdout_rescore_20260930.jsonl (XFOIL-conditioned, run 2
0.00372) and results/f0cm_holdout_rescore.json (operator-conditioned, run 1 0.00262). The corpus is the one named in
run_config, looked up by basename in VORTFOIL_CORPUS_DIR (the AirfRANS-derived training corpora are not shipped); the tokens
are weights/airfrans_surface_tokens.npz, the release name of _nf_shape_tokens_airfrans.npz. Both must match their recorded sha256.
  VORTFOIL_CORPUS_DIR=/path/to/corpora python code/field/field_holdout_score.py weights/field_decoder_f0a_s{0,1,2}.npz
  VORTFOIL_CORPUS_DIR=/path/to/corpora python code/field/field_holdout_score.py weights/field_decoder_f0cm_s{0,1,2}.npz
"""
import os, sys, json, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # <release>/code
import vortfoil_paths as VP  # noqa: E402
REPO = VP.add_paths()
SF = os.environ.get("VORTFOIL_CORPUS_DIR", os.path.join(VP.FIELD, "supervised_field"))
TOKENS = os.path.join(VP.WEIGHTS, "airfrans_surface_tokens.npz")   # cd27519b, every run_config tokens_sha256
from supervised_field.nf_eval_shape import fwd_np, encode_np  # noqa: E402

FIELDS = ("ux", "uy", "p", "nut")
_CACHE = {}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load(path, want_sha):
    if path not in _CACHE:
        got = sha256(path)
        assert got == want_sha, f"{path}: sha {got[:12]} != run_config {want_sha[:12]}"
        _CACHE[path] = np.load(path, allow_pickle=True)
    return _CACHE[path]


def score(wpath):
    w = dict(np.load(wpath, allow_pickle=True))
    cfg = json.loads(str(w["run_config"]))
    z = load(os.path.join(SF, os.path.basename(cfg["corpus"].replace("\\", "/"))), cfg["corpus_sha256"])
    tz = load(TOKENS, cfg["tokens_sha256"])
    names = [str(x) for x in z["names"]]
    cid = z["case_id"].astype(np.int64)
    tmap = {str(n): i for i, n in enumerate(tz["names"])}
    tok = tz["surf_tokens"].astype(np.float32)
    s_case = np.asarray(z["s"], np.float32)
    err, rows = [], 0
    for nm in cfg["holdout_names"]:
        ci = names.index(nm)
        m = cid == ci
        zc = encode_np(tok[tmap[nm]].astype(np.float64), w)
        pred = fwd_np(z["X"][m].astype(np.float64), np.full(int(m.sum()), float(s_case[ci])), z["PAINT"][m].astype(np.float64), zc, w)
        err.append(((pred - z["TRUTH"][m].astype(np.float64)) / w["std"][None, :]) ** 2)
        rows += int(m.sum())
    assert rows == cfg["holdout_rows"] and len(cfg["holdout_names"]) == cfg["holdout_cases"], (rows, cfg["holdout_rows"])
    e = np.concatenate(err).mean(0)
    return dict(file=os.path.relpath(wpath, REPO) if wpath.startswith(REPO) else wpath, sha256=sha256(wpath), seed=cfg.get("train_seed"),
                corpus=os.path.basename(cfg["corpus"].replace("\\", "/")), holdout_cases=len(cfg["holdout_names"]), holdout_rows=rows,
                **{f: float(v) for f, v in zip(FIELDS, e)}, score=float(e.mean()))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = score(os.path.abspath(p))
        print(json.dumps(r), flush=True)
