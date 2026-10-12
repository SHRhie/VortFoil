# -*- coding: utf-8 -*-
"""code/wing_view.html, the template of the application's 3D wing page (code/wing_flow.py fills it in): its static contract, and,
where a Chromium-based browser (Chrome, Chromium or Edge) is installed, a headless render of a small synthetic wing in the page's
data contract (the comment at the top of the template). No window opens."""
import base64
import gzip
import json
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "code", "wing_view.html")
BROWSERS = ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")


def browser():
    found = [shutil.which(n) for n in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome", "msedge", "microsoft-edge")]
    return next((p for p in found + list(BROWSERS) if p and os.path.exists(p)), None)


def pack(o):
    """a record as the page's data block: JSON, gzip, base64 ('' for no record)"""
    return base64.b64encode(gzip.compress(json.dumps(o, separators=(",", ":"), allow_nan=False).encode(), 6, mtime=0)).decode() if o else ""


def fill(rec, fld, path):
    with open(PAGE, encoding="utf-8") as f:
        page = f.read().replace("__WING_DATA__", pack(rec)).replace("__FIELD_DATA__", pack(fld))
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path


def synthetic(n=8, nx=9, viscous=True, **wing):
    """(wing record, field record or None): a rectangular wing, half-span 2 m and chord 1 m, a thin lens section, elliptic loading"""
    th = np.linspace(0, np.pi, n + 1)
    yn, y = -2.0 * np.cos(th), -2.0 * np.cos(0.5 * (th[:-1] + th[1:]))
    x = np.linspace(0, 1, nx)
    t = 0.24 * x * (1 - x)
    side = lambda sg, cp: dict(x=np.tile(x, (n, 1)).round(5).tolist(), y=np.repeat(y[:, None], nx, 1).round(5).tolist(),
                               z=np.tile(sg * t, (n, 1)).round(5).tolist(), cp=np.tile(cp, (n, 1)).round(4).tolist())
    ell = np.sqrt(1 - (y / 2.0) ** 2)
    G = 1.5 * ell
    Gp = np.r_[0.0, G, 0.0]
    case = dict(alpha=5.0, CL=0.512, CDi=0.0101, CD0=0.0082, e=0.953, y=y.round(5).tolist(), chord=[1.0] * n, cl=(0.6 * ell).round(5).tolist(),
                alpha_eff=(5 - 1.5 * ell).round(4).tolist(), gamma=G.round(5).tolist(),
                surface=dict(upper=side(1, -1.2 * np.sin(np.pi * x) - 0.1), lower=side(-1, 0.4 * np.sin(np.pi * x))),
                trailing=[dict(y=float(yn[k]), x0=0.25, z0=0.0, strength=float(Gp[k] - Gp[k + 1])) for k in range(n + 1)],
                strips_past_stall=0, residual=1e-12, warnings=["strip 3 near stall (a < b)"])
    polar = [dict(alpha=float(a), CL=0.1 * a + 0.012, CDi=0.001 * a * a, CD0=0.008, e=0.95) for a in range(-4, 13)]
    rec = dict(wing=dict(name="Test wing, NACA 2412", section="NACA 2412", span=4.0, S=4.0, AR=4.0, V=30.0, Re_root=2e6, Re_tip=2e6, n=n,
                         stations=[dict(y=0.0, chord=1.0, twist=0.0, x=0.0, z=0.0, section="NACA 2412"),
                                   dict(y=2.0, chord=1.0, twist=0.0, x=0.0, z=0.0, section="NACA 2412")], **wing),
               cases=[case], polar=polar, focus=5.0, warnings=["aspect ratio 3.5 < 4: loads are unreliable"], wake_model="straight trailing legs")
    if not viscous:
        return rec, None
    line = lambda yy: dict(x=np.linspace(-0.5, 2.0, 12).round(4).tolist(), y=[yy] * 12, z=(0.05 * np.sin(np.linspace(0, 3, 12))).round(4).tolist(),
                           cp=(0.3 * np.sin(np.linspace(0, 6, 12))).round(4).tolist())
    q = lambda nu, nv: base64.b64encode(bytes((np.arange(nu * nv) % 255).astype(np.uint8))).decode()
    planes = [dict(kind="xz", at=0.0, u0=-0.3, u1=1.3, v0=-0.3, v1=0.3, nu=6, nv=5, q=q(6, 5)),
              dict(kind="yz", at=1.4, u0=-2.2, u1=2.2, v0=-0.4, v1=0.2, nu=7, nv=5, q=q(7, 5))]
    sheet = dict(nj=3, ni=4, x=[1.0 + i * 0.2 for j in range(3) for i in range(4)], y=[-1.0 + j for j in range(3) for i in range(4)],
                 z=[0.0] * 12, q=[0.2 + 0.05 * i for j in range(3) for i in range(4)])
    fld = dict(kind="viscous", decoder="test.npz", decoder_warnings=["Reynolds number 1.8e6 lies outside the decoder's range"],
               cases=[dict(alpha=5.0, lines=[line(v) for v in (-1.5, -0.5, 0.5, 1.5)], planes=planes, sheet=sheet)])
    return rec, fld


def test_template_contract():
    """two placeholders in two gzip + base64 blocks; English only; the research viewer's oblique preset first; no height exaggeration"""
    html = open(PAGE, encoding="utf-8").read()
    for ident, mark in (("wing-data", "__WING_DATA__"), ("field-data", "__FIELD_DATA__")):
        assert f'<script type="application/gzip+base64" id="{ident}">{mark}</script>' in html and html.count(mark) == 1   # nowhere else: write_page replaces every one
    assert not re.search(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]", html), "Korean text in the page"
    for text in ("Streamlines", "Viscous wake", "Section planes", "Trailing vortices", "Surface colour", "Oblique", "Top", "Side", "Coefficients",
                 "Warnings", "DecompressionStream", "oblique: [225, 34]", "top: [0, 89.5]", "side: [270, 0.5]"):
        assert text in html, text
    assert not re.search(r"×\s*4|exag", html) and "view.az = PRESETS" not in html
    assert "view = { az: PRESETS.oblique[0] * DEG, el: PRESETS.oblique[1] * DEG" in html            # the default view is the oblique preset
    assert "(min-width: 901px) and (orientation: landscape)" in html and "(max-width: 900px), (orientation: portrait)" in html
    assert "aspect-ratio" not in html          # the two layouts are exact complements: a square window (1:1 matched both rules) is the stacked one
    assert len(html) < 80_000


def test_data_blocks_round_trip():
    rec, fld = synthetic()
    for o in (rec, fld):
        assert json.loads(gzip.decompress(base64.b64decode(pack(o)))) == json.loads(json.dumps(o))
    assert pack(None) == ""


def dump(path):
    """the DOM of the page after headless Chromium has run its script (virtual time, no window), without the script elements"""
    prof = tempfile.mkdtemp(prefix="vf-page-")
    try:
        r = subprocess.run([browser(), "--headless=new", f"--user-data-dir={prof}", "--window-size=1500,950", "--virtual-time-budget=8000",
                            "--no-first-run", "--use-mock-keychain", "--dump-dom", "file:///" + path.replace("\\", "/").lstrip("/")],   # no keychain dialog on macOS
                           capture_output=True, encoding="utf-8", errors="replace", timeout=120)
    finally:
        shutil.rmtree(prof, ignore_errors=True)
    return re.sub(r"<script.*?</script>", "", r.stdout, flags=re.S)


needs_browser = pytest.mark.skipif(browser() is None, reason="no Chrome, Chromium or Edge to render the page")


@needs_browser
def test_page_renders_with_a_flow_field(tmp_path):
    rec, fld = synthetic()
    dom = dump(fill(rec, fld, str(tmp_path / "wing.html")))
    assert re.search(r'<html[^>]* data-view="ready"', dom) or re.search(r'<html[^>]* data-view="no-webgl"', dom), dom[:300]
    assert '<h1 id="title">Test wing, NACA 2412</h1>' in dom and "<title>Test wing, NACA 2412 · VortFoil 3D view</title>" in dom
    for value in ("0.512", "0.01010", "0.953", "0.00820", "0.01830", "28.0"):                      # C_L, C_D,i, e, C_D,0, C_D, L/D
        assert f'class="v">{value}<' in dom, value
    assert 'class="v">—<' in dom                                                                     # C_M was not given: a dash
    assert "aspect ratio 3.5 &lt; 4: loads are unreliable" in dom and "strip 3 near stall (a &lt; b)" in dom
    assert "Flow field: Reynolds number 1.8e6 lies outside the decoder's range" in dom
    assert "⚠ 3 warnings" in dom and "Re <b>2.0×10⁶</b> root" in dom and "V<sub>∞</sub> <b>30 m/s</b>" in dom
    assert re.search(r'id="opt-stream"(?![^>]*disabled)', dom) and not re.search(r'id="tog-planes"[^>]*hidden', dom) and 'class="err"' not in dom
    assert "<polyline" in dom and re.search(r'id="alpha-group"[^>]*hidden', dom)                    # charts drawn; one incidence: no buttons


@needs_browser
def test_page_renders_without_a_flow_field_and_with_a_state(tmp_path):
    rec, _ = synthetic(viscous=False, state="sst")
    rec["cases"][0].update(CM=-0.0512, CD=0.0183, LD=28.0, CD0_pressure=0.003, CD0_friction=0.0052)
    dom = dump(fill(rec, None, str(tmp_path / "none.html")))
    assert re.search(r'data-view="(ready|no-webgl)"', dom)
    assert 'class="v">−0.0512<' in dom and "C<sub>D,0</sub> pressure" in dom and "RANS state" in dom
    assert re.search(r'id="opt-stream"[^>]*disabled', dom) and re.search(r'id="opt-sheet"[^>]*disabled', dom) and re.search(r'id="tog-planes"[^>]*hidden', dom)


@needs_browser
def test_clip_planes_hold_the_rolled_up_wake(tmp_path):
    """the depth range is 3 half-spans either side of the camera distance, and more when a free wake runs further downstream than that
    (the camera may look from there, and the near plane would cut the wake off)"""
    depth = lambda dom: float(re.search(r'<html[^>]* data-depth="([0-9.]+)"', dom).group(1))
    rec, fld = synthetic()
    assert depth(dump(fill(rec, fld, str(tmp_path / "straight.html")))) == 3.0
    case = rec["cases"][0]                                    # a wake 4.3 half-spans long (the half-span is 2 m); the view's target is 0.2 half-spans downstream
    case["wake"] = [[[t["x0"] + s, t["y"], t["z0"]] for s in np.linspace(0, 8.6, 51).round(4).tolist()] for t in case["trailing"]]
    far = max(np.hypot(p[0] - 0.4, np.hypot(p[1], p[2])) for pl in case["wake"] for p in pl) / 2.0
    assert far > 3 and depth(dump(fill(rec, fld, str(tmp_path / "free.html")))) == pytest.approx(1.1 * far, abs=0.006)


@needs_browser
def test_page_with_broken_data_says_so(tmp_path):
    rec, _ = synthetic()
    page = fill(rec, None, str(tmp_path / "broken.html"))
    html = open(page, encoding="utf-8").read().replace(pack(rec), "AAAA")
    open(page, "w", encoding="utf-8").write(html)
    dom = dump(page)
    assert re.search(r'<html[^>]* data-view="error"', dom) and "Could not unpack the data of this page" in dom   # on <html>: the page's CSS has the bare text too
