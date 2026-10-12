# -*- coding: utf-8 -*-
"""The 3D view button of the application (code/vortfoil_app.py): the options window, the page made in the worker (a stand-in for
code/wing_flow.py), the browser started on the Tk thread, progress and Cancel, and a finished page opened again. Tk is the recording
stand-in of test_app_wing; no window opens, nothing is computed and no browser starts."""
import os
import pathlib
import re
import sys
import time
import types
import webbrowser

import numpy as np
import pytest

from test_app_wing import Button, Radiobutton, small_wing, settle, texts, tk  # noqa: F401  (tk is a fixture)
import vortfoil_app as A  # noqa: E402

RES = dict(wing=dict(name="Small test wing, NACA 2412"), alpha=5.0, V=30.0, state="xfoil", warnings=[], strips=dict(Re=np.array([3e6, 2e6])))
STEM = "vortfoil_wing_Small_test_wing_NACA_2412_a5_V30_xfoil"


ERRORS = dict(value=lambda: ValueError("no flow here"), runtime=lambda: RuntimeError("boom"), silent=RuntimeError, memory=MemoryError)


class FakeFlow(types.ModuleType):
    """a stand-in for code/wing_flow.py: records its calls, reports progress, waits for Cancel or fails when told to"""

    def __init__(self):
        super().__init__("wing_flow")
        self.calls, self.mode = [], "ok"

    def view_data(self, res, field="viscous", free_wake=False, progress=None, cancel=None):
        self.calls.append((field, free_wake))
        progress(0.0, "first unit")
        progress(0.5, "half way")
        if self.mode == "wait":                                       # until Cancel (the worker's cancel() is the app's flag)
            t0 = time.time()
            while not cancel():
                assert time.time() - t0 < 60
                time.sleep(0.01)
            raise RuntimeError("cancelled")
        if self.mode in ERRORS:
            raise ERRORS[self.mode]()
        return {"wing": {"name": res["wing"]["name"]}, "field": None}

    def re_floor(self, res):
        return 1.5e6                                                  # the released decoder's: 3/4 of its lower training Re

    def write_page(self, data, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("<html>" + data["wing"]["name"] + "</html>")
        return path


@pytest.fixture
def view3d(tk, monkeypatch, tmp_path):
    """(app in the 3D mode with a wing result, the stand-in flow module, the URLs the browser was asked to open)"""
    flow, opened = FakeFlow(), []
    monkeypatch.setitem(sys.modules, "wing_flow", flow)
    monkeypatch.setattr(A, "PAGES", str(tmp_path / "pages"))
    monkeypatch.setattr(webbrowser, "open", lambda url, *a, **k: opened.append(url) or True)
    app, root = tk()
    settle(app, lambda: app.section is not None)
    small_wing(app)
    app.wres = RES
    app.sync()
    return app, flow, opened


def cancels(app):
    """what Run's command may be while a case runs: the Cancel flag's set, or the application's debounced cancel_run where it has one"""
    return (app.cancel.set, getattr(app, "cancel_run", app.cancel.set))


def press_open(app, field=None, free=None):
    app.open_view3d()
    if field is not None:
        app.v3field.set(field)
    if free is not None:
        app.v3free.set(free)
    texts(app.v3win)["Open"].kw["command"]()


def test_button_sits_next_to_open_cut_and_follows_the_result(tk):
    """In the last wing row, after Open cut in 2D; off until there is a wing result, and while a wing (or a page) is being made."""
    app, root = tk()
    t = texts(app.f3)
    btn, cut = t["3D view"], t["Open cut in 2D"]
    assert isinstance(btn, Button) and btn is app.view3d_btn and btn.master is cut.master and btn.master.master is app.f3
    assert btn.kw["command"] == app.open_view3d and btn["state"] == "disabled" and app.v3win is None
    app.wres = RES
    app.sync()
    assert btn["state"] == "normal"
    app.lock(True, wing=True)                                              # a wing runs: Run reads Cancel
    assert app.run_btn["text"] == "Cancel" and btn["state"] == "disabled"
    app.lock(False)
    assert app.run_btn["text"] == "Run" and btn["state"] == "normal"
    app.wres = None
    app.sync()
    assert btn["state"] == "disabled"


def test_a_case_that_starts_does_not_switch_export_back_on_while_it_writes(view3d):
    """Export data (CSV) is grey while its files are written (export() and export_wing() do that, done_export() ends it): the case that
    starts meanwhile (a wing Run, a page of the 3D view, a 2D Run) switches the 3D view button, not Export, and a second export cannot
    begin in the same folder."""
    app, flow, opened = view3d
    for wing in (True, False):
        app.csv_btn.configure(state="disabled")
        app.lock(True, wing=wing)
        assert app.csv_btn["state"] == "disabled" and app.view3d_btn["state"] == ("disabled" if wing else "normal")
        app.lock(False)
        assert app.csv_btn["state"] == "disabled" and app.view3d_btn["state"] == "normal"
        app.sync()                                                         # the export's end
        assert app.csv_btn["state"] == "normal"


def test_options_window_is_modeless_and_remembers_its_choice(view3d):
    app, flow, opened = view3d
    app.open_view3d()
    win, t = app.v3win, texts(app.v3win)
    assert win.visible and win.name == "3D view" and not win.destroyed
    radios = [w for w in win.walk() if isinstance(w, Radiobutton)]
    assert [(r.kw["text"], r.kw["value"]) for r in radios] == [("Viscous (RANS field decoder)", "viscous"), ("Inviscid streamlines", "inviscid"),
                                                              ("None", "none")]
    assert all(r.kw["variable"] is app.v3field for r in radios) and app.v3field.get() == "viscous"
    assert t["Rolled-up wake (free wake; slower)"].kw["variable"] is app.v3free and app.v3free.get() is False
    assert {"Flow field:", "Open", "Cancel"} <= set(t) and "grab" not in str(win.bound)
    t["Cancel"].kw["command"]()                                            # hidden, not destroyed: the choice stays
    assert not win.visible and not win.destroyed
    app.v3field.set("none"); app.v3free.set(True)
    for hide in (lambda: win.bound["<Escape>"](None), win.bound["WM_DELETE_WINDOW"]):
        app.open_view3d()
        assert win.visible and app.v3win is win and app.v3field.get() == "none" and app.v3free.get() is True
        hide()
        assert not win.visible
    app.run_btn.configure(text="Cancel")                                   # a case runs: nothing opens
    app.open_view3d()
    assert not win.visible
    app.run_btn.configure(text="Run"); app.wres = None
    app.open_view3d()
    assert not win.visible and flow.calls == [] and opened == []


def test_viscous_is_off_below_the_decoders_reynolds_floor(view3d):
    """A wing with a strip below wing_flow.re_floor gets no viscous field: the window opens on Inviscid streamlines, Viscous is grey
    with the reason under it; for a wing inside the range both come back (and the choice made meanwhile stays)."""
    app, flow, opened = view3d
    app.wres = dict(RES, strips=dict(Re=np.array([4e5, 2e5])))
    app.open_view3d()
    radios = {r.kw["value"]: r for r in app.v3win.walk() if isinstance(r, Radiobutton)}
    assert radios["viscous"] is app.v3visc and app.v3visc["state"] == "disabled" and app.v3field.get() == "inviscid"
    assert app.v3note.mapped and app.v3note.master is app.v3visc.master and app.v3note.kw["foreground"] == "gray"
    assert app.v3note.kw["text"] == "Not for this wing: the viscous field needs Re >= 1.5e6 on every strip (this one goes down to Re 2e5)."
    assert app.v3note.kw["wraplength"] == 360                              # it wraps: one 100-character line would double the window's width
    app.v3field.set("none"); app.v3win.withdraw()
    app.wres = RES
    app.open_view3d()
    assert app.v3visc["state"] == "normal" and not app.v3note.mapped and app.v3field.get() == "none" and app.v3win.visible


def test_a_result_that_arrives_while_the_window_is_up_updates_it(view3d, monkeypatch):
    """The window is modeless, so a wing run can finish under it: the viscous field follows the new result at once (greyed for a
    small wing, so [Open] makes the inviscid page instead of a refusal; back for a big one)."""
    app, flow, opened = view3d
    monkeypatch.setattr(app, "draw", lambda layout: None)                  # (the figure of a real result: test_app_wing)
    monkeypatch.setattr(A, "wing_summary", lambda res, dt: "summary")
    app.open_view3d()
    assert app.v3visc["state"] == "normal" and app.v3field.get() == "viscous"
    app.done_wing((), (dict(RES, strips=dict(Re=np.array([4e5, 2e5]))), 1.0), None)
    assert app.v3win.visible and app.v3visc["state"] == "disabled" and app.v3note.mapped and app.v3field.get() == "inviscid"
    texts(app.v3win)["Open"].kw["command"]()
    settle(app, lambda: opened)
    assert flow.calls == [("inviscid", False)]
    app.open_view3d()
    app.done_wing((), (RES, 1.0), None)
    assert app.v3win.visible and app.v3visc["state"] == "normal" and not app.v3note.mapped


def test_options_window_goes_with_the_3d_mode(view3d):
    """Unticking 3D wing (Open cut in 2D does that too) hides the options window: hidden, not destroyed, so its choice stays."""
    app, flow, opened = view3d
    app.open_view3d()
    win = app.v3win
    assert win.visible
    app.wing.set(False); app.show()
    assert not win.visible and not win.destroyed and app.v3win is win


def test_open_after_a_case_started_says_why_nothing_happens(view3d):
    """A case that started while the options window was open: [Open] closes the window, queues nothing and says what to do."""
    app, flow, opened = view3d
    app.open_view3d()
    app.lock(True, wing=True)
    texts(app.v3win)["Open"].kw["command"]()
    assert not app.v3win.visible and flow.calls == [] and opened == [] and app.run_btn["text"] == "Cancel"
    assert app.status.kw["text"] == "A case is running: press 3D view again when it has finished."
    app.lock(False)
    assert app.view3d_btn["state"] == "normal"


def test_open_makes_the_page_in_the_worker_and_starts_the_browser(view3d, tmp_path):
    app, flow, opened = view3d
    press_open(app, "inviscid", True)
    assert not app.v3win.visible                                           # [Open] closes the options window
    assert app.run_btn["text"] == "Cancel" and app.run_btn["state"] == "normal" and app.run_btn.kw["command"] in cancels(app)
    assert app.chk["state"] == "disabled" and app.view3d_btn["state"] == "disabled" and app.bar.kw["mode"] == "determinate"
    assert 0 <= time.monotonic() - app.started < 30                        # the time Run's Cancel debounce counts from
    assert app.status.kw["text"] == "Preparing the 3D view of Small test wing, NACA 2412: flow field inviscid, rolled-up wake ..."
    settle(app, lambda: opened)
    path = str(tmp_path / "pages" / (STEM + "_inviscid_fw.html"))
    assert flow.calls == [("inviscid", True)] and open(path).read() == "<html>Small test wing, NACA 2412</html>"
    assert opened == [pathlib.Path(path).as_uri()] and path == A.page_path(RES, "inviscid", True)
    assert app.run_btn["text"] == "Run" and app.chk["state"] == "normal" and app.view3d_btn["state"] == "normal"
    assert re.fullmatch(r"3D view ready in [0-9.]+ s and opened in your browser: " + re.escape(path), app.status.kw["text"])


def test_the_preparing_status_says_what_flow_field_it_makes(view3d):
    """'flow field viscous', 'flow field inviscid', and for None 'no flow field' (not 'flow field none')."""
    app, flow, opened = view3d
    for field, free, text in (("none", False, "no flow field ..."), ("none", True, "no flow field, rolled-up wake ..."),
                              ("inviscid", False, "flow field inviscid ...")):
        press_open(app, field, free)
        assert app.status.kw["text"] == "Preparing the 3D view of Small test wing, NACA 2412: " + text
        n = len(opened)
        settle(app, lambda: len(opened) > n and app.run_btn["text"] == "Run")


def test_progress_and_cancel(view3d):
    app, flow, opened = view3d
    flow.mode = "wait"
    press_open(app)
    base = "Preparing the 3D view of Small test wing, NACA 2412: flow field viscous ..."
    settle(app, lambda: app.status.kw["text"].endswith("half way"))
    assert app.status.kw["text"] == base + " half way" and app.bar.kw["value"] == 50.0 and app.bar.kw["mode"] == "determinate"
    app.started = 0.0                                                      # (the click of a user comes later than a double click's 2nd: no debounce)
    app.run_btn.kw["command"]()                                            # Run reads Cancel
    settle(app, lambda: app.run_btn["text"] == "Run")
    assert app.status.kw["text"] == "Cancelled. The 3D view was not made." and opened == [] and app.pages == {}
    assert app.view3d_btn["state"] == "normal" and app.bar.kw["value"] == 0


@pytest.mark.parametrize("mode, text", [
    ("value", "The 3D view failed: no flow here"), ("runtime", "The 3D view failed: boom"),
    ("silent", "The 3D view failed: RuntimeError"),                       # an error with no message is named, not left blank
    ("memory", "The 3D view failed: not enough memory (close other programs, or choose Inviscid streamlines or None)")])
def test_failures_reach_the_status_line(view3d, mode, text, capsys):
    app, flow, opened = view3d
    flow.mode = mode
    press_open(app)
    settle(app, lambda: app.run_btn["text"] == "Run" and app.status.kw["text"] != "Preparing the 3D view of Small test wing, NACA 2412: flow field viscous ...")
    assert app.status.kw["text"] == text and opened == [] and app.pages == {} and app.view3d_btn["state"] == "normal"
    assert ("Traceback" in capsys.readouterr().err) == (mode != "value")        # a ValueError is the user's to fix: no trace


def test_a_finished_page_is_opened_again_for_the_same_result_and_options(view3d):
    app, flow, opened = view3d
    press_open(app, "viscous", False)
    settle(app, lambda: len(opened) == 1)
    path = app.pages[("viscous", False)]
    press_open(app, "viscous", False)                                      # same result, same options: no recomputation
    assert len(opened) == 2 and opened[0] == opened[1] and flow.calls == [("viscous", False)] and app.run_btn["text"] == "Run"
    assert app.status.kw["text"] == f"3D view opened again in your browser: {path}"
    press_open(app, "none", False)                                         # other options: made
    settle(app, lambda: len(opened) == 3)
    assert flow.calls == [("viscous", False), ("none", False)] and len(app.pages) == 2
    os.remove(path)                                                        # a page that was deleted is made again
    press_open(app, "viscous", False)
    settle(app, lambda: len(opened) == 4)
    assert len(flow.calls) == 3 and os.path.exists(path)
    app.wres = dict(RES)                                                   # a new wing result starts afresh
    app.sync()
    press_open(app, "viscous", False)
    settle(app, lambda: len(opened) == 5)
    assert len(flow.calls) == 4 and list(app.pages) == [("viscous", False)] and app.pages_of is app.wres


def test_no_browser_says_where_the_page_is(view3d, monkeypatch):
    app, flow, opened = view3d
    for behaviour in (lambda url, *a, **k: False, lambda url, *a, **k: (_ for _ in ()).throw(OSError("no launcher"))):
        monkeypatch.setattr(webbrowser, "open", behaviour)
        app.pages_of = None                                                # (make the page again each time)
        press_open(app)
        settle(app, lambda: "No browser" in app.status.kw["text"])
        assert re.fullmatch(r"3D view ready in [0-9.]+ s\. No browser could be started; open this file in one: .*viscous\.html", app.status.kw["text"])


def test_page_names():
    """<cache>/vortfoil/wing3d/<the figure's stem>_<field>[_fw].html, with the folder next to the UIUC cache"""
    assert pathlib.Path(A.PAGES) == pathlib.Path(A.CACHE).parent / "wing3d"
    res = dict(RES, wing=dict(name="Kinked wing (4415 to 2412, dihedral), NACA 4415"), state="sst", V=60.0)
    assert os.path.basename(A.page_path(res, "viscous", False)) == "vortfoil_wing_Kinked_wing_4415_to_2412_dihedral_NACA_4415_a5_V60_sst_viscous.html"
    assert os.path.basename(A.page_path(RES, "none", True)) == STEM + "_none_fw.html"
    assert A.App.TASKS["wing3d"] is A.wing3d_page
