# -*- coding: utf-8 -*-
"""The application's 3D wing mode (code/vortfoil_app.py) without a display: the pure helpers it is built from, then the App
itself on a recording stand-in for Tk (strict: an unknown widget option or method fails, as it would in Tk), driven through
its worker queue like the real event loop does. No window opens and nothing is fetched from the network."""
import json
import os
import sys
import threading
import time
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "code", "field"))

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib.backend_bases import MouseEvent, NavigationToolbar2  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
import vortfoil_app as A  # noqa: E402
import view_airfoil as VA  # noqa: E402
import view_wing as WV  # noqa: E402
import wing_nll as WG  # noqa: E402

FULL, KINKED, CLARK = "Full-scale wing (span 10 m)", "Kinked wing (4415 to 2412, dihedral)", "Clark Y wing"
SMALL = "# y chord twist x z section\n0 1.0 0 0 0 current\n3, 0.6, -2, 0, 0, NACA 4412\n"      # 8 strips: the tests' wing


# ------------------------------------------------------------------------------------------------ pure helpers
def test_stations_text_round_trip():
    """Every preset survives text -> stations; the text takes comments, commas, tabs, quotes and 'NACA 2412'."""
    for name, d in WG.PRESETS.items():
        st = A.parse_stations(A.format_stations(d["stations"]))
        assert [s["section"] for s in st] == [s["section"] for s in d["stations"]], name
        for a, b in zip(st, d["stations"]):
            assert all(np.isclose(a[k], b[k], rtol=1e-7, atol=1e-12) for k in A.COLUMNS), name
    assert A.format_stations(WG.PRESETS[FULL]["stations"]).splitlines()[0] == "# y_m  chord_m  twist_deg  x_c4_m  z_c4_m  section"
    st = A.parse_stations(SMALL + '\n  # a comment\n\n5\t0.5 0 0 0 "C:\\my wings\\sec 1.dat"\n')
    assert [s["section"] for s in st] == ["current", "NACA 4412", "C:\\my wings\\sec 1.dat"] and st[1]["y"] == 3.0 and st[1]["twist"] == -2.0
    assert WG.validate(dict(WG.PRESETS[FULL], stations=st[:2]))["stations"][1]["section"] == "4412"


@pytest.mark.parametrize("text, message", [
    ("0 1 0 0 0", "line 1: expected y, chord, twist, x, z and a section"),
    ("# header\n0 1 0 0 0 2412\n3 0.5 two 0 0 2412", "line 3: y, chord, twist, x and z must be numbers"),
    ("0 1 0 0", "line 1: expected"),
])
def test_stations_errors_name_the_line(text, message):
    with pytest.raises(ValueError, match=message):
        A.parse_stations(text)


def test_wing_definition_scales_and_validates():
    """Scale x multiplies every length (not the twist), Strips is a whole number 8 to 200; errors name the form's field."""
    d = A.wing_definition(SMALL, "2", "12", "Mine")
    assert d["name"] == "Mine, scale x 2" and d["n"] == 12 and d["stations"][1]["y"] == 6.0 and d["stations"][1]["chord"] == 1.2
    assert d["stations"][1]["twist"] == -2.0 and A.wing_definition(SMALL)["name"] == "Custom wing"
    for args, message in [((SMALL, "x"), "Scale x: 'x' is not a number"), ((SMALL, "0"), "Scale x: must be positive"),
                          ((SMALL, "1", "4"), "Strips: must be a whole number from 8 to 200"), ((SMALL, "1", "40.5"), "Strips"),
                          (("", ), "stations: give at least two"), (("0 1 0 0 0 2412\n3 -1 0 0 0 2412", ), "station 2 chord")]:
        with pytest.raises(ValueError, match=message):
            A.wing_definition(*args)


def test_pick_wing(tmp_path):
    assert A.pick_wing(FULL) is WG.PRESETS[FULL] and A.pick_wing("kinked")["name"] == KINKED and A.pick_wing(" clark y ")["name"] == CLARK
    WG.save_wing(WG.PRESETS[FULL], tmp_path / "mine.json")
    assert A.pick_wing(str(tmp_path / "mine.json"))["V"] == 45.0
    for spec, word in (("rectangular", "ambiguous"), ("nothing like it", "neither a preset nor"), ("", "ambiguous")):
        with pytest.raises(ValueError, match=word) as e:
            A.pick_wing(spec)
        assert FULL in str(e.value)


def test_resolve_sections():
    """'current' becomes its NACA designation (the RANS state knows those) or goes in as coordinates; 'uiuc:' is fetched once per name."""
    loop, loaded = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]), []
    st = A.parse_stations("0 1 0 0 0 current\n1 1 0 0 0 uiuc:e387\n2 1 0 0 0 uiuc:e387\n3 1 0 0 0 4412")
    d = dict(WG.PRESETS[FULL], stations=st)
    got, sec = A.resolve_sections(d, (loop, "2412"), lambda n: loaded.append(n) or loop * 2)
    assert got["stations"][0]["section"] == "2412" and sec.keys() == {"uiuc:e387"} and loaded == ["e387"] and np.all(sec["uiuc:e387"] == 2 * loop)
    got, sec = A.resolve_sections(d, (loop, None), lambda n: loop)
    assert got["stations"][0]["section"] == "current" and sec.keys() == {"current", "uiuc:e387"} and np.all(sec["current"] == loop)
    for args, message in (((d, None, lambda n: loop), "'current' needs an airfoil"), ((d, (loop, None), None), "could not load it")):
        with pytest.raises(ValueError, match=message):
            A.resolve_sections(*args)
    assert d["stations"] is st and st[0]["section"] == "current"                      # the caller's definition is left alone


def test_selftest_arguments():
    a = A.make_parser().parse_args(["--selftest", "out", "--wing", "kinked", "--V", "55", "--state", "sst", "--alpha", "6"])
    assert (a.selftest, a.wing, a.V, a.state, a.alpha, a.airfoil) == ("out", "kinked", 55.0, "sst", 6.0, "naca:2412")
    a = A.make_parser().parse_args(["--selftest", "out"])
    assert (a.wing, a.V, a.state) == (None, None, "xfoil")
    with pytest.raises(SystemExit):
        A.make_parser().parse_args(["--state", "rans"])


def test_wing_names_and_summary():
    res = dict(wing=dict(name="Kinked wing (4415 to 2412, dihedral)"), alpha=6.0, V=60.0, state="sst", warnings=["a", "b"],
               coef=dict(CL=0.7234, CD=0.031234, CM=-0.0549), strips=dict(Re=np.array([6e6, 4e6, 2e6])), sst_range=dict(message=None))
    assert A.wing_base(res) == "vortfoil_wing_Kinked_wing_4415_to_2412_dihedral_a6_V60_sst"
    s = A.wing_summary(res, 18.84)
    assert s == ("Kinked wing (4415 to 2412, dihedral), 6 deg, 60 m/s (RANS state): C_L 0.723, C_D 0.0312, C_M -0.055, done in 18.8 s; "
                 "2 warning(s), listed in the figure.")                                   # no sentence on dragging: the figure's titles say it
    assert A.wing_summary(dict(res, warnings=[], state="xfoil"), 1.0).endswith("done in 1.0 s; no range warnings.")
    out = A.wing_summary(dict(res, coef=dict(res["coef"], in_range=False)), 2.0)                  # the requested incidence is outside the state's training range
    assert out.endswith("done in 2.0 s; 2 warning(s), listed in the figure. This incidence is outside the RANS state's training range: the numbers are extrapolated.")
    assert A.wing_summary(dict(res, state="xfoil", coef=dict(res["coef"], in_range=False)), 2.0).endswith("outside the XFOIL state's training range: the numbers are extrapolated.")
    assert "outside the" not in A.wing_summary(dict(res, coef=dict(res["coef"], in_range=True)), 2.0).split("done in")[1]
    far = A.wing_summary(dict(res, coef=dict(res["coef"], in_range=False), sst_range=dict(message="RANS state outside the field decoder's Reynolds range")), 2.0)
    assert far.startswith("RANS state outside the field decoder's Reynolds range. ") and "training range: the numbers are extrapolated" not in far   # its lead sentence says it
    assert A.naca_code("NACA 23012", "naca") == "23012" and A.naca_code("NACA 2412", None) is None and A.naca_code("clarky", "naca") is None
    assert A.wing_base(dict(res, wing=dict(name="직사각형 날개"))) == "vortfoil_wing_직사각형_날개_a6_V60_sst"      # a name in any script stays one
    assert A.wing_base(dict(res, wing=dict(name="테이퍼 날개"))) != A.wing_base(dict(res, wing=dict(name="직사각형 날개")))


def test_the_summary_leads_with_what_makes_the_numbers_meaningless():
    """The RANS state far outside its decoder's Reynolds range, the XFOIL state far outside the surface model's, a result that is not
    finite: the status line says so first; the run itself still ends as a result."""
    res = dict(wing=dict(name="Model wing"), alpha=6.0, V=20.0, state="sst", warnings=[], coef=dict(CL=-3.767, CD=2.0011, CM=3.157),
               strips=dict(Re=np.full(40, 3e5)))
    lead = ("RANS state outside the field decoder's Reynolds range (2e6-6e6) on 40 of 40 strips: the RANS-state numbers are not meaningful here; "
            "use the XFOIL state")
    s = A.wing_summary(dict(res, sst_range=dict(message=lead)), 19.8)
    assert s == lead + ". Model wing, 6 deg, 20 m/s (RANS state): C_L -3.767, C_D 2.0011, C_M 3.157, done in 19.8 s; no range warnings."
    x = A.wing_summary(dict(res, state="xfoil", strips=dict(Re=np.r_[np.full(21, 1e4), np.full(19, 3e6)])), 1.0)
    assert x.startswith("Reynolds number outside the surface model's range (5e4-1e7) on 21 of 40 strips: the numbers are not meaningful here. Model wing, ")
    assert A.wing_summary(dict(res, state="xfoil", strips=dict(Re=np.r_[np.full(20, 1e4), np.full(20, 3e6)])), 1.0).startswith("Model wing, ")   # half is not more than half
    nf = A.wing_summary(dict(res, state="xfoil", strips=dict(Re=np.full(40, 3e6)), coef=dict(CL=np.nan, CD=np.nan, CM=np.nan)), 0.5)
    assert nf.startswith("The result is not finite: the numbers are not meaningful (check the speed and the size). Model wing, ")
    assert A.wing_summary(dict(res, state="xfoil", strips=dict(Re=np.full(40, 3e6))), 0.5).startswith("Model wing, ")
    nc = A.wing_summary(dict(res, state="xfoil", strips=dict(Re=np.full(40, 3e6)), newton=dict(residual=0.3, iterations=60)), 1.7)
    assert nc.startswith("No lifting-line solution (residual 3e-01): the numbers are not meaningful. Model wing, ")
    assert A.wing_summary(dict(res, state="xfoil", strips=dict(Re=np.full(40, 3e6)), newton=dict(residual=1e-9, iterations=9)), 1.7).startswith("Model wing, ")
    both = A.wing_summary(dict(res, sst_range=dict(message=lead), newton=dict(residual=8.0, iterations=60)), 1.0)
    assert both.startswith(lead + ". Model wing, ")                                           # the decoder's range first: that is the cause


def test_section_names_and_cut_names():
    assert [A.section_name(t) for t in ("2412", "NACA 23012", "naca4412", "uiuc:clarky", "/tmp/my foils/e387.dat")] == [
        "NACA 2412", "NACA 23012", "NACA 4412", "clarky", "e387"]


# ------------------------------------------------------------------------------------------------ a recording stand-in for Tk
class Var:
    def __init__(self, master=None, value=None):
        self.v, self.traces = "" if value is None else value, []

    def get(self):
        return self.v

    def set(self, v):
        self.v = v
        for callback in self.traces:                                                              # a write trace runs at every set, as in Tk
            callback("PY_VAR0", "", "write")

    def trace_add(self, mode, callback):
        assert mode == "write"
        self.traces.append(callback)


COMMON = {"style", "takefocus", "cursor", "class_"}
GEOMETRY = {"pack": {"side", "fill", "expand", "anchor", "padx", "pady", "ipadx", "ipady", "before", "after", "in_"},
            "grid": {"row", "column", "rowspan", "columnspan", "sticky", "padx", "pady", "ipadx", "ipady", "in_"}}


class Widget:
    """a widget that records what it is given and fails on an option or a method Tk does not have"""
    OPTS = frozenset()

    def __init__(self, master=None, **kw):
        self.master, self.kw, self.bound, self.children, self.destroyed, self.mapped, self.cell = master, {}, {}, [], False, False, {}
        self.configure(**kw)
        if master is not None:
            master.children.append(self)

    def configure(self, **kw):
        assert set(kw) <= self.OPTS, f"{type(self).__name__} has no option {sorted(set(kw) - self.OPTS)}"
        self.kw.update(kw)
    config = configure

    def cget(self, key):
        return self.kw[key]

    def __getitem__(self, key):
        return self.kw.get(key, "normal" if key == "state" else "")

    def bind(self, sequence, func, add=None):
        self.bound[sequence] = func

    def pack(self, **kw):
        assert set(kw) <= GEOMETRY["pack"], sorted(kw)
        self.mapped = True

    def pack_forget(self):
        self.mapped = False

    def grid(self, **kw):
        assert set(kw) <= GEOMETRY["grid"], sorted(kw)
        self.mapped, self.cell = True, kw or self.cell                         # no options: the cell it was removed from (Tk remembers it)

    def grid_remove(self):
        self.mapped = False

    def columnconfigure(self, index, **kw):
        assert set(kw) <= {"weight", "minsize", "pad", "uniform"}
    rowconfigure = columnconfigure

    def destroy(self):
        self.destroyed = True

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


class Frame(Widget):
    OPTS = COMMON | {"padding", "borderwidth", "relief", "width", "height"}


class LabelFrame(Frame):
    OPTS = Frame.OPTS | {"text", "labelanchor", "labelwidget", "underline"}


class Label(Widget):
    OPTS = COMMON | {"text", "textvariable", "foreground", "background", "wraplength", "justify", "anchor", "font", "image", "compound",
                     "padding", "width", "relief", "underline", "state"}


class Entry(Widget):
    OPTS = COMMON | {"textvariable", "width", "state", "show", "justify", "font", "foreground", "background", "exportselection",
                     "validate", "validatecommand", "invalidcommand", "xscrollcommand"}


class Combobox(Entry):
    OPTS = Entry.OPTS | {"values", "height", "postcommand"}


class Button(Widget):
    OPTS = COMMON | {"text", "command", "state", "width", "image", "compound", "underline", "padding", "default", "textvariable"}


class Radiobutton(Widget):
    OPTS = COMMON | {"text", "value", "variable", "command", "state", "width", "image", "compound", "underline", "padding", "textvariable"}


class Checkbutton(Widget):
    OPTS = COMMON | {"text", "variable", "onvalue", "offvalue", "command", "state", "width", "image", "compound", "underline", "padding", "textvariable"}


class Progressbar(Widget):
    OPTS = COMMON | {"mode", "value", "maximum", "length", "orient", "variable", "phase"}
    running = False

    def start(self, interval=None):
        self.running = True

    def stop(self):
        self.running = False


class Scrollbar(Widget):
    OPTS = COMMON | {"orient", "command"}

    def set(self, first, last):
        pass


class Listbox(Widget):
    OPTS = {"height", "width", "exportselection", "yscrollcommand", "xscrollcommand", "selectmode", "font", "activestyle"}

    def insert(self, index, *items):
        pass

    def delete(self, first, last=None):
        pass

    def curselection(self):
        return ()

    def yview(self, *a):
        pass


class Text(Widget):
    """a Text with Tk's indices ('L.C', 'L.end', 'end', 'insert'), the insert mark and a selection (select()); the text is kept without
    the final newline Tk's Text always has"""
    OPTS = {"width", "height", "wrap", "font", "yscrollcommand", "xscrollcommand", "undo", "state", "padx", "pady", "background", "foreground"}

    def __init__(self, master=None, **kw):
        super().__init__(master, **kw)
        self.s, self.mod, self.cursor, self.sel = "", False, 0, ()                            # sel: the selection's (first, last) offsets

    def offset(self, index):
        index = str(index)
        if index in ("end", "insert"):
            return len(self.s) if index == "end" else self.cursor
        line, col = index.split(".")
        lines = self.s.split("\n")
        n = min(int(line), len(lines))
        return sum(len(x) + 1 for x in lines[:n - 1]) + (len(lines[n - 1]) if col == "end" else min(int(col), len(lines[n - 1])))

    def at(self, o):
        """the Tk index 'L.C' of offset o"""
        return f"{self.s.count(chr(10), 0, o) + 1}.{o - (self.s.rfind(chr(10), 0, o) + 1)}"

    def index(self, index):
        return self.at(self.offset(index))

    def insert(self, index, text):
        o = self.offset(index)
        self.s, self.mod, self.sel = self.s[:o] + text + self.s[o:], True, ()                # any change sets the modified flag
        self.cursor += len(text) if self.cursor >= o else 0                                  # the insert mark has right gravity

    def delete(self, first, last=None):
        a = self.offset(first)
        b = self.offset(last) if last is not None else a + 1
        self.s, self.mod, self.sel = self.s[:a] + self.s[b:], True, ()
        self.cursor = a if a < self.cursor <= b else self.cursor - (b - a) if self.cursor > b else self.cursor

    def get(self, first, last=None):
        full = self.s + "\n"                                                                  # Tk's Text always ends in a newline
        return full[self.offset(first):len(full) if str(last) == "end" else self.offset(last) if last is not None else self.offset(first) + 1]

    def mark_set(self, name, index):
        assert name == "insert"
        self.cursor = self.offset(index)

    def tag_ranges(self, tag):
        assert tag == "sel"
        return tuple(self.at(o) for o in self.sel)

    def select(self, first, last):
        """what a person dragging over the text does: the selection tag on first..last"""
        self.sel = (self.offset(first), self.offset(last))

    def edit_modified(self, flag=None):
        if flag is None:
            return self.mod
        self.mod = flag

    def yview(self, *a):
        pass

    def type(self, text):
        """what a person typing does: the text changes, then the <<Modified>> event is delivered"""
        self.insert("end", text)
        self.bound["<<Modified>>"](None)


class Toplevel(Widget):
    OPTS = frozenset()
    visible = True                                                                            # a Tk toplevel shows at once

    def title(self, text):
        self.name = text

    def transient(self, master):
        pass

    def withdraw(self):
        self.visible = False

    def deiconify(self):
        self.visible = True

    def lift(self):
        pass

    def protocol(self, name, func):
        self.bound[name] = func

    def geometry(self, spec):
        self.spec = spec


class Root(Widget):
    def __init__(self, screen):
        super().__init__()
        self.screen, self.after_calls, self.exit = screen, [], False

    def winfo_fpixels(self, spec):
        return 100.0

    def winfo_screenwidth(self):
        return self.screen[0]

    def winfo_screenheight(self):
        return self.screen[1]

    def title(self, text):
        self.name = text

    def geometry(self, spec):
        pass

    def winfo_width(self):
        return 1590

    def winfo_height(self):
        return 960

    def after(self, ms, func=None, *args):
        self.after_calls.append((ms, func))
        return len(self.after_calls)

    def after_cancel(self, ident):
        pass

    def destroy(self):
        self.exit = True


class Canvas(FigureCanvasAgg):
    """FigureCanvasTkAgg without Tk: an Agg canvas whose widget has a settable size (pixels)"""

    def __init__(self, figure, master=None):
        super().__init__(figure)
        self.widget, self.size, self.idle = Widget(master), (1, 1), 0
        self.widget.winfo_width, self.widget.winfo_height = (lambda: self.size[0]), (lambda: self.size[1])

    def get_tk_widget(self):
        return self.widget

    def draw_idle(self, *a, **k):
        self.idle += 1


class Toolbar(NavigationToolbar2):
    """the real toolbar logic (mode, history, messages) without the Tk widgets"""

    def __init__(self, canvas, window):
        super().__init__(canvas)

    def set_message(self, s):
        pass


@pytest.fixture
def tk(monkeypatch):
    """make(screen=(1728, 1117), selftest=None, canvas=(1000, 1250), experimental=True) -> (app, root): the application on the stand-in
    for Tk; experimental: the window of vortfoil_exp (the 3D wing this file tests), else vortfoil's (2D only)"""
    ttk, fd, tkm, be = (types.ModuleType(n) for n in ("tkinter.ttk", "tkinter.filedialog", "tkinter", "matplotlib.backends.backend_tkagg"))
    for c in (Frame, LabelFrame, Label, Entry, Combobox, Button, Radiobutton, Checkbutton, Progressbar, Scrollbar):
        setattr(ttk, c.__name__, c)
    tkm.StringVar = tkm.BooleanVar = Var
    for dialog in ("askdirectory", "asksaveasfilename", "askopenfilename"):         # a dialog nobody expects would block a real Tk
        setattr(fd, dialog, lambda **kw: pytest.fail("a file dialog opened"))
    tkm.Listbox, tkm.Text, tkm.Toplevel, tkm.ttk, tkm.filedialog = Listbox, Text, Toplevel, ttk, fd
    tkm.mainloop, tkm.Misc = (lambda n=0: None), type("Misc", (), {"mainloop": lambda self, n=0: None})   # matplotlib looks for them
    be.FigureCanvasTkAgg, be.NavigationToolbar2Tk = Canvas, Toolbar
    for name, mod in (("tkinter", tkm), ("tkinter.ttk", ttk), ("tkinter.filedialog", fd), ("matplotlib.backends.backend_tkagg", be)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setitem(sys.modules, "PIL.ImageGrab", None)                   # no screen capture, whatever the window does
    monkeypatch.setitem(A.App.TASKS, "index", lambda: [])                      # no network
    monkeypatch.setitem(A.DETAIL, "normal (0.005 c)", 0.05)                    # a coarse field: the 2D run takes seconds

    def make(screen=(1728, 1117), selftest=None, canvas=(1000, 1250), experimental=True):
        root = Root(screen)
        app = A.App(root, selftest=selftest, experimental=experimental)
        app.can.size = canvas
        return app, root
    make.fd = fd
    return make


def settle(app, done, timeout=240):
    """the event loop's polling of the worker queue, until done() (the status line in the message when it takes too long)"""
    t0 = time.time()
    while not done():
        assert time.time() - t0 < timeout, app.status.kw["text"]
        app.poll()
        time.sleep(0.02)


def texts(top):
    """{text: widget} of everything with a text in the widget tree under `top`"""
    return {w.kw["text"]: w for w in top.walk() if "text" in w.kw}


FLOW_2D = {"Incidence (deg)", "Reynolds number", "Field detail", "Show", "XFOIL state", "RANS state",
           "Trained on: XFOIL state -10 to 18 deg, Re 5e4 to 1e7; RANS state -5 to 15 deg, Re 2e6 to 6e6. Outside, drawn faded."}
REXT = "Re-extended field decoder (experimental; NACA sections; field Re 1e6 to 1e7)"
CREDIT = "UIUC coordinates: M. S. Selig, UIUC Applied Aerodynamics Group, downloaded from its site."


def test_window_builds_in_both_layouts(tk):
    """The window is the released one plus a '3D wing' box at the right end of the Section box's name row. Off, the Flow box is the
    released 2D one in the released places; ticked, the same cell shows the wing rows and the Wing geometry window holds the table
    (its words explain what a station is). The wide and the upright screen both build (the upright one in two columns)."""
    for screen in ((1728, 1117), (1080, 1920)):
        app, root = tk(screen)
        tall, t = screen[1] > screen[0], texts(root)
        # 3D off: the released window
        assert app.mode() == "section" and app.f2.mapped and not app.f3.mapped and app.f2.master is app.f3.master is app.flow
        assert app.f2.cell == app.f3.cell == dict(row=0, column=0, sticky="ew") and t["Flow"] is app.flow
        assert set(texts(app.f2)) == FLOW_2D | ({REXT} if app.rext_path else set())
        t2 = texts(app.f2)                                                    # the state boxes sit in the free third column of the first rows: no new row
        assert t2["Show"].cell == dict(row=0, column=2, sticky="w", padx=(10, 0)) and isinstance(t2["XFOIL state"], Checkbutton)
        assert t2["XFOIL state"].cell == dict(row=1, column=2, sticky="w", padx=(10, 0)) and t2["RANS state"].cell == dict(row=2, column=2, sticky="w", padx=(10, 0))
        assert t2["XFOIL state"].kw["variable"] is app.show_x and t2["RANS state"].kw["variable"] is app.show_r and app.states() == ("xfoil", "rans")
        assert app.show_x.get() is True and app.show_r.get() is True and t2["XFOIL state"].master is app.f2 is t2["Show"].master
        assert isinstance(t["3D wing"], Checkbutton) and t["3D wing"].master is t["Section"] is app.pname.master and app.chk is t["3D wing"]
        assert app.chk.kw["variable"] is app.wing and app.chk.kw["command"] == app.show and app.wing.get() is False and app.chk["state"] == "normal"
        assert app.lst.kw["height"] == (9 if tall else 14) and app.status.kw["wraplength"] == t[CREDIT].kw["wraplength"] == app.pname.kw["wraplength"] == 340
        act = t["Run"].master
        assert act is t["Save figure ..."].master is t["Export data (CSV) ..."].master and t["Save figure ..."]["state"] == "disabled"
        cells = {w: w.cell for w in (t["Airfoil"], t["Section"], app.flow, act, app.bar, app.status, t[CREDIT])}
        if tall:                                                              # two columns above the figure
            assert cells[t["Airfoil"]] == dict(row=0, column=0, rowspan=6, sticky="nsew", padx=(0, 10))
            for i, w in enumerate((t["Section"], app.flow, act, app.bar, app.status, t[CREDIT])):
                assert cells[w] == dict(row=i, column=1, sticky="new", pady=(0 if i == 0 else 6, 0))
        else:                                                                 # one column left of the figure
            for i, w in enumerate((t["Airfoil"], t["Section"], app.flow, act, app.bar, app.status)):
                assert cells[w] == dict(row=i, column=0, sticky="ew", pady=(0 if i == 0 else 8, 0))
            assert cells[t[CREDIT]] == dict(row=6, column=0, sticky="sw")
        assert app.fig.texts[0].get_text() == A.HINTS["section"] == "Choose an airfoil, set the flow and press Run"
        # 3D on: the wing rows in the Flow box's cell, the table in its own window
        app.wing.set(True); app.show()
        assert app.mode() == "wing" and app.f3.mapped and not app.f2.mapped and app.flow.cell == cells[app.flow]
        assert set(texts(app.f3)) == {"Planform", "Geometry ...", "Speed (m/s)", "Incidence (deg)", "Section model", "XFOIL state", "RANS state",
                                      "Surface", "Spanwise", "Polar", "Open cut in 2D", "3D view"}
        assert set(texts(app.stw)) == {A.GEOMETRY_TEXT, A.SECTION_HINT, "Airfoil box's section on:", "Selected row(s)", "All rows",
                                       "Scale x", "Strips", "Load ...", "Save ...", "Close"} and app.stw.name == "Wing geometry"
        assert app.lst.kw["height"] == 9                                       # 3D: the list is as short as on an upright screen (room for the gray line)
        assert A.GEOMETRY_TEXT.startswith("Each row is a spanwise station: the wing's shape at one position along the half-span, root (y = 0) first. "
                                          "Between rows the chord, twist, sweep, dihedral and section change linearly.")
        assert all(c.bound.get("<MouseWheel>") and c.bound["<MouseWheel>"](None) == "break" for c in app.f3.walk() if isinstance(c, Combobox))
        box = {w.kw["textvariable"]: w for w in app.f3.walk() if isinstance(w, Combobox)}
        assert box[app.planform].kw["values"] == list(WG.PRESETS) and "Custom" not in WG.PRESETS and box[app.planform].kw["state"] == "readonly"
        assert box[app.surface].kw["values"] == WV.SURFACE_QUANTITIES["xfoil"] and box[app.span].kw["values"] == WV.SPAN_QUANTITIES
        assert box[app.polar].kw["values"] == WV.POLAR_PLOTS and box[app.surface] is app.surface_box and len(box) == 4
        assert [w.kw["value"] for w in app.f3.walk() if isinstance(w, Radiobutton)] == ["xfoil", "sst"]
        assert [e.kw["textvariable"] for e in app.f3.walk() if type(e) is Entry] == [app.speed, app.walpha] and app.walpha is not app.alpha
        assert app.speed is not app.re and app.stations.kw["width"] == 56 and app.stw.spec == ("+0-0" if tall else "-0+0") and not app.stw.visible
        lines = app.stations.get("1.0", "end").splitlines()
        assert lines[0] == A.STATIONS_HEADER and len(lines) == 3 and all(ln.endswith("current") for ln in lines[1:])
        assert (app.planform.get(), app.speed.get(), app.walpha.get(), app.wstate.get(), app.strips.get(), app.scale.get()) == (
            FULL, "45", "6", "xfoil", "40", "1")
        assert app.cur.get() == "current = (no airfoil loaded yet)"
        assert app.fig.texts[0].get_text() == A.HINTS["wing"] and app.save_btn["state"] == app.csv_btn["state"] == app.cut_btn["state"] == "disabled"
        assert app.status.kw["text"] == f"{FULL} with the Airfoil box's section (none loaded yet) on every station; load another airfoil to change it."
        app.open_stations()                                                   # modeless, hidden by Close, Escape and the window's [x]
        assert app.stw.visible
        for hide in (texts(app.stw)["Close"].kw["command"], lambda: app.stw.bound["<Escape>"](None), app.stw.bound["WM_DELETE_WINDOW"]):
            app.open_stations(); hide()
            assert not app.stw.visible
        app.open_stations()
        app.wing.set(False); app.show()                                       # unticking: the released window again, the table hidden
        assert app.f2.mapped and not app.f3.mapped and not app.stw.visible and app.fig.texts[0].get_text() == A.HINTS["section"]
        assert app.lst.kw["height"] == (9 if tall else 14)                     # the released list again
        assert app.cut_btn["state"] == "disabled" and app.f2.cell == dict(row=0, column=0, sticky="ew")


def test_planform_fill_and_edit(tk, tmp_path):
    """A preset fills the stations, speed and strips: a one-section preset with the Airfoil box's section ('current') on every station,
    the Kinked and Clark Y presets with their own sections, each with a status text that says which airfoil the wing has; typing makes
    the wing Custom; filling is not typing; a definition from a file keeps its sections."""
    app, root = tk()
    status = lambda: app.status.kw["text"]
    app.planform.set(FULL); app.pick_planform()
    assert status() == f"{FULL} with the Airfoil box's section (none loaded yet) on every station; load another airfoil to change it. Press Run."
    settle(app, lambda: app.section is not None)
    for name, d in WG.PRESETS.items():
        one = len({s["section"] for s in d["stations"]}) == 1 and WG.is_naca(d["stations"][0]["section"])
        app.planform.set(name); app.pick_planform()
        assert one == (name not in (KINKED, CLARK))
        want = [dict(s, section="current") for s in d["stations"]] if one else d["stations"]
        assert app.stations.get("1.0", "end").strip() == A.format_stations(want) and app.speed.get() == f"{d['V']:g}" and app.wname == name
        assert app.strips.get() == "40" and app.scale.get() == "1" and app.wdesc == d["description"] and WG.PRESETS[name] is d
        assert not any(s["section"] == "current" for s in d["stations"])                  # the preset itself is left alone
        assert status() == {True: f"{name} with the Airfoil box's NACA 2412 on every station; load another airfoil to change it. Press Run.",
                            False: {KINKED: f"{KINKED}: NACA 4415 to NACA 2412 (its own sections) Press Run.",
                                    CLARK: f"{CLARK}: clarky (its own sections) Press Run."}.get(name)}[one]
        assert app.stations.edit_modified() is False
        app.stations.bound["<<Modified>>"](None)                              # the event of the fill, delivered late
        assert app.planform.get() == name and app.wname == name
    app.scale.set("3")
    app.stations.type("# a note\n")
    assert app.planform.get() == "Custom" and app.wname == "Custom wing" and app.wdesc == "" and app.stations.edit_modified() is False
    app.planform.set("Swept 30 deg"); app.pick_planform()
    assert app.scale.get() == "1" and app.wname == "Swept 30 deg" and app.planform.get() == "Swept 30 deg"
    WG.save_wing(WG.PRESETS[FULL], tmp_path / "full.json")                    # a file's definition: its own sections, the planform Custom
    d = WG.load_wing(str(tmp_path / "full.json"))
    assert app.fill_wing(d) is False and app.planform.get() == "Custom" and app.wname == FULL
    assert [s["section"] for s in A.parse_stations(app.stations.get("1.0", "end"))] == ["2412", "2412"]


def test_the_status_says_which_airfoil_the_wing_has(tk):
    """Ticking 3D, choosing a planform, loading a wing file and loading another airfoil all say it: a one-section planform takes the
    Airfoil box's section; a wing with its own sections says so; a table that mixes them says which is which."""
    app, root = tk()
    settle(app, lambda: app.section is not None)
    status = lambda: app.status.kw["text"]
    assert status() == "NACA 2412 loaded. Set the flow and press Run."                         # 2D: the released text
    app.wing.set(True); app.show()
    assert status() == f"{FULL} with the Airfoil box's NACA 2412 on every station; load another airfoil to change it."
    app.choose("naca", "4412"); settle(app, lambda: app.section[1] == "NACA 4412")
    assert status() == f"NACA 4412 loaded. {FULL} with the Airfoil box's NACA 4412 on every station; load another airfoil to change it."
    app.planform.set(KINKED); app.pick_planform()
    assert status() == f"{KINKED}: NACA 4415 to NACA 2412 (its own sections) Press Run."
    app.choose("naca", "23012"); settle(app, lambda: app.section[1] == "NACA 23012")
    assert status() == f"NACA 23012 loaded. {KINKED}: NACA 4415 to NACA 2412 (its own sections)"      # the airfoil box does not touch this wing
    app.stations.delete("1.0", "end"); app.stations.insert("1.0", SMALL)
    app.stations.bound["<<Modified>>"](None)
    assert app.wname == "Custom wing" and app.sections_note() == (
        "Custom wing: NACA 23012 to NACA 4412 (its own sections; 'current' is the Airfoil box's NACA 23012)")
    app.stations.delete("1.0", "end"); app.stations.insert("1.0", "0 1 0 0 0 2412\n1 1 0 0 0 4412\n2 1 0 0 0 uiuc:clarky\n3 1 0 0 0 0012\n4 1 0 0 0 /x/e387.dat")
    assert app.sections_note() == "Custom wing: 5 sections (its own sections)"
    app.stations.delete("1.0", "end"); app.stations.insert("1.0", "0 1 0 0")
    assert app.sections_note().startswith("The geometry table has a mistake: stations, line 1: expected y, chord, twist, x, z and a section")
    app.stations.delete("1.0", "end"); app.stations.insert("1.0", "# nothing\n")
    assert app.sections_note() == "The geometry table is empty."
    said = status()
    app.wing.set(False); app.show()
    assert status() == said                                                                   # unticking leaves the status alone


def small_wing(app, **over):
    """the tests' wing in the 3D mode: root 'current' (the Airfoil box's section), tip NACA 4412, 8 strips, 30 m/s, 5 deg"""
    app.wing.set(True)
    app.show()
    app.fill_wing(dict(name="Small test wing", stations=A.parse_stations(SMALL), V=30.0, n=8, **over))
    app.walpha.set("5")


def mouse(app, kind, eta, button=1):
    """a mouse event at span position eta in the span panel (in display pixels, as the canvas delivers them)"""
    ax = app.view.ax_span
    x, y = ax.transData.transform((eta, float(np.mean(ax.get_ylim()))))
    app.fig.canvas.callbacks.process(kind, MouseEvent(kind, app.fig.canvas, x, y, button=button))


def test_wing_run_display_cut_and_modes(tk, monkeypatch, tmp_path):
    """The 3D mode end to end on the real models (XFOIL state, a small wing): Run reads Cancel and the 3D box is off limits during the
    run, the figure, a display option, the cut by mouse, the modes keeping their results (and the turn and the cut) without
    recomputing, Open cut in 2D, the CSV parts dialog and Save figure."""
    app, root = tk()
    settle(app, lambda: app.section is not None)
    assert app.code == "2412" and app.cur.get() == "current = NACA 2412 (the Airfoil box's section)"
    small_wing(app)
    app.open_cut()                                                             # no wing result yet: ignored
    assert app.mode() == "wing" and app.section[1] == "NACA 2412"
    app.run_wing()
    assert (app.run_btn["text"], app.run_btn["state"], app.run_btn.kw["command"]) == ("Cancel", "normal", app.cancel_run)
    assert app.chk["state"] == "disabled" and app.bar.running
    sent = []
    with monkeypatch.context() as m:                                           # Return in an entry while it runs starts nothing
        m.setattr(app, "submit", lambda *a: sent.append(a))
        app.run(); app.run_wing()
    assert not sent and app.status.kw["text"].startswith("Computing Small test wing (XFOIL state) at 5 deg, 30 m/s ... (the first run also loads")
    settle(app, lambda: app.wres is not None)
    res, c = app.wres, app.wres["coef"]
    assert res["state"] == "xfoil" and 0.4 < c["CL"] < 0.7 and res["defn"]["n"] == 8
    assert [t["section"] for t in res["defn"]["stations"]] == ["2412", "4412"]            # 'current' went in as the designation
    text = app.status.kw["text"]
    assert text.startswith(f"Small test wing, 5 deg, 30 m/s (XFOIL state): C_L {c['CL']:.3f}, C_D {c['CD']:.4f}, C_M {c['CM']:.3f}, done in ")
    assert text.endswith("; no range warnings.") and "Drag" not in text
    assert (app.run_btn["text"], app.run_btn["state"], app.run_btn.kw["command"]) == ("Run", "normal", app.run)
    assert app.chk["state"] == "normal" and not app.bar.running and app.bar["value"] == 0
    assert all(app.cut_btn["state"] == b["state"] == "normal" for b in (app.save_btn, app.csv_btn)) and app.view is not None
    assert app.layout == "portrait" and len(app.fig.axes) >= 5 and np.allclose(app.fig.get_size_inches(), (10, 12.5))
    assert app.surface_box.kw["values"] == WV.SURFACE_QUANTITIES["xfoil"]

    app.surface.set("H"); app.wing_option("surface", "H")                      # a display choice: redrawn, re-frozen for the zoom
    ax = app.view.ax_span
    assert app.view.surface == "H" and app.wopts["surface"] == "H" and np.allclose(app.fig.get_size_inches(), (10, 12.5))
    f0 = ax.title.get_fontsize()
    app.fig.set_size_inches(5, 6.25, forward=False); app.on_resize()
    assert abs(ax.title.get_fontsize() - f0 / 2) < 1e-9
    app.fig.set_size_inches(10, 12.5, forward=False); app.on_resize()

    idle, eta = app.fig.canvas.idle, res["strips"]["eta"]                      # the cut: dragged in the span panel (no entry: the figure prints 2y/b)
    mouse(app, "button_press_event", 0.8); mouse(app, "motion_notify_event", 0.2)
    k = int(np.argmin(np.abs(eta - 0.2)))
    assert app.view.cut_index == k and app.fig.canvas.idle > idle
    mouse(app, "button_release_event", 0.2); mouse(app, "motion_notify_event", -0.9)
    assert app.view.cut_index == k                                             # released: no more dragging
    app.view.ax3d.view_init(elev=11, azim=-33)                                 # the user turns the wing: the view is its elevation and azimuth
    assert app.view.pose == (11, -33)
    app.surface.set("theta/c"); app.wing_option("surface", "theta/c")          # a display choice keeps the view and the cut
    assert app.view.cut_index == k and app.view.pose == (11, -33) and app.view.surface == "theta/c"

    with monkeypatch.context() as m:                                           # the modes keep their results: nothing is recomputed
        m.setattr(A.WG, "compute", lambda *a, **kw: pytest.fail("recomputed"))
        app.wing.set(False); app.show()
        assert app.view is None and app.f2.mapped and not app.f3.mapped and not app.stw.visible
        assert app.save_btn["state"] == app.csv_btn["state"] == "disabled" and app.fig.texts[0].get_text() == A.HINTS["section"]
        app.wing.set(True); app.show()
    assert app.f3.mapped and not app.f2.mapped and app.save_btn["state"] == app.csv_btn["state"] == "normal"
    assert app.view.cut_index == k and app.view.pose == (11, -33) and app.view.surface == "theta/c"
    app.can.size = (1500, 800); app.on_configure(None)                         # a resize that flips the layout redraws in it, same cut and view
    app.root.after_calls[-1][1]()
    assert app.layout in ("landscape", "square") and app.view.cut_index == k and app.view.pose == (11, -33)

    with monkeypatch.context() as m:                                           # Open cut in 2D is ignored while a wing runs
        m.setattr(app, "submit", lambda *a: pytest.fail("started"))
        app.lock(True, wing=True); app.open_cut(); app.idle()
    assert app.mode() == "wing" and app.section[1] == "NACA 2412"
    calls = []                                                                 # Open cut in 2D: the box unticks, the 2D case starts
    with monkeypatch.context() as m:
        m.setattr(app, "submit", lambda task, *a: calls.append((task, a)))
        app.open_cut()
    st, label = res["strips"], res["loops"]["label"][res["loops"]["index"][k]]
    name = f"{label[label.index('blend'):]}, 2y/b {eta[k]:.2f}"                # short: the 2D figure's header must stay in its cell
    assert label.startswith("NACA 2412 to NACA 4412 blend ") and len(name) <= 22
    assert app.mode() == "section" and app.wing.get() is False and app.f2.mapped and not app.f3.mapped and not app.stw.visible
    assert app.section[1] == name and app.pname.kw["text"].startswith(name) and app.code is None   # a blend: no designation
    assert app.cutlabel == label and app.cutof == " From the wing Small test wing."               # the wing's name goes to the status line
    assert app.cur.get() == f"current = {name} (the Airfoil box's section)"
    assert np.isclose(float(app.alpha.get()), st["alpha_eff_deg"][k], rtol=1e-3) and np.isclose(float(app.re.get()), st["Re"][k], rtol=1e-3)
    (task, args), = calls
    lp = res["loops"]
    assert task == "run" and np.array_equal(args[0], lp["coords"][lp["index"][k]]) and args[4:] == (name, "naca", None) and args[3] == 0.05   # a 2412/4412 blend: NACA family
    assert np.isclose(args[1], st["alpha_eff_deg"][k], rtol=1e-3) and np.isclose(args[2], st["Re"][k], rtol=1e-3)
    assert app.run_btn["state"] == app.chk["state"] == "disabled"              # the 2D case runs (its submit was stubbed): finish it
    app.idle()

    app.wing.set(True); app.show()                                             # the CSV parts: a dialog, then the folder, in the worker
    app.export()
    dlg = [w for w in root.walk() if isinstance(w, Toplevel)][-1]
    ticks = [w for w in dlg.walk() if isinstance(w, Checkbutton)]
    buttons = {w.kw["text"]: w for w in dlg.walk() if isinstance(w, Button)}
    parts = [w.kw["text"].split(":")[0] for w in ticks]
    assert parts == list(WV.CSV_PARTS) and all(w.kw["variable"].get() for w in ticks) and set(buttons) == {"All", "None", "Export ...", "Cancel"}
    assert [w.kw["text"] for w in ticks] == [f"{sfx}: {txt}" for sfx, txt in WV.CSV_PARTS.values()]
    buttons["None"].kw["command"]()
    assert not any(w.kw["variable"].get() for w in ticks)
    buttons["Export ..."].kw["command"]()
    assert dlg.destroyed and app.status.kw["text"] == "No part ticked: nothing written."
    app.export()
    dlg = [w for w in root.walk() if isinstance(w, Toplevel)][-1]
    ticks = [w for w in dlg.walk() if isinstance(w, Checkbutton)]
    {w.kw["text"]: w for w in dlg.walk() if isinstance(w, Button)}["None"].kw["command"]()
    for w in ticks:
        w.kw["variable"].set(w.kw["text"].split(":")[0] in ("summary", "cut"))
    asked = []
    monkeypatch.setattr(tk.fd, "askdirectory", lambda **kw: asked.append(kw) or str(tmp_path / "out"))
    {w.kw["text"]: w for w in dlg.walk() if isinstance(w, Button)}["Export ..."].kw["command"]()
    assert app.csv_btn["state"] == "disabled" and asked and app.status.kw["text"].startswith("Writing the CSV files to")
    settle(app, lambda: app.status.kw["text"].startswith("Wrote"))
    stem = A.wing_base(res)
    assert sorted(os.listdir(tmp_path / "out")) == sorted(f"{stem}_{s}" for s in ("summary.csv", "cut.csv", "about.txt"))
    assert app.status.kw["text"] == f"Wrote 3 files to {tmp_path / 'out'}: summary, cut, about" and app.csv_btn["state"] == "normal"
    cut = np.genfromtxt(tmp_path / "out" / f"{stem}_cut.csv", delimiter=",", names=True, dtype=None, encoding=None)
    assert np.allclose(cut["eta"], eta[k], atol=1e-6)                          # the cut strip as dragged, not the default

    saved, sizes = [], []                                                      # Save figure: its own name, the design size, 400 dpi
    monkeypatch.setattr(tk.fd, "asksaveasfilename", lambda **kw: saved.append(kw) or str(tmp_path / "w.png"))
    monkeypatch.setattr(app.fig, "savefig", lambda path, dpi: sizes.append((path, dpi, tuple(app.fig.get_size_inches()))))
    before = tuple(app.fig.get_size_inches())
    app.save()
    assert saved[0]["initialfile"] == stem + ".png" == "vortfoil_wing_Small_test_wing_a5_V30_xfoil.png" and saved[0]["defaultextension"] == ".png"
    assert sizes == [(str(tmp_path / "w.png"), 400, VA.DESIGNS[app.layout])] and tuple(app.fig.get_size_inches()) == before


def test_wing_errors_reach_the_status_line(tk, monkeypatch, capsys):
    """Every way a run can fail is a sentence in the status line; nothing is submitted for a form that cannot be read, a
    worker-side error leaves Run and the 3D box usable and the last result alone, and a user's mistake prints no traceback."""
    app, root = tk()
    settle(app, lambda: app.section is not None)
    app.wing.set(True); app.show()
    status = lambda: app.status.kw["text"]
    usable = lambda: app.run_btn["text"] == "Run" and app.run_btn["state"] == app.chk["state"] == "normal" and not app.bar.running
    good = app.stations.get("1.0", "end")
    sent = []
    with monkeypatch.context() as m:
        m.setattr(app, "submit", lambda *a: sent.append(a))
        for text, speed, alpha, strips, message in [
                ("0 1 0 0 0", "20", "6", "40", "The wing cannot be computed: stations, line 1: expected y, chord, twist, x, z and a section"),
                (good, "fast", "6", "40", "The speed and the incidence must be numbers (e.g. 45 and 6)."),
                (good, "-3", "6", "40", "The speed must be positive and the incidence between -30 and 30 deg."),
                (good, "20", "31", "40", "The speed must be positive and the incidence between -30 and 30 deg."),
                (good, "20", "6", "3", "The wing cannot be computed: Strips: must be a whole number from 8 to 200 (got 3)"),
                (good, "20", "6", "nan", "The wing cannot be computed: Strips: must be a whole number from 8 to 200 (got nan)"),
                (good.replace("current", ""), "20", "6", "40", "The wing cannot be computed: stations, line 2: expected")]:
            app.stations.delete("1.0", "end"); app.stations.insert("1.0", text)
            app.speed.set(speed); app.walpha.set(alpha); app.strips.set(strips)
            app.run()
            assert status().startswith(message) and not sent and usable(), message
    app.speed.set("20"); app.walpha.set("6"); app.strips.set("40")
    for tweak, message in [("nosuchfile.dat", "The wing computation failed: section 'nosuchfile.dat': not a NACA"),
                           ("uiuc:e387", "The wing computation failed: station 1 section 'uiuc:e387': could not load it (offline)"),
                           ("current", "The wing computation failed: station 1 section: 'current' needs an airfoil")]:
        with monkeypatch.context() as m:
            m.setattr(A, "load_section", lambda kind, value: (_ for _ in ()).throw(OSError("offline")))
            if tweak == "current":
                m.setattr(app, "section", None)
            app.stations.delete("1.0", "end"); app.stations.insert("1.0", good.replace("current", tweak))
            app.run()
            settle(app, lambda: status().startswith("The wing computation failed"))
        assert status().startswith(message) and app.wres is None and usable()
    assert capsys.readouterr().err == ""                                       # ValueErrors are the user's to fix: no trace

    monkeypatch.setattr(A.WG, "compute", lambda *a, **k: 1 / 0)                # a failure that is not the user's: shown and traced
    app.stations.delete("1.0", "end"); app.stations.insert("1.0", good)
    app.run()
    settle(app, lambda: status().startswith("The wing computation failed"))
    assert status() == "The wing computation failed: division by zero" and "ZeroDivisionError" in capsys.readouterr().err and usable()
    for err, message in ((MemoryError(), "The wing computation failed: not enough memory (close other programs and press Run again)"),
                         (RuntimeError(), "The wing computation failed: RuntimeError")):     # an error with no message is named, not left blank
        monkeypatch.setattr(A.WG, "compute", lambda *a, err=err, **k: (_ for _ in ()).throw(err))
        app.run()
        settle(app, lambda: status() == message)
        assert usable()
    capsys.readouterr()
    app.jobs.put(("wing", (), None, None))                                     # a result that cannot be shown does not stop the polling
    app.poll()
    assert status().startswith("The result could not be shown:") and root.after_calls[-1][1] == app.poll and usable()
    capsys.readouterr()


def test_default_wing_through_the_app_is_the_regression_case(tk):
    """Tick 3D wing and press Run with nothing changed: the Full-scale wing with the Airfoil box's NACA 2412 on every station, 45 m/s,
    6 deg, XFOIL state, is the research regression case (C_L 0.6265 within 3e-3), with no warnings, in the wing's own status text."""
    app, root = tk()
    settle(app, lambda: app.section is not None)
    app.wing.set(True); app.show()
    app.run()
    settle(app, lambda: app.wres is not None)
    res = app.wres
    assert res["wing"]["name"] == f"{FULL}, NACA 2412" and res["state"] == "xfoil" and (res["V"], res["alpha"], res["defn"]["n"]) == (45.0, 6.0, 40)
    assert abs(res["coef"]["CL"] - 0.6265) < 3e-3 and res["warnings"] == []
    assert app.status.kw["text"].startswith(f"{FULL}, NACA 2412, 6 deg, 45 m/s (XFOIL state): C_L 0.626, ")
    assert app.status.kw["text"].endswith("; no range warnings.") and app.layout == "portrait" and abs(app.view.cut_eta - 0.5) < 0.05   # the cut starts at mid half-span


def test_run_wing_hands_the_worker_what_it_needs(tk, monkeypatch):
    """Run reads the forms on the Tk thread and hands the worker the validated definition (scale x applied; the section's name in the wing's
    name when every station is 'current'), speed, incidence, state, the Airfoil box's section as (coordinates, NACA designation, name) and the
    progress and cancel callbacks; the worker's progress only queues."""
    app, root = tk()
    settle(app, lambda: app.section is not None)
    app.wing.set(True); app.show()
    sent = []
    monkeypatch.setattr(app, "submit", lambda *a: sent.append(a))
    app.scale.set("2"); app.strips.set("24"); app.speed.set("12.5"); app.walpha.set("-3"); app.wstate.set("sst")
    app.run()
    (task, defn, speed, alpha, state, current, progress, cancel), = sent
    assert task == "wing" and (speed, alpha, state) == (12.5, -3.0, "sst") and defn["n"] == 24
    assert defn["name"] == f"{FULL}, NACA 2412, scale x 2" and [s["y"] for s in defn["stations"]] == [0.0, 10.0]
    assert [s["chord"] for s in defn["stations"]] == [3.0, 1.5] and [s["twist"] for s in defn["stations"]] == [0.0, -2.0]   # lengths scale, twist does not
    assert [s["section"] for s in defn["stations"]] == ["current", "current"]      # resolved in the worker
    assert current[1:] == ("2412", "NACA 2412") and np.array_equal(current[0], app.section[0]) and cancel == app.cancel.is_set and progress == app.progress
    progress(0.5, "tables 2 of 4")                                              # from the worker: queued, not shown
    assert app.jobs.get_nowait() == ("progress", (), (0.5, "tables 2 of 4"), None) and app.bar["value"] == 0
    app.idle(); app.scale.set("1"); app.strips.set("40")
    app.planform.set(KINKED); app.pick_planform(); app.run()                    # sections of its own: the wing keeps its name
    assert sent[1][1]["name"] == KINKED and [s["section"] for s in sent[1][1]["stations"]] == ["4415", "4415", "2412"]
    app.idle()
    app.section = (app.section[0], "clarky", None); app.code = None             # a section from a file: no designation
    app.planform.set(FULL); app.pick_planform(); app.run()
    assert sent[2][1]["name"] == f"{FULL}, clarky" and sent[2][5][1] is None and sent[2][5][0] is app.section[0]


def test_progress_and_cancel(tk, monkeypatch):
    """The RANS state's bar is determinate and follows compute's progress (queued from the worker); Run reads Cancel, and Cancel stops
    the run at the next tick and keeps the last result; the XFOIL state's bar is the running kind."""
    app, root = tk()
    settle(app, lambda: app.section is not None)
    app.wing.set(True); app.show()
    gate = threading.Event()

    def slow(defn, speed, alpha, state, sections, progress, cancel, labels=None):
        progress(0.25, "tables 1 of 4")
        gate.wait(30)
        if cancel():
            raise RuntimeError("cancelled")
        raise AssertionError("not cancelled")
    monkeypatch.setattr(A.WG, "compute", slow)
    app.wstate.set("sst")
    app.run_wing()
    assert app.bar["mode"] == "determinate" and not app.bar.running and app.run_btn["text"] == "Cancel" and app.run_btn["state"] == "normal"
    settle(app, lambda: "tables 1 of 4" in app.status.kw["text"])
    assert app.bar["value"] == 25
    assert app.status.kw["text"] == "Computing Full-scale wing (span 10 m), NACA 2412 (RANS state) at 6 deg, 45 m/s ... tables 1 of 4"
    app.run_btn.kw["command"]()                                                # the second click of a double click on Run: too early, ignored
    assert not app.cancel.is_set()
    app.started -= 1
    app.wres = last = dict(sentinel=1)
    app.run_btn.kw["command"](); gate.set()                                    # the button reads Cancel: its command is the flag
    settle(app, lambda: app.status.kw["text"].startswith("Cancelled"))
    assert app.status.kw["text"] == "Cancelled. The last wing result is kept." and app.wres is last
    assert app.run_btn["text"] == "Run" and app.run_btn["state"] == "normal" and app.chk["state"] == "normal" and app.bar["value"] == 0
    app.wres = None
    gate.clear()                                                               # the flag is cleared by the next run
    app.wstate.set("xfoil"); app.run_wing()
    assert app.bar["mode"] == "indeterminate" and app.bar.running and not app.cancel.is_set()
    app.started -= 1
    app.run_btn.kw["command"](); gate.set()
    settle(app, lambda: app.status.kw["text"].startswith("Cancelled"))
    assert not app.bar.running and app.run_btn["text"] == "Run" and app.status.kw["text"] == "Cancelled."     # no result to keep


def test_2d_keeps_working(tk, tmp_path, monkeypatch):
    """The 2D analysis in the window with the 3D box: during a run the box and Run are off; the figure, Save figure's name and the CSV
    files are the released ones; ticking shows the wing hint and unticking the 2D figure again without recomputing."""
    app, root = tk()
    settle(app, lambda: app.section is not None)
    app.run()
    assert app.run_btn["state"] == app.chk["state"] == "disabled" and app.run_btn["text"] == "Run"
    assert app.status.kw["text"].startswith("Computing NACA 2412 at 4 deg, Re 3e+06 ...")
    settle(app, lambda: app.last is not None)
    assert app.run_btn["state"] == app.chk["state"] == "normal" and len(app.fig.axes) == 12 and app.layout == "portrait"
    assert app.status.kw["text"].startswith("NACA 2412, 4 deg, Re 3e+06: done in ") and app.status.kw["text"].endswith("; inside both training domains.")
    assert app.save_btn["state"] == app.csv_btn["state"] == "normal" and app.cut_btn["state"] == "disabled"
    with monkeypatch.context() as m:
        m.setattr(A.V, "compute", lambda *a: pytest.fail("recomputed"))
        app.wing.set(True); app.show()
        assert app.fig.texts[0].get_text() == A.HINTS["wing"] and not app.fig.axes and app.save_btn["state"] == app.csv_btn["state"] == "disabled"
        app.wing.set(False); app.show()
        assert len(app.fig.axes) == 12 and app.save_btn["state"] == app.csv_btn["state"] == "normal"
    saved = []
    monkeypatch.setattr(tk.fd, "asksaveasfilename", lambda **kw: saved.append(kw) or "")
    app.save()
    assert saved[0]["initialfile"] == "vortfoil_NACA_2412_a4_Re3e+06.png"
    monkeypatch.setattr(tk.fd, "askdirectory", lambda **kw: str(tmp_path / "csv2d"))
    app.export()
    settle(app, lambda: app.status.kw["text"].startswith("Wrote"))
    assert set(os.listdir(tmp_path / "csv2d")) == {f"vortfoil_NACA_2412_a4_Re3e+06_{s}" for s in (
        "about.txt", "boundary_layer.csv", "boundary_layer_rans.csv", "coefficients.csv", "field.csv", "section.csv", "surface.csv", "wall.csv")}
    assert len(os.listdir(tmp_path / "csv2d")) == 8 and app.csv_btn["state"] == "normal"


def test_state_boxes_choose_the_states_of_the_2d_figure(tk, monkeypatch):
    """The XFOIL state and RANS state boxes (both ticked at first) choose the states the surface panels, the boundary layer and the
    coefficients show: ticking or unticking redraws the figure from the last result without recomputing, the last tick cannot be taken
    off (it says so), a new run keeps the choice, and with no result yet nothing is drawn."""
    app, root = tk()
    settle(app, lambda: app.section is not None)
    boxes = {w.kw["text"]: w for w in app.f2.walk() if isinstance(w, Checkbutton)}
    legend = lambda: [t.get_text() for t in app.fig.axes[8].get_legend().get_texts()]
    app.show_r.set(False); boxes["RANS state"].kw["command"]()                  # no result yet: the hint stays
    assert app.fig.texts[0].get_text() == A.HINTS["section"] and not app.fig.axes and app.states() == ("xfoil",)
    app.show_r.set(True); boxes["RANS state"].kw["command"]()
    app.run(); settle(app, lambda: app.last is not None)
    assert app.states() == ("xfoil", "rans") and len(app.fig.axes) == 12 and legend() == ["XFOIL state", "RANS state"]
    last = app.last
    with monkeypatch.context() as m:
        m.setattr(A.V, "compute", lambda *a: pytest.fail("recomputed"))
        app.show_r.set(False); boxes["RANS state"].kw["command"]()
        assert app.states() == ("xfoil",) and len(app.fig.axes) == 11 and legend() == ["XFOIL state"] and app.layout == "portrait"
        assert "RANS state" not in "".join(t.get_text() for t in app.fig.axes[-1].texts)
        app.show_r.set(True); app.show_x.set(False); boxes["XFOIL state"].kw["command"]()
        assert app.states() == ("rans",) and len(app.fig.axes) == 12 and legend() == ["RANS state"]
        assert "XFOIL" not in "".join(t.get_text() for t in app.fig.axes[-1].texts) and app.last is last
        app.show_r.set(False); boxes["RANS state"].kw["command"]()              # the last tick: ticked again, with a word
        assert app.show_r.get() is True and app.states() == ("rans",) and app.status.kw["text"] == "At least one state stays ticked."
        assert legend() == ["RANS state"]
        app.show_x.set(True); boxes["XFOIL state"].kw["command"]()
        assert app.states() == ("xfoil", "rans") and len(app.fig.axes) == 12
        app.wing.set(True); app.show()                                         # the 3D mode: the boxes are not its
        app.show_x.set(False); boxes["XFOIL state"].kw["command"]()
        assert app.mode() == "wing" and app.fig.texts[0].get_text() == A.HINTS["wing"] and not app.fig.axes
        app.wing.set(False); app.show()
        assert app.states() == ("rans",) and legend() == ["RANS state"]
    app.run(); settle(app, lambda: app.last is not last)                      # a new run keeps the choice
    assert app.states() == ("rans",) and legend() == ["RANS state"] and len(app.fig.axes) == 12


def test_a_state_outside_its_training_range_is_drawn_faded_and_the_status_says_so(tk):
    """17 deg is inside the surface model's training range and outside the field decoder's: the RANS state is drawn faded and the status
    line says it; 20 deg is outside both; a case inside both says nothing of it."""
    import view_airfoil as V
    app, root = tk()
    settle(app, lambda: app.section is not None)
    status = lambda: app.status.kw["text"]
    legend = lambda: [t.get_text() for t in app.fig.axes[8].get_legend().get_texts()]
    for alpha, said, labels in (("17", " The RANS state is outside its training range: drawn faded.", ["XFOIL state", "RANS state (outside training range)"]),
                                ("20", " The XFOIL state and the RANS state are outside their training range: drawn faded.",
                                 ["XFOIL state (outside training range)", "RANS state (outside training range)"]),
                                ("4", "", ["XFOIL state", "RANS state"])):
        app.alpha.set(alpha); app.last = None
        app.run(); settle(app, lambda: app.last is not None)
        assert legend() == labels and (said in status() if said else status().endswith("; inside both training domains."))
        if said:
            assert "drawn faded." in status() and (status().endswith(said) or "listed in the figure." in status())
            assert {ln.get_alpha() for ln in app.fig.axes[8].lines[3:5]} == {V.FADE}
    assert status().startswith("NACA 2412, 4 deg, Re 3e+06: done in ")


def test_selftest_runs_a_wing(tk, tmp_path, capsys, monkeypatch):
    """--selftest with --wing: the 3D mode is filled from the preset (the Airfoil box's NACA 2412 on every station) or from a wing file
    (its own sections, its name), with --V, --alpha and --state, run, and the figure, the canvas and every CSV part are saved; a wing
    that does not exist, or one that cannot run, ends it with the reason."""
    out = tmp_path / "st"
    app, root = tk(selftest=(str(out), "naca:2412", 6.0, 3e6, "rectangular, ar 6", 25.0, "xfoil"))
    settle(app, lambda: any(f == app.finish_selftest for _, f in root.after_calls))
    assert app.mode() == "wing" and app.f3.mapped and (app.speed.get(), app.walpha.get(), app.wstate.get()) == ("25", "6", "xfoil")
    assert app.planform.get() == "Rectangular, AR 6"
    app.finish_selftest()
    text = capsys.readouterr().out
    assert "SELFTEST OK Rectangular, AR 6, NACA 2412, 6 deg, 25 m/s (XFOIL state): C_L 0." in text
    assert "window capture skipped" in text and "layout portrait" in text
    assert {"app_canvas.png", "app_figure.png", "csv"} == set(os.listdir(out)) and app.exit_code == 0
    stem = "vortfoil_wing_Rectangular_AR_6_NACA_2412_a6_V25_xfoil"
    assert sorted(os.listdir(out / "csv")) == sorted([f"{stem}_{p}.csv" for p in WV.CSV_PARTS] + [f"{stem}_about.txt"])
    assert any(f == root.destroy for _, f in root.after_calls)
    WG.save_wing(dict(WG.PRESETS[FULL], name="Json wing", n=8), tmp_path / "json_wing.json")             # a wing file: --V, --alpha
    app, root = tk(selftest=(str(out), "naca:2412", 5.0, 3e6, str(tmp_path / "json_wing.json"), 33.0, "xfoil"))
    settle(app, lambda: any(f == app.finish_selftest for _, f in root.after_calls))
    assert (app.speed.get(), app.walpha.get(), app.strips.get(), app.planform.get(), app.wname) == ("33", "5", "8", "Custom", "Json wing")
    assert [s["section"] for s in A.parse_stations(app.stations.get("1.0", "end"))] == ["2412", "2412"]   # the file's sections, not 'current'
    app.finish_selftest()
    assert "SELFTEST OK Json wing, 5 deg, 33 m/s (XFOIL state): C_L 0." in capsys.readouterr().out and app.exit_code == 0
    seen = []                                                                                           # --state sst reaches the worker
    with monkeypatch.context() as m:
        m.setitem(A.App.TASKS, "wing", lambda *a: seen.append(a[1:4]) or (_ for _ in ()).throw(ValueError("stop here")))
        app, root = tk(selftest=(str(out), "naca:2412", 6.0, 3e6, "kinked", 55.0, "sst"))
        settle(app, lambda: app.exit_code == 1)
    assert seen == [(55.0, 6.0, "sst")]                                                                 # speed, incidence, state
    assert "SELFTEST FAILED: The wing computation failed: stop here" in capsys.readouterr().out
    app, root = tk(selftest=(str(out), "naca:2412", 6.0, 3e6, "no such wing", None, "xfoil"))            # a wing that does not exist
    settle(app, lambda: app.exit_code == 1)
    assert "SELFTEST FAILED: 'no such wing' is neither a preset nor a wing definition file; the presets are: Model wing" in capsys.readouterr().out
    WG.save_wing(dict(WG.PRESETS[FULL], stations=[dict(s, section="nosuch.dat") for s in WG.PRESETS[FULL]["stations"]]), tmp_path / "bad.json")
    app, root = tk(selftest=(str(out), "naca:2412", 6.0, 3e6, str(tmp_path / "bad.json"), None, "xfoil"))  # a wing file whose section is missing
    settle(app, lambda: app.exit_code == 1)
    assert "SELFTEST FAILED: The wing computation failed: section '" in capsys.readouterr().out


def test_uiuc_and_file_sections_run(tk, tmp_path, monkeypatch):
    """A station's 'uiuc:<name>' is fetched in the worker (its progress is queued), a file path is read, both reach the lifting line
    as coordinates (the figure's section label is the file's name); a non-NACA section in the Airfoil box goes in as coordinates
    for 'current' and names the wing. Here the network is a local file."""
    import predict_field as PF
    clark = tmp_path / "clarky.dat"
    np.savetxt(clark, PF.naca_coords("4412"), header="Clark Y stand-in", comments="")
    fetched = []
    monkeypatch.setattr(A, "fetch", lambda url, path: fetched.append((url, path)) or str(clark))
    app, root = tk()
    settle(app, lambda: app.section is not None)
    seen = []
    done_progress = app.done_progress
    monkeypatch.setattr(app, "done_progress", lambda args, res, err: seen.append(res[1]) or done_progress(args, res, err))
    small_wing(app)
    app.stations.delete("1.0", "end"); app.stations.insert("1.0", f"0 1 0 0 0 uiuc:clarky\n3 0.6 0 0 0 {clark}\n")
    app.run_wing()
    settle(app, lambda: app.wres is not None)
    assert [u for u, _ in fetched] == [A.UIUC + "coord/clarky.dat"] and "fetching the UIUC section clarky ..." in seen and seen[-1] == "done"
    assert all("clarky" in lb for lb in app.wres["loops"]["label"]) and set(app.wres["loops"]["family"]) == {None} and app.wres["coef"]["CL"] > 0.3
    assert app.status.kw["text"].startswith("Small test wing, 5 deg, 30 m/s (XFOIL state): C_L 0.")
    app.choose("file", str(clark))                                             # a section from a file in the Airfoil box: no designation
    settle(app, lambda: app.section[1] == "clarky")
    assert app.code is None and app.cur.get() == "current = clarky (the Airfoil box's section)"
    app.planform.set("Rectangular, AR 4"); app.pick_planform()
    assert app.status.kw["text"] == "Rectangular, AR 4 with the Airfoil box's clarky on every station; load another airfoil to change it. Press Run."
    app.strips.set("8"); app.wres = None
    app.run_wing()
    settle(app, lambda: app.wres is not None)
    assert app.wres["wing"]["name"] == "Rectangular, AR 4, clarky" and app.wres["defn"]["stations"][0]["section"] == "clarky"   # never the token 'current'
    assert set(app.wres["loops"]["label"]) == {"clarky"} and set(app.wres["loops"]["family"]) == {None}


def test_wing_files_load_and_save(tk, tmp_path, monkeypatch):
    """Save ... writes the form's wing (scaled, with its speed), a custom wing is named after its file, Load ... reads it back into the
    form (its own sections, the planform Custom); a file that is not a wing says so in the status line. The dialogs belong to the
    Wing geometry window."""
    app, root = tk()
    status = lambda: app.status.kw["text"]
    app.planform.set(KINKED); app.pick_planform()
    assert status() == f"{KINKED}: NACA 4415 to NACA 2412 (its own sections) Press Run."
    app.scale.set("2"); app.speed.set("33"); app.strips.set("24")
    paths, asked = iter([str(tmp_path / "scaled.json"), str(tmp_path / "mine.json"), ""]), []
    monkeypatch.setattr(tk.fd, "asksaveasfilename", lambda **kw: asked.append(kw) or next(paths))
    texts_ = texts(app.stw)
    texts_["Save ..."].kw["command"]()
    d = json.load(open(tmp_path / "scaled.json"))
    assert d["name"] == KINKED + ", scale x 2" and d["V"] == 33.0 and d["n"] == 24 and d["stations"][-1]["y"] == 10.0 and d["stations"][-1]["twist"] == -3.0
    assert status() == f"Saved the wing {d['name']} to {tmp_path / 'scaled.json'}"
    app.stations.type("# edited\n")
    app.save_wing()
    assert json.load(open(tmp_path / "mine.json"))["name"] == "mine"            # a custom wing takes its file's name
    app.save_wing()                                                             # a cancelled dialog writes nothing
    assert sorted(f for f in os.listdir(tmp_path) if f.endswith(".json")) == ["mine.json", "scaled.json"]
    assert len(asked) == 3 and all(kw["parent"] is app.stw for kw in asked) and asked[0]["initialfile"].endswith("scale_x_2.json")
    app.stations.delete("1.0", "end"); app.stations.insert("1.0", "0 1 0 0 0")
    app.save_wing()
    assert status().startswith("The wing cannot be saved: stations, line 1: expected")

    asked.clear()
    monkeypatch.setattr(tk.fd, "askopenfilename", lambda **kw: asked.append(kw) or str(tmp_path / "scaled.json"))
    texts_["Load ..."].kw["command"]()
    assert (app.planform.get(), app.wname, app.speed.get(), app.strips.get(), app.scale.get()) == ("Custom", d["name"], "33", "24", "1")
    assert A.parse_stations(app.stations.get("1.0", "end")) == [{k: s[k] for k in (*A.COLUMNS, "section")} for s in d["stations"]]
    assert [s["section"] for s in A.parse_stations(app.stations.get("1.0", "end"))] == ["4415", "4415", "2412"]   # a file keeps its sections
    assert status() == f"Loaded {tmp_path / 'scaled.json'}. {d['name']}: NACA 4415 to NACA 2412 (its own sections) Press Run."
    (tmp_path / "bad.json").write_text("{")
    monkeypatch.setattr(tk.fd, "askopenfilename", lambda **kw: asked.append(kw) or str(tmp_path / "bad.json"))
    app.load_wing()
    assert status().startswith("Could not load bad.json: bad.json: not valid JSON") and all(kw["parent"] is app.stw for kw in asked)


def test_sst_state_through_the_app(tk, tmp_path, monkeypatch):
    """The RANS state end to end (a coarse incidence grid and a scratch table cache): the determinate bar follows compute's
    progress, the Surface choices become the state's, a leftover XFOIL-only choice falls back to C_p, and the SST figure draws."""
    monkeypatch.setattr(WG, "TABLE_S", np.arange(-8.0, 16.001, 4.0))
    monkeypatch.setattr(WG, "CACHE", str(tmp_path / "tables"))
    monkeypatch.setattr(WG, "_MEM", {})
    app, root = tk()
    settle(app, lambda: app.section is not None)
    small_wing(app)
    app.surface.set("H"); app.wing_option("surface", "H")                      # an XFOIL-only quantity, chosen before any result
    app.wstate.set("sst")
    bars = []
    done_progress = app.done_progress
    monkeypatch.setattr(app, "done_progress", lambda args, res, err: done_progress(args, res, err) or bars.append(app.bar["value"]))
    app.run_wing()
    assert app.bar["mode"] == "determinate" and not app.bar.running
    settle(app, lambda: app.wres is not None)
    res = app.wres
    assert res["state"] == "sst" and app.surface_box.kw["values"] == ["C_p", "C_f"] and app.surface.get() == "C_p" and app.view.surface == "C_p"
    assert bars[-1] == 100 and all(b >= a for a, b in zip(bars, bars[1:])) and len(bars) > 5 and app.bar["value"] == 0
    lead = "RANS state outside the field decoder's Reynolds range (2e6-6e6) on 8 of 8 strips: the RANS-state numbers are not meaningful here; use the XFOIL state. "
    assert app.status.kw["text"].startswith(lead + "Small test wing, 5 deg, 30 m/s (RANS state): C_L 0.")      # Re 1.2e6 to 2e6: far outside, it says so first
    assert res["sst_range"]["below"] == 8 and app.status.kw["text"].endswith("done in " + app.status.kw["text"].split("done in ")[1])
    app.surface.set("C_f"); app.wing_option("surface", "C_f")
    assert app.view.surface == "C_f" and app.wopts["surface"] == "C_f" and np.allclose(app.fig.get_size_inches(), (10, 12.5))
    app.wstate.set("xfoil")                                                    # the radio is for the next run: the shown result keeps its state
    assert app.surface_box.kw["values"] == ["C_p", "C_f"]


@pytest.mark.parametrize("experimental", [True, False])
def test_selftest_runs_a_section_as_before(tk, tmp_path, capsys, experimental):
    """Without --wing the self-test is the 2D one, in both windows: its run, the canvas and figure files, the 2D CSV files,
    'SELFTEST OK Saved ...'."""
    out = tmp_path / "st2d"
    app, root = tk(selftest=(str(out), "naca:2412", 4.0, 3e6, None, None, "xfoil"), experimental=experimental)
    settle(app, lambda: any(f == app.finish_selftest for _, f in root.after_calls))
    assert app.mode() == "section" and app.re.get() == "3e+06" and app.alpha.get() == "4"
    app.finish_selftest()
    text = capsys.readouterr().out
    assert f"SELFTEST OK Saved {out / 'app_figure.png'}" in text and "layout portrait" in text and app.exit_code == 0
    assert {"app_canvas.png", "app_figure.png", "csv"} == set(os.listdir(out)) and len(os.listdir(out / "csv")) == 8


# ------------------------------------------------------------------------------------------------ the section at the cut, files, odd numbers
def test_open_cut_is_the_real_section_with_a_short_name(tk, monkeypatch):
    """Open cut in 2D hands the 2D case the strip's own section: a NACA strip as the designation with family 'naca', so its numbers are
    those of a direct 2D run of that NACA at the strip's incidence and Reynolds number, bit for bit. Its 2D name is short ('NACA 2212,
    2y/b 0.51': a long header squeezes the figure's panels), the wing's name goes to the status line, and the next wing run is named
    after the wing once, with the section's own name."""
    import predict_field as PF
    app, root = tk()
    settle(app, lambda: app.section is not None)
    app.choose("naca", "2212"); settle(app, lambda: app.section[1] == "NACA 2212")
    app.wing.set(True); app.show()
    app.strips.set("8"); app.walpha.set("5"); app.speed.set("40")
    app.run_wing()
    settle(app, lambda: app.wres is not None)
    res = app.wres
    assert res["wing"]["name"] == f"{FULL}, NACA 2212" and set(res["loops"]["label"]) == {"NACA 2212"} and set(res["loops"]["family"]) == {"naca"}
    k = app.view.cut_index
    mouse(app, "button_press_event", 0.6); mouse(app, "button_release_event", 0.6)
    k = app.view.cut_index
    app.open_cut()
    name = f"NACA 2212, 2y/b {res['strips']['eta'][k]:.2f}"
    assert app.section[1] == name and len(name) <= 22 and app.code == "2212" and app.cutlabel == "NACA 2212"
    assert app.cutof == f" From the wing {FULL}, NACA 2212."
    settle(app, lambda: app.last is not None)
    assert app.args[0] == name and app.status.kw["text"].startswith(f"{name}, ") and app.status.kw["text"].endswith(f" From the wing {FULL}, NACA 2212.")
    alpha, Re = float(f"{res['strips']['alpha_eff_deg'][k]:.4g}"), float(f"{res['strips']['Re'][k]:.4g}")
    assert app.args[1:] == (alpha, Re)
    direct = A.V.compute(PF.naca_coords("2212"), alpha, Re, 0.05, "naca", None)
    r, surf_w, out, meta = app.last[3:]
    for a, b in ((r.cl, direct[0].cl), (r.cd, direct[0].cd), (r.cm, direct[0].cm), (r.cp, direct[0].cp), (r.theta, direct[0].theta)):
        assert np.array_equal(a, b)
    assert surf_w == direct[1] and meta["warnings"] == direct[3]["warnings"]
    same = lambda a, b: np.array_equal(np.asarray(a), np.asarray(b), equal_nan=np.asarray(a).dtype.kind in "fc" and np.asarray(b).dtype.kind in "fc")
    for key, val in direct[2].items():
        assert same(out[key], val), key
    for key, val in direct[3]["wall"].items():
        assert same(meta["wall"][key], val), key
    app.wing.set(True); app.show()                                             # the next wing run: the section's name, not the cut's
    sent = []
    with monkeypatch.context() as m:
        m.setattr(app, "submit", lambda *a: sent.append(a))
        app.run_wing()
    assert sent[0][1]["name"] == f"{FULL}, NACA 2212" and sent[0][5][2] == "NACA 2212" and sent[0][5][1] == "2212"
    assert app.status.kw["text"].startswith(f"Computing {FULL}, NACA 2212 (XFOIL state)")
    app.idle()
    app.choose("naca", "4412"); settle(app, lambda: app.section[1] == "NACA 4412")     # another airfoil in the box is not a cut any more
    assert app.cutlabel is None and app.cutof == "" and app.sname() == "NACA 4412"


def test_a_2d_figure_after_a_wing_figure_is_a_fresh_one(tk):
    """The 2D figure drawn into the figure a wing figure was in (3D axes, colour bar, mouse handling, frozen layout) is the picture a
    fresh figure gives for the same case, pixel for pixel, in the layout the window chooses."""
    from matplotlib.figure import Figure
    app, root = tk()
    settle(app, lambda: app.section is not None)
    app.run(); settle(app, lambda: app.last is not None)
    app.wing.set(True); app.show()
    small_wing(app); app.run_wing(); settle(app, lambda: app.wres is not None)
    assert app.view is not None and any(a.name == "3d" for a in app.fig.axes)
    app.wing.set(False); app.show()                                            # back to the 2D figure in the same Figure
    assert app.view is None and not any(a.name == "3d" for a in app.fig.axes) and len(app.fig.axes) == 12
    assert app.fig.wing_view._cids == [] and "draw" not in vars(app.fig.canvas)                      # no wing handler left on the canvas
    app.fig.canvas.draw()
    shown = np.asarray(app.fig.canvas.buffer_rgba()).copy()
    fresh = Figure(figsize=VA.DESIGN, dpi=app.fig.dpi)
    FigureCanvasAgg(fresh)
    VA.figure(*app.last, fig=fresh, layout=app.layout)
    on_resize = VA.scale_with_window(fresh)
    fresh.set_size_inches(*app.fig.get_size_inches(), forward=False); on_resize()
    fresh.canvas.draw()
    assert shown.shape == np.asarray(fresh.canvas.buffer_rgba()).shape and np.array_equal(shown, np.asarray(fresh.canvas.buffer_rgba()))


def test_the_toolbars_history_does_not_outlive_the_figure(tk):
    """mplot3d pushes every turn onto the toolbar's history (Back, Home); it holds axes positions, so it is cleared when the figure is
    rebuilt and when the window is resized: Home cannot put the old layout back."""
    from matplotlib.backend_bases import ResizeEvent
    app, root = tk()
    settle(app, lambda: app.section is not None)
    app.wing.set(True); app.show()
    small_wing(app); app.run_wing(); settle(app, lambda: app.wres is not None)
    tb = app.can.toolbar
    assert tb._nav_stack() is None
    bb = app.view.ax3d.get_window_extent()
    for kind in ("button_press_event", "button_release_event"):
        app.fig.canvas.callbacks.process(kind, MouseEvent(kind, app.fig.canvas, bb.x0 + bb.width / 2, bb.y0 + bb.height / 2, button=1))
    assert tb._nav_stack() is not None                                         # the turn is in the history
    app.surface.set("H"); app.wing_option("surface", "H")
    assert tb._nav_stack() is None                                             # the rebuilt figure's is empty
    for kind in ("button_press_event", "button_release_event"):
        app.fig.canvas.callbacks.process(kind, MouseEvent(kind, app.fig.canvas, bb.x0 + bb.width / 2, bb.y0 + bb.height / 2, button=1))
    assert tb._nav_stack() is not None
    app.can.callbacks.process("resize_event", ResizeEvent("resize_event", app.can))
    assert tb._nav_stack() is None


def test_save_figure_errors_reach_the_status_line(tk, monkeypatch, tmp_path):
    """An unwritable place or an unknown file type is a sentence in the status line, and the figure keeps its size."""
    app, root = tk()
    settle(app, lambda: app.section is not None)
    app.wing.set(True); app.show()
    small_wing(app); app.run_wing(); settle(app, lambda: app.wres is not None)
    before = tuple(app.fig.get_size_inches())
    for path, word in (("/no/such/dir/x.png", "No such file or directory"), (str(tmp_path / "x.unknownext"), "not supported")):
        monkeypatch.setattr(tk.fd, "asksaveasfilename", lambda path=path, **kw: path)
        app.save()
        assert app.status.kw["text"].startswith(f"Could not save {path}: ") and word in app.status.kw["text"]
        assert tuple(app.fig.get_size_inches()) == before
    monkeypatch.setattr(tk.fd, "asksaveasfilename", lambda **kw: str(tmp_path / "ok.png"))
    app.save()
    assert app.status.kw["text"] == f"Saved {tmp_path / 'ok.png'}" and (tmp_path / "ok.png").stat().st_size > 10000


def test_saving_a_wing_does_not_leave_current_behind(tk, monkeypatch, tmp_path):
    """A one-section wing's file names the NACA section of the Airfoil box; for a section that has no designation 'current' stays, and
    the status line says what it means."""
    app, root = tk()
    settle(app, lambda: app.section is not None)
    app.choose("naca", "4412"); settle(app, lambda: app.section[1] == "NACA 4412")
    paths = iter([str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    monkeypatch.setattr(tk.fd, "asksaveasfilename", lambda **kw: next(paths))
    app.save_wing()
    d = json.load(open(tmp_path / "a.json"))
    assert [s["section"] for s in d["stations"]] == ["4412", "4412"] and app.status.kw["text"] == f"Saved the wing {FULL} to {tmp_path / 'a.json'}"
    assert [s["section"] for s in A.parse_stations(app.stations.get("1.0", "end"))] == ["current", "current"]     # the table itself is as it was
    app.section = (app.section[0], "clarky", None); app.code = None
    app.save_wing()
    assert [s["section"] for s in json.load(open(tmp_path / "b.json"))["stations"]] == ["current", "current"]
    assert app.status.kw["text"].endswith("; its 'current' sections mean the Airfoil box's (clarky) when it is loaded")


def test_odd_numbers_end_in_a_sentence_not_in_numbers(tk, capsys):
    """A speed past what a float holds, a wing far too small or a speed far too low for the surface model: the status line says what
    is wrong (and a result that comes out says so first), with no traceback for a mistake of the user's."""
    app, root = tk()
    settle(app, lambda: app.section is not None)
    app.wing.set(True); app.show()
    app.strips.set("8")
    status = lambda: app.status.kw["text"]
    app.speed.set("1e300"); app.run_wing()
    settle(app, lambda: status().startswith("The wing computation failed"))
    assert status() == "The wing computation failed: the speed or the size is out of range." and capsys.readouterr().err == ""
    assert app.run_btn["text"] == "Run" and app.chk["state"] == "normal"
    app.speed.set("45"); app.scale.set("1e-6"); app.run_wing()
    settle(app, lambda: app.wres is not None)
    assert status().startswith("Reynolds number outside the surface model's range (5e4-1e7) on 8 of 8 strips: the numbers are not meaningful here. ")
    app.wres = None
    app.scale.set("1"); app.speed.set("1e-300"); app.run_wing()
    settle(app, lambda: app.wres is not None or status().startswith("The wing computation failed"))
    assert status().startswith(("Reynolds number outside the surface model's range", "The result is not finite", "The wing computation failed"))


def test_current_is_the_designation_in_the_wing(tk, tmp_path, monkeypatch):
    """With a NACA section in the Airfoil box, a wing whose stations say 'current' is the same wing as the one whose stations carry
    the digits, in both states: the same numbers, a NACA section for the decoder (no non-NACA warning), the section named in the labels."""
    import predict_field as PF
    monkeypatch.setattr(WG, "TABLE_S", np.arange(-8.0, 16.001, 4.0))
    monkeypatch.setattr(WG, "CACHE", str(tmp_path / "tables"))
    monkeypatch.setattr(WG, "_MEM", {})
    coords = PF.naca_coords("4412")
    d_cur = A.wing_definition("0 1.5 0 0 0 current\n4 0.8 -1 0 0 current\n", n="8")
    d_typ = A.wing_definition("0 1.5 0 0 0 4412\n4 0.8 -1 0 0 NACA 4412\n", n="8")
    for state in ("xfoil", "sst"):
        a, _ = A.wing_case(d_cur, 50.0, 5.0, state, (coords, "4412", "NACA 4412"))
        b, _ = A.wing_case(d_typ, 50.0, 5.0, state, None)
        assert a["coef"] == b["coef"] and a["warnings"] == b["warnings"] and not any("not NACA" in w for w in a["warnings"]), state
        for key in a["strips"]:
            assert np.array_equal(a["strips"][key], b["strips"][key]), (state, key)
        assert set(a["loops"]["family"]) == {"naca"} and set(a["loops"]["label"]) == {"NACA 4412"} and a["loops"]["label"] == b["loops"]["label"]
        assert [s["section"] for s in a["defn"]["stations"]] == ["4412", "4412"] == [s["section"] for s in b["defn"]["stations"]]


def test_a_failing_selftest_write_ends_the_selftest(tk, tmp_path, capsys, monkeypatch):
    """A failure while the self-test writes its files must end it with the reason (and exit code 1), not leave the window open."""
    out = tmp_path / "st"
    app, root = tk(selftest=(str(out), "naca:2412", 6.0, 3e6, "rectangular, ar 6", 25.0, "xfoil"))
    settle(app, lambda: any(f == app.finish_selftest for _, f in root.after_calls))
    monkeypatch.setattr(A.WV, "export_csv", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    app.finish_selftest()
    assert "SELFTEST FAILED: disk full" in capsys.readouterr().out and app.exit_code == 1
    assert any(f == root.destroy for _, f in root.after_calls)


# ------------------------------------------------------------------------------------------------ the gray row of ranges; the Airfoil box's section on rows
SPEEDS = "Speeds that keep every strip inside the training Re: XFOIL state 1-102 m/s, RANS state 40-61.2 m/s. "     # the 40 strips' chords
INCIDENCE = "Strip incidence: XFOIL -10 to 18 deg, RANS -5 to 15 deg."
FAMILY = "RANS state: trained on NACA 4/5-digit sections only; this wing's other sections give indicative values; prefer the XFOIL state."


def test_the_gray_row_of_ranges(tk, monkeypatch):
    """The 3D Flow box has one gray row of wrapping text after the Section model row. It says what the wing's inputs must respect
    (wing_nll.input_ranges: no model runs) and follows every input at once: the planform, the geometry table, Scale x, the speed, the
    Airfoil box's section, the chosen state and Strips (the speeds are those of the strips Run computes). In range, below range, no speed
    fits, the UIUC family note, and a table, Scale x or Strips that cannot be read (a note, never an exception, also for a size no float
    holds: the section load that calls it still completes)."""
    app, root = tk()
    settle(app, lambda: app.section is not None)
    for owner, name in ((A.WG, "compute"), (A.WG, "_model"), (A.V, "compute")):                   # nothing here may run a model
        monkeypatch.setattr(owner, name, lambda *a, **k: pytest.fail("a model ran"))
    app.wing.set(True); app.show()
    row = app.range_note.get
    # where it is and how it looks: the fourth row of the wing rows, gray, wrapped like the 2D box's note, wider where the box is
    label, = [w for w in app.f3.walk() if isinstance(w, Label) and w.kw.get("textvariable") is app.range_note]
    rows = app.f3.children
    assert len(rows) == 6 and rows[3].children == [label] and label.master is rows[3]
    assert [r.cell for r in rows] == [dict(row=i, column=0, sticky="ew", pady=(0 if i == 0 else 3, 0)) for i in range(6)]
    assert any(isinstance(w, Radiobutton) for w in rows[2].walk()) and any(w is app.surface_box for w in rows[4].walk())      # after Section model, before Surface
    assert label.kw["foreground"] == "gray" and label.kw["wraplength"] == app.status.kw["wraplength"] == 340 and label.kw["justify"] == "left"
    for width, wrap in ((1, 340), (300, 340), (373, 365), (500, 492)):                            # the Flow box's width minus a margin, never below 3.4 in
        label.bound["<Configure>"](types.SimpleNamespace(width=width))
        assert label.kw["wraplength"] == wrap
    # the Full-scale wing at its own 45 m/s (NACA 2412 in the Airfoil box: a NACA section, so no family note)
    assert row() == SPEEDS + INCIDENCE + " 45 m/s: inside both."
    at30 = "30 m/s: XFOIL state only; RANS state below its Re range at the outboard strips."
    app.speed.set("30")                                                                           # typing a speed
    assert row() == SPEEDS + INCIDENCE + " " + at30
    app.wstate.set("sst")                                                                         # the chosen state that does not fit says so
    assert row() == SPEEDS + INCIDENCE + " " + at30 + " The RANS state you chose does not fit this speed."
    app.speed.set("70")
    assert row().endswith(" 70 m/s: XFOIL state only; RANS state above its Re range at the inboard strips. The RANS state you chose does not fit this speed.")
    app.speed.set("45")
    assert row() == SPEEDS + INCIDENCE + " 45 m/s: inside both."
    app.wstate.set("xfoil")
    app.speed.set("70")
    assert row().endswith("RANS state above its Re range at the inboard strips.")                 # the XFOIL state fits: nothing more
    for junk in ("", "fast", "-3", "0", "nan", "inf", "4,5"):                                     # a speed that is not a number (yet) is not judged
        app.speed.set(junk)
        assert row() == SPEEDS + INCIDENCE, junk
    app.speed.set("45")
    app.scale.set("2")                                                                            # twice the size: half the speeds
    assert row() == ("Speeds that keep every strip inside the training Re: XFOIL state 0.5-51 m/s, RANS state 20-30.6 m/s. " + INCIDENCE
                      + " 45 m/s: XFOIL state only; RANS state above its Re range at the inboard strips.")
    for bad, note in (("x", "Scale x: 'x' is not a number"), ("0", "Scale x: must be positive (got 0)")):    # Run's own words (wing_definition)
        app.scale.set(bad)
        assert row() == f"Ranges not shown ({note})."
    app.scale.set("1")
    assert row() == SPEEDS + INCIDENCE + " 45 m/s: inside both."
    app.strips.set("16")                                                                          # fewer strips: the innermost one has less chord
    assert row() == ("Speeds that keep every strip inside the training Re: XFOIL state 0.996-105 m/s, RANS state 39.9-63 m/s. " + INCIDENCE
                     + " 45 m/s: inside both.")
    app.strips.set("4")
    assert row() == "Ranges not shown (Strips: must be a whole number from 8 to 200 (got 4))."
    app.strips.set("40")
    app.speed.set("61")                                                                           # above the stations' 60 m/s, but no strip reaches the root chord
    assert row() == SPEEDS + INCIDENCE + " 61 m/s: inside both."
    for tiny in ("1e-320", "1e200"):                                                              # a size no float holds: a note, and the section still loads
        app.scale.set(tiny)
        assert row() == "Ranges not shown (the size is out of range)."
    app.choose("naca", "23012"); settle(app, lambda: app.section[1] == "NACA 23012")
    assert app.pname.kw["text"].startswith("NACA 23012") and app.code == "23012"
    app.scale.set("1"); app.speed.set("45")
    assert row() == SPEEDS + INCIDENCE + " 45 m/s: inside both."
    # the planform: the Model wing's own 20 m/s is far below the RANS state's Re range
    app.planform.set("Model wing (span 2 m)"); app.pick_planform()
    assert row() == ("Speeds that keep every strip inside the training Re: XFOIL state 5-510 m/s, RANS state 200-306 m/s. " + INCIDENCE
                     + " 20 m/s: XFOIL state only; RANS state below its Re range on every strip.")
    # the geometry table: half a row typed, then the impossible wing (chord ratio 3.75 against the RANS state's Re ratio 3)
    app.planform.set(FULL); app.pick_planform()
    app.stations.type("\n3 0.5 0 0")
    assert row() == "Ranges not shown (stations, line 4: expected y, chord, twist, x, z and a section (found 4 of 6 fields))." and app.planform.get() == "Custom"
    app.stations.delete("1.0", "end"); app.stations.insert("1.0", "0 1.5 0 0 0 2412\n5 0.4 0 0 0 2412")
    app.stations.bound["<<Modified>>"](None)
    assert row() == ("Speeds that keep every strip inside the training Re: XFOIL state 1.88-102 m/s, RANS state no speed (strip chord ratio 3.63 > its Re ratio 3). "
                     + INCIDENCE + " 45 m/s: XFOIL state only; RANS state below its Re range at the outboard strips.")
    # a section that is not NACA 4/5-digit: a UIUC station, then a 'current' that is not NACA, then NACA again
    app.planform.set(KINKED); app.pick_planform()
    assert row() == ("Speeds that keep every strip inside the training Re: XFOIL state 1.5-101 m/s, RANS state 59.9-60.7 m/s. " + INCIDENCE
                     + " 60 m/s: inside both.")
    lines = app.stations.get("1.0", "end").splitlines()
    app.stations.delete("1.0", "end"); app.stations.insert("1.0", "\n".join(lines[:2] + [lines[2].replace("4415", "uiuc:e387")] + lines[3:]))
    app.stations.bound["<<Modified>>"](None)
    assert row().endswith(" 60 m/s: inside both. " + FAMILY) and "RANS state 59.9-60.7 m/s" in row()
    app.planform.set(FULL); app.pick_planform()
    assert FAMILY not in row()
    app.done_section(("file", "/x/clarky.dat"), (np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]), "clarky", None), None)    # the Airfoil box's section changes
    assert row() == SPEEDS + INCIDENCE + " 45 m/s: inside both. " + FAMILY
    app.speed.set("30"); app.wstate.set("sst")                                                    # the chosen state's clause follows the verdict, before the family note
    assert row() == SPEEDS + INCIDENCE + " " + at30 + " The RANS state you chose does not fit this speed. " + FAMILY
    app.wstate.set("xfoil"); app.speed.set("45")
    app.wing.set(False); app.show()
    app.choose("naca", "4412"); settle(app, lambda: app.section[1] == "NACA 4412")
    app.wing.set(True); app.show()                                                                # the section changed while the box was off
    assert row() == SPEEDS + INCIDENCE + " 45 m/s: inside both."


def test_section_rows_change_only_the_section_column():
    """The table text with the section column of station rows set: the five numbers keep their text and separators; comments, blank lines
    and lines without five numbers (words too) are not rows; first and last choose lines; a row with no section gets one; lines are
    numbered at newlines only, as Tk's Text numbers them."""
    text = ("# y chord twist x z section\n0 1.5 0 0 0 current\n\n  # a note 1 2 3 4 5 6\n2, 1.3, -1, 0, 0, NACA 4415\n"
            "3\t1.0\t-2\t0\t0\t/my foils/e 387.dat\n5 0.5 -3 0 0\nnot a row\n7 0.3 0 0 0 4412 \n"
            "y chord twist x z section\nthis is my note about the wing\n")
    assert A.section_rows(text, "uiuc:e387") == {2: "0 1.5 0 0 0 uiuc:e387", 5: "2, 1.3, -1, 0, 0, uiuc:e387", 6: "3\t1.0\t-2\t0\t0\tuiuc:e387",
                                                  7: "5 0.5 -3 0 0  uiuc:e387", 9: "7 0.3 0 0 0 uiuc:e387"}
    assert A.section_rows(text, "2412", 5, 6) == {5: "2, 1.3, -1, 0, 0, 2412", 6: "3\t1.0\t-2\t0\t0\t2412"} and A.section_rows(text, "2412", 5, 5).keys() == {5}
    assert A.section_rows(text, "2412", 3, 4) == {} and A.section_rows(text, "2412", 8, 8) == {} and A.section_rows("", "2412") == {}
    assert A.section_rows("0 1 0 0 0 2412\x0c\n5 1 0 0 0 2412\u2028", "4412") == {1: "0 1 0 0 0 4412", 2: "5 1 0 0 0 4412"}
    assert [s["section"] for s in A.parse_stations("0 1 0 0 0 2412\x0c\n5 1 0 0 0 4412\u2028")] == ["2412", "4412"]
    with pytest.raises(ValueError, match="stations, line 2: expected"):                         # the window's line, not splitlines()' third
        A.parse_stations("0 1 0 0 0 2412\x0c\n3 0.5 0 0")


def test_the_airfoil_boxs_section_goes_on_rows(tk, tmp_path, monkeypatch):
    """The Wing geometry window's Selected row(s) and All rows put the Airfoil box's section on the table's rows with its own name (a NACA
    designation, uiuc:<name> or the file's path, never 'current'), in the section column only: the selected rows, else the cursor's row, else
    every row; numbers, separators, comments and blank lines stay; the wing becomes Custom and its ranges follow; refusals say why."""
    import predict_field as PF
    clark = tmp_path / "clarky.dat"
    np.savetxt(clark, PF.naca_coords("4412"), header="Clark Y stand-in", comments="")
    monkeypatch.setattr(A, "fetch", lambda url, path: str(clark))
    app, root = tk()
    settle(app, lambda: app.section is not None)
    app.wing.set(True); app.show()
    t, status, rows = app.stations, (lambda: app.status.kw["text"]), (lambda: app.stations.get("1.0", "end").splitlines())
    btn = {k: v for k, v in texts(app.stw).items() if k in ("Selected row(s)", "All rows")}
    assert set(btn) == {"Selected row(s)", "All rows"} and all(isinstance(b, Button) for b in btn.values())
    btn["Selected row(s)"].kw["command"]()                                                                         # nothing was clicked: the cursor is on the header
    assert rows() == [A.STATIONS_HEADER, "0  1.5  0  0  0  current", "5  0.75  -2  0  0  current"]
    assert status().startswith("No station row there: click a row of the table (or select several), then press Selected row(s).")
    assert A.SECTION_HINT in texts(app.stw) and "(current)" in A.SECTION_HINT                                    # the hint says what 'current' does
    # a NACA section on the cursor's row
    app.choose("naca", "4412"); settle(app, lambda: app.section[1] == "NACA 4412")
    t.mark_set("insert", "3.5"); t.edit_modified(False)
    btn["Selected row(s)"].kw["command"]()
    assert rows() == [A.STATIONS_HEADER, "0  1.5  0  0  0  current", "5  0.75  -2  0  0  4412"] and t.index("insert") == "3.5"
    assert status() == "Set the section of 1 row to 4412. Custom wing: NACA 4412 (its own sections; 'current' is the Airfoil box's NACA 4412)"
    assert app.planform.get() == "Custom" and app.wname == "Custom wing" and t.edit_modified() is False
    # a UIUC section on the selected rows: a selection that ends at the start of a line does not take that line
    app.choose("uiuc", "clarky"); settle(app, lambda: app.section[1] == "clarky")
    assert app.origin == "uiuc:clarky" and app.code is None
    t.select("2.0", "3.0")
    btn["Selected row(s)"].kw["command"]()
    assert rows() == [A.STATIONS_HEADER, "0  1.5  0  0  0  uiuc:clarky", "5  0.75  -2  0  0  4412"]
    assert status() == "Set the section of 1 row to uiuc:clarky. Custom wing: clarky to NACA 4412 (its own sections)"
    assert app.range_note.get().endswith(FAMILY)                                                                   # the table has a UIUC section now
    t.select("2.0", "3.end")                                                                                       # both rows
    btn["Selected row(s)"].kw["command"]()
    assert rows()[1:] == ["0  1.5  0  0  0  uiuc:clarky", "5  0.75  -2  0  0  uiuc:clarky"] and status().startswith("Set the section of 2 rows to uiuc:clarky.")
    # a file: its path; every row
    app.choose("file", str(clark)); settle(app, lambda: app.section[1] == "clarky" and app.origin == str(clark))
    btn["All rows"].kw["command"]()
    assert rows()[1:] == [f"0  1.5  0  0  0  {clark}", f"5  0.75  -2  0  0  {clark}"] and status().startswith(f"Set the section of 2 rows to {clark}.")
    assert [s["section"] for s in A.parse_stations(t.get("1.0", "end"))] == [str(clark)] * 2                      # a path with no quotes parses back
    # a preset's 'current' becomes the digits; comments, blank lines, commas and tabs stay
    app.planform.set(FULL); app.pick_planform()
    app.choose("naca", "23012"); settle(app, lambda: app.section[1] == "NACA 23012")
    btn["All rows"].kw["command"]()
    assert rows() == [A.STATIONS_HEADER, "0  1.5  0  0  0  23012", "5  0.75  -2  0  0  23012"] and "current" not in t.get("1.0", "end")
    t.delete("1.0", "end"); t.insert("1.0", "# my wing\n0, 1.5, 0, 0, 0, current\n\n# the tip\n5\t0.75\t-2\t0\t0\tuiuc:e387\n")
    btn["All rows"].kw["command"]()
    assert rows() == ["# my wing", "0, 1.5, 0, 0, 0, 23012", "", "# the tip", "5\t0.75\t-2\t0\t0\t23012", ""]
    before = t.get("1.0", "end")
    t.select("3.0", "4.end")                                                                                       # a selection with no row in it
    btn["Selected row(s)"].kw["command"]()
    assert t.get("1.0", "end") == before and status().startswith("No station row there")
    # refusals: no section yet, a wing's cut (no name a station can use)
    keep = app.section
    app.section, app.code, app.origin = None, None, None
    btn["All rows"].kw["command"]()
    assert status() == "Load a section in the Airfoil box first (NACA, UIUC or your file)." and t.get("1.0", "end") == before
    app.section = (keep[0], "blend 0.5, 2y/b 0.51", "naca")
    btn["All rows"].kw["command"]()
    assert status() == ("The Airfoil box's section (blend 0.5, 2y/b 0.51) is a wing's cut: no name a station can use. "
                        "Load a NACA, UIUC or file section.") and t.get("1.0", "end") == before


def test_a_uiuc_section_put_on_the_wing_by_the_helper_runs(tk, tmp_path, monkeypatch):
    """A UIUC section picked in the Airfoil box and put on the rows by the Wing geometry window is fetched in the worker when Run
    resolves its uiuc: token (the network is a local file here), and reaches the lifting line as coordinates."""
    import predict_field as PF
    clark = tmp_path / "clarky.dat"
    np.savetxt(clark, PF.naca_coords("4412"), header="Clark Y stand-in", comments="")
    fetched = []
    monkeypatch.setattr(A, "fetch", lambda url, path: fetched.append(url) or str(clark))
    app, root = tk()
    settle(app, lambda: app.section is not None)
    small_wing(app)
    app.choose("uiuc", "clarky"); settle(app, lambda: app.section[1] == "clarky")
    url = A.UIUC + "coord/clarky.dat"
    assert fetched == [url]                                                                                        # the Airfoil box's own load
    texts(app.stw)["All rows"].kw["command"]()
    assert [s["section"] for s in A.parse_stations(app.stations.get("1.0", "end"))] == ["uiuc:clarky"] * 2
    app.run_wing()
    settle(app, lambda: app.wres is not None)
    assert fetched == [url, url]                                                                                   # and Run's, once for both rows
    res = app.wres
    assert res["wing"]["name"] == "Custom wing" and all("clarky" in lb for lb in res["loops"]["label"]) and set(res["loops"]["family"]) == {None}
    assert res["coef"]["CL"] > 0.3 and app.range_note.get().endswith(FAMILY)


# ------------------------------------------------------------------------------------------------ vortfoil (2D) and vortfoil_exp (+ 3D)
def describe(w, skip=()):
    """a widget subtree as nested tuples (class, options, grid cell, mapped, children) that compare across two windows: a variable by
    its value, a command by its name; widgets whose id is in `skip` are left out"""
    def val(v):
        return ("var", v.get()) if isinstance(v, Var) else getattr(v, "__name__", v) if callable(v) else v
    return (type(w).__name__, sorted((k, repr(val(v))) for k, v in w.kw.items()), repr(w.cell), w.mapped,
            [describe(c, skip) for c in w.children if id(c) not in skip])


def test_vortfoil_window_is_2d_only(tk):
    """`vortfoil` (App without experimental) builds nothing of the 3D wing: no '3D wing' box, no wing rows, no Wing geometry window, so
    the wing mode cannot be reached. Widget for widget (options, cells, mapping) it is the experimental window without them, in both
    layouts, with the released 2D Flow box; only the experimental window's title says so."""
    for screen in ((1728, 1117), (1080, 1920)):
        two, r2 = tk(screen, experimental=False)
        exp, rx = tk(screen)
        for app in (two, exp):
            settle(app, lambda: app.section is not None)
        assert "3D wing" not in texts(r2) and not any(hasattr(two, k) for k in ("chk", "f3", "stw", "cur", "cut_btn", "view3d_btn", "range_note", "origin"))
        assert describe(r2) == describe(rx, skip={id(exp.chk), id(exp.f3), id(exp.stw)}) and len(r2.children) == len(rx.children) - 1
        assert set(texts(two.f2)) == FLOW_2D | ({REXT} if two.rext_path else set()) and two.f2.mapped and two.mode() == "section"
        assert (r2.name, rx.name) == ("VortFoil", "VortFoil (experimental)") and two.status.kw["text"] == "NACA 2412 loaded. Set the flow and press Run."


def test_vortfoil_runs_2d_as_the_experimental_window(tk, tmp_path, monkeypatch):
    """The 2D case in vortfoil's window: Run greys out while it runs; the figure (pixel for pixel, at its design size), Save figure's
    name, the CSV files (byte for byte) and the status line are those of the experimental window."""
    import re
    from matplotlib.image import imread
    got = {}
    for exp in (False, True):
        app, root = tk(experimental=exp)
        settle(app, lambda: app.section is not None)
        app.run()
        assert app.run_btn["state"] == "disabled" and app.save_btn["state"] == "disabled"
        settle(app, lambda: app.last is not None)
        assert app.run_btn["state"] == app.save_btn["state"] == app.csv_btn["state"] == "normal" and len(app.fig.axes) == 12
        status = re.sub(r"done in [0-9.]+ s", "done in t s", app.status.kw["text"])
        app.save_to(str(tmp_path / f"{exp}.png"))
        saved = []
        monkeypatch.setattr(tk.fd, "asksaveasfilename", lambda **kw: saved.append(kw["initialfile"]) or "")
        app.save()
        monkeypatch.setattr(tk.fd, "askdirectory", lambda **kw: str(tmp_path / f"csv{exp}"))
        app.export()
        settle(app, lambda: app.status.kw["text"].startswith("Wrote"))
        files = sorted(os.listdir(tmp_path / f"csv{exp}"))
        got[exp] = (status, saved, files, [open(tmp_path / f"csv{exp}" / f, "rb").read() for f in files], imread(str(tmp_path / f"{exp}.png")))
    (s0, n0, f0, b0, p0), (s1, n1, f1, b1, p1) = got[False], got[True]
    assert s0 == s1 and s0.startswith("NACA 2412, 4 deg, Re 3e+06: done in t s") and n0 == n1 == ["vortfoil_NACA_2412_a4_Re3e+06.png"]
    assert f0 == f1 and len(f0) == 8 and b0 == b1 and p0.shape == p1.shape and np.array_equal(p0, p1)


def test_vortfoil_exp_command(monkeypatch, capsys):
    """vortfoil_exp (cli.main_exp): no arguments and `app ...` open the window as experimental; view, predict, sweep, about and
    selfcheck are vortfoil's (the same calls, the same output, its own name in the usage); vortfoil's window is the 2D one."""
    from vortfoil import app as P, cli
    import vortfoil.selfcheck as SC
    seen = []
    monkeypatch.setattr(P, "app", lambda argv, experimental=False: seen.append(("app", argv, experimental)) or 0)
    monkeypatch.setattr(P, "view", lambda argv: seen.append(("view", argv)) or 0)
    monkeypatch.setattr(SC, "run", lambda **kw: seen.append(("selfcheck", kw)) or 0)
    monkeypatch.setattr(sys, "argv", ["vortfoil_exp"])
    for f, argv in ((cli.main_exp, None), (cli.main_exp, ["app", "--selftest", "out", "--wing", "kinked"]), (cli.main, []),
                    (cli.main, ["app", "--shortcut"]), (cli.main_exp, ["view", "--airfoil", "2412"]), (cli.main, ["view", "--airfoil", "2412"]),
                    (cli.main_exp, ["selfcheck", "--data", "x.npz"]), (cli.main, ["selfcheck", "--data", "x.npz"])):
        assert f(argv) == 0
    assert seen == [("app", [], True), ("app", ["--selftest", "out", "--wing", "kinked"], True), ("app", [], False),
                    ("app", ["--shortcut"], False)] + [("view", ["--airfoil", "2412"])] * 2 + [("selfcheck", dict(mode="graph", device="cpu", data="x.npz"))] * 2
    capsys.readouterr()
    for f, prog in ((cli.main, "vortfoil"), (cli.main_exp, "vortfoil_exp")):
        with pytest.raises(SystemExit):
            f(["predict", "--help"])
        assert capsys.readouterr().out.startswith(f"usage: {prog} predict ")
    cst = "0.17,0.16,0.18,0.15,0.17,0.12,0.13,0.11,-0.12,-0.08,-0.09,-0.05,-0.07,-0.04,-0.05,-0.02"
    for argv in (["about"], ["predict", "--cst", cst, "--alpha", "4", "--Re", "3e6"], ["sweep", "--cst", cst, "--alpha-range", "0,8,3"]):
        assert cli.main(argv) == 0
        out = capsys.readouterr().out
        assert cli.main_exp(argv) == 0 and capsys.readouterr().out == out and out.strip(), argv


def test_the_wing_needs_experimental(monkeypatch, capsys):
    """--selftest DIR --wing without --experimental stops before any window with the reason; with it (or without --wing) the window is
    made with the matching experimental flag. The vortfoil command (main(..., experimental=False)) is 2D only: it refuses --experimental,
    abbreviated too, before any window; vortfoil_exp (True) is experimental without the option."""
    monkeypatch.setitem(sys.modules, "tkinter", None)                             # a window would need tkinter: none may open
    with pytest.raises(SystemExit) as e:
        A.main(["--selftest", "out", "--wing", "kinked"])
    assert e.value.code == 2 and "the 3D wing is experimental; run it with vortfoil_exp, or add --experimental" in capsys.readouterr().err
    made = []
    monkeypatch.setitem(sys.modules, "tkinter", types.SimpleNamespace(Tk=lambda: types.SimpleNamespace(mainloop=lambda: None)))
    monkeypatch.setattr(A, "App", lambda root, selftest=None, experimental=False: made.append((selftest, experimental)) or types.SimpleNamespace(exit_code=0))
    assert A.main(["--selftest", "out", "--wing", "kinked", "--experimental"]) == A.main(["--selftest", "out"]) == A.main([]) == 0
    assert made == [(("out", "naca:2412", 4.0, 3e6, "kinked", None, "xfoil"), True), (("out", "naca:2412", 4.0, 3e6, None, None, "xfoil"), False),
                    (None, False)]
    made.clear()
    assert A.main(["--selftest", "out", "--wing", "kinked"], True) == A.main([], True) == A.main([], False) == A.main([], None) == 0
    assert made == [(("out", "naca:2412", 4.0, 3e6, "kinked", None, "xfoil"), True), (None, True), (None, False), (None, False)]
    from vortfoil import cli
    for main, argv in [(A.main, (a, False)) for a in (["--experimental"], ["--exp"], ["--shortcut", "--experimental"])] + [(cli.main, (["app", "--experimental"],))]:
        with pytest.raises(SystemExit) as e:
            main(*argv)
        assert e.value.code == 2 and "--experimental: the vortfoil command is 2D only; the experimental 3D wing is vortfoil_exp" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        A.main(["--selftest", "out", "--wing", "kinked"], False)
    assert e.value.code == 2 and capsys.readouterr().err.rstrip().endswith("--wing: the 3D wing is experimental; run it with vortfoil_exp")
    assert len(made) == 4                                                       # no window for any of these


@pytest.mark.parametrize("experimental", [False, True])
def test_shortcuts(monkeypatch, tmp_path, capsys, experimental):
    """--shortcut makes the 2D launcher as released; with --experimental a separate one, 'VortFoil (experimental)', that passes
    --experimental: the Windows desktop shortcut (the PowerShell it runs), the Linux menu entry (under a stand-in home), the macOS
    terminal command. Nothing is written to this computer's desktop or menus."""
    import shutil
    app, flag = os.path.abspath(A.__file__), " --experimental" if experimental else ""
    name, cmd = ("VortFoil (experimental)", "vortfoil_exp") if experimental else ("VortFoil", "vortfoil")
    ran = []
    monkeypatch.setattr(A.subprocess, "run", lambda c, **kw: ran.append(c) or types.SimpleNamespace(stdout="C:\\Desktop\\made.lnk\n"))
    monkeypatch.setattr(sys, "platform", "win32")
    assert A.make_shortcut(experimental) == "C:\\Desktop\\made.lnk" and ran[0][:3] == ["powershell", "-NoProfile", "-Command"]
    assert f"GetFolderPath('Desktop'), '{name}.lnk'));" in ran[0][3] and f"$s.Arguments = '\"{app}\"{flag}';" in ran[0][3]
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    path = A.make_shortcut(experimental)
    assert path == str(tmp_path / ".local" / "share" / "applications" / ("vortfoil-exp.desktop" if experimental else "vortfoil.desktop"))
    entry = open(path).read()
    assert f"\nName={name}\n" in entry and f'\nExec="{sys.executable}" "{app}"{flag}\n' in entry and len(os.listdir(os.path.dirname(path))) == 1
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(shutil, "which", lambda c: None)
    line = f'"{sys.executable}" "{app}"{flag}'
    assert A.make_shortcut(experimental) == (f"no file written on macOS; start it from the terminal with\n  {line}\n"
                                             f"or add this line to ~/.zshrc and type {cmd}:\n  alias {cmd}='{line}'")
    monkeypatch.setattr(shutil, "which", lambda c: "/usr/local/bin/" + c)
    assert A.make_shortcut(experimental) == f"no file written on macOS; the {cmd} command is installed: type {cmd} in a terminal"
    monkeypatch.setattr(A, "make_shortcut", lambda e=False: f"made, experimental {e}")
    assert A.main(["--shortcut"] + flag.split()) == 0 and capsys.readouterr().out == f"shortcut: made, experimental {experimental}\n"


def test_console_scripts():
    """pyproject.toml: vortfoil and vortfoil_exp are the command line's main and main_exp"""
    toml = pytest.importorskip("tomllib" if sys.version_info >= (3, 11) else "tomli")
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        assert toml.load(f)["project"]["scripts"] == {"vortfoil": "vortfoil.cli:main", "vortfoil_exp": "vortfoil.cli:main_exp"}
