# -*- coding: utf-8 -*-
"""VortFoil as a desktop application: choose an airfoil, set the incidence and the Reynolds number, press Run, and
read the flow field and the surface results in the same window. With --experimental (the vortfoil_exp command) the window
also builds a wing from sections and reads its loads: an experimental quasi-3D wing, not part of the paper's validated results.

    vortfoil                       (installed with `uv tool install --python 3.12 --managed-python vortfoil`)
    python code/vortfoil_app.py    (from a checkout)
    vortfoil_exp                   (the same with the experimental 3D wing; from a checkout: add --experimental)

The airfoil comes from one of three places:
  NACA     a 4- or 5-digit designation, generated exactly as the AirfRANS sections were;
  UIUC     the UIUC Airfoil Coordinates Database (M. S. Selig, UIUC Applied Aerodynamics Group,
           https://m-selig.ae.illinois.edu/ads/coord_database.html). Its list and the chosen section are downloaded
           from the site when needed and cached on this computer; VortFoil does not redistribute them
           (DATA_LICENSES.md);
  file     any coordinate file (Selig or Lednicer .dat), for a section that is in no public database.
Export data (CSV) writes every number of the result (coefficients, surface C_p, the boundary layers, the field at every
grid point, the section) as CSV files with a text file that explains their columns. The result is the figure of
code/view_airfoil.py: the field of the decoder trained on AirfRANS (wind frame) and the surface pressure, boundary layer
and coefficients in two states, with every range warning, side by side in a wide window and stacked in a tall one. The
boxes XFOIL state (the vortfoil package: vortex sheet and boundary-layer operator, trained on XFOIL) and RANS state (the
field decoder's wall values and the boundary layer read from its flow) choose which of them the figure shows, both by
default; a state outside its training range is drawn faded and labelled. Resizing the window zooms the figure, and Save
figure writes it at 400 dpi (png, pdf or svg). It needs matplotlib and Tk (part of most Python installs).

Experimental, in the window of vortfoil_exp (or --experimental) only: next to the section's name, the box "3D wing" turns
the Flow box into a wing: a planform (presets, or the table of
"Geometry ...": one station per row, with its span position y, chord, twist, quarter-chord x and z for sweep and dihedral,
and a section: a NACA designation, `current` (the Airfoil box's section, which the one-section presets use),
`uiuc:<name>` or a coordinate file; any size in metres, kinks at any station), a flow speed in m/s and an incidence. A
nonlinear lifting line (code/wing_nll.py) carries the sections in one of two states: the surface model (XFOIL state, free
transition) or the RANS field decoder (RANS state, fully turbulent). The figure (code/view_wing.py) shows the wing in 3D at
true scale (drag it to turn it, double-click to start over) coloured by C_p or another surface quantity, a
spanwise quantity with a section cut that you move by dragging, the section at that cut, the wing polar, the coefficients
and every warning. Each mode keeps its last result: Run, Save figure and Export data (CSV) act on the mode that is on (the
wing's export lets you tick the parts to write), and Open cut in 2D analyses the section at the cut as an airfoil. A gray
line below the Section model row gives the speeds that keep every strip inside each state's training Reynolds number, the
strips' incidence limits, whether the speed typed fits, and a note when a section is not NACA 4/5-digit (the RANS state was
trained on those only). In the Geometry window, Selected row(s) and All rows put the Airfoil box's section (a NACA
designation, a UIUC section picked from its list, or your file) on rows of the table.

3D view (next to Open cut in 2D) opens the shown wing result in your web browser as a standalone page (code/wing_flow.py
writes it from code/wing_view.html; WebGL, nothing is fetched): the wing coloured by C_p, streamlines of a flow field (the RANS
field decoder's, or inviscid ones), the viscous wake and the trailing vortices (rolled up if you ask), turned and zoomed fast
with the mouse. The page shows the incidence of the result; the pages are kept in <cache>/vortfoil/wing3d. The viscous flow
field needs Reynolds numbers near the decoder's range (at least 1.5e6, 3/4 of its lower training Re) on every strip: a
model-scale wing takes the inviscid one (the viscous choice is greyed out for it).

  --shortcut       Windows: a desktop shortcut that starts the application with this Python environment; Linux: an
                   entry in the applications menu; macOS: print the terminal command (and an alias), write nothing
  --selftest DIR   run one case without a person (--airfoil naca:2412 | uiuc:<name> | file:<path>), save the figure
                   and a capture of the window in DIR, and exit; with --wing PRESET_OR_JSON (a preset name or a wing
                   definition file; --V speed in m/s, --state xfoil|sst, --alpha incidence) a wing instead, its CSV
                   files included (with --experimental)
  --experimental   the window with the experimental 3D wing (what vortfoil_exp opens; the vortfoil command refuses it);
                   with --shortcut, a separate launcher, "VortFoil (experimental)"
"""
import os, re, sys, time, queue, threading, argparse, traceback, subprocess, urllib.request
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")    # before numpy: OpenBLAS threads beside TF's made a panel solve 0.35 s on Windows, 1.5 ms single
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vortfoil_paths  # noqa: E402
vortfoil_paths.add_paths()            # this checkout's vortfoil package (src) and code/field, ahead of other installs
import view_airfoil as V  # noqa: E402
import view_wing as WV  # noqa: E402
import wing_nll as WG  # noqa: E402

UIUC = "https://m-selig.ae.illinois.edu/ads/"
CACHE = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".cache"), "vortfoil", "uiuc")
DETAIL = {"fast (0.01 c)": 0.01, "normal (0.005 c)": 0.005, "fine (0.0025 c)": 0.0025}
TRAINED_ON = ("Trained on: XFOIL state -10 to 18 deg, Re 5e4 to 1e7; RANS state -5 to 15 deg, Re 2e6 to 6e6. "
              "Outside, drawn faded.")


def uiuc_index(html):
    """[(name, description)] of the coordinate links on the database page."""
    return [(m.group(1), " ".join(m.group(2).split()).strip(" \\"))
            for m in re.finditer(r'href="coord/([^"/]+)\.dat"[^>]*>[^<]*</a>([^<]*)', html)]


def fetch(url, path):
    """The file at url, downloaded once into the cache."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "VortFoil application"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        with open(path + ".part", "wb") as f:
            f.write(data)
        os.replace(path + ".part", path)
    return path


def load_section(kind, value):
    """(coordinates, display name, family) of a NACA designation, a UIUC database name or a coordinate file; family is
    "naca" for a designation (the decoder's training family) and None otherwise (the field gets a family warning)."""
    import predict_field as PF
    if kind == "naca":
        code = value.upper().replace("NACA", "").strip()
        if not re.fullmatch(r"\d{4,5}", code):
            raise ValueError(f"'{value}' is not a NACA 4- or 5-digit designation")
        return PF.naca_coords(code), f"NACA {code}", "naca"
    if kind == "uiuc":
        return PF.read_coords(fetch(f"{UIUC}coord/{value}.dat", os.path.join(CACHE, value + ".dat"))), value, None
    return PF.read_coords(value), os.path.splitext(os.path.basename(value))[0], None


def load_index():
    path = fetch(UIUC + "coord_database.html", os.path.join(CACHE, "coord_database.html"))
    with open(path, encoding="latin-1") as f:
        return uiuc_index(f.read())


def run_case(loop, alpha, Re, h, name, family, decoder=None):
    t0 = time.perf_counter()
    res = V.compute(loop, alpha, Re, h, family, decoder)
    return res, time.perf_counter() - t0


# ---------------------------------------------------------------------------------------------- the 3D wing
STATE_LABEL = {"xfoil": "XFOIL state", "sst": "RANS state"}
HINTS = {"section": "Choose an airfoil, set the flow and press Run",
         "wing": "Choose a planform (Geometry ... edits it; one-section planforms use the Airfoil box's section),\n"
                 "set the speed and press Run. XFOIL state: surface model, free transition;\n"
                 "RANS state: SST field decoder, fully turbulent. Then drag in the spanwise plot to\n"
                 "move the section cut; drag the wing to turn it, double-click to reset."}
GEOMETRY_TEXT = ("Each row is a spanwise station: the wing's shape at one position along the half-span, root (y = 0) first. "
                 "Between rows the chord, twist, sweep, dihedral and section change linearly.\n"
                 "Columns: y chord twist x z section. Lengths in m, twist in deg (nose up); x and z of the quarter chord give the "
                 "sweep and the dihedral. The left half mirrors the right.\n"
                 "section: a NACA 4/5-digit designation, current (the Airfoil box's section), uiuc:<name> or a file.")
SECTION_HINT = ("Pick a section in the Airfoil box (UIUC: search its list), click a row here and press Selected row(s), or press All rows. "
                "A one-section planform already uses the Airfoil box's section (current).")
STATIONS_HEADER = "# y_m  chord_m  twist_deg  x_c4_m  z_c4_m  section"
COLUMNS = ("y", "chord", "twist", "x", "z")
STATION = re.compile(r"(\s*(?:[-+.\d][^\s,]*[,\s]+){4}[-+.\d][^\s,]*)([,\s]+)?.*")    # a row: five numbers as typed, their separator, then the section (the rest)


def section_name(token):
    """a station's section as people read it: NACA 4412, the UIUC name or the file's name ('current' is the Airfoil box's: not here)"""
    m = re.fullmatch(r"(?i)\s*(?:naca)?\s*(\d{4,5})\s*", token)
    return f"NACA {m.group(1)}" if m else os.path.splitext(os.path.basename(token.replace("uiuc:", "")))[0]


def naca_code(label, family):
    """the designation digits of a section labelled 'NACA 2412' whose family is 'naca', else None"""
    m = re.fullmatch(r"NACA (\d{4,5})", label)
    return m.group(1) if m and family == "naca" else None


def format_stations(stations):
    """the Wing geometry window's text for a wing's stations: a header comment, then y chord twist x z section per line, root first"""
    return "\n".join([STATIONS_HEADER] + ["  ".join([f"{s[k]:.8g}" for k in COLUMNS] + [s["section"]]) for s in stations])


def parse_stations(text):
    """the stations (dicts as wing_nll takes them) of the Wing geometry window's text: per line y chord twist x z and the section, which
    is the rest of the line (a NACA designation, 'current', 'uiuc:<name>' or a file path); blank and '#' lines are skipped. Lines end at
    newlines only, as Tk's Text counts them (the line numbers of the messages are the window's)"""
    out = []
    for n, line in enumerate(text.split("\n"), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = re.split(r"[,\s]+", line, maxsplit=5)
        if len(f) < 6:
            raise ValueError(f"stations, line {n}: expected y, chord, twist, x, z and a section (found {len(f)} of 6 fields)")
        try:
            v = [float(x) for x in f[:5]]
        except ValueError:
            raise ValueError(f"stations, line {n}: y, chord, twist, x and z must be numbers (got '{' '.join(f[:5])}')") from None
        out.append(dict(zip(COLUMNS, v), section=os.path.expanduser(f[5].strip("\"' "))))
    return out


def section_rows(text, token, first=None, last=None):
    """{line number: the row with its section set to `token`} for the station rows of the geometry table `text` from line `first` to `last`
    (every row when None): only the section column changes (the five numbers keep their text and separators); comments and blank lines are
    not rows, and a line without five numbers is left alone; lines are numbered as Tk's Text numbers them (at newlines only)"""
    out = {}
    for n, line in enumerate(text.split("\n"), 1):
        m = None if line.lstrip().startswith("#") or (first is not None and not first <= n <= last) else STATION.fullmatch(line)
        if m:
            out[n] = m.group(1) + (m.group(2) or "  ") + token
    return out


def wing_definition(text, scale="1", n="40", name="Custom wing", description=""):
    """the wing of the 3D form as a validated definition (wing_nll.validate): the stations of `text` with every length times
    `scale`, and `n` strips; ValueError in English, naming the field"""
    def num(label, v):
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{label}: '{v}' is not a number") from None
    k, n = num("Scale x", scale), num("Strips", n)
    if not 0 < k < float("inf"):
        raise ValueError(f"Scale x: must be positive (got {k:g})")
    if not 8 <= n <= 200 or n != int(n):
        raise ValueError(f"Strips: must be a whole number from 8 to 200 (got {n:g})")
    st = parse_stations(text)
    for s in st:
        for q in ("y", "chord", "x", "z"):
            s[q] *= k
    return WG.validate(dict(name=name if k == 1 else f"{name}, scale x {k:g}", description=description, n=n, stations=st))


def pick_wing(spec):
    """a wing definition from a preset name (exact, or any part of one when that is unambiguous; case ignored) or a file"""
    if spec in WG.PRESETS:
        return WG.PRESETS[spec]
    if os.path.isfile(spec):
        return WG.load_wing(spec)
    hit = [k for k in WG.PRESETS if spec.strip().lower() in k.lower()]
    if len(hit) == 1:
        return WG.PRESETS[hit[0]]
    raise ValueError(f"'{spec}' is " + ("ambiguous" if hit else "neither a preset nor a wing definition file")
                     + "; the presets are: " + "; ".join(WG.PRESETS))


def resolve_sections(defn, current=None, load=None):
    """(definition, sections) for wing_nll.compute: the stations' 'current' and 'uiuc:<name>' sections resolved. current =
    (coordinates, NACA designation or None[, name]) of the section in the Airfoil box: a designation goes into the stations
    (the RANS state knows NACA sections), other coordinates into `sections`; load(name) -> coordinates fetches a UIUC section."""
    d, sections = WG.validate(defn), {}
    for k, s in enumerate(d["stations"], 1):
        tok = s["section"]
        if tok == "current":
            if current is None:
                raise ValueError(f"station {k} section: 'current' needs an airfoil loaded in the Airfoil box")
            if current[1]:
                s["section"] = current[1]
            else:
                sections[tok] = current[0]
        elif tok.startswith("uiuc:") and tok not in sections:
            try:
                sections[tok] = load(tok[5:])
            except Exception as e:                                       # offline, no such section, a damaged file
                raise ValueError(f"station {k} section '{tok}': could not load it ({e})") from None
    return d, sections


def wing_case(defn, speed, alpha, state, current, progress=None, cancel=None):
    """the wing of `defn` in `state` (wing_nll.compute with its 'current' and 'uiuc:' sections resolved); (result, seconds).
    progress(fraction, text) and cancel() are compute's, called from the worker thread; current: see resolve_sections"""
    t0 = time.perf_counter()

    def uiuc(name):
        if progress:
            progress(0.0, f"fetching the UIUC section {name} ...")
        return load_section("uiuc", name)[0]
    d, sections = resolve_sections(defn, current, uiuc)
    labels = {"current": current[2]} if current and len(current) > 2 else None    # the label, cut and CSV name it, not 'current'
    res = WG.compute(d, speed, alpha, state, sections, progress, cancel, labels=labels)
    return res, time.perf_counter() - t0


def wing_base(res):
    """the file name stem of a wing result's figure and CSV files"""
    return re.sub(r"(?:[^\w.+-]|_)+", "_", f"vortfoil_wing_{res['wing']['name']}_a{res['alpha']:g}_V{res['V']:g}_{res['state']}")   # \w: a name in any script stays one


def reason(err, advice):
    """why a case failed, for the status line: the error's message, or its name when it has none (a MemoryError often has none);
    for a MemoryError, what to do about it"""
    return f"not enough memory ({advice})" if isinstance(err, MemoryError) else (str(err) or type(err).__name__)


def wing_summary(res, dt):
    """the status line of a finished wing run, led by what makes its numbers mean nothing: the RANS state far outside its decoder's
    Reynolds range, no lifting-line solution, the XFOIL state far outside the surface model's Reynolds range, a result that is not finite"""
    c, n, Re = res["coef"], len(res["warnings"]), res["strips"]["Re"]
    lead, resid = WV.range_message(res), res.get("newton", {}).get("residual", 0.0)
    if lead is None and resid > WV.NEWTON_TOL:
        lead = f"No lifting-line solution (residual {resid:.0e}): the numbers are not meaningful."
    if lead is None and res["state"] == "xfoil":
        from vortfoil.api import TRAIN_RE
        lo, hi = TRAIN_RE
        out = int(np.sum((Re < lo) | (Re > hi)))
        if 2 * out > len(Re):
            lead = (f"Reynolds number outside the surface model's range ({lo:.0e}-{hi:.0e}) on {out} of {len(Re)} strips: "
                    "the numbers are not meaningful here.").replace("e+0", "e")
    if lead is None and not all(np.isfinite(v) for v in (c["CL"], c["CD"], c["CM"])):
        lead = "The result is not finite: the numbers are not meaningful (check the speed and the size)."
    return ((lead.rstrip(".") + ". " if lead else "") +
            f"{res['wing']['name']}, {res['alpha']:g} deg, {res['V']:g} m/s ({STATE_LABEL[res['state']]}): C_L {c['CL']:.3f}, "
            f"C_D {c['CD']:.4f}, C_M {c['CM']:.3f}, done in {dt:.1f} s; "
            + (f"{n} warning(s), listed in the figure." if n else "no range warnings.")
            + (f" This incidence is outside the {STATE_LABEL[res['state']]}'s training range: the numbers are extrapolated." if lead is None and not c.get("in_range", True) else ""))


class App:
    TASKS = {"index": load_index, "section": load_section, "run": run_case, "export": V.export_csv,
             "wing": wing_case, "wing_export": WV.export_csv}

    def __init__(self, root, selftest=None, experimental=False):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        import matplotlib.pyplot as plt
        import predict_field as PF
        plt.rcParams.update(PF.STYLE)
        self.root, self.tk, self.ttk = root, tk, ttk
        self.jobs, self.section, self.index, self.shown, self.args, self.on_resize = queue.Queue(), None, [], [], None, None
        self.last, self.layout, self.relayout = None, "landscape", None
        # the 3D wing: its last result, the live figure controller (WingView), the NACA designation of the Airfoil box's section,
        # the status prefix of the running case, whether its bar is determinate, the display options, the wing's name, its description
        # and the Cancel flag (with the time the run began: a double click on Run must not cancel it); where the 2D case came from and
        # what the wing calls it, when it is the section at a wing's cut
        self.wres, self.view, self.code, self.wbase, self.determinate = None, None, None, "", False
        self.wopts = dict(surface="C_p", span="c_l", polar=WV.POLAR_PLOTS[0], cut_eta=0.5)
        self.wname, self.wdesc, self.cancel = "Full-scale wing (span 10 m)", "", threading.Event()
        self.started, self.cutof, self.cutlabel = 0.0, "", None
        self.selftest, self.exit_code, self.experimental = selftest, 0, experimental     # experimental: the 3D wing (vortfoil_exp)
        self.rext_path = PF.DECODER_REXT if experimental and os.path.exists(PF.DECODER_REXT) else None   # experimental: vortfoil_exp only
        dpi = root.winfo_fpixels("1i")                       # the screen's real resolution: a sharp figure
        root.title("VortFoil (experimental)" if experimental else "VortFoil")
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{int(0.92 * sw)}x{int(0.86 * sh)}+{int(0.04 * sw)}+{int(0.03 * sh)}")
        tall = sh > sw                                        # an upright screen: the controls go above the figure
        panel = ttk.Frame(root, padding=8)
        panel.pack(side="top" if tall else "left", fill="x" if tall else "y")
        right = ttk.Frame(root); right.pack(side="top" if tall else "left", fill="both", expand=True)
        wrap = int(3.4 * dpi)

        box = ttk.LabelFrame(panel, text="Airfoil", padding=6)
        self.kind = tk.StringVar(value="naca")
        self.naca = tk.StringVar(value="2412")
        ttk.Radiobutton(box, text="NACA 4/5-digit", value="naca", variable=self.kind).grid(row=0, column=0, sticky="w")
        e = ttk.Entry(box, textvariable=self.naca, width=10); e.grid(row=0, column=1, sticky="w")
        e.bind("<Return>", lambda ev: self.choose("naca", self.naca.get()))
        ttk.Button(box, text="Load", command=lambda: self.choose("naca", self.naca.get())).grid(row=0, column=2, sticky="e")
        ttk.Radiobutton(box, text="UIUC database (online), search:", value="uiuc", variable=self.kind).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.search = tk.StringVar(); self.search.trace_add("write", lambda *a: self.filter())
        ttk.Entry(box, textvariable=self.search).grid(row=2, column=0, columnspan=3, sticky="ew")
        lf = ttk.Frame(box); lf.grid(row=3, column=0, columnspan=3, sticky="ew")
        self.lst = tk.Listbox(lf, height=9 if tall else 14, width=46, exportselection=False)
        sb = ttk.Scrollbar(lf, orient="vertical", command=self.lst.yview); self.lst.configure(yscrollcommand=sb.set)
        self.lst.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y")
        self.lst.bind("<<ListboxSelect>>", lambda ev: self.pick_uiuc())
        self.note = ttk.Label(box, text="loading the list ...", foreground="gray"); self.note.grid(row=4, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(box, text="My coordinate file (.dat)", value="file", variable=self.kind).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(box, text="Open ...", command=self.open_file).grid(row=5, column=2, sticky="e", pady=(8, 0))

        prev = ttk.LabelFrame(panel, text="Section", padding=6)
        self.pfig = Figure(figsize=(3.4, 1.0), dpi=dpi)
        self.pax = self.pfig.add_axes([0.02, 0.05, 0.96, 0.9]); self.pax.axis("off")
        self.pcan = FigureCanvasTkAgg(self.pfig, master=prev); self.pcan.get_tk_widget().pack()
        self.wing = tk.BooleanVar(value=False)
        if experimental:                                      # vortfoil's window has no 3D: nothing of the wing is built
            self.chk = ttk.Checkbutton(prev, text="3D wing", variable=self.wing, command=self.show); self.chk.pack(side="right")
        self.pname = ttk.Label(prev, text="no airfoil loaded", wraplength=wrap); self.pname.pack(anchor="w")

        self.flow = ttk.LabelFrame(panel, text="Flow", padding=6); self.flow.columnconfigure(0, weight=1)
        fl = self.f2 = ttk.Frame(self.flow); fl.grid(row=0, column=0, sticky="ew")
        self.alpha, self.re, self.detail = tk.StringVar(value="4"), tk.StringVar(value="3e6"), tk.StringVar(value="normal (0.005 c)")
        for i, (lab, var) in enumerate((("Incidence (deg)", self.alpha), ("Reynolds number", self.re))):
            ttk.Label(fl, text=lab).grid(row=i, column=0, sticky="w")
            en = ttk.Entry(fl, textvariable=var, width=12); en.grid(row=i, column=1, sticky="w"); en.bind("<Return>", lambda ev: self.run())
        ttk.Label(fl, text="Field detail").grid(row=2, column=0, sticky="w")
        ttk.Combobox(fl, textvariable=self.detail, values=list(DETAIL), state="readonly", width=16).grid(row=2, column=1, sticky="w")
        self.show_x, self.show_r = tk.BooleanVar(value=True), tk.BooleanVar(value=True)      # the states the surface panels show
        ttk.Label(fl, text="Show").grid(row=0, column=2, sticky="w", padx=(10, 0))
        for i, (text, var) in enumerate((("XFOIL state", self.show_x), ("RANS state", self.show_r)), 1):
            ttk.Checkbutton(fl, text=text, variable=var, command=lambda v=var: self.states_changed(v)).grid(row=i, column=2, sticky="w", padx=(10, 0))
        ttk.Label(fl, text=TRAINED_ON, foreground="gray", wraplength=wrap).grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.rext = tk.BooleanVar(value=False)
        if self.rext_path:
            self.rext_chk = ttk.Checkbutton(fl, text="Re-extended field decoder (experimental; NACA sections; field Re 1e6 to 1e7)",
                                            variable=self.rext)
            self.rext_chk.grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))
        if experimental:
            self.origin = None                                # how a station names the Airfoil box's section when it is no NACA designation: uiuc:<name> or the file
            self.f3 = ttk.Frame(self.flow); self.f3.grid(row=0, column=0, sticky="ew")     # the 3D rows: same cell, one mapped at a time
            self.wing_rows(self.f3, wrap); self.f3.grid_remove()
            self.stations_window()
            self.fill_wing(WG.PRESETS[self.wname])

        act = ttk.Frame(panel)
        self.run_btn = ttk.Button(act, text="Run", command=self.run); self.run_btn.pack(side="left")
        self.save_btn = ttk.Button(act, text="Save figure ...", command=self.save, state="disabled"); self.save_btn.pack(side="left", padx=6)
        self.csv_btn = ttk.Button(act, text="Export data (CSV) ...", command=self.export, state="disabled"); self.csv_btn.pack(side="left")
        self.bar = ttk.Progressbar(panel, mode="indeterminate")
        self.status = ttk.Label(panel, text="Choose an airfoil and press Run.", wraplength=wrap, justify="left")
        cred = ttk.Label(panel, text="UIUC coordinates: M. S. Selig, UIUC Applied Aerodynamics Group, downloaded from its site.",
                         foreground="gray", wraplength=wrap)
        if tall:                                              # two columns above the figure
            box.grid(row=0, column=0, rowspan=6, sticky="nsew", padx=(0, 10))
            for i, w in enumerate((prev, self.flow, act, self.bar, self.status, cred)):
                w.grid(row=i, column=1, sticky="new", pady=(0 if i == 0 else 6, 0))
        else:                                                 # one column left of the figure
            for i, w in enumerate((box, prev, self.flow, act, self.bar, self.status)):
                w.grid(row=i, column=0, sticky="ew", pady=(0 if i == 0 else 8, 0))
            panel.rowconfigure(6, weight=1)
            cred.grid(row=6, column=0, sticky="sw")

        self.fig = Figure(figsize=V.DESIGN, dpi=dpi)
        self.can = FigureCanvasTkAgg(self.fig, master=right)
        tb = NavigationToolbar2Tk(self.can, right); tb.update()
        self.can.mpl_connect("resize_event", lambda ev: tb.update())      # its history holds axes positions: Home after a resize would undo the zoom
        self.can.get_tk_widget().pack(fill="both", expand=True)
        self.can.get_tk_widget().bind("<Configure>", self.on_configure, add="+")
        self.fig.text(0.5, 0.5, HINTS["section"], ha="center", va="center", fontsize=14, color="0.45")
        self.can.draw()

        root.after(100, self.poll)
        self.submit("index")
        if selftest:
            kind, _, value = selftest[1].partition(":")
            self.kind.set(kind); self.alpha.set(f"{selftest[2]:g}"); self.re.set(f"{selftest[3]:g}")
            self.choose(kind, value)
        else:
            self.choose("naca", self.naca.get())

    # ------------------------------------------------------------------ background work
    def submit(self, task, *args):
        def work():
            try:
                self.jobs.put((task, args, self.TASKS[task](*args), None))
            except Exception as e:                                        # shown in the window, not lost in a thread
                self.jobs.put((task, args, None, e))
        threading.Thread(target=work, daemon=True).start()

    def progress(self, fraction, text):
        """wing_nll.compute's progress, called from the worker thread: it only queues (Tk is touched on the Tk thread)"""
        self.jobs.put(("progress", (), (fraction, text), None))

    def poll(self):
        try:
            while True:
                task, args, res, err = self.jobs.get_nowait()
                try:
                    getattr(self, "done_" + task)(args, res, err)
                except Exception as e:                                    # a result that cannot be shown must not stop the polling
                    traceback.print_exc()
                    self.idle()
                    self.status.configure(text=f"The result could not be shown: {e}")
                    if self.selftest:
                        self.finish(1)
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    # ------------------------------------------------------------------ airfoil
    def done_index(self, args, res, err):
        if err:
            self.note.configure(text=f"UIUC list unavailable ({err.__class__.__name__}); NACA and your own file still work")
            return
        self.index = res
        self.note.configure(text=f"{len(res)} airfoils")
        self.filter()

    def filter(self):
        q = self.search.get().strip().lower()
        self.shown = [(n, d) for n, d in self.index if q in n.lower() or q in d.lower()]
        self.lst.delete(0, "end")
        for n, d in self.shown:
            self.lst.insert("end", f"{n:<12s} {d}")

    def pick_uiuc(self):
        sel = self.lst.curselection()
        if sel:
            self.kind.set("uiuc")
            self.choose("uiuc", self.shown[sel[0]][0])

    def open_file(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(title="Airfoil coordinates", filetypes=[("Coordinate files", "*.dat *.txt *.csv"), ("All files", "*.*")])
        if path:
            self.kind.set("file")
            self.choose("file", path)

    def choose(self, kind, value):
        self.cutof = ""
        self.status.configure(text=f"Loading {value} ...")
        self.submit("section", kind, value)

    def done_section(self, args, res, err):
        if err:
            self.status.configure(text=f"Could not load {args[1]}: {err}")
            if self.selftest:
                self.finish(1)
            return
        self.section = res
        loop, name, family = res
        self.code, self.cutlabel = naca_code(name, family), None
        if self.experimental:
            self.origin = f"uiuc:{args[1]}" if args[0] == "uiuc" else args[1] if args[0] == "file" else None
            self.cur.set(f"current = {name} (the Airfoil box's section)")
            self.ranges()                                     # 'current' may be a NACA designation now, or no longer one
        self.pax.clear(); self.pax.axis("off")
        self.pax.fill(loop[:, 0], loop[:, 1], color="0.75", lw=0.8, ec="0.2"); self.pax.set_aspect("equal")
        self.pcan.draw_idle()
        self.pname.configure(text=f"{name}: {len(loop)} points")
        self.status.configure(text=f"{name} loaded. " + (self.sections_note() if self.mode() == "wing" else "Set the flow and press Run." + (
            "" if family == "naca" else " Not a NACA 4/5-digit section: the field decoder was trained on those only (see the warnings).")))
        if self.selftest:
            self.selftest_wing() if self.selftest[4] else self.run()

    # ------------------------------------------------------------------ run and show
    def run(self):
        if self.mode() == "wing":                                         # Run acts on the mode that is on
            return self.run_wing()
        if self.section is None or str(self.run_btn["state"]) == "disabled":
            return
        try:
            alpha, Re = float(self.alpha.get()), float(self.re.get())
        except ValueError:
            self.status.configure(text="The incidence and the Reynolds number must be numbers (e.g. 4 and 3e6).")
            return
        loop, name, family = self.section
        decoder = self.rext_path if self.rext.get() and family == "naca" else None
        note = (" The Re-extended decoder is for NACA sections; this one uses the released decoder."
                if self.rext.get() and family != "naca" else "")
        self.lock(True); self.bar.configure(mode="indeterminate"); self.bar.start(12)
        self.status.configure(text=f"Computing {name} at {alpha:g} deg, Re {Re:.3g} ... (the first run also loads the models)" + note)
        self.submit("run", loop, alpha, Re, DETAIL[self.detail.get()], name, family, decoder)

    def done_run(self, args, res, err):
        self.idle()
        if err:
            self.status.configure(text=f"The computation failed: {err}")
            traceback.print_exception(err)
            if self.selftest:
                self.finish(1)
            return
        (r, surf_w, out, meta), dt = res
        loop, alpha, Re, h, name, family, decoder = args
        self.args = (name, alpha, Re)
        self.last = (name, alpha, Re, r, surf_w, out, meta)
        n = len(surf_w) + len(meta["warnings"])
        out_of = [s for s, k in (("XFOIL state", "xfoil_in_range"), ("RANS state", "rans_in_range")) if not meta.get(k, True)]
        self.status.configure(text=f"{name}, {alpha:g} deg, Re {Re:.3g}: done in {dt:.1f} s"
                              + (f"; {n} warning(s), listed in the figure." if n else "; inside both training domains.")
                              + (f" The {' and the '.join(out_of)} {'is' if len(out_of) == 1 else 'are'} outside {'its' if len(out_of) == 1 else 'their'} training range: drawn faded." if out_of else "")
                              + self.cutof)
        self.draw(self.orientation())                         # after the status line: its height is the panel's, which sets the canvas's
        self.sync()
        if self.selftest:
            self.root.after(500, self.finish_selftest)

    def orientation(self):
        """The layout that shows the figure largest on the canvas as it is now."""
        w = self.can.get_tk_widget().winfo_width() / self.fig.dpi; h = self.can.get_tk_widget().winfo_height() / self.fig.dpi
        zoom = {k: min(w / W, h / H) for k, (W, H) in V.DESIGNS.items()}
        return max(zoom, key=zoom.get)

    def draw(self, layout):
        """The mode's last result in a side-by-side or stacked layout, frozen and zoomed to the canvas."""
        self.layout = layout
        self.drop_view()
        if self.mode() == "wing":
            self.view = WV.WingView(self.fig, self.wres, layout, dict(self.wopts))
            self.view.draw()
        else:
            V.figure(*self.last, fig=self.fig, layout=layout, states=self.states())
        self.on_resize = V.scale_with_window(self.fig)
        self.fit()

    def states(self):
        """The states the 2D figure's surface panels show (V.figure's `states`), from the two boxes."""
        return tuple(s for s, v in (("xfoil", self.show_x), ("rans", self.show_r)) if v.get())

    def states_changed(self, var):
        """A state's box: the figure shows the ticked states (redrawn, nothing recomputed). One stays ticked: the last box is ticked again."""
        if not (self.show_x.get() or self.show_r.get()):
            var.set(True)
            self.status.configure(text="At least one state stays ticked.")
        elif self.last is not None and self.mode() == "section":
            self.draw(self.orientation())

    def drop_view(self):
        """The wing figure's mouse handling goes before its figure is reused; its view (the turn) and cut are kept for the next one."""
        if self.view is not None:
            self.wopts["view"], self.wopts["cut_eta"] = self.view.pose, self.view.cut_eta
            self.view.disconnect(); self.view = None

    def blank(self, text):
        """An empty figure with a hint: the mode has no result yet."""
        self.drop_view()
        self.fig.clf()
        self.fig.text(0.5, 0.5, text, ha="center", va="center", fontsize=14, color="0.45")
        self.on_resize = V.scale_with_window(self.fig)
        self.fit()

    def mode(self):
        """'wing' while the 3D wing box is ticked, else 'section'."""
        return "wing" if self.wing.get() else "section"

    def result(self):
        """The mode's last result (None before its first run)."""
        return self.wres if self.mode() == "wing" else self.last

    def sync(self):
        """Save figure and Export data act on the mode's last result, so they are on only when it has one."""
        state = "normal" if self.result() is not None else "disabled"
        # ponytail: this also switches Export data (CSV) on while an export is being written, when the mode is switched or a case ends
        # meanwhile (a second or two, the same files); an `exporting` flag here if that ever matters. lock() leaves Export alone.
        self.save_btn.configure(state=state); self.csv_btn.configure(state=state)
        if self.experimental:
            self.cut_btn.configure(state="normal" if self.wres is not None else "disabled")
            self.sync3d()

    def sync3d(self):
        """The 3D view button is on with a wing result, and off while a wing or a page is made (Run reads Cancel then)."""
        self.view3d_btn.configure(state="normal" if self.wres is not None and str(self.run_btn["text"]) != "Cancel" else "disabled")

    def show(self):
        """The 3D wing box (also Open cut in 2D and the self-test): the Flow box shows the mode's rows and the figure the mode's
        last result (nothing is recomputed), or its hint when it has none yet."""
        on = self.mode() == "wing"
        (self.f3 if on else self.f2).grid(); (self.f2 if on else self.f3).grid_remove()
        # the wing's rows and their gray line need the height (a 14-inch Mac: the credit line would be cut off): in 3D the list is as short as on an upright screen
        self.lst.configure(height=9 if on or self.root.winfo_screenheight() > self.root.winfo_screenwidth() else 14)
        if not on:
            if self.v3win is not None:                        # the options of the 3D view belong to the 3D mode
                self.v3win.withdraw()
            self.stw.withdraw()
        else:
            self.status.configure(text=self.sections_note())
            self.ranges()                                     # (the Airfoil box's section may have changed while the box was off)
        self.sync()
        if self.result() is None:
            self.blank(HINTS[self.mode()])
        else:
            self.draw(self.orientation())

    def lock(self, on, wing=False):
        """A case runs (on): the 3D box is off limits, and Run is disabled, or for a wing it reads Cancel."""
        self.run_btn.configure(text="Cancel" if on and wing else "Run", command=self.cancel_run if on and wing else self.run,
                               state="disabled" if on and not wing else "normal")
        if self.experimental:
            self.chk.configure(state="disabled" if on else "normal")
            self.sync3d()                                     # (not sync(): that would switch Export data (CSV) back on while an export is written)

    def cancel_run(self):
        """Run reads Cancel while a wing runs. Not in its first 0.6 s: the second click of a double click on Run lands on Cancel."""
        if time.monotonic() - self.started > 0.6:
            self.cancel.set()

    def idle(self):
        """No case is running: the progress bar rests and Run is Run again."""
        self.bar.stop(); self.bar.configure(mode="determinate", value=0)
        self.lock(False)

    def on_configure(self, event):
        """A resize that makes the other layout the larger one redraws in it; other resizes only zoom."""
        if self.result() is None or self.orientation() == self.layout:
            return
        if self.relayout is not None:
            self.root.after_cancel(self.relayout)
        self.relayout = self.root.after(300, lambda: self.result() is None or self.draw(self.orientation()))   # (the mode may have changed)

    def fit(self):
        """Zoom the frozen layout to the canvas once the panel has its new size (later resizes do this through the resize handler).
        The panel's changes (the Flow box's rows, the status line) reach the canvas first, so the figure is drawn once, at its final
        size: by the canvas's resize when they resize it, else here (a draw asked for before them ran first, at the old size)."""
        pending = getattr(self.can, "_idle_draw_id", None)   # a draw still queued from before this figure was built (a resize's, the relayout's)
        if pending:                                           # would run first in the update, of the new figure unzoomed: the one draw below replaces it
            self.can.get_tk_widget().after_cancel(pending); self.can._idle_draw_id = None
        resized = []
        cid = self.can.mpl_connect("resize_event", resized.append)
        self.root.update_idletasks()
        self.can.mpl_disconnect(cid)
        if not resized:
            w = self.can.get_tk_widget().winfo_width(); h = self.can.get_tk_widget().winfo_height()
            if w > 1 and h > 1:
                self.fig.set_size_inches(w / self.fig.dpi, h / self.fig.dpi, forward=False)
            self.on_resize()
        self.can.toolbar.update()                             # the figure was rebuilt: the toolbar's history (mplot3d fills it) holds the old axes

    def save(self):
        from tkinter import filedialog
        if self.mode() == "wing":
            stem = wing_base(self.wres)
        else:
            name, alpha, Re = self.args
            stem = re.sub(r"[^A-Za-z0-9.+-]+", "_", f"vortfoil_{name}_a{alpha:g}_Re{Re:.3g}")
        path = filedialog.asksaveasfilename(title="Save figure", defaultextension=".png", initialfile=stem + ".png",
                                            filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if path:
            try:
                self.save_to(path)
            except (OSError, ValueError) as e:                # an unwritable place, an unknown file type
                self.status.configure(text=f"Could not save {path}: {e}")

    def export(self):
        """Every number of the result as CSV files in a chosen folder (written in the background)."""
        from tkinter import filedialog
        if self.mode() == "wing":
            return self.export_wing()
        folder = filedialog.askdirectory(title="Folder for the CSV files", mustexist=False)
        if folder:
            self.status.configure(text=f"Writing the CSV files to {folder} ...")
            self.csv_btn.configure(state="disabled")
            self.submit("export", folder, *self.last)

    def done_export(self, args, res, err):
        self.sync()
        self.status.configure(text=f"Could not write the CSV files: {err}" if err else
                              f"Wrote {len(res)} files to {args[0]}: coefficients, surface C_p, boundary layer, field, wall, RANS boundary layer, section, about")

    def save_to(self, path):
        """The figure at its design size and 400 dpi, whatever the window size."""
        w, h = self.fig.get_size_inches()
        self.fig.set_size_inches(*V.DESIGNS[self.layout], forward=False); self.on_resize()
        try:
            self.fig.savefig(path, dpi=400)
        finally:
            self.fig.set_size_inches(w, h, forward=False); self.on_resize()
        self.status.configure(text=f"Saved {path}")

    # ------------------------------------------------------------------ 3D wing
    def wing_rows(self, f, wrap):
        """The Flow box in 3D: planform (and its geometry table), speed and incidence, section state, the gray line of what these inputs must
        respect (ranges(); wrapped at `wrap` like the 2D box's note, wider where the box is), what the figure shows (redrawn, never
        recomputed), polar and Open cut in 2D."""
        tk, ttk = self.tk, self.ttk
        self.planform, self.speed, self.walpha = tk.StringVar(value=self.wname), tk.StringVar(), tk.StringVar(value="6")
        self.wstate, self.range_note = tk.StringVar(value="xfoil"), tk.StringVar()
        self.speed.trace_add("write", self.ranges); self.wstate.trace_add("write", self.ranges)       # the line follows the speed and the chosen state
        self.surface, self.span, self.polar = (tk.StringVar(value=self.wopts[k]) for k in ("surface", "span", "polar"))
        f.columnconfigure(0, weight=1)

        def row(i):
            r = ttk.Frame(f); r.grid(row=i, column=0, sticky="ew", pady=(0 if i == 0 else 3, 0)); return r

        def label(r, text):
            ttk.Label(r, text=text).pack(side="left", padx=(0, 4))

        def entry(r, text, var):
            label(r, text)
            e = ttk.Entry(r, textvariable=var, width=6); e.pack(side="left", padx=(0, 10)); e.bind("<Return>", lambda ev: self.run())

        def still(c):                                                     # the wheel over a closed list must not pick an item (the planform's would replace the table)
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                c.bind(seq, lambda ev: "break")
            return c

        def choice(r, text, var, values, width, key):
            label(r, text)
            c = still(ttk.Combobox(r, textvariable=var, values=values, state="readonly", width=width)); c.pack(side="left", padx=(0, 10))
            c.bind("<<ComboboxSelected>>", lambda ev: self.wing_option(key, var.get()))
            return c

        r = row(0); label(r, "Planform")
        ttk.Button(r, text="Geometry ...", command=self.open_stations).pack(side="right")
        c = still(ttk.Combobox(r, textvariable=self.planform, values=list(WG.PRESETS), state="readonly", width=20))
        c.pack(side="left", fill="x", expand=True, padx=(0, 6)); c.bind("<<ComboboxSelected>>", lambda ev: self.pick_planform())
        r = row(1); entry(r, "Speed (m/s)", self.speed); entry(r, "Incidence (deg)", self.walpha)
        r = row(2); label(r, "Section model")                             # read at Run: a shown result keeps its own state
        for v, t in (("xfoil", "XFOIL state"), ("sst", "RANS state")):
            ttk.Radiobutton(r, text=t, value=v, variable=self.wstate).pack(side="left", padx=(0, 10))
        r = row(3)
        note = ttk.Label(r, textvariable=self.range_note, foreground="gray", wraplength=wrap, justify="left", width=1); note.pack(fill="x")
        note.bind("<Configure>", lambda ev: note.configure(wraplength=max(wrap, f.winfo_width() - 8)))     # the box's width where it is wider than 3.4 in (a 14-inch Mac: 373 px, not 287)
        # width=1: the other rows set the box's width, never this text (it widened the box it wrapped at: ten redraws a tick). The box's width, not
        # the event's: after a tick the line's own event still carries the 2D box's width when the box already has the 3D rows' (one more redraw)
        r = row(4)
        self.surface_box = choice(r, "Surface", self.surface, WV.SURFACE_QUANTITIES["xfoil"], 8, "surface")
        choice(r, "Spanwise", self.span, WV.SPAN_QUANTITIES, 11, "span")
        r = row(5); choice(r, "Polar", self.polar, WV.POLAR_PLOTS, 12, "polar")
        self.cut_btn = ttk.Button(r, text="Open cut in 2D", command=self.open_cut, state="disabled"); self.cut_btn.pack(side="left")
        self.view3d_btn = ttk.Button(r, text="3D view", command=self.open_view3d, state="disabled"); self.view3d_btn.pack(side="left", padx=(6, 0))
        self.v3win, self.pages, self.pages_of = None, {}, None   # its options window (built at the first click); the pages made for one result

    def stations_window(self):
        """The Wing geometry window: modeless, built once and only hidden, so its values survive. Its Text is the only store of the
        wing's stations (Run reads it); typing in it makes the planform Custom."""
        tk, ttk = self.tk, self.ttk
        top = self.stw = tk.Toplevel(self.root); top.withdraw(); top.title("Wing geometry"); top.transient(self.root)
        tall = self.root.winfo_screenheight() > self.root.winfo_screenwidth()   # over the figure's far corner: the Airfoil box, whose
        top.geometry("+0-0" if tall else "-0+0")                                  # section the buttons below copy, stays in view
        top.protocol("WM_DELETE_WINDOW", top.withdraw); top.bind("<Escape>", lambda ev: top.withdraw())
        body = ttk.Frame(top, padding=8); body.pack(fill="both", expand=True)
        wl = int(4.2 * self.root.winfo_fpixels("1i"))
        ttk.Label(body, justify="left", text=GEOMETRY_TEXT, wraplength=wl).pack(anchor="w")
        self.cur = tk.StringVar(value="current = (no airfoil loaded yet)")
        ttk.Label(body, textvariable=self.cur, foreground="gray").pack(anchor="w", pady=(4, 0))
        f = ttk.Frame(body); f.pack(fill="both", expand=True, pady=(4, 0))
        self.stations = tk.Text(f, width=56, height=6, wrap="none", font="TkFixedFont", undo=True)   # Ctrl+Z takes back a section button
        sb = ttk.Scrollbar(f, orient="vertical", command=self.stations.yview); self.stations.configure(yscrollcommand=sb.set)
        self.stations.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y")
        self.stations.bind("<<Modified>>", self.edit_stations)
        ttk.Label(body, justify="left", foreground="gray", text=SECTION_HINT, wraplength=wl).pack(anchor="w", pady=(4, 0))
        sec = ttk.Frame(body); sec.pack(fill="x", pady=(4, 0))                  # the Airfoil box's section on rows of the table
        ttk.Label(sec, text="Airfoil box's section on:").pack(side="left", padx=(0, 6))
        ttk.Button(sec, text="Selected row(s)", command=lambda: self.set_sections(False)).pack(side="left")
        ttk.Button(sec, text="All rows", command=lambda: self.set_sections(True)).pack(side="left", padx=6)
        self.scale, self.strips = tk.StringVar(value="1"), tk.StringVar(value="40")
        for v in (self.scale, self.strips):                   # (the speeds scale with the size, and are the strips')
            v.trace_add("write", self.ranges)
        bt = ttk.Frame(body); bt.pack(fill="x", pady=(6, 0))
        for text, var, width in (("Scale x", self.scale, 5), ("Strips", self.strips, 4)):
            ttk.Label(bt, text=text).pack(side="left", padx=(0, 4))
            e = ttk.Entry(bt, textvariable=var, width=width); e.pack(side="left", padx=(0, 10)); e.bind("<Return>", lambda ev: self.run())
        ttk.Button(bt, text="Close", command=top.withdraw).pack(side="right")
        ttk.Button(bt, text="Save ...", command=self.save_wing).pack(side="right", padx=6)
        ttk.Button(bt, text="Load ...", command=self.load_wing).pack(side="right")

    def open_stations(self):
        self.stw.deiconify(); self.stw.lift()

    def fill_wing(self, d):
        """A wing definition (a preset, or a file's) into the Wing geometry window and the Flow box. A preset whose stations all carry one
        NACA section gets the Airfoil box's section ('current') on every station; returns whether it did."""
        preset = WG.PRESETS.get(d.get("name")) is d
        st = d["stations"]
        cur = preset and len({s["section"] for s in st}) == 1 and WG.is_naca(st[0]["section"])
        if cur:
            st = [dict(s, section="current") for s in st]
        self.stations.delete("1.0", "end"); self.stations.insert("1.0", format_stations(st))
        self.stations.mark_set("insert", "1.0")               # (the cursor is on the header, not on the last row: Selected row(s) takes the row that was clicked)
        self.stations.edit_modified(False)                    # filling is not editing (the <<Modified>> event comes later)
        self.planform.set(d["name"] if preset else "Custom")
        self.wname, self.wdesc = d.get("name", "Custom wing"), d.get("description", "")
        self.scale.set("1"); self.strips.set(str(int(d.get("n", 40))))
        if "V" in d:
            self.speed.set(f"{d['V']:g}")
        self.ranges()
        return cur

    def pick_planform(self):
        self.fill_wing(WG.PRESETS[self.planform.get()])
        self.status.configure(text=self.sections_note() + " Press Run.")

    def sections_note(self):
        """Which airfoil the wing in the geometry table has, for the status line: a one-section planform takes the Airfoil box's
        ('current' in its table), the others have their own."""
        try:
            toks = [t["section"] for t in parse_stations(self.stations.get("1.0", "end"))]
        except ValueError as e:
            return f"The geometry table has a mistake: {e}."
        if not toks:
            return "The geometry table is empty."
        box = self.sname() or "section (none loaded yet)"
        if set(toks) == {"current"}:
            return f"{self.wname} with the Airfoil box's {box} on every station; load another airfoil to change it."
        names = list(dict.fromkeys(box if t == "current" else section_name(t) for t in toks))
        return (f"{self.wname}: " + (" to ".join(names) if len(names) < 4 else f"{len(names)} sections")
                + (f" (its own sections; 'current' is the Airfoil box's {box})" if "current" in toks else " (its own sections)"))

    def sname(self):
        """The Airfoil box's section as the wing names it (the section a wing's cut is the cut of, not the cut's own 2D name), or None."""
        return self.cutlabel or (self.section[1] if self.section else None)

    def edit_stations(self, event=None):
        """Typing in the stations makes the wing a custom one (and its ranges follow)."""
        if self.stations.edit_modified():
            self.stations.edit_modified(False)
            self.planform.set("Custom"); self.wname, self.wdesc = "Custom wing", ""
            self.ranges()

    def ranges(self, *args):
        """The gray line of the 3D Flow box: what the wing's inputs must respect, from wing_nll.input_ranges (numbers only, no model runs) on
        the wing Run computes (wing_definition: its strips, its size). It follows the planform, the geometry table, Scale x, Strips, the
        speed, the Airfoil box's section and the chosen state (a Tk trace passes arguments, ignored here). Inputs that cannot be read give
        a short note, never an exception (it runs in Tk traces and in done_section); a speed that is not a number is not judged."""
        try:
            V = float(self.speed.get())
        except ValueError:
            V = None
        V = V if V is not None and 0 < V < float("inf") else None
        try:
            r = WG.input_ranges(wing_definition(self.stations.get("1.0", "end"), self.scale.get(), self.strips.get()), V=V,
                                current_is_naca=self.code is not None)
        except ValueError as e:
            return self.range_note.set(f"Ranges not shown ({e}).")
        except ArithmeticError:                               # lengths no float carries (Scale x 1e-320): Run says the same
            return self.range_note.set("Ranges not shown (the size is out of range).")
        say, s = dict(r["sentences"]), r["sst" if self.wstate.get() == "sst" else "xfoil"]
        if s["inside"] is False:                              # the state Run will use, said right after the speed's verdict
            say["at_V"] += f" The {s['label']} you chose does not fit this speed."
        self.range_note.set(" ".join(t for t in say.values() if t))

    def set_sections(self, everywhere):
        """The Airfoil box's section on rows of the geometry table: the selected rows (the cursor's row when none is selected), or every row. It
        writes the section's own name (a NACA designation, uiuc:<name> or the file's path, never 'current'), and only in the section column."""
        token, t = self.code or self.origin, self.stations
        if token is None:
            return self.status.configure(text="Load a section in the Airfoil box first (NACA, UIUC or your file)." if self.section is None else
                                         f"The Airfoil box's section ({self.section[1]}) is a wing's cut: no name a station can use. Load a NACA, UIUC or file section.")
        first = last = None
        if not everywhere:
            sel = t.tag_ranges("sel")
            a, b = (str(sel[0]), str(sel[-1])) if sel else (str(t.index("insert")),) * 2
            first, last = int(a.split(".")[0]), int(b.split(".")[0])
            if last > first and b.endswith(".0"):             # a selection that ends at the start of a line does not take that line
                last -= 1
        rows = section_rows(t.get("1.0", "end"), token, first, last)
        if not rows:
            return self.status.configure(text="No station row there: click a row of the table (or select several), then press Selected row(s).")
        cursor = t.index("insert")
        for n, line in rows.items():                          # only these lines change: the view and the other rows stay
            t.delete(f"{n}.0", f"{n}.end"); t.insert(f"{n}.0", line)
        t.mark_set("insert", cursor)
        self.edit_stations()                                  # the wing is a custom one now; Tk delivers its <<Modified>> later
        self.status.configure(text=f"Set the section of {len(rows)} row{'s' if len(rows) > 1 else ''} to {token}. {self.sections_note()}")

    def load_wing(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(parent=self.stw, title="Wing definition", filetypes=[("Wing definition (JSON)", "*.json"), ("All files", "*.*")])
        if path:
            try:
                d = WG.load_wing(path)
            except (OSError, ValueError) as e:
                self.status.configure(text=f"Could not load {os.path.basename(path)}: {e}")
                return
            self.fill_wing(d)
            self.status.configure(text=f"Loaded {path}. {self.sections_note()} Press Run.")

    def save_wing(self):
        """The form's wing (lengths scaled, speed included) as a wing definition file."""
        from tkinter import filedialog
        try:
            d = wing_definition(self.stations.get("1.0", "end"), self.scale.get(), self.strips.get(), self.wname, self.wdesc)
        except ValueError as e:
            self.status.configure(text=f"The wing cannot be saved: {e}")
            return
        path = filedialog.asksaveasfilename(parent=self.stw, title="Save wing definition", defaultextension=".json",
                                            filetypes=[("Wing definition (JSON)", "*.json")], initialfile=re.sub(r"(?:[^\w.+-]|_)+", "_", d["name"]) + ".json")
        if path:
            if self.wname == "Custom wing":                   # a custom wing is named after its file
                d["name"] = os.path.splitext(os.path.basename(path))[0]
            try:
                d["V"] = float(self.speed.get())
            except ValueError:
                pass                                          # no speed in the file
            note = ""
            if any(t["section"] == "current" for t in d["stations"]):     # the file must not depend on what the Airfoil box holds later
                if self.code:
                    for t in d["stations"]:
                        t["section"] = self.code if t["section"] == "current" else t["section"]
                else:
                    note = f"; its 'current' sections mean the Airfoil box's ({self.sname() or 'none loaded'}) when it is loaded"
            try:
                WG.save_wing(d, path)
                self.status.configure(text=f"Saved the wing {d['name']} to {path}" + note)
            except (OSError, ValueError) as e:
                self.status.configure(text=f"Could not save {path}: {e}")

    def refuse(self, text):
        """A run that cannot start: the reason in the status line (and the self-test ends)."""
        self.status.configure(text=text)
        if self.selftest:
            print("SELFTEST FAILED:", text, flush=True)
            self.finish(1)

    def run_wing(self):
        """Run in 3D: the stations, scale, strips, speed and incidence of the forms in the chosen section state (in the background)."""
        if str(self.run_btn["text"]) == "Cancel":                         # a wing is running: Run reads Cancel, Return does nothing
            return
        try:
            speed, alpha = float(self.speed.get()), float(self.walpha.get())
        except ValueError:
            return self.refuse("The speed and the incidence must be numbers (e.g. 45 and 6).")
        if not (speed > 0 and abs(alpha) <= 30):
            return self.refuse("The speed must be positive and the incidence between -30 and 30 deg.")
        text, name = self.stations.get("1.0", "end"), self.wname
        try:
            if self.section and all(s["section"] == "current" for s in parse_stations(text)):
                name += f", {self.sname()}"                               # the Airfoil box's section is in the title and the file names
            defn = wing_definition(text, self.scale.get(), self.strips.get(), name, self.wdesc)
        except ValueError as e:
            return self.refuse(f"The wing cannot be computed: {e}")
        state = self.wstate.get()
        self.cancel.clear(); self.lock(True, wing=True); self.started = time.monotonic()
        self.determinate = state == "sst"                     # the RANS state reports its progress; the XFOIL state takes seconds
        self.bar.configure(mode="determinate" if self.determinate else "indeterminate", value=0)
        if not self.determinate:
            self.bar.start(12)
        self.wbase = f"Computing {defn['name']} ({STATE_LABEL[state]}) at {alpha:g} deg, {speed:g} m/s ..."
        self.status.configure(text=self.wbase + " (the first run also loads the models)")
        current = (self.section[0], self.code, self.sname()) if self.section else None
        self.submit("wing", defn, speed, alpha, state, current, self.progress, self.cancel.is_set)

    def done_progress(self, args, res, err):
        fraction, text = res
        if self.determinate:
            self.bar.configure(value=100 * fraction)
        self.status.configure(text=f"{self.wbase} {text}")

    def done_wing(self, args, res, err):
        self.idle()
        if err:
            if self.cancel.is_set():
                self.status.configure(text="Cancelled." + (" The last wing result is kept." if self.wres is not None else ""))
            else:
                why = ("the speed or the size is out of range." if isinstance(err, OverflowError)
                       else reason(err, "close other programs and press Run again"))
                self.status.configure(text="The wing computation failed: " + why)
                if not isinstance(err, (ValueError, OverflowError)):      # those are numbers or a section the user can fix: no trace
                    traceback.print_exception(err)
            if self.selftest:
                print("SELFTEST FAILED:", self.status.cget("text"), flush=True)
                self.finish(1)
            return
        out, dt = res
        self.wres = out
        vals = WV.SURFACE_QUANTITIES[out["state"]]            # the Surface choices are the result's state's
        self.surface_box.configure(values=vals)
        if self.surface.get() not in vals:
            self.surface.set("C_p")
        self.wopts["surface"] = self.surface.get()
        self.status.configure(text=wing_summary(out, dt))     # before the figure: the status line's height is the panel's, which sets the canvas's
        self.draw(self.orientation())
        self.sync()
        if self.v3win is not None:                            # the 3D view's options (a modeless window) follow the new result
            self.view3d_floor()
        if self.selftest:
            self.root.after(500, self.finish_selftest)

    def wing_option(self, key, value):
        """A display choice redraws the wing figure; nothing is recomputed."""
        self.wopts[key] = value
        if self.view is not None:
            self.view.set_option(key, value)
            if tuple(self.fig.get_size_inches()) == V.DESIGNS[self.layout]:   # rebuilt at the design size: freeze it anew for the zoom
                self.on_resize = V.scale_with_window(self.fig)
                self.fit()

    def open_cut(self):
        """The section at the cut into the Airfoil box, analysed in 2D at the strip's own incidence and Reynolds number."""
        res, view = self.wres, self.view
        if res is None or view is None or str(self.run_btn["text"]) == "Cancel":
            return
        i, lp, st = view.cut_index, res["loops"], res["strips"]
        j = int(lp["index"][i])
        label = lp["label"][j]
        m = re.search(r"blend [0-9.]+$", label)
        name = f"{m.group() if m else WV.short(label, 11)}, 2y/b {view.cut_eta:.2f}"     # short: a long 2D header squeezes the figure's panels
        self.done_section(("cut", name), (lp["coords"][j], name, lp["family"][j]), None)
        self.code, self.cutlabel, self.cutof = naca_code(label, lp["family"][j]), label, f" From the wing {res['wing']['name']}."
        self.ranges()                                         # done_section judged the cut's 2D name, no designation: the hidden gray line was wrong and changed at the next tick (redraws)
        self.alpha.set(f"{st['alpha_eff_deg'][i]:.4g}"); self.re.set(f"{st['Re'][i]:.4g}")
        self.wing.set(False); self.show()
        self.run()

    def export_wing(self):
        """Which parts of the wing's data to write: a small dialog (all ticked), then the folder; written in the background."""
        from tkinter import filedialog
        tk, ttk = self.tk, self.ttk
        top = tk.Toplevel(self.root); top.title("Export data (CSV)"); top.transient(self.root)
        body = ttk.Frame(top, padding=10); body.pack(fill="both", expand=True)
        ttk.Label(body, text="Parts to write (one CSV file each; the text that explains the columns is always written):").pack(anchor="w")
        ticks = {k: tk.BooleanVar(value=True) for k in WV.CSV_PARTS}
        for k, (suffix, text) in WV.CSV_PARTS.items():
            ttk.Checkbutton(body, text=f"{suffix}: {text}", variable=ticks[k]).pack(anchor="w")
        res, cut = self.wres, (self.view.cut_index if self.view else None)

        def go():
            parts = [k for k, v in ticks.items() if v.get()]
            top.destroy()
            if not parts:
                self.status.configure(text="No part ticked: nothing written.")
                return
            folder = filedialog.askdirectory(title="Folder for the CSV files", mustexist=False)
            if folder:
                self.status.configure(text=f"Writing the CSV files to {folder} ...")
                self.csv_btn.configure(state="disabled")
                self.submit("wing_export", folder, res, parts, cut)
        bt = ttk.Frame(body); bt.pack(anchor="w", pady=(8, 0))
        ttk.Button(bt, text="All", command=lambda: [v.set(True) for v in ticks.values()]).pack(side="left")
        ttk.Button(bt, text="None", command=lambda: [v.set(False) for v in ticks.values()]).pack(side="left", padx=6)
        ttk.Button(bt, text="Export ...", command=go).pack(side="left", padx=(12, 6))
        ttk.Button(bt, text="Cancel", command=top.destroy).pack(side="left")

    def done_wing_export(self, args, res, err):
        self.sync()
        self.status.configure(text=f"Could not write the CSV files: {err}" if err else
                              f"Wrote {len(res)} files to {args[0]}: {', '.join(args[2])}, about")

    # ------------------------------------------------------------------ 3D view in the browser
    def open_view3d(self):
        """3D view: what the page shows, in a small window (modeless, built at the first click and only hidden after, so the
        last choice is kept); [Open] makes the page."""
        if self.wres is None or str(self.run_btn["text"]) == "Cancel":
            return
        if self.v3win is None:
            tk, ttk = self.tk, self.ttk
            top = self.v3win = tk.Toplevel(self.root); top.withdraw(); top.title("3D view"); top.transient(self.root)
            top.geometry("+40+40")
            top.protocol("WM_DELETE_WINDOW", top.withdraw); top.bind("<Escape>", lambda ev: top.withdraw())
            body = ttk.Frame(top, padding=10); body.pack(fill="both", expand=True)
            ttk.Label(body, justify="left", text="Opens the wing in your web browser: drag to turn it, wheel to zoom.\n"
                      "The flow field is computed at the wing's incidence only.").pack(anchor="w")
            self.v3field, self.v3free = tk.StringVar(value="viscous"), tk.BooleanVar(value=False)
            ttk.Label(body, text="Flow field:").pack(anchor="w", pady=(8, 0))
            for value, text in VIEW3D_FIELDS:
                rb = ttk.Radiobutton(body, text=text, value=value, variable=self.v3field); rb.pack(anchor="w", padx=(12, 0))
                if value == "viscous":
                    self.v3visc, self.v3note = rb, ttk.Label(body, text="", foreground="gray", wraplength=360)   # the note goes under it when it is off
            ttk.Checkbutton(body, text="Rolled-up wake (free wake; slower)", variable=self.v3free).pack(anchor="w", pady=(8, 0))
            bt = ttk.Frame(body); bt.pack(anchor="e", pady=(10, 0))
            ttk.Button(bt, text="Open", command=self.run_view3d).pack(side="left")
            ttk.Button(bt, text="Cancel", command=top.withdraw).pack(side="left", padx=(6, 0))
        self.view3d_floor()
        self.v3win.deiconify(); self.v3win.lift()

    def view3d_floor(self):
        """The options window follows the shown wing result (at each click, and when a result arrives while the window is up): for
        a wing with a strip below the field decoder's Reynolds floor (wing_flow.re_floor) the viscous field is greyed out with the
        reason, and the choice moves to the inviscid one."""
        import wing_flow as WF
        floor, low = WF.re_floor(self.wres), float(np.min(self.wres["strips"]["Re"]))
        if low < floor:
            e = lambda x: f"{x:.2g}".replace("e+0", "e").replace("e+", "e")
            self.v3visc.configure(state="disabled")
            self.v3note.configure(text=f"Not for this wing: the viscous field needs Re >= {e(floor)} on every strip (this one goes down to Re {e(low)}).")
            self.v3note.pack(anchor="w", padx=(30, 0), after=self.v3visc)
            if self.v3field.get() == "viscous":
                self.v3field.set("inviscid")
        else:
            self.v3visc.configure(state="normal"); self.v3note.pack_forget()

    def run_view3d(self):
        """[Open]: the page of the shown wing result with the chosen options, made in the background (determinate progress and
        Cancel, as for a wing) and opened in the default browser. A page already made for this result and these options is
        opened again; a new wing result starts afresh."""
        self.v3win.withdraw()
        res = self.wres
        if res is None:
            return
        if str(self.run_btn["text"]) == "Cancel":                         # a case started while the window was open
            self.status.configure(text="A case is running: press 3D view again when it has finished.")
            return
        field, free = self.v3field.get(), self.v3free.get()
        if self.pages_of is not res:
            self.pages, self.pages_of = {}, res
        path = self.pages.get((field, free))
        if path and os.path.exists(path):
            return self.show_page(path)
        self.cancel.clear(); self.lock(True, wing=True); self.started = time.monotonic()    # (as in run_wing: Cancel ignores a double click's 2nd click)
        self.determinate = True
        self.bar.configure(mode="determinate", value=0)
        what = "no flow field" if field == "none" else f"flow field {field}"
        self.wbase = f"Preparing the 3D view of {res['wing']['name']}: {what}{', rolled-up wake' if free else ''} ..."
        self.status.configure(text=self.wbase)
        self.submit("wing3d", res, field, free, page_path(res, field, free), self.progress, self.cancel.is_set)

    def done_wing3d(self, args, res, err):
        self.idle()
        if err:
            if self.cancel.is_set():
                self.status.configure(text="Cancelled. The 3D view was not made.")
            else:
                self.status.configure(text=f"The 3D view failed: {reason(err, 'close other programs, or choose Inviscid streamlines or None')}")
                if not isinstance(err, ValueError):
                    traceback.print_exception(err)
            return
        path, dt = res
        self.pages[(args[1], args[2])] = path
        self.show_page(path, dt)

    def show_page(self, path, dt=None):
        """The page in the default browser (started here, on the Tk thread), and where it is in the status line."""
        import pathlib, webbrowser
        try:
            opened = webbrowser.open(pathlib.Path(path).as_uri())
        except Exception:                                     # no browser launcher on this system
            opened = False
        if dt is None:                                        # a page made before
            text = (f"3D view opened again in your browser: {path}" if opened else f"No browser could be started; open this file in one: {path}")
        else:
            text = (f"3D view ready in {dt:.1f} s and opened in your browser: {path}" if opened else
                    f"3D view ready in {dt:.1f} s. No browser could be started; open this file in one: {path}")
        self.status.configure(text=text)

    # ------------------------------------------------------------------ self-test
    def selftest_wing(self):
        """--selftest --wing: the 3D mode filled from the preset or file and run, as a person would."""
        _, _, alpha, _, wing, speed, state = self.selftest
        try:
            d = pick_wing(wing)
        except (OSError, ValueError) as e:
            return self.refuse(str(e))
        self.wing.set(True); self.show(); self.fill_wing(d)
        if speed:
            self.speed.set(f"{speed:g}")
        self.walpha.set(f"{alpha:g}"); self.wstate.set(state)
        self.run_wing()

    def finish_selftest(self):
        try:
            self.write_selftest()
        except Exception as e:                                # a failing write must end the self-test, not leave the window open
            traceback.print_exc()
            print("SELFTEST FAILED:", e, flush=True)
            return self.finish(1)
        self.finish(0)

    def write_selftest(self):
        out, done = self.selftest[0], self.status.cget("text")
        os.makedirs(out, exist_ok=True)
        self.fig.savefig(os.path.join(out, "app_canvas.png"), dpi=self.fig.dpi)   # the canvas as shown, at its size
        cw, ch = self.can.get_tk_widget().winfo_width(), self.can.get_tk_widget().winfo_height()
        print(f"window {self.root.winfo_width()}x{self.root.winfo_height()}, canvas {cw}x{ch}, {self.fig.dpi:.0f} dpi, layout {self.layout}", flush=True)
        self.save_to(os.path.join(out, "app_figure.png"))
        if self.mode() == "wing":
            WV.export_csv(os.path.join(out, "csv"), self.wres, list(WV.CSV_PARTS), self.view.cut_index)
            self.status.configure(text=done)                  # the wing's self-test reports its run (the 2D one prints save_to's line)
        else:
            V.export_csv(os.path.join(out, "csv"), *self.last)
        try:
            from PIL import ImageGrab
            self.root.update()
            x, y, w, h = self.root.winfo_rootx(), self.root.winfo_rooty(), self.root.winfo_width(), self.root.winfo_height()
            ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(os.path.join(out, "app_window.png"))
        except Exception as e:                                            # no screen capture on this platform
            print("window capture skipped:", e)
        print("SELFTEST OK", self.status.cget("text"), flush=True)

    def finish(self, code):
        self.exit_code = code
        self.root.after(200, self.root.destroy)


# ---------------------------------------------------------------------------------------------- the 3D view in the browser
PAGES = os.path.join(os.path.dirname(CACHE), "wing3d")      # <cache>/vortfoil/wing3d: the pages of the 3D view
VIEW3D_FIELDS = (("viscous", "Viscous (RANS field decoder)"), ("inviscid", "Inviscid streamlines"), ("none", "None"))


def page_path(res, field, free_wake):
    """where the page of a wing result with these options is written: <cache>/vortfoil/wing3d/<wing_base>_<field>[_fw].html"""
    return os.path.join(PAGES, f"{wing_base(res)}_{field}{'_fw' if free_wake else ''}.html")


def wing3d_page(res, field, free_wake, path, progress=None, cancel=None):
    """the standalone page of a wing result (code/wing_flow.py: the flow field, the wake and the page's data; code/wing_view.html: the
    page), written to `path`; (path, seconds). progress(fraction, text) and cancel() are wing_flow's, called from the worker thread."""
    import wing_flow as WF
    t0 = time.perf_counter()
    WF.write_page(WF.view_data(res, field, free_wake, progress, cancel), path)
    return path, time.perf_counter() - t0


App.TASKS["wing3d"] = wing3d_page


def make_shortcut(experimental=False):
    """A launcher for this Python environment and this checkout; returns what was done. On macOS nothing is written:
    the application is started from the terminal, and the command (and a shell alias for it) is returned. experimental: a
    separate launcher, "VortFoil (experimental)", whose window has the 3D wing (--experimental, the vortfoil_exp command)."""
    app = os.path.abspath(__file__)
    root = os.path.dirname(HERE)
    name, flag, cmd_name = ("VortFoil (experimental)", " --experimental", "vortfoil_exp") if experimental else ("VortFoil", "", "vortfoil")
    if sys.platform == "win32":                               # VortFoil.lnk on the desktop, through pythonw (no console)
        pyw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
        pyw = pyw if os.path.exists(pyw) else sys.executable
        ps = (f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut([IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), '{name}.lnk')); "
              f"$s.TargetPath = '{pyw}'; $s.Arguments = '\"{app}\"{flag}'; $s.WorkingDirectory = '{root}'; $s.IconLocation = '{pyw},0'; "
              "$s.Description = 'VortFoil: airfoil field and surface results'; $s.Save(); $s.FullName")
        return subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, check=True).stdout.strip()
    if sys.platform == "darwin":                              # the terminal is the launcher: print, write nothing
        import shutil
        if shutil.which(cmd_name):
            return f"no file written on macOS; the {cmd_name} command is installed: type {cmd_name} in a terminal"
        cmd = f'"{sys.executable}" "{app}"{flag}'
        return (f"no file written on macOS; start it from the terminal with\n  {cmd}\n"
                f"or add this line to ~/.zshrc and type {cmd_name}:\n  alias {cmd_name}='{cmd}'")
    else:                                                     # an entry in the applications menu
        path = os.path.expanduser(f"~/.local/share/applications/{'vortfoil-exp' if experimental else 'vortfoil'}.desktop")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(f'[Desktop Entry]\nType=Application\nName={name}\nComment=Airfoil field and surface results\n'
                    f'Exec="{sys.executable}" "{app}"{flag}\nPath={root}\nTerminal=false\nCategories=Science;Education;\n')
    os.chmod(path, 0o755)
    return path


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shortcut", action="store_true", help="a launcher for this Python environment (macOS: print the terminal command) and exit")
    ap.add_argument("--selftest", metavar="DIR", help="run one case unattended, save the figure and a window capture in DIR")
    ap.add_argument("--airfoil", default="naca:2412", help="for --selftest: naca:<digits>, uiuc:<name> or file:<path>")
    ap.add_argument("--alpha", type=float, default=4.0, help="for --selftest: the incidence in deg (the airfoil's, or the wing's)")
    ap.add_argument("--Re", type=float, default=3e6, help="for --selftest: the airfoil's Reynolds number")
    ap.add_argument("--wing", metavar="PRESET_OR_JSON", help="for --selftest with --experimental: run this wing (a preset name, or part of one, or a wing "
                    "definition .json) instead of the airfoil; the CSV files of all its parts are written")
    ap.add_argument("--V", type=float, help="for --selftest --wing: the flow speed in m/s (default: the wing's own)")
    ap.add_argument("--state", choices=("xfoil", "sst"), default="xfoil", help="for --selftest --wing: the section state (default %(default)s)")
    ap.add_argument("--experimental", action="store_true", help="the window with the experimental 3D wing (what vortfoil_exp opens; "
                    "the vortfoil command refuses it); with --shortcut, a separate launcher for it")
    return ap


def main(argv=None, experimental=None):
    """experimental: None from a checkout (--experimental decides), False for the vortfoil command (2D only: it refuses
    --experimental, however abbreviated), True for vortfoil_exp"""
    ap = make_parser()
    a = ap.parse_args(argv)
    if experimental is False and a.experimental:
        ap.error("--experimental: the vortfoil command is 2D only; the experimental 3D wing is vortfoil_exp")
    a.experimental = a.experimental or bool(experimental)
    if a.wing and not a.experimental:                         # vortfoil's window is 2D only
        ap.error("--wing: the 3D wing is experimental; run it with vortfoil_exp" + ("" if experimental is False else ", or add --experimental"))
    if a.shortcut:
        print("shortcut:", make_shortcut(a.experimental))
        return 0
    if sys.stdout is None or sys.stderr is None:              # pythonw (no console): keep messages in a log file
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        log = open(os.path.join(os.path.dirname(CACHE), "app.log"), "a", encoding="utf-8")
        sys.stdout, sys.stderr = sys.stdout or log, sys.stderr or log
    if sys.platform == "win32":
        try:                                                              # sharp widgets and figure on scaled displays
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    import tkinter as tk
    root = tk.Tk()
    app = App(root, selftest=(a.selftest, a.airfoil, a.alpha, a.Re, a.wing, a.V, a.state) if a.selftest else None, experimental=a.experimental)
    root.mainloop()
    return app.exit_code


if __name__ == "__main__":
    sys.exit(main())
