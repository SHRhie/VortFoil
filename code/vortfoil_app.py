# -*- coding: utf-8 -*-
"""VortFoil as a desktop application: choose an airfoil, set the incidence and the Reynolds number, press Run, and
read the flow field and the surface results in the same window.

    vortfoil                       (installed with `uv tool install vortfoil`)
    python code/vortfoil_app.py    (from a checkout)

The airfoil comes from one of three places:
  NACA     a 4- or 5-digit designation, generated exactly as the AirfRANS sections were;
  UIUC     the UIUC Airfoil Coordinates Database (M. S. Selig, UIUC Applied Aerodynamics Group,
           https://m-selig.ae.illinois.edu/ads/coord_database.html). Its list and the chosen section are downloaded
           from the site when needed and cached on this computer; VortFoil does not redistribute them
           (DATA_LICENSES.md);
  file     any coordinate file (Selig or Lednicer .dat), for a section that is in no public database.
Export data (CSV) writes every number of the result (coefficients, surface C_p, boundary layer, the field at every grid
point, the section) as CSV files with a text file that explains their columns. The result is the figure of
code/view_airfoil.py: the field of the decoder trained on AirfRANS (wind frame) and the
surface results of the vortfoil package, with every range warning, side by side in a wide window and stacked in a
tall one. Resizing the window zooms the figure, and Save figure writes it at 400 dpi (png, pdf or svg). It needs
matplotlib and Tk (part of most Python installs).

  --shortcut       Windows: a desktop shortcut that starts the application with this Python environment; Linux: an
                   entry in the applications menu; macOS: print the terminal command (and an alias), write nothing
  --selftest DIR   run one case without a person (--airfoil naca:2412 | uiuc:<name> | file:<path>), save the figure
                   and a capture of the window in DIR, and exit
"""
import os, re, sys, time, queue, threading, argparse, traceback, subprocess, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vortfoil_paths  # noqa: E402
vortfoil_paths.add_paths()            # this checkout's vortfoil package (src) and code/field, ahead of other installs
import view_airfoil as V  # noqa: E402

UIUC = "https://m-selig.ae.illinois.edu/ads/"
CACHE = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".cache"), "vortfoil", "uiuc")
DETAIL = {"fast (0.01 c)": 0.01, "normal (0.005 c)": 0.005, "fine (0.0025 c)": 0.0025}


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
    """(coordinates, display name) of a NACA designation, a UIUC database name or a coordinate file."""
    import predict_field as PF
    if kind == "naca":
        code = value.upper().replace("NACA", "").strip()
        if not re.fullmatch(r"\d{4,5}", code):
            raise ValueError(f"'{value}' is not a NACA 4- or 5-digit designation")
        return PF.naca_coords(code), f"NACA {code}"
    if kind == "uiuc":
        return PF.read_coords(fetch(f"{UIUC}coord/{value}.dat", os.path.join(CACHE, value + ".dat"))), value
    return PF.read_coords(value), os.path.splitext(os.path.basename(value))[0]


def load_index():
    path = fetch(UIUC + "coord_database.html", os.path.join(CACHE, "coord_database.html"))
    with open(path, encoding="latin-1") as f:
        return uiuc_index(f.read())


def run_case(loop, alpha, Re, h, name):
    t0 = time.perf_counter()
    res = V.compute(loop, alpha, Re, h)
    return res, time.perf_counter() - t0


class App:
    TASKS = {"index": load_index, "section": load_section, "run": run_case, "export": V.export_csv}

    def __init__(self, root, selftest=None):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        import matplotlib.pyplot as plt
        import predict_field as PF
        plt.rcParams.update(PF.STYLE)
        self.root, self.tk = root, tk
        self.jobs, self.section, self.index, self.shown, self.args, self.on_resize = queue.Queue(), None, [], [], None, None
        self.last, self.layout, self.relayout = None, "landscape", None
        self.selftest, self.exit_code = selftest, 0
        dpi = root.winfo_fpixels("1i")                       # the screen's real resolution: a sharp figure
        root.title("VortFoil")
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
        self.pname = ttk.Label(prev, text="no airfoil loaded", wraplength=wrap); self.pname.pack(anchor="w")

        fl = ttk.LabelFrame(panel, text="Flow", padding=6)
        self.alpha, self.re, self.detail = tk.StringVar(value="4"), tk.StringVar(value="3e6"), tk.StringVar(value="normal (0.005 c)")
        for i, (lab, var) in enumerate((("Incidence (deg)", self.alpha), ("Reynolds number", self.re))):
            ttk.Label(fl, text=lab).grid(row=i, column=0, sticky="w")
            en = ttk.Entry(fl, textvariable=var, width=12); en.grid(row=i, column=1, sticky="w"); en.bind("<Return>", lambda ev: self.run())
        ttk.Label(fl, text="Field detail").grid(row=2, column=0, sticky="w")
        ttk.Combobox(fl, textvariable=self.detail, values=list(DETAIL), state="readonly", width=16).grid(row=2, column=1, sticky="w")
        ttk.Label(fl, text="Trained on: surface -10 to 18 deg, Re 5e4 to 1e7; field -5 to 15 deg, Re 2e6 to 6e6. "
                  "Outside, the figure lists warnings.", foreground="gray", wraplength=wrap).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

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
            for i, w in enumerate((prev, fl, act, self.bar, self.status, cred)):
                w.grid(row=i, column=1, sticky="new", pady=(0 if i == 0 else 6, 0))
        else:                                                 # one column left of the figure
            for i, w in enumerate((box, prev, fl, act, self.bar, self.status)):
                w.grid(row=i, column=0, sticky="ew", pady=(0 if i == 0 else 8, 0))
            panel.rowconfigure(6, weight=1)
            cred.grid(row=6, column=0, sticky="sw")

        self.fig = Figure(figsize=V.DESIGN, dpi=dpi)
        self.can = FigureCanvasTkAgg(self.fig, master=right)
        NavigationToolbar2Tk(self.can, right).update()
        self.can.get_tk_widget().pack(fill="both", expand=True)
        self.can.get_tk_widget().bind("<Configure>", self.on_configure, add="+")
        self.fig.text(0.5, 0.5, "Choose an airfoil, set the flow and press Run", ha="center", va="center", fontsize=14, color="0.45")
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

    def poll(self):
        try:
            while True:
                task, args, res, err = self.jobs.get_nowait()
                getattr(self, "done_" + task)(args, res, err)
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
        self.status.configure(text=f"Loading {value} ...")
        self.submit("section", kind, value)

    def done_section(self, args, res, err):
        if err:
            self.status.configure(text=f"Could not load {args[1]}: {err}")
            if self.selftest:
                self.finish(1)
            return
        self.section = res
        loop, name = res
        self.pax.clear(); self.pax.axis("off")
        self.pax.fill(loop[:, 0], loop[:, 1], color="0.75", lw=0.8, ec="0.2"); self.pax.set_aspect("equal")
        self.pcan.draw_idle()
        self.pname.configure(text=f"{name}: {len(loop)} points")
        self.status.configure(text=f"{name} loaded. Set the flow and press Run.")
        if self.selftest:
            self.run()

    # ------------------------------------------------------------------ run and show
    def run(self):
        if self.section is None or str(self.run_btn["state"]) == "disabled":
            return
        try:
            alpha, Re = float(self.alpha.get()), float(self.re.get())
        except ValueError:
            self.status.configure(text="The incidence and the Reynolds number must be numbers (e.g. 4 and 3e6).")
            return
        loop, name = self.section
        self.run_btn.configure(state="disabled"); self.bar.configure(mode="indeterminate"); self.bar.start(12)
        self.status.configure(text=f"Computing {name} at {alpha:g} deg, Re {Re:.3g} ... (the first run also loads the models)")
        self.submit("run", loop, alpha, Re, DETAIL[self.detail.get()], name)

    def done_run(self, args, res, err):
        self.bar.stop(); self.bar.configure(mode="determinate", value=0); self.run_btn.configure(state="normal")
        if err:
            self.status.configure(text=f"The computation failed: {err}")
            traceback.print_exception(err)
            if self.selftest:
                self.finish(1)
            return
        (r, surf_w, out, meta), dt = res
        loop, alpha, Re, h, name = args
        self.args = (name, alpha, Re)
        self.last = (name, alpha, Re, r, surf_w, out, meta)
        self.draw(self.orientation())
        self.save_btn.configure(state="normal"); self.csv_btn.configure(state="normal")
        n = len(surf_w) + len(meta["warnings"])
        self.status.configure(text=f"{name}, {alpha:g} deg, Re {Re:.3g}: done in {dt:.1f} s"
                              + (f"; {n} warning(s), listed in the figure." if n else "; inside both training domains."))
        if self.selftest:
            self.root.after(500, self.finish_selftest)

    def orientation(self):
        """The layout that shows the figure largest on the canvas as it is now."""
        w = self.can.get_tk_widget().winfo_width() / self.fig.dpi; h = self.can.get_tk_widget().winfo_height() / self.fig.dpi
        zoom = {k: min(w / W, h / H) for k, (W, H) in V.DESIGNS.items()}
        return max(zoom, key=zoom.get)

    def draw(self, layout):
        """The last result in a side-by-side or stacked layout, frozen and zoomed to the canvas."""
        self.layout = layout
        V.figure(*self.last, fig=self.fig, layout=layout)
        self.on_resize = V.scale_with_window(self.fig)
        self.fit()

    def on_configure(self, event):
        """A resize that makes the other layout the larger one redraws in it; other resizes only zoom."""
        if self.last is None or self.orientation() == self.layout:
            return
        if self.relayout is not None:
            self.root.after_cancel(self.relayout)
        self.relayout = self.root.after(300, lambda: self.draw(self.orientation()))

    def fit(self):
        """Zoom the frozen layout to the canvas as it is now (later resizes do this through the resize handler)."""
        w = self.can.get_tk_widget().winfo_width(); h = self.can.get_tk_widget().winfo_height()
        if w > 1 and h > 1:
            self.fig.set_size_inches(w / self.fig.dpi, h / self.fig.dpi, forward=False)
        self.on_resize()

    def save(self):
        from tkinter import filedialog
        name, alpha, Re = self.args
        path = filedialog.asksaveasfilename(title="Save figure", defaultextension=".png",
                                            initialfile=re.sub(r"[^A-Za-z0-9.+-]+", "_", f"vortfoil_{name}_a{alpha:g}_Re{Re:.3g}") + ".png",
                                            filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if path:
            self.save_to(path)

    def export(self):
        """Every number of the result as CSV files in a chosen folder (written in the background)."""
        from tkinter import filedialog
        folder = filedialog.askdirectory(title="Folder for the CSV files", mustexist=False)
        if folder:
            self.status.configure(text=f"Writing the CSV files to {folder} ...")
            self.csv_btn.configure(state="disabled")
            self.submit("export", folder, *self.last)

    def done_export(self, args, res, err):
        self.csv_btn.configure(state="normal")
        self.status.configure(text=f"Could not write the CSV files: {err}" if err else
                              f"Wrote {len(res)} files to {args[0]}: coefficients, surface C_p, boundary layer, field, section, about")

    def save_to(self, path):
        """The figure at its design size and 400 dpi, whatever the window size."""
        w, h = self.fig.get_size_inches()
        self.fig.set_size_inches(*V.DESIGNS[self.layout], forward=False); self.on_resize()
        try:
            self.fig.savefig(path, dpi=400)
        finally:
            self.fig.set_size_inches(w, h, forward=False); self.on_resize()
        self.status.configure(text=f"Saved {path}")

    # ------------------------------------------------------------------ self-test
    def finish_selftest(self):
        out = self.selftest[0]
        os.makedirs(out, exist_ok=True)
        self.fig.savefig(os.path.join(out, "app_canvas.png"), dpi=self.fig.dpi)   # the canvas as shown, at its size
        cw, ch = self.can.get_tk_widget().winfo_width(), self.can.get_tk_widget().winfo_height()
        print(f"window {self.root.winfo_width()}x{self.root.winfo_height()}, canvas {cw}x{ch}, {self.fig.dpi:.0f} dpi, layout {self.layout}", flush=True)
        self.save_to(os.path.join(out, "app_figure.png"))
        V.export_csv(os.path.join(out, "csv"), *self.last)
        try:
            from PIL import ImageGrab
            self.root.update()
            x, y, w, h = self.root.winfo_rootx(), self.root.winfo_rooty(), self.root.winfo_width(), self.root.winfo_height()
            ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(os.path.join(out, "app_window.png"))
        except Exception as e:                                            # no screen capture on this platform
            print("window capture skipped:", e)
        print("SELFTEST OK", self.status.cget("text"), flush=True)
        self.finish(0)

    def finish(self, code):
        self.exit_code = code
        self.root.after(200, self.root.destroy)


def make_shortcut():
    """A launcher for this Python environment and this checkout; returns what was done. On macOS nothing is written:
    the application is started from the terminal, and the command (and a shell alias for it) is returned."""
    app = os.path.abspath(__file__)
    root = os.path.dirname(HERE)
    if sys.platform == "win32":                               # VortFoil.lnk on the desktop, through pythonw (no console)
        pyw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
        pyw = pyw if os.path.exists(pyw) else sys.executable
        ps = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut([IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), 'VortFoil.lnk')); "
              f"$s.TargetPath = '{pyw}'; $s.Arguments = '\"{app}\"'; $s.WorkingDirectory = '{root}'; $s.IconLocation = '{pyw},0'; "
              "$s.Description = 'VortFoil: airfoil field and surface results'; $s.Save(); $s.FullName")
        return subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, check=True).stdout.strip()
    if sys.platform == "darwin":                              # the terminal is the launcher: print, write nothing
        import shutil
        if shutil.which("vortfoil"):
            return "no file written on macOS; the vortfoil command is installed: type vortfoil in a terminal"
        cmd = f'"{sys.executable}" "{app}"'
        return (f"no file written on macOS; start it from the terminal with\n  {cmd}\n"
                f"or add this line to ~/.zshrc and type vortfoil:\n  alias vortfoil='{cmd}'")
    else:                                                     # an entry in the applications menu
        path = os.path.expanduser("~/.local/share/applications/vortfoil.desktop")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(f'[Desktop Entry]\nType=Application\nName=VortFoil\nComment=Airfoil field and surface results\n'
                    f'Exec="{sys.executable}" "{app}"\nPath={root}\nTerminal=false\nCategories=Science;Education;\n')
    os.chmod(path, 0o755)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shortcut", action="store_true", help="a launcher for this Python environment (macOS: print the terminal command) and exit")
    ap.add_argument("--selftest", metavar="DIR", help="run one case unattended, save the figure and a window capture in DIR")
    ap.add_argument("--airfoil", default="naca:2412", help="for --selftest: naca:<digits>, uiuc:<name> or file:<path>")
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--Re", type=float, default=3e6)
    a = ap.parse_args(argv)
    if a.shortcut:
        print("shortcut:", make_shortcut())
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
    app = App(root, selftest=(a.selftest, a.airfoil, a.alpha, a.Re) if a.selftest else None)
    root.mainloop()
    return app.exit_code


if __name__ == "__main__":
    sys.exit(main())
