"""Interactive desktop Hodgkin-Huxley membrane dynamics simulator (hub UI)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure
import numpy as np

from src.analysis.spike_metrics import SpikeMetrics, summarize_voltage_trace
from src.models.hodgkin_huxley import (
    HodgkinHuxleyNeuron,
    HodgkinHuxleyParameters,
    HodgkinHuxleyState,
)
from src.simulation.config import SimulationConfig
from src.simulation.protocols import (
    ConductanceChange,
    ConductanceSchedule,
    CurrentPulse,
    MultiPulseCurrent,
)
from src.simulation.runner import SimulationResult, simulate
from src.utils.export import export_simulation_csv


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INTEGRATION_DT_MS = 0.01
CONTROL_CONTENT_WIDTH = 330
FULL_BUTTON_WIDTH = 54
HALF_BUTTON_WIDTH = 25

SNAP_THRESHOLD_PX = 18

GRAPH_ORDER: tuple[str, ...] = (
    "voltage",
    "current",
    "net_ionic",
    "conductance",
    "gating",
)
GRAPH_LABELS: dict[str, str] = {
    "voltage": "Voltage",
    "current": "Injected Current",
    "net_ionic": "Net Ionic Current",
    "conductance": "Conductances",
    "gating": "Ion Channel Functional States",
}
GRAPH_STEMS: dict[str, str] = {
    "voltage": "voltage",
    "current": "injected_current",
    "net_ionic": "net_ionic_current",
    "conductance": "sodium_potassium_conductances",
    "gating": "ion_channel_functional_states",
}
DEFAULT_SAVE_DIMS: tuple[float, float] = (6.4, 4.0)


PARAMETER_TOOLTIPS: dict[str, str] = {
    "current_baseline": (
        "Baseline (uA/cm^2): constant background current that is always present. "
        "All pulses add on top of it.\n\n"
        "Example: baseline = 0 keeps the neuron quiet between pulses. "
        "baseline = 6 provides steady drive that can push the cell into "
        "repetitive firing even without a pulse. Negative values (e.g. -2) "
        "hyperpolarize and raise the threshold for spiking."
    ),
    "membrane_capacitance": (
        "C_m (uF/cm^2): membrane capacitance density. Sets how quickly the "
        "voltage responds to injected current. Larger C_m slows voltage "
        "changes; smaller C_m speeds them up.\n\n"
        "Example: C_m = 1.0 (default) gives the classic HH spike shape. "
        "C_m = 2.0 broadens spikes and delays firing onset. "
        "C_m = 0.5 sharpens spikes and lowers the current needed to fire."
    ),
    "g_l": (
        "g_L (mS/cm^2): leak conductance density. Sets how strongly the "
        "membrane is pulled toward the leak reversal E_L. Larger g_L clamps "
        "V near E_L and raises the current needed to spike.\n\n"
        "Example: g_L = 0.3 (default) allows normal firing near 10 uA/cm^2. "
        "g_L = 1.0 sharply raises threshold and can silence the cell. "
        "g_L = 0.05 makes the cell hyperexcitable and easier to spike."
    ),
    "e_l": (
        "E_L (mV): leak reversal potential. The voltage the leak current "
        "drives V toward at rest.\n\n"
        "Example: E_L = -54.387 (default) balances Na and K leak so V rests "
        "near -65 mV. Making E_L more positive (e.g. -40) depolarizes rest "
        "and can cause tonic firing. More negative (e.g. -80) hyperpolarizes "
        "rest and suppresses spikes."
    ),
}


class Tooltip:
    """Simple hover tooltip that appears next to a widget."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, _event: tk.Event | None = None) -> None:
        if self.tip is not None:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + 22
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self.tip,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            wraplength=340,
            font=("Segoe UI", 9),
            padx=8,
            pady=6,
        ).pack()

    def _hide(self, _event: tk.Event | None = None) -> None:
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


@dataclass(frozen=True)
class ControlDefaults:
    """Default GUI values that reproduce the classic HH current-step response."""

    duration_ms: float = 50.0
    resting_voltage_mV: float = -65.0
    initial_voltage_mV: float = -65.0
    current_amplitude: float = 10.0
    current_start_ms: float = 5.0
    current_end_ms: float = 35.0
    current_baseline: float = 0.0
    g_na: float = 120.0
    g_k: float = 36.0
    g_l: float = 0.3
    e_na: float = 50.0
    e_k: float = -77.0
    e_l: float = -54.387
    membrane_capacitance: float = 1.0


class HodgkinHuxleySimulatorApp:
    """Small hub window that opens Parameters, Graphs, Save Plots, and Help as tabs."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("HH Simulator")
        self.root.resizable(False, False)

        self.defaults = ControlDefaults()

        # Persistent shadow of parameter values (survives Parameters-window close/reopen).
        self._param_snapshot: dict[str, str] = {
            name: str(value) for name, value in self.defaults.__dict__.items()
        }
        self._pulse_snapshot: list[tuple[str, str, str]] = [("", "", "")]
        self._g_na_snapshot: list[tuple[str, str, str]] = [("", "", "")]
        self._g_k_snapshot: list[tuple[str, str, str]] = [("", "", "")]

        # Live widgets (rebuilt when Parameters window opens).
        self.variables: dict[str, tk.StringVar] = {}
        self.extra_pulse_rows: list[tuple[tk.StringVar, tk.StringVar, tk.StringVar]] = []
        self.g_na_change_rows: list[tuple[tk.StringVar, tk.StringVar, tk.StringVar]] = []
        self.g_k_change_rows: list[tuple[tk.StringVar, tk.StringVar, tk.StringVar]] = []
        self.extra_pulses_container: ttk.Frame | None = None
        self.g_na_changes_container: ttk.Frame | None = None
        self.g_k_changes_container: ttk.Frame | None = None
        self.metrics_var: tk.StringVar | None = None

        # Simulation state.
        self.result: SimulationResult | None = None
        self.metrics: SpikeMetrics | None = None
        self.current_parameters: HodgkinHuxleyParameters | None = None

        # Windows.
        self._parameters_window: tk.Toplevel | None = None
        self._save_window: tk.Toplevel | None = None
        self._graph_windows: dict[str, dict] = {}
        self._help_canvases: list[FigureCanvasTkAgg] = []

        # Ion channel isolate state (persists across window close/reopen).
        self._gating_visible: dict[str, bool] = {"m": True, "h": True, "n": True}

        # Shared visible-time window (Init / Step / Full buttons).
        self._visible_time_end_ms: float | None = None

        # Save-plot per-graph dimensions (inches), plus separate "All" dims.
        self._save_dimensions: dict[str, tuple[float, float]] = {
            kind: DEFAULT_SAVE_DIMS for kind in GRAPH_ORDER
        }
        self._save_all_dimensions: tuple[float, float] = DEFAULT_SAVE_DIMS

        self._configure_style()
        self._build_hub()
        self.root.protocol("WM_DELETE_WINDOW", self._on_hub_close)

        # Prime an initial simulation so any graph opened has data ready.
        self._run_simulation_from_snapshot()

    # ------------------------------------------------------------------ style

    def _configure_style(self) -> None:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("Metric.TLabel", font=("Consolas", 10))
        style.configure("Run.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("HelpTitle.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("HelpHeading.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("HelpBody.TLabel", font=("Segoe UI", 10))
        style.configure("HelpChoice.TButton", font=("Segoe UI", 11, "bold"), padding=(14, 8))

    # ------------------------------------------------------------------ hub

    def _build_hub(self) -> None:
        self.root.geometry("520x220")
        frame = ttk.Frame(self.root, padding=(20, 18, 20, 18))
        frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        for col in range(4):
            frame.columnconfigure(col, weight=1, uniform="hub")

        common = dict(
            font=("Segoe UI", 10, "bold"),
            relief="raised",
            bd=1,
            padx=6,
            pady=10,
            takefocus=0,
            highlightthickness=0,
            bg="#ffffff",
            activebackground="#f0f0f0",
        )
        tk.Button(frame, text="Parameters", command=self.open_parameters_tab, **common).grid(
            row=0, column=0, sticky="ew", padx=4
        )
        self._graphs_button = tk.Button(
            frame, text="Graphs ▾", command=self._show_graphs_menu, **common
        )
        self._graphs_button.grid(row=0, column=1, sticky="ew", padx=4)
        tk.Button(frame, text="Save Plots", command=self.open_save_plots, **common).grid(
            row=0, column=2, sticky="ew", padx=4
        )
        tk.Button(frame, text="Help", command=self.show_help, **common).grid(
            row=0, column=3, sticky="ew", padx=4
        )

        ttk.Label(
            frame,
            text="Select a feature to open.",
            font=("Segoe UI", 10),
            foreground="#333333",
        ).grid(row=1, column=0, columnspan=4, pady=(24, 0))

        status_row = ttk.Frame(frame)
        status_row.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(16, 0))
        status_row.columnconfigure(0, weight=1)
        status_row.columnconfigure(1, weight=0)
        status_row.columnconfigure(2, weight=0)
        self.status_var = tk.StringVar(value="")
        ttk.Label(
            status_row,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            foreground="#666666",
        ).grid(row=0, column=0, sticky="w")
        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress_bar = ttk.Progressbar(
            status_row,
            orient="horizontal",
            mode="determinate",
            length=110,
            maximum=100,
            variable=self._progress_var,
        )
        self._progress_percent_var = tk.StringVar(value="")
        self._progress_percent_label = ttk.Label(
            status_row,
            textvariable=self._progress_percent_var,
            font=("Segoe UI", 9),
            foreground="#555555",
            width=5,
            anchor="e",
        )
        # progress bar + label are hidden by default; shown only while a run is in progress

    def _show_graphs_menu(self) -> None:
        menu = tk.Menu(self.root, tearoff=0)
        for kind in GRAPH_ORDER:
            menu.add_command(
                label=GRAPH_LABELS[kind],
                command=lambda k=kind: self.open_graph_window(k),
            )
        x = self._graphs_button.winfo_rootx()
        y = self._graphs_button.winfo_rooty() + self._graphs_button.winfo_height()
        menu.tk_popup(x, y)

    def _on_hub_close(self) -> None:
        if self._any_child_open():
            confirm = messagebox.askyesno(
                "Close simulator?",
                "Closing this window will end application and all unsaved data will be lost, "
                "are you sure you want to close it?",
                icon="warning",
                default="no",
                parent=self.root,
            )
            if not confirm:
                return
        self.root.destroy()

    def _any_child_open(self) -> bool:
        for window in (self._parameters_window, self._save_window):
            if window is not None and self._window_visible(window):
                return True
        for state in self._graph_windows.values():
            if self._window_visible(state["window"]):
                return True
        return False

    @staticmethod
    def _window_visible(window: tk.Toplevel) -> bool:
        try:
            return bool(window.winfo_exists()) and window.state() != "withdrawn"
        except tk.TclError:
            return False

    # ------------------------------------------------------ Parameters window

    def open_parameters_tab(self) -> None:
        if self._parameters_window is not None and self._parameters_window.winfo_exists():
            self._parameters_window.deiconify()
            self._parameters_window.lift()
            self._parameters_window.focus_force()
            return

        window = tk.Toplevel(self.root)
        window.title("Parameters")
        window.geometry("400x900")
        window.minsize(380, 500)
        self._parameters_window = window
        window.protocol("WM_DELETE_WINDOW", self._on_parameters_close)

        controls = self._build_scrollable_controls(window)
        self._build_controls(controls)
        self._hydrate_controls_from_snapshot()

    def _on_parameters_close(self) -> None:
        self._snapshot_controls()
        if self._parameters_window is not None:
            self._parameters_window.destroy()
        self._parameters_window = None
        self.variables = {}
        self.extra_pulse_rows = []
        self.g_na_change_rows = []
        self.g_k_change_rows = []
        self.metrics_var = None
        self.extra_pulses_container = None
        self.g_na_changes_container = None
        self.g_k_changes_container = None

    def _snapshot_controls(self) -> None:
        if self.variables:
            for name, var in self.variables.items():
                self._param_snapshot[name] = var.get()
        if self.extra_pulses_container is not None:
            self._pulse_snapshot = self._snapshot_rows(self.extra_pulse_rows)
        if self.g_na_changes_container is not None:
            self._g_na_snapshot = self._snapshot_rows(self.g_na_change_rows)
        if self.g_k_changes_container is not None:
            self._g_k_snapshot = self._snapshot_rows(self.g_k_change_rows)

    @staticmethod
    def _snapshot_rows(
        rows: list[tuple[tk.StringVar, tk.StringVar, tk.StringVar]],
    ) -> list[tuple[str, str, str]]:
        return [(a.get(), b.get(), c.get()) for a, b, c in rows]

    def _hydrate_controls_from_snapshot(self) -> None:
        for name, value in self._param_snapshot.items():
            if name in self.variables:
                self.variables[name].set(value)

        self._clear_row_inputs(self.extra_pulses_container, self.extra_pulse_rows)
        self._clear_row_inputs(self.g_na_changes_container, self.g_na_change_rows)
        self._clear_row_inputs(self.g_k_changes_container, self.g_k_change_rows)

        pulse_snap = self._pulse_snapshot or [("", "", "")]
        for index, (start, end, amp) in enumerate(pulse_snap):
            self.add_pulse_row(start, end, amp, removable=(index > 0))

        na_snap = self._g_na_snapshot or [("", "", "")]
        for index, (t, e, v) in enumerate(na_snap):
            self.add_g_na_change_row(t, e, v, removable=(index > 0))

        k_snap = self._g_k_snapshot or [("", "", "")]
        for index, (t, e, v) in enumerate(k_snap):
            self.add_g_k_change_row(t, e, v, removable=(index > 0))

        self._update_metrics()

    # -------------------------------------------------- scrollable controls

    def _build_scrollable_controls(self, parent: tk.Widget) -> ttk.Frame:
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas, padding=(8, 8, 8, 8))
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def update_scroll_region(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def update_width(event: tk.Event) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        def scroll_controls(event: tk.Event) -> str | None:
            widget = parent.winfo_containing(event.x_root, event.y_root)
            inside = False
            walk = widget
            while walk is not None:
                if walk in {canvas, content, scrollbar}:
                    inside = True
                    break
                walk = walk.master
            if not inside:
                return None
            if getattr(event, "num", None) == 4:
                canvas.yview_scroll(-3, "units")
            elif getattr(event, "num", None) == 5:
                canvas.yview_scroll(3, "units")
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)) * 3, "units")
            return "break"

        content.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", update_width)
        parent.bind("<MouseWheel>", scroll_controls, add="+")
        parent.bind("<Button-4>", scroll_controls, add="+")
        parent.bind("<Button-5>", scroll_controls, add="+")

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        return content

    # ---------------------------------------------------------- controls UI

    def _build_controls(self, parent: ttk.Frame) -> None:
        simulation = ttk.LabelFrame(parent, text="Simulation", style="Section.TLabelframe")
        simulation.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self._add_entry(simulation, "duration_ms", "Duration (ms)", 0, label_width=22)
        self._add_entry(simulation, "resting_voltage_mV", "Resting V (mV)", 1, label_width=22)
        self._add_entry(simulation, "initial_voltage_mV", "Initial V (mV)", 2, label_width=22)
        step_buttons = ttk.Frame(simulation)
        step_buttons.grid(row=3, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 6))
        step_buttons.columnconfigure((0, 1, 2, 3), weight=1)
        ttk.Button(step_buttons, text="Init", command=self.init_view).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(step_buttons, text="Step 0.5 ms", command=lambda: self.advance_view(0.5)).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(step_buttons, text="Step 5 ms", command=lambda: self.advance_view(5.0)).grid(
            row=0, column=2, sticky="ew", padx=4
        )
        ttk.Button(step_buttons, text="Full", command=self.reset_full_view).grid(
            row=0, column=3, sticky="ew", padx=(4, 0)
        )

        current = ttk.LabelFrame(parent, text="Injected Current", style="Section.TLabelframe")
        current.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self._add_entry(current, "current_amplitude", "Amplitude (uA/cm^2)", 0, label_width=22)
        self._add_entry(current, "current_start_ms", "Start (ms)", 1, label_width=22)
        self._add_entry(current, "current_end_ms", "End (ms)", 2, label_width=22)
        self._add_entry(current, "current_baseline", "Baseline (uA/cm^2)", 3, label_width=22)
        pulse_frame = ttk.Frame(current)
        pulse_frame.grid(row=4, column=0, columnspan=2, sticky="w", padx=4, pady=(8, 4))
        ttk.Label(pulse_frame, text="Additional pulses").grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )
        self.extra_pulses_container = ttk.Frame(pulse_frame)
        self.extra_pulses_container.grid(row=1, column=0, sticky="w")
        ttk.Button(
            pulse_frame,
            text="Add Pulse",
            command=self.add_pulse_row,
            width=FULL_BUTTON_WIDTH,
        ).grid(row=2, column=0, sticky="w", pady=(6, 0))

        conductance = ttk.LabelFrame(parent, text="Conductances", style="Section.TLabelframe")
        conductance.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self._add_entry(conductance, "g_na", "g_Na max (mS/cm^2)", 0, label_width=22)
        self._add_entry(conductance, "g_k", "g_K max (mS/cm^2)", 1, label_width=22)
        self._add_entry(conductance, "g_l", "g_L (mS/cm^2)", 2, label_width=22)
        self._add_entry(conductance, "membrane_capacitance", "C_m (uF/cm^2)", 3, label_width=22)
        self.g_na_changes_container = self._add_schedule_section(
            conductance, "g_Na changes", "Add g_Na Change", self.add_g_na_change_row, 4
        )
        self.g_k_changes_container = self._add_schedule_section(
            conductance, "g_K changes", "Add g_K Change", self.add_g_k_change_row, 5
        )

        reversal = ttk.LabelFrame(parent, text="Reversal Potentials", style="Section.TLabelframe")
        reversal.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self._add_entry(reversal, "e_na", "E_Na (mV)", 0)
        self._add_entry(reversal, "e_k", "E_K (mV)", 1)
        self._add_entry(reversal, "e_l", "E_L (mV)", 2)

        buttons = ttk.Frame(parent)
        buttons.grid(row=4, column=0, sticky="ew", pady=(2, 10))
        buttons.columnconfigure((0, 1), weight=1, uniform="bottom")
        ttk.Button(
            buttons,
            text="Run Simulation",
            style="Run.TButton",
            command=self.run_simulation,
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Button(buttons, text="Export CSV", command=self.export_csv).grid(
            row=1, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(buttons, text="Reset Defaults", command=self.reset_defaults).grid(
            row=1, column=1, sticky="ew", padx=(4, 0)
        )

        metrics = ttk.LabelFrame(parent, text="Trace Metrics", style="Section.TLabelframe")
        metrics.grid(row=5, column=0, sticky="ew")
        self.metrics_var = tk.StringVar(value="")
        ttk.Label(
            metrics,
            textvariable=self.metrics_var,
            style="Metric.TLabel",
            justify="left",
            padding=(6, 6),
        ).grid(row=0, column=0, sticky="ew")

    def _add_entry(
        self,
        parent: ttk.Frame,
        name: str,
        label: str,
        row: int,
        *,
        label_width: int | None = None,
    ) -> None:
        variable = tk.StringVar()
        tooltip_text = PARAMETER_TOOLTIPS.get(name)
        if label_width is not None:
            entry_row = ttk.Frame(parent)
            entry_row.grid(row=row, column=0, columnspan=2, sticky="w", padx=(4, 0), pady=4)
            ttk.Label(entry_row, text=label, width=label_width).grid(row=0, column=0, sticky="w")
            entry = ttk.Entry(entry_row, textvariable=variable, width=10)
            entry.grid(row=0, column=1, sticky="w", padx=(0, 4))
            entry.bind("<Return>", lambda _event: self.run_simulation())
            if tooltip_text is not None:
                self._add_help_emblem(entry_row, 0, 2, tooltip_text)
            self.variables[name] = variable
            return

        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(4, 2), pady=4)
        entry = ttk.Entry(parent, textvariable=variable, width=10)
        entry.grid(row=row, column=1, sticky="w", padx=(2, 4), pady=4)
        entry.bind("<Return>", lambda _event: self.run_simulation())
        if tooltip_text is not None:
            self._add_help_emblem(parent, row, 2, tooltip_text)
        parent.columnconfigure(1, weight=0)
        self.variables[name] = variable

    def _add_help_emblem(
        self,
        parent: ttk.Frame,
        row: int,
        column: int,
        tooltip_text: str,
    ) -> None:
        emblem = tk.Label(
            parent,
            text="?",
            font=("Segoe UI", 8, "bold"),
            fg="#000000",
            bg="#ffffff",
            width=2,
            padx=0,
            pady=0,
            borderwidth=1,
            relief="solid",
            cursor="question_arrow",
        )
        emblem.grid(row=row, column=column, sticky="w", padx=(2, 0))
        Tooltip(emblem, tooltip_text)

    def _add_small_entry(
        self,
        parent: ttk.Frame,
        variable: tk.StringVar,
        row: int,
        column: int,
        title: str,
        suffix: str,
    ) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=column, sticky="w", padx=2, pady=2)
        ttk.Label(frame, text=title, anchor="center").grid(
            row=0, column=0, sticky="ew", pady=(0, 2)
        )
        entry = ttk.Entry(frame, textvariable=variable, width=5)
        entry.grid(row=1, column=0, sticky="w")
        entry.bind("<Return>", lambda _event: self.run_simulation())
        ttk.Label(frame, text=suffix).grid(row=1, column=1, sticky="w", padx=(4, 0))

    def _add_schedule_section(
        self,
        parent: ttk.Frame,
        title: str,
        button_text: str,
        command,
        row: int,
    ) -> ttk.Frame:
        section = ttk.Frame(parent)
        section.grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(8, 4))
        ttk.Label(section, text=title).grid(row=0, column=0, columnspan=2, sticky="w")
        container = ttk.Frame(section)
        container.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Button(section, text=button_text, command=command, width=FULL_BUTTON_WIDTH).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        return container

    def add_pulse_row(
        self,
        start_ms: str = "",
        end_ms: str = "",
        amplitude: str = "",
        *,
        removable: bool = True,
    ) -> None:
        if self.extra_pulses_container is None:
            return
        start_var = tk.StringVar(value=start_ms)
        end_var = tk.StringVar(value=end_ms)
        amplitude_var = tk.StringVar(value=amplitude)
        entry = (start_var, end_var, amplitude_var)
        row_frame = ttk.Frame(self.extra_pulses_container)
        row_frame.grid(sticky="w", pady=1)
        self._add_small_entry(row_frame, start_var, 0, 0, "Start", "ms")
        self._add_small_entry(row_frame, end_var, 0, 1, "End", "ms")
        self._add_small_entry(row_frame, amplitude_var, 0, 2, "Amplitude", "uA/cm^2")
        if removable:
            self._add_row_remove_button(row_frame, 0, 3, entry, self.extra_pulse_rows)
        self.extra_pulse_rows.append(entry)

    def add_g_na_change_row(
        self,
        time_ms: str = "",
        end_ms: str = "",
        value: str = "",
        *,
        removable: bool = True,
    ) -> None:
        if self.g_na_changes_container is None:
            return
        self._add_conductance_change_row(
            self.g_na_changes_container,
            self.g_na_change_rows,
            time_ms,
            end_ms,
            value,
            "mS/cm^2",
            removable=removable,
        )

    def add_g_k_change_row(
        self,
        time_ms: str = "",
        end_ms: str = "",
        value: str = "",
        *,
        removable: bool = True,
    ) -> None:
        if self.g_k_changes_container is None:
            return
        self._add_conductance_change_row(
            self.g_k_changes_container,
            self.g_k_change_rows,
            time_ms,
            end_ms,
            value,
            "mS/cm^2",
            removable=removable,
        )

    def _add_conductance_change_row(
        self,
        parent: ttk.Frame,
        rows: list[tuple[tk.StringVar, tk.StringVar, tk.StringVar]],
        time_ms: str,
        end_ms: str,
        value: str,
        unit: str,
        *,
        removable: bool = True,
    ) -> None:
        time_var = tk.StringVar(value=time_ms)
        end_var = tk.StringVar(value=end_ms)
        value_var = tk.StringVar(value=value)
        entry = (time_var, end_var, value_var)
        row_frame = ttk.Frame(parent)
        row_frame.grid(sticky="w", pady=1)
        self._add_small_entry(row_frame, time_var, 0, 0, "Time", "ms")
        self._add_small_entry(row_frame, end_var, 0, 1, "End", "ms")
        self._add_small_entry(row_frame, value_var, 0, 2, "Value", unit)
        if removable:
            self._add_row_remove_button(row_frame, 0, 3, entry, rows)
        rows.append(entry)

    def _add_row_remove_button(
        self,
        row_frame: ttk.Frame,
        row: int,
        column: int,
        entry: tuple[tk.StringVar, tk.StringVar, tk.StringVar],
        rows: list,
    ) -> None:
        button = tk.Button(
            row_frame,
            text="✕",
            font=("Segoe UI", 8, "bold"),
            fg="#000000",
            width=2,
            relief="flat",
            bd=0,
            padx=0,
            pady=0,
            highlightthickness=0,
            takefocus=0,
            cursor="hand2",
            command=lambda: self._remove_dynamic_row(row_frame, entry, rows),
        )
        button.grid(row=row, column=column, sticky="w", padx=(4, 0), pady=(14, 0))

    def _remove_dynamic_row(
        self,
        row_frame: ttk.Frame,
        entry: tuple[tk.StringVar, tk.StringVar, tk.StringVar],
        rows: list,
    ) -> None:
        try:
            rows.remove(entry)
        except ValueError:
            pass
        row_frame.destroy()

    def reset_defaults(self) -> None:
        for field_name, value in self.defaults.__dict__.items():
            self._param_snapshot[field_name] = str(value)
            if field_name in self.variables:
                self.variables[field_name].set(str(value))
        self._pulse_snapshot = [("", "", "")]
        self._g_na_snapshot = [("", "", "")]
        self._g_k_snapshot = [("", "", "")]
        if self.extra_pulses_container is not None:
            self._clear_row_inputs(self.extra_pulses_container, self.extra_pulse_rows)
            self.add_pulse_row(removable=False)
        if self.g_na_changes_container is not None:
            self._clear_row_inputs(self.g_na_changes_container, self.g_na_change_rows)
            self.add_g_na_change_row(removable=False)
        if self.g_k_changes_container is not None:
            self._clear_row_inputs(self.g_k_changes_container, self.g_k_change_rows)
            self.add_g_k_change_row(removable=False)
        self.status_var.set("Default HH parameters loaded")

    def _clear_row_inputs(self, parent: ttk.Frame | None, rows: list) -> None:
        if parent is not None:
            for child in parent.winfo_children():
                child.destroy()
        rows.clear()

    # ----------------------------------------------------- simulation logic

    def _read_float(self, name: str) -> float:
        raw = self.variables[name].get() if name in self.variables else self._param_snapshot.get(name, "")
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number.") from exc

    def _extra_pulse_source(self) -> list[tuple[str, str, str]]:
        if self.extra_pulse_rows:
            return self._snapshot_rows(self.extra_pulse_rows)
        return list(self._pulse_snapshot)

    def _g_na_change_source(self) -> list[tuple[str, str, str]]:
        if self.g_na_change_rows:
            return self._snapshot_rows(self.g_na_change_rows)
        return list(self._g_na_snapshot)

    def _g_k_change_source(self) -> list[tuple[str, str, str]]:
        if self.g_k_change_rows:
            return self._snapshot_rows(self.g_k_change_rows)
        return list(self._g_k_snapshot)

    def _parse_extra_current_pulses(self) -> tuple[CurrentPulse, ...]:
        pulses: list[CurrentPulse] = []
        for start, end, amp in self._extra_pulse_source():
            values = (start.strip(), end.strip(), amp.strip())
            if not any(values):
                continue
            if not all(values):
                raise ValueError("Additional pulse rows must include start, end, and amplitude.")
            start_ms, end_ms, amplitude = (float(value) for value in values)
            pulses.append(CurrentPulse(amplitude=amplitude, start_ms=start_ms, end_ms=end_ms))
        return tuple(pulses)

    def _parse_conductance_changes(
        self,
        rows: list[tuple[str, str, str]],
        label: str,
    ) -> tuple[ConductanceChange, ...]:
        changes: list[ConductanceChange] = []
        for time_ms, end_ms, value in rows:
            values = (time_ms.strip(), end_ms.strip(), value.strip())
            if not any(values):
                continue
            if not all(values):
                raise ValueError(f"{label} rows must include time, end, and conductance.")
            t, e, v = (float(item) for item in values)
            changes.append(ConductanceChange(time_ms=t, end_ms=e, value=v))
        return tuple(changes)

    def _build_simulation_objects(self):
        parameters = HodgkinHuxleyParameters(
            membrane_capacitance=self._read_float("membrane_capacitance"),
            g_na=self._read_float("g_na"),
            g_k=self._read_float("g_k"),
            g_l=self._read_float("g_l"),
            e_na=self._read_float("e_na"),
            e_k=self._read_float("e_k"),
            e_l=self._read_float("e_l"),
        )
        config = SimulationConfig(
            duration_ms=self._read_float("duration_ms"),
            dt_ms=DEFAULT_INTEGRATION_DT_MS,
            method="rk4",
            resting_voltage_mV=self._read_float("resting_voltage_mV"),
        )
        primary_pulse = CurrentPulse(
            amplitude=self._read_float("current_amplitude"),
            start_ms=self._read_float("current_start_ms"),
            end_ms=self._read_float("current_end_ms"),
        )
        protocol = MultiPulseCurrent(
            baseline=self._read_float("current_baseline"),
            pulses=(primary_pulse, *self._parse_extra_current_pulses()),
        )

        neuron = HodgkinHuxleyNeuron(parameters)
        resting_voltage = self._read_float("resting_voltage_mV")
        initial_voltage = self._read_float("initial_voltage_mV")
        resting_state = HodgkinHuxleyState.from_voltage(neuron, resting_voltage)
        initial_state = HodgkinHuxleyState(
            voltage=initial_voltage,
            m=resting_state.m,
            h=resting_state.h,
            n=resting_state.n,
        )
        g_na_schedule = ConductanceSchedule(
            base_value=parameters.g_na,
            changes=self._parse_conductance_changes(self._g_na_change_source(), "g_Na changes"),
        )
        g_k_schedule = ConductanceSchedule(
            base_value=parameters.g_k,
            changes=self._parse_conductance_changes(self._g_k_change_source(), "g_K changes"),
        )
        return neuron, config, protocol, initial_state, g_na_schedule, g_k_schedule

    def _on_sim_progress(self, fraction: float) -> None:
        percent = max(0.0, min(100.0, fraction * 100.0))
        self._progress_var.set(percent)
        if 0.0 < percent < 100.0:
            self._show_progress_bar()
            self._progress_percent_var.set(f"{percent:.0f}%")
            self.status_var.set("Simulating...")
        else:
            self._hide_progress_bar()
        try:
            self.root.update_idletasks()
        except tk.TclError:
            pass

    def _show_progress_bar(self) -> None:
        try:
            self._progress_bar.grid(row=0, column=1, sticky="e", padx=(6, 4))
            self._progress_percent_label.grid(row=0, column=2, sticky="e")
        except tk.TclError:
            pass

    def _hide_progress_bar(self) -> None:
        try:
            self._progress_bar.grid_remove()
            self._progress_percent_label.grid_remove()
        except tk.TclError:
            pass
        self._progress_percent_var.set("")

    def _run_simulation_from_snapshot(self) -> None:
        try:
            neuron, config, protocol, initial_state, g_na_schedule, g_k_schedule = (
                self._build_simulation_objects()
            )
            self.result = simulate(
                neuron=neuron,
                config=config,
                current_protocol=protocol,
                initial_state=initial_state,
                g_na_schedule=g_na_schedule,
                g_k_schedule=g_k_schedule,
                progress_callback=self._on_sim_progress,
            )
            self.metrics = summarize_voltage_trace(self.result)
            self.current_parameters = neuron.parameters
            self._progress_var.set(100.0)
            self._hide_progress_bar()
            self.status_var.set("Initial simulation ready")
        except Exception:  # noqa: BLE001
            self._progress_var.set(0.0)
            self._hide_progress_bar()
            self.status_var.set("Initial simulation failed; open Parameters to fix")

    def run_simulation(self) -> None:
        try:
            neuron, config, protocol, initial_state, g_na_schedule, g_k_schedule = (
                self._build_simulation_objects()
            )
            self.result = simulate(
                neuron=neuron,
                config=config,
                current_protocol=protocol,
                initial_state=initial_state,
                g_na_schedule=g_na_schedule,
                g_k_schedule=g_k_schedule,
                progress_callback=self._on_sim_progress,
            )
            self.metrics = summarize_voltage_trace(self.result)
            self.current_parameters = neuron.parameters
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Simulation error", str(exc), parent=self._parameters_window or self.root)
            self._hide_progress_bar()
            self.status_var.set("Simulation failed")
            return

        self._visible_time_end_ms = None
        self._snapshot_controls()
        self._update_metrics()
        self._redraw_all_graphs()
        self._apply_visible_time_end_to_graphs()
        self._refresh_save_targets()
        self._progress_var.set(100.0)
        self._hide_progress_bar()
        self.status_var.set("Simulation complete")

    def _update_metrics(self) -> None:
        if self.metrics_var is None or self.metrics is None:
            return
        first_spike = (
            f"{self.metrics.first_spike_time_ms:.2f} ms"
            if self.metrics.first_spike_time_ms is not None
            else "none"
        )
        self.metrics_var.set(
            "spikes:      "
            f"{self.metrics.spike_count}\n"
            "rate:        "
            f"{self.metrics.firing_rate_hz:.2f} Hz\n"
            "Spike detection: 0 mV\n"
            "First spike: "
            f"{first_spike}\n"
            "Max V:       "
            f"{self.metrics.peak_voltage_mV:.2f} mV\n"
            "Min V:       "
            f"{self.metrics.trough_voltage_mV:.2f} mV\n"
        )

    # ------------------------------------------------------ Graph windows

    def open_graph_window(self, kind: str) -> None:
        if kind in self._graph_windows:
            existing = self._graph_windows[kind]
            if existing["window"].winfo_exists():
                existing["window"].deiconify()
                existing["window"].lift()
                existing["window"].focus_force()
                self._draw_graph(kind)
                return
            self._graph_windows.pop(kind, None)

        window = tk.Toplevel(self.root)
        window.title(GRAPH_LABELS[kind])
        window.geometry("640x430")
        window.minsize(260, 150)
        window.protocol("WM_DELETE_WINDOW", lambda k=kind: self._on_graph_close(k))

        figure = Figure(figsize=(5.8, 3.8), dpi=100)
        axis = figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        toolbar_frame = ttk.Frame(window)
        toolbar_frame.pack(side="bottom", fill="x")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="left")

        state: dict = {
            "window": window,
            "figure": figure,
            "canvas": canvas,
            "axis": axis,
            "toolbar": toolbar,
        }

        def _on_resize(_event: tk.Event, k: str = kind) -> None:
            active = self._graph_windows.get(k)
            if active is None:
                return
            try:
                active["figure"].tight_layout()
                active["canvas"].draw_idle()
            except Exception:  # noqa: BLE001
                pass

        canvas.get_tk_widget().bind("<Configure>", _on_resize)

        def _on_window_configure(event: tk.Event, k: str = kind, w: tk.Toplevel = window) -> None:
            if event.widget is not w:
                return
            self._maybe_snap_window(k)

        window.bind("<Configure>", _on_window_configure)

        if kind == "gating":
            isolate_frame = ttk.Frame(window)
            isolate_frame.pack(side="bottom", fill="x", pady=(4, 6))
            inner = ttk.Frame(isolate_frame)
            inner.pack(anchor="center")
            ttk.Label(
                inner,
                text="Isolate Functional States:",
                font=("Segoe UI", 9),
            ).grid(row=0, column=0, padx=(0, 8))
            buttons: dict[str, tk.Button] = {}
            for index, (key, label) in enumerate(
                (("m", "Na+ Open"), ("h", "Na+ Inactivated"), ("n", "K+ Open"))
            ):
                button = tk.Button(
                    inner,
                    text=label,
                    font=("Segoe UI", 9),
                    relief="raised",
                    bd=1,
                    padx=10,
                    pady=3,
                    highlightthickness=0,
                    takefocus=0,
                    command=lambda gate=key: self._toggle_gating_line(gate),
                )
                button.grid(row=0, column=1 + index, padx=3)
                buttons[key] = button
            state["isolate_buttons"] = buttons
            state["lines"] = {}

        self._graph_windows[kind] = state
        self._draw_graph(kind)
        self._refresh_save_targets()

    def _maybe_snap_window(self, kind: str) -> None:
        state = self._graph_windows.get(kind)
        if state is None or state.get("_snapping"):
            return
        window = state["window"]
        if not self._window_visible(window):
            return
        try:
            x = window.winfo_x()
            y = window.winfo_y()
            wd = window.winfo_width()
            ht = window.winfo_height()
        except tk.TclError:
            return

        new_x, new_y = x, y
        snap_x, snap_y = False, False
        for other_kind, other_state in self._graph_windows.items():
            if other_kind == kind:
                continue
            other = other_state["window"]
            if not self._window_visible(other):
                continue
            try:
                ox = other.winfo_x()
                oy = other.winfo_y()
                owd = other.winfo_width()
                oht = other.winfo_height()
            except tk.TclError:
                continue

            if not snap_x:
                if abs((x + wd) - ox) < SNAP_THRESHOLD_PX:
                    new_x, snap_x = ox - wd, True
                elif abs(x - (ox + owd)) < SNAP_THRESHOLD_PX:
                    new_x, snap_x = ox + owd, True
                elif abs(x - ox) < SNAP_THRESHOLD_PX:
                    new_x, snap_x = ox, True
                elif abs((x + wd) - (ox + owd)) < SNAP_THRESHOLD_PX:
                    new_x, snap_x = ox + owd - wd, True

            if not snap_y:
                if abs((y + ht) - oy) < SNAP_THRESHOLD_PX:
                    new_y, snap_y = oy - ht, True
                elif abs(y - (oy + oht)) < SNAP_THRESHOLD_PX:
                    new_y, snap_y = oy + oht, True
                elif abs(y - oy) < SNAP_THRESHOLD_PX:
                    new_y, snap_y = oy, True
                elif abs((y + ht) - (oy + oht)) < SNAP_THRESHOLD_PX:
                    new_y, snap_y = oy + oht - ht, True

            if snap_x and snap_y:
                break

        if new_x == x and new_y == y:
            return
        state["_snapping"] = True
        try:
            window.geometry(f"+{int(new_x)}+{int(new_y)}")
        except tk.TclError:
            pass
        window.after(40, lambda s=state: s.__setitem__("_snapping", False))

    def _on_graph_close(self, kind: str) -> None:
        state = self._graph_windows.pop(kind, None)
        if state is not None and state["window"].winfo_exists():
            state["window"].destroy()
        self._refresh_save_targets()

    def _redraw_all_graphs(self) -> None:
        for kind in list(self._graph_windows.keys()):
            self._draw_graph(kind)

    def init_view(self) -> None:
        if self.result is None:
            return
        self._visible_time_end_ms = 0.0
        self._apply_visible_time_end_to_graphs()

    def advance_view(self, step_ms: float) -> None:
        if self.result is None:
            return
        duration = float(self.result.time_ms[-1])
        current = self._visible_time_end_ms if self._visible_time_end_ms is not None else 0.0
        self._visible_time_end_ms = min(duration, current + step_ms)
        self._apply_visible_time_end_to_graphs()

    def reset_full_view(self) -> None:
        self._visible_time_end_ms = None
        self._apply_visible_time_end_to_graphs()

    def _apply_visible_time_end_to_graphs(self) -> None:
        if self.result is None:
            return
        t_start = float(self.result.time_ms[0])
        t_end = float(self.result.time_ms[-1])
        end = self._visible_time_end_ms
        if end is None:
            left, right = t_start, t_end
        else:
            span = max(end - t_start, 1.0)
            left = -0.5
            right = end + max(0.5, 0.09 * span)
        for state in self._graph_windows.values():
            state["axis"].set_xlim(left, right)
            state["figure"].tight_layout()
            state["canvas"].draw_idle()

    def _draw_graph(self, kind: str) -> None:
        state = self._graph_windows.get(kind)
        if state is None or self.result is None:
            return
        axis = state["axis"]
        axis.clear()

        draw_map = {
            "voltage": self._draw_voltage,
            "current": self._draw_injected_current,
            "net_ionic": self._draw_net_ionic_current,
            "conductance": self._draw_conductance,
            "gating": self._draw_gating,
        }
        draw_map[kind](state)
        state["figure"].tight_layout()
        state["canvas"].draw_idle()

    def _draw_voltage(self, state: dict) -> None:
        axis = state["axis"]
        axis.plot(
            self.result.time_ms,
            self.result.voltage_mV,
            color="black",
            linewidth=1.6,
            label="Voltage",
        )
        if self.current_parameters is not None:
            axis.axhline(self.current_parameters.e_na, color="black", linestyle="--", linewidth=1.0)
            axis.axhline(self.current_parameters.e_k, color="black", linestyle="--", linewidth=1.0)
            label_kwargs = dict(
                transform=axis.get_yaxis_transform(),
                ha="right",
                fontsize=8,
                bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85},
            )
            axis.text(0.995, self.current_parameters.e_na, "E_Na", va="bottom", **label_kwargs)
            axis.text(0.995, self.current_parameters.e_k, "E_K", va="top", **label_kwargs)
        axis.set_xlabel("Time (ms)")
        axis.set_ylabel("V (mV)")
        axis.set_title("Voltage")
        axis.grid(True, alpha=0.25)

    def _draw_injected_current(self, state: dict) -> None:
        axis = state["axis"]
        axis.plot(
            self.result.time_ms,
            self.result.injected_current_uA_cm2,
            color="black",
            linewidth=1.4,
            label="Injected current",
        )
        axis.set_xlabel("Time (ms)")
        axis.set_ylabel("I (uA/cm^2)")
        axis.set_title("Injected Current")
        axis.grid(True, alpha=0.25)

    def _draw_net_ionic_current(self, state: dict) -> None:
        axis = state["axis"]
        axis.plot(
            self.result.time_ms,
            self.result.net_ionic_current_uA_cm2,
            color="black",
            linewidth=1.4,
            label="I_Na + I_K + I_L",
        )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        axis.set_xlabel("Time (ms)")
        axis.set_ylabel("uA/cm^2")
        axis.set_title("Net Ionic Current")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", fontsize=8, framealpha=0.9)

    def _draw_conductance(self, state: dict) -> None:
        axis = state["axis"]
        axis.plot(self.result.time_ms, self.result.sodium_conductance_mS_cm2,
                  color="red", linewidth=1.5, label="g_Na m^3 h")
        axis.plot(self.result.time_ms, self.result.g_na_max_mS_cm2,
                  color="red", linewidth=1.0, linestyle="--", label="g_Na max(t)")
        axis.plot(self.result.time_ms, self.result.potassium_conductance_mS_cm2,
                  color="blue", linewidth=1.5, label="g_K n^4")
        axis.plot(self.result.time_ms, self.result.g_k_max_mS_cm2,
                  color="blue", linewidth=1.0, linestyle="--", label="g_K max(t)")
        axis.set_xlabel("Time (ms)")
        axis.set_ylabel("mS/cm^2")
        axis.set_title("Sodium and Potassium Conductances")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", fontsize=8, framealpha=0.9)

    def _draw_gating(self, state: dict) -> None:
        axis = state["axis"]
        lines: dict[str, object] = {}
        (lines["m"],) = axis.plot(self.result.time_ms, self.result.m,
                                  color="red", linewidth=1.4, label="Na+ Open State")
        (lines["h"],) = axis.plot(self.result.time_ms, self.result.h,
                                  color="green", linewidth=1.4, label="Na+ Inactivated State")
        (lines["n"],) = axis.plot(self.result.time_ms, self.result.n,
                                  color="blue", linewidth=1.4, label="K+ Open State")
        axis.set_xlabel("Time (ms)")
        axis.set_ylabel("Probability")
        axis.set_title("Ion Channel Functional States Probability")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", fontsize=8, framealpha=0.9)
        state["lines"] = lines
        self._apply_gating_visibility(state)
        self._update_gating_toggle_buttons(state)

    def _toggle_gating_line(self, gate: str) -> None:
        self._gating_visible[gate] = not self._gating_visible[gate]
        state = self._graph_windows.get("gating")
        if state is None:
            return
        self._apply_gating_visibility(state)
        self._update_gating_toggle_buttons(state)
        state["canvas"].draw_idle()

    def _apply_gating_visibility(self, state: dict) -> None:
        lines = state.get("lines", {})
        axis = state["axis"]
        visible_handles = []
        for key, line in lines.items():
            visible = self._gating_visible.get(key, True)
            line.set_visible(visible)
            if visible:
                visible_handles.append((line, line.get_label()))
        legend = axis.get_legend()
        if visible_handles:
            axis.legend(
                [h for h, _ in visible_handles],
                [lbl for _, lbl in visible_handles],
                loc="upper right",
                fontsize=8,
                framealpha=0.9,
            )
        elif legend is not None:
            legend.set_visible(False)

    def _update_gating_toggle_buttons(self, state: dict) -> None:
        buttons = state.get("isolate_buttons", {})
        for key, button in buttons.items():
            button.configure(relief="sunken" if self._gating_visible.get(key, True) else "raised")

    # ------------------------------------------------------- Save Plots tab

    def open_save_plots(self) -> None:
        if self._save_window is not None and self._save_window.winfo_exists():
            self._save_window.deiconify()
            self._save_window.lift()
            self._save_window.focus_force()
            self._refresh_save_targets()
            return

        window = tk.Toplevel(self.root)
        window.title("Save Plots")
        window.geometry("520x360")
        window.minsize(420, 260)
        self._save_window = window
        window.protocol("WM_DELETE_WINDOW", self._on_save_close)

        frame = ttk.Frame(window, padding=(16, 14, 16, 14))
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text="Save individual plots or every open figure. Set width/height in inches.",
            font=("Segoe UI", 9),
            foreground="#333",
            wraplength=470,
            justify="left",
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))

        ttk.Label(frame, text="Figure").grid(row=1, column=0, sticky="w")
        ttk.Label(frame, text="Width (in)").grid(row=1, column=1, sticky="w", padx=(8, 4))
        ttk.Label(frame, text="Height (in)").grid(row=1, column=2, sticky="w", padx=(4, 8))

        self._save_rows_frame = ttk.Frame(frame)
        self._save_rows_frame.grid(row=2, column=0, columnspan=4, sticky="nsew", pady=(4, 8))
        frame.rowconfigure(2, weight=1)
        frame.columnconfigure(0, weight=1)

        self._save_empty_label = ttk.Label(
            frame,
            text="No graphs are open. Open a graph via the Graphs menu, then return here.",
            foreground="#a03030",
            wraplength=470,
        )

        button_row = ttk.Frame(frame)
        button_row.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        button_row.columnconfigure((0, 1), weight=1)
        ttk.Button(button_row, text="Close", command=self._on_save_close).grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(button_row, text="Save Selected", command=self._save_selected).grid(
            row=0, column=1, sticky="ew", padx=(6, 0)
        )

        self._save_selected_var = tk.StringVar()
        self._save_dim_vars: dict[str, tuple[tk.StringVar, tk.StringVar]] = {}
        self._refresh_save_targets()

    def _on_save_close(self) -> None:
        self._snapshot_save_dimensions()
        if self._save_window is not None and self._save_window.winfo_exists():
            self._save_window.destroy()
        self._save_window = None
        self._save_dim_vars = {}

    def _snapshot_save_dimensions(self) -> None:
        for key, (w, h) in self._save_dim_vars.items():
            try:
                width = float(w.get())
                height = float(h.get())
            except ValueError:
                continue
            if key == "all":
                self._save_all_dimensions = (width, height)
            else:
                self._save_dimensions[key] = (width, height)

    def _refresh_save_targets(self) -> None:
        if self._save_window is None or not self._save_window.winfo_exists():
            return
        self._snapshot_save_dimensions()
        for child in self._save_rows_frame.winfo_children():
            child.destroy()
        self._save_dim_vars = {}

        open_kinds = [
            kind for kind in GRAPH_ORDER if kind in self._graph_windows
            and self._graph_windows[kind]["window"].winfo_exists()
        ]

        if not open_kinds:
            self._save_empty_label.grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 8))
            self._save_rows_frame.grid_remove()
            return

        self._save_empty_label.grid_remove()
        self._save_rows_frame.grid()

        if self._save_selected_var.get() not in open_kinds + ["all"]:
            self._save_selected_var.set(open_kinds[0])

        row = 0
        for kind in open_kinds:
            self._build_save_row(
                row, kind, GRAPH_LABELS[kind], self._save_dimensions.get(kind, DEFAULT_SAVE_DIMS)
            )
            row += 1
        self._build_save_row(
            row, "all", "All open figures (uses these dims for every save)",
            self._save_all_dimensions,
        )

    def _build_save_row(
        self,
        row: int,
        key: str,
        label: str,
        dims: tuple[float, float],
    ) -> None:
        radio = ttk.Radiobutton(
            self._save_rows_frame,
            text=label,
            variable=self._save_selected_var,
            value=key,
        )
        radio.grid(row=row, column=0, sticky="w", pady=2)
        width_var = tk.StringVar(value=f"{dims[0]:g}")
        height_var = tk.StringVar(value=f"{dims[1]:g}")
        ttk.Entry(self._save_rows_frame, textvariable=width_var, width=6).grid(
            row=row, column=1, sticky="w", padx=(8, 4)
        )
        ttk.Entry(self._save_rows_frame, textvariable=height_var, width=6).grid(
            row=row, column=2, sticky="w", padx=(4, 8)
        )
        self._save_dim_vars[key] = (width_var, height_var)
        self._save_rows_frame.columnconfigure(0, weight=1)

    def _save_selected(self) -> None:
        if self.result is None:
            messagebox.showwarning("No simulation", "Run a simulation first.", parent=self._save_window)
            return
        selection = self._save_selected_var.get()
        if not selection:
            messagebox.showwarning("No selection", "Select a figure to save.", parent=self._save_window)
            return

        self._snapshot_save_dimensions()

        if selection == "all":
            open_kinds = [
                kind for kind in GRAPH_ORDER if kind in self._graph_windows
                and self._graph_windows[kind]["window"].winfo_exists()
            ]
            if not open_kinds:
                messagebox.showwarning("No open graphs", "Open one or more graphs first.", parent=self._save_window)
                return
            directory = filedialog.askdirectory(
                title="Choose folder for all figures",
                initialdir=str(PROJECT_ROOT / "output"),
                parent=self._save_window,
            )
            if not directory:
                return
            output_dir = Path(directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            width, height = self._save_all_dimensions
            for kind in open_kinds:
                path = output_dir / f"{GRAPH_STEMS[kind]}.png"
                self._render_graph_to_file(kind, path, width, height)
            self.status_var.set(f"Saved {len(open_kinds)} figure(s) to {output_dir}")
            return

        default_path = PROJECT_ROOT / "output" / f"{GRAPH_STEMS[selection]}.png"
        path = filedialog.asksaveasfilename(
            title=f"Save {GRAPH_LABELS[selection]}",
            initialdir=str(default_path.parent),
            initialfile=default_path.name,
            defaultextension=".png",
            filetypes=(("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")),
            parent=self._save_window,
        )
        if not path:
            return
        width, height = self._save_dimensions.get(selection, DEFAULT_SAVE_DIMS)
        self._render_graph_to_file(selection, Path(path), width, height)
        self.status_var.set(f"Saved {GRAPH_LABELS[selection]}: {path}")

    def _render_graph_to_file(
        self,
        kind: str,
        path: Path,
        width: float,
        height: float,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure = Figure(figsize=(max(1.0, width), max(1.0, height)), dpi=200)
        axis = figure.add_subplot(111)
        temp_state = {"figure": figure, "axis": axis, "lines": {}}
        draw_map = {
            "voltage": self._draw_voltage,
            "current": self._draw_injected_current,
            "net_ionic": self._draw_net_ionic_current,
            "conductance": self._draw_conductance,
            "gating": self._draw_gating,
        }
        draw_map[kind](temp_state)
        if kind == "gating":
            self._apply_gating_visibility(temp_state)
        figure.tight_layout()
        figure.savefig(path, dpi=200, bbox_inches="tight")

    # ------------------------------------------------------ Export CSV / Help

    def export_csv(self) -> None:
        if self.result is None:
            messagebox.showwarning("No simulation", "Run a simulation before exporting.",
                                   parent=self._parameters_window or self.root)
            return
        default_path = PROJECT_ROOT / "output" / "interactive_simulation.csv"
        path = filedialog.asksaveasfilename(
            title="Export simulation CSV",
            initialdir=str(default_path.parent),
            initialfile=default_path.name,
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
            parent=self._parameters_window or self.root,
        )
        if not path:
            return
        export_simulation_csv(self.result, path)
        self.status_var.set(f"CSV exported: {path}")

    def show_help(self) -> None:
        help_window = tk.Toplevel(self.root)
        help_window.title("Parameters and Interpreting Graphs")
        help_window.transient(self.root)
        self._center_window(help_window, 1080, 840)

        page = self._create_help_page(help_window)
        self._build_graph_help(page)

    def show_graph_equations(self) -> None:
        equations_window = tk.Toplevel(self.root)
        equations_window.title("Graph Equations")
        equations_window.transient(self.root)
        self._center_window(equations_window, 1080, 760)

        page = self._create_help_page(equations_window)
        row = 0
        row = self._add_help_heading(page, row, "Graph Equations")
        row = self._add_help_figure(page, row, self._build_graph_equations_figure())
        self._add_help_body(
            page,
            row,
            "The net-current graph displays only the net sum. Sodium, potassium, and leak are "
            "still used to compute that sum and remain available in CSV export.",
        )

    def _center_window(self, window: tk.Toplevel, width: int, height: int) -> None:
        window.update_idletasks()
        x = max(0, (window.winfo_screenwidth() - width) // 2)
        y = max(0, (window.winfo_screenheight() - height) // 2)
        window.geometry(f"{width}x{height}+{x}+{y}")

    def _create_help_page(self, parent: tk.Widget) -> ttk.Frame:
        canvas = tk.Canvas(parent, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas, padding=(18, 16, 18, 20))
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def update_scroll_region(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def update_width(event: tk.Event) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        content.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", update_width)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        return content

    def _add_help_heading(self, parent: ttk.Frame, row: int, text: str) -> int:
        ttk.Label(parent, text=text, style="HelpHeading.TLabel").grid(
            row=row, column=0, sticky="w", pady=(0, 4)
        )
        return row + 1

    def _add_help_body(self, parent: ttk.Frame, row: int, text: str) -> int:
        ttk.Label(
            parent, text=text, style="HelpBody.TLabel",
            justify="left", wraplength=1000,
        ).grid(row=row, column=0, sticky="w", pady=(0, 12))
        return row + 1

    def _add_help_figure(self, parent: ttk.Frame, row: int, figure: Figure) -> int:
        canvas = FigureCanvasTkAgg(figure, master=parent)
        self._help_canvases.append(canvas)
        canvas.draw()
        canvas.get_tk_widget().grid(row=row, column=0, sticky="ew", pady=(0, 14))
        return row + 1

    def _build_graph_equations_figure(self) -> Figure:
        figure = Figure(figsize=(10.0, 6.6), dpi=160, facecolor="white")
        axis = figure.add_subplot(111)
        axis.axis("off")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)

        axis.text(0.02, 0.96, "Voltage", fontsize=12, fontweight="bold", va="top")
        axis.text(0.02, 0.90, r"$C_m\frac{dV}{dt}=I_{\mathrm{inj}}(t)-I_{\mathrm{net}}(t)$", fontsize=16, va="top")
        axis.text(0.02, 0.82, r"$I_{\mathrm{net}}=I_{\mathrm{Na}}+I_{\mathrm{K}}+I_{\mathrm{L}}$", fontsize=14, va="top")

        axis.text(0.02, 0.70, "Injected Current", fontsize=12, fontweight="bold", va="top")
        axis.text(0.02, 0.64, r"$I_{\mathrm{inj}}(t)=I_{\mathrm{baseline}}+\sum_p I_p(t)$", fontsize=15, va="top")
        axis.text(0.02, 0.57, r"$I_p(t)=A_p$ inside its pulse window, otherwise $0$", fontsize=12, va="top")

        axis.text(0.02, 0.45, "Conductances", fontsize=12, fontweight="bold", va="top")
        axis.text(0.02, 0.39, r"$g_{\mathrm{Na}}(t)=g_{\mathrm{Na,max}}(t)\,m^3h$", fontsize=15, va="top")
        axis.text(0.02, 0.31, r"$g_{\mathrm{K}}(t)=g_{\mathrm{K,max}}(t)\,n^4$", fontsize=15, va="top")

        axis.text(0.52, 0.96, "Ion Channel Functional States", fontsize=12, fontweight="bold", va="top")
        axis.text(0.52, 0.90, r"$\frac{dm}{dt}=\alpha_m(V)(1-m)-\beta_m(V)m$", fontsize=14, va="top")
        axis.text(0.52, 0.82, r"$\frac{dh}{dt}=\alpha_h(V)(1-h)-\beta_h(V)h$", fontsize=14, va="top")
        axis.text(0.52, 0.74, r"$\frac{dn}{dt}=\alpha_n(V)(1-n)-\beta_n(V)n$", fontsize=14, va="top")

        axis.text(0.52, 0.58, "Net Ionic Current", fontsize=12, fontweight="bold", va="top")
        axis.text(0.52, 0.52, r"$I_{\mathrm{Na}}=g_{\mathrm{Na,max}}(t)m^3h(V-E_{\mathrm{Na}})$", fontsize=13, va="top")
        axis.text(0.52, 0.44, r"$I_{\mathrm{K}}=g_{\mathrm{K,max}}(t)n^4(V-E_{\mathrm{K}})$", fontsize=13, va="top")
        axis.text(0.52, 0.36, r"$I_{\mathrm{L}}=g_{\mathrm{L}}(V-E_{\mathrm{L}})$", fontsize=13, va="top")
        axis.text(0.52, 0.28, r"$I_{\mathrm{net}}=I_{\mathrm{Na}}+I_{\mathrm{K}}+I_{\mathrm{L}}$", fontsize=14, va="top")
        return figure

    def _build_graph_help(self, parent: ttk.Frame) -> None:
        row = 0
        row = self._add_help_heading(parent, row, "Parameters")
        row = self._add_help_body(
            parent, row,
            "Resting V initializes the gate variables at steady state. Initial V is the membrane voltage at t = 0.",
        )
        row = self._add_help_body(
            parent, row,
            "Baseline current is always present. The main pulse and additional pulses are added on top of "
            "baseline. Conductance changes replace g_Na,max or g_K,max only during their start/end windows.",
        )
        row = self._add_help_body(
            parent, row,
            "Trace Metrics report Max V, Min V, spike count, firing rate, and first spike time. "
            "Spike detection uses upward crossings of 0 mV with a 2 ms refractory window.",
        )

        row = self._add_help_heading(parent, row, "Voltage Graph")
        row = self._add_help_body(
            parent, row,
            "Shows membrane voltage over time. The E_Na and E_K reference lines mark the sodium and "
            "potassium reversal potentials.",
        )

        row = self._add_help_heading(parent, row, "Injected Current Graph")
        row = self._add_help_body(
            parent, row,
            "Shows the current command delivered to the model. Overlapping pulses sum, and baseline is "
            "included for the full simulation.",
        )

        row = self._add_help_heading(parent, row, "Conductance Graph")
        row = self._add_help_body(
            parent, row,
            "Dashed lines are the scheduled maximum conductances: g_Na,max(t) and g_K,max(t). Solid lines "
            "are the actual conductances used by the model: g_Na,max(t)m^3h and g_K,max(t)n^4.",
        )

        row = self._add_help_heading(parent, row, "Net Ionic Current Graph")
        row = self._add_help_body(
            parent, row,
            "Shows only the total ionic current in black. It is computed as I_Na + I_K + I_L using the "
            "simulator's outward-positive sign convention.",
        )

        row = self._add_help_heading(parent, row, "Ion Channel Functional States")
        row = self._add_help_body(
            parent, row,
            "Na+ Open State corresponds to sodium activation (m), Na+ Inactivated State to sodium "
            "inactivation (h), and K+ Open State to potassium activation (n). Use the Isolate buttons "
            "to hide any subset of the three traces.",
        )

        row = self._add_help_heading(parent, row, "Saving")
        row = self._add_help_body(
            parent, row,
            "Save Plots lists the currently-open graph windows. Set individual width/height in inches for "
            "each figure, or use the 'All open figures' entry to save every open graph at one dimension.",
        )


def main() -> None:
    root = tk.Tk()
    HodgkinHuxleySimulatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
