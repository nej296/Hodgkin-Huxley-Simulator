"""Interactive desktop Hodgkin-Huxley membrane dynamics simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

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


class SimulatorNavigationToolbar(NavigationToolbar2Tk):
    """Matplotlib toolbar with only the controls useful for this simulator."""

    toolitems = tuple(
        item
        for item in NavigationToolbar2Tk.toolitems
        if item[0] not in {"Pan", "Zoom", "Subplots"}
    )

    def __init__(self, canvas, window, *, pack_toolbar=True, reset_callback=None):
        self._reset_callback = reset_callback
        super().__init__(canvas, window, pack_toolbar=pack_toolbar)

    def home(self, *args):
        if self._reset_callback is not None:
            self._reset_callback()
            return
        super().home(*args)


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


@dataclass
class PlotPanState:
    """State captured while the pan tool is dragging."""

    axis: object
    press_x: float
    press_y: float
    x_limits: dict[object, tuple[float, float]]
    y_limits: dict[object, tuple[float, float]]
    x_scale: float
    y_scale: float


class HodgkinHuxleySimulatorApp:
    """Tkinter GUI for interactively running HH membrane simulations."""

    def __init__(self, root: tk.Tk) -> None:
        """Build the simulator window and run the initial default simulation."""

        self.root = root
        self.root.title("Hodgkin-Huxley Membrane Dynamics Simulator")
        self.root.geometry("1800x930")
        self.root.minsize(1460, 760)

        self.defaults = ControlDefaults()
        self.variables: dict[str, tk.StringVar] = {}
        self.extra_pulse_rows: list[tuple[tk.StringVar, tk.StringVar, tk.StringVar]] = []
        self.g_na_change_rows: list[tuple[tk.StringVar, tk.StringVar, tk.StringVar]] = []
        self.g_k_change_rows: list[tuple[tk.StringVar, tk.StringVar, tk.StringVar]] = []
        self.result: SimulationResult | None = None
        self.metrics: SpikeMetrics | None = None
        self.current_parameters: HodgkinHuxleyParameters | None = None
        self.visible_time_end_ms: float | None = None
        self.plot_mode: str | None = None
        self._pan_mode_button: tk.Button | None = None
        self._zoom_mode_button: tk.Button | None = None
        self._pan_mode_icon: tk.PhotoImage | None = None
        self._zoom_mode_icon: tk.PhotoImage | None = None
        self._pan_drag_state: PlotPanState | None = None
        self._help_canvases: list[FigureCanvasTkAgg] = []

        self._configure_style()
        self._build_layout()
        self.reset_defaults()
        self.run_simulation()

    def _configure_style(self) -> None:
        """Apply a restrained desktop style that keeps controls scannable."""

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

    def _build_layout(self) -> None:
        """Create the control panel, plot area, and status bar."""

        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        controls_container = ttk.Frame(self.root, padding=(12, 12, 0, 8))
        controls_container.grid(row=0, column=0, sticky="ns")
        controls = self._build_scrollable_controls(controls_container)

        plot_frame = ttk.Frame(self.root, padding=(0, 8, 12, 8))
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self._build_controls(controls)
        self._build_plot(plot_frame)

        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            padding=(12, 4),
        )
        status.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _build_scrollable_controls(self, parent: ttk.Frame) -> ttk.Frame:
        """Create a vertically scrollable control panel."""

        canvas = tk.Canvas(parent, width=430, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def update_scroll_region(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def update_width(event: tk.Event) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        content.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", update_width)
        canvas.grid(row=0, column=0, sticky="ns")
        scrollbar.grid(row=0, column=1, sticky="ns")
        parent.rowconfigure(0, weight=1)
        return content

    def _build_controls(self, parent: ttk.Frame) -> None:
        """Create grouped numeric controls for model, protocol, and output."""

        simulation = ttk.LabelFrame(parent, text="Simulation", style="Section.TLabelframe")
        simulation.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self._add_entry(simulation, "duration_ms", "Duration (ms)", 0)
        self._add_entry(simulation, "resting_voltage_mV", "Resting V (mV)", 1)
        self._add_entry(simulation, "initial_voltage_mV", "Initial V (mV)", 2)
        step_buttons = ttk.Frame(simulation)
        step_buttons.grid(row=3, column=0, columnspan=2, sticky="ew", padx=6, pady=(4, 6))
        step_buttons.columnconfigure((0, 1, 2, 3), weight=1)
        ttk.Button(step_buttons, text="Init", command=self.init_view).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 4),
        )
        ttk.Button(step_buttons, text="Step 0.5 ms", command=lambda: self.advance_view(0.5)).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=4,
        )
        ttk.Button(step_buttons, text="Step 5 ms", command=lambda: self.advance_view(5.0)).grid(
            row=0,
            column=2,
            sticky="ew",
            padx=4,
        )
        ttk.Button(step_buttons, text="Full", command=self.reset_original_view).grid(
            row=0,
            column=3,
            sticky="ew",
            padx=(4, 0),
        )

        current = ttk.LabelFrame(parent, text="Injected Current", style="Section.TLabelframe")
        current.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self._add_entry(current, "current_amplitude", "Amplitude (uA/cm^2)", 0)
        self._add_entry(current, "current_start_ms", "Start (ms)", 1)
        self._add_entry(current, "current_end_ms", "End (ms)", 2)
        self._add_entry(current, "current_baseline", "Baseline (uA/cm^2)", 3)
        pulse_frame = ttk.Frame(current)
        pulse_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=6, pady=(8, 4))
        pulse_frame.columnconfigure((0, 1, 2), weight=1)
        ttk.Label(pulse_frame, text="Additional pulses").grid(
            row=0,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(0, 4),
        )
        self.extra_pulses_container = ttk.Frame(pulse_frame)
        self.extra_pulses_container.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.extra_pulses_container.columnconfigure((0, 1, 2), weight=1, uniform="pulsecols")
        ttk.Button(pulse_frame, text="Add Pulse", command=self.add_pulse_row).grid(
            row=2,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(6, 0),
        )

        conductance = ttk.LabelFrame(parent, text="Conductances", style="Section.TLabelframe")
        conductance.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self._add_entry(conductance, "g_na", "g_Na max (mS/cm^2)", 0)
        self._add_entry(conductance, "g_k", "g_K max (mS/cm^2)", 1)
        self._add_entry(conductance, "g_l", "g_L (mS/cm^2)", 2)
        self._add_entry(conductance, "membrane_capacitance", "C_m (uF/cm^2)", 3)
        self.g_na_changes_container = self._add_schedule_section(
            conductance,
            "g_Na changes",
            "Add g_Na Change",
            self.add_g_na_change_row,
            4,
        )
        self.g_k_changes_container = self._add_schedule_section(
            conductance,
            "g_K changes",
            "Add g_K Change",
            self.add_g_k_change_row,
            5,
        )

        reversal = ttk.LabelFrame(parent, text="Reversal Potentials", style="Section.TLabelframe")
        reversal.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self._add_entry(reversal, "e_na", "E_Na (mV)", 0)
        self._add_entry(reversal, "e_k", "E_K (mV)", 1)
        self._add_entry(reversal, "e_l", "E_L (mV)", 2)

        buttons = ttk.Frame(parent)
        buttons.grid(row=4, column=0, sticky="ew", pady=(2, 10))
        buttons.columnconfigure((0, 1), weight=1)
        ttk.Button(
            buttons,
            text="Run Simulation",
            style="Run.TButton",
            command=self.run_simulation,
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Button(buttons, text="Export CSV", command=self.export_csv).grid(
            row=1,
            column=0,
            sticky="ew",
            padx=(0, 4),
        )
        ttk.Button(buttons, text="Save Plot", command=self.save_plot).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(4, 0),
        )
        ttk.Button(buttons, text="Reset Defaults", command=self.reset_defaults).grid(
            row=2,
            column=0,
            sticky="ew",
            padx=(0, 4),
            pady=(6, 0),
        )
        ttk.Button(buttons, text="Help", command=self.show_help).grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(4, 0),
            pady=(6, 0),
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

    def _add_entry(self, parent: ttk.Frame, name: str, label: str, row: int) -> None:
        """Add a labeled numeric entry and store its StringVar by name."""

        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        variable = tk.StringVar()
        entry = ttk.Entry(parent, textvariable=variable, width=14)
        entry.grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        entry.bind("<Return>", lambda _event: self.run_simulation())
        parent.columnconfigure(1, weight=1)
        self.variables[name] = variable

    def _add_small_entry(
        self,
        parent: ttk.Frame,
        variable: tk.StringVar,
        row: int,
        column: int,
        title: str,
        suffix: str,
    ) -> None:
        """Add a compact value field with a unit suffix."""

        frame = ttk.Frame(parent)
        frame.grid(row=row, column=column, sticky="ew", padx=2, pady=2)
        frame.columnconfigure(0, weight=1)
        title_padx = (0, 56)
        if title in {"Amplitude", "Value"}:
            title_padx = (0, 74)
        ttk.Label(frame, text=title, anchor="center").grid(
            row=0,
            column=0,
            sticky="ew",
            padx=title_padx,
            pady=(0, 2),
        )
        entry_row = ttk.Frame(frame)
        entry_row.grid(row=1, column=0, sticky="ew")
        entry_row.columnconfigure(0, weight=1)
        entry = ttk.Entry(entry_row, textvariable=variable, width=7)
        entry.grid(row=0, column=0, sticky="ew")
        entry.bind("<Return>", lambda _event: self.run_simulation())
        ttk.Label(entry_row, text=suffix).grid(row=0, column=1, sticky="w", padx=(4, 0))

    def _header_padx_for_unit(self, unit: str, extra_right: int = 0) -> tuple[int, int]:
        """Shift a section header left so it sits over the entry box, not the unit."""

        font_name = ttk.Style().lookup("TLabel", "font") or "TkDefaultFont"
        font = tkfont.nametofont(font_name)
        return (0, max(12, font.measure(unit) + 4 + extra_right))

    def _add_schedule_section(
        self,
        parent: ttk.Frame,
        title: str,
        button_text: str,
        command,
        row: int,
    ) -> ttk.Frame:
        """Add a labeled row-input section for conductance changes."""

        section = ttk.Frame(parent)
        section.grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=(8, 4))
        section.columnconfigure((0, 1), weight=1)
        ttk.Label(section, text=title).grid(row=0, column=0, columnspan=2, sticky="w")
        container = ttk.Frame(section)
        container.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        container.columnconfigure((0, 1, 2), weight=1, uniform="changecols")
        ttk.Button(section, text=button_text, command=command).grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(6, 0),
        )
        return container

    def add_pulse_row(
        self,
        start_ms: str = "",
        end_ms: str = "",
        amplitude: str = "",
    ) -> None:
        """Add one explicit additional-current-pulse row."""

        row = len(self.extra_pulse_rows)
        start_var = tk.StringVar(value=start_ms)
        end_var = tk.StringVar(value=end_ms)
        amplitude_var = tk.StringVar(value=amplitude)
        self.extra_pulse_rows.append((start_var, end_var, amplitude_var))
        self._add_small_entry(self.extra_pulses_container, start_var, row, 0, "Start", "ms")
        self._add_small_entry(self.extra_pulses_container, end_var, row, 1, "End", "ms")
        self._add_small_entry(self.extra_pulses_container, amplitude_var, row, 2, "Amplitude", "uA/cm^2")

    def add_g_na_change_row(
        self,
        time_ms: str = "",
        end_ms: str = "",
        value: str = "",
    ) -> None:
        """Add one g_Na schedule row."""

        self._add_conductance_change_row(
            self.g_na_changes_container,
            self.g_na_change_rows,
            time_ms,
            end_ms,
            value,
            "mS/cm^2",
        )

    def add_g_k_change_row(
        self,
        time_ms: str = "",
        end_ms: str = "",
        value: str = "",
    ) -> None:
        """Add one g_K schedule row."""

        self._add_conductance_change_row(
            self.g_k_changes_container,
            self.g_k_change_rows,
            time_ms,
            end_ms,
            value,
            "mS/cm^2",
        )

    def _add_conductance_change_row(
        self,
        parent: ttk.Frame,
        rows: list[tuple[tk.StringVar, tk.StringVar, tk.StringVar]],
        time_ms: str,
        end_ms: str,
        value: str,
        unit: str,
    ) -> None:
        """Add one explicit conductance schedule row."""

        row = len(rows)
        time_var = tk.StringVar(value=time_ms)
        end_var = tk.StringVar(value=end_ms)
        value_var = tk.StringVar(value=value)
        rows.append((time_var, end_var, value_var))
        self._add_small_entry(parent, time_var, row, 0, "Time", "ms")
        self._add_small_entry(parent, end_var, row, 1, "End", "ms")
        self._add_small_entry(parent, value_var, row, 2, "Value", unit)

    def reset_defaults(self) -> None:
        """Restore controls to the classic Hodgkin-Huxley default protocol."""

        for field_name, value in self.defaults.__dict__.items():
            self.variables[field_name].set(str(value))
        self._clear_row_inputs(self.extra_pulses_container, self.extra_pulse_rows)
        self._clear_row_inputs(self.g_na_changes_container, self.g_na_change_rows)
        self._clear_row_inputs(self.g_k_changes_container, self.g_k_change_rows)
        self.add_pulse_row()
        self.add_g_na_change_row()
        self.add_g_k_change_row()
        self.visible_time_end_ms = None
        self.status_var.set("Default HH parameters loaded")

    def _clear_row_inputs(self, parent: ttk.Frame, rows: list) -> None:
        """Remove all dynamic input rows from a container."""

        for child in parent.winfo_children():
            child.destroy()
        rows.clear()

    def _read_float(self, name: str) -> float:
        """Parse a float control and raise a clear error for invalid input."""

        try:
            return float(self.variables[name].get())
        except ValueError as exc:
            raise ValueError(f"{name} must be a number.") from exc

    def _parse_extra_current_pulses(self) -> tuple[CurrentPulse, ...]:
        """Parse additional current pulses from explicit row inputs."""

        pulses: list[CurrentPulse] = []
        for start_var, end_var, amplitude_var in self.extra_pulse_rows:
            values = (start_var.get().strip(), end_var.get().strip(), amplitude_var.get().strip())
            if not any(values):
                continue
            if not all(values):
                raise ValueError("Additional pulse rows must include start, end, and amplitude.")
            start_ms, end_ms, amplitude = (float(value) for value in values)
            pulses.append(CurrentPulse(amplitude=amplitude, start_ms=start_ms, end_ms=end_ms))
        return tuple(pulses)

    def _parse_conductance_changes(
        self,
        rows: list[tuple[tk.StringVar, tk.StringVar, tk.StringVar]],
        label: str,
    ) -> tuple[ConductanceChange, ...]:
        """Parse scheduled maximum-conductance changes from row inputs."""

        changes: list[ConductanceChange] = []
        for time_var, end_var, value_var in rows:
            values = (time_var.get().strip(), end_var.get().strip(), value_var.get().strip())
            if not any(values):
                continue
            if not all(values):
                raise ValueError(f"{label} rows must include time, end, and conductance.")
            time_ms, end_ms, value = (float(item) for item in values)
            changes.append(ConductanceChange(time_ms=time_ms, end_ms=end_ms, value=value))
        return tuple(changes)

    def _build_simulation_objects(
        self,
    ) -> tuple[
        HodgkinHuxleyNeuron,
        SimulationConfig,
        MultiPulseCurrent,
        HodgkinHuxleyState,
        ConductanceSchedule,
        ConductanceSchedule,
    ]:
        """Convert GUI control values into model, config, protocol, and schedules."""

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
            changes=self._parse_conductance_changes(self.g_na_change_rows, "g_Na changes"),
        )
        g_k_schedule = ConductanceSchedule(
            base_value=parameters.g_k,
            changes=self._parse_conductance_changes(self.g_k_change_rows, "g_K changes"),
        )
        return neuron, config, protocol, initial_state, g_na_schedule, g_k_schedule

    def run_simulation(self) -> None:
        """Run the HH model with current controls and redraw the membrane trace."""

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
            )
            self.metrics = summarize_voltage_trace(self.result)
            self.current_parameters = neuron.parameters
        except Exception as exc:  # noqa: BLE001 - GUI should report validation errors.
            messagebox.showerror("Simulation error", str(exc))
            self.status_var.set("Simulation failed")
            return

        self.visible_time_end_ms = None
        self._pan_drag_state = None
        self._draw_result()
        self._update_metrics()
        self.status_var.set("Simulation complete")

    def advance_view(self, step_ms: float) -> None:
        """Advance the visible trace window by a fixed number of milliseconds."""

        if self.result is None:
            self.run_simulation()
        if self.result is None:
            return
        duration = float(self.result.time_ms[-1])
        current_end = self.visible_time_end_ms if self.visible_time_end_ms is not None else 0.0
        self.visible_time_end_ms = min(duration, current_end + step_ms)
        self._draw_result()
        self.status_var.set(f"Showing 0 to {self.visible_time_end_ms:.1f} ms")

    def init_view(self) -> None:
        """Reset the visible trace to the beginning of the current simulation."""

        if self.result is None:
            self.run_simulation()
        if self.result is None:
            return
        self.visible_time_end_ms = 0.0
        self._draw_result()
        self.status_var.set("Showing initial time point")

    def reset_original_view(self) -> None:
        """Return to the full unzoomed simulation view and clear plot modes."""

        self.plot_mode = None
        self._pan_drag_state = None
        self._update_plot_mode_buttons()
        if self.result is None:
            self.run_simulation()
            return
        self.visible_time_end_ms = None
        self._draw_result()
        self.canvas.draw()
        self.status_var.set("Showing full trace")

    def show_full_trace(self) -> None:
        """Backward-compatible alias for the reset-view action."""

        self.reset_original_view()

    def _build_plot(self, parent: ttk.Frame) -> None:
        """Create the embedded Matplotlib figure used for all simulations."""

        self.figure = Figure(figsize=(12.8, 7.8), dpi=100)
        grid = self.figure.add_gridspec(
            2,
            2,
            width_ratios=(1.35, 1.0),
            wspace=0.34,
            hspace=0.28,
        )
        self.voltage_axis = self.figure.add_subplot(grid[0, 0])
        self.current_axis = self.figure.add_subplot(grid[1, 0], sharex=self.voltage_axis)
        self.sodium_conductance_axis = self.figure.add_subplot(grid[0, 1], sharex=self.voltage_axis)
        self.potassium_conductance_axis = self.figure.add_subplot(grid[1, 1], sharex=self.voltage_axis)

        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        toolbar_frame.columnconfigure(1, weight=1)

        mode_frame = ttk.Frame(toolbar_frame)
        mode_frame.grid(row=0, column=0, sticky="w", padx=(0, 8))
        self._pan_mode_icon = self._create_pan_icon()
        self._zoom_mode_icon = self._create_zoom_icon()
        self._pan_mode_button = tk.Button(
            mode_frame,
            image=self._pan_mode_icon,
            command=lambda: self.toggle_plot_mode("pan"),
            relief="raised",
            bd=1,
            padx=4,
            pady=4,
            highlightthickness=0,
            takefocus=0,
        )
        self._pan_mode_button.grid(row=0, column=0, padx=(0, 4))
        self._zoom_mode_button = tk.Button(
            mode_frame,
            image=self._zoom_mode_icon,
            command=lambda: self.toggle_plot_mode("zoom"),
            relief="raised",
            bd=1,
            padx=4,
            pady=4,
            highlightthickness=0,
            takefocus=0,
        )
        self._zoom_mode_button.grid(row=0, column=1)

        nav_frame = ttk.Frame(toolbar_frame)
        nav_frame.grid(row=0, column=1, sticky="w")
        SimulatorNavigationToolbar(
            self.canvas,
            nav_frame,
            pack_toolbar=False,
            reset_callback=self.reset_original_view,
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )
        self.canvas.mpl_connect("button_press_event", self._on_plot_button_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
        self.canvas.mpl_connect("button_release_event", self._on_plot_button_release)

    def _plot_axes(self) -> tuple:
        """Return all axes that participate in plot interactions."""

        return (
            self.voltage_axis,
            self.current_axis,
            self.sodium_conductance_axis,
            self.potassium_conductance_axis,
        )

    def _create_pan_icon(self) -> tk.PhotoImage:
        """Create a small crosshair-style pan icon."""

        image = tk.PhotoImage(width=18, height=18)
        color = "#2b2b2b"
        for x in range(3, 15):
            image.put(color, (x, 9))
        for y in range(3, 15):
            image.put(color, (9, y))
        for offset in range(-1, 2):
            image.put(color, (9 + offset, 9))
            image.put(color, (9, 9 + offset))
        return image

    def _create_zoom_icon(self) -> tk.PhotoImage:
        """Create a small magnifying-glass-style zoom icon."""

        image = tk.PhotoImage(width=18, height=18)
        color = "#2b2b2b"
        cx = 7
        cy = 7
        radius = 4
        for x in range(18):
            for y in range(18):
                distance = math.hypot(x - cx, y - cy)
                if radius - 0.65 <= distance <= radius + 0.65:
                    image.put(color, (x, y))
        for offset in range(6):
            image.put(color, (11 + offset, 11 + offset))
        for offset in range(2):
            image.put(color, (11 + offset, 10 + offset))
        return image

    def toggle_plot_mode(self, mode: str) -> None:
        """Toggle the active plot interaction mode."""

        self.plot_mode = None if self.plot_mode == mode else mode
        self._update_plot_mode_buttons()

    def _update_plot_mode_buttons(self) -> None:
        """Reflect the active plot mode on the toolbar buttons."""

        if self._pan_mode_button is not None:
            self._pan_mode_button.configure(relief="sunken" if self.plot_mode == "pan" else "raised")
        if self._zoom_mode_button is not None:
            self._zoom_mode_button.configure(relief="sunken" if self.plot_mode == "zoom" else "raised")

    def _on_plot_button_press(self, event) -> None:
        """Apply a zoom or pan action on a single click."""

        if event.inaxes not in self._plot_axes():
            return
        if self.plot_mode not in {"pan", "zoom"}:
            return
        if self.plot_mode == "pan":
            if event.button != 1:
                return
            if event.x is None or event.y is None:
                return
            x_limits = {axis: axis.get_xlim() for axis in self._plot_axes()}
            y_limits = {axis: axis.get_ylim() for axis in self._plot_axes()}
            bbox = event.inaxes.bbox
            self._pan_drag_state = PlotPanState(
                axis=event.inaxes,
                press_x=float(event.x),
                press_y=float(event.y),
                x_limits=x_limits,
                y_limits=y_limits,
                x_scale=(x_limits[event.inaxes][1] - x_limits[event.inaxes][0]) / max(bbox.width, 1.0),
                y_scale=(y_limits[event.inaxes][1] - y_limits[event.inaxes][0]) / max(bbox.height, 1.0),
            )
        elif self.plot_mode == "zoom":
            if event.xdata is None or event.ydata is None:
                return
            if event.button == 1:
                self._zoom_to_click(event.inaxes, float(event.xdata), float(event.ydata), 0.5)
            elif event.button == 3:
                self._zoom_to_click(event.inaxes, float(event.xdata), float(event.ydata), 2.0)

    def _on_plot_motion(self, event) -> None:
        """Pan the plot while the pan mode is held down and dragged."""

        state = self._pan_drag_state
        if state is None or self.plot_mode != "pan":
            return
        if event.x is None or event.y is None:
            return

        delta_x = (float(event.x) - state.press_x) * state.x_scale
        delta_y = (float(event.y) - state.press_y) * state.y_scale
        for axis, (x0, x1) in state.x_limits.items():
            axis.set_xlim(x0 - delta_x, x1 - delta_x)
        y0, y1 = state.y_limits[state.axis]
        state.axis.set_ylim(y0 - delta_y, y1 - delta_y)
        self.canvas.draw_idle()

    def _on_plot_button_release(self, _event) -> None:
        """Clear the drag state after a pan interaction."""

        self._pan_drag_state = None

    def _zoom_to_click(self, axis, center_x: float, center_y: float, factor: float) -> None:
        """Zoom in or out around the clicked point."""

        x0, x1 = axis.get_xlim()
        y0, y1 = axis.get_ylim()
        x_span = (x1 - x0) * factor
        y_span = (y1 - y0) * factor
        new_x0 = center_x - x_span / 2.0
        new_x1 = center_x + x_span / 2.0
        new_y0 = center_y - y_span / 2.0
        new_y1 = center_y + y_span / 2.0
        for target_axis in self._plot_axes():
            target_axis.set_xlim(new_x0, new_x1)
        axis.set_ylim(new_y0, new_y1)
        self.canvas.draw_idle()

    def _draw_result(self) -> None:
        """Update voltage, current, conductance, and gating plots."""

        if self.result is None:
            return

        axes = (
            self.voltage_axis,
            self.current_axis,
            self.sodium_conductance_axis,
            self.potassium_conductance_axis,
        )
        for axis in axes:
            axis.clear()

        time_start = float(self.result.time_ms[0])
        time_end = float(self.result.time_ms[-1])
        visible_end = self.visible_time_end_ms if self.visible_time_end_ms is not None else time_end
        visible_end = min(visible_end, time_end)
        time_span = max(visible_end - time_start, 1.0)
        label_x = visible_end + 0.015 * time_span
        right_margin = max(0.5, 0.09 * time_span)
        left_bound = -0.5

        self.voltage_axis.plot(
            self.result.time_ms,
            self.result.voltage_mV,
            color="#1f5f99",
            linewidth=1.6,
        )
        if self.current_parameters is not None:
            self.voltage_axis.axhline(
                self.current_parameters.e_na,
                color="black",
                linestyle="--",
                linewidth=1.0,
            )
            self.voltage_axis.axhline(
                self.current_parameters.e_k,
                color="black",
                linestyle="--",
                linewidth=1.0,
            )
            self.voltage_axis.annotate(
                "E_Na",
                xy=(label_x, self.current_parameters.e_na),
                xytext=(0, -4),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=8,
                color="black",
            )
            self.voltage_axis.annotate(
                "E_K",
                xy=(label_x, self.current_parameters.e_k),
                xytext=(0, 4),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=8,
                color="black",
            )
        self.voltage_axis.set_ylabel("V (mV)")
        self.voltage_axis.set_title("Voltage")
        self.voltage_axis.grid(True, alpha=0.25)

        self.current_axis.plot(
            self.result.time_ms,
            self.result.injected_current_uA_cm2,
            color="#7a3b12",
            linewidth=1.4,
        )
        self.current_axis.set_ylabel("I (uA/cm^2)")
        self.current_axis.set_xlabel("Time (ms)")
        self.current_axis.set_title("Injected Current")
        self.current_axis.grid(True, alpha=0.25)

        self.sodium_conductance_axis.plot(
            self.result.time_ms,
            self.result.sodium_conductance_mS_cm2,
            color="#255f85",
            linewidth=1.5,
            label="g_Na m^3 h",
        )
        self.sodium_conductance_axis.plot(
            self.result.time_ms,
            self.result.g_na_max_mS_cm2,
            color="#6c7a89",
            linewidth=1.0,
            linestyle="--",
            label="g_Na max(t)",
        )
        self.sodium_conductance_axis.set_ylabel("mS/cm^2")
        self.sodium_conductance_axis.set_title("Sodium Conductance")
        self.sodium_conductance_axis.grid(True, alpha=0.25)
        self.sodium_conductance_axis.legend(
            loc="upper right",
            bbox_to_anchor=(1.0, 0.92),
            fontsize=7,
            framealpha=0.9,
            borderaxespad=0.2,
        )

        self.potassium_conductance_axis.plot(
            self.result.time_ms,
            self.result.potassium_conductance_mS_cm2,
            color="#386641",
            linewidth=1.5,
            label="g_K n^4",
        )
        self.potassium_conductance_axis.plot(
            self.result.time_ms,
            self.result.g_k_max_mS_cm2,
            color="#6c7a89",
            linewidth=1.0,
            linestyle="--",
            label="g_K max(t)",
        )
        self.potassium_conductance_axis.set_ylabel("mS/cm^2")
        self.potassium_conductance_axis.set_xlabel("Time (ms)")
        self.potassium_conductance_axis.set_title("Potassium Conductance")
        self.potassium_conductance_axis.grid(True, alpha=0.25)
        self.potassium_conductance_axis.legend(
            loc="upper right",
            bbox_to_anchor=(1.0, 0.92),
            fontsize=7,
            framealpha=0.9,
            borderaxespad=0.2,
        )

        for axis in axes:
            axis.set_xlim(left_bound, visible_end + right_margin)
        self.figure.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.94)
        self.canvas.draw_idle()

    def _update_metrics(self) -> None:
        """Display spike-count and voltage extrema for the latest simulation."""

        if self.metrics is None:
            self.metrics_var.set("")
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

    def show_help(self) -> None:
        """Open a compact chooser for the two help topics."""

        chooser = tk.Toplevel(self.root)
        chooser.title("Help")
        chooser.resizable(False, False)
        chooser.transient(self.root)
        chooser.grab_set()

        frame = ttk.Frame(chooser, padding=(18, 16, 18, 16))
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="Help", style="HelpTitle.TLabel").grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 6),
        )
        ttk.Label(
            frame,
            text="Choose a topic",
            style="HelpBody.TLabel",
            justify="center",
        ).grid(row=1, column=0, sticky="ew", pady=(0, 12))
        frame.rowconfigure(2, weight=1)

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=2, column=0, sticky="nsew")
        button_frame.columnconfigure(0, weight=1)
        button_box = ttk.Frame(button_frame)
        button_box.place(relx=0.5, rely=0.5, anchor="center")

        ttk.Button(
            button_box,
            text="Hodgkin-Huxley Equations",
            style="HelpChoice.TButton",
            command=lambda: self._open_help_topic(chooser, "Hodgkin-Huxley Equations", self._build_equations_help),
        ).grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(
            button_box,
            text="Parameters and Interpreting Graphs",
            style="HelpChoice.TButton",
            command=lambda: self._open_help_topic(
                chooser,
                "Parameters and Interpreting Graphs",
                self._build_graph_help,
            ),
        ).grid(row=1, column=0, sticky="ew")

        self._center_window(chooser, 640, 240)

    def _center_window(self, window: tk.Toplevel, width: int, height: int) -> None:
        """Center a window on the screen."""

        window.update_idletasks()
        x = max(0, (window.winfo_screenwidth() - width) // 2)
        y = max(0, (window.winfo_screenheight() - height) // 2)
        window.geometry(f"{width}x{height}+{x}+{y}")

    def _open_help_topic(
        self,
        chooser: tk.Toplevel,
        title: str,
        builder,
    ) -> None:
        """Open a dedicated help page for one topic."""

        chooser.destroy()
        help_window = tk.Toplevel(self.root)
        help_window.title(title)
        help_window.transient(self.root)
        self._center_window(help_window, 1080, 840)

        page = self._create_help_page(help_window)
        builder(page)

    def _create_help_page(self, parent: ttk.Frame) -> ttk.Frame:
        """Create a vertically scrollable help page."""

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
        """Add a bold heading to a help page and return the next row index."""

        ttk.Label(parent, text=text, style="HelpHeading.TLabel").grid(
            row=row,
            column=0,
            sticky="w",
            pady=(0, 4),
        )
        return row + 1

    def _add_help_body(self, parent: ttk.Frame, row: int, text: str) -> int:
        """Add wrapped body text to a help page and return the next row index."""

        ttk.Label(
            parent,
            text=text,
            style="HelpBody.TLabel",
            justify="left",
            wraplength=1000,
        ).grid(row=row, column=0, sticky="w", pady=(0, 12))
        return row + 1

    def _add_help_figure(self, parent: ttk.Frame, row: int, figure: Figure) -> int:
        """Embed a rendered math figure into a help page."""

        canvas = FigureCanvasTkAgg(figure, master=parent)
        self._help_canvases.append(canvas)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.grid(row=row, column=0, sticky="ew", pady=(0, 14))
        return row + 1

    def _build_equations_figure(self) -> Figure:
        """Render the main Hodgkin-Huxley equations as paper-style math."""

        figure = Figure(figsize=(10.0, 5.0), dpi=160, facecolor="white")
        axis = figure.add_subplot(111)
        axis.axis("off")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)

        axis.text(0.02, 0.96, "Membrane balance", fontsize=12, fontweight="bold", va="top")
        axis.text(
            0.02,
            0.90,
            r"$C_m \frac{dV}{dt} = I_{\mathrm{inj}}(t) - I_{\mathrm{Na}}(t) - I_{\mathrm{K}}(t) - I_{\mathrm{L}}(t)$",
            fontsize=17,
            va="top",
        )

        axis.text(0.02, 0.74, "Ionic currents", fontsize=12, fontweight="bold", va="top")
        axis.text(
            0.02,
            0.68,
            r"$I_{\mathrm{Na}} = g_{\mathrm{Na,max}}(t)\,m^3 h\,(V - E_{\mathrm{Na}})$",
            fontsize=15,
            va="top",
        )
        axis.text(
            0.02,
            0.58,
            r"$I_{\mathrm{K}} = g_{\mathrm{K,max}}(t)\,n^4\,(V - E_{\mathrm{K}})$",
            fontsize=15,
            va="top",
        )
        axis.text(
            0.02,
            0.48,
            r"$I_{\mathrm{L}} = g_{\mathrm{L}}\,(V - E_{\mathrm{L}})$",
            fontsize=15,
            va="top",
        )

        axis.text(0.52, 0.96, "Gate dynamics", fontsize=12, fontweight="bold", va="top")
        axis.text(
            0.52,
            0.90,
            r"$\frac{dm}{dt} = \alpha_m(V)(1-m) - \beta_m(V)m$",
            fontsize=15,
            va="top",
        )
        axis.text(
            0.52,
            0.80,
            r"$\frac{dh}{dt} = \alpha_h(V)(1-h) - \beta_h(V)h$",
            fontsize=15,
            va="top",
        )
        axis.text(
            0.52,
            0.70,
            r"$\frac{dn}{dt} = \alpha_n(V)(1-n) - \beta_n(V)n$",
            fontsize=15,
            va="top",
        )
        axis.text(0.52, 0.56, "Voltage-dependent rates", fontsize=12, fontweight="bold", va="top")
        axis.text(0.52, 0.50, r"$\alpha_m(V) = 0.1\,\mathrm{vtrap}(V + 40, 10)$", fontsize=12, va="top")
        axis.text(0.52, 0.42, r"$\beta_m(V) = 4.0\,e^{-(V + 65)/18}$", fontsize=12, va="top")
        axis.text(0.52, 0.34, r"$\alpha_h(V) = 0.07\,e^{-(V + 65)/20}$", fontsize=12, va="top")
        axis.text(0.52, 0.26, r"$\beta_h(V) = \frac{1}{1 + e^{-(V + 35)/10}}$", fontsize=12, va="top")
        axis.text(0.52, 0.18, r"$\alpha_n(V) = 0.01\,\mathrm{vtrap}(V + 55, 10)$", fontsize=12, va="top")
        axis.text(0.52, 0.10, r"$\beta_n(V) = 0.125\,e^{-(V + 65)/80}$", fontsize=12, va="top")

        return figure

    def _build_graph_help_figure(self) -> Figure:
        """Render the key graph relationships in clean math notation."""

        figure = Figure(figsize=(10.0, 4.2), dpi=160, facecolor="white")
        axis = figure.add_subplot(111)
        axis.axis("off")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)

        axis.text(0.02, 0.94, "Injected current", fontsize=12, fontweight="bold", va="top")
        axis.text(
            0.02,
            0.86,
            r"$I_{\mathrm{inj}}(t) = I_{\mathrm{baseline}} + \sum_p I_p(t)$",
            fontsize=16,
            va="top",
        )
        axis.text(0.02, 0.72, "Sodium conductance", fontsize=12, fontweight="bold", va="top")
        axis.text(
            0.02,
            0.64,
            r"$g_{\mathrm{Na,actual}}(t) = g_{\mathrm{Na,max}}(t)\,m^3 h$",
            fontsize=15,
            va="top",
        )
        axis.text(0.02, 0.52, "Potassium conductance", fontsize=12, fontweight="bold", va="top")
        axis.text(
            0.02,
            0.44,
            r"$g_{\mathrm{K,actual}}(t) = g_{\mathrm{K,max}}(t)\,n^4$",
            fontsize=15,
            va="top",
        )

        axis.text(0.52, 0.94, "Metrics and thresholds", fontsize=12, fontweight="bold", va="top")
        axis.text(
            0.52,
            0.86,
            r"Spike detection counts upward crossings of $0\,\mathrm{mV}$ with a $2\,\mathrm{ms}$ refractory window.",
            fontsize=12,
            va="top",
        )
        axis.text(
            0.52,
            0.72,
            r"$\mathrm{Max}\,V$ and $\mathrm{Min}\,V$ are the extrema of the displayed trace.",
            fontsize=12,
            va="top",
        )
        axis.text(
            0.52,
            0.60,
            r"$V_{\mathrm{rest}}$ initializes the gates at steady state.",
            fontsize=12,
            va="top",
        )
        axis.text(
            0.52,
            0.50,
            r"$V_0$ is the actual starting membrane voltage at $t = 0$.",
            fontsize=12,
            va="top",
        )
        axis.text(
            0.52,
            0.38,
            r"The dashed conductance traces are $g_{\mathrm{Na,max}}(t)$ and $g_{\mathrm{K,max}}(t)$.",
            fontsize=12,
            va="top",
        )
        axis.text(
            0.52,
            0.28,
            r"The solid traces are the actual conductances after the gates are applied.",
            fontsize=12,
            va="top",
        )

        return figure

    def _build_equations_help(self, parent: ttk.Frame) -> None:
        """Populate the equations tab in the help window."""

        row = 0
        row = self._add_help_heading(parent, row, "Hodgkin-Huxley Equations")
        row = self._add_help_figure(parent, row, self._build_equations_figure())
        row = self._add_help_body(
            parent,
            row,
            "V is membrane voltage. m is sodium activation, h is sodium inactivation, and n is potassium activation. "
            "Each gate is a probability from 0 to 1. g_Na,max(t) and g_K,max(t) are the scheduled maximum conductances "
            "that you can change over time from the left panel. E_Na, E_K, and E_L are reversal potentials, and C_m is membrane capacitance.",
        )
        row = self._add_help_body(
            parent,
            row,
            "The equations work together like this: injected current pushes voltage up or down, the voltage changes the gate rates, "
            "and the gates then control the sodium and potassium currents that shape the action potential. The simulator uses fixed-step RK4 "
            f"with dt = {DEFAULT_INTEGRATION_DT_MS:.2f} ms.",
        )

    def _build_graph_help(self, parent: ttk.Frame) -> None:
        """Populate the graph-interpretation page in the help window."""

        row = 0
        row = self._add_help_heading(parent, row, "Parameters")
        row = self._add_help_body(
            parent,
            row,
            "Resting V sets the steady-state gate values used before the run starts. Initial V sets the actual membrane voltage at t = 0. Baseline current is always present, and any additional pulses are added on top of it. The conductance change controls adjust the scheduled maximum conductances over time. Trace Metrics count spikes at upward crossings of 0 mV with a 2 ms refractory window.",
        )

        row = self._add_help_heading(parent, row, "Voltage Graph")
        row = self._add_help_body(
            parent,
            row,
            "The voltage graph is the membrane voltage trace over time. A very positive Initial V will appear at the start of the trace, even if the model quickly relaxes toward the potassium reversal region. The Full button restores the full simulation window after zooming or panning.",
        )

        row = self._add_help_heading(parent, row, "Injected Current Graph")
        row = self._add_help_body(
            parent,
            row,
            "The injected-current graph shows baseline plus all current pulses. The main pulse and Additional pulses are additive, so overlapping pulses sum. This graph controls the input drive that pushes the membrane voltage up or down.",
        )

        row = self._add_help_heading(parent, row, "Sodium Conductance Graph")
        row = self._add_help_body(
            parent,
            row,
            "The dashed sodium trace is g_Na,max(t), the scheduled maximum conductance. The solid sodium trace is the actual conductance g_Na,max(t) m^3 h. If the dashed line is lowered, the ceiling moves down, but the gates still determine how much of that ceiling is used.",
        )

        row = self._add_help_heading(parent, row, "Potassium Conductance Graph")
        row = self._add_help_body(
            parent,
            row,
            "The dashed potassium trace is g_K,max(t), the scheduled maximum conductance. The solid potassium trace is the actual conductance g_K,max(t) n^4. This is the same idea as sodium: the schedule sets the ceiling, and the gate term determines the open-channel fraction.",
        )

    def export_csv(self) -> None:
        """Save the current simulation trace to a user-selected CSV file."""

        if self.result is None:
            messagebox.showwarning("No simulation", "Run a simulation before exporting.")
            return

        default_path = PROJECT_ROOT / "output" / "interactive_simulation.csv"
        path = filedialog.asksaveasfilename(
            title="Export simulation CSV",
            initialdir=default_path.parent,
            initialfile=default_path.name,
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if not path:
            return

        export_simulation_csv(self.result, path)
        self.status_var.set(f"CSV exported: {path}")

    def save_plot(self) -> None:
        """Save the current voltage/current figure to an image file."""

        if self.result is None:
            messagebox.showwarning("No simulation", "Run a simulation before saving a plot.")
            return

        default_path = PROJECT_ROOT / "output" / "interactive_simulation.png"
        path = filedialog.asksaveasfilename(
            title="Save simulation plot",
            initialdir=default_path.parent,
            initialfile=default_path.name,
            defaultextension=".png",
            filetypes=(("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")),
        )
        if not path:
            return

        self.figure.savefig(path, dpi=200, bbox_inches="tight")
        self.status_var.set(f"Plot saved: {path}")


def main() -> None:
    """Launch the interactive membrane dynamics simulator."""

    root = tk.Tk()
    HodgkinHuxleySimulatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
