"""Interactive desktop Hodgkin-Huxley membrane dynamics simulator.

This Tkinter application provides a usable interface for the research simulator.
It lets users change conductances, reversal potentials, integration settings,
and current-injection parameters, then rerun the single-compartment model and
export the resulting voltage trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from src.analysis.spike_metrics import SpikeMetrics, summarize_voltage_trace
from src.models.hodgkin_huxley import HodgkinHuxleyNeuron, HodgkinHuxleyParameters
from src.simulation.config import SimulationConfig
from src.simulation.protocols import StepCurrent
from src.simulation.runner import SimulationResult, simulate
from src.utils.export import export_simulation_csv


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ControlDefaults:
    """Default GUI values that reproduce the classic HH current-step response."""

    duration_ms: float = 50.0
    dt_ms: float = 0.01
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
    method: str = "rk4"


class HodgkinHuxleySimulatorApp:
    """Tkinter GUI for interactively running HH membrane simulations."""

    def __init__(self, root: tk.Tk) -> None:
        """Build the simulator window and run the initial default simulation."""

        self.root = root
        self.root.title("Hodgkin-Huxley Membrane Dynamics Simulator")
        self.root.geometry("1280x820")
        self.root.minsize(1040, 680)

        self.defaults = ControlDefaults()
        self.variables: dict[str, tk.StringVar] = {}
        self.result: SimulationResult | None = None
        self.metrics: SpikeMetrics | None = None

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

    def _build_layout(self) -> None:
        """Create the control panel, plot area, and status bar."""

        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        controls = ttk.Frame(self.root, padding=(12, 12, 8, 8))
        controls.grid(row=0, column=0, sticky="ns")

        plot_frame = ttk.Frame(self.root, padding=(4, 8, 12, 8))
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

    def _build_controls(self, parent: ttk.Frame) -> None:
        """Create grouped numeric controls for model, protocol, and output."""

        simulation = ttk.LabelFrame(parent, text="Simulation", style="Section.TLabelframe")
        simulation.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self._add_entry(simulation, "duration_ms", "Duration (ms)", 0)
        self._add_entry(simulation, "dt_ms", "Time step (ms)", 1)
        self._add_entry(simulation, "initial_voltage_mV", "Initial V (mV)", 2)
        self._add_method_control(simulation, 3)

        current = ttk.LabelFrame(parent, text="Injected Current", style="Section.TLabelframe")
        current.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self._add_entry(current, "current_amplitude", "Amplitude (uA/cm^2)", 0)
        self._add_entry(current, "current_start_ms", "Start (ms)", 1)
        self._add_entry(current, "current_end_ms", "End (ms)", 2)
        self._add_entry(current, "current_baseline", "Baseline (uA/cm^2)", 3)

        conductance = ttk.LabelFrame(parent, text="Conductances", style="Section.TLabelframe")
        conductance.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self._add_entry(conductance, "g_na", "g_Na (mS/cm^2)", 0)
        self._add_entry(conductance, "g_k", "g_K (mS/cm^2)", 1)
        self._add_entry(conductance, "g_l", "g_L (mS/cm^2)", 2)
        self._add_entry(conductance, "membrane_capacitance", "C_m (uF/cm^2)", 3)

        reversal = ttk.LabelFrame(parent, text="Reversal Potentials", style="Section.TLabelframe")
        reversal.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self._add_entry(reversal, "e_na", "E_Na (mV)", 0)
        self._add_entry(reversal, "e_k", "E_K (mV)", 1)
        self._add_entry(reversal, "e_l", "E_L (mV)", 2)

        buttons = ttk.Frame(parent)
        buttons.grid(row=4, column=0, sticky="ew", pady=(2, 10))
        buttons.columnconfigure((0, 1), weight=1)
        ttk.Button(buttons, text="Run Simulation", style="Run.TButton", command=self.run_simulation).grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6)
        )
        ttk.Button(buttons, text="Export CSV", command=self.export_csv).grid(
            row=1, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(buttons, text="Save Plot", command=self.save_plot).grid(
            row=1, column=1, sticky="ew", padx=(4, 0)
        )
        ttk.Button(buttons, text="Reset Defaults", command=self.reset_defaults).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0)
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

    def _add_method_control(self, parent: ttk.Frame, row: int) -> None:
        """Add the numerical integration method selector."""

        ttk.Label(parent, text="Method").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        variable = tk.StringVar(value=self.defaults.method)
        selector = ttk.Combobox(
            parent,
            textvariable=variable,
            values=("rk4", "euler"),
            state="readonly",
            width=11,
        )
        selector.grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        selector.bind("<<ComboboxSelected>>", lambda _event: self.run_simulation())
        self.variables["method"] = variable

    def _build_plot(self, parent: ttk.Frame) -> None:
        """Create the embedded Matplotlib figure used for all simulations."""

        self.figure = Figure(figsize=(9.5, 6.8), dpi=100)
        self.voltage_axis = self.figure.add_subplot(211)
        self.current_axis = self.figure.add_subplot(212, sharex=self.voltage_axis)

        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False).grid(
            row=0, column=0, sticky="w"
        )

    def reset_defaults(self) -> None:
        """Restore controls to the classic Hodgkin-Huxley default protocol."""

        for field_name, value in self.defaults.__dict__.items():
            self.variables[field_name].set(str(value))
        self.status_var.set("Default HH parameters loaded")

    def _read_float(self, name: str) -> float:
        """Parse a float control and raise a clear error for invalid input."""

        try:
            return float(self.variables[name].get())
        except ValueError as exc:
            raise ValueError(f"{name} must be a number.") from exc

    def _build_simulation_objects(
        self,
    ) -> tuple[HodgkinHuxleyNeuron, SimulationConfig, StepCurrent, float]:
        """Convert GUI control values into model, config, and protocol objects."""

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
            dt_ms=self._read_float("dt_ms"),
            method=self.variables["method"].get(),
        )
        protocol = StepCurrent(
            amplitude=self._read_float("current_amplitude"),
            start_ms=self._read_float("current_start_ms"),
            end_ms=self._read_float("current_end_ms"),
            baseline=self._read_float("current_baseline"),
        )
        initial_voltage = self._read_float("initial_voltage_mV")
        return HodgkinHuxleyNeuron(parameters), config, protocol, initial_voltage

    def run_simulation(self) -> None:
        """Run the HH model with current controls and redraw the membrane trace."""

        try:
            neuron, config, protocol, initial_voltage = self._build_simulation_objects()
            initial_state = None
            if initial_voltage != -65.0:
                from src.models.hodgkin_huxley import HodgkinHuxleyState

                initial_state = HodgkinHuxleyState.from_voltage(neuron, initial_voltage)

            self.result = simulate(
                neuron=neuron,
                config=config,
                current_protocol=protocol,
                initial_state=initial_state,
            )
            self.metrics = summarize_voltage_trace(self.result)
        except Exception as exc:  # noqa: BLE001 - GUI should report validation errors.
            messagebox.showerror("Simulation error", str(exc))
            self.status_var.set("Simulation failed")
            return

        self._draw_result()
        self._update_metrics()
        self.status_var.set("Simulation complete")

    def _draw_result(self) -> None:
        """Update voltage and injected-current plots from the latest result."""

        if self.result is None:
            return

        self.voltage_axis.clear()
        self.current_axis.clear()

        self.voltage_axis.plot(
            self.result.time_ms,
            self.result.voltage_mV,
            color="#1f5f99",
            linewidth=1.6,
        )
        self.voltage_axis.set_ylabel("V (mV)")
        self.voltage_axis.set_title("Membrane voltage response")
        self.voltage_axis.grid(True, alpha=0.25)

        self.current_axis.plot(
            self.result.time_ms,
            self.result.injected_current_uA_cm2,
            color="#7a3b12",
            linewidth=1.4,
        )
        self.current_axis.set_ylabel("I (uA/cm^2)")
        self.current_axis.set_xlabel("Time (ms)")
        self.current_axis.grid(True, alpha=0.25)

        self.figure.tight_layout()
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
            "peak V:      "
            f"{self.metrics.peak_voltage_mV:.2f} mV\n"
            "trough V:    "
            f"{self.metrics.trough_voltage_mV:.2f} mV\n"
            "first spike: "
            f"{first_spike}"
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
