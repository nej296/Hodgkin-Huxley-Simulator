# Hodgkin-Huxley Neuron Simulator

An interactive Python desktop app for exploring single-compartment
Hodgkin-Huxley membrane dynamics, current injection, spike generation, and
basic action-potential metrics.

## Download the Latest Version

Use the current `main` branch if you want the newest simulator code without
cloning the repository:

- [Download the latest ZIP](https://github.com/nej296/Hodgkin-Huxley-Simulator/archive/refs/heads/main.zip)
- [Open the repository on GitHub](https://github.com/nej296/Hodgkin-Huxley-Simulator)

The ZIP link always points to the latest commit on `main`.

## Quick Start: Open the Desktop App

These steps run the same Python/Tkinter simulator used in this repository.

### Windows

1. Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/).
2. Download this repository from GitHub, use the ZIP link above, or clone it with git.
3. Open PowerShell in the `Hodgkin-Huxley-Simulator` folder.
4. Run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

After the first setup, reopen the app from the same folder with:

```powershell
.\.venv\Scripts\activate
python app.py
```

### macOS and Linux

1. Install Python 3.10 or newer.
2. Download this repository from GitHub or clone it with git.
3. Open a terminal in the `Hodgkin-Huxley-Simulator` folder.
4. Run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

After the first setup, reopen the app from the same folder with:

```bash
source .venv/bin/activate
python app.py
```

## What the Simulator Does

The app runs a classical Hodgkin-Huxley model for one isopotential membrane
compartment. It lets users change the biophysical parameters and injected
current protocol, then immediately rerun the model and inspect the resulting
voltage trace.

The simulator is designed for learning and exploration. It makes it easy to see
how sodium conductance, potassium conductance, leak conductance, reversal
potentials, membrane capacitance, current amplitude, pulse timing, and scheduled
conductance changes affect action-potential behavior.

## App Controls

The desktop interface includes controls for:

- Simulation duration, resting voltage, initial voltage, and step controls
- Step-current amplitude, start time, end time, baseline current, and additional pulses
- Sodium and potassium maximum conductance settings, plus scheduled conductance changes with start and end times
- Leak conductance density
- Sodium, potassium, and leak reversal potentials
- Membrane capacitance

The app uses fixed-step fourth-order Runge-Kutta (`rk4`) integration. Euler is
not exposed in the app.

## Displayed Outputs

The standard four-graph view displays:

- Membrane voltage over time
- Injected current over time
- Sodium and potassium conductances on one graph, including scheduled maximum conductances
- Net ionic current over time, computed from sodium, potassium, and leak components
- Sodium and potassium reversal-potential reference lines

The `Show Gating Variables` button switches to a two-graph view with voltage on
top and gating variables `m`, `h`, and `n` below it.

Trace metrics display:

- Spike count
- Firing rate
- Max voltage
- Min voltage
- First spike time

The `Trace` button lets users click a plotted function and follow its x/y
coordinates. The current simulation can also be exported as a CSV file. The full
plot layout can be saved as an image or PDF, or each visible graph can be saved
individually.

## Repository Layout

```text
app.py                         Desktop simulator interface
src/models/hodgkin_huxley.py   Hodgkin-Huxley equations and state derivatives
src/simulation/                Time stepping, protocols, and simulation config
src/analysis/                  Spike detection and trace summary metrics
src/utils/                     CSV export helpers
tests/                         Unit tests for model and simulation behavior
experiments/                   Optional scripted simulation examples
neuron_validation/             Optional NEURON comparison utilities
rstudio/                       Optional plotting scripts for validation outputs
```

The NEURON validation utilities are included as supporting material, but the
main purpose of this repository is the interactive Hodgkin-Huxley simulator.

## Equations Used in the Simulator

The simulator represents the neuron as one isopotential compartment with state
vector:

```text
y = [V, m, h, n]^T
```

where `V` is membrane voltage in mV, `m` is sodium activation, `h` is sodium
inactivation, and `n` is potassium activation. Time is in ms, current density is
in `uA/cm2`, conductance density is in `mS/cm2`, and capacitance density is in
`uF/cm2`.

### Default Parameters

```text
C_m  = 1.0      uF/cm2
g_Na = 120.0    mS/cm2
g_K  = 36.0     mS/cm2
g_L  = 0.3      mS/cm2
E_Na = 50.0     mV
E_K  = -77.0    mV
E_L  = -54.387  mV
```

### Membrane Voltage

The membrane equation is:

```text
C_m dV/dt = I_inj(t) - I_Na - I_K - I_L
```

Equivalently:

```text
dV/dt = (I_inj(t) - I_Na - I_K - I_L) / C_m
```

Positive injected current depolarizes the membrane. Positive ionic current is
outward and hyperpolarizing.

### Ionic Currents

```text
g_Na,actual(t) = g_Na,max(t) m^3 h
g_K,actual(t)  = g_K,max(t)  n^4

I_Na = g_Na,actual(t) (V - E_Na)
I_K  = g_K,actual(t)  (V - E_K)
I_L  = g_L            (V - E_L)

I_net = I_Na + I_K + I_L
```

### Gating Variable Dynamics

For each gating variable, the simulator uses first-order voltage-dependent
kinetics:

```text
dm/dt = alpha_m(V) (1 - m) - beta_m(V) m
dh/dt = alpha_h(V) (1 - h) - beta_h(V) h
dn/dt = alpha_n(V) (1 - n) - beta_n(V) n
```

The rate functions are:

```text
alpha_m(V) = 0.1 * vtrap(V + 40, 10)
beta_m(V)  = 4.0 * exp(-(V + 65) / 18)

alpha_h(V) = 0.07 * exp(-(V + 65) / 20)
beta_h(V)  = 1 / (1 + exp(-(V + 35) / 10))

alpha_n(V) = 0.01 * vtrap(V + 55, 10)
beta_n(V)  = 0.125 * exp(-(V + 65) / 80)
```

The helper `vtrap(x, y)` is used to evaluate the removable singularities in
`alpha_m` and `alpha_n`:

```text
vtrap(x, y) = x / (1 - exp(-x / y))
```

Near `x = 0`, the simulator uses the Taylor approximation:

```text
vtrap(x, y) ~= y * (1 + x / (2y))
```

### Initial Voltage and Resting Voltage

The app has both `Resting V` and `Initial V`.

`Resting V` initializes the gate variables at steady state:

```text
m_inf(V) = alpha_m(V) / (alpha_m(V) + beta_m(V))
h_inf(V) = alpha_h(V) / (alpha_h(V) + beta_h(V))
n_inf(V) = alpha_n(V) / (alpha_n(V) + beta_n(V))
```

`Initial V` sets the actual membrane voltage at `t = 0`. This lets users start
the membrane voltage away from the voltage used to initialize the gates.

The defaults are:

```text
V_rest = -65 mV
V_0    = -65 mV
m_0    = m_inf(V_rest)
h_0    = h_inf(V_rest)
n_0    = n_inf(V_rest)
```

### Injected Current Protocols

For constant current:

```text
I_inj(t) = A
```

For the rectangular step-current protocol:

```text
I_inj(t) = baseline + amplitude,  start_ms <= t <= end_ms
I_inj(t) = baseline,              otherwise
```

For multiple current pulses, the app adds every active pulse to the baseline:

```text
I_inj(t) = baseline + sum(active pulse amplitudes)
```

Pulses can overlap. If they overlap, their amplitudes add.

### Scheduled Maximum Conductances

Sodium and potassium maximum conductances can change during specified time
windows:

```text
g_Na,max(t) = scheduled sodium maximum conductance
g_K,max(t)  = scheduled potassium maximum conductance
```

Outside a scheduled interval, the conductance returns to its base maximum value.
The displayed solid conductance traces are:

```text
g_Na,actual(t) = g_Na,max(t) m^3 h
g_K,actual(t)  = g_K,max(t)  n^4
```

### Time Grid

The simulation uses a fixed time step:

```text
t_k = k dt
```

from `t = 0` through the configured simulation duration.

### Fourth-Order Runge-Kutta Integration

RK4 is the only integration method used by the app. The injected current and
scheduled maximum conductances are held constant over each integration step:

```text
k1 = f(y_k, I_k, g_Na,max,k, g_K,max,k)
k2 = f(y_k + 0.5 dt k1, I_k, g_Na,max,k, g_K,max,k)
k3 = f(y_k + 0.5 dt k2, I_k, g_Na,max,k, g_K,max,k)
k4 = f(y_k + dt k3, I_k, g_Na,max,k, g_K,max,k)

y_{k+1} = y_k + (dt / 6) * (k1 + 2k2 + 2k3 + k4)
```

### Gate Probability Clipping

After each integration step, the gating variables are clipped to the physical
probability range:

```text
m <- min(max(m, 0), 1)
h <- min(max(h, 0), 1)
n <- min(max(n, 0), 1)
```

### Spike Detection and Displayed Metrics

The simulator detects spikes from upward voltage threshold crossings. With
default threshold `0 mV` and refractory window `2 ms`, a spike is counted when:

```text
V_k < threshold
V_{k+1} >= threshold
t_{k+1} - last_spike_time >= refractory_ms
```

The displayed summary metrics are:

```text
spike_count = number of accepted threshold crossings
firing_rate_hz = spike_count / ((t_end - t_start) / 1000)
max_voltage_mV = max(V)
min_voltage_mV = min(V)
first_spike_time_ms = time of the first accepted threshold crossing
```
