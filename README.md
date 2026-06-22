# Building and Validating a Single-Compartment Hodgkin-Huxley Simulator in Python

An interactive Python desktop simulator for exploring Hodgkin-Huxley membrane
dynamics and validating the model against NEURON.

## Quick Start: Open the Desktop App

These steps run the same Python/Tkinter simulator used in this repository.

1. Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/).
2. Download this repository from GitHub or clone it with git.
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

## Abstract

The simulator implements voltage-dependent sodium, potassium, and leak currents
using the classical Hodgkin-Huxley conductance-based framework. Users can adjust
ionic conductances, reversal potentials, membrane capacitance, current-injection
settings, simulation duration, time step, and numerical integration method. The
main scientific question was whether a simple Python implementation could
reproduce the firing behavior of an equivalent single-compartment
Hodgkin-Huxley model in NEURON.

Validation was performed across injected current amplitudes from 0 to
200 uA/cm2. The Python simulator and NEURON reference model matched exactly in
spike count, firing rate, threshold current, and the overall frequency-current
relationship. Both models began firing at 3 uA/cm2, reached a maximum firing
rate of 133.33 Hz at 60 uA/cm2, and showed reduced repetitive firing at higher
current amplitudes consistent with depolarization block. Voltage-level
differences were small: the mean absolute peak-voltage difference was
approximately 0.171 mV, the mean absolute trough-voltage difference was
approximately 0.044 mV, and the mean absolute first-spike latency difference was
approximately 0.003 ms.

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
I_Na = g_Na m^3 h (V - E_Na)
I_K  = g_K  n^4   (V - E_K)
I_L  = g_L        (V - E_L)
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

### Initial Gate Values

When an initial gate value is not manually supplied, gates are initialized at
their steady-state values for the starting voltage:

```text
m_inf(V) = alpha_m(V) / (alpha_m(V) + beta_m(V))
h_inf(V) = alpha_h(V) / (alpha_h(V) + beta_h(V))
n_inf(V) = alpha_n(V) / (alpha_n(V) + beta_n(V))
```

The default initial voltage is:

```text
V_0 = -65 mV
m_0 = m_inf(V_0)
h_0 = h_inf(V_0)
n_0 = n_inf(V_0)
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

### Time Grid

The simulation uses a fixed time step:

```text
t_k = k dt
```

from `t = 0` through the configured simulation duration.

### Euler Integration

For state vector `y_k = [V_k, m_k, h_k, n_k]^T`, derivative function
`f(y_k, I_k)`, and fixed step `dt`:

```text
y_{k+1} = y_k + dt * f(y_k, I_k)
```

### Fourth-Order Runge-Kutta Integration

RK4 is the default method. The injected current is held constant over each
integration step:

```text
k1 = f(y_k, I_k)
k2 = f(y_k + 0.5 dt k1, I_k)
k3 = f(y_k + 0.5 dt k2, I_k)
k4 = f(y_k + dt k3, I_k)

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
peak_voltage_mV = max(V)
trough_voltage_mV = min(V)
first_spike_time_ms = time of the first accepted threshold crossing
```
