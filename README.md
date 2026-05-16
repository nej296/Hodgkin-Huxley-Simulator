# Hodgkin-Huxley Membrane Dynamics Simulator

An interactive Python simulator for exploring the classic Hodgkin-Huxley
single-compartment neuron model. The project is designed for educational use,
research portfolio presentation, and future expansion into more advanced
computational neuroscience workflows.

The simulator lets students and researchers change ionic conductances, reversal
potentials, current injection, and numerical integration settings, then observe
how those parameters shape action-potential generation and membrane voltage
dynamics.

## Educational Purpose

This project is intended to help users understand how conductance-based neuron
models produce spikes from biophysical mechanisms. It is suitable for:

- neuroscience and neuroengineering coursework
- computational biology demonstrations
- electrophysiology modeling labs
- portfolio projects focused on scientific Python
- early-stage research prototypes before moving to larger simulators

Key learning goals:

- connect membrane voltage dynamics to ionic currents
- visualize the roles of sodium and potassium reversal potentials
- study how `g_Na` and `g_K` affect spike shape, threshold, and firing rate
- compare numerical methods such as RK4 and Euler integration
- export simulation traces for reproducible analysis

## Interactive Simulator

Launch the desktop simulator:

```bash
python app.py
```

The application opens a control panel and two plots:

- membrane voltage over time
- injected current over time

The voltage plot includes black dashed reference lines for:

- `E_Na`, the sodium reversal potential
- `E_K`, the potassium reversal potential

These labels appear at the right side of the voltage plot. The leak reversal
potential, `E_L`, is adjustable in the controls but is not drawn on the main
plot, keeping the display focused on the main action-potential driving forces.

The app includes controls for:

- simulation duration and time step
- initial membrane voltage
- RK4 or Euler integration
- current-step amplitude, start time, end time, and baseline
- sodium, potassium, and leak conductances
- sodium, potassium, and leak reversal potentials
- membrane capacitance

The metrics panel reports:

- spike count
- firing rate
- peak voltage
- trough voltage
- first spike time

Use **Run Simulation** after changing parameters. Use **Export CSV** to save the
active trace and **Save Plot** to save the current figure.

## Biological Background

The Hodgkin-Huxley model describes action-potential generation as the interaction
between membrane capacitance and voltage-gated ionic conductances. Sodium
channels drive rapid depolarization, potassium channels drive repolarization and
after-hyperpolarization, and leak conductance provides a passive background
current.

Changing sodium and potassium conductance values changes:

- spike amplitude
- spike width
- threshold behavior
- firing frequency
- after-hyperpolarization
- recovery between spikes

## Model Equations

The membrane equation is:

```text
C_m dV/dt = I_injected - I_Na - I_K - I_L
```

The ionic currents are:

```text
I_Na = g_Na m^3 h (V - E_Na)
I_K  = g_K n^4 (V - E_K)
I_L  = g_L (V - E_L)
```

The gating variables follow first-order kinetics:

```text
dm/dt = alpha_m(V) (1 - m) - beta_m(V) m
dh/dt = alpha_h(V) (1 - h) - beta_h(V) h
dn/dt = alpha_n(V) (1 - n) - beta_n(V) n
```

Variables:

- `V`: membrane voltage in mV
- `m`: sodium activation gate
- `h`: sodium inactivation gate
- `n`: potassium activation gate
- `C_m`: membrane capacitance density in uF/cm^2
- `I_injected`: applied current density in uA/cm^2

Default parameters follow the classic squid giant axon model:

| Parameter | Default | Units | Interpretation |
| --- | ---: | --- | --- |
| `C_m` | 1.0 | uF/cm^2 | membrane capacitance density |
| `g_Na` | 120.0 | mS/cm^2 | maximum sodium conductance |
| `g_K` | 36.0 | mS/cm^2 | maximum potassium conductance |
| `g_L` | 0.3 | mS/cm^2 | leak conductance |
| `E_Na` | 50.0 | mV | sodium reversal potential |
| `E_K` | -77.0 | mV | potassium reversal potential |
| `E_L` | -54.387 | mV | leak reversal potential |

## Reversal Potentials

Reversal potentials define the voltage at which a particular ionic current would
have no net driving force.

- `E_Na` is usually positive, so sodium current tends to depolarize the membrane
  during spike initiation.
- `E_K` is usually negative, so potassium current tends to repolarize and
  hyperpolarize the membrane after a spike.
- `E_L` represents the passive leak reversal potential. It helps set the
  background tendency of the membrane but is not identical to the resting
  membrane voltage. Resting voltage emerges from the balance of sodium,
  potassium, leak, and any injected current.

## Numerical Methods

The simulator uses fixed-step explicit integration:

- `rk4`: fourth-order Runge-Kutta, the default method
- `euler`: forward Euler, useful for teaching numerical-method differences

Current protocols are evaluated on the simulation time grid and held constant
over each integration step. The gating-rate equations include numerically stable
handling of removable singularities so rate functions remain finite at the
classic Hodgkin-Huxley singular voltages.

## Installation

Clone the repository:

```bash
git clone https://github.com/nej296/Hodgkin-Huxley-Simulator.git
cd Hodgkin-Huxley-Simulator
```

Create and activate a virtual environment on Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Start the interactive simulator:

```bash
python app.py
```

Run the default command-line simulation:

```bash
python main.py
```

This writes:

- `output/simulation.csv`
- `output/simulation.png`

Customize a command-line simulation:

```bash
python main.py --current 12 --start-ms 5 --end-ms 40 --g-na 140 --g-k 30
```

Run a reproducible basic experiment:

```bash
python experiments/run_basic_simulation.py
```

Run sodium and potassium conductance sweeps:

```bash
python experiments/conductance_sweep.py
```

## Data Export

CSV exports include:

- time in ms
- membrane voltage in mV
- gating variables `m`, `h`, and `n`
- injected current in uA/cm^2

The interactive app does not show the gating variables on the main page, but
they remain available in exported data for analysis, validation, and plotting.

## Repository Structure

```text
.
|-- app.py               # interactive desktop simulator
|-- main.py              # command-line simulation entry point
|-- src/
|   |-- models/          # Hodgkin-Huxley model and parameters
|   |-- simulation/      # configuration, protocols, integration loop
|   |-- visualization/   # matplotlib plotting utilities
|   |-- analysis/        # spike metrics and validation comparisons
|   `-- utils/           # CSV export helpers
|-- experiments/         # reproducible simulation scripts
|-- docs/                # validation and scientific notes
|-- tests/               # automated tests
|-- data/                # input/reference data
|-- output/              # generated files, ignored by git
|-- notebooks/           # exploratory analysis notebooks
|-- requirements.txt
`-- pyproject.toml
```

## Testing

Run the test suite:

```bash
python -m pytest
```

The tests currently verify:

- stable HH rate functions at singular voltages
- action-potential generation under a standard depolarizing current step
- current-step protocol behavior

## Current Capabilities

- interactive educational desktop simulator
- standard Hodgkin-Huxley single-compartment equations
- sodium, potassium, and leak currents
- adjustable conductances and reversal potentials
- adjustable current-step protocol
- RK4 and Euler integration
- voltage and injected-current visualization
- Na/K reversal-potential overlays on the voltage graph
- CSV export for reproducible analysis
- spike-count and firing-rate summary metrics
- sodium and potassium conductance sweep experiments
- trace-comparison utilities for future validation against NEURON

## Limitations

This is currently a deterministic, single-compartment Hodgkin-Huxley simulator.
It does not yet include spatial morphology, multicompartment cable equations,
synaptic conductances, stochastic ion channels, or network simulations.

## Roadmap

Planned extensions:

- NEURON comparison experiments using matched protocols and reference traces
- multicompartment cable models
- SWC morphology loading
- morphology-dependent simulations
- synaptic conductances
- stochastic ion-channel variants
- structured parameter sweeps
- notebook-based analysis reports
- GPU-accelerated integration backends
- network simulation support

## References

Hodgkin, A. L., and Huxley, A. F. (1952). A quantitative description of
membrane current and its application to conduction and excitation in nerve.
Journal of Physiology, 117(4), 500-544.

Dayan, P., and Abbott, L. F. (2001). Theoretical Neuroscience: Computational
and Mathematical Modeling of Neural Systems. MIT Press.

Koch, C. (1999). Biophysics of Computation: Information Processing in Single
Neurons. Oxford University Press.
