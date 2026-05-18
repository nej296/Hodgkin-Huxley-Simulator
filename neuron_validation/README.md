# NEURON Validation Workflow

This folder contains the source files needed to compare the Python
Hodgkin-Huxley simulator against NEURON using matched current-density input.

## Why `dclamp` Exists

The Python simulator injects current as membrane current density in `uA/cm2`.
NEURON's built-in `IClamp` injects point current in `nA`, which would require
an area-based conversion. For validation, this project uses a custom NMODL
density mechanism, `dclamp`, so both simulators receive the same current-density
protocol.

The mechanism accepts `amp` in `uA/cm2` and converts internally to NEURON's
distributed-current unit, `mA/cm2`:

```text
i = -0.001 * iinj
```

The minus sign follows NEURON's outward-positive membrane-current convention;
positive `iinj` is therefore depolarizing.

## Step 1: Compile The NEURON Mechanism

From the repository root:

```powershell
nrnivmodl neuron_validation
```

This creates local NEURON build artifacts such as `nrnmech.dll`. These files are
ignored by git and should not be committed.

## Step 2: Generate Matched Traces

Run the Python simulator sweep:

```powershell
python neuron_validation\run_python_validation_sweep.py
```

Run the NEURON simulator sweep:

```powershell
python neuron_validation\run_neuron_dclamp_sweep.py
```

Both scripts use the shared settings in `validation_config.py`:

- current levels: `0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 150, 200 uA/cm2`
- simulation duration: `100 ms`
- timestep: `0.01 ms`
- initial voltage: `-65 mV`
- current step: `5 ms` to `80 ms`
- spike threshold: `0 mV`
- refractory period: `2 ms`

Generated outputs are written under:

```text
output/validation/
```

## Next Quantitative Analysis

The next validation step is to compare the Python and NEURON traces
current-by-current using:

- voltage RMSE
- voltage MAE
- maximum absolute voltage error
- spike count difference
- firing-rate difference
- peak and trough voltage differences
- first-spike latency difference
- F-I curve error
