# RStudio Plotting Scripts

This folder stores the R code used to create validation and article figures.

## Scripts

- `validation_summary_plots.R`
  - reads the Python-vs-NEURON summary comparison files from
    `Desktop/statistical analysis`
  - creates peak-voltage difference, mean-absolute-difference, peak-voltage
    scatter, and first-spike-latency plots

- `neuron_fi_curve_0_to_60.R`
  - reads `neuron_dclamp_fi_summary.csv`
  - filters the NEURON F-I curve to `0-60 uA/cm^2`
  - avoids the depolarization-block range for a cleaner educational figure

## Required R Packages

Install these once in RStudio:

```r
install.packages("ggplot2")
install.packages("readr")
install.packages("dplyr")
install.packages("tidyr")
```

Then run the relevant script from RStudio.
