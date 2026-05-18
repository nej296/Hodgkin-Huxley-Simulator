# RStudio plots for Python HH vs NEURON summary-level validation.
#
# Expected input files:
#   Desktop/statistical analysis/per_current_summary_differences.csv
#   Desktop/statistical analysis/aggregate_summary_statistics.csv
#
# Required packages:
#   install.packages("ggplot2")
#   install.packages("readr")
#   install.packages("dplyr")
#   install.packages("tidyr")

library(ggplot2)
library(readr)
library(dplyr)
library(tidyr)

data_dir <- "C:/Users/Nicholas Johnson/Desktop/statistical analysis"

per_current <- read_csv(file.path(data_dir, "per_current_summary_differences.csv"))
aggregate <- read_csv(file.path(data_dir, "aggregate_summary_statistics.csv"))

theme_set(theme_minimal(base_size = 13))

# Peak voltage difference across current levels.
peak_difference_plot <- ggplot(
  per_current,
  aes(
    x = current_uA_cm2,
    y = peak_voltage_mV_difference_python_minus_neuron
  )
) +
  geom_hline(yintercept = 0, linewidth = 0.6, color = "black") +
  geom_line(linewidth = 0.9, color = "#1f5f99") +
  geom_point(size = 2.5, color = "#1f5f99") +
  labs(
    title = "Peak Voltage Difference Between Python HH and NEURON",
    x = "Injected current density (uA/cm^2)",
    y = "Peak voltage difference (mV)"
  ) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

peak_difference_plot

ggsave(
  file.path(data_dir, "peak_voltage_difference_plot.png"),
  peak_difference_plot,
  width = 8,
  height = 5,
  dpi = 300
)

# Mean absolute difference by summary metric.
aggregate_long <- aggregate %>%
  filter(metric %in% c(
    "spike_count",
    "firing_rate_hz",
    "peak_voltage_mV",
    "trough_voltage_mV",
    "first_spike_latency_ms"
  )) %>%
  mutate(
    metric_label = recode(
      metric,
      "spike_count" = "Spike count",
      "firing_rate_hz" = "Firing rate",
      "peak_voltage_mV" = "Peak voltage",
      "trough_voltage_mV" = "Trough voltage",
      "first_spike_latency_ms" = "First spike latency"
    )
  )

metric_error_plot <- ggplot(
  aggregate_long,
  aes(
    x = reorder(metric_label, mean_absolute_difference),
    y = mean_absolute_difference
  )
) +
  geom_col(fill = "#1f5f99", width = 0.7) +
  coord_flip() +
  labs(
    title = "Mean Absolute Difference by Summary Metric",
    x = NULL,
    y = "Mean absolute difference"
  ) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

metric_error_plot

ggsave(
  file.path(data_dir, "mean_absolute_difference_by_metric.png"),
  metric_error_plot,
  width = 8,
  height = 5,
  dpi = 300
)

# Python vs NEURON peak voltage scatter plot.
peak_scatter <- ggplot(
  per_current,
  aes(
    x = neuron_peak_voltage_mV,
    y = python_peak_voltage_mV
  )
) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed",
    color = "black",
    linewidth = 0.7
  ) +
  geom_point(aes(color = current_uA_cm2), size = 3) +
  scale_color_gradient(low = "#1f5f99", high = "#b8322a") +
  labs(
    title = "Peak Voltage Agreement Between Python HH and NEURON",
    x = "NEURON peak voltage (mV)",
    y = "Python peak voltage (mV)",
    color = "Current\n(uA/cm^2)"
  ) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

peak_scatter

ggsave(
  file.path(data_dir, "python_vs_neuron_peak_voltage_scatter.png"),
  peak_scatter,
  width = 7,
  height = 6,
  dpi = 300
)

# First spike latency difference across current levels.
latency_plot <- per_current %>%
  filter(!is.na(first_spike_latency_ms_difference_python_minus_neuron)) %>%
  ggplot(
    aes(
      x = current_uA_cm2,
      y = first_spike_latency_ms_difference_python_minus_neuron
    )
  ) +
  geom_hline(yintercept = 0, linewidth = 0.6, color = "black") +
  geom_line(linewidth = 0.9, color = "#7a3b12") +
  geom_point(size = 2.5, color = "#7a3b12") +
  labs(
    title = "First Spike Latency Difference",
    x = "Injected current density (uA/cm^2)",
    y = "Latency difference (ms)"
  ) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

latency_plot

ggsave(
  file.path(data_dir, "first_spike_latency_difference.png"),
  latency_plot,
  width = 8,
  height = 5,
  dpi = 300
)
