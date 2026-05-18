# RStudio plot for the NEURON F-I curve before depolarization block.
#
# Expected input file:
#   Desktop/HH Simulator Article files and figures/CSV Files/NEURON CSV Files/
#     neuron_dclamp_fi_summary.csv
#
# Required packages:
#   install.packages("ggplot2")
#   install.packages("readr")
#   install.packages("dplyr")

library(ggplot2)
library(readr)
library(dplyr)

data_dir <- "C:/Users/Nicholas Johnson/Desktop/HH Simulator Article files and figures/CSV Files/NEURON CSV Files"

neuron_summary <- read_csv(file.path(data_dir, "neuron_dclamp_fi_summary.csv"))

neuron_fi_0_60 <- neuron_summary %>%
  filter(current_uA_cm2 <= 60)

neuron_fi_plot <- ggplot(
  neuron_fi_0_60,
  aes(
    x = current_uA_cm2,
    y = firing_rate_hz
  )
) +
  geom_line(linewidth = 1.1, color = "#1f5f99") +
  geom_point(size = 3, color = "#1f5f99") +
  scale_x_continuous(
    limits = c(0, 60),
    breaks = c(0, 10, 20, 30, 40, 50, 60)
  ) +
  scale_y_continuous(
    limits = c(0, 140),
    breaks = c(0, 20, 40, 60, 80, 100, 120, 140)
  ) +
  labs(
    title = "NEURON Hodgkin-Huxley F-I Curve",
    x = "Injected current density (uA/cm^2)",
    y = "Firing rate (Hz)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    panel.grid.minor = element_blank()
  )

neuron_fi_plot

ggsave(
  file.path(data_dir, "neuron_fi_curve_0_to_60_RStudio.png"),
  neuron_fi_plot,
  width = 8,
  height = 5,
  dpi = 300
)
