library(ggplot2)
library(readr)

neuron_summary <- read_csv(
  "C:/Users/Nicholas Johnson/Desktop/HH Simulator Article files and figures/CSV Files/NEURON CSV Files/neuron_dclamp_fi_summary.csv"
)

neuron_fi_plot <- ggplot(
  neuron_summary,
  aes(x = current_uA_cm2, y = firing_rate_hz)
) +
  geom_line(color = "#1f5f99", linewidth = 1.1) +
  geom_point(color = "#1f5f99", size = 2.4) +
  scale_x_continuous(
    breaks = c(0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200),
    limits = c(0, 200)
  ) +
  scale_y_continuous(
    breaks = seq(0, 140, 20),
    limits = c(0, 140)
  ) +
  labs(
    title = "NEURON Reference Model F-I Curve",
    x = expression("Injected Current ("*mu*"A/cm"^2*")"),
    y = "Firing Rate (Hz)"
  ) +
  theme_classic(base_size = 13) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title = element_text(face = "bold"),
    axis.text = element_text(color = "black")
  )

print(neuron_fi_plot)

ggsave(
  filename = "C:/Users/Nicholas Johnson/Desktop/HH Simulator Article files and figures/All Figures/Figures in Article/NEURON_FI_Curve_Full_Range.png",
  plot = neuron_fi_plot,
  width = 7,
  height = 4.5,
  dpi = 300
)
