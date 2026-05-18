"""Shared settings for Python-vs-NEURON Hodgkin-Huxley validation."""

from __future__ import annotations


CURRENT_LEVELS_UA_CM2: tuple[float, ...] = (
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    7.0,
    10.0,
    15.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    70.0,
    80.0,
    90.0,
    100.0,
    120.0,
    140.0,
    150.0,
    200.0,
)

EXAMPLE_CURRENTS_UA_CM2: tuple[float, ...] = (
    0.0,
    3.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
)

TSTOP_MS: float = 100.0
DT_MS: float = 0.01
INITIAL_VOLTAGE_MV: float = -65.0
CURRENT_START_MS: float = 5.0
CURRENT_END_MS: float = 80.0
CURRENT_DURATION_MS: float = CURRENT_END_MS - CURRENT_START_MS
SPIKE_THRESHOLD_MV: float = 0.0
REFRACTORY_MS: float = 2.0
