TITLE Density current clamp for HH validation

NEURON {
    SUFFIX dclamp
    RANGE amp, delay, dur, i, iinj
    NONSPECIFIC_CURRENT i
}

UNITS {
    (uA) = (microamp)
    (mA) = (milliamp)
}

PARAMETER {
    amp = 0 (uA/cm2)
    delay = 5 (ms)
    dur = 75 (ms)
}

ASSIGNED {
    v (mV)
    i (mA/cm2)
    iinj (uA/cm2)
}

BREAKPOINT {
    if (t >= delay && t < delay + dur) {
        iinj = amp
    } else {
        iinj = 0
    }

    : NEURON membrane currents are positive outward. A positive
    : injected current should depolarize, so it is represented as
    : negative outward current after converting uA/cm2 to mA/cm2.
    i = -(0.001) * iinj
}
