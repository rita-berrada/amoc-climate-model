"""Physical and numerical parameters for the AMOC box model."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ModelConfig:
    """
    Container for all model parameters.

    Physical meaning
    ----------------
    Smoy  : reference (mean) salinity, psu
    mu    : flow sensitivity to density gradient, Sv / (°C or psu)
    delta : haline contraction / thermal expansion ratio (dimensionless)
    tau   : SST restoring timescale, years
    Vol   : box volume scaling (controls advection rate)
    """
    Smoy:  float = 35.0
    mu:    float = 1.5
    delta: float = 8.0
    tau:   float = 10.0
    Vol:   float = 300.0

    # Stochastic noise amplitudes per variable [T1, T2, S1]
    sigma: np.ndarray = field(
        default_factory=lambda: np.array([0.05, 0.05, 0.03])
    )

    # ODE solver settings
    t_start: float = 0.0
    t_end:   float = 500.0
    n_eval:  int   = 1000
    m_clip:  tuple = (-2.0, 2.0)

    # Euler-Maruyama settings
    em_T:  float = 10_000.0
    em_dt: float = 0.1
