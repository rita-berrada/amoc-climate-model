"""
Three-box thermohaline circulation model (Stommel-type).

State vector:  y = [T1, T2, S1]
Diagnostic:    S2 = 2·Smoy − S1       (global salt conservation)
               m  = μ·[(T2−T1) − δ·(S2−S1)]   [Sv]

Equations
---------
dT1/dt = −(T1 − T1r)/τ  +  |m|/V · (T2 − T1)
dT2/dt = −(T2 − T2r)/τ  +  |m|/V · (T1 − T2)
dS1/dt =                    |m|/V · (S2 − S1)  +  F
"""
from __future__ import annotations
import numpy as np
from .config import ModelConfig


class BoxModel:
    """
    Stommel-type 3-box AMOC model with temperature and salinity prognostics.

    Parameters
    ----------
    cfg : ModelConfig
        Physical and numerical parameters.
    T1r : float
        Restoring temperature for box 1 (high latitudes, 60–80°N), °C.
    T2r : float
        Restoring temperature for box 2 (equatorial, −10 to +10°), °C.
    """

    def __init__(self, cfg: ModelConfig, T1r: float, T2r: float) -> None:
        self.cfg = cfg
        self.T1r = float(T1r)
        self.T2r = float(T2r)

    def tendency(self, y: np.ndarray, F: float) -> np.ndarray:
        """
        Compute dY/dt for state vector y = [T1, T2, S1].

        Parameters
        ----------
        y : array-like, shape (3,)
            State vector [T1, T2, S1].
        F : float
            Freshwater forcing (psu yr⁻¹). Negative = freshening of box 1.

        Returns
        -------
        dydt : np.ndarray, shape (3,)
        """
        cfg = self.cfg
        T1, T2, S1 = y
        S2 = 2.0 * cfg.Smoy - S1
        m  = cfg.mu * ((T2 - T1) - cfg.delta * (S2 - S1))
        m  = np.clip(m, *cfg.m_clip)

        dT1dt = -(T1 - self.T1r) / cfg.tau + np.abs(m) / cfg.Vol * (T2 - T1)
        dT2dt = -(T2 - self.T2r) / cfg.tau + np.abs(m) / cfg.Vol * (T1 - T2)
        dS1dt = np.abs(m) / cfg.Vol * (S2 - S1) + F

        return np.array([dT1dt, dT2dt, dS1dt])

    def amoc_flow(self, y: np.ndarray) -> float:
        """Return AMOC transport m [Sv] for a given state vector."""
        cfg = self.cfg
        T1, T2, S1 = y
        S2 = 2.0 * cfg.Smoy - S1
        m  = cfg.mu * ((T2 - T1) - cfg.delta * (S2 - S1))
        return float(np.clip(m, *cfg.m_clip))

    def scipy_rhs(self, t: float, y: np.ndarray, F: float) -> np.ndarray:
        """Wrapper matching the scipy.integrate.solve_ivp signature (t, y, *args)."""
        return self.tendency(y, F)
