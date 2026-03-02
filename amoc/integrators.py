"""
Numerical integration routines for the AMOC box model.

Provides
--------
deterministic_solve  : scipy LSODA wrapper (stiff ODE solver)
euler_maruyama       : explicit stochastic Euler–Maruyama scheme
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .model import BoxModel


@dataclass
class Solution:
    """Container for integration output."""
    t:  np.ndarray   # time axis
    T1: np.ndarray   # temperature box 1 (high latitudes), °C
    T2: np.ndarray   # temperature box 2 (equatorial), °C
    S1: np.ndarray   # salinity box 1, psu
    m:  np.ndarray   # AMOC transport, Sv

    @property
    def S2(self) -> np.ndarray:
        """Salinity of box 2, derived from global salt conservation."""
        smoy = 35.0  # default; exact value stored in cfg but sufficient here
        return 2.0 * smoy - self.S1


def deterministic_solve(
    model: BoxModel,
    F: float,
    y0: np.ndarray | None = None,
) -> Solution:
    """
    Integrate the ODE system with scipy LSODA (handles stiff regimes).

    Parameters
    ----------
    model : BoxModel
    F     : freshwater forcing (psu yr⁻¹)
    y0    : initial conditions [T1, T2, S1]; defaults to [15, 20, 35]

    Returns
    -------
    Solution dataclass
    """
    cfg = model.cfg
    if y0 is None:
        y0 = np.array([15.0, 20.0, 35.0])

    t_eval = np.linspace(cfg.t_start, cfg.t_end, cfg.n_eval)
    sol = solve_ivp(
        model.scipy_rhs,
        [cfg.t_start, cfg.t_end],
        y0,
        method="LSODA",
        t_eval=t_eval,
        args=(F,),
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    S2 = 2.0 * cfg.Smoy - sol.y[2]
    m  = cfg.mu * ((sol.y[1] - sol.y[0]) - cfg.delta * (S2 - sol.y[2]))

    return Solution(t=sol.t, T1=sol.y[0], T2=sol.y[1], S1=sol.y[2], m=m)


def euler_maruyama(
    model: BoxModel,
    F: float,
    y0: np.ndarray,
    sigma: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> Solution:
    """
    Euler–Maruyama stochastic integration.

    Scheme:  yₙ₊₁ = yₙ + f(yₙ, F)·Δt + σ·√Δt·ξₙ,  ξₙ ~ N(0, I)

    Parameters
    ----------
    model : BoxModel
    F     : freshwater forcing (psu yr⁻¹)
    y0    : initial conditions [T1, T2, S1]
    sigma : noise amplitudes per variable; defaults to cfg.sigma
    rng   : numpy Generator for reproducibility (e.g. np.random.default_rng(42))

    Returns
    -------
    Solution dataclass
    """
    cfg   = model.cfg
    sigma = sigma if sigma is not None else cfg.sigma
    rng   = rng or np.random.default_rng()
    dt    = cfg.em_dt
    n_steps = int(cfg.em_T / dt)

    trajectory = np.empty((n_steps + 1, 3))
    trajectory[0] = y0
    y = y0.copy().astype(float)

    for i in range(n_steps):
        noise = sigma * np.sqrt(dt) * rng.standard_normal(size=3)
        y = y + model.tendency(y, F) * dt + noise
        trajectory[i + 1] = y

    S2 = 2.0 * cfg.Smoy - trajectory[:, 2]
    m  = cfg.mu * (
        (trajectory[:, 1] - trajectory[:, 0])
        - cfg.delta * (S2 - trajectory[:, 2])
    )

    t = np.arange(n_steps + 1) * dt
    return Solution(
        t=t, T1=trajectory[:, 0], T2=trajectory[:, 1],
        S1=trajectory[:, 2], m=m,
    )
