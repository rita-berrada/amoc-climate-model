"""
Scientific analysis utilities.

Provides
--------
bifurcation_sweep   : scan freshwater forcing values and record steady-state m
lorenz_attractor    : generate Lorenz reference trajectory for phase-space comparison
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .model import BoxModel
from .integrators import deterministic_solve


def bifurcation_sweep(
    model: BoxModel,
    F_values: np.ndarray,
    y0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quasi-static bifurcation scan over freshwater forcing values.

    Initial conditions are chained between runs so that the trajectory
    follows the stable branch as F is varied (hysteresis tracking).

    Parameters
    ----------
    model    : BoxModel
    F_values : 1-D array of forcing values to sweep
    y0       : starting initial conditions; defaults to [15, 20, 35]

    Returns
    -------
    F_values, m_steady : arrays of shape (len(F_values),)
    """
    m_steady = np.zeros_like(F_values)
    current_y0 = y0 if y0 is not None else np.array([15.0, 20.0, 35.0])

    for i, F in enumerate(F_values):
        sol = deterministic_solve(model, float(F), y0=current_y0)
        m_steady[i] = sol.m[-1]
        # Chain initial conditions for quasi-static branch continuity
        current_y0 = np.array([sol.T1[-1], sol.T2[-1], sol.S1[-1]])

    return F_values, m_steady


def lorenz_attractor(
    sigma: float = 10.0,
    beta: float = 8.0 / 3.0,
    rho: float = 28.0,
    tmax: float = 100.0,
    n: int = 10_000,
    ic: tuple = (0.0, 1.0, 1.05),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate the Lorenz system and return (t, x, y, z).

    Useful for qualitative comparison with the stochastic AMOC
    trajectory in 3-D phase space.

    Parameters
    ----------
    sigma, beta, rho : Lorenz parameters (standard: 10, 8/3, 28)
    tmax             : integration end time
    n                : number of output time points
    ic               : initial conditions (x0, y0, z0)

    Returns
    -------
    t, x, y, z : np.ndarrays of shape (n,)
    """
    def _lorenz(t, state):
        u, v, w = state
        return [
            sigma * (v - u),
            rho * u - v - u * w,
            -beta * w + u * v,
        ]

    soln = solve_ivp(_lorenz, (0.0, tmax), list(ic), dense_output=True)
    t = np.linspace(0.0, tmax, n)
    x, y, z = soln.sol(t)
    return t, x, y, z
