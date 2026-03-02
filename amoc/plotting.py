"""
Visualisation utilities for the AMOC box model.

All public functions return a matplotlib Figure object so callers can
display or save them independently.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .integrators import Solution

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.figsize": (12, 8), "font.size": 14})


def plot_time_series(sol: Solution, title_suffix: str = "") -> plt.Figure:
    """
    Four-panel time series: T1, T2, S1, and AMOC transport m.

    Parameters
    ----------
    sol          : Solution from deterministic_solve or euler_maruyama
    title_suffix : appended to each subplot title (e.g. "F = -0.01")
    """
    fig, axes = plt.subplots(4, 1, figsize=(10, 20), sharex=True)
    pairs = [
        (sol.T1, "T1", "°C",  "tab:blue"),
        (sol.T2, "T2", "°C",  "tab:orange"),
        (sol.S1, "S1", "psu", "tab:green"),
        (sol.m,  "m",  "Sv",  "tab:purple"),
    ]
    for ax, (data, label, unit, color) in zip(axes, pairs):
        ax.plot(sol.t, data, color=color, label=label)
        ax.set_ylabel(f"{label} ({unit})")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.annotate(
            f"{data[-1]:.2f} {unit}",
            xy=(sol.t[-1], data[-1]),
            xytext=(-60, 10),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->"),
        )
    axes[-1].set_xlabel("Time (yr)")
    fig.suptitle(f"AMOC box model time series {title_suffix}", y=1.01)
    fig.tight_layout()
    return fig


def plot_global_sst(oce_ds) -> plt.Figure:
    """
    Robinson-projection map of time-mean sea surface temperature.

    Parameters
    ----------
    oce_ds : xarray.Dataset with variable 'to' (temperature)
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(14, 7))
    ax  = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.add_feature(cfeature.LAND, zorder=1)
    oce_ds.to.mean("time").plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="plasma",
        vmin=-2,
        vmax=30,
        cbar_kwargs={"label": "Temperature (°C)", "shrink": 0.6},
    )
    ax.set_title("Time-mean ocean temperature (ARMOR3D)")
    return fig


def plot_amoc_validation(
    t_model: np.ndarray,
    m_model: np.ndarray,
    amoc_obs,
) -> plt.Figure:
    """
    Overlay modelled and observed AMOC transport.

    Parameters
    ----------
    t_model  : time axis for model output
    m_model  : model AMOC transport [Sv]
    amoc_obs : xarray.DataArray with observed AMOC index [Sv]
    """
    fig, ax = plt.subplots()
    ax.plot(t_model, m_model, label="Box model", color="steelblue")
    amoc_obs.plot(ax=ax, label="CMEMS obs", color="coral", linestyle="--")
    ax.set_ylabel("AMOC transport (Sv)")
    ax.set_xlabel("Time")
    ax.set_title("Model vs. Observations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_bifurcation(F_values: np.ndarray, m_steady: np.ndarray) -> plt.Figure:
    """
    Bifurcation diagram: steady-state AMOC transport vs. freshwater forcing F.

    Parameters
    ----------
    F_values : array of freshwater forcing values
    m_steady : corresponding steady-state AMOC transport [Sv]
    """
    fig, ax = plt.subplots()
    ax.plot(F_values, m_steady, "o-", markersize=3, color="darkblue")
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8, label="AMOC collapse")
    ax.set_xlabel("Freshwater forcing F (psu yr⁻¹)")
    ax.set_ylabel("Steady-state AMOC (Sv)")
    ax.set_title("Bifurcation / tipping-point diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_phase_space_3d(
    T1: np.ndarray,
    T2: np.ndarray,
    S1: np.ndarray,
    title: str = "Phase space",
) -> plt.Figure:
    """
    3-D phase space trajectory coloured by time progression.

    Parameters
    ----------
    T1, T2, S1 : state variable arrays
    title      : figure title
    """
    n    = len(T1)
    step = max(n // 200, 1)
    cmap = cm.viridis

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(projection="3d")
    for i in range(0, n - step, step):
        ax.plot(
            T1[i:i + step + 1],
            T2[i:i + step + 1],
            S1[i:i + step + 1],
            color=cmap(i / n),
            alpha=0.5,
            linewidth=0.7,
        )
    ax.set_xlabel("T1 (°C)")
    ax.set_ylabel("T2 (°C)")
    ax.set_zlabel("S1 (psu)")
    ax.set_title(title)
    return fig


def plot_lorenz(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> plt.Figure:
    """3-D Lorenz attractor coloured by time (reference for chaotic dynamics)."""
    n    = len(x)
    step = 10
    cmap = cm.winter

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(projection="3d")
    for i in range(0, n - step, step):
        ax.plot(
            x[i:i + step + 1],
            y[i:i + step + 1],
            z[i:i + step + 1],
            color=cmap(i / n),
            alpha=0.4,
            linewidth=0.5,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Lorenz attractor (reference)")
    return fig
