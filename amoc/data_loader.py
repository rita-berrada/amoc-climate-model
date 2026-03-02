"""Load and preprocess CMEMS observational datasets."""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import xarray as xr

# Default location: repo_root/data/
_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"


class ObsState(NamedTuple):
    """Box-averaged observational fields."""
    T1_obs:     xr.DataArray   # mean temperature 60–80°N
    T2_obs:     xr.DataArray   # mean temperature −10 to +10°
    S1_obs:     xr.DataArray   # mean salinity   60–80°N
    S2_obs:     xr.DataArray   # mean salinity   −10 to +10°
    amoc_index: xr.DataArray   # observed AMOC transport [Sv]


def load_obs(data_dir: str | Path = _DEFAULT_DATA_DIR) -> ObsState:
    """
    Load ARMOR3D ocean reanalysis and CMEMS AMOC index.

    Parameters
    ----------
    data_dir : path to directory containing AMOC_index.nc and ocean_obs.nc

    Returns
    -------
    ObsState namedtuple

    Raises
    ------
    FileNotFoundError if any required file is missing.
    """
    data_dir = Path(data_dir)
    _check_files(data_dir)

    oce  = xr.open_mfdataset(data_dir / "ocean_obs.nc",  combine="by_coords")
    amoc = xr.open_mfdataset(data_dir / "AMOC_index.nc", combine="by_coords")

    return ObsState(
        T1_obs=oce.to.sel(latitude=slice(60, 80)).mean(("latitude", "longitude")),
        T2_obs=oce.to.sel(latitude=slice(-10, 10)).mean(("latitude", "longitude")),
        S1_obs=oce.so.sel(latitude=slice(60, 80)).mean(("latitude", "longitude")),
        S2_obs=oce.so.sel(latitude=slice(-10, 10)).mean(("latitude", "longitude")),
        amoc_index=amoc.amoc_mean * 1e-6,   # m³/s → Sv
    )


def _check_files(data_dir: Path) -> None:
    missing = [
        f for f in ("AMOC_index.nc", "ocean_obs.nc")
        if not (data_dir / f).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing data files in {data_dir}: {missing}\n"
            "Run:  python scripts/download_data.py"
        )
