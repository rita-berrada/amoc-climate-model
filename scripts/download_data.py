"""
Download the ARMOR3D ocean reanalysis dataset from CMEMS.

This script fetches the 80 MB ocean temperature and salinity dataset
(ocean_obs.nc) from the Copernicus Marine Service, which is required
for obs-driven model initialisation and validation plots.

Requirements
------------
    pip install copernicusmarine

Usage
-----
    python scripts/download_data.py

You will be prompted for your free CMEMS account credentials.
Register at: https://marine.copernicus.eu/

The AMOC observational index (AMOC_index.nc, 13 KB) is already
committed to data/ and does not need to be downloaded.
"""
from __future__ import annotations
import os
from pathlib import Path

# ── Output location ────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "ocean_obs.nc"

# ── CMEMS dataset configuration ────────────────────────────────────────────────
DATASET_ID = "cmems_obs-mob_glo_phy-cur_my_0.25deg_P1M-m"   # ARMOR3D REP monthly
VARIABLES  = ["thetao", "so"]          # potential temperature, salinity
LAT_RANGE  = (-10.0, 80.0)
LON_RANGE  = (-80.0, 20.0)
TIME_RANGE = ("1993-01-01", "2019-12-31")
DEPTH      = (0.0, 1.0)               # surface only


def main() -> None:
    try:
        import copernicusmarine
    except ImportError:
        raise SystemExit(
            "copernicusmarine is not installed.\n"
            "Run:  pip install copernicusmarine"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading ARMOR3D data to {OUTPUT_FILE} …")
    copernicusmarine.subset(
        dataset_id=DATASET_ID,
        variables=VARIABLES,
        minimum_latitude=LAT_RANGE[0],
        maximum_latitude=LAT_RANGE[1],
        minimum_longitude=LON_RANGE[0],
        maximum_longitude=LON_RANGE[1],
        start_datetime=TIME_RANGE[0],
        end_datetime=TIME_RANGE[1],
        minimum_depth=DEPTH[0],
        maximum_depth=DEPTH[1],
        output_filename=str(OUTPUT_FILE),
        force_download=True,
    )
    print(f"Done. File saved to {OUTPUT_FILE}")
    print(f"Size: {OUTPUT_FILE.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
