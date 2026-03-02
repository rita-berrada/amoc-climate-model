"""Tests for amoc.data_loader."""
import pytest
from pathlib import Path
from amoc.data_loader import _check_files

DATA_DIR = Path(__file__).parent.parent / "data"


def test_check_files_raises_on_missing(tmp_path):
    """FileNotFoundError must be raised when data files are absent."""
    with pytest.raises(FileNotFoundError, match="ocean_obs.nc"):
        _check_files(tmp_path)


def test_amoc_index_present():
    """AMOC_index.nc is committed to the repo and must always be present."""
    assert (DATA_DIR / "AMOC_index.nc").exists(), (
        "AMOC_index.nc is missing from data/ — it should be committed to the repo.\n"
        "This file is 13 KB and is the primary observational target."
    )


@pytest.mark.requires_data
def test_load_obs_returns_obs_state():
    """Full integration test: requires ocean_obs.nc (80 MB, not in CI)."""
    from amoc.data_loader import load_obs, ObsState
    obs = load_obs(DATA_DIR)
    assert isinstance(obs, ObsState)
    assert obs.T1_obs.dims == ("time",)
    assert obs.amoc_index.dims == ("time",)
