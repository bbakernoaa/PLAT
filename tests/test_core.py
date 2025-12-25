
"""Unit tests for the core trajectory model."""

import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_allclose

from plat.core import run_trajectory


@pytest.fixture
def sample_velocity_field() -> xr.Dataset:
    """Create a sample velocity field for testing."""
    lat = np.arange(-90, 91, 10)
    lon = np.arange(-180, 181, 20)
    lon_rad, lat_rad = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
    u = -np.sin(lat_rad) * np.cos(lon_rad)
    v = np.sin(lon_rad)
    return xr.Dataset(
        {'u': (('lat', 'lon'), u), 'v': (('lat', 'lon'), v)},
        coords={'lat': lat, 'lon': lon},
    )


def test_run_trajectory_output_structure(sample_velocity_field):
    """Test the basic output structure of the run_trajectory function."""
    start = {'lat': 40.0, 'lon': -120.0}
    num_steps = 10
    trajectory = run_trajectory(start, sample_velocity_field, num_steps)

    assert isinstance(trajectory, xr.Dataset)
    assert 'lat' in trajectory
    assert 'lon' in trajectory
    assert 'time' in trajectory.coords
    assert len(trajectory['time']) == num_steps + 1
    assert 'history' in trajectory.attrs


def test_run_trajectory_calculation(sample_velocity_field):
    """Test the trajectory calculation for a known case."""
    start = {'lat': 0.0, 'lon': 0.0}
    num_steps = 1
    trajectory = run_trajectory(start, sample_velocity_field, num_steps)

    # At (0, 0), u should be 0 and v should be 0 from the sample field
    expected_lat = xr.DataArray([0.0, 0.0], dims=['time'], coords={'time': [0, 1]})
    expected_lon = xr.DataArray([0.0, 0.0], dims=['time'], coords={'time': [0, 1]})

    assert_allclose(trajectory['lat'], expected_lat)
    assert_allclose(trajectory['lon'], expected_lon)
