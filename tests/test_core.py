
"""Unit tests for the core trajectory model."""

import numpy as np
import pytest
import xarray as xr
from plat.core import run_trajectory


@pytest.fixture
def zero_velocity_field() -> xr.Dataset:
    """Create a velocity field with zero velocity everywhere."""
    lat = np.arange(-90, 91, 10)
    lon = np.arange(-180, 181, 20)
    u = np.zeros((len(lat), len(lon)))
    v = np.zeros((len(lat), len(lon)))
    return xr.Dataset(
        {'u': (('lat', 'lon'), u), 'v': (('lat', 'lon'), v)},
        coords={'lat': lat, 'lon': lon},
    )


@pytest.fixture
def constant_velocity_field() -> xr.Dataset:
    """Create a velocity field with constant velocity everywhere."""
    lat = np.arange(-90, 91, 10)
    lon = np.arange(-180, 181, 20)
    u = np.ones((len(lat), len(lon)))
    v = np.ones((len(lat), len(lon)))
    return xr.Dataset(
        {'u': (('lat', 'lon'), u), 'v': (('lat', 'lon'), v)},
        coords={'lat': lat, 'lon': lon},
    )


def test_run_trajectory_stationary_particle(zero_velocity_field):
    """
    Test that a particle in a zero-velocity field remains stationary.
    """
    start = {'lat': 40.0, 'lon': -120.0}
    num_steps = 10
    trajectory = run_trajectory(start, zero_velocity_field, num_steps)

    # --- Check that the particle has not moved ---
    assert np.all(trajectory['lat'].values == start['lat'])
    assert np.all(trajectory['lon'].values == start['lon'])


def test_run_trajectory_constant_velocity(constant_velocity_field):
    """
    Test that a particle in a constant-velocity field moves as expected.
    """
    start = {'lat': 40.0, 'lon': -120.0}
    num_steps = 10
    trajectory = run_trajectory(start, constant_velocity_field, num_steps)

    # --- Check that the particle has moved the correct distance ---
    # The expected position is the starting position plus the velocity
    # multiplied by the number of steps.
    expected_lat = start['lat'] + num_steps
    expected_lon = start['lon'] + num_steps

    assert trajectory['lat'].values[-1] == expected_lat
    assert trajectory['lon'].values[-1] == expected_lon
