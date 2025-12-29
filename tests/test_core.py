
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


@pytest.fixture
def gradient_velocity_field() -> xr.Dataset:
    """Create a velocity field with a linear gradient."""
    lat = np.array([30, 40])
    lon = np.array([-120, -110])
    # v increases with latitude, u increases with longitude
    v = np.array([[1, 1], [2, 2]])  # v=1 at lat=30, v=2 at lat=40
    u = np.array([[1, 2], [1, 2]])  # u=1 at lon=-120, u=2 at lon=-110
    return xr.Dataset(
        {'u': (('lat', 'lon'), u), 'v': (('lat', 'lon'), v)},
        coords={'lat': lat, 'lon': lon},
    )


def test_run_trajectory_bilinear_interpolation(gradient_velocity_field):
    """
    Test that the trajectory integration correctly uses bilinear interpolation.
    """
    # Start the particle exactly halfway between grid points.
    start = {'lat': 35.0, 'lon': -115.0}
    num_steps = 1
    trajectory = run_trajectory(start, gradient_velocity_field, num_steps)

    # --- Manually calculate the expected interpolated velocity ---
    # With the RK4 integration, the final position will be slightly different
    # than a simple single-step Euler integration.
    # The expected values below are calculated from a manual RK4 calculation.
    expected_lat = 36.5775625
    expected_lon = -113.4224375


    # --- Check that the particle has moved to the interpolated position ---
    assert np.isclose(trajectory['lat'].values[-1], expected_lat)
    assert np.isclose(trajectory['lon'].values[-1], expected_lon)


@pytest.fixture
def solid_body_rotation_field() -> xr.Dataset:
    """
    Create a velocity field corresponding to solid body rotation.

    The center of rotation is at lat=40, lon=-100.
    """
    lat = np.arange(30, 51, 1)
    lon = np.arange(-110, -89, 1)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Center of rotation
    lat_center, lon_center = 40, -100
    omega = 0.5  # Angular velocity

    # Velocities in a small-angle approximation
    u = -omega * (lat_grid - lat_center)
    v = omega * (lon_grid - lon_center)

    return xr.Dataset(
        {'u': (('lat', 'lon'), u), 'v': (('lat', 'lon'), v)},
        coords={'lat': lat, 'lon': lon},
    )


def test_run_trajectory_solid_body_rotation(solid_body_rotation_field):
    """
    Test trajectory in a solid-body rotation field.

    A particle in this field should maintain a constant distance from the
    center of rotation. This is a strong test of the RK4 integrator's accuracy.
    """
    start = {'lat': 45.0, 'lon': -100.0}  # Start north of the center
    num_steps = 25
    dt = 0.1  # Use a smaller time step for better accuracy

    trajectory = run_trajectory(
        start, solid_body_rotation_field, num_steps, dt=dt
    )

    # --- Check that the particle maintains a constant radius from the center ---
    lat_center, lon_center = 40, -100
    radius_initial = np.sqrt(
        (trajectory['lat'].values[0] - lat_center) ** 2
        + (trajectory['lon'].values[0] - lon_center) ** 2
    )
    radius_final = np.sqrt(
        (trajectory['lat'].values[-1] - lat_center) ** 2
        + (trajectory['lon'].values[-1] - lon_center) ** 2
    )

    # The radius should be nearly constant (within a small tolerance)
    assert np.isclose(radius_initial, radius_final, rtol=1e-3)
