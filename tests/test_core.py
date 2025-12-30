
"""Unit tests for the core trajectory model."""

import numpy as np
import pytest
import xarray as xr
from plat.core import run_trajectory


@pytest.fixture
def zero_velocity_field() -> xr.Dataset:
    """Create a 3D velocity field with zero velocity everywhere."""
    lat = np.arange(-90, 91, 10)
    lon = np.arange(-180, 181, 20)
    level = np.array([1000, 850, 500])
    u = np.zeros((len(level), len(lat), len(lon)))
    v = np.zeros((len(level), len(lat), len(lon)))
    w = np.zeros((len(level), len(lat), len(lon)))
    return xr.Dataset(
        {
            'u': (('level', 'lat', 'lon'), u),
            'v': (('level', 'lat', 'lon'), v),
            'w': (('level', 'lat', 'lon'), w),
        },
        coords={'level': level, 'lat': lat, 'lon': lon},
    )


@pytest.fixture
def constant_velocity_field() -> xr.Dataset:
    """Create a 3D velocity field with constant velocity everywhere."""
    lat = np.arange(-90, 91, 10)
    lon = np.arange(-180, 181, 20)
    level = np.array([1000, 850, 500])
    u = np.ones((len(level), len(lat), len(lon)))
    v = np.ones((len(level), len(lat), len(lon)))
    w = np.zeros((len(level), len(lat), len(lon)))  # No vertical motion
    return xr.Dataset(
        {
            'u': (('level', 'lat', 'lon'), u),
            'v': (('level', 'lat', 'lon'), v),
            'w': (('level', 'lat', 'lon'), w),
        },
        coords={'level': level, 'lat': lat, 'lon': lon},
    )


def test_run_trajectory_stationary_particle(zero_velocity_field):
    """
    Test that a particle in a zero-velocity field remains stationary.
    """
    start = {'lat': [40.0], 'lon': [-120.0], 'level': [850.0]}
    num_steps = 10
    trajectory = run_trajectory(start, zero_velocity_field, num_steps)

    # --- Check that the particle has not moved ---
    assert np.all(trajectory['lat'].values == start['lat'])
    assert np.all(trajectory['lon'].values == start['lon'])
    assert np.all(trajectory['level'].values == start['level'])


def test_run_trajectory_constant_velocity(constant_velocity_field):
    """
    Test that a particle in a constant-velocity field moves as expected.
    """
    start = {'lat': [40.0], 'lon': [-120.0], 'level': [850.0]}
    num_steps = 10
    trajectory = run_trajectory(start, constant_velocity_field, num_steps)

    # --- Check that the particle has moved the correct distance ---
    expected_lat = start['lat'][0] + num_steps
    expected_lon = start['lon'][0] + num_steps

    assert trajectory['lat'].values[0, -1] == expected_lat
    assert trajectory['lon'].values[0, -1] == expected_lon
    assert trajectory['level'].values[0, -1] == start['level'][0]


@pytest.fixture
def gradient_velocity_field_3d() -> xr.Dataset:
    """Create a 3D velocity field with a linear gradient."""
    lat = np.array([30, 40])
    lon = np.array([-120, -110])
    level = np.array([800, 700])
    # u increases with longitude, v with latitude, w with level
    u_2d = np.array([[1, 2], [1, 2]])
    v_2d = np.array([[1, 1], [2, 2]])
    w_2d = np.array([[-10, -10], [-10, -10]])  # w=-10 at level=800

    u = np.stack([u_2d, u_2d])
    v = np.stack([v_2d, v_2d])
    w = np.stack([w_2d, w_2d * 2])  # w=-20 at level=700

    return xr.Dataset(
        {
            'u': (('level', 'lat', 'lon'), u),
            'v': (('level', 'lat', 'lon'), v),
            'w': (('level', 'lat', 'lon'), w),
        },
        coords={'level': level, 'lat': lat, 'lon': lon},
    )


def test_run_trajectory_trilinear_interpolation(gradient_velocity_field_3d):
    """
    Test that the trajectory integration correctly uses trilinear interpolation.
    """
    # Start the particle exactly halfway between grid points.
    start = {'lat': [35.0], 'lon': [-115.0], 'level': [750.0]}
    num_steps = 1
    trajectory = run_trajectory(start, gradient_velocity_field_3d, num_steps)

    # The expected values are from a manual calculation of a single RK4 step
    # with the given gradient field.
    expected_lat = 36.5775625
    expected_lon = -113.4224375
    expected_level = 734.224375

    # --- Check that the particle has moved to the interpolated position ---
    assert np.isclose(trajectory['lat'].values[0, -1], expected_lat)
    assert np.isclose(trajectory['lon'].values[0, -1], expected_lon)
    assert np.isclose(trajectory['level'].values[0, -1], expected_level)


@pytest.fixture
def solid_body_rotation_field() -> xr.Dataset:
    """
    Create a 3D velocity field corresponding to solid body rotation.
    The vertical velocity is zero.
    """
    lat = np.arange(30, 51, 1)
    lon = np.arange(-110, -89, 1)
    level = np.array([1000, 850, 500])
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Center of rotation
    lat_center, lon_center = 40, -100
    omega = 0.5  # Angular velocity

    # Velocities in a small-angle approximation
    u_2d = -omega * (lat_grid - lat_center)
    v_2d = omega * (lon_grid - lon_center)

    # Add a level dimension
    u = np.stack([u_2d] * len(level))
    v = np.stack([v_2d] * len(level))
    w = np.zeros_like(u)

    return xr.Dataset(
        {
            'u': (('level', 'lat', 'lon'), u),
            'v': (('level', 'lat', 'lon'), v),
            'w': (('level', 'lat', 'lon'), w),
        },
        coords={'level': level, 'lat': lat, 'lon': lon},
    )


def test_run_trajectory_solid_body_rotation(solid_body_rotation_field):
    """
    Test trajectory in a solid-body rotation field.

    A particle in this field should maintain a constant distance from the
    center of rotation and a constant vertical level.
    """
    start = {'lat': [45.0], 'lon': [-100.0], 'level': [850.0]}
    num_steps = 25
    dt = 0.1  # Use a smaller time step for better accuracy

    trajectory = run_trajectory(
        start, solid_body_rotation_field, num_steps, dt=dt
    )

    # --- Check that the particle maintains a constant radius from the center ---
    lat_center, lon_center = 40, -100
    radius_initial = np.sqrt(
        (trajectory['lat'].values[0, 0] - lat_center) ** 2
        + (trajectory['lon'].values[0, 0] - lon_center) ** 2
    )
    radius_final = np.sqrt(
        (trajectory['lat'].values[0, -1] - lat_center) ** 2
        + (trajectory['lon'].values[0, -1] - lon_center) ** 2
    )

    # The radius should be nearly constant (within a small tolerance)
    assert np.isclose(radius_initial, radius_final, rtol=1e-3)
    # The level should be constant
    assert np.all(trajectory['level'].values == start['level'])


@pytest.fixture
def constant_vertical_velocity_field() -> xr.Dataset:
    """Create a 3D velocity field with constant vertical velocity."""
    lat = np.arange(30, 41, 1)
    lon = np.arange(-110, -99, 1)
    level = np.arange(1000, 400, -100)
    u = np.zeros((len(level), len(lat), len(lon)))
    v = np.zeros((len(level), len(lat), len(lon)))
    w = np.full((len(level), len(lat), len(lon)), -50.0)  # Constant decent
    return xr.Dataset(
        {
            'u': (('level', 'lat', 'lon'), u),
            'v': (('level', 'lat', 'lon'), v),
            'w': (('level', 'lat', 'lon'), w),
        },
        coords={'level': level, 'lat': lat, 'lon': lon},
    )


def test_run_trajectory_3d_vertical_motion(constant_vertical_velocity_field):
    """
    Test a 3D trajectory with only vertical motion.
    """
    start = {'lat': [35.0], 'lon': [-105.0], 'level': [850.0]}
    num_steps = 5
    dt = 1.0  # 1 hour time step

    trajectory = run_trajectory(
        start, constant_vertical_velocity_field, num_steps, dt=dt
    )

    # --- Check that the particle has moved vertically ---
    expected_level = start['level'][0] + (-50.0 * num_steps)
    assert np.isclose(trajectory['level'].values[0, -1], expected_level)

    # --- Check that the particle has not moved horizontally ---
    assert np.all(trajectory['lat'].values == start['lat'])
    assert np.all(trajectory['lon'].values == start['lon'])


def test_run_trajectory_multiple_particles(constant_velocity_field):
    """
    Test running the model with multiple particles simultaneously.
    """
    start_points = {
        'lat': [40.0, 42.0],
        'lon': [-120.0, -122.0],
        'level': [850.0, 850.0],
    }
    num_steps = 10
    trajectory = run_trajectory(
        start_points, constant_velocity_field, num_steps
    )

    # --- Check that both particles moved the correct distance ---
    expected_lat_p1 = start_points['lat'][0] + num_steps
    expected_lon_p1 = start_points['lon'][0] + num_steps
    expected_lat_p2 = start_points['lat'][1] + num_steps
    expected_lon_p2 = start_points['lon'][1] + num_steps

    assert trajectory['lat'].values[0, -1] == expected_lat_p1
    assert trajectory['lon'].values[0, -1] == expected_lon_p1
    assert trajectory['lat'].values[1, -1] == expected_lat_p2
    assert trajectory['lon'].values[1, -1] == expected_lon_p2
