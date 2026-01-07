
"""Unit tests for the core trajectory model."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from unittest.mock import patch
from plat.core import _calculate_interpolation_weights_jit, run_trajectory


def test_calculate_interpolation_weights_jit():
    """
    Test the _calculate_interpolation_weights_jit function.

    This test validates two scenarios:
    1.  Standard case: A point between two grid cells.
    2.  Edge case: A point on a grid cell with a co-located neighbor (delta=0),
        which should return weights of 1.0 and 0.0.
    """
    grid = np.array([10.0, 20.0, 30.0, 30.0])

    # --- Test standard interpolation ---
    # Point is 25% of the way between 10.0 and 20.0
    point1 = 12.5
    i1 = 0
    w1, w2, i1_out, i2_out = _calculate_interpolation_weights_jit(point1, grid, i1)
    # Expect w1=0.75 (closer to 10.0), w2=0.25 (further from 20.0)
    assert np.isclose(w1, 0.75)
    assert np.isclose(w2, 0.25)
    assert i1_out == 0
    assert i2_out == 1

    # --- Test edge case with delta = 0 ---
    point2 = 30.0
    i2 = 2
    w1, w2, i1_out, i2_out = _calculate_interpolation_weights_jit(point2, grid, i2)
    # Expect weights to be 1.0 and 0.0, avoiding division by zero
    assert np.isclose(w1, 1.0)
    assert np.isclose(w2, 0.0)
    assert i1_out == 2
    assert i2_out == 3


@pytest.fixture
def zero_velocity_field() -> xr.Dataset:
    """Create a 4D velocity field with zero velocity everywhere."""
    lat = np.arange(-90, 91, 10)
    lon = np.arange(-180, 181, 20)
    level = np.array([1000, 850, 500])
    time = pd.to_datetime(['2023-01-01T00:00:00', '2023-01-01T01:00:00'])
    u = np.zeros((len(time), len(level), len(lat), len(lon)))
    v = np.zeros((len(time), len(level), len(lat), len(lon)))
    w = np.zeros((len(time), len(level), len(lat), len(lon)))
    return xr.Dataset(
        {
            'u': (('time', 'level', 'lat', 'lon'), u),
            'v': (('time', 'level', 'lat', 'lon'), v),
            'w': (('time', 'level', 'lat', 'lon'), w),
        },
        coords={'time': time, 'level': level, 'lat': lat, 'lon': lon},
    )


@pytest.fixture
def constant_velocity_field() -> xr.Dataset:
    """Create a 4D velocity field with constant velocity everywhere."""
    lat = np.arange(-90, 91, 10)
    lon = np.arange(-180, 181, 20)
    level = np.array([1000, 850, 500])
    time = pd.to_datetime(['2023-01-01T00:00:00', '2023-01-01T01:00:00'])
    u = np.ones((len(time), len(level), len(lat), len(lon)))
    v = np.ones((len(time), len(level), len(lat), len(lon)))
    w = np.zeros((len(time), len(level), len(lat), len(lon)))  # No vertical motion
    return xr.Dataset(
        {
            'u': (('time', 'level', 'lat', 'lon'), u),
            'v': (('time', 'level', 'lat', 'lon'), v),
            'w': (('time', 'level', 'lat', 'lon'), w),
        },
        coords={'time': time, 'level': level, 'lat': lat, 'lon': lon},
    )


def test_run_trajectory_stationary_particle(zero_velocity_field):
    """
    Test that a particle in a zero-velocity field remains stationary.
    """
    start = {
        'lat': [40.0],
        'lon': [-120.0],
        'level': [850.0],
        'time': pd.Timestamp('2023-01-01T00:00:00'),
    }
    num_steps = 10
    trajectory = run_trajectory(start, zero_velocity_field, num_steps)

    # --- Check that the particle has not moved ---
    assert np.all(np.isclose(trajectory['lat'].values, start['lat']))
    assert np.all(np.isclose(trajectory['lon'].values, start['lon']))
    assert np.all(np.isclose(trajectory['level'].values, start['level']))


def test_run_trajectory_constant_velocity(constant_velocity_field):
    """
    Test that a particle in a constant-velocity field moves as expected.
    """
    start = {
        'lat': [40.0],
        'lon': [-120.0],
        'level': [850.0],
        'time': pd.Timestamp('2023-01-01T00:00:00'),
    }
    num_steps = 10
    trajectory = run_trajectory(start, constant_velocity_field, num_steps)

    # --- Check that the particle has moved the correct distance ---
    expected_lat = start['lat'][0] + num_steps
    expected_lon = start['lon'][0] + num_steps

    final_point = trajectory.isel(particle=0, step=-1)
    assert np.isclose(final_point['lat'], expected_lat)
    assert np.isclose(final_point['lon'], expected_lon)
    assert np.isclose(final_point['level'], start['level'][0])


@pytest.fixture
def gradient_velocity_field_4d() -> xr.Dataset:
    """Create a 4D velocity field with a linear gradient."""
    lat = np.array([30, 40])
    lon = np.array([-120, -110])
    level = np.array([800, 700])
    time = pd.to_datetime(['2023-01-01T00:00:00', '2023-01-01T01:00:00'])
    # u increases with longitude, v with latitude, w with level
    u_2d = np.array([[1, 2], [1, 2]])
    v_2d = np.array([[1, 1], [2, 2]])
    w_2d = np.array([[-10, -10], [-10, -10]])  # w=-10 at level=800

    u = np.stack([u_2d, u_2d])
    v = np.stack([v_2d, v_2d])
    w = np.stack([w_2d, w_2d * 2])  # w=-20 at level=700

    u_4d = np.stack([u, u])
    v_4d = np.stack([v, v])
    w_4d = np.stack([w, w])

    return xr.Dataset(
        {
            'u': (('time', 'level', 'lat', 'lon'), u_4d),
            'v': (('time', 'level', 'lat', 'lon'), v_4d),
            'w': (('time', 'level', 'lat', 'lon'), w_4d),
        },
        coords={'time': time, 'level': level, 'lat': lat, 'lon': lon},
    )


def test_run_trajectory_quadrilinear_interpolation(gradient_velocity_field_4d):
    """
    Test that the trajectory integration correctly uses quadrilinear interpolation.
    """
    # Start the particle exactly halfway between grid points.
    start = {
        'lat': [35.0],
        'lon': [-115.0],
        'level': [750.0],
        'time': pd.Timestamp('2023-01-01T00:00:00'),
    }
    num_steps = 1
    trajectory = run_trajectory(start, gradient_velocity_field_4d, num_steps)

    # The expected values are from a manual calculation of a single RK4 step
    # with the given gradient field.
    expected_lat = 36.5775625
    expected_lon = -113.4224375
    expected_level = 734.224375

    # --- Check that the particle has moved to the interpolated position ---
    final_point = trajectory.isel(particle=0, step=-1)
    assert np.isclose(final_point['lat'], expected_lat)
    assert np.isclose(final_point['lon'], expected_lon)
    assert np.isclose(final_point['level'], expected_level)


@pytest.fixture
def solid_body_rotation_field() -> xr.Dataset:
    """
    Create a 4D velocity field corresponding to solid body rotation.
    The vertical velocity is zero.
    """
    lat = np.arange(30, 51, 1)
    lon = np.arange(-110, -89, 1)
    level = np.array([1000, 850, 500])
    time = pd.to_datetime(['2023-01-01T00:00:00', '2023-01-01T06:00:00'])
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Center of rotation
    lat_center, lon_center = 40, -100
    omega = 0.5  # Angular velocity

    # Velocities in a small-angle approximation
    u_2d = -omega * (lat_grid - lat_center)
    v_2d = omega * (lon_grid - lon_center)

    # Add level and time dimensions
    u = np.stack([u_2d] * len(level))
    v = np.stack([v_2d] * len(level))
    w = np.zeros_like(u)
    u_4d = np.stack([u] * len(time))
    v_4d = np.stack([v] * len(time))
    w_4d = np.stack([w] * len(time))

    return xr.Dataset(
        {
            'u': (('time', 'level', 'lat', 'lon'), u_4d),
            'v': (('time', 'level', 'lat', 'lon'), v_4d),
            'w': (('time', 'level', 'lat', 'lon'), w_4d),
        },
        coords={'time': time, 'level': level, 'lat': lat, 'lon': lon},
    )


def test_run_trajectory_solid_body_rotation(solid_body_rotation_field):
    """
    Test trajectory in a solid-body rotation field.

    A particle in this field should maintain a constant distance from the
    center of rotation and a constant vertical level.
    """
    start = {
        'lat': [45.0],
        'lon': [-100.0],
        'level': [850.0],
        'time': pd.Timestamp('2023-01-01T00:00:00'),
    }
    num_steps = 25
    dt = 0.1  # Use a smaller time step for better accuracy

    trajectory = run_trajectory(
        start, solid_body_rotation_field, num_steps, dt=dt
    )

    # --- Check that the particle maintains a constant radius from the center ---
    lat_center, lon_center = 40, -100
    initial_point = trajectory.isel(particle=0, step=0)
    final_point = trajectory.isel(particle=0, step=-1)

    radius_initial = np.sqrt(
        (initial_point['lat'] - lat_center) ** 2
        + (initial_point['lon'] - lon_center) ** 2
    )
    radius_final = np.sqrt(
        (final_point['lat'] - lat_center) ** 2
        + (final_point['lon'] - lon_center) ** 2
    )

    # The radius should be nearly constant (within a small tolerance)
    assert np.isclose(radius_initial, radius_final, rtol=1e-3)
    # The level should be constant
    assert np.all(np.isclose(trajectory['level'].values, start['level']))


@pytest.fixture
def constant_vertical_velocity_field() -> xr.Dataset:
    """Create a 4D velocity field with constant vertical velocity."""
    lat = np.arange(30, 41, 1)
    lon = np.arange(-110, -99, 1)
    level = np.arange(1000, 400, -100)
    time = pd.to_datetime(['2023-01-01T00:00:00', '2023-01-01T01:00:00'])
    u = np.zeros((len(time), len(level), len(lat), len(lon)))
    v = np.zeros((len(time), len(level), len(lat), len(lon)))
    w = np.full((len(time), len(level), len(lat), len(lon)), -50.0)  # Constant decent
    return xr.Dataset(
        {
            'u': (('time', 'level', 'lat', 'lon'), u),
            'v': (('time', 'level', 'lat', 'lon'), v),
            'w': (('time', 'level', 'lat', 'lon'), w),
        },
        coords={'time': time, 'level': level, 'lat': lat, 'lon': lon},
    )


def test_run_trajectory_4d_vertical_motion(constant_vertical_velocity_field):
    """
    Test a 4D trajectory with only vertical motion.
    """
    start = {
        'lat': [35.0],
        'lon': [-105.0],
        'level': [850.0],
        'time': pd.Timestamp('2023-01-01T00:00:00'),
    }
    num_steps = 5
    dt = 1.0  # 1 hour time step

    trajectory = run_trajectory(
        start, constant_vertical_velocity_field, num_steps, dt=dt
    )

    # --- Check that the particle has moved vertically ---
    expected_level = start['level'][0] + (-50.0 * num_steps)
    final_level = trajectory.isel(particle=0, step=-1)['level']
    assert np.isclose(final_level, expected_level)

    # --- Check that the particle has not moved horizontally ---
    assert np.all(np.isclose(trajectory['lat'].values, start['lat']))
    assert np.all(np.isclose(trajectory['lon'].values, start['lon']))


def test_run_trajectory_multiple_particles(constant_velocity_field):
    """
    Test running the model with multiple particles simultaneously.
    """
    start_points = {
        'lat': [40.0, 42.0],
        'lon': [-120.0, -122.0],
        'level': [850.0, 850.0],
        'time': pd.Timestamp('2023-01-01T00:00:00'),
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

    final_point_p1 = trajectory.isel(particle=0, step=-1)
    final_point_p2 = trajectory.isel(particle=1, step=-1)

    assert np.isclose(final_point_p1['lat'], expected_lat_p1)
    assert np.isclose(final_point_p1['lon'], expected_lon_p1)
    assert np.isclose(final_point_p2['lat'], expected_lat_p2)
    assert np.isclose(final_point_p2['lon'], expected_lon_p2)


@pytest.fixture
def time_varying_velocity_field() -> xr.Dataset:
    """Create a 4D velocity field where u increases with time."""
    lat = np.array([30, 40])
    lon = np.array([-120, -110])
    level = np.array([800])
    time = pd.to_datetime(['2023-01-01T00:00:00', '2023-01-01T01:00:00', '2023-01-01T02:00:00'])

    # At T0, u is 1.0 everywhere. At T1, u is 2.0. At T2, u is 3.0
    u_t0 = np.ones((len(level), len(lat), len(lon)))
    u_t1 = np.full((len(level), len(lat), len(lon)), 2.0)
    u_t2 = np.full((len(level), len(lat), len(lon)), 3.0)
    u = np.stack([u_t0, u_t1, u_t2])

    v = np.zeros_like(u)
    w = np.zeros_like(u)

    return xr.Dataset(
        {
            'u': (('time', 'level', 'lat', 'lon'), u),
            'v': (('time', 'level', 'lat', 'lon'), v),
            'w': (('time', 'level', 'lat', 'lon'), w),
        },
        coords={'time': time, 'level': level, 'lat': lat, 'lon': lon},
    )


def test_run_trajectory_time_interpolation(time_varying_velocity_field):
    """
    Test that the trajectory correctly interpolates velocity in time.
    """
    # Start the particle halfway in time between the two time slices.
    start = {
        'lat': [35.0],
        'lon': [-115.0],
        'level': [800.0],
        'time': pd.Timestamp('2023-01-01T00:30:00'),
    }
    num_steps = 1
    dt = 1.0  # 1 hour step

    trajectory = run_trajectory(
        start, time_varying_velocity_field, num_steps, dt=dt
    )

    # Manual RK4 calculation for this specific case:
    # u_initial is 1.5 (halfway between 1.0 and 2.0)
    # The final velocity is (k1 + 2*k2 + 2*k3 + k4) / 6
    # k1_u = 1.5
    # k2_u, k3_u are at t=1h, so u=2.0
    # k4 is at t=1.5h, so u is 2.5
    # u_final = (1.5 + 2*2.0 + 2*2.0 + 2.5)/6 = 12.0/6 = 2.0
    expected_lon = start['lon'][0] + 2.0 * dt

    final_point = trajectory.isel(particle=0, step=-1)
    assert np.isclose(final_point['lon'], expected_lon)
    assert np.isclose(final_point['lat'], start['lat'][0])
    assert np.isclose(final_point['level'], start['level'][0])
    assert final_point['time'] == pd.Timestamp('2023-01-01T01:30:00')


@patch('plat.core._integrate_jit')
def test_run_trajectory_lazy_subsetting(mock_integrate_jit):
    """
    Test that run_trajectory only loads a subset of a large, Dask-backed dataset.
    """
    # --- 1. Create a large, Dask-chunked velocity field ---
    # The grid is 100x100x100x10, which would be large if loaded into memory.
    lat = np.linspace(0, 90, 100)
    lon = np.linspace(-180, 180, 100)
    level = np.linspace(1000, 100, 100)
    time = pd.to_datetime(pd.date_range('2023-01-01', periods=10, freq='1H'))

    # Create a dummy velocity field of all ones
    ds = xr.Dataset(
        {
            'u': (('time', 'level', 'lat', 'lon'), np.ones((10, 100, 100, 100))),
            'v': (('time', 'level', 'lat', 'lon'), np.ones((10, 100, 100, 100))),
            'w': (('time', 'level', 'lat', 'lon'), np.zeros((10, 100, 100, 100))),
        },
        coords={'time': time, 'level': level, 'lat': lat, 'lon': lon},
    )
    # Use Dask to make the dataset lazy. Choose small chunks to ensure
    # that the subsetting logic is properly tested.
    velocity_field_lazy = ds.chunk({'lat': 10, 'lon': 10, 'level': 10, 'time': 1})

    # --- 2. Define starting points and run the trajectory ---
    start = {
        'lat': [45.0],
        'lon': [0.0],
        'level': [500.0],
        'time': pd.Timestamp('2023-01-01T00:00:00'),
    }
    # Run for a short duration to ensure the subset is small.
    num_steps = 2
    dt = 1.0

    # Mock the return of _integrate_jit to avoid running the actual integration
    mock_integrate_jit.return_value = (
        np.zeros((1, num_steps + 1)),
        np.zeros((1, num_steps + 1)),
        np.zeros((1, num_steps + 1)),
        np.zeros((1, num_steps + 1)),
    )
    run_trajectory(start, velocity_field_lazy, num_steps, dt=dt)

    # --- 3. Assert that the subset passed to the JIT kernel is small ---
    # Get the arguments that were passed to the mocked _integrate_jit function.
    # The arrays are the 8th, 9th, and 10th arguments.
    args, _ = mock_integrate_jit.call_args
    u_data_subset = args[8]
    v_data_subset = args[9]
    w_data_subset = args[10]

    # The original dataset has 10 * 100 * 100 * 100 = 10,000,000 points per variable.
    # The subset should be much smaller. The exact size depends on the
    # velocity and trajectory duration, but it should be a tiny fraction of the original.
    # Here, we assert that it's less than 1% of the original size.
    assert u_data_subset.size < (velocity_field_lazy['u'].size * 0.01)
    assert v_data_subset.size < (velocity_field_lazy['v'].size * 0.01)
    assert w_data_subset.size < (velocity_field_lazy['w'].size * 0.01)

    # Also, assert that the input data is a NumPy array, not a Dask array,
    # which proves that the data has been loaded into memory *after* subsetting.
    assert isinstance(u_data_subset, np.ndarray)
    assert not hasattr(u_data_subset, 'dask')


def test_run_trajectory_multiple_start_times(constant_velocity_field):
    """
    Test that subsetting works correctly with multiple particles starting at
    different times.
    """
    # Use a velocity field with more time slices to test the selection
    time_coords = pd.to_datetime(pd.date_range('2023-01-01', periods=5, freq='1H'))
    vel_field = constant_velocity_field.reindex({'time': time_coords}, method='pad')

    start_points = {
        'lat': [40.0, 42.0],
        'lon': [-120.0, -122.0],
        'level': [850.0, 850.0],
        'time': [
            pd.Timestamp('2023-01-01T00:00:00'),
            pd.Timestamp('2023-01-01T01:00:00'),
        ],
    }
    num_steps = 2
    trajectory = run_trajectory(start_points, vel_field, num_steps)

    # --- Check that both particles moved the correct distance ---
    # Particle 1 starts at T0 and moves for 2 steps
    expected_lat_p1 = start_points['lat'][0] + num_steps
    expected_lon_p1 = start_points['lon'][0] + num_steps
    # Particle 2 starts at T1 and moves for 2 steps
    expected_lat_p2 = start_points['lat'][1] + num_steps
    expected_lon_p2 = start_points['lon'][1] + num_steps

    # The final time coordinate for particle 2 will be T1 + 2 hours = T3
    # The final time coordinate for particle 1 will be T0 + 2 hours = T2
    final_point_p1 = trajectory.isel(particle=0, step=-1)
    final_point_p2 = trajectory.isel(particle=1, step=-1)

    assert np.isclose(final_point_p1['lat'], expected_lat_p1)
    assert np.isclose(final_point_p1['lon'], expected_lon_p1)
    assert np.isclose(final_point_p2['lat'], expected_lat_p2)
    assert np.isclose(final_point_p2['lon'], expected_lon_p2)
