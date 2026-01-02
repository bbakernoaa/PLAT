
"""Core atmospheric trajectory model."""

from typing import Dict, Union

import numba
import numpy as np
import pandas as pd
import xarray as xr


@numba.jit(nopython=True)
def _calculate_interpolation_weights_jit(
    point: float, grid: np.ndarray, i: int
) -> tuple[float, float, int, int]:
    """
    Calculate interpolation weights and clamp indices for a single dimension.

    Parameters
    ----------
    point : float
        The coordinate of the point for which to calculate weights.
    grid : np.ndarray
        The 1D grid of coordinates.
    i : int
        The lower-bound index for the point in the grid.

    Returns
    -------
    tuple[float, float, int, int]
        A tuple containing the weight for the lower grid point (w1), the
        weight for the upper grid point (w2), and the clamped lower (i1)
        and upper (i2) indices.
    """
    # Clamp index to be within bounds
    i = max(0, min(i, len(grid) - 2))
    i1, i2 = i, i + 1

    # Grid points surrounding the current location
    grid1, grid2 = grid[i1], grid[i2]

    # Calculate interpolation weights, avoiding division by zero
    delta = grid2 - grid1
    if delta == 0:
        return 1.0, 0.0, i1, i2

    w1 = (grid2 - point) / delta
    w2 = (point - grid1) / delta

    return w1, w2, i1, i2


@numba.jit(nopython=True)
def _quadrilinear_interpolation_jit(
    lat: float,
    lon: float,
    level: float,
    time: float,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    grid_level: np.ndarray,
    grid_time: np.ndarray,
    data: np.ndarray,
) -> float:
    """
    Perform quadrilinear interpolation on a 4D grid.

    This function is JIT-compiled with Numba for performance.

    Parameters
    ----------
    lat : float
        Latitude of the interpolation point.
    lon : float
        Longitude of the interpolation point.
    level : float
        Vertical level of the interpolation point.
    time : float
        Time of the interpolation point.
    grid_lat : np.ndarray
        A 1D array of latitude coordinates for the grid.
    grid_lon : np.ndarray
        A 1D array of longitude coordinates for the grid.
    grid_level : np.ndarray
        A 1D array of vertical level coordinates for the grid.
    grid_time : np.ndarray
        A 1D array of time coordinates for the grid.
    data : np.ndarray
        A 4D array of data values corresponding to the grid. The expected
        dimension order is (time, level, lat, lon).

    Returns
    -------
    float
        The interpolated value at the given lat/lon/level/time point.
    """
    # Find lower-bound indices
    i = np.searchsorted(grid_lat, lat) - 1
    j = np.searchsorted(grid_lon, lon) - 1
    k = np.searchsorted(grid_level, level) - 1
    m = np.searchsorted(grid_time, time) - 1

    # Calculate weights and clamped indices
    w_lat1, w_lat2, i1, i2 = _calculate_interpolation_weights_jit(lat, grid_lat, i)
    w_lon1, w_lon2, j1, j2 = _calculate_interpolation_weights_jit(lon, grid_lon, j)
    w_level1, w_level2, k1, k2 = _calculate_interpolation_weights_jit(
        level, grid_level, k
    )
    w_time1, w_time2, m1, m2 = _calculate_interpolation_weights_jit(time, grid_time, m)

    # Data values at the 16 corners of the hypercube
    # Time slice 1
    q1111 = data[m1, k1, i1, j1]
    q1112 = data[m1, k1, i1, j2]
    q1121 = data[m1, k1, i2, j1]
    q1122 = data[m1, k1, i2, j2]
    q1211 = data[m1, k2, i1, j1]
    q1212 = data[m1, k2, i1, j2]
    q1221 = data[m1, k2, i2, j1]
    q1222 = data[m1, k2, i2, j2]

    # Time slice 2
    q2111 = data[m2, k1, i1, j1]
    q2112 = data[m2, k1, i1, j2]
    q2121 = data[m2, k1, i2, j1]
    q2122 = data[m2, k1, i2, j2]
    q2211 = data[m2, k2, i1, j1]
    q2212 = data[m2, k2, i1, j2]
    q2221 = data[m2, k2, i2, j1]
    q2222 = data[m2, k2, i2, j2]

    # Interpolate along longitude (x-axis) for both time slices
    c000 = w_lon1 * q1111 + w_lon2 * q1112
    c001 = w_lon1 * q1121 + w_lon2 * q1122
    c010 = w_lon1 * q1211 + w_lon2 * q1212
    c011 = w_lon1 * q1221 + w_lon2 * q1222
    c100 = w_lon1 * q2111 + w_lon2 * q2112
    c101 = w_lon1 * q2121 + w_lon2 * q2122
    c110 = w_lon1 * q2211 + w_lon2 * q2212
    c111 = w_lon1 * q2221 + w_lon2 * q2222

    # Interpolate along latitude (y-axis) for both time slices
    c00 = w_lat1 * c000 + w_lat2 * c001
    c01 = w_lat1 * c010 + w_lat2 * c011
    c10 = w_lat1 * c100 + w_lat2 * c101
    c11 = w_lat1 * c110 + w_lat2 * c111

    # Interpolate along level (z-axis) for both time slices
    c0 = w_level1 * c00 + w_level2 * c01
    c1 = w_level1 * c10 + w_level2 * c11

    # Interpolate along time (t-axis)
    interpolated_value = w_time1 * c0 + w_time2 * c1

    return interpolated_value


@numba.jit(nopython=True)
def _bilinear_interpolation_jit(
    lat: float,
    lon: float,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    data: np.ndarray,
) -> float:
    """
    Perform bilinear interpolation on a 2D grid.

    This function is JIT-compiled with Numba for performance.

    Parameters
    ----------
    lat : float
        Latitude of the interpolation point.
    lon : float
        Longitude of the interpolation point.
    grid_lat : np.ndarray
        A 1D array of latitude coordinates for the grid.
    grid_lon : np.ndarray
        A 1D array of longitude coordinates for the grid.
    data : np.ndarray
        A 2D array of data values corresponding to the grid.

    Returns
    -------
    float
        The interpolated value at the given lat/lon point.
    """
    # Find lower-bound indices for lat and lon
    i = np.searchsorted(grid_lat, lat) - 1
    j = np.searchsorted(grid_lon, lon) - 1

    # Calculate weights and clamped indices
    w_lat1, w_lat2, i1, i2 = _calculate_interpolation_weights_jit(lat, grid_lat, i)
    w_lon1, w_lon2, j1, j2 = _calculate_interpolation_weights_jit(lon, grid_lon, j)

    # Data values at the grid corners
    q11 = data[i1, j1]
    q21 = data[i2, j1]
    q12 = data[i1, j2]
    q22 = data[i2, j2]

    # Perform the interpolation
    f_lon1 = w_lat1 * q11 + w_lat2 * q21
    f_lon2 = w_lat1 * q12 + w_lat2 * q22
    interpolated_value = w_lon1 * f_lon1 + w_lon2 * f_lon2

    return interpolated_value


@numba.jit(nopython=True)
def _rk4_step_jit(
    lat: float,
    lon: float,
    level: float,
    time: float,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    grid_level: np.ndarray,
    grid_time: np.ndarray,
    u_data: np.ndarray,
    v_data: np.ndarray,
    w_data: np.ndarray,
    dt_seconds: float,
) -> tuple[float, float, float]:
    """
    Perform a single Runge-Kutta 4th order (RK4) integration step in 4D.

    This function is JIT-compiled with Numba for performance.

    Parameters
    ----------
    lat : float
        The current latitude of the particle.
    lon : float
        The current longitude of the particle.
    level : float
        The current vertical level of the particle.
    time : float
        The current time of the particle.
    grid_lat : np.ndarray
        The latitude coordinates of the velocity field grid.
    grid_lon : np.ndarray
        The longitude coordinates of the velocity field grid.
    grid_level : np.ndarray
        The vertical level coordinates of the velocity field grid.
    grid_time : np.ndarray
        The time coordinates of the velocity field grid.
    u_data : np.ndarray
        A 4D array of the 'u' velocity component.
    v_data : np.ndarray
        A 4D array of the 'v' velocity component.
    w_data : np.ndarray
        A 4D array of the 'w' velocity component.
    dt_seconds : float
        The time step for the integration in seconds.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing the new latitude, longitude, and level.
    """
    dt_hours = dt_seconds / 3600.0
    # --- RK4 k1 ---
    u1 = _quadrilinear_interpolation_jit(lat, lon, level, time, grid_lat, grid_lon, grid_level, grid_time, u_data)
    v1 = _quadrilinear_interpolation_jit(lat, lon, level, time, grid_lat, grid_lon, grid_level, grid_time, v_data)
    w1 = _quadrilinear_interpolation_jit(lat, lon, level, time, grid_lat, grid_lon, grid_level, grid_time, w_data)

    # --- RK4 k2 ---
    lat2 = lat + v1 * dt_hours / 2
    lon2 = lon + u1 * dt_hours / 2
    level2 = level + w1 * dt_hours / 2
    time2 = time + dt_seconds / 2
    u2 = _quadrilinear_interpolation_jit(lat2, lon2, level2, time2, grid_lat, grid_lon, grid_level, grid_time, u_data)
    v2 = _quadrilinear_interpolation_jit(lat2, lon2, level2, time2, grid_lat, grid_lon, grid_level, grid_time, v_data)
    w2 = _quadrilinear_interpolation_jit(lat2, lon2, level2, time2, grid_lat, grid_lon, grid_level, grid_time, w_data)

    # --- RK4 k3 ---
    lat3 = lat + v2 * dt_hours / 2
    lon3 = lon + u2 * dt_hours / 2
    level3 = level + w2 * dt_hours / 2
    time3 = time + dt_seconds / 2
    u3 = _quadrilinear_interpolation_jit(lat3, lon3, level3, time3, grid_lat, grid_lon, grid_level, grid_time, u_data)
    v3 = _quadrilinear_interpolation_jit(lat3, lon3, level3, time3, grid_lat, grid_lon, grid_level, grid_time, v_data)
    w3 = _quadrilinear_interpolation_jit(lat3, lon3, level3, time3, grid_lat, grid_lon, grid_level, grid_time, w_data)

    # --- RK4 k4 ---
    lat4 = lat + v3 * dt_hours
    lon4 = lon + u3 * dt_hours
    level4 = level + w3 * dt_hours
    time4 = time + dt_seconds
    u4 = _quadrilinear_interpolation_jit(lat4, lon4, level4, time4, grid_lat, grid_lon, grid_level, grid_time, u_data)
    v4 = _quadrilinear_interpolation_jit(lat4, lon4, level4, time4, grid_lat, grid_lon, grid_level, grid_time, v_data)
    w4 = _quadrilinear_interpolation_jit(lat4, lon4, level4, time4, grid_lat, grid_lon, grid_level, grid_time, w_data)

    # --- Final velocity and position update ---
    u_final = (u1 + 2 * u2 + 2 * u3 + u4) / 6.0
    v_final = (v1 + 2 * v2 + 2 * v3 + v4) / 6.0
    w_final = (w1 + 2 * w2 + 2 * w3 + w4) / 6.0

    new_lat = lat + v_final * dt_hours
    new_lon = lon + u_final * dt_hours
    new_level = level + w_final * dt_hours

    return new_lat, new_lon, new_level


@numba.jit(nopython=True, parallel=True)
def _integrate_jit(
    trajectory_lat: np.ndarray,
    trajectory_lon: np.ndarray,
    trajectory_level: np.ndarray,
    trajectory_time: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    grid_level: np.ndarray,
    grid_time: np.ndarray,
    u_data: np.ndarray,
    v_data: np.ndarray,
    w_data: np.ndarray,
    num_steps: int,
    dt_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform the core trajectory integration using a Numba JIT-compiled loop.
    This function is designed for performance-critical computation and operates
    exclusively on NumPy arrays. It uses the Runge-Kutta 4th order (RK4)
    method for improved physical accuracy. The outer loop over particles is
    parallelized.
    Parameters
    ----------
    trajectory_lat : np.ndarray
        A 2D array (particle, time) to be filled with particle latitudes.
    trajectory_lon : np.ndarray
        A 2D array (particle, time) to be filled with particle longitudes.
    trajectory_level : np.ndarray
        A 2D array (particle, time) to be filled with particle vertical levels.
    trajectory_time : np.ndarray
        A 2D array (particle, time) to be filled with particle times.
    grid_lat : np.ndarray
        The latitude coordinates of the velocity field grid.
    grid_lon : np.ndarray
        The longitude coordinates of the velocity field grid.
    grid_level : np.ndarray
        The vertical level coordinates of the velocity field grid.
    grid_time : np.ndarray
        The time coordinates of the velocity field grid.
    u_data : np.ndarray
        A 4D array of the 'u' velocity component.
    v_data : np.ndarray
        A 4D array of the 'v' velocity component.
    w_data : np.ndarray
        A 4D array of the 'w' velocity component.
    num_steps : int
        The number of integration steps to perform.
    dt_seconds : float
        The time step for the integration in seconds.
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the populated trajectory arrays for lat, lon, level, and time.
    """
    num_particles = trajectory_lat.shape[0]
    for p in numba.prange(num_particles):  # Parallel loop over particles
        for i in range(num_steps):
            new_lat, new_lon, new_level = _rk4_step_jit(
                trajectory_lat[p, i],
                trajectory_lon[p, i],
                trajectory_level[p, i],
                trajectory_time[p, i],
                grid_lat,
                grid_lon,
                grid_level,
                grid_time,
                u_data,
                v_data,
                w_data,
                dt_seconds,
            )
            trajectory_lat[p, i + 1] = new_lat
            trajectory_lon[p, i + 1] = new_lon
            trajectory_level[p, i + 1] = new_level
            trajectory_time[p, i + 1] = trajectory_time[p, i] + dt_seconds

    return trajectory_lat, trajectory_lon, trajectory_level, trajectory_time


def run_trajectory(
    starting_points: Dict[str, Union[np.ndarray, list, pd.Timestamp]],
    velocity_field: xr.Dataset,
    num_steps: int,
    dt: float = 1.0,
) -> xr.Dataset:
    """
    Simulate multiple particle trajectories through a 4D velocity field.
    This function integrates particle positions using the Runge-Kutta 4th
    order (RK4) method. The core integration loop is JIT-compiled with Numba
    and parallelized to handle multiple particles efficiently.
    Parameters
    ----------
    starting_points : Dict[str, Union[np.ndarray, list, pd.Timestamp]]
        A dictionary defining the initial positions of the particles.
        Must contain 'lat', 'lon', 'level', and 'time' keys.
    velocity_field : xr.Dataset
        An xarray Dataset containing the velocity components 'u', 'v', and 'w'.
        The dataset must have 'lat', 'lon', 'level', and 'time' as coordinates.
    num_steps : int
        The number of integration steps to perform.
    dt : float, optional
        The time step for the integration in hours (default is 1.0).
    Returns
    -------
    xr.Dataset
        A new xarray Dataset containing the trajectories of the particles.
        The dataset will have 'time' and 'particle' coordinates.
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import xarray as xr
    >>> # Create a sample velocity field (e.g., solid body rotation)
    >>> lat_grid = np.arange(-90, 91, 10)
    >>> lon_grid = np.arange(-180, 181, 20)
    >>> level_grid = np.arange(1000, 900, -50)
    >>> time_grid = pd.to_datetime(['2023-01-01T00:00', '2023-01-01T06:00'])
    >>> lon_rad, lat_rad = np.meshgrid(np.deg2rad(lon_grid), np.deg2rad(lat_grid))
    >>> u = -np.sin(lat_rad) * np.cos(lon_rad) * 10
    >>> v = np.sin(lon_rad) * 10
    >>> w = np.zeros_like(u)
    >>> # Create a 4D velocity field
    >>> u_4d = np.tile(u[np.newaxis, np.newaxis, :, :], (len(time_grid), len(level_grid), 1, 1))
    >>> v_4d = np.tile(v[np.newaxis, np.newaxis, :, :], (len(time_grid), len(level_grid), 1, 1))
    >>> w_4d = np.tile(w[np.newaxis, np.newaxis, :, :], (len(time_grid), len(level_grid), 1, 1))
    >>> velocity_field = xr.Dataset(
    ...     {
    ...         'u': (('time', 'level', 'lat', 'lon'), u_4d),
    ...         'v': (('time', 'level', 'lat', 'lon'), v_4d),
    ...         'w': (('time', 'level', 'lat', 'lon'), w_4d),
    ...     },
    ...     coords={'time': time_grid, 'lat': lat_grid, 'lon': lon_grid, 'level': level_grid}
    ... )
    >>> # Define starting points for multiple particles
    >>> start_points = {
    ...     'lat': [40.0, 45.0],
    ...     'lon': [-120.0, -115.0],
    ...     'level': [950, 950],
    ...     'time': pd.Timestamp('2023-01-01T01:00')
    ... }
    >>> trajectory_ds = run_trajectory(start_points, velocity_field, 10)
    >>> print(trajectory_ds)
    <xarray.Dataset>
    Dimensions:   (particle: 2, time: 11)
    Coordinates:
      * particle  (particle) int64 0 1
      * time      (time) datetime64[ns] 2023-01-01T01:00:00 ... 2023-01-01T11:00:00
    Data variables:
        lat       (particle, time) float64 40.0 39.86 39.73 ... 43.51 43.4
        lon       (particle, time) float64 -120.0 -119.5 -119.0 ... -111.3 -110.8
        level     (particle, time) float64 950.0 950.0 950.0 ... 950.0 950.0
    Attributes:
        history:  4D particle trajectory calculated for 2 particles using RK4 integ...
    """
    # --- Prepare data for Numba kernel ---
    start_lat = np.atleast_1d(starting_points['lat'])
    start_lon = np.atleast_1d(starting_points['lon'])
    start_level = np.atleast_1d(starting_points['level'])
    start_time = pd.to_datetime(np.atleast_1d(starting_points['time']))
    num_particles = len(start_lat)

    # Convert timestamps to float64 for Numba compatibility
    start_time_numeric = start_time.astype(np.int64) / 1e9
    time_step_seconds = pd.Timedelta(hours=dt).total_seconds()

    # Pre-allocate numpy arrays for performance
    trajectory_lat = np.zeros((num_particles, num_steps + 1), dtype=np.float64)
    trajectory_lon = np.zeros((num_particles, num_steps + 1), dtype=np.float64)
    trajectory_level = np.zeros((num_particles, num_steps + 1), dtype=np.float64)
    trajectory_time = np.zeros((num_particles, num_steps + 1), dtype=np.float64)

    trajectory_lat[:, 0] = start_lat
    trajectory_lon[:, 0] = start_lon
    trajectory_level[:, 0] = start_level
    trajectory_time[:, 0] = start_time_numeric

    # Extract NumPy arrays from the velocity field for the JIT function.
    grid_lat = velocity_field['lat'].values
    grid_lon = velocity_field['lon'].values
    grid_level = velocity_field['level'].values
    grid_time = velocity_field['time'].values.astype('datetime64[s]').astype(np.float64)
    u_data = velocity_field['u'].values
    v_data = velocity_field['v'].values
    w_data = velocity_field['w'].values

    # --- Run the JIT-compiled integration ---
    traj_lat, traj_lon, traj_level, traj_time = _integrate_jit(
        trajectory_lat,
        trajectory_lon,
        trajectory_level,
        trajectory_time,
        grid_lat,
        grid_lon,
        grid_level,
        grid_time,
        u_data,
        v_data,
        w_data,
        num_steps,
        time_step_seconds,
    )

    # --- Package the results into an xarray Dataset ---
    # The time coordinate is the same for all particles, so we can take the first row.
    time_coords = pd.to_datetime(traj_time[0, :], unit='s')

    trajectory_ds = xr.Dataset(
        {
            'lat': (('particle', 'time'), traj_lat),
            'lon': (('particle', 'time'), traj_lon),
            'level': (('particle', 'time'), traj_level),
        },
        coords={
            'particle': range(num_particles),
            'time': time_coords,
        },
    )

    # --- Scientific Hygiene: Update Attributes ---
    history_log = (
        f"4D particle trajectory calculated for {num_particles} particles using RK4 "
        f"integration with {num_steps} steps and dt={dt} hours."
    )
    trajectory_ds.attrs['history'] = history_log

    return trajectory_ds
