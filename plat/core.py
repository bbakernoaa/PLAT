
"""Core atmospheric trajectory model."""

from typing import Dict, Union

import numba
import numpy as np
import xarray as xr


@numba.jit(nopython=True)
def _trilinear_interpolation_jit(
    lat: float,
    lon: float,
    level: float,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    grid_level: np.ndarray,
    data: np.ndarray,
) -> float:
    """
    Perform trilinear interpolation on a 3D grid.

    This function is JIT-compiled with Numba for performance.

    Parameters
    ----------
    lat : float
        Latitude of the interpolation point.
    lon : float
        Longitude of the interpolation point.
    level : float
        Vertical level of the interpolation point.
    grid_lat : np.ndarray
        A 1D array of latitude coordinates for the grid.
    grid_lon : np.ndarray
        A 1D array of longitude coordinates for the grid.
    grid_level : np.ndarray
        A 1D array of vertical level coordinates for the grid.
    data : np.ndarray
        A 3D array of data values corresponding to the grid. The expected
        dimension order is (level, lat, lon).

    Returns
    -------
    float
        The interpolated value at the given lat/lon/level point.
    """
    # Find lower-bound indices for lat, lon, and level
    i = np.searchsorted(grid_lat, lat) - 1
    j = np.searchsorted(grid_lon, lon) - 1
    k = np.searchsorted(grid_level, level) - 1

    # Clamp indices to be within bounds
    i = max(0, min(i, len(grid_lat) - 2))
    j = max(0, min(j, len(grid_lon) - 2))
    k = max(0, min(k, len(grid_level) - 2))

    # Grid points surrounding the current location
    lat1, lat2 = grid_lat[i], grid_lat[i + 1]
    lon1, lon2 = grid_lon[j], grid_lon[j + 1]
    level1, level2 = grid_level[k], grid_level[k + 1]

    # Data values at the 8 corners of the cube
    q111 = data[k, i, j]
    q112 = data[k, i, j + 1]
    q121 = data[k, i + 1, j]
    q122 = data[k, i + 1, j + 1]
    q211 = data[k + 1, i, j]
    q212 = data[k + 1, i, j + 1]
    q221 = data[k + 1, i + 1, j]
    q222 = data[k + 1, i + 1, j + 1]

    # Calculate interpolation weights, avoiding division by zero
    d_lat = lat2 - lat1 if (lat2 - lat1) != 0 else 1
    d_lon = lon2 - lon1 if (lon2 - lon1) != 0 else 1
    d_level = level2 - level1 if (level2 - level1) != 0 else 1

    w_lat1 = (lat2 - lat) / d_lat
    w_lat2 = (lat - lat1) / d_lat
    w_lon1 = (lon2 - lon) / d_lon
    w_lon2 = (lon - lon1) / d_lon
    w_level1 = (level2 - level) / d_level
    w_level2 = (level - level1) / d_level

    # Interpolate along the longitude axis (x-axis)
    c00 = w_lon1 * q111 + w_lon2 * q112
    c01 = w_lon1 * q121 + w_lon2 * q122
    c10 = w_lon1 * q211 + w_lon2 * q212
    c11 = w_lon1 * q221 + w_lon2 * q222

    # Interpolate along the latitude axis (y-axis)
    c0 = w_lat1 * c00 + w_lat2 * c01
    c1 = w_lat1 * c10 + w_lat2 * c11

    # Interpolate along the level axis (z-axis)
    interpolated_value = w_level1 * c0 + w_level2 * c1

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

    # Clamp indices to be within bounds
    i = max(0, min(i, len(grid_lat) - 2))
    j = max(0, min(j, len(grid_lon) - 2))

    # Grid points surrounding the current location
    lat1, lat2 = grid_lat[i], grid_lat[i + 1]
    lon1, lon2 = grid_lon[j], grid_lon[j + 1]

    # Data values at the grid corners
    q11 = data[i, j]
    q21 = data[i + 1, j]
    q12 = data[i, j + 1]
    q22 = data[i + 1, j + 1]

    # Calculate interpolation weights
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1

    # Avoid division by zero if grid is singular
    if d_lat == 0 or d_lon == 0:
        return q11

    w1 = (lat2 - lat) / d_lat
    w2 = (lat - lat1) / d_lat
    w3 = (lon2 - lon) / d_lon
    w4 = (lon - lon1) / d_lon

    # Perform the interpolation
    f_lon1 = w1 * q11 + w2 * q21
    f_lon2 = w1 * q12 + w2 * q22
    interpolated_value = w3 * f_lon1 + w4 * f_lon2

    return interpolated_value


@numba.jit(nopython=True)
def _rk4_step_jit(
    lat: float,
    lon: float,
    level: float,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    grid_level: np.ndarray,
    u_data: np.ndarray,
    v_data: np.ndarray,
    w_data: np.ndarray,
    dt: float,
) -> tuple[float, float, float]:
    """
    Perform a single Runge-Kutta 4th order (RK4) integration step in 3D.

    This function is JIT-compiled with Numba for performance.

    Parameters
    ----------
    lat : float
        The current latitude of the particle.
    lon : float
        The current longitude of the particle.
    level : float
        The current vertical level of the particle.
    grid_lat : np.ndarray
        The latitude coordinates of the velocity field grid.
    grid_lon : np.ndarray
        The longitude coordinates of the velocity field grid.
    grid_level : np.ndarray
        The vertical level coordinates of the velocity field grid.
    u_data : np.ndarray
        A 3D array of the 'u' velocity component.
    v_data : np.ndarray
        A 3D array of the 'v' velocity component.
    w_data : np.ndarray
        A 3D array of the 'w' velocity component.
    dt : float
        The time step for the integration.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing the new latitude, longitude, and level.
    """
    # --- RK4 k1 ---
    u1 = _trilinear_interpolation_jit(lat, lon, level, grid_lat, grid_lon, grid_level, u_data)
    v1 = _trilinear_interpolation_jit(lat, lon, level, grid_lat, grid_lon, grid_level, v_data)
    w1 = _trilinear_interpolation_jit(lat, lon, level, grid_lat, grid_lon, grid_level, w_data)

    # --- RK4 k2 ---
    lat2 = lat + v1 * dt / 2
    lon2 = lon + u1 * dt / 2
    level2 = level + w1 * dt / 2
    u2 = _trilinear_interpolation_jit(lat2, lon2, level2, grid_lat, grid_lon, grid_level, u_data)
    v2 = _trilinear_interpolation_jit(lat2, lon2, level2, grid_lat, grid_lon, grid_level, v_data)
    w2 = _trilinear_interpolation_jit(lat2, lon2, level2, grid_lat, grid_lon, grid_level, w_data)

    # --- RK4 k3 ---
    lat3 = lat + v2 * dt / 2
    lon3 = lon + u2 * dt / 2
    level3 = level + w2 * dt / 2
    u3 = _trilinear_interpolation_jit(lat3, lon3, level3, grid_lat, grid_lon, grid_level, u_data)
    v3 = _trilinear_interpolation_jit(lat3, lon3, level3, grid_lat, grid_lon, grid_level, v_data)
    w3 = _trilinear_interpolation_jit(lat3, lon3, level3, grid_lat, grid_lon, grid_level, w_data)

    # --- RK4 k4 ---
    lat4 = lat + v3 * dt
    lon4 = lon + u3 * dt
    level4 = level + w3 * dt
    u4 = _trilinear_interpolation_jit(lat4, lon4, level4, grid_lat, grid_lon, grid_level, u_data)
    v4 = _trilinear_interpolation_jit(lat4, lon4, level4, grid_lat, grid_lon, grid_level, v_data)
    w4 = _trilinear_interpolation_jit(lat4, lon4, level4, grid_lat, grid_lon, grid_level, w_data)

    # --- Final velocity and position update ---
    u_final = (u1 + 2 * u2 + 2 * u3 + u4) / 6
    v_final = (v1 + 2 * v2 + 2 * v3 + v4) / 6
    w_final = (w1 + 2 * w2 + 2 * w3 + w4) / 6

    new_lat = lat + v_final * dt
    new_lon = lon + u_final * dt
    new_level = level + w_final * dt

    return new_lat, new_lon, new_level


@numba.jit(nopython=True, parallel=True)
def _integrate_jit(
    trajectory_lat: np.ndarray,
    trajectory_lon: np.ndarray,
    trajectory_level: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    grid_level: np.ndarray,
    u_data: np.ndarray,
    v_data: np.ndarray,
    w_data: np.ndarray,
    num_steps: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    grid_lat : np.ndarray
        The latitude coordinates of the velocity field grid.
    grid_lon : np.ndarray
        The longitude coordinates of the velocity field grid.
    grid_level : np.ndarray
        The vertical level coordinates of the velocity field grid.
    u_data : np.ndarray
        A 3D array of the 'u' velocity component.
    v_data : np.ndarray
        A 3D array of the 'v' velocity component.
    w_data : np.ndarray
        A 3D array of the 'w' velocity component.
    num_steps : int
        The number of integration steps to perform.
    dt : float
        The time step for the integration.
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the populated trajectory arrays for lat, lon, and level.
    """
    num_particles = trajectory_lat.shape[0]
    for p in numba.prange(num_particles):  # Parallel loop over particles
        for i in range(num_steps):
            new_lat, new_lon, new_level = _rk4_step_jit(
                trajectory_lat[p, i],
                trajectory_lon[p, i],
                trajectory_level[p, i],
                grid_lat,
                grid_lon,
                grid_level,
                u_data,
                v_data,
                w_data,
                dt,
            )
            trajectory_lat[p, i + 1] = new_lat
            trajectory_lon[p, i + 1] = new_lon
            trajectory_level[p, i + 1] = new_level

    return trajectory_lat, trajectory_lon, trajectory_level


def run_trajectory(
    starting_points: Dict[str, Union[np.ndarray, list]],
    velocity_field: xr.Dataset,
    num_steps: int,
    dt: float = 1.0,
) -> xr.Dataset:
    """
    Simulate multiple particle trajectories through a 3D velocity field.
    This function integrates particle positions using the Runge-Kutta 4th
    order (RK4) method. The core integration loop is JIT-compiled with Numba
    and parallelized to handle multiple particles efficiently.
    Parameters
    ----------
    starting_points : Dict[str, Union[np.ndarray, list]]
        A dictionary defining the initial positions of the particles.
        Must contain 'lat', 'lon', and 'level' keys with lists or NumPy
        arrays of the same length.
    velocity_field : xr.Dataset
        An xarray Dataset containing the velocity components 'u', 'v', and 'w'.
        The dataset must have 'lat', 'lon', and 'level' as coordinates.
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
    >>> import xarray as xr
    >>> # Create a sample velocity field (e.g., solid body rotation)
    >>> lat_grid = np.arange(-90, 91, 10)
    >>> lon_grid = np.arange(-180, 181, 20)
    >>> level_grid = np.arange(1000, 900, -50)
    >>> lon_rad, lat_rad = np.meshgrid(np.deg2rad(lon_grid), np.deg2rad(lat_grid))
    >>> u = -np.sin(lat_rad) * np.cos(lon_rad) * 10
    >>> v = np.sin(lon_rad) * 10
    >>> w = np.zeros_like(u)
    >>> # Create a 3D velocity field
    >>> u_3d = np.tile(u[np.newaxis, :, :], (len(level_grid), 1, 1))
    >>> v_3d = np.tile(v[np.newaxis, :, :], (len(level_grid), 1, 1))
    >>> w_3d = np.tile(w[np.newaxis, :, :], (len(level_grid), 1, 1))
    >>> velocity_field = xr.Dataset(
    ...     {
    ...         'u': (('level', 'lat', 'lon'), u_3d),
    ...         'v': (('level', 'lat', 'lon'), v_3d),
    ...         'w': (('level', 'lat', 'lon'), w_3d),
    ...     },
    ...     coords={'lat': lat_grid, 'lon': lon_grid, 'level': level_grid}
    ... )
    >>> # Define starting points for multiple particles
    >>> start_points = {'lat': [40.0, 45.0], 'lon': [-120.0, -115.0], 'level': [950, 950]}
    >>> trajectory_ds = run_trajectory(start_points, velocity_field, 10)
    >>> print(trajectory_ds)
    <xarray.Dataset>
    Dimensions:   (time: 11, particle: 2)
    Coordinates:
      * time      (time) int64 0 1 2 3 4 5 6 7 8 9 10
      * particle  (particle) int64 0 1
    Data variables:
        lat       (particle, time) float64 40.0 39.86 39.73 ... 43.51 43.4
        lon       (particle, time) float64 -120.0 -119.5 -119.0 ... -111.3 -110.8
        level     (particle, time) float64 950.0 950.0 950.0 ... 950.0 950.0
    Attributes:
        history:  3D particle trajectory calculated using RK4 integration with 10 ...
    """
    # --- Prepare data for Numba kernel ---
    start_lat = np.atleast_1d(starting_points['lat'])
    start_lon = np.atleast_1d(starting_points['lon'])
    start_level = np.atleast_1d(starting_points['level'])
    num_particles = len(start_lat)

    # Pre-allocate numpy arrays for performance
    trajectory_lat = np.zeros((num_particles, num_steps + 1), dtype=np.float64)
    trajectory_lon = np.zeros((num_particles, num_steps + 1), dtype=np.float64)
    trajectory_level = np.zeros((num_particles, num_steps + 1), dtype=np.float64)

    trajectory_lat[:, 0] = start_lat
    trajectory_lon[:, 0] = start_lon
    trajectory_level[:, 0] = start_level

    # Extract NumPy arrays from the velocity field for the JIT function.
    grid_lat = velocity_field['lat'].values
    grid_lon = velocity_field['lon'].values
    grid_level = velocity_field['level'].values
    u_data = velocity_field['u'].values
    v_data = velocity_field['v'].values
    w_data = velocity_field['w'].values

    # --- Run the JIT-compiled integration ---
    trajectory_lat, trajectory_lon, trajectory_level = _integrate_jit(
        trajectory_lat,
        trajectory_lon,
        trajectory_level,
        grid_lat,
        grid_lon,
        grid_level,
        u_data,
        v_data,
        w_data,
        num_steps,
        dt,
    )

    # --- Package the results into an xarray Dataset ---
    trajectory_ds = xr.Dataset(
        {
            'lat': (('particle', 'time'), trajectory_lat),
            'lon': (('particle', 'time'), trajectory_lon),
            'level': (('particle', 'time'), trajectory_level),
        },
        coords={
            'time': range(num_steps + 1),
            'particle': range(num_particles),
        },
    )

    # --- Scientific Hygiene: Update Attributes ---
    history_log = (
        f"3D particle trajectory calculated for {num_particles} particles using RK4 "
        f"integration with {num_steps} steps and dt={dt}."
    )
    trajectory_ds.attrs['history'] = history_log

    return trajectory_ds
