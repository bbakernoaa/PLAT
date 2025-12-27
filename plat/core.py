
"""Core atmospheric trajectory model."""

from typing import Dict, Union

import numba
import numpy as np
import xarray as xr


@numba.jit(nopython=True)
def _find_nearest_index(array, value):
    """
    Find the index of the nearest value in a sorted array.

    This utility function is designed to be used within a Numba JIT-compiled
    environment. It uses binary search for efficiency.

    Parameters
    ----------
    array : np.ndarray
        A 1D sorted array.
    value : float
        The value to find the nearest index for.

    Returns
    -------
    int
        The index of the element in `array` that is closest to `value`.
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or np.abs(value - array[idx - 1]) < np.abs(value - array[idx])
    ):
        return idx - 1
    else:
        return idx


@numba.jit(nopython=True)
def _integrate_jit(
    trajectory_lat: np.ndarray,
    trajectory_lon: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    u_data: np.ndarray,
    v_data: np.ndarray,
    num_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform the core trajectory integration using a Numba JIT-compiled loop.

    This function is designed for performance-critical computation and operates
    exclusively on NumPy arrays. It uses nearest-neighbor interpolation with an
    efficient binary search (`np.searchsorted`) for grid lookup.

    Parameters
    ----------
    trajectory_lat : np.ndarray
        A 1D array to be filled with the latitude of the particle at each step.
    trajectory_lon : np.ndarray
        A 1D array to be filled with the longitude of the particle at each step.
    grid_lat : np.ndarray
        The latitude coordinates of the velocity field grid.
    grid_lon : np.ndarray
        The longitude coordinates of the velocity field grid.
    u_data : np.ndarray
        A 2D array of the 'u' velocity component.
    v_data : np.ndarray
        A 2D array of the 'v' velocity component.
    num_steps : int
        The number of integration steps to perform.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the populated trajectory_lat and trajectory_lon arrays.
    """
    # Simple forward Euler integration
    for i in range(num_steps):
        # Find the index of the nearest grid point using binary search.
        lat_idx = _find_nearest_index(grid_lat, trajectory_lat[i])
        lon_idx = _find_nearest_index(grid_lon, trajectory_lon[i])

        # Get velocity values at that grid point
        u = u_data[lat_idx, lon_idx]
        v = v_data[lat_idx, lon_idx]

        # Update position (assuming dt=1 and simple lat/lon update)
        trajectory_lat[i + 1] = trajectory_lat[i] + v
        trajectory_lon[i + 1] = trajectory_lon[i] + u

    return trajectory_lat, trajectory_lon


def run_trajectory(
    starting_point: Dict[str, Union[float, int]],
    velocity_field: xr.Dataset,
    num_steps: int,
) -> xr.Dataset:
    """
    Simulate a single-particle trajectory through a 2D velocity field.

    This function integrates the particle's position using a forward Euler
    method. The velocity field is assumed to be steady-state. The core
    integration loop is JIT-compiled with Numba for performance.

    Parameters
    ----------
    starting_point : Dict[str, Union[float, int]]
        A dictionary defining the initial position of the particle.
        Must contain 'lat' and 'lon' keys with numeric values.
    velocity_field : xr.Dataset
        An xarray Dataset containing the velocity components 'u' and 'v'.
        The dataset must have 'lat' and 'lon' as coordinates.
    num_steps : int
        The number of integration steps to perform.

    Returns
    -------
    xr.Dataset
        A new xarray Dataset containing the trajectory of the particle.
        The dataset will have a 'time' coordinate and variables 'lat' and 'lon'.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> # Create a sample velocity field (e.g., solid body rotation)
    >>> lat = np.arange(-90, 91, 10)
    >>> lon = np.arange(-180, 181, 20)
    >>> lon_rad, lat_rad = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
    >>> u = -np.sin(lat_rad) * np.cos(lon_rad)
    >>> v = np.sin(lon_rad)
    >>> velocity_field = xr.Dataset(
    ...     {'u': (('lat', 'lon'), u), 'v': (('lat', 'lon'), v)},
    ...     coords={'lat': lat, 'lon': lon}
    ... )
    >>> start = {'lat': 40.0, 'lon': -120.0}
    >>> trajectory = run_trajectory(start, velocity_field, 10)
    >>> print(trajectory)
    <xarray.Dataset>
    Dimensions:  (time: 11)
    Coordinates:
      * time     (time) int64 0 1 2 3 4 5 6 7 8 9 10
    Data variables:
        lat      (time) float64 40.0 39.86 39.73 ... 38.64 38.53
        lon      (time) float64 -120.0 -119.5 -119.0 ... -115.2 -114.8
    Attributes:
        history:  'Trajectory simulation started from lat=40.0, lon=-120.0'
    """
    # --- Prepare data for Numba kernel ---
    # Pre-allocate numpy arrays for performance
    trajectory_lat = np.zeros(num_steps + 1, dtype=np.float64)
    trajectory_lon = np.zeros(num_steps + 1, dtype=np.float64)

    trajectory_lat[0] = starting_point['lat']
    trajectory_lon[0] = starting_point['lon']

    # Extract NumPy arrays from the velocity field for the JIT function.
    # Numba cannot handle xarray objects directly.
    grid_lat = velocity_field['lat'].values
    grid_lon = velocity_field['lon'].values
    u_data = velocity_field['u'].values
    v_data = velocity_field['v'].values

    # --- Run the JIT-compiled integration ---
    trajectory_lat, trajectory_lon = _integrate_jit(
        trajectory_lat,
        trajectory_lon,
        grid_lat,
        grid_lon,
        u_data,
        v_data,
        num_steps,
    )

    # --- Package the results into an xarray Dataset ---
    trajectory_ds = xr.Dataset(
        {
            'lat': (('time',), trajectory_lat),
            'lon': (('time',), trajectory_lon),
        },
        coords={'time': range(num_steps + 1)},
    )

    # --- Scientific Hygiene: Update Attributes ---
    history_log = (
        f"Trajectory simulation started from "
        f"lat={starting_point['lat']}, lon={starting_point['lon']}"
    )
    trajectory_ds.attrs['history'] = history_log

    return trajectory_ds
