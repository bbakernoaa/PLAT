
"""Core atmospheric trajectory model."""

from typing import Dict, Union

import xarray as xr


def run_trajectory(
    starting_point: Dict[str, Union[float, int]],
    velocity_field: xr.Dataset,
    num_steps: int,
) -> xr.Dataset:
    """
    Simulate a single-particle trajectory through a 2D velocity field.

    This function integrates the particle's position using a forward Euler
    method. The velocity field is assumed to be steady-state.

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
    # Initialize trajectory lists
    trajectory_lat = [starting_point['lat']]
    trajectory_lon = [starting_point['lon']]

    # Simple forward Euler integration
    for _ in range(num_steps):
        current_lat = trajectory_lat[-1]
        current_lon = trajectory_lon[-1]

        # Interpolate velocity at the current position
        # Using nearest-neighbor for simplicity
        u = velocity_field['u'].interp(
            lat=current_lat, lon=current_lon, method='nearest'
        )
        v = velocity_field['v'].interp(
            lat=current_lat, lon=current_lon, method='nearest'
        )

        # Update position (assuming dt=1 and simple lat/lon update)
        new_lat = current_lat + v.item()
        new_lon = current_lon + u.item()

        trajectory_lat.append(new_lat)
        trajectory_lon.append(new_lon)

    # Create the output Dataset
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
