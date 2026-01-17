
"""
Basic Trajectory Plotting Example

This script demonstrates a complete, end-to-end workflow for the PLAT model:
1.  **Set up the Environment**: Import necessary libraries.
2.  **Create a Velocity Field**: Generate a synthetic velocity field representing
    a simple, steady eastward wind. This mimics the kind of data that would
    typically be loaded from a GRIB or NetCDF file.
3.  **Define Starting Points**: Specify the initial conditions (latitude,
    longitude, level, and time) for two particles.
4.  **Run the Trajectory Model**: Use `plat.core.run_trajectory` to integrate
    the particle paths through the velocity field.
5.  **Visualize the Results**: Use the new `plat.plotting.plot_trajectories`
    function to create a publication-quality map of the trajectories.
6.  **Save the Output**: Save the resulting plot to a file.
"""

import numpy as np
import pandas as pd
import xarray as xr

from plat.core import run_trajectory
from plat.plotting import plot_trajectories


def main() -> None:
    """Main function to run the trajectory simulation and plotting."""
    print("--- 1. Creating a synthetic velocity field ---")
    # Define the grid for the velocity field
    lat_grid = np.arange(30, 61, 10)  # Latitudes from 30N to 60N
    lon_grid = np.arange(-130, -69, 10)  # Longitudes from 130W to 70W
    level_grid = np.array([1000, 950, 900])  # Pressure levels in hPa
    time_grid = pd.to_datetime(
        ["2023-01-01T00:00", "2023-01-01T06:00", "2023-01-01T12:00"]
    )

    # Create a steady eastward wind (u=10 m/s, v=0 m/s, w=0 hPa/hr)
    # The shape of the data arrays must match the coordinates
    u_data = np.full(
        (len(time_grid), len(level_grid), len(lat_grid), len(lon_grid)), 10.0
    )
    v_data = np.zeros_like(u_data)
    w_data = np.zeros_like(u_data)

    velocity_field = xr.Dataset(
        {
            "u": (("time", "level", "lat", "lon"), u_data),
            "v": (("time", "level", "lat", "lon"), v_data),
            "w": (("time", "level", "lat", "lon"), w_data),
        },
        coords={
            "time": time_grid,
            "level": level_grid,
            "lat": lat_grid,
            "lon": lon_grid,
        },
    )
    print(velocity_field)

    print("\n--- 2. Defining starting points for trajectories ---")
    start_points = {
        "lat": [40.0, 45.0],
        "lon": [-120.0, -115.0],
        "level": [950, 950],
        "time": pd.Timestamp("2023-01-01T01:00"),
    }
    print(start_points)

    print("\n--- 3. Running the trajectory model ---")
    # Run for 10 steps with a 1-hour time step
    trajectory_ds = run_trajectory(start_points, velocity_field, num_steps=10, dt=1.0)
    print(trajectory_ds)

    print("\n--- 4. Visualizing the trajectory results ---")
    fig = plot_trajectories(trajectory_ds)

    output_filename = "examples/basic_trajectory_plot.png"
    fig.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"\n--- 5. Saved plot to '{output_filename}' ---")


if __name__ == "__main__":
    main()
