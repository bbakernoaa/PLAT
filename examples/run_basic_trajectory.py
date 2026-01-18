"""
=========================
Run a Basic Trajectory
=========================

This example demonstrates the core end-to-end workflow of the PLAT model.

It covers three main steps:
1.  **Creating a Velocity Field**: A synthetic, 4D velocity field is
    generated using NumPy and xarray. This represents a simplified
    "solid-body rotation" wind field.
2.  **Running the Trajectory Model**: The `plat.core.run_trajectory`
    function is called with a set of starting points to simulate the
    paths of two particles.
3.  **Visualizing the Output**: The resulting trajectory dataset is passed
    to the `plat.plotting.plot_trajectories` function to generate a
    publication-quality map, which is then saved to a file.
"""

import numpy as np
import pandas as pd
import xarray as xr

# Import the core model and plotting functions
from plat.core import run_trajectory
from plat.plotting import plot_trajectories


def create_synthetic_velocity_field() -> xr.Dataset:
    """
    Creates a synthetic velocity field representing solid body rotation.

    This is a classic meteorological test case where wind vectors circulate
    around the poles. This helps verify that the trajectory model correctly
    advects particles along curved paths.

    Returns
    -------
    xr.Dataset
        A 4D xarray Dataset containing 'u', 'v', and 'w' velocity
        components on a lat/lon/level/time grid.
    """
    lat_grid = np.arange(-90, 91, 10)
    lon_grid = np.arange(-180, 181, 20)
    level_grid = np.arange(1000, 900, -50)
    time_grid = pd.to_datetime(["2023-01-01T00:00", "2023-01-01T06:00"])

    # Create a meshgrid for vectorized calculations
    lon_rad, lat_rad = np.meshgrid(np.deg2rad(lon_grid), np.deg2rad(lat_grid))

    # Define wind components for solid body rotation
    u = -np.sin(lat_rad) * np.cos(lon_rad) * 10  # Zonal wind
    v = np.sin(lon_rad) * 10  # Meridional wind
    w = np.zeros_like(u)  # No vertical motion

    # Tile these 2D fields to create a 4D dataset
    u_4d = np.tile(
        u[np.newaxis, np.newaxis, :, :], (len(time_grid), len(level_grid), 1, 1)
    )
    v_4d = np.tile(
        v[np.newaxis, np.newaxis, :, :], (len(time_grid), len(level_grid), 1, 1)
    )
    w_4d = np.tile(
        w[np.newaxis, np.newaxis, :, :], (len(time_grid), len(level_grid), 1, 1)
    )

    velocity_field = xr.Dataset(
        {
            "u": (("time", "level", "lat", "lon"), u_4d),
            "v": (("time", "level", "lat", "lon"), v_4d),
            "w": (("time", "level", "lat", "lon"), w_4d),
        },
        coords={
            "time": time_grid,
            "lat": lat_grid,
            "lon": lon_grid,
            "level": level_grid,
        },
    )
    velocity_field.attrs["history"] = "Created synthetic solid-body rotation field."
    return velocity_field


def main() -> None:
    """
    Main function to run the trajectory simulation and plotting.
    """
    # --- 1. Create a sample velocity field ---
    print("Creating synthetic velocity field...")
    velocity_field = create_synthetic_velocity_field()

    # --- 2. Define starting points for multiple particles ---
    start_points = {
        "lat": [40.0, 45.0],
        "lon": [-120.0, -115.0],
        "level": [950, 950],
        "time": pd.Timestamp("2023-01-01T01:00"),
    }

    # --- 3. Run the trajectory model ---
    print("Running trajectory model...")
    trajectory_ds = run_trajectory(
        starting_points=start_points,
        velocity_field=velocity_field,
        num_steps=10,
        dt=1.0,  # Time step in hours
    )

    print("\nTrajectory Simulation Complete:")
    print(trajectory_ds)

    # --- 4. Plot the results ---
    print("\nGenerating trajectory plot...")
    fig = plot_trajectories(trajectory_ds)

    # Save the figure
    output_filename = "trajectory_plot.png"
    fig.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved to '{output_filename}'")


if __name__ == "__main__":
    main()
