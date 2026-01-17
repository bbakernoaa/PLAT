
"""Plotting utilities for PLAT trajectories."""

from typing import Optional

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import xarray as xr


def plot_trajectories(
    trajectory_ds: xr.Dataset,
    projection: Optional[ccrs.Projection] = None,
    transform: ccrs.Projection = ccrs.PlateCarree(),
) -> Figure:
    """
    Plot particle trajectories on a map.

    This function generates a publication-quality plot of trajectories using
    Matplotlib and Cartopy. It plots each particle's path, marking the start
    (green circle) and end (red 'x') points.

    Parameters
    ----------
    trajectory_ds : xr.Dataset
        An xarray Dataset containing the trajectory data, as produced by
        `plat.core.run_trajectory`. It must contain 'lon' and 'lat'
        data variables with 'particle' and 'step' dimensions.
    projection : Optional[ccrs.Projection], optional
        The Cartopy projection for the map axes. If None, defaults to
        ccrs.PlateCarree().
    transform : ccrs.Projection, optional
        The projection of the input data coordinates, by default ccrs.PlateCarree().
        This is the standard for lat/lon data.

    Returns
    -------
    matplotlib.figure.Figure
        The Matplotlib figure object containing the plot.

    Examples
    --------
    .. code-block:: python

        # This is a conceptual example. A runnable script will be
        # created in the examples/ directory.
        import numpy as np
        import pandas as pd
        import xarray as xr
        from plat.core import run_trajectory
        from plat.plotting import plot_trajectories

        # 1. Create a sample velocity field
        lat = np.arange(20, 61, 10)
        lon = np.arange(-130, -79, 10)
        level = [1000, 950]
        time = pd.to_datetime(['2023-01-01T00:00', '2023-01-01T06:00'])
        u = np.full((2, 2, 5, 6), 10)  # Constant eastward wind
        v = np.full((2, 2, 5, 6), 0)   # No north-south wind
        w = np.full((2, 2, 5, 6), 0)   # No vertical motion
        velocity_field = xr.Dataset(
            {'u': (('time', 'level', 'lat', 'lon'), u),
             'v': (('time', 'level', 'lat', 'lon'), v),
             'w': (('time', 'level', 'lat', 'lon'), w)},
            coords={'time': time, 'level': level, 'lat': lat, 'lon': lon}
        )

        # 2. Define starting points
        start_points = {
            'lat': [40.0, 50.0],
            'lon': [-120.0, -110.0],
            'level': [950, 950],
            'time': pd.Timestamp('2023-01-01T01:00')
        }

        # 3. Run the trajectory model
        trajectory_ds = run_trajectory(start_points, velocity_field, num_steps=5, dt=1.0)

        # 4. Plot the results
        fig = plot_trajectories(trajectory_ds)
        # fig.savefig("trajectory_plot.png")
    """
    if projection is None:
        projection = ccrs.PlateCarree()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    ax.set_title("Particle Trajectories")
    ax.coastlines()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Iterate over each particle and plot its trajectory
    for particle in trajectory_ds.particle:
        lons = trajectory_ds['lon'].sel(particle=particle)
        lats = trajectory_ds['lat'].sel(particle=particle)

        # Plot the trajectory line
        ax.plot(lons, lats, transform=transform, label=f"Particle {particle.item()}")
        # Mark the start point
        ax.plot(
            lons.isel(step=0),
            lats.isel(step=0),
            "go",
            markersize=7,
            transform=transform,
            label=f"Start {particle.item()}",
        )
        # Mark the end point (ensure we select the last valid point)
        last_step = lons.dropna(dim="step").step[-1]
        ax.plot(
            lons.sel(step=last_step),
            lats.sel(step=last_step),
            "rx",
            markersize=9,
            transform=transform,
            label=f"End {particle.item()}",
        )

    ax.legend()
    # Close the plot to prevent automatic display in non-interactive environments
    # The figure object is returned for further manipulation or saving.
    plt.close(fig)
    return fig
