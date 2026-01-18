
"""Tests for plotting utilities."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib.figure import Figure

from plat.plotting import plot_trajectories


@pytest.fixture
def sample_trajectory_dataset() -> xr.Dataset:
    """
    Create a sample trajectory dataset for testing.

    This fixture generates a simple xarray.Dataset with two particles, each
    with a short, linear trajectory. This mimics the output of
    `plat.core.run_trajectory` and serves as a consistent input for
    testing plotting functions.

    Returns
    -------
    xr.Dataset
        A sample trajectory dataset.
    """
    num_particles = 2
    num_steps = 6  # 5 steps + initial position
    base_time = pd.to_datetime("2023-01-01T00:00:00")

    # Create coordinates
    particle_coord = np.arange(num_particles)
    step_coord = np.arange(num_steps)
    time_coord = np.array(
        [base_time + pd.to_timedelta(i, unit="h") for i in range(num_steps)]
    )
    time_coord_2d = np.tile(time_coord, (num_particles, 1))

    # Create data variables
    # Particle 1: Moves east
    lat1 = np.full(num_steps, 40.0)
    lon1 = np.linspace(-120, -115, num_steps)
    # Particle 2: Moves north-east
    lat2 = np.linspace(45, 47, num_steps)
    lon2 = np.linspace(-110, -108, num_steps)

    # Add a NaN to simulate a particle exiting the domain
    lat2[4:] = np.nan
    lon2[4:] = np.nan

    lat_data = np.vstack([lat1, lat2])
    lon_data = np.vstack([lon1, lon2])
    level_data = np.full((num_particles, num_steps), 950.0)

    trajectory_ds = xr.Dataset(
        {
            "lat": (("particle", "step"), lat_data),
            "lon": (("particle", "step"), lon_data),
            "level": (("particle", "step"), level_data),
        },
        coords={
            "particle": particle_coord,
            "step": step_coord,
            "time": (("particle", "step"), time_coord_2d),
        },
    )
    return trajectory_ds


def test_plot_trajectories_returns_figure(
    sample_trajectory_dataset: xr.Dataset,
) -> None:
    """
    Test that plot_trajectories runs and returns a Matplotlib Figure.

    This test verifies the basic functionality of the plotting function. It
    ensures that the function executes without raising an exception and that its
    return type is a `matplotlib.figure.Figure` object, which confirms that the
    plotting machinery was successfully invoked.

    Parameters
    ----------
    sample_trajectory_dataset : xr.Dataset
        The sample trajectory dataset from the pytest fixture.
    """
    # Act: Call the function to be tested
    fig = plot_trajectories(sample_trajectory_dataset)

    # Assert: Check the return type and basic properties
    assert isinstance(fig, Figure)
    # The figure should have one axes object for the map
    assert len(fig.get_axes()) == 1

    # Clean up the figure to avoid displaying it in test runners
    plt.close(fig)
