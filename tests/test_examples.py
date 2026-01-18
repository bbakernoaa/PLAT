"""Unit tests for the example scripts."""

import numpy as np
import xarray as xr

from examples.run_basic_trajectory import create_synthetic_velocity_field


def test_create_synthetic_velocity_field():
    """
    Test the output of the create_synthetic_velocity_field function.

    This test verifies that the generated xarray.Dataset has the correct
    dimensions, coordinates, data variables, and attributes.
    """
    ds = create_synthetic_velocity_field()

    # --- 1. Check that it's an xarray.Dataset ---
    assert isinstance(ds, xr.Dataset)

    # --- 2. Check for expected dimensions ---
    expected_dims = {"lat", "lon", "level", "time"}
    assert set(ds.dims) == expected_dims

    # --- 3. Check for expected coordinates ---
    expected_coords = {"lat", "lon", "level", "time"}
    assert set(ds.coords) == expected_coords

    # --- 4. Check for expected data variables ---
    expected_vars = {"u", "v", "w"}
    assert set(ds.data_vars) == expected_vars

    # --- 5. Check the shapes of the data variables ---
    expected_shape = (
        len(ds.time),
        len(ds.level),
        len(ds.lat),
        len(ds.lon),
    )
    assert ds.u.shape == expected_shape
    assert ds.v.shape == expected_shape
    assert ds.w.shape == expected_shape

    # --- 6. Check the dtypes of the arrays ---
    assert ds.u.dtype == np.float64
    assert ds.lat.dtype == np.int64
    assert ds.time.dtype == "datetime64[ns]"

    # --- 7. Check the history attribute ---
    assert "history" in ds.attrs
    assert ds.attrs["history"] == "Created synthetic solid-body rotation field."
