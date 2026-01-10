
"""Unit tests for the MetDataset class in plat/met_handler.py."""

import os
from datetime import datetime, timedelta

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from plat.met_handler import MetDataset


@pytest.fixture(scope="module")
def sample_netcdf_file(tmpdir_factory):
    """
    Creates a temporary NetCDF file with sample meteorological data for testing.

    The data includes variables with aliased names to test normalization.
    """
    tmpdir = tmpdir_factory.mktemp("data")
    file_path = os.path.join(str(tmpdir), "test_met_data.nc")

    # Define coordinates
    time = [datetime(2023, 1, 1, 0) + timedelta(hours=i) for i in range(12)]
    latitude = np.arange(30, 51, 10)  # 30, 40, 50
    longitude = np.arange(-125, -104, 10)  # -125, -115, -105

    # Create a sample dataset with aliased variable names
    ds = xr.Dataset(
        {
            'UGRD': (('time', 'latitude', 'longitude'), np.random.rand(12, 3, 3)),
            'VGRD': (('time', 'latitude', 'longitude'), np.random.rand(12, 3, 3)),
            'TMP': (('time', 'latitude', 'longitude'), np.random.rand(12, 3, 3)),
        },
        coords={'time': time, 'latitude': latitude, 'longitude': longitude},
    )

    ds.to_netcdf(file_path)
    return file_path


def test_metdataset_init_and_normalize(sample_netcdf_file):
    """
    Tests that MetDataset initializes correctly and normalizes variable names.
    """
    met_data = MetDataset(sample_netcdf_file)
    assert isinstance(met_data.ds, xr.Dataset)

    # Check for standardized variable names
    assert 'u' in met_data.ds
    assert 'v' in met_data.ds
    assert 't' in met_data.ds

    # Ensure original aliased names are gone
    assert 'UGRD' not in met_data.ds
    assert 'VGRD' not in met_data.ds
    assert 'TMP' not in met_data.ds


def test_metdataset_subset(sample_netcdf_file):
    """
    Tests the spatial and temporal subsetting of the MetDataset.
    """
    met_data = MetDataset(sample_netcdf_file)

    # Define subset boundaries
    time_range = ('2023-01-01T02:00', '2023-01-01T05:00')
    lat_bounds = (35.0, 45.0)
    lon_bounds = (-120.0, -110.0)

    subset_ds = met_data.subset(time_range, lat_bounds, lon_bounds)

    # Verify time dimension
    expected_times = np.array([
        '2023-01-01T02:00',
        '2023-01-01T03:00',
        '2023-01-01T04:00',
        '2023-01-01T05:00',
    ], dtype='datetime64[ns]')
    np.testing.assert_array_equal(subset_ds['time'].values, expected_times)

    # Verify latitude dimension
    expected_lats = [40.0]
    np.testing.assert_array_equal(subset_ds['latitude'].values, expected_lats)

    # Verify longitude dimension
    expected_lons = [-115.0]
    np.testing.assert_array_equal(subset_ds['longitude'].values, expected_lons)


def test_metdataset_lazy_loading(sample_netcdf_file):
    """
    Checks if the dataset loaded by MetDataset is a Dask-backed dataset.
    """
    met_data = MetDataset(sample_netcdf_file, chunks='auto')
    # The data should be a Dask array, not a NumPy array in memory
    assert isinstance(met_data.ds['u'].data, da.Array)


def test_metdataset_eager_loading(sample_netcdf_file):
    """
    Tests that passing chunks=None results in an in-memory NumPy array.
    """
    met_data = MetDataset(sample_netcdf_file, chunks=None)
    # The data should be a NumPy array, not a Dask array
    assert isinstance(met_data.ds['u'].data, np.ndarray)
    assert not hasattr(met_data.ds['u'].data, 'dask')


def test_subset_provenance(sample_netcdf_file):
    """
    Tests that the subset method adds a history attribute for provenance.
    """
    met_data = MetDataset(sample_netcdf_file)

    # Define subset boundaries
    time_range = ('2023-01-01T02:00', '2023-01-01T05:00')
    lat_bounds = (35.0, 45.0)
    lon_bounds = (-120.0, -110.0)

    subset_ds = met_data.subset(time_range, lat_bounds, lon_bounds)

    # Check for the 'history' attribute
    assert 'history' in subset_ds.attrs
    # Check that the history attribute contains the subsetting information
    assert "Subsetted data" in subset_ds.attrs['history']


def test_metdataset_repr(sample_netcdf_file):
    """
    Tests the __repr__ method for a clear, informative representation.
    """
    met_data = MetDataset(sample_netcdf_file)
    repr_str = repr(met_data)

    # Check for the file path
    assert sample_netcdf_file in repr_str

    # Check for coordinate names
    assert "Coordinates: ['time', 'latitude', 'longitude']" in repr_str

    # Check for normalized data variable names
    assert "Data Variables: ['u', 'v', 't']" in repr_str

def test_init_provenance(sample_netcdf_file):
    """
    Tests that the __init__ method adds a history attribute for provenance.
    """
    met_data = MetDataset(sample_netcdf_file)
    assert 'history' in met_data.ds.attrs
    assert "Opened file" in met_data.ds.attrs['history']


def test_get_coord_names_variants(sample_netcdf_file):
    """
    Tests that _get_coord_names correctly identifies various coordinate aliases.
    """
    # Instantiate MetDataset, we'll overwrite the ds attribute for testing
    met_data = MetDataset(sample_netcdf_file)

    # Define test cases as a list of tuples: (coords, expected_mapping)
    test_cases = [
        (
            {'lat': [40.0], 'lon': [-120.0], 'pressure': [1000.0]},
            {'lat': 'lat', 'lon': 'lon', 'level': 'pressure'},
        ),
        (
            {'latitude': [40.0], 'longitude': [-120.0], 'isobaricInhPa': [850.0]},
            {'lat': 'latitude', 'lon': 'longitude', 'level': 'isobaricInhPa'},
        ),
        (
            {'lat': [40.0], 'lon': [-120.0], 'z': [500.0]},
            {'lat': 'lat', 'lon': 'lon', 'level': 'z'},
        ),
        # New test case to prove robustness with a previously unsupported alias
        (
            {'latitude': [40.0], 'lon': [-110.0], 'isobaric': [700.0]},
            {'lat': 'latitude', 'lon': 'lon', 'level': 'isobaric'},
        ),
    ]

    for coords, expected in test_cases:
        coords['time'] = [datetime(2023, 1, 1)]  # Add time coord for consistency
        ds_variant = xr.Dataset(coords=coords)
        met_data.ds = ds_variant
        coord_names = met_data._get_coord_names()
        assert coord_names == expected
