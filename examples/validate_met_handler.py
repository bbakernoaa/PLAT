"""
Example script to validate the MetDataset handler.

This script creates a temporary NetCDF file with non-standard variable names,
uses the MetDataset to open and normalize it, and asserts that the
normalization was successful.
"""

import os
import tempfile
from datetime import datetime

import numpy as np
import xarray as xr

# Adjust the path to import from the parent directory
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plat.met_handler import MetDataset


def create_sample_netcdf(file_path: str) -> None:
    """Creates a sample NetCDF file with aliased variable names."""
    ds = xr.Dataset(
        {
            'UGRD': (('time', 'latitude', 'longitude'), np.random.rand(1, 3, 3)),
            'VGRD': (('time', 'latitude', 'longitude'), np.random.rand(1, 3, 3)),
            'TMP': (('time', 'latitude', 'longitude'), np.random.rand(1, 3, 3)),
        },
        coords={
            'time': [datetime(2023, 1, 1)],
            'latitude': [30.0, 40.0, 50.0],
            'longitude': [-120.0, -110.0, -100.0],
        },
    )
    ds.to_netcdf(file_path)
    print(f"Created temporary sample file: {file_path}")


def main():
    """Main function to run the validation example."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmpfile:
        file_path = tmpfile.name

    try:
        # 1. Create the sample data
        create_sample_netcdf(file_path)

        # 2. Initialize MetDataset
        print("Initializing MetDataset...")
        met_data = MetDataset(file_path)
        print("MetDataset initialized.")
        print(f"Initial representation:\n{met_data}")

        # 3. Assert that normalization was successful
        print("\nAsserting correct normalization...")
        # Check for standard variable names
        assert 'u' in met_data.ds
        assert 'v' in met_data.ds
        assert 't' in met_data.ds
        # Check for standard coordinate names
        assert 'lat' in met_data.ds.coords
        assert 'lon' in met_data.ds.coords
        assert 'time' in met_data.ds.coords

        # Ensure original names are gone
        assert 'UGRD' not in met_data.ds
        assert 'VGRD' not in met_data.ds
        assert 'latitude' not in met_data.ds.coords

        print("\n✅ All assertions passed. Normalization is working correctly.")

    finally:
        # 4. Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up temporary file: {file_path}")


if __name__ == "__main__":
    main()
