"""
Example script to validate the MetDataset handler with a real GFS file from S3.

This script downloads a sample GFS GRIB2 file from the NOAA Big Data Program
S3 bucket, uses the MetDataset to open and normalize it, and asserts that the
normalization of common GFS variables was successful.

This script is runnable directly and can also be collected by pytest.
"""

import os
import sys
import tempfile
from typing import Generator

import pytest
import s3fs

# Adjust the path to import from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plat.met_handler import MetDataset

# A stable URL to a sample GFS file on the public NOAA S3 bucket.
S3_URL = "s3://noaa-gfs-bdp-pds/gfs.20241106/00/atmos/gfs.t00z.pgrb2.1p00.f000"


@pytest.fixture(scope="module")
def gfs_file_path() -> Generator[str, None, None]:
    """
    Downloads a GFS file from S3 to a temporary location for the test session.
    Yields the file path and cleans up afterwards.
    This is used when running with pytest.
    """
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmpfile:
        local_path = tmpfile.name

    try:
        print(f"Downloading GFS file from: {S3_URL}")
        # Use s3fs for robust, anonymous access to the public bucket
        fs = s3fs.S3FileSystem(anon=True)
        fs.get(S3_URL, local_path)
        print(f"Downloaded to temporary file: {local_path}")
        yield local_path
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)
            print(f"Cleaned up temporary file: {local_path}")


def run_validation_logic(file_path: str):
    """
    Core logic to validate the MetDataset with a given GFS file.
    """
    # 1. Initialize MetDataset with the path to the GFS file
    print("\nInitializing MetDataset with GFS file...")
    met_data = MetDataset(file_path)
    print("MetDataset initialized.")
    print(f"Initial representation:\n{met_data}")

    # 2. Assert that normalization was successful
    print("\nAsserting correct normalization of GFS variables...")
    assert 'u' in met_data.ds, "Variable 'u' not found after normalization."
    assert 'v' in met_data.ds, "Variable 'v' not found after normalization."
    assert 'z' in met_data.ds, "Variable 'z' (geopotential height) not found."
    assert 't' in met_data.ds, "Variable 't' (temperature) not found."
    assert 'lat' in met_data.ds.coords, "Coordinate 'lat' not found."
    assert 'lon' in met_data.ds.coords, "Coordinate 'lon' not found."
    assert 'level' in met_data.ds.coords, "Coordinate 'level' not found."
    assert 'time' in met_data.ds.coords, "Coordinate 'time' not found."

    # Ensure some common GFS aliases are gone
    assert 'UGRD' not in met_data.ds, "Original name 'UGRD' was not removed."
    assert 'isobaricInhPa' not in met_data.ds.coords, "Original name 'isobaricInhPa' was not removed."

    print("\n✅ All assertions passed. GFS file normalization is working correctly.")


def test_gfs_file_normalization(gfs_file_path: str):
    """
    Pytest entry point for the validation logic.
    """
    run_validation_logic(gfs_file_path)


def main():
    """
    Main execution block for running the script directly.
    Handles file download from S3 and cleanup.
    """
    local_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmpfile:
            local_path = tmpfile.name

        print(f"Downloading GFS file from: {S3_URL}")
        fs = s3fs.S3FileSystem(anon=True)
        fs.get(S3_URL, local_path)
        print(f"Downloaded to temporary file: {local_path}")

        run_validation_logic(local_path)

    finally:
        if local_path and os.path.exists(local_path):
            os.remove(local_path)
            print(f"Cleaned up temporary file: {local_path}")


if __name__ == "__main__":
    main()
