"""
Meteorological data handler for PLAT.

This module provides the MetDataset class, which is responsible for ingesting,
 normalizing, and subsetting meteorological data from NetCDF or GRIB2 files.
"""

from typing import Dict, Tuple

import xarray as xr


class MetDataset:
    """
    A handler for meteorological datasets that normalizes variable names
    and provides subsetting capabilities using xarray and Dask.

    Attributes:
        ds (xr.Dataset): The lazily-loaded xarray Dataset.
    """

    # HYSPLIT standard keys and potential aliases found in weather model outputs.
    VARIABLE_MAP: Dict[str, Tuple[str, ...]] = {
        'u': ('u', 'UGRD', 'u_wind'),
        'v': ('v', 'VGRD', 'v_wind'),
        'w': ('w', 'W', 'W_wind', 'VVEL'),
        't': ('t', 'TMP', 'temperature'),
        'z': ('z', 'HGT', 'geopotential_height'),
    }

    def __init__(self, file_path: str):
        """
        Initializes the MetDataset by opening a meteorological data file.

        It attempts to open GRIB2 files using the 'cfgrib' engine and falls
        back to the default engine for other formats like NetCDF. Dask is
        used for lazy loading to handle large datasets efficiently.

        Args:
            file_path (str): The path to the meteorological data file.
        """
        try:
            self.ds = xr.open_dataset(file_path, engine='cfgrib', chunks='auto')
        except ValueError:
            # Fallback for non-GRIB formats like NetCDF
            self.ds = xr.open_dataset(file_path, chunks='auto')

        self._normalize_variable_names()

    def _normalize_variable_names(self):
        """
        Renames variables in the dataset to standardized HYSPLIT keys.
        """
        rename_dict = {}
        for std_name, aliases in self.VARIABLE_MAP.items():
            for alias in aliases:
                if alias in self.ds.variables:
                    rename_dict[alias] = std_name
                    break
        self.ds = self.ds.rename(rename_dict)

    def subset(
        self,
        time_range: Tuple[str, str],
        lat_bounds: Tuple[float, float],
        lon_bounds: Tuple[float, float],
    ) -> xr.Dataset:
        """
        Selects a spatial and temporal subset of the meteorological data.

        Args:
            time_range (Tuple[str, str]): A tuple containing the start and end
                time strings (e.g., ('2023-01-01T00:00', '2023-01-01T06:00')).
            lat_bounds (Tuple[float, float]): A tuple containing the minimum and
                maximum latitude bounds (e.g., (30.0, 50.0)).
            lon_bounds (Tuple[float, float]): A tuple containing the minimum and
                maximum longitude bounds (e.g., (-125.0, -110.0)).

        Returns:
            xr.Dataset: A new xarray Dataset containing the sliced data.
        """
        time_slice = slice(time_range[0], time_range[1])
        lat_slice = slice(lat_bounds[0], lat_bounds[1])
        lon_slice = slice(lon_bounds[0], lon_bounds[1])

        return self.ds.sel(time=time_slice, latitude=lat_slice, longitude=lon_slice)
