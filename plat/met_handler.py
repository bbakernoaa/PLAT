"""
Meteorological data handler for PLAT.

This module provides the MetDataset class, which is responsible for ingesting,
 normalizing, and subsetting meteorological data from NetCDF or GRIB2 files.
"""

from datetime import datetime
from typing import Dict, Optional, Tuple

import xarray as xr


class MetDataset:
    """A handler for meteorological datasets.

    This class provides a consistent interface to meteorological data from
    various file formats (e.g., GRIB2, NetCDF). It handles lazy loading with Dask,
    normalizes variable names to HYSPLIT standards, and provides methods for
    accessing and subsetting the data.

    Attributes
    ----------
    ds : xr.Dataset
        The lazily-loaded xarray Dataset.
    VARIABLE_MAP : Dict[str, Tuple[str, ...]]
        A class-level dictionary mapping standard HYSPLIT variable names
        to potential aliases found in source meteorological files.

    """

    # HYSPLIT standard keys and potential aliases found in weather model outputs.
    VARIABLE_MAP: Dict[str, Tuple[str, ...]] = {
        'u': ('u', 'UGRD', 'u_wind'),
        'v': ('v', 'VGRD', 'v_wind'),
        'w': ('w', 'W', 'W_wind', 'VVEL'),
        't': ('t', 'TMP', 'temperature'),
        'z': ('z', 'HGT', 'geopotential_height'),
    }

    COORD_MAP: Dict[str, Tuple[str, ...]] = {
    'lat': ('latitude', 'lat'),
    'lon': ('longitude', 'lon'),
    'level': ('level', 'pressure', 'isobaricInhPa', 'z', 'isobaric'),
    }

    def __init__(self, file_path: str, chunks: Optional[Dict] = 'auto'):
        """Initialize the MetDataset.
        This constructor opens a meteorological data file (e.g., GRIB2, NetCDF)
        and prepares it for use. It uses Dask for lazy loading to efficiently
        handle datasets that are larger than memory. The constructor attempts
        to use the 'cfgrib' engine for GRIB2 files and falls back to the default
        engine for other formats. After opening the file, it normalizes the
        variable names.
        Parameters
        ----------
        file_path : str
            The local or remote path to the meteorological data file.
        chunks : Optional[Dict], optional
            A dictionary specifying the chunking strategy for Dask, by default 'auto'.
            Example: `{'time': 24, 'latitude': 100, 'longitude': 100}`.
            Set to `None` to disable chunking (loads data into memory).
        """
        self.file_path = file_path
        try:
            self.ds = xr.open_dataset(file_path, engine='cfgrib', chunks=chunks)
        except ValueError:
            # Fallback for non-GRIB formats like NetCDF
            self.ds = xr.open_dataset(file_path, chunks=chunks)

        # --- Scientific Hygiene: Update Attributes ---
        timestamp = datetime.utcnow().isoformat()
        self.ds.attrs['history'] = f"[{timestamp}] Opened file: {file_path}"

        self._normalize_variable_names()

    def __repr__(self) -> str:
        """Provide a developer-friendly string representation."""
        header = f"MetDataset(file_path='{self.file_path}')"
        coords = f"Coordinates: {list(self.ds.coords.keys())}"
        variables = f"Data Variables: {list(self.ds.data_vars.keys())}"
        return f"{header}\n  {coords}\n  {variables}"

    def _normalize_variable_names(self):
        """Normalize meteorological variable names to HYSPLIT standards.

        This private method iterates through the `VARIABLE_MAP` to find known
        aliases for standard meteorological variables (like 'u', 'v', 't')
        and renames them in the xarray Dataset. This ensures consistent
        data access regardless of the source model's naming conventions.

        """
        rename_dict = {}
        for std_name, aliases in self.VARIABLE_MAP.items():
            for alias in aliases:
                if alias in self.ds.variables:
                    rename_dict[alias] = std_name
                    break
        self.ds = self.ds.rename(rename_dict)

    def _get_coord_names(self) -> Dict[str, str]:
        """Dynamically detect standard coordinate names in the dataset.

        This private helper searches the dataset's coordinates for common
        aliases of latitude, longitude, and vertical level dimensions and
        returns a standardized dictionary mapping generic names to the names
        found in the dataset. This allows other methods to remain agnostic
        to the specific naming conventions of the input data file.

        Returns
        -------
        Dict[str, str]
            A dictionary mapping generic coordinate names ('lat', 'lon', 'level')
            to their actual names as found in the dataset's coordinates.
            Example: {'lat': 'latitude', 'lon': 'longitude', 'level': 'pressure'}

        """
        coord_names = {}
        for std_name, aliases in self.COORD_MAP.items():
            for alias in aliases:
                if alias in self.ds.coords:
                    coord_names[std_name] = alias
                    break
        return coord_names

    def subset(
        self,
        time_range: Tuple[str, str],
        lat_bounds: Tuple[float, float],
        lon_bounds: Tuple[float, float],
        level_bounds: Optional[Tuple[float, float]] = None,
    ) -> xr.Dataset:
        """Select a spatial and temporal subset of the data.
        This method uses xarray's `.sel()` to perform a selection based on
        time, latitude, longitude, and an optional vertical level. The
        selection is lazy and returns a new view of the original dataset
        without loading data into memory.
        Parameters
        ----------
        time_range : Tuple[str, str]
            A tuple containing the start and end time strings for the slice.
            The format should be compatible with xarray's time indexing
            (e.g., 'YYYY-MM-DDTHH:MM').
        lat_bounds : Tuple[float, float]
            A tuple containing the minimum and maximum latitude bounds for the
            selection (e.g., (30.0, 50.0)).
        lon_bounds : Tuple[float, float]
            A tuple containing the minimum and maximum longitude bounds for the
            selection (e.g., (-125.0, -110.0)).
        level_bounds : Optional[Tuple[float, float]], optional
            A tuple containing the minimum and maximum vertical level bounds
            for the selection (e.g., (1000.0, 500.0)), by default None.
        Returns
        -------
        xr.Dataset
            A new xarray Dataset view containing the sliced data.
        """
        coord_names = self._get_coord_names()
        lat_name = coord_names.get('lat', 'latitude')
        lon_name = coord_names.get('lon', 'longitude')

        slicers = {
            'time': slice(time_range[0], time_range[1]),
            lat_name: slice(lat_bounds[0], lat_bounds[1]),
            lon_name: slice(lon_bounds[0], lon_bounds[1]),
        }

        history_log = (
            f"Subsetted data to time_range={time_range}, "
            f"lat_bounds={lat_bounds}, lon_bounds={lon_bounds}"
        )

        if level_bounds and 'level' in coord_names:
            level_name = coord_names['level']
            slicers[level_name] = slice(level_bounds[0], level_bounds[1])
            history_log += f", level_bounds={level_bounds}"


        subset_ds = self.ds.sel(**slicers)

        # --- Scientific Hygiene: Update Attributes ---
        # Get existing history, if any
        existing_history = subset_ds.attrs.get('history')

        # Append new log entry
        if existing_history:
            subset_ds.attrs['history'] = f"{existing_history}\n{history_log}"
        else:
            subset_ds.attrs['history'] = history_log
        return subset_ds
