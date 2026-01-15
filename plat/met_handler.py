"""
Meteorological data handler for PLAT.

This module provides the MetDataset class, which is responsible for ingesting,
normalizing, and subsetting meteorological data from NetCDF or GRIB2 files.
"""

from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Union

import xarray as xr


class MetDataset:
    """A handler for meteorological datasets.

    This class provides a consistent interface to meteorological data from
    various file formats (e.g., GRIB2, NetCDF). It handles lazy loading with Dask,
    normalizes variable and coordinate names to PLAT standards, and provides
    methods for accessing and subsetting the data.

    Attributes
    ----------
    ds : xr.Dataset
        The lazily-loaded xarray Dataset.
    file_path : str
        The path to the meteorological data file.
    VARIABLE_MAP : Dict[str, Tuple[str, ...]]
        A class-level dictionary mapping standard PLAT variable names
        to potential aliases found in source meteorological files.
    COORD_MAP : Dict[str, Tuple[str, ...]]
        A class-level dictionary mapping standard PLAT coordinate names
        to potential aliases.
    """

    # PLAT standard keys and potential aliases found in weather model outputs.
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
        'time': ('time', 'valid_time'),
    }

    def __init__(
        self, file_path: str, chunks: Optional[Dict[str, Union[int, str]]] = 'auto'
    ) -> None:
        """Initialize the MetDataset.

        This constructor opens a meteorological data file (e.g., GRIB2, NetCDF)
        and prepares it for use. It uses Dask for lazy loading to efficiently
        handle datasets that are larger than memory. The constructor attempts
        to use the 'cfgrib' engine for GRIB2 files and falls back to the default
        engine for other formats. After opening the file, it normalizes
        variable and coordinate names.

        Parameters
        ----------
        file_path : str
            The local or remote path to the meteorological data file.
        chunks : Optional[Dict[str, Union[int, str]]], optional
            A dictionary specifying the chunking strategy for Dask,
            by default 'auto'. Example: `{'time': 24, 'latitude': 100}`.
            Set to `None` to disable chunking (loads data into memory).
        """
        self.file_path: str = file_path
        try:
            self.ds: xr.Dataset = xr.open_dataset(
                file_path, engine='cfgrib', chunks=chunks
            )
        except (ValueError, EOFError):
            # Fallback for non-GRIB formats like NetCDF
            self.ds = xr.open_dataset(file_path, chunks=chunks)

        # --- Scientific Hygiene: Update Attributes ---
        timestamp: str = datetime.now(timezone.utc).isoformat()
        self.ds.attrs['history'] = f"[{timestamp}] Opened file: {file_path}"

        self._normalize_names()

    def _normalize_names(self) -> None:
        """
        Normalize meteorological variable and coordinate names to PLAT standards.

        This private method uses a data-driven approach based on the class-level
        `VARIABLE_MAP` and `COORD_MAP`. It intelligently handles cases where
        multiple conflicting aliases for the same standard name exist (e.g., both
        `latitude` and `lat`). The logic prioritizes keeping the alias that is
        a dimension, drops the others, and then renames the canonical alias to
        the standard name.
        """
        all_ds_vars = set(self.ds.variables)
        ds_dims = set(self.ds.dims)
        rename_map: Dict[str, str] = {}
        vars_to_drop: list[str] = []

        combined_map = {**self.VARIABLE_MAP, **self.COORD_MAP}

        for std_name, aliases in combined_map.items():
            found_aliases = [
                alias for alias in aliases if alias in all_ds_vars
            ]

            if not found_aliases:
                continue

            # Choose the canonical variable to keep.
            # Preference is given to aliases that are also dimension coordinates.
            canonical_alias = found_aliases[0]  # Default to the first one found
            is_dim = canonical_alias in ds_dims

            for alias in found_aliases[1:]:
                alias_is_dim = alias in ds_dims
                if alias_is_dim and not is_dim:
                    # This alias is a dimension, the current canonical is not.
                    # Promote this alias to canonical.
                    canonical_alias = alias
                    is_dim = True
                elif alias_is_dim == is_dim:
                    # If both are dims or both are not, prefer the one that
                    # matches the standard name.
                    if alias == std_name:
                        canonical_alias = alias

            # Rename the canonical variable if its name is not the standard.
            if canonical_alias != std_name:
                rename_map[canonical_alias] = std_name

            # Drop all other found aliases for this standard name.
            for alias in found_aliases:
                if alias != canonical_alias:
                    vars_to_drop.append(alias)

        # The operations must be ordered: drop conflicting variables first, then rename.
        if vars_to_drop:
            # Ensure we don't drop a variable that is also slated for renaming.
            # This can happen if the canonical alias is not the standard name.
            final_drop_list = [
                v for v in vars_to_drop if v not in rename_map
            ]
            if final_drop_list:
                self.ds = self.ds.drop_vars(final_drop_list, errors="ignore")

        if rename_map:
            self.ds = self.ds.rename(rename_map)

    def __repr__(self) -> str:
        """Provide a developer-friendly string representation."""
        header = f"MetDataset(file_path='{self.file_path}')"
        coords = f"Coordinates: {list(self.ds.coords.keys())}"
        variables = f"Data Variables: {list(self.ds.data_vars.keys())}"
        return f"{header}\n  {coords}\n  {variables}"


    def subset(
        self,
        time_range: Tuple[str, str],
        lat_bounds: Tuple[float, float],
        lon_bounds: Tuple[float, float],
        level_bounds: Optional[Tuple[float, float]] = None,
    ) -> xr.Dataset:
        """
        Select a spatial and temporal subset of the data.

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
            A tuple containing the minimum and maximum latitude bounds.
        lon_bounds : Tuple[float, float]
            A tuple containing the minimum and maximum longitude bounds.
        level_bounds : Optional[Tuple[float, float]], optional
            A tuple containing the minimum and maximum vertical level bounds,
            by default None.

        Returns
        -------
        xr.Dataset
            A new xarray Dataset view containing the sliced data.

        Examples
        --------
        >>> met_data = MetDataset("path/to/your/data.grib2")
        >>> subset = met_data.subset(
        ...     time_range=('2023-01-01T00:00', '2023-01-01T12:00'),
        ...     lat_bounds=(30.0, 50.0),
        ...     lon_bounds=(-125.0, -110.0),
        ...     level_bounds=(1000.0, 500.0)
        ... )
        """
        slicers: Dict[str, slice] = {
            'time': slice(time_range[0], time_range[1]),
            'lat': slice(lat_bounds[0], lat_bounds[1]),
            'lon': slice(lon_bounds[0], lon_bounds[1]),
        }

        if level_bounds:
            slicers['level'] = slice(level_bounds[0], level_bounds[1])

        subset_ds: xr.Dataset = self.ds.sel(**slicers)

        # --- Scientific Hygiene: Update Attributes ---
        timestamp: str = datetime.now(timezone.utc).isoformat()
        history_log = (
            f"[{timestamp}] Subsetted data to time_range={time_range}, "
            f"lat_bounds={lat_bounds}, lon_bounds={lon_bounds}"
        )
        if level_bounds:
            history_log += f", level_bounds={level_bounds}"

        existing_history: Optional[str] = subset_ds.attrs.get('history')
        if existing_history:
            subset_ds.attrs['history'] = f"{existing_history}\n{history_log}"
        else:
            subset_ds.attrs['history'] = history_log

        return subset_ds
