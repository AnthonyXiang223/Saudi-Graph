"""
xarray data layer for the Saudi extreme event knowledge graph.

Wraps xarray to load NetCDF indicator files with LRU caching.
Handles 160 x 220 grid at ~0.1 degrees, covering 16.0N-31.9N, 34.0E-55.9E.
"""

from __future__ import annotations

import functools
import os
import re
from difflib import get_close_matches
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------------
# Grid utility helpers
# ---------------------------------------------------------------------------

def find_nearest_grid(
    lat: float,
    lon: float,
    ds: xr.Dataset,
) -> Tuple[int, int, float, float]:
    """Find the nearest grid point indices to the requested lat/lon.

    Handles datasets that use ``latitude``/``longitude`` or ``lat``/``lon``
    as their spatial dimension names.

    Parameters
    ----------
    lat : float
        Target latitude (degrees north, 16.0 -- 31.9).
    lon : float
        Target longitude (degrees east, 34.0 -- 55.9).
    ds : xr.Dataset
        An open dataset containing at least one spatial pair of coords.

    Returns
    -------
    lat_idx : int
        Index along the latitude dimension.
    lon_idx : int
        Index along the longitude dimension.
    nearest_lat : float
        Actual latitude value at the nearest grid point.
    nearest_lon : float
        Actual longitude value at the nearest grid point.
    """
    lat_arr, lon_arr = get_coordinates(ds)

    lat_idx = int(np.abs(lat_arr - lat).argmin())
    lon_idx = int(np.abs(lon_arr - lon).argmin())

    return lat_idx, lon_idx, float(lat_arr[lat_idx]), float(lon_arr[lon_idx])


def get_coordinates(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lat_array, lon_array) from a dataset.

    Tries ``latitude``/``longitude`` first, then falls back to
    ``lat``/``lon``.

    Parameters
    ----------
    ds : xr.Dataset

    Returns
    -------
    lat : np.ndarray
        1-D latitude values.
    lon : np.ndarray
        1-D longitude values.

    Raises
    ------
    KeyError
        If neither naming convention is present.
    """
    for lat_name, lon_name in [("latitude", "longitude"), ("lat", "lon")]:
        if lat_name in ds.coords and lon_name in ds.coords:
            return ds[lat_name].values, ds[lon_name].values

    raise KeyError(
        "Could not find spatial coordinates. "
        "Expected 'latitude'/'longitude' or 'lat'/'lon'; "
        f"found coords: {list(ds.coords)}"
    )


# ---------------------------------------------------------------------------
# DataLayer
# ---------------------------------------------------------------------------

class DataLayer:
    """Spatiotemporal data access layer with LRU-cached NetCDF loading.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing ``saudi_indicators_YYYYMMDD.nc``
        files.
    """

    def __init__(self, data_dir: str) -> None:
        self._data_dir = os.path.abspath(data_dir)

        # Cache statistics tracked manually because functools.lru_cache
        # provides hits/misses on the underlying function, which we expose
        # via cache_info().
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _normalise_date(date_str: str) -> str:
        """Convert *date_str* to ``YYYYMMDD``."""
        cleaned = date_str.strip().replace("-", "").replace("/", "")
        if len(cleaned) != 8 or not cleaned.isdigit():
            raise ValueError(
                f"Expected date in 'YYYYMMDD' or 'YYYY-MM-DD' format, got {date_str!r}"
            )
        return cleaned

    def _build_path(self, date_str: str) -> str:
        """Return the full path to the NetCDF file for *date_str*."""
        normalised = self._normalise_date(date_str)
        filename = f"saudi_indicators_{normalised}.nc"
        return os.path.join(self._data_dir, filename)

    @functools.lru_cache(maxsize=7)
    def _load_dataset(self, cache_key: str) -> xr.Dataset:
        """Cached loader for a single day's NetCDF file.

        The *cache_key* is the normalised date string (``YYYYMMDD``).
        Using ``lru_cache`` on this method means repeated requests for the
        same date skip disk I/O.

        ``.load()`` is called so the entire dataset is in memory and later
        operations do not inadvertently trigger lazy re-reads.
        """
        path = self._build_path(cache_key)

        if not os.path.isfile(path):
            raise FileNotFoundError(f"NetCDF file not found: {path}")

        ds = xr.open_dataset(path, engine="netcdf4")
        ds.load()
        return ds

    # -- public API ---------------------------------------------------------

    def load_day(self, date_str: str) -> xr.Dataset:
        """Load a single day's NetCDF file with LRU caching.

        Parameters
        ----------
        date_str : str
            Date in ``"YYYYMMDD"`` or ``"YYYY-MM-DD"`` format.

        Returns
        -------
        xr.Dataset
            Fully in-memory dataset (``.load()`` has been called).

        Raises
        ------
        FileNotFoundError
            If the file does not exist on disk.
        """
        cache_key = self._normalise_date(date_str)

        # Record miss/hit before the actual call so we can track ourselves.
        # functools.lru_cache stores its stats internally, but the
        # _load_dataset method is static-like and we want per-instance stats.
        if cache_key in getattr(self._load_dataset, "cache", {}):
            # We're about to get a hit -- but lru_cache is on the function,
            # not per-instance, so we track hits/misses manually via a
            # wrapper approach: check the underlying cache by probing with
            # cache_info and inferring.
            pass  # tracked below via cache_info delta

        # Instead of trying to infer hit/miss externally, we wrap the call
        # and compare cache_info before and after.
        info_before = self._load_dataset.cache_info()
        ds = self._load_dataset(cache_key)
        info_after = self._load_dataset.cache_info()

        if info_after.hits > info_before.hits:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

        return ds

    def get_timeseries(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        variables: Union[str, List[str]],
    ) -> pd.DataFrame:
        """Extract a time series for a specific grid point.

        Parameters
        ----------
        lat : float
            Target latitude.
        lon : float
            Target longitude.
        start_date : str
            First date (inclusive), ``"YYYYMMDD"`` or ``"YYYY-MM-DD"``.
        end_date : str
            Last date (inclusive).
        variables : str or list of str
            Variable name(s) to extract.

        Returns
        -------
        pd.DataFrame
            Index = ``pd.DatetimeIndex``, columns = variable names.

        Raises
        ------
        FileNotFoundError
            If any day's file is missing.
        KeyError
            If a requested variable is not present in the dataset.
        """
        if isinstance(variables, str):
            variables = [variables]

        start = pd.Timestamp(self._normalise_date(start_date))
        end = pd.Timestamp(self._normalise_date(end_date))
        date_range = pd.date_range(start, end, freq="D")

        # Determine the grid indices from the first available day.
        first_ds = self.load_day(start.strftime("%Y%m%d"))
        lat_idx, lon_idx, nearest_lat, nearest_lon = find_nearest_grid(
            lat, lon, first_ds
        )

        # Validate all variables exist in the first dataset.
        self._check_variables(first_ds, variables)

        # Determine which spatial dim names this variable uses.
        lat_name, lon_name = self._resolve_spatial_dims(first_ds, variables[0])

        records: Dict[str, list] = {v: [] for v in variables}
        dates: List[pd.Timestamp] = []

        for day in date_range:
            day_key = day.strftime("%Y%m%d")
            ds = self.load_day(day_key)
            dates.append(day)

            for var in variables:
                try:
                    da = ds[var]
                except KeyError:
                    raise KeyError(
                        f"Variable {var!r} not found in dataset for {day_key}. "
                        f"Available variables: {list(ds.data_vars)}"
                    )

                # Slice the data array at the nearest indices.
                val = da.isel({lat_name: lat_idx, lon_name: lon_idx}).values
                records[var].append(float(val))

        df = pd.DataFrame(records, index=pd.DatetimeIndex(dates, name="time"))
        df.index.freq = "D"
        df.attrs["nearest_lat"] = nearest_lat
        df.attrs["nearest_lon"] = nearest_lon
        df.attrs["lat_idx"] = lat_idx
        df.attrs["lon_idx"] = lon_idx
        return df

    def get_spatial_snapshot(
        self,
        date_str: str,
        variable: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a 2-D spatial slice for a single variable on a single date.

        Parameters
        ----------
        date_str : str
            Date string.
        variable : str
            Variable name.

        Returns
        -------
        data : np.ndarray
            2-D array (lat, lon).
        lat : np.ndarray
            1-D latitude coordinates.
        lon : np.ndarray
            1-D longitude coordinates.

        Raises
        ------
        KeyError
            If *variable* does not exist.
        """
        ds = self.load_day(date_str)
        self._check_variables(ds, [variable])
        lat_arr, lon_arr = get_coordinates(ds)

        lat_name, _ = self._resolve_spatial_dims(ds, variable)
        # Re-derive lon_name in case it differs.
        _, lon_name = self._resolve_spatial_dims(ds, variable)

        data = ds[variable].values

        # Ensure we return a 2-D array; some variables might have extra dims
        # (e.g. time). Squeeze any size-1 dimensions but keep (lat, lon).
        if data.ndim > 2:
            # Attempt to squeeze dimensions that are 1 -- but preserve
            # lat/lon axes.
            extra_axes = [
                ax
                for ax, name in enumerate(ds[variable].dims)
                if name not in (lat_name, lon_name) and ds[variable].shape[ax] == 1
            ]
            if extra_axes:
                data = np.squeeze(data, axis=tuple(extra_axes))

        if data.ndim != 2:
            raise ValueError(
                f"Expected 2-D data for variable {variable!r}, "
                f"got shape {data.shape} with dims {list(ds[variable].dims)}"
            )

        return data, lat_arr, lon_arr

    def get_multi_variable_snapshot(
        self,
        date_str: str,
        variables: List[str],
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return a dict mapping each variable to its (data, lat, lon) triple.

        Loads the dataset **once** for the day and extracts every requested
        variable from that single dataset.

        Parameters
        ----------
        date_str : str
            Date string.
        variables : list of str
            Variable names.

        Returns
        -------
        dict
            ``{variable: (data_2d, lat_1d, lon_1d)}``
        """
        ds = self.load_day(date_str)
        self._check_variables(ds, variables)

        lat_arr, lon_arr = get_coordinates(ds)

        result: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        for var in variables:
            lat_name, lon_name = self._resolve_spatial_dims(ds, var)

            data = ds[var].values
            if data.ndim > 2:
                extra_axes = [
                    ax
                    for ax, name in enumerate(ds[var].dims)
                    if name not in (lat_name, lon_name) and ds[var].shape[ax] == 1
                ]
                if extra_axes:
                    data = np.squeeze(data, axis=tuple(extra_axes))

            if data.ndim != 2:
                raise ValueError(
                    f"Expected 2-D data for variable {var!r}, "
                    f"got shape {data.shape} with dims {list(ds[var].dims)}"
                )

            result[var] = (data, lat_arr.copy(), lon_arr.copy())

        return result

    def cache_info(self) -> dict:
        """Return cache hit/miss statistics.

        Returns
        -------
        dict
            Keys: ``"hits"``, ``"misses"``, ``"currsize"``, ``"maxsize"``.
        """
        func_info = self._load_dataset.cache_info()
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "currsize": func_info.currsize,
            "maxsize": func_info.maxsize,
        }

    # -- internal helpers ---------------------------------------------------

    def _check_variables(
        self, ds: xr.Dataset, variables: List[str]
    ) -> None:
        """Raise KeyError with helpful suggestions if a variable is missing."""
        available = list(ds.data_vars)
        missing = [v for v in variables if v not in ds.data_vars]

        if not missing:
            return

        # Build a suggestion message for the first missing variable.
        suggestions = get_close_matches(missing[0], available, n=5, cutoff=0.4)
        hint = ""
        if suggestions:
            hint = f" Did you mean: {suggestions}?"
        raise KeyError(
            f"Variable(s) {missing} not found in dataset. "
            f"Available variables ({len(available)} total): {available}{hint}"
        )

    @staticmethod
    def _resolve_spatial_dims(
        ds: xr.Dataset, variable: str
    ) -> Tuple[str, str]:
        """Return the (lat_dim, lon_dim) names that *variable* actually uses."""
        var_dims = set(ds[variable].dims)

        for lat_candidate, lon_candidate in [
            ("latitude", "longitude"),
            ("lat", "lon"),
        ]:
            if lat_candidate in var_dims and lon_candidate in var_dims:
                return lat_candidate, lon_candidate

        # Fallback: try to match any dim that contains 'lat' / 'lon'
        lat_dim = None
        lon_dim = None
        for d in var_dims:
            low = d.lower()
            if "lat" in low and lat_dim is None:
                lat_dim = d
            elif "lon" in low and lon_dim is None:
                lon_dim = d

        if lat_dim and lon_dim:
            return lat_dim, lon_dim

        raise KeyError(
            f"Cannot determine spatial dimensions for variable {variable!r}. "
            f"Variable dims: {list(ds[variable].dims)}. "
            f"Expected a pair containing 'lat'/'latitude' and 'lon'/'longitude'."
        )
