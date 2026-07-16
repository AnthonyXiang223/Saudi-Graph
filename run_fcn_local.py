"""
本地 ERA5 预报 — 支持 Pangu6 / FCN / SFNO
用法:
  python download_era5.py --date 2025-07-01   # 先下载
  python run_fcn_local.py --date 2025-07-01 --days 3 --model pangu6
"""

import numpy as np
import xarray as xr
import os, argparse, datetime
from collections import OrderedDict

ERA5_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "era5_data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forecast")
SAUDI_LAT = (16.0, 32.0)
SAUDI_LON = (34.0, 56.0)


class LocalERA5Source:
    """earth2studio-compatible data source wrapping CDS ERA5 NetCDF files."""

    def __init__(self, date_str: str, model_type: str = "pangu"):
        self.date = date_str
        self.model_type = model_type
        date_compact = date_str.replace("-", "")
        self.sfc = xr.open_dataset(os.path.join(ERA5_DIR, f"era5_surface_{date_compact}.nc"))
        self.pl  = xr.open_dataset(os.path.join(ERA5_DIR, f"era5_pressure_{date_compact}.nc"))

    def __call__(self, time, variable):
        """Return xarray DataArray with dims (time, variable, lat, lon)."""
        arrays = []; var_names = []
        for v in variable:
            da = self._get_var(v)
            da = da.rename({"latitude": "lat", "longitude": "lon"})
            arrays.append(da.values)
            var_names.append(v)

        stacked = np.stack(arrays)
        lat = self.sfc["latitude"].values
        lon = self.sfc["longitude"].values
        return xr.DataArray(
            stacked[np.newaxis, ...],
            dims=["time", "variable", "lat", "lon"],
            coords={"time": time, "variable": var_names, "lat": lat, "lon": lon},
        )

    def _get_var(self, v: str):
        """Map variable name to CDS data."""
        # Surface variables
        surface_map = {
            "t2m": "t2m", "msl": "msl", "sp": "sp", "tcwv": "tcwv",
            "u10m": "u10", "v10m": "v10",
            "u100m": "u10", "v100m": "v10",  # proxy
        }
        if v in surface_map:
            return self.sfc[surface_map[v]].isel(valid_time=0)

        # Pressure-level: parse varname_level e.g. "z500" -> ("z", 500)
        for prefix in ["z", "q", "t", "u", "v", "r"]:
            if v.startswith(prefix) and v[len(prefix):].isdigit():
                pl_var = {"z": "z", "q": "q", "t": "t", "u": "u", "v": "v", "r": "r"}[prefix]
                level = int(v[len(prefix):])
                level_idx = np.argmin(np.abs(self.pl["pressure_level"].values - level))
                return self.pl[pl_var].isel(valid_time=0, pressure_level=level_idx)

        raise KeyError(f"Unknown variable: {v}")


def run(date: str, days: int = 3, model_name: str = "pangu6"):
    from earth2studio.run import deterministic
    from earth2studio.io import NetCDF4Backend

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"加载本地 ERA5: {date}")
    data = LocalERA5Source(date, model_name)

    # Load model
    if model_name == "pangu6":
        from earth2studio.models.px import Pangu6
        print("模型: Pangu-Weather (华为, 6h 步长)")
        model = Pangu6.load_model(Pangu6.load_default_package())
    elif model_name == "fcn":
        from earth2studio.models.px import FCN
        print("模型: FourCastNet v1")
        model = FCN.load_model(FCN.load_default_package())
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Output
    out_coords = OrderedDict({
        "lat": np.arange(SAUDI_LAT[0], SAUDI_LAT[1] + 0.25, 0.25),
        "lon": np.arange(SAUDI_LON[0], SAUDI_LON[1] + 0.25, 0.25),
    })

    out_path = os.path.join(OUT_DIR, f"{model_name}_era5_{date.replace('-','')}.nc")
    if os.path.exists(out_path):
        os.remove(out_path)
    io = NetCDF4Backend(out_path)

    nsteps = days * 4  # all supported models use 6h steps
    t0 = np.datetime64(date + "T00:00")
    print(f"\n预报: {nsteps} 步 = {days} 天")

    io = deterministic(
        time=[t0], nsteps=nsteps, prognostic=model,
        data=data, io=io, output_coords=out_coords,
    )

    import shutil
    latest = os.path.join(OUT_DIR, "fcn_forecast.nc")
    if os.path.exists(latest):
        os.remove(latest)
    shutil.copy2(out_path, latest)

    print(f"\n完成 → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--model", type=str, default="pangu6")
    args = parser.parse_args()
    run(date=args.date, days=args.days, model_name=args.model)


if __name__ == "__main__":
    main()
