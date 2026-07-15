"""
FCN 本地 ERA5 预报 — 从 CDS 下载的 NetCDF 直接运行
用法:
  python download_era5.py --date 2025-07-01   # 先下载
  python run_fcn_local.py --date 2025-07-01 --days 3
"""

import numpy as np
import xarray as xr
import os, argparse, datetime
from collections import OrderedDict

ERA5_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "era5_data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forecast")
SAUDI_LAT = (16.0, 32.0)
SAUDI_LON = (34.0, 56.0)

# FCN variable name -> (CDS file type, CDS variable name, pressure level or None)
VAR_MAP = {
    "u10m":  ("surface",  "u10",  None),
    "v10m":  ("surface",  "v10",  None),
    "t2m":   ("surface",  "t2m",  None),
    "sp":    ("surface",  "sp",   None),
    "msl":   ("surface",  "msl",  None),
    "tcwv":  ("surface",  "tcwv", None),
    "u100m": ("surface",  "u10",  None),  # proxy
    "v100m": ("surface",  "v10",  None),  # proxy
    "t850":  ("pressure", "t",    850),
    "u1000": ("pressure", "u",    1000),
    "v1000": ("pressure", "v",    1000),
    "z1000": ("pressure", "z",    1000),
    "u850":  ("pressure", "u",    850),
    "v850":  ("pressure", "v",    850),
    "z850":  ("pressure", "z",    850),
    "u500":  ("pressure", "u",    500),
    "v500":  ("pressure", "v",    500),
    "z500":  ("pressure", "z",    500),
    "t500":  ("pressure", "t",    500),
    "r500":  ("pressure", "r",    500),
    "z50":   ("pressure", "z",    50),
    "r850":  ("pressure", "r",    850),
    "u250":  ("pressure", "u",    250),
    "v250":  ("pressure", "v",    250),
    "z250":  ("pressure", "z",    250),
    "t250":  ("pressure", "t",    250),
}


class LocalERA5Source:
    """earth2studio-compatible data source wrapping CDS ERA5 NetCDF files."""

    def __init__(self, date_str: str):
        self.date = date_str
        date_compact = date_str.replace("-", "")
        self.sfc = xr.open_dataset(os.path.join(ERA5_DIR, f"era5_surface_{date_compact}.nc"))
        self.pl  = xr.open_dataset(os.path.join(ERA5_DIR, f"era5_pressure_{date_compact}.nc"))

    def __call__(self, time, variable):
        """Return global DataArray — Saudi data embedded in 721×1440 grid."""
        import numpy as np
        arrays = []; var_names = []
        sfc_lat = self.sfc["latitude"].values
        sfc_lon = self.sfc["longitude"].values

        for v in variable:
            info = VAR_MAP[v]
            if info[0] == "surface":
                da = self.sfc[info[1]].isel(valid_time=0)
            else:
                level_idx = np.argmin(np.abs(self.pl["pressure_level"].values - info[2]))
                da = self.pl[info[1]].isel(valid_time=0, pressure_level=level_idx)
            arrays.append(da.values)
            var_names.append(v)

        saudi = np.stack(arrays)  # (N_var, nlat, nlon)

        # Embed into global 721×1440 grid (FCN requires global input)
        global_lat = np.arange(90, -90.25, -0.25)
        global_lon = np.arange(0, 360, 0.25)
        lat0 = np.argmin(np.abs(global_lat - sfc_lat[0]))
        lon0 = np.argmin(np.abs(global_lon - sfc_lon[0]))
        nlat, nlon = saudi.shape[1], saudi.shape[2]

        global_arr = np.zeros((len(var_names), len(global_lat), len(global_lon)), dtype=np.float32)
        global_arr[:, lat0:lat0+nlat, lon0:lon0+nlon] = saudi

        return xr.DataArray(
            global_arr[np.newaxis, ...],
            dims=["time", "variable", "lat", "lon"],
            coords={"time": time, "variable": var_names,
                    "lat": global_lat, "lon": global_lon},
        )


def run(date: str, days: int = 3):
    from earth2studio.models.px import FCN
    from earth2studio.run import deterministic
    from earth2studio.io import NetCDF4Backend

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"加载本地 ERA5: {date}")
    data = LocalERA5Source(date)

    print("加载 FourCastNet...")
    model = FCN.load_model(FCN.load_default_package())

    # Saudi output coords
    out_coords = OrderedDict({
        "lat": np.arange(SAUDI_LAT[0], SAUDI_LAT[1] + 0.25, 0.25),
        "lon": np.arange(SAUDI_LON[0], SAUDI_LON[1] + 0.25, 0.25),
    })

    out_path = os.path.join(OUT_DIR, f"fcn_era5_{date.replace('-','')}.nc")
    if os.path.exists(out_path):
        os.remove(out_path)
    io = NetCDF4Backend(out_path)

    nsteps = days * 4
    t0 = np.datetime64(date + "T00:00")
    print(f"\nFCN ERA5 预报: {nsteps} 步 = {days} 天, GPU")

    io = deterministic(
        time=[t0],
        nsteps=nsteps,
        prognostic=model,
        data=data,
        io=io,
        output_coords=out_coords,
    )

    # Copy to latest
    import shutil
    latest = os.path.join(OUT_DIR, "fcn_forecast.nc")
    if os.path.exists(latest):
        os.remove(latest)
    shutil.copy2(out_path, latest)

    print(f"\n完成。输出: {out_path} (→ fcn_forecast.nc)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--days", type=int, default=3)
    args = parser.parse_args()
    run(date=args.date, days=args.days)


if __name__ == "__main__":
    main()
