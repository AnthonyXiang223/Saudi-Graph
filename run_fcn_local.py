"""
FCN 本地 ERA5 预报 — 从 CDS 下载的 NetCDF 文件直接运行
用法: python run_fcn_local.py --date 2025-07-01 --days 3
前置: python download_era5.py --date 2025-07-01
"""

import numpy as np
import xarray as xr
import os, argparse, datetime
from collections import OrderedDict

ERA5_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "era5_data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forecast")
SAUDI_LAT = (16.0, 32.0)
SAUDI_LON = (34.0, 56.0)


def load_era5_local(date_str: str):
    """Load CDS ERA5 files and remap to FCN's 26-variable format."""
    date_compact = date_str.replace("-", "")
    sfc = xr.open_dataset(os.path.join(ERA5_DIR, f"era5_surface_{date_compact}.nc"))
    pl  = xr.open_dataset(os.path.join(ERA5_DIR, f"era5_pressure_{date_compact}.nc"))

    # Squeeze time dim
    sfc = sfc.isel(valid_time=0)
    pl  = pl.isel(valid_time=0)

    # Rename coords to match FCN
    sfc = sfc.rename({"latitude": "lat", "longitude": "lon"})
    pl  = pl.rename({"latitude": "lat", "longitude": "lon"})

    # Build FCN variable dict (variable_name -> 2D numpy array)
    data = {}

    # Surface
    data["u10m"]  = sfc["u10"].values
    data["v10m"]  = sfc["v10"].values
    data["t2m"]   = sfc["t2m"].values
    data["sp"]    = sfc["sp"].values
    data["msl"]   = sfc["msl"].values
    data["tcwv"]  = sfc["tcwv"].values
    data["u100m"] = sfc["u10"].values   # proxy: use 10m wind for 100m
    data["v100m"] = sfc["v10"].values

    # Pressure level -> variable mapping
    level_map = {
        "t850":  ("t", 850),
        "u1000": ("u", 1000), "v1000": ("v", 1000), "z1000": ("z", 1000),
        "u850":  ("u", 850),  "v850":  ("v", 850),  "z850":  ("z", 850),
        "u500":  ("u", 500),  "v500":  ("v", 500),  "z500":  ("z", 500),
        "t500":  ("t", 500),  "r500":  ("r", 500),
        "z50":   ("z", 50),
        "r850":  ("r", 850),
        "u250":  ("u", 250),  "v250":  ("v", 250),  "z250":  ("z", 250),
        "t250":  ("t", 250),
    }

    pressure_levels = pl["pressure_level"].values
    for fcn_var, (pl_var, pl_level) in level_map.items():
        idx = np.where(np.abs(pressure_levels - pl_level) < 1)[0]
        if len(idx) > 0:
            data[fcn_var] = pl[pl_var].isel(pressure_level=idx[0]).values

    sfc.close(); pl.close()
    return data


def run(date: str, days: int = 3):
    from earth2studio.models.px import FCN
    from earth2studio.io import NetCDF4Backend

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Load local ERA5 ──
    print(f"加载本地 ERA5: {date}")
    era5_data = load_era5_local(date)

    # Verify 26 variables
    expected = ['u10m', 'v10m', 't2m', 'sp', 'msl', 't850', 'u1000', 'v1000',
                'z1000', 'u850', 'v850', 'z850', 'u500', 'v500', 'z500', 't500',
                'z50', 'r500', 'r850', 'tcwv', 'u100m', 'v100m', 'u250', 'v250',
                'z250', 't250']
    missing = set(expected) - set(era5_data.keys())
    if missing:
        print(f"缺失变量: {missing}")
    print(f"可用变量: {len(era5_data)}/26")

    # ── 2. Load FCN model ──
    print("加载 FourCastNet...")
    model = FCN.load_model(FCN.load_default_package())

    # Get model input coords
    input_coords = model.input_coords()

    # ── 3. Stack into input tensor (global grid) ──
    var_order = list(input_coords["variable"])
    stacked = np.stack([era5_data[v] for v in var_order])  # (26, 721, 1440)
    x = stacked[np.newaxis, ...]  # → (1, 26, 721, 1440)

    # ── 4. Output coords (Saudi region) ──
    global_lat = input_coords["lat"]
    global_lon = input_coords["lon"]
    lat_start = np.argmin(np.abs(global_lat - SAUDI_LAT[0]))
    lat_end   = np.argmin(np.abs(global_lat - SAUDI_LAT[1])) + 1
    lon_start = np.argmin(np.abs(global_lon - SAUDI_LON[0]))
    lon_end   = np.argmin(np.abs(global_lon - SAUDI_LON[1])) + 1
    out_coords = OrderedDict({
        "lat": np.arange(SAUDI_LAT[0], SAUDI_LAT[1] + 0.25, 0.25),
        "lon": np.arange(SAUDI_LON[0], SAUDI_LON[1] + 0.25, 0.25),
    })

    # ── 5. Run forecast step by step ──
    out_path = os.path.join(OUT_DIR, f"fcn_era5_{date.replace('-','')}.nc")
    if os.path.exists(out_path):
        os.remove(out_path)
    io = NetCDF4Backend(out_path)

    nsteps = days * 4
    print(f"\nFCN ERA5 预报: {nsteps} 步 = {days} 天, GPU")
    import torch

    for step in range(nsteps + 1):
        saudi_x = x[:, :, lat_start:lat_end, lon_start:lon_end]
        io.write(saudi_x, OrderedDict({
            "time": np.array([np.datetime64(date) + np.timedelta64(step * 6, "h")]),
            "lead_time": np.array([step * 6 * 3_600_000_000_000]),
            "lat": out_coords["lat"],
            "lon": out_coords["lon"],
            "variable": var_order,
        }))

        if step < nsteps:
            with torch.no_grad():
                x = model(x, input_coords)

        if step % 4 == 0:
            print(f"  +{step*6}h", end=" ", flush=True)

    print(f"\n完成。输出: {out_path}")
    # Copy to latest
    import shutil
    latest = os.path.join(OUT_DIR, "fcn_forecast.nc")
    if os.path.exists(latest):
        os.remove(latest)
    shutil.copy2(out_path, latest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--days", type=int, default=3)
    args = parser.parse_args()
    run(date=args.date, days=args.days)


if __name__ == "__main__":
    main()
