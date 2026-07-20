"""
Kaggle Pangu6 — 华为 Nature 2023, 精度优于 FCN v1
"""

import subprocess, sys

# ── 1. 安装 ──
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "earth2studio==0.16.0", "cdsapi", "scipy", "onnxruntime"])

import os, numpy as np, xarray as xr
from collections import OrderedDict
import cdsapi

os.environ["CDSAPI_URL"] = "https://cds.climate.copernicus.eu/api"
os.environ["CDSAPI_KEY"] = "07f04bcc-987d-4b07-b9fd-22b3b3547eaf"

# ── 2. 四季×7天 ──
DATES = [
    "2025-01-15",
    "2025-04-15",
    "2025-07-01",
    "2025-10-15",
]

SURFACE_VARS = ["10m_u_component_of_wind","10m_v_component_of_wind",
    "2m_temperature","surface_pressure","mean_sea_level_pressure",
    "total_column_water_vapour"]
PRESSURE_LEVELS = [50,100,150,200,250,300,400,500,600,700,850,925,1000]
PRESSURE_VARS = ["temperature","u_component_of_wind","v_component_of_wind",
    "geopotential","specific_humidity"]

SFC_MAP = {"t2m":"t2m","msl":"msl","sp":"sp","tcwv":"tcwv",
           "u10m":"u10","v10m":"v10","u100m":"u10","v100m":"v10"}
PL_MAP = {}
for pfx, plv in [("t","t"),("u","u"),("v","v"),("z","z"),("q","q")]:
    for lv in PRESSURE_LEVELS:
        PL_MAP[f"{pfx}{lv}"] = (plv, lv)

SAUDI_LAT = np.arange(16.0, 32.25, 0.25)
SAUDI_LON = np.arange(34.0, 56.25, 0.25)

# ── 3. 加载 Pangu6 ──
from earth2studio.models.px import Pangu6
from earth2studio.run import deterministic
from earth2studio.io import NetCDF4Backend

print("加载 Pangu6...")
model = Pangu6.load_model(Pangu6.load_default_package())

c = cdsapi.Client()
for date in DATES:
    print(f"\n=== {date} ===")
    date_c = date.replace("-","")
    sfc_path = f"era5_sfc_{date_c}.nc"
    pl_path  = f"era5_pl_{date_c}.nc"
    out_path = f"pangu_{date_c}.nc"

    if os.path.exists(out_path):
        print("  已存在，跳过")
        continue

    for var_list, ds_name, fname in [
        (SURFACE_VARS, "reanalysis-era5-single-levels", sfc_path),
        (PRESSURE_VARS, "reanalysis-era5-pressure-levels", pl_path),
    ]:
        if os.path.exists(fname): continue
        req = {"product_type":"reanalysis","variable":var_list,"date":date,
               "time":"00:00","number":"0","data_format":"netcdf"}
        if "pressure" in ds_name:
            req["pressure_level"] = [str(p) for p in PRESSURE_LEVELS]
        c.retrieve(ds_name, req, fname)

    sfc = xr.open_dataset(sfc_path); pl = xr.open_dataset(pl_path)
    lat = sfc["latitude"].values; lon = sfc["longitude"].values

    class GSrc:
        def __call__(self, time, variable):
            arrays=[]; names=[]
            for v in variable:
                if v in SFC_MAP:
                    da = sfc[SFC_MAP[v]].isel(valid_time=0).values
                else:
                    plv, lv = PL_MAP[v]
                    idx = np.argmin(np.abs(pl["pressure_level"].values-lv))
                    da = pl[plv].isel(valid_time=0, pressure_level=idx).values
                arrays.append(da); names.append(v)
            return xr.DataArray(np.stack(arrays)[None],
                dims=["time","variable","lat","lon"],
                coords={"time":time,"variable":names,"lat":lat,"lon":lon})

    io = NetCDF4Backend(out_path)
    io = deterministic(time=[np.datetime64(date+"T00:00")], nsteps=2,
                       prognostic=model, data=GSrc(), io=io, device="cpu",
                       output_coords=OrderedDict({"lat":SAUDI_LAT,"lon":SAUDI_LON}))

    sfc.close(); pl.close()
    for tmp in [sfc_path, pl_path]:
        if os.path.exists(tmp): os.remove(tmp)

    # 验证
    check = xr.open_dataset(out_path)
    tmax = float(check['t2m'].values.max()); check.close()
    if tmax > 1e6:
        print("  ⚠ 垃圾值!")
        os.remove(out_path)
    else:
        print(f"  → {out_path}")

# 打包
import zipfile
files = [f for f in os.listdir() if f.startswith("pangu_")]
total_mb = sum(os.path.getsize(f) for f in files)/1e6
print(f"\n{len(files)} 个文件, {total_mb:.1f} MB")
with zipfile.ZipFile("pangu_results.zip","w") as zf:
    for f in files: zf.write(f)
print("完成 → pangu_results.zip")
