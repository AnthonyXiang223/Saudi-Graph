"""
Kaggle FCN v1 全球预报批量 — 直接下载全球ERA5，不做缓冲裁切
"""

import subprocess, sys, os, numpy as np, xarray as xr
from collections import OrderedDict

# ── 1. 安装 ──
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "earth2studio[fcn]==0.16.0", "cdsapi", "scipy"])

import cdsapi

os.environ["CDSAPI_URL"] = "https://cds.climate.copernicus.eu/api"
os.environ["CDSAPI_KEY"] = "07f04bcc-987d-4b07-b9fd-22b3b3547eaf"

# ── 2. 四季×7天 ──
DATES = [
    "2025-01-15","2025-01-16","2025-01-17","2025-01-18","2025-01-19","2025-01-20","2025-01-21",
    "2025-04-15","2025-04-16","2025-04-17","2025-04-18","2025-04-19","2025-04-20","2025-04-21",
    "2025-07-01","2025-07-02","2025-07-03","2025-07-04","2025-07-05","2025-07-06","2025-07-07",
    "2025-10-15","2025-10-16","2025-10-17","2025-10-18","2025-10-19","2025-10-20","2025-10-21",
]

SURFACE_VARS = ["10m_u_component_of_wind","10m_v_component_of_wind",
    "2m_temperature","surface_pressure","mean_sea_level_pressure",
    "total_column_water_vapour"]
PRESSURE_LEVELS = [50, 250, 500, 850, 1000]
PRESSURE_VARS = ["temperature","u_component_of_wind","v_component_of_wind",
    "geopotential","relative_humidity"]

SFC_MAP = {"t2m":"t2m","msl":"msl","sp":"sp","tcwv":"tcwv",
           "u10m":"u10","v10m":"v10","u100m":"u10","v100m":"v10"}
PL_MAP = {}
for pfx, plv in [("t","t"),("u","u"),("v","v"),("z","z"),("r","r")]:
    for lv in PRESSURE_LEVELS:
        PL_MAP[f"{pfx}{lv}"] = (plv, lv)

SAUDI_LAT = np.arange(16.0, 32.25, 0.25)
SAUDI_LON = np.arange(34.0, 56.25, 0.25)
out_coords = OrderedDict({"lat": SAUDI_LAT, "lon": SAUDI_LON})

# ── 3. 加载模型 ──
from earth2studio.models.px import FCN
from earth2studio.run import deterministic
from earth2studio.io import NetCDF4Backend
import torch

print("加载 FCN v1...")
model = FCN.load_model(FCN.load_default_package())

c = cdsapi.Client()
for date in DATES:
    print(f"\n=== {date} ===")
    date_c = date.replace("-","")
    sfc_path = f"era5_sfc_{date_c}.nc"
    pl_path  = f"era5_pl_{date_c}.nc"
    out_path = f"fcn_{date_c}.nc"

    if os.path.exists(out_path):
        print("  已存在，跳过")
        continue

    # 下载全球 ERA5
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

    class GlobalSrc:
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
            stacked = np.stack(arrays)
            return xr.DataArray(stacked[None],
                dims=["time","variable","lat","lon"],
                coords={"time":time,"variable":names,"lat":lat,"lon":lon})

    io = NetCDF4Backend(out_path)
    io = deterministic(time=[np.datetime64(f"{date}T00:00")], nsteps=2,
                       prognostic=model, data=GlobalSrc(), io=io,
                       output_coords=out_coords)

    sfc.close(); pl.close()

    # 验证输出
    check = xr.open_dataset(out_path)
    tmax = float(check['t2m'].values.max())
    check.close()

    if tmax > 1e6:
        print("  ⚠ 垃圾值，重载模型重试...")
        os.remove(out_path)
        del model; torch.cuda.empty_cache()
        from earth2studio.models.px import FCN as FCN2
        model = FCN2.load_model(FCN2.load_default_package())
        sfc = xr.open_dataset(sfc_path); pl = xr.open_dataset(pl_path)
        io2 = NetCDF4Backend(out_path)
        io2 = deterministic(time=[np.datetime64(date + 'T00:00')], nsteps=2,
                            prognostic=model, data=GlobalSrc(), io=io2,
                            output_coords=out_coords)
        sfc.close(); pl.close()

    # 清理临时文件
    for tmp in [sfc_path, pl_path]:
        if os.path.exists(tmp): os.remove(tmp)
    print(f"  → {out_path}")

# 打包（只打包 FCN 输出，不含临时文件）
import zipfile
fc_files = [f for f in os.listdir() if f.startswith("fcn_")]
total_mb = sum(os.path.getsize(f) for f in fc_files) / 1e6
print(f"\n{len(fc_files)} 个 FCN 文件, 共 {total_mb:.1f} MB")
with zipfile.ZipFile("fcn_results.zip", "w") as zf:
    for f in fc_files:
        zf.write(f)
        print(f"  已添加: {f}")
print(f"完成 → fcn_results.zip ({os.path.getsize('fcn_results.zip')/1e6:.1f} MB)")
