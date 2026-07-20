"""
Kaggle FCN v1 GPU — 7天预报 (nsteps=28, 每步6h)
PyTorch 原生 AFNO，不需要 ONNX / torch-harmonics
"""
import subprocess, sys

# ── 1. 安装 (不需要 ONNX, 不需要 torch-harmonics) ──
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "earth2studio[fcn]==0.16.0", "cdsapi", "scipy"])

import os, numpy as np, xarray as xr
from collections import OrderedDict
import cdsapi
import torch

print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {DEVICE}")

os.environ["CDSAPI_URL"] = "https://cds.climate.copernicus.eu/api"
os.environ["CDSAPI_KEY"] = "07f04bcc-987d-4b07-b9fd-22b3b3547eaf"

# ── 2. 配置 ──
# 每季1天 × 7天预报 = 4×28步，无重叠，足够评估模型能力
DATES = [
    "2025-01-15",  # 冬
    "2025-04-15",  # 春
    "2025-07-01",  # 夏
    "2025-10-15",  # 秋
]

NSTEPS = 28

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

print(f"\n加载 FCN v1 (PyTorch AFNO)...")
model = FCN.load_model(FCN.load_default_package())

c = cdsapi.Client()
total_ok = 0; total_bad = 0; total_skip = 0

for di, date in enumerate(DATES):
    print(f"\n[{di+1}/{len(DATES)}] {date}")
    date_c = date.replace("-","")
    sfc_path = f"era5_sfc_{date_c}.nc"
    pl_path  = f"era5_pl_{date_c}.nc"
    out_path = f"fcn_gpu_{date_c}.nc"

    if os.path.exists(out_path):
        print(f"  已存在，跳过")
        total_skip += 1
        continue

    # 下载全球 ERA5
    for var_list, ds_name, fname in [
        (SURFACE_VARS, "reanalysis-era5-single-levels", sfc_path),
        (PRESSURE_VARS, "reanalysis-era5-pressure-levels", pl_path),
    ]:
        if os.path.exists(fname): continue
        print(f"  下载 {fname}...")
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
            return xr.DataArray(np.stack(arrays)[None],
                dims=["time","variable","lat","lon"],
                coords={"time":time,"variable":names,"lat":lat,"lon":lon})

    # 推理
    print(f"  推理 {NSTEPS}步 ({DEVICE})...")
    io = NetCDF4Backend(out_path)
    io = deterministic(time=[np.datetime64(f"{date}T00:00")], nsteps=NSTEPS,
                       prognostic=model, data=GlobalSrc(), io=io,
                       device=DEVICE, output_coords=out_coords)
    sfc.close(); pl.close()

    # 验证
    check = xr.open_dataset(out_path)
    t2m = check['t2m'].values
    check.close()

    if float(t2m.max()) > 1e6 or np.any(~np.isfinite(t2m)):
        print(f"  *** 垃圾值! 重载模型重试...")
        os.remove(out_path)
        del model; torch.cuda.empty_cache()
        from earth2studio.models.px import FCN as FCN2
        model = FCN2.load_model(FCN2.load_default_package())
        total_bad += 1
    else:
        total_ok += 1
        print(f"  -> {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB, {t2m.shape[1]}步)")
        print(f"     T2m: {np.nanmin(t2m)-273.15:.1f} ~ {np.nanmax(t2m)-273.15:.1f} C")

    for tmp in [sfc_path, pl_path]:
        if os.path.exists(tmp): os.remove(tmp)

print(f"\n完成: {total_ok} OK, {total_bad} 坏值, {total_skip} 跳过")

# 打包
import zipfile
nc_files = [f for f in os.listdir() if f.startswith("fcn_gpu_") and f.endswith(".nc")]
if nc_files:
    with zipfile.ZipFile("fcn_gpu_results.zip", "w") as zf:
        for f in nc_files: zf.write(f)
    print(f"打包 -> fcn_gpu_results.zip ({os.path.getsize('fcn_gpu_results.zip')/1e6:.1f} MB)")
