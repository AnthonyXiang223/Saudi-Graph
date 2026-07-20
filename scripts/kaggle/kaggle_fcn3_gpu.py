"""
Kaggle FCNv3 GPU — 7天预报 (nsteps=28, 每步6h)
SFNO架构, fp16半精度, 适配 T4 16GB
"""
import subprocess, sys, os

# ── 1. 安装 GPU 依赖 ──
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "torch", "xarray", "netcdf4", "scipy", "cdsapi"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "earth2studio==0.16.0"])

# CUDA 扩展 — 优先直连 GitHub（Kaggle 海外环境），失败再用代理
os.environ["FORCE_CUDA_EXTENSION"] = "1"
for proxy_url in [
    "git+https://github.com/NVIDIA/torch-harmonics.git",
    "git+https://gh-proxy.com/github.com/NVIDIA/torch-harmonics.git",
]:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", proxy_url])
        break
    except:
        continue

for proxy_url in [
    "git+https://github.com/NVIDIA/makani.git",
    "git+https://gh-proxy.com/github.com/NVIDIA/makani.git",
]:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", proxy_url])
        break
    except:
        continue

# 重编译确保 CUDA 扩展生效
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "torch-harmonics", "-y", "-q"])
for proxy_url in [
    "git+https://github.com/NVIDIA/torch-harmonics.git",
    "git+https://gh-proxy.com/github.com/NVIDIA/torch-harmonics.git",
]:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", proxy_url])
        break
    except:
        continue

import numpy as np, xarray as xr
from collections import OrderedDict
import cdsapi, torch

# ── 2. 配置 ──
os.environ["CDSAPI_URL"] = "https://cds.climate.copernicus.eu/api"
os.environ["CDSAPI_KEY"] = "07f04bcc-987d-4b07-b9fd-22b3b3547eaf"

DATES = [
    # 冬季 (7天)
    "2025-01-15","2025-01-16","2025-01-17","2025-01-18","2025-01-19","2025-01-20","2025-01-21",
    # 春季 (7天)
    "2025-04-15","2025-04-16","2025-04-17","2025-04-18","2025-04-19","2025-04-20","2025-04-21",
    # 夏季 (7天)
    "2025-07-01","2025-07-02","2025-07-03","2025-07-04","2025-07-05","2025-07-06","2025-07-07",
    # 秋季 (7天)
    "2025-10-15","2025-10-16","2025-10-17","2025-10-18","2025-10-19","2025-10-20","2025-10-21",
]

NSTEPS = 28          # 7天 × 4步/天 (6h timestep)
TIMESTEP_H = 6

SURFACE_VARS = ["10m_u_component_of_wind","10m_v_component_of_wind",
    "2m_temperature","surface_pressure","mean_sea_level_pressure",
    "total_column_water_vapour"]
PRESSURE_LEVELS = [50,100,150,200,250,300,400,500,600,700,850,925,1000]
PRESSURE_VARS = ["temperature","u_component_of_wind","v_component_of_wind",
    "geopotential","specific_humidity"]

BUFFER = {"N": 40, "W": 25, "S": 10, "E": 65}

SFC_MAP = {"t2m":"t2m","msl":"msl","sp":"sp","tcwv":"tcwv",
           "u10m":"u10","v10m":"v10","u100m":"u10","v100m":"v10"}
PL_MAP = {}
for pfx, plv in [("t","t"),("u","u"),("v","v"),("z","z"),("q","q")]:
    for lv in PRESSURE_LEVELS:
        PL_MAP[f"{pfx}{lv}"] = (plv, lv)

GLOBAL_LAT = np.arange(90, -90.25, -0.25)
GLOBAL_LON = np.arange(0, 360, 0.25)
SAUDI_LAT = np.arange(16.0, 32.25, 0.25)
SAUDI_LON = np.arange(34.0, 56.25, 0.25)

# ── 3. 加载 FCNv3 (fp16, GPU) ──
from earth2studio.models.px import FCN3
from earth2studio.run import deterministic
from earth2studio.io import NetCDF4Backend

print("=" * 60)
print("  FCNv3 (SFNO) GPU — 7天预报")
print("=" * 60)

print(f"\n加载 FCNv3 (fp16)...")
model = FCN3.load_model(FCN3.load_default_package())
model = model.half()  # fp16: VRAM减半
model = model.cuda()  # 移到 GPU

print(f"  CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info(0)
    print(f"  VRAM: {total/1e9:.1f} GB, 空闲: {free/1e9:.1f} GB")

print(f"\n配置: {NSTEPS}步 × {TIMESTEP_H}h = {NSTEPS*TIMESTEP_H/24:.0f}天预报")
print(f"日期数: {len(DATES)} ({len(DATES)//7}季 × 7天)")

c = cdsapi.Client()
total_ok = 0
total_bad = 0
total_skip = 0
total_mb = 0.0

for di, date in enumerate(DATES):
    print(f"\n--- [{di+1}/{len(DATES)}] {date} ---")
    date_c = date.replace("-","")
    sfc_path = f"era5_sfc_{date_c}.nc"
    pl_path  = f"era5_pl_{date_c}.nc"
    out_path = f"fcn3_gpu_{date_c}.nc"

    if os.path.exists(out_path):
        print(f"  已存在，跳过")
        total_skip += 1
        total_mb += os.path.getsize(out_path) / 1e6
        continue

    # 下载 ERA5 初始场 (区域裁切以加速)
    for var_list, ds_name, fname in [
        (SURFACE_VARS, "reanalysis-era5-single-levels", sfc_path),
        (PRESSURE_VARS, "reanalysis-era5-pressure-levels", pl_path),
    ]:
        if os.path.exists(fname): continue
        print(f"  下载 {fname}...")
        req = {"product_type":"reanalysis","variable":var_list,"date":date,
               "time":"00:00","number":"0","data_format":"netcdf",
               "area": [BUFFER["N"], BUFFER["W"], BUFFER["S"], BUFFER["E"]]}
        if "pressure" in ds_name:
            req["pressure_level"] = [str(p) for p in PRESSURE_LEVELS]
        c.retrieve(ds_name, req, fname)

    sfc = xr.open_dataset(sfc_path); pl = xr.open_dataset(pl_path)
    buf_lat = sfc["latitude"].values; buf_lon = sfc["longitude"].values
    lat0 = np.argmin(np.abs(GLOBAL_LAT - buf_lat[0]))
    lon0 = np.argmin(np.abs(GLOBAL_LON - buf_lon[0]))
    nlat, nlon = len(buf_lat), len(buf_lon)

    class BufSrc:
        def __call__(self, time, variable):
            arrays=[]; names=[]
            for v in variable:
                if v in SFC_MAP:
                    da = sfc[SFC_MAP[v]].isel(valid_time=0).values
                else:
                    plv, lv = PL_MAP[v]
                    idx = np.argmin(np.abs(pl["pressure_level"].values - lv))
                    da = pl[plv].isel(valid_time=0, pressure_level=idx).values
                arrays.append(da); names.append(v)
            arr = np.zeros((len(arrays), 721, 1440), dtype=np.float32)
            for i in range(len(arrays)):
                arr[i, lat0:lat0+nlat, lon0:lon0+nlon] = arrays[i]
            return xr.DataArray(arr[None],
                dims=["time","variable","lat","lon"],
                coords={"time":time,"variable":names,"lat":GLOBAL_LAT,"lon":GLOBAL_LON})

    # ── GPU 推理 ──
    print(f"  推理中... ({NSTEPS}步, GPU fp16)")
    try:
        io = NetCDF4Backend(out_path)
        io = deterministic(
            time=[np.datetime64(f"{date}T00:00")],
            nsteps=NSTEPS,
            prognostic=model,
            data=BufSrc(),
            io=io,
            output_coords=OrderedDict({"lat": SAUDI_LAT, "lon": SAUDI_LON}),
        )
    except Exception as e:
        print(f"  *** 推理失败: {e}")
        sfc.close(); pl.close()
        continue

    sfc.close(); pl.close()

    # 验证
    check = xr.open_dataset(out_path)
    t2m = check['t2m'].values
    n_steps = t2m.shape[1]
    check.close()

    if np.any(~np.isfinite(t2m)) or float(t2m.max()) > 1e6:
        print(f"  *** 垃圾值! 删除")
        os.remove(out_path)
        total_bad += 1
        # 重载模型重试 (FCN 已知问题)
        del model
        torch.cuda.empty_cache()
        print(f"  重载模型重试...")
        from earth2studio.models.px import FCN3 as FCN3_reload
        model = FCN3_reload.load_model(FCN3_reload.load_default_package())
        model = model.half().cuda()
    else:
        size_mb = os.path.getsize(out_path) / 1e6
        total_ok += 1
        total_mb += size_mb
        print(f"  -> {out_path} ({size_mb:.1f} MB, {n_steps}步)")
        tmin_c = float(np.nanmin(t2m)) - 273.15
        tmax_c = float(np.nanmax(t2m)) - 273.15
        print(f"     T2m: {tmin_c:.1f} ~ {tmax_c:.1f} C")

    # 清理临时文件
    for tmp in [sfc_path, pl_path]:
        if os.path.exists(tmp): os.remove(tmp)

print(f"\n{'='*60}")
print(f"  完成: {total_ok} OK, {total_bad} 垃圾值, {total_skip} 跳过")
print(f"  总大小: {total_mb:.1f} MB")
print(f"{'='*60}")

# 打包
files = [f for f in os.listdir() if f.startswith("fcn3_gpu_") and f.endswith(".nc")]
if files:
    import zipfile
    with zipfile.ZipFile("fcn3_gpu_results.zip", "w") as zf:
        for f in files:
            zf.write(f)
    zip_mb = os.path.getsize("fcn3_gpu_results.zip") / 1e6
    print(f"打包完成 -> fcn3_gpu_results.zip ({zip_mb:.1f} MB)")
