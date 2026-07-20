"""
Kaggle Pangu6 GPU — 7天预报 (nsteps=28, 每步6h)
"""
import subprocess, sys

# ══════════════════════════════════════════════════════════════
# 1. 安装依赖
#    关键: 不降级 NumPy，改用新版 onnxruntime-gpu (支持 NumPy 2)
# ══════════════════════════════════════════════════════════════
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "earth2studio==0.16.0", "cdsapi", "scipy"])

import os, numpy as np
import torch
print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# 先卸载旧版，从最新往最旧试，找到 NumPy 2 + CUDA 双兼容的版本
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "onnxruntime", "onnxruntime-gpu", "-y", "-q"])

HAS_GPU_ORT = False
# 最新版优先 (onnxruntime>=1.21 支持 NumPy 2)
ORT_VERSIONS = [
    "onnxruntime-gpu",          # 最新 (1.27)
    "onnxruntime-gpu==1.24.1",
    "onnxruntime-gpu==1.21.0",
    "onnxruntime-gpu==1.19.0",
]

if torch.cuda.is_available():
    for ort_ver in ORT_VERSIONS:
        try:
            print(f"  尝试 {ort_ver}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", ort_ver], timeout=120)
            import onnxruntime as ort
            providers = ort.get_available_providers()
            print(f"    providers: {providers}")
            if "CUDAExecutionProvider" in providers:
                print(f"  [OK] GPU ONNX 可用!")
                HAS_GPU_ORT = True
                break
            else:
                print(f"    无 CUDA provider，卸载重试...")
                subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "onnxruntime-gpu", "-y", "-q"])
        except Exception as e:
            print(f"    失败: {type(e).__name__}: {e}")
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "onnxruntime-gpu", "-y", "-q"])

if not HAS_GPU_ORT:
    print("  GPU ONNX 全部失败，回退 CPU")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"])
    import onnxruntime as ort

DEVICE = "cuda" if HAS_GPU_ORT else "cpu"
print(f"推理设备: {DEVICE}")

# ══════════════════════════════════════════════════════════════
# 2. 配置
# ══════════════════════════════════════════════════════════════
import xarray as xr
from collections import OrderedDict
import cdsapi

os.environ["CDSAPI_URL"] = "https://cds.climate.copernicus.eu/api"
os.environ["CDSAPI_KEY"] = "07f04bcc-987d-4b07-b9fd-22b3b3547eaf"

DATES = [
    "2025-01-15",
    "2025-04-15",
    "2025-07-01",
    "2025-10-15",
]

NSTEPS = 28          # 7天 × 4步/天 (6h timestep)

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

# ══════════════════════════════════════════════════════════════
# 3. 加载模型 & 推理
# ══════════════════════════════════════════════════════════════
from earth2studio.models.px import Pangu6
from earth2studio.run import deterministic
from earth2studio.io import NetCDF4Backend

print(f"\n加载 Pangu6...")
model = Pangu6.load_model(Pangu6.load_default_package())

c = cdsapi.Client()
total_ok = 0; total_mb = 0.0

for date in DATES:
    print(f"\n{'='*50}")
    print(f"  {date} — {NSTEPS}步 = {NSTEPS*6/24:.0f}天预报 ({DEVICE})")
    print(f"{'='*50}")
    date_c = date.replace("-","")
    sfc_path = f"era5_sfc_{date_c}.nc"
    pl_path  = f"era5_pl_{date_c}.nc"
    out_path = f"pangu_gpu_{date_c}.nc"

    if os.path.exists(out_path):
        print("  已存在，跳过")
        total_ok += 1
        total_mb += os.path.getsize(out_path) / 1e6
        continue

    # 下载 ERA5 初始场
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

    class GSrc:
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
            return xr.DataArray(np.stack(arrays)[None],
                dims=["time","variable","lat","lon"],
                coords={"time":time,"variable":names,"lat":lat,"lon":lon})

    print(f"  推理中...")
    io = NetCDF4Backend(out_path)
    io = deterministic(
        time=[np.datetime64(date + "T00:00")],
        nsteps=NSTEPS,
        prognostic=model,
        data=GSrc(),
        io=io,
        device=DEVICE,
        output_coords=OrderedDict({"lat": SAUDI_LAT, "lon": SAUDI_LON}),
    )

    sfc.close(); pl.close()
    for tmp in [sfc_path, pl_path]:
        if os.path.exists(tmp): os.remove(tmp)

    # 验证
    check = xr.open_dataset(out_path)
    t2m = check['t2m'].values; n_steps = t2m.shape[1]
    check.close()

    if float(t2m.max()) > 1e6:
        print(f"  *** 垃圾值! 删除")
        os.remove(out_path)
    else:
        size_mb = os.path.getsize(out_path) / 1e6
        total_ok += 1; total_mb += size_mb
        print(f"  -> {out_path} ({size_mb:.1f} MB, {n_steps}步)")
        print(f"     T2m: {t2m.min()-273.15:.1f} ~ {t2m.max()-273.15:.1f} C")

print(f"\n{'='*50}")
print(f"  完成: {total_ok} 文件, {total_mb:.1f} MB")
print(f"{'='*50}")

# 打包
import zipfile
nc_files = [f for f in os.listdir() if f.startswith("pangu_gpu_") and f.endswith(".nc")]
if nc_files:
    with zipfile.ZipFile("pangu_gpu_results.zip", "w") as zf:
        for f in nc_files: zf.write(f)
    print(f"打包 -> pangu_gpu_results.zip ({os.path.getsize('pangu_gpu_results.zip')/1e6:.1f} MB)")
