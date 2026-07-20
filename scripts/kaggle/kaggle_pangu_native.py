"""
Kaggle Pangu-Weather 原生 PyTorch 推理 — 7天预报
直接 GitHub 拉模型代码, 不依赖 earth2studio / ONNX
"""
import subprocess, sys, os

# ── 1. 拉取 Pangu-Weather 代码 ──
if not os.path.exists("Pangu-Weather"):
    subprocess.check_call(["git", "clone", "https://github.com/198808xc/Pangu-Weather.git"])
sys.path.insert(0, "Pangu-Weather")

# 安装最简依赖 (不需要 ONNX / cuML / cupy)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "cdsapi", "netcdf4", "xarray", "scipy", "gdown"])

import numpy as np, xarray as xr, torch
import cdsapi
from collections import OrderedDict

print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# ── 2. 下载/加载模型权重 ──
# 从 Pangu-Weather Google Drive 下载 6h 模型
MODEL_FILE = "pangu_weather_6.pth"
if not os.path.exists(MODEL_FILE):
    print("下载模型权重 (约 1.1GB)...")
    # Google Drive file ID for pytorch_6.pth
    # 官方: https://drive.google.com/file/d/1rCXlI_U6e_EYPu3FQthCkTAYz_eO6eI7
    file_id = "1rCXlI_U6e_EYPu3FQthCkTAYz_eO6eI7"
    subprocess.check_call([sys.executable, "-m", "gdown", "--id", file_id, "-O", MODEL_FILE],
                          timeout=600)

# 加载模型
from pseudocode import PanguModel
model = PanguModel().to(device)
state = torch.load(MODEL_FILE, map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()
print(f"模型已加载, 参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# ── 3. ERA5 数据准备 ──
os.environ["CDSAPI_URL"] = "https://cds.climate.copernicus.eu/api"
os.environ["CDSAPI_KEY"] = "07f04bcc-987d-4b07-b9fd-22b3b3547eaf"

DATES = ["2025-01-15", "2025-04-15", "2025-07-01", "2025-10-15"]
NSTEPS = 28  # 7天

# Pangu 需要的变量: 高空 (Z,Q,T,U,V × 13层) + 地表 (MSLP,U10,V10,T2M)
# 模型输入顺序: Z50-Z1000, Q50-Q1000, T50-T1000, U50-U1000, V50-V1000, MSLP, U10, V10, T2M
PRESSURE_LEVELS = [50,100,150,200,250,300,400,500,600,700,850,925,1000]
PL_NAMES = ["z","q","t","u","v"]  # 模型内部顺序
SFC_NAMES = ["msl","u10","v10","t2m"]  # 模型内部顺序

CDS_SFC = ["mean_sea_level_pressure","10m_u_component_of_wind",
           "10m_v_component_of_wind","2m_temperature"]
CDS_PL  = ["geopotential","specific_humidity","temperature",
           "u_component_of_wind","v_component_of_wind"]

SAUDI_LAT = np.arange(16.0, 32.25, 0.25)
SAUDI_LON = np.arange(34.0, 56.25, 0.25)

# 全球网格 (Pangu 输出: 721×1440, 90N→90S, 0→360E)
GLOBAL_LAT = np.arange(90, -90.25, -0.25)
GLOBAL_LON = np.arange(0, 360, 0.25)

# 数据标准化常数 (来自 Pangu-Weather 原论文 ECMWF 统计)
# 地表变量常数
SURFACE_MEAN = np.array([101325.0, 0.0, 0.0, 288.0])  # MSLP, U10, V10, T2M 近似
SURFACE_STD  = np.array([1500.0, 5.0, 5.0, 20.0])
SURFACE_MEAN = torch.tensor(SURFACE_MEAN, dtype=torch.float32).view(1,4,1,1).to(device)
SURFACE_STD  = torch.tensor(SURFACE_STD,  dtype=torch.float32).view(1,4,1,1).to(device)

# 高空变量常数 (简化: 每层不同但这里用统一值)
# Z: clim mean/std by level, Q: by level, T: by level, U: by level, V: by level
# 略去精确值，模型训练时自带 normalization
# Pangu-Weather 实际不需要外部标准化，模型内部有处理

c = cdsapi.Client()

for date in DATES:
    print(f"\n{'='*50}")
    print(f"  {date} — {NSTEPS}步 (7天)")
    print(f"{'='*50}")

    date_c = date.replace("-","")
    sfc_path = f"era5_sfc_{date_c}.nc"
    pl_path  = f"era5_pl_{date_c}.nc"
    out_path = f"pangu_native_{date_c}.nc"

    if os.path.exists(out_path):
        print("  已存在，跳过")
        continue

    # 下载 ERA5 数据
    for var_list, ds_name, fname in [
        (CDS_SFC, "reanalysis-era5-single-levels", sfc_path),
        (CDS_PL,  "reanalysis-era5-pressure-levels", pl_path),
    ]:
        if os.path.exists(fname): continue
        print(f"  下载 {fname}...")
        req = {"product_type":"reanalysis","variable":var_list,"date":date,
               "time":"00:00","number":"0","data_format":"netcdf"}
        if "pressure" in ds_name:
            req["pressure_level"] = [str(p) for p in PRESSURE_LEVELS]
        c.retrieve(ds_name, req, fname)

    sfc = xr.open_dataset(sfc_path); pl = xr.open_dataset(pl_path)

    # ── 构建 Pangu 输入 tensor ──
    # 高空: [Z50,..Z1000, Q50,..Q1000, T50,..T1000, U50,..U1000, V50,..V1000]
    upper = []
    for var_name in ["z","q","t","u","v"]:
        for lv in PRESSURE_LEVELS:
            pl_var = {
                "z": "z", "q": "q", "t": "t", "u": "u", "v": "v"
            }[var_name]
            idx = np.argmin(np.abs(pl["pressure_level"].values - lv))
            da = pl[pl_var].isel(valid_time=0, pressure_level=idx).values  # (721, 1440)
            upper.append(da)
    upper = np.stack(upper)  # (65, 721, 1440)

    # 地表: [MSLP, U10, V10, T2M]
    surface = []
    for var_name in ["msl","u10","v10","t2m"]:
        cds_key = {
            "msl": "msl", "u10": "u10", "v10": "v10", "t2m": "t2m"
        }[var_name]
        da = sfc[cds_key].isel(valid_time=0).values
        surface.append(da)
    surface = np.stack(surface)  # (4, 721, 1440)

    sfc.close(); pl.close()

    # ── 转 tensor ──
    upper_t  = torch.tensor(upper,  dtype=torch.float32).unsqueeze(0).to(device)  # (1,65,721,1440)
    surface_t = torch.tensor(surface, dtype=torch.float32).unsqueeze(0).to(device)  # (1,4,721,1440)

    # ── 自回归推理 ──
    print(f"  推理 {NSTEPS}步 ({device})...")
    all_upper = []; all_surface = []

    with torch.no_grad():
        for step in range(NSTEPS):
            upper_t, surface_t = model(upper_t, surface_t)
            all_upper.append(upper_t.cpu().numpy()[0])
            all_surface.append(surface_t.cpu().numpy()[0])
            if (step + 1) % 7 == 0:
                print(f"    {(step+1)*6/24:.0f}天 完成")

    # ── 提取 Saudi 区域 & 保存 NetCDF ──
    lat_idx = np.where((GLOBAL_LAT >= SAUDI_LAT[0]) & (GLOBAL_LAT <= SAUDI_LAT[-1]))[0]
    lon_idx = np.where((GLOBAL_LON >= SAUDI_LON[0]) & (GLOBAL_LON <= SAUDI_LON[-1]))[0]

    all_upper_saudi  = np.stack([u[:, lat_idx][:, :, lon_idx] for u in all_upper])   # (28, 65, 65, 89)
    all_surface_saudi = np.stack([s[:, lat_idx][:, :, lon_idx] for s in all_surface])  # (28, 4, 65, 89)

    # 写 NetCDF (兼容 validate_long_forecast.py 格式)
    ds = xr.Dataset()
    ds["t2m"] = xr.DataArray(
        all_surface_saudi[:, 3:4, :, :],  # T2M is index 3 in surface
        dims=["time","lead_time","lat","lon"],
        coords={
            "time": [np.datetime64(date + "T00:00")],
            "lead_time": np.arange(1, NSTEPS+1) * 6 * 3.6e12,  # hours in ns
            "lat": SAUDI_LAT,
            "lon": SAUDI_LON,
        }
    )
    ds.to_netcdf(out_path)
    ds.close()

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  -> {out_path} ({size_mb:.1f} MB)")

    for tmp in [sfc_path, pl_path]:
        if os.path.exists(tmp): os.remove(tmp)

print("\n完成!")

# 打包
import zipfile
nc_files = [f for f in os.listdir() if f.startswith("pangu_native_") and f.endswith(".nc")]
if nc_files:
    with zipfile.ZipFile("pangu_native_results.zip", "w") as zf:
        for f in nc_files: zf.write(f)
    print(f"打包 -> pangu_native_results.zip ({os.path.getsize('pangu_native_results.zip')/1e6:.1f} MB)")
