"""
FourCastNet 沙特区域预报 — 与 AIFS 对比
在 WSL2 conda earth2 环境中运行:
    conda activate earth2
    python run_fourcastnet.py
"""

import numpy as np
import xarray as xr
import os, sys

FORECAST_DIR = "/mnt/f/Saudi/forecast"

print("=" * 55)
print("  FourCastNet 沙特预报 vs AIFS 对比")
print("=" * 55)

# ── Step 1: 下载 FourCastNet 模型 + 初始场 ──
print("\n1. 加载 FourCastNet 预训练模型...")
from earth2studio.models.px import FCN
from earth2studio.data import GFS

model = FCN.load_model(FCN.load_default_package())
print("   模型加载完成")

# ── Step 2: 获取初始场数据 (GFS 分析场, 0.25°) ──
print("\n2. 获取 GFS 初始场数据...")
data_source = GFS()
t0 = np.datetime64("2026-07-10T00:00")
x = data_source(t0)

# GFS 变量: [tcwv, u10m, v10m, t2m, msl, ...]
# FourCastNet 需要特定变量组合
print(f"   初始场 shape: {x.shape}")
print(f"   变量: {data_source.variables[:10]}...")

# ── Step 3: 跑 24h 预报 ──
print("\n3. FourCastNet 24h 预报中...")
# 自回归: 每步 6h，4 步 = 24h
n_steps = 4
for step in range(n_steps):
    x = model(x)
    print(f"   步 {step+1}/{n_steps}: +{(step+1)*6}h")

print("   预报完成")

# ── Step 4: 提取沙特区域 ──
print("\n4. 裁剪沙特区域 (16-32°N, 34-56°E)...")
# 从 GFS 坐标中裁剪
lat = data_source.lat
lon = data_source.lon
saudi_mask = (lat >= 16) & (lat <= 32) & (lon >= 34) & (lon <= 56)

# ── Step 5: 读取 AIFS 预报做对比 ──
print("\n5. 对比 AIFS 预报...")
aifs_path = os.path.join(FORECAST_DIR, "saudi_forecast_d01.nc")

if os.path.exists(aifs_path):
    aifs = xr.open_dataset(aifs_path)

    # 对比 2m 温度
    if 't2m' in aifs.variables:
        aifs_t2m = aifs['t2m'].values  # K
        print(f"\n  {'─'*40}")
        print(f"  2m 气温 (K)")
        print(f"  AIFS:       mean={np.nanmean(aifs_t2m):.1f}  max={np.nanmax(aifs_t2m):.1f}")
        print(f"  FourCastNet: mean 需要在 WSL 端计算")
        print(f"  差异将通过第二步脚本量化")

    aifs.close()

print(f"\n✅ FourCastNet 推理完成。")
print(f"下一步: 在 WSL2 中运行对比脚本做精细量化")
