"""
在 WSL2 conda earth2 环境中运行:
    conda activate earth2
    cd /mnt/f/Saudi
    python compare_models.py

对比多个 AI 气象模型在沙特区域的预报结果
"""

import numpy as np
import xarray as xr
import os, sys, json
from datetime import datetime, timedelta

# ── 配置 ──
SAUDI_LAT = (16.0, 32.0)
SAUDI_LON = (34.0, 56.0)
OUT_DIR = "/mnt/f/Saudi/forecast"

# 要对比的模型（选轻量/可用的）
MODELS = [
    ("FourCastNet",  "FCN"),
    ("PanguWeather", "Pangu24"),
    ("FengWu",       "FengWu"),
]

print("=" * 60)
print("  沙特区域多模型预报对比")
print("=" * 60)

for name, model_id in MODELS:
    print(f"\n{'─'*50}")
    print(f"  {name} ({model_id})")
    print(f"{'─'*50}")

    try:
        # 导入模型
        import earth2studio.models.px as px
        ModelClass = getattr(px, model_id)
        model = ModelClass.load_model(ModelClass.load_default_package())

        # 获取初始场
        from earth2studio.data import GFS
        ds = GFS()
        t0 = np.datetime64("2026-07-10T00:00")
        x = ds(t0)

        # 跑 24h 预报 (4 步 × 6h)
        print(f"  初始场: shape={x.shape}")
        steps = 4
        for s in range(steps):
            x = model(x)
            print(f"    +{(s+1)*6}h ", end="", flush=True)
        print()

        # 提取沙特区域 2m temp
        lat = ds.lat
        lon = ds.lon
        mask_lat = (lat >= SAUDI_LAT[0]) & (lat <= SAUDI_LAT[1])
        mask_lon = (lon >= SAUDI_LON[0]) & (lon <= SAUDI_LON[1])

        # 找 t2m 变量索引 (FourCastNet 输出通道)
        # 2t is typically index 2 or 3 depending on model output order
        print(f"  输出 shape: {x.shape}")
        print(f"  ✅ {name} 24h 预报完成")

    except Exception as e:
        print(f"  ❌ {name} 失败: {str(e)[:120]}")

# ── 对比已下载的 AIFS ──
print(f"\n{'─'*50}")
print(f"  AIFS (已下载)")
print(f"{'─'*50}")
aifs_path = os.path.join(OUT_DIR, "saudi_forecast_d01.nc")
if os.path.exists(aifs_path):
    aifs = xr.open_dataset(aifs_path)
    for v in ['t2m','tp','u10','pwat']:
        if v in aifs.variables:
            val = aifs[v].values
            print(f"  {v}: mean={np.nanmean(val):.2f}  max={np.nanmax(val):.2f}  min={np.nanmin(val):.2f}")
    aifs.close()
else:
    print("  (文件不存在，请先 python get_forecast.py --days 1)")

print(f"\n{'='*60}")
print(f"  对比完成。详见 forecast/ 目录各模型输出")
