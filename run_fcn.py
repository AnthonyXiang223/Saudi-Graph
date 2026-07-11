"""
FourCastNet 沙特区域预报 — 完整替换 AIFS
在 WSL2 中运行:
    conda activate earth2
    export HF_ENDPOINT=https://hf-mirror.com
    cd /mnt/f/Saudi
    python run_fcn.py --days 7
"""

import numpy as np
import xarray as xr
import os, sys, argparse
from datetime import datetime, timedelta

SAUDI_LAT = (16.0, 32.0)
SAUDI_LON = (34.0, 56.0)
OUT_DIR = "/mnt/f/Saudi/forecast"


def run_fcn_forecast(days: int = 7):
    from earth2studio.models.px import FCN
    from earth2studio.data import GFS

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 加载模型 ──
    print("加载 FourCastNet (287M params)...")
    model = FCN.load_model(FCN.load_default_package())
    fcn_vars = list(model.input_coords()["variable"])
    print(f"  输入变量: {len(fcn_vars)} 个")
    print(f"  {fcn_vars}")

    # ── 获取初始场 ──
    print("\n获取 GFS 初始场 (0.25 deg)...")
    ds = GFS()
    t0 = np.datetime64("2026-07-10T00:00")

    # 对 FCN 变量名做 GFS 映射
    gfs_vars = []
    for v in fcn_vars:
        if v in ["u10m","v10m","t2m","sp","msl","tcwv"]:
            gfs_vars.append(v)
        elif v.startswith("u") and v[1:].isdigit():
            gfs_vars.append("u")  # GFS has u on pressure levels
        elif v.startswith("v") and v[1:].isdigit():
            gfs_vars.append("v")
        elif v.startswith("t") and v[1:].isdigit():
            gfs_vars.append("t")
        elif v.startswith("z") and v[1:].isdigit():
            gfs_vars.append("z")
        elif v.startswith("r") and v[1:].isdigit():
            gfs_vars.append("r")
        else:
            gfs_vars.append(v)

    # 去重
    gfs_vars = list(dict.fromkeys(gfs_vars))
    print(f"  GFS 变量: {gfs_vars}")

    x = ds(t0, gfs_vars)
    print(f"  初始场 shape: {x.shape}")

    # ── 归一化 ──
    # center/scale are (1, 26, 1, 1) tensors — index by variable position
    fcn_vars = list(model.input_coords()["variable"])
    center_vals = model.center[0, :, 0, 0]  # (26,)
    scale_vals = model.scale[0, :, 0, 0]    # (26,)
    channel_names = list(x.coords["variable"].values)
    for i, ch in enumerate(channel_names):
        if ch in fcn_vars:
            j = fcn_vars.index(ch)
            x.values[:, i, :, :] = (x.values[:, i, :, :] - float(center_vals[j])) / float(scale_vals[j])

    # ── 自回归预报 ──
    n_steps = days * 4  # 4 steps/day (6h each)
    print(f"\nFourCastNet 自回归预报 ({n_steps} 步 = {days} 天)...")

    lat = x.coords["lat"].values
    lon = x.coords["lon"].values
    mlat = (lat >= SAUDI_LAT[0]) & (lat <= SAUDI_LAT[1])
    mlon = (lon >= SAUDI_LON[0]) & (lon <= SAUDI_LON[1])
    out_var_list = list(model.output_coords(model.input_coords())["variable"])

    # Convert to tensor + coords format
    x_tensor = x.values  # numpy
    coords = {k: x.coords[k].values for k in x.coords}

    from collections import OrderedDict
    import torch
    for step in range(1, n_steps + 1):
        co = OrderedDict((k, coords[k]) for k in coords)
        x_tensor, coords = model(torch.from_numpy(x_tensor).float().cuda(), co)
        x_tensor = x_tensor.detach().cpu().numpy()
        coords = {k: v for k, v in coords.items()}
        if step % 10 == 0 or step == 1:
            print(f"  +{step*6}h", end=" ", flush=True)

        if step % 4 == 0:  # 每 24h 保存
            day = step // 4
            vals = x_tensor  # numpy: (1, N_vars, 721, 1440)

            out_vars = {}
            for v in ["t2m", "u10m", "v10m", "tcwv", "msl", "sp"]:
                if v in out_var_list:
                    idx = out_var_list.index(v)
                    raw = vals[0, idx, :, :]
                    if v in fcn_vars:
                        j = fcn_vars.index(v)
                        raw = raw * float(scale_vals[j]) + float(center_vals[j])
                    out_vars[v] = raw[mlat][:, mlon]

            out_ds = xr.Dataset(
                {k: (["lat", "lon"], v) for k, v in out_vars.items()},
                coords={"lat": coords["lat"][mlat], "lon": coords["lon"][mlon]}
            )
            out_ds.attrs["source"] = "FourCastNet (NVIDIA Earth-2)"
            out_ds.attrs["forecast_day"] = day
            out_ds.attrs["init_time"] = str(t0)

            path = os.path.join(OUT_DIR, f"fcn_forecast_d{day:02d}.nc")
            out_ds.to_netcdf(path)
            kb = os.path.getsize(path)//1024
            print(f"\n  Day {day}: {kb}KB, vars={list(out_vars.keys())}")

    print(f"\n完成。{days} 天预报 -> {OUT_DIR}/fcn_forecast_d*.nc")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    run_fcn_forecast(days=args.days)


if __name__ == "__main__":
    main()
