"""
FourCastNet 沙特区域预报
在 WSL2 运行: conda activate earth2 && export HF_ENDPOINT=https://hf-mirror.com
    cd /mnt/f/Saudi && python run_fcn.py --days 7
"""

import numpy as np
import xarray as xr
import os, argparse
from collections import OrderedDict

SAUDI_LAT = (16.0, 32.0)
SAUDI_LON = (34.0, 56.0)
OUT_DIR = "/mnt/f/Saudi/forecast"


def run(days: int = 7):
    from earth2studio.models.px import FCN
    from earth2studio.data import GFS
    from earth2studio.run import deterministic
    from earth2studio.io import NetCDF4Backend

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. 模型 ──
    print("加载 FourCastNet...")
    model = FCN.load_model(FCN.load_default_package())
    print(f"  输入变量: {len(model.input_coords()['variable'])} 个")

    # ── 2. 数据 ──
    print("连接 GFS 数据源...")
    data = GFS()

    # ── 3. 输出坐标（仅沙特区域） ──
    out_coords = OrderedDict({
        "lat": np.arange(SAUDI_LAT[0], SAUDI_LAT[1] + 0.25, 0.25),
        "lon": np.arange(SAUDI_LON[0], SAUDI_LON[1] + 0.25, 0.25),
    })

    # ── 4. IO 后端（NetCDF） ──
    io = NetCDF4Backend(os.path.join(OUT_DIR, "fcn_forecast.nc"))

    # ── 5. 运行 ──
    nsteps = days * 4  # 4 steps/day
    t0 = np.datetime64("2026-07-10T00:00")
    times = [t0]

    print(f"\nFourCastNet 预报: {nsteps} 步 = {days} 天, 沙特区域")
    print(f"  GPU: RTX 4060")

    io = deterministic(
        time=times,
        nsteps=nsteps,
        prognostic=model,
        data=data,
        io=io,
        output_coords=out_coords,
    )

    print(f"\n完成。输出: {OUT_DIR}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    run(days=args.days)


if __name__ == "__main__":
    main()
