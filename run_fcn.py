"""
FourCastNet 沙特区域预报（自动获取最新 GFS 分析场）
在 WSL2 运行: conda activate earth2 && export HF_ENDPOINT=https://hf-mirror.com
    cd /mnt/f/Saudi && python run_fcn.py --days 7
每日自动: 见 run_fcn_daily.sh
"""

import numpy as np
import xarray as xr
import os, argparse, datetime
from collections import OrderedDict

SAUDI_LAT = (16.0, 32.0)
SAUDI_LON = (34.0, 56.0)
OUT_DIR = "/mnt/f/Saudi/forecast"


def run(days: int = 7, init_time: str = None):
    from earth2studio.models.px import FCN
    from earth2studio.data import GFS
    from earth2studio.run import deterministic
    from earth2studio.io import NetCDF4Backend

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. 数据源与初始化时间 ──
    print("连接 GFS 数据源...")
    data = GFS()

    # 尝试获取最新可用 GFS 时次（中国访问 AWS 可能失败）
    t0 = None
    try:
        available = sorted(data.available_times())
        if available:
            t0 = available[-1]
            print(f"最新 GFS 时次: {t0}")
    except Exception as e:
        print(f"获取 GFS 时次列表失败: {e}")

    if t0 is None:
        # 降级策略：用户指定的时间 → 今天 00Z → 已知可用的时间
        if init_time:
            t0 = np.datetime64(init_time)
        else:
            today = datetime.date.today()
            t0 = np.datetime64(today.isoformat() + "T00:00")
        print(f"使用初始化时间: {t0}（需确保该时次 GFS 数据在 AWS 上可用）")
        print(f"  如果报 FileNotFoundError，尝试: python run_fcn.py --init 2026-07-10")

    # ── 2. 模型 ──
    print("加载 FourCastNet...")
    model = FCN.load_model(FCN.load_default_package())
    print(f"  输入变量: {len(model.input_coords()['variable'])} 个")

    # ── 3. 输出坐标（仅沙特区域，0.25°） ──
    out_coords = OrderedDict({
        "lat": np.arange(SAUDI_LAT[0], SAUDI_LAT[1] + 0.25, 0.25),
        "lon": np.arange(SAUDI_LON[0], SAUDI_LON[1] + 0.25, 0.25),
    })

    # ── 5. IO ──
    out_path = os.path.join(OUT_DIR, "fcn_forecast.nc")
    io = NetCDF4Backend(out_path)

    # ── 6. 运行 ──
    nsteps = days * 4
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

    # 标记初始化时间到文件属性
    ds = xr.open_dataset(out_path)
    ds.attrs["fcn_init_time"] = str(t0)
    ds.attrs["fcn_run_time"] = datetime.datetime.now().isoformat()
    ds.close()

    print(f"\n完成。输出: {out_path}")
    print(f"下次更新: 明天运行 python run_fcn.py --days 7")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7, help="预报天数（默认7天）")
    parser.add_argument("--init", type=str, default=None, help="初始化时间 YYYY-MM-DD（默认今天）")
    args = parser.parse_args()
    run(days=args.days, init_time=args.init)


if __name__ == "__main__":
    main()
