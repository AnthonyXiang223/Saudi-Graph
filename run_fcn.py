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


def run(days: int = 7, init_time: str = None, source: str = "gfs"):
    """Run FCN forecast.

    Args:
        days: forecast length in days
        init_time: initialization date YYYY-MM-DD (default: latest available)
        source: data source — "gfs" (real-time, any date) or "era5" (low bias, ~5-day latency)
    """
    from earth2studio.models.px import FCN
    from earth2studio.run import deterministic
    from earth2studio.io import NetCDF4Backend

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. 数据源 ──
    if source == "era5":
        from earth2studio.data import NCAR_ERA5
        import s3fs

        # Patch: NCAR ERA5 bucket is in us-west-2, s3fs defaults to us-east-1
        # Bare s3fs works but earth2studio doesn't set region → 302 redirect timeout
        _orig_read = NCAR_ERA5._read_s3_dataset
        @staticmethod
        def _patched_read(nc_file_uri, time, variable, **kw):
            fs = s3fs.S3FileSystem(anon=True, asynchronous=False,
                                   client_kwargs={"region_name": "us-west-2"})
            import xarray as xr
            with fs.open(nc_file_uri, "rb", block_size=4 * 1400 * 720) as f:
                ds = xr.open_dataset(f, engine="h5netcdf", chunks={})
                ds = ds.sel(time=time)
                da = ds[variable].isel(time=slice(0, kw.get("lead_time", 0) + 1))
                return da.values
        NCAR_ERA5._read_s3_dataset = _patched_read

        print("数据源: ERA5 (ECMWF 再分析 — FCN 训练数据同源，偏差最小)")
        data = NCAR_ERA5()
    else:
        from earth2studio.data import GFS
        print("数据源: GFS (NOAA 全球预报 — 实时可用，但与 ERA5 存在系统性差异)")
        data = GFS()

    # 候选时间列表 — ERA5 不需要回退（CDS 数据必定存在），GFS 需要
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    day_before = today - datetime.timedelta(days=2)

    candidates = []
    if init_time:
        candidates.append(np.datetime64(init_time))
    if source == "era5":
        # ERA5 只需指定日期，CDS 上历史数据始终可用
        if not candidates:
            candidates.append(np.datetime64(yesterday.isoformat() + "T00:00"))
    else:
        for date in [yesterday, day_before]:
            for hour in ["18", "12", "06", "00"]:
                candidates.append(np.datetime64(date.isoformat() + f"T{hour}:00"))
        candidates.append(np.datetime64("2026-07-10T00:00"))

    t0 = candidates[0]
    print(f"尝试时次: {t0}（共 {len(candidates)} 个候选）")

    # ── 2. 模型 ──
    print("加载 FourCastNet...")
    model = FCN.load_model(FCN.load_default_package())
    print(f"  输入变量: {len(model.input_coords()['variable'])} 个")

    # ── 3. 输出坐标（仅沙特区域，0.25°） ──
    out_coords = OrderedDict({
        "lat": np.arange(SAUDI_LAT[0], SAUDI_LAT[1] + 0.25, 0.25),
        "lon": np.arange(SAUDI_LON[0], SAUDI_LON[1] + 0.25, 0.25),
    })

    # ── 5. IO（按初始化日期命名，避免互相覆盖） ──
    init_date_str = str(t0)[:10].replace("-", "")
    out_path = os.path.join(OUT_DIR, f"fcn_{init_date_str}.nc")
    if os.path.exists(out_path):
        os.remove(out_path)
    # 同时更新 latest 符号链接，agent 始终读取最新文件
    latest_path = os.path.join(OUT_DIR, "fcn_forecast.nc")
    if os.path.exists(latest_path):
        os.remove(latest_path)
    io = NetCDF4Backend(out_path)

    print(f"输出: {os.path.basename(out_path)}")

    # ── 6. 运行（自动回退到可用时次） ──
    nsteps = days * 4

    for i, t_candidate in enumerate(candidates):
        times = [t_candidate]
        try:
            print(f"\nFourCastNet 预报 [{i+1}/{len(candidates)}]: "
                  f"{nsteps} 步 = {days} 天, 初始化 {t_candidate}")
            print(f"  GPU: RTX 4060")

            io = deterministic(
                time=times,
                nsteps=nsteps,
                prognostic=model,
                data=data,
                io=io,
                output_coords=out_coords,
            )
            t0 = t_candidate  # success
            print(f"成功！使用 GFS 时次: {t0}")
            break

        except FileNotFoundError:
            if i + 1 < len(candidates):
                print(f"  GFS 数据 {t_candidate} 不可用，尝试下一候选...")
                # Re-create IO backend (it was partially written)
                import earth2studio.io
                io = earth2studio.io.NetCDF4Backend(out_path)
            else:
                raise
        except Exception as e:
            if "NoSuchKey" in str(e) or "does not exist" in str(e):
                if i + 1 < len(candidates):
                    print(f"  GFS {t_candidate} 不存在于 AWS，尝试下一候选...")
                    import earth2studio.io
                    io = earth2studio.io.NetCDF4Backend(out_path)
                else:
                    raise
            else:
                raise

    # 标记初始化时间到文件属性
    ds = xr.open_dataset(out_path)
    ds.attrs["fcn_init_time"] = str(t0)
    ds.attrs["fcn_run_time"] = datetime.datetime.now().isoformat()
    ds.close()

    # 同时更新 latest 副本，Agent 始终读 fcn_forecast.nc
    import shutil
    shutil.copy2(out_path, latest_path)

    print(f"\n完成。输出: {out_path} (→ {os.path.basename(latest_path)})")
    print(f"下次更新: python run_fcn.py --days 7")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7, help="预报天数（默认7天）")
    parser.add_argument("--init", type=str, default=None, help="初始化时间 YYYY-MM-DD")
    parser.add_argument("--source", type=str, default="gfs", choices=["gfs", "era5"],
                       help="数据源: gfs(实时可用) / era5(低偏差,需CDS API,延迟约5天)")
    args = parser.parse_args()

    if args.source == "era5":
        cdsrc = os.path.expanduser("~/.cdsapirc")
        if not os.path.exists(cdsrc):
            print("错误: ERA5 需要 CDS API key")
            print(f"  创建 {cdsrc} 并填入:")
            print("  url: https://cds.climate.copernicus.eu/api")
            print("  key: <your-uid>:<your-api-key>")
            return

    run(days=args.days, init_time=args.init, source=args.source)


if __name__ == "__main__":
    main()
