"""
从 ECMWF Open Data 下载 AIFS 沙特区域预报 NetCDF。
AIFS 预报在 ECMWF Open Data 上公开免费提供。

使用:
    python get_forecast.py                    # 最新可用预报, 7 天
    python get_forecast.py --days 10           # 10 天预报
"""

import os, sys, argparse
from datetime import datetime, timedelta
import xarray as xr
import numpy as np

SAUDI_BBOX = [32, 34, 16, 56]  # [北, 西, 南, 东]
FORECAST_DIR = "forecast"

# 常用表面变量（映射到你的 91 个指标需要的原始变量）
# AIFS Open Data 支持的表面变量 (shortName)
VARIABLES = [
    "2t",      # 2m temperature (K) → t2m_c
    "10u",     # 10m u-wind → wind10_speed
    "10v",     # 10m v-wind
    "2d",     # 2m dewpoint (K) → d2m_c
    "tp",     # total precipitation (m) → daily_precip_total
    "ssr",    # surface solar radiation → sw_net
    "str",    # surface thermal radiation → lw_net
    "sp",     # surface pressure → surface_pressure
    "tcc",    # total cloud cover
    "tcwv",   # total column water vapour → pwat
    # 以下变量 AIFS Open Data 可能不支持, 可用备选
    # "mucape" → CAPE (需验证)
    # "sshf" / "slhf" → 可能需要从其他数据集获取
]

# 压力层变量（850hPa）
PRESSURE_VARS = {
    "850": ["u", "v", "q", "w"],  # wind, humidity, omega
    "500": ["gh", "w"],          # geopotential height, omega
}


def download_aifs_open_data(date_str: str, lead_days: int):
    """从 ECMWF Open Data 下载 AIFS 预报并裁剪到沙特区域"""
    from ecmwf.opendata import Client

    os.makedirs(FORECAST_DIR, exist_ok=True)
    client = Client(source="ecmwf")

    # 最新可用数据
    print(f"ECMWF Open Data AIFS 预报")
    print(f"  区域: 16-32°N, 34-56°E (沙特)")
    print(f"  天数: {lead_days}")
    print()

    try:
        for day in range(1, lead_days + 1):
            out_path = os.path.join(FORECAST_DIR, f"saudi_aifs_d{day:02d}.grib2")
            if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
                print(f"  Day {day}: 已存在，跳过")
                continue

            print(f"  Day {day}: 下载 AIFS 预报...")
            try:
                # Open Data 下载所有 leadtime steps
                client.retrieve(
                    step=day * 24,  # 24h, 48h, 72h...
                    param=VARIABLES,
                    type="fc",
                    target=out_path,
                    area=SAUDI_BBOX,  # [北,西,南,东]
                )
                size_kb = os.path.getsize(out_path)//1024 if os.path.exists(out_path) else 0
                print(f"    OK: {out_path} ({size_kb} KB)")
            except Exception as e:
                print(f"    失败 Day {day}: {str(e)[:120]}")

    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("可能原因: ECMWF Open Data 暂未发布最新预报（有 24h 延迟）")
        print("手动下载: https://data.ecmwf.int/forecasts/")

    # 统计
    count = len([f for f in os.listdir(FORECAST_DIR) if f.endswith(('.nc','.grib2','.grib'))])
    print(f"\n完成。{count} 个预报文件 -> {FORECAST_DIR}/")
    if count > 0:
        print(f"运行: python agent.py → 输入: '未来3天利雅得会有极端高温吗'")
    return count


def main():
    parser = argparse.ArgumentParser(description="ECMWF Open Data AIFS 沙特预报下载")
    parser.add_argument("--days", type=int, default=7, help="预报天数，默认 7")
    args = parser.parse_args()

    download_aifs_open_data(
        date_str=datetime.now().strftime("%Y%m%d"),
        lead_days=args.days,
    )


if __name__ == "__main__":
    main()
