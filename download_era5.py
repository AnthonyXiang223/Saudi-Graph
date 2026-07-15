"""
CDS API 下载 ERA5 数据供 FCN 使用
用法: python download_era5.py --date 2025-07-01
需要: CDS API key (~/.cdsapirc), pip install cdsapi
"""

import cdsapi
import argparse, os

# FCN 需要的 26 个变量
SURFACE_VARS = [
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "2m_temperature", "surface_pressure", "mean_sea_level_pressure",
    "total_column_water_vapour",
]

PRESSURE_LEVELS = [50, 250, 500, 850, 1000]  # hPa
PRESSURE_VARS = [
    "temperature",           # T at all levels
    "u_component_of_wind",   # U at all levels
    "v_component_of_wind",   # V at all levels
    "geopotential",          # Z at all levels
    "relative_humidity",     # R at all levels
]

OUT_DIR = "era5_data"


def download(date: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    c = cdsapi.Client()

    # ── 1. Surface single-level ──
    print(f"下载地表变量: {date}")
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": SURFACE_VARS,
            "date": date,
            "time": "00:00",
            "number": "0",
            "data_format": "netcdf",
            "area": [45, 20, 5, 65],  # N, W, S, E — 中东区域（约全球 1/6）
        },
        os.path.join(OUT_DIR, f"era5_surface_{date.replace('-','')}.nc"),
    )

    # ── 2. Pressure-level ──
    print(f"下载气压层变量: {date}")
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": PRESSURE_VARS,
            "pressure_level": [str(p) for p in PRESSURE_LEVELS],
            "date": date,
            "time": "00:00",
            "number": "0",
            "data_format": "netcdf",
            "area": [32, 34, 16, 56],
        },
        os.path.join(OUT_DIR, f"era5_pressure_{date.replace('-','')}.nc"),
    )

    print(f"完成 → {OUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    download(args.date)
