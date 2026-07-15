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
    date_compact = date.replace("-", "")

    # ── 1. Surface single-level ──
    sfc_path = os.path.join(OUT_DIR, f"era5_surface_{date_compact}.nc")
    if os.path.exists(sfc_path):
        print(f"地表变量已存在，跳过: {sfc_path}")
    else:
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
            },
            sfc_path,
        )

    # ── 2. Pressure-level ──
    pl_path = os.path.join(OUT_DIR, f"era5_pressure_{date_compact}.nc")
    if os.path.exists(pl_path):
        print(f"气压层变量已存在，跳过: {pl_path}")
    else:
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
            },
            pl_path,
        )

    print(f"完成 → {OUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    download(args.date)
