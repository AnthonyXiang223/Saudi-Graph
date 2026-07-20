"""
Download ERA5 at forecast valid times (06Z/12Z/18Z) for same-hour validation.
Forecast init is always 00Z, so T+6h=06Z, T+12h=12Z, T+18h=18Z.
"""
import os, numpy as np, xarray as xr
import cdsapi

os.environ["CDSAPI_URL"] = "https://cds.climate.copernicus.eu/api"
os.environ["CDSAPI_KEY"] = "07f04bcc-987d-4b07-b9fd-22b3b3547eaf"

# ── Forecast init dates (Pangu + clean FCN) ──
# 每个初始日 00Z 启动, 需要验证的 ERA5 时刻: 06Z, 12Z, 18Z
INIT_DATES = [
    "2025-01-15",  # 冬
    "2025-04-15",  # 春
    "2025-07-01",  # 夏
    "2025-10-15",  # 秋
]

VALID_TIMES = ["06:00", "12:00", "18:00"]  # 对应 T+6h, T+12h, T+18h

# 只需要地表变量用于 t2m 验证
SURFACE_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
]

# Saudi 区域 (和预报网格一致)
# 预报网格: 16.0-32.25N, 34.0-56.25E, 0.25°
# 下载稍微扩大一点区域以便插值
AREA = [33, 33, 15, 57]  # [N, W, S, E] — CDS 格式

OUTPUT_DIR = "era5_validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

c = cdsapi.Client()

for init_date in INIT_DATES:
    # 计算该初始日的所有验证时刻
    # 当前日期
    date_no_dash = init_date.replace("-", "")

    for valid_time in VALID_TIMES:
        out_file = os.path.join(OUTPUT_DIR, f"era5_sfc_{date_no_dash}_{valid_time.replace(':', '')}Z.nc")

        if os.path.exists(out_file):
            print(f"[SKIP] {out_file} 已存在")
            continue

        print(f"[DOWNLOAD] ERA5 surface {init_date} {valid_time}Z → {out_file}")

        try:
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": SURFACE_VARS,
                    "date": init_date,
                    "time": valid_time,
                    "number": "0",
                    "data_format": "netcdf",
                    "area": AREA,  # 限制区域, 加速下载
                },
                out_file,
            )
            print("  [OK] done")
        except Exception as e:
            print(f"  [FAIL] {e}")

print(f"\n所有下载完成 → {OUTPUT_DIR}/")
