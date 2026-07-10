"""
从 ECMWF Open Data 下载 AIFS 预报 → 裁剪沙特区域 → 输出 NetCDF。
变量名对齐 indicators/ 格式，直接兼容 datalayer.py。

使用:
    python get_forecast.py                  # 最新预报, 7 天
    python get_forecast.py --days 10         # 10 天预报
    python get_forecast.py --date 2026-07-20 # 指定日期 (需已发布)
"""

import os, sys, argparse
from datetime import datetime, timedelta
import xarray as xr
import numpy as np

SAUDI_BBOX_LAT = (16.0, 32.0)
SAUDI_BBOX_LON = (34.0, 56.0)
FORECAST_DIR = "forecast"

# AIFS 变量 → NetCDF 输出变量名（对齐 indicators/ 的命名）
# GRIB shortName → 输出变量名
VARIABLE_MAP = {
    "2t":   "t2m",      # 2m 气温 (K)
    "2d":   "d2m",      # 2m 露点 (K)
    "10u":  "u10",      # 10m 纬向风 (m/s)
    "10v":  "v10",      # 10m 经向风 (m/s)
    "tp":   "tp",       # 累计降水 (m) → 转换为 mm
    "sp":   "sp",       # 地面气压 (Pa)
    "tcc":  "tcc",      # 总云量 (0-1)
    "tcwv": "pwat",     # 柱水汽 (kg/m²)
    "ssr":  "ssr",      # 净短波辐射 (J/m² 累积)
    "str":  "str",      # 净长波辐射 (J/m² 累积)
}


def download_and_process(days: int = 7):
    """下载 AIFS 预报 → 裁剪沙特 → 转 NetCDF"""
    from ecmwf.opendata import Client
    import xarray as xr

    os.makedirs(FORECAST_DIR, exist_ok=True)
    client = Client(source="ecmwf")
    params = list(VARIABLE_MAP.keys())

    print(f"ECMWF AIFS 预报 → 沙特区域裁剪")
    print(f"  变量: {params}")
    print(f"  区域: {SAUDI_BBOX_LAT[0]}-{SAUDI_BBOX_LAT[1]}°N, {SAUDI_BBOX_LON[0]}-{SAUDI_BBOX_LON[1]}°E")
    print(f"  天数: {days}")
    print()

    for day in range(1, days + 1):
        grib_path = os.path.join(FORECAST_DIR, f"_tmp_d{day:02d}.grib2")
        nc_path   = os.path.join(FORECAST_DIR, f"saudi_forecast_d{day:02d}.nc")

        if os.path.exists(nc_path) and os.path.getsize(nc_path) > 500:
            ds = xr.open_dataset(nc_path)
            actual_vars = [v for v in ds.variables if v not in ('latitude','longitude','time')]
            ds.close()
            if len(actual_vars) >= 6:
                print(f"  Day {day}: 已存在 ({len(actual_vars)} vars)，跳过")
                continue

        # ── Step 1: 下载全球 GRIB2 ──
        print(f"  Day {day}: 下载 AIFS...")
        try:
            client.retrieve(
                step=day * 24,
                param=params,
                type="fc",
                target=grib_path,
            )
        except Exception as e:
            print(f"    下载失败: {str(e)[:100]}")
            continue

        # ── Step 2: 用 cfgrib 读取 → 裁剪 → 保存 NetCDF ──
        print(f"    裁剪沙特区域 + 转 NetCDF...")
        try:
            # cfgrib 一次只读一个变量类型（surface/pressure），分开读
            ds_list = []
            for param in params:
                try:
                    ds_var = xr.open_dataset(
                        grib_path, engine='cfgrib',
                        backend_kwargs={'filter_by_keys': {'shortName': param}}
                    )
                    ds_list.append(ds_var)
                except Exception:
                    pass  # 变量不在这个文件中，跳过

            if len(ds_list) < 3:
                print(f"    可读变量不足 ({len(ds_list)})，跳过")
                continue

            # 合并所有变量
            ds = xr.merge(ds_list, compat='override')

            # ── 裁剪到沙特区域 ──
            ds = ds.sel(
                latitude=slice(SAUDI_BBOX_LAT[1], SAUDI_BBOX_LAT[0]),   # 北→南
                longitude=slice(SAUDI_BBOX_LON[0], SAUDI_BBOX_LON[1]),  # 西→东
            )

            # ── 统一坐标名（对齐 indicators/ 的 latitude/longitude） ──
            if 'latitude' in ds.dims and 'longitude' in ds.dims:
                lat_vals = ds['latitude'].values
                lon_vals = ds['longitude'].values
                ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})

            # ── 单位转换（对齐 indicators/） ──
            rename_map = {}
            for src, dst in VARIABLE_MAP.items():
                if src in ds.variables:
                    if dst != src:
                        rename_map[src] = dst

            # tp: m → mm
            if 'tp' in ds.variables:
                ds['tp'] = ds['tp'] * 1000.0
                ds['tp'].attrs['units'] = 'mm'
                ds['tp'].attrs['long_name'] = 'Total precipitation'

            # t2m: K → degC (等 compute_indicators 做)
            if 't2m' in ds.variables:
                ds['t2m'].attrs['units'] = 'K'
                ds['t2m'].attrs['long_name'] = '2 metre temperature'

            ds = ds.rename(rename_map)
            ds.attrs['source'] = 'ECMWF AIFS Open Data'
            ds.attrs['forecast_day'] = day
            ds.attrs['bbox'] = f'{SAUDI_BBOX_LAT[0]}-{SAUDI_BBOX_LAT[1]}N_{SAUDI_BBOX_LON[0]}-{SAUDI_BBOX_LON[1]}E'

            # 保存
            ds.to_netcdf(nc_path)
            ds.close()

            # 统计
            var_count = len([v for v in ds if v not in ('lat','lon')]) if 'ds' in dir() else 0
            size_kb = os.path.getsize(nc_path) // 1024 if os.path.exists(nc_path) else 0
            print(f"    ✅ saudi_forecast_d{day:02d}.nc ({size_kb} KB, {params} vars)")
        except Exception as e:
            print(f"    处理失败: {str(e)[:150]}")
        finally:
            # 清理临时 GRIB
            if os.path.exists(grib_path):
                os.remove(grib_path)

    # 统计
    count = len([f for f in os.listdir(FORECAST_DIR) if f.endswith('.nc') and 'saudi_forecast' in f])
    print(f"\n完成。{count} 个预报 NetCDF → {FORECAST_DIR}/")
    if count > 0:
        print(f"下一步: python agent.py → '未来3天利雅得会有极端高温吗'")
    return count


def main():
    parser = argparse.ArgumentParser(description="ECMWF AIFS 沙特预报下载 + 裁剪")
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    download_and_process(days=args.days)


if __name__ == "__main__":
    main()
