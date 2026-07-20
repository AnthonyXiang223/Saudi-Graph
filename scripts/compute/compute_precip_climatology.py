"""
降水格点气候态计算 — 基于 365 天 ERA5 再分析数据

对每个 0.1° 格点:
  1. 收集 365 天的 daily_precip_total
  2. 计算经验百分位 (P50, P75, P90, P95, P98, P99)
  3. 矩法拟合 Gamma 分布 (shape, scale)
  4. 保存为 forecast/precip_climatology.nc

输出变量:
  - precip_percentiles: (percentile, lat, lon) — [P50, P75, P90, P95, P98, P99]
  - gamma_shape, gamma_scale: (lat, lon)
  - precip_mean, precip_std, precip_max: (lat, lon)
  - n_rainy_days: (lat, lon)

用法:
  python compute_precip_climatology.py
"""

import numpy as np
import xarray as xr
import os, sys, glob, time
from datetime import datetime


def fit_gamma_mom(precip_series, min_rainy=15):
    """
    Method of moments Gamma fit (closed-form, no scipy optimization needed).
    Returns (shape, scale, quality):
      quality: 'mm' = method of moments, 'dry' = too few rainy days
    """
    valid = precip_series[np.isfinite(precip_series)]
    nonzero = valid[valid > 0.05]  # >0.05mm counts as rain

    if len(nonzero) < min_rainy:
        return (0.5, 1.0, 'dry')

    mu = max(float(np.mean(nonzero)), 0.01)
    var = float(np.var(nonzero))
    if var < 1e-6:
        return (1.0, mu, 'mm')

    shape = np.clip(mu ** 2 / var, 0.05, 100.0)
    scale = np.clip(var / mu, 0.01, 200.0)
    return (float(shape), float(scale), 'mm')


def main():
    t0 = time.time()
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    indicators_dir = os.path.join(project_dir, "indicators")
    forecast_dir = os.path.join(project_dir, "forecast")

    from netCDF4 import Dataset as nc_open

    # ── 1. Find files ──
    nc_files = sorted(glob.glob(os.path.join(indicators_dir, "saudi_indicators_*.nc")))
    if not nc_files:
        print("ERROR: 未找到指标文件")
        return
    print(f"找到 {len(nc_files)} 天指标数据")

    # ── 2. First file → grid metadata (xarray just once for coords) ──
    ds0 = xr.open_dataset(nc_files[0], decode_times=False)
    if "latitude" in ds0:
        lat_raw = ds0["latitude"].values
        lon_raw = ds0["longitude"].values
    else:
        lat_raw = ds0["lat"].values
        lon_raw = ds0["lon"].values
        if len(lat_raw) > 200:
            lat_raw = lat_raw[:160]
        if len(lon_raw) > 250:
            lon_raw = lon_raw[:220]
    nlat, nlon = len(lat_raw), len(lon_raw)
    lat = lat_raw[:nlat].astype(np.float32)
    lon = lon_raw[:nlon].astype(np.float32)
    ds0.close()

    ncells = nlat * nlon
    print(f"格点: {nlat} x {nlon} = {ncells}")

    # ── 3. Read all precip via raw netCDF4 (38x faster than xarray) ──
    ndays = len(nc_files)
    print(f"读取 {ndays} 天降水数据 (netCDF4 raw)...")
    precip = np.zeros((ndays, nlat, nlon), dtype=np.float32)
    skipped = 0

    for day_idx, fpath in enumerate(nc_files):
        try:
            nc = nc_open(fpath, "r")
            p = nc.variables["daily_precip_total"][:]
            nc.close()

            # Squeeze extra dims
            while p.ndim > 2:
                if p.shape[0] == 1:
                    p = p[0]
                else:
                    p = p[0]
            precip[day_idx] = p[:nlat, :nlon]

            if (day_idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  {day_idx+1}/{ndays} ({elapsed:.0f}s)", end="\r")
        except Exception as e:
            skipped += 1
            continue

    elapsed = time.time() - t0
    print(f"\n  完成: {ndays - skipped}/{ndays} 天 ({elapsed:.0f}s)")
    print(f"  降水范围: {np.nanmin(precip):.1f} - {np.nanmax(precip):.1f} mm")

    # ── 4. Compute percentiles (vectorized, fast) ──
    percentiles = [50, 75, 90, 95, 98, 99]
    print(f"计算百分位: {percentiles} ...")
    pct_data = np.zeros((len(percentiles), nlat, nlon), dtype=np.float32)
    for i, pv in enumerate(percentiles):
        pct_data[i] = np.percentile(precip, pv, axis=0)
        # Only show P95 for brevity
        if pv >= 90:
            print(f"  P{pv}: {np.min(pct_data[i]):.2f} - {np.max(pct_data[i]):.2f} mm")

    # Basic stats (vectorized)
    precip_mean = np.mean(precip, axis=0).astype(np.float32)
    precip_std = np.std(precip, axis=0).astype(np.float32)
    precip_max = np.max(precip, axis=0).astype(np.float32)
    n_rainy = (precip > 0.05).sum(axis=0).astype(np.int16)

    # ── 5. Gamma fit per cell (method of moments, fast) ──
    print("拟合 Gamma 分布 (矩法)...")
    gamma_shape = np.full((nlat, nlon), np.nan, dtype=np.float32)
    gamma_scale = np.full((nlat, nlon), np.nan, dtype=np.float32)
    n_mm, n_dry = 0, 0

    for i in range(nlat):
        for j in range(nlon):
            series = precip[:, i, j]
            sh, sc, qual = fit_gamma_mom(series)
            gamma_shape[i, j] = sh
            gamma_scale[i, j] = sc
            if qual == 'mm':
                n_mm += 1
            else:
                n_dry += 1
        if (i + 1) % 40 == 0:
            print(f"  row {i+1}/{nlat} ({((i+1)*nlon)} cells)", end="\r")
    print(f"\n  Done. fit={n_mm} dry={n_dry}")

    # ── 6. Save ──
    output_path = os.path.join(forecast_dir, "precip_climatology.nc")
    print(f"\n保存: {output_path}")

    ds_out = xr.Dataset(
        {
            "precip_percentiles": (
                ["percentile", "lat", "lon"], pct_data,
                {"description": "降水百分位 (mm/day)",
                 "percentile_values": str(percentiles)}),
            "gamma_shape": (["lat", "lon"], gamma_shape,
                            {"description": "Gamma shape (k)"}),
            "gamma_scale": (["lat", "lon"], gamma_scale,
                            {"description": "Gamma scale (theta, mm)"}),
            "precip_mean": (["lat", "lon"], precip_mean,
                            {"description": "日均降水 (mm)"}),
            "precip_std": (["lat", "lon"], precip_std,
                           {"description": "日降水标准差 (mm)"}),
            "precip_max": (["lat", "lon"], precip_max,
                           {"description": "年最大日降水 (mm)"}),
            "n_rainy_days": (["lat", "lon"], n_rainy,
                             {"description": "降水>0.05mm天数"}),
        },
        coords={
            "lat": (["lat"], lat, {"units": "degrees_north"}),
            "lon": (["lon"], lon, {"units": "degrees_east"}),
            "percentile": (["percentile"], np.array(percentiles, dtype=np.int32)),
        },
        attrs={
            "description": "Saudi precipitation climatology (2025 ERA5)",
            "n_days": ndays - skipped,
            "gamma_fit_method": "method_of_moments",
            "created": datetime.now().isoformat(),
        }
    )
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars}
    ds_out.to_netcdf(output_path, encoding=encoding)
    ds_out.close()

    # ── 7. Print key locations ──
    print(f"\n{'='*55}")
    print(f"  气候态计算完成  ({time.time() - t0:.0f}s)")
    print(f"{'='*55}")

    locations = {
        "吉达": (21.54, 39.17),
        "麦加": (21.39, 39.86),
        "利雅得": (24.71, 46.68),
        "达曼": (26.42, 50.10),
        "艾卜哈": (18.22, 42.51),
        "塔伊夫": (21.27, 40.42),
    }
    for name, (clat, clon) in locations.items():
        i = int(np.argmin(np.abs(lat - clat)))
        j = int(np.argmin(np.abs(lon - clon)))
        p95 = pct_data[3, i, j]
        p99 = pct_data[5, i, j]
        print(f"  {name} ({clat}N,{clon}E): "
              f"P95={p95:.1f}mm  P99={p99:.1f}mm  "
              f"max={precip_max[i,j]:.1f}mm  rainy={n_rainy[i,j]}d")

    print(f"\n  输出: {output_path}")


if __name__ == "__main__":
    main()
