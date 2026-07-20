"""
沿海湿热联合概率 — Empirical Copula (Gumbel)

对沿海格点(红海+波斯湾)的 365 天 ERA5 数据:
  指标 + 方向 (ALL 都需要同时极端才触发):
    - sst_celsius:       越高越极端 (右尾) → rank
    - rh2m:              越高越极端 (右尾) → rank
    - t2m_c:             越高越极端 (右尾) → rank
    - wind10_speed:      越低越极端 (左尾) → negate → rank

  Joint prob = min(F_sst, F_rh, F_t2m, F_1/wind)
  只有四条件都极端(高温+高湿+低风+暖海水)时 joint 才高

输出: forecast/humid_heat_joint_climatology.nc

用法:
  python compute_humid_heat_joint_prob.py
"""

import numpy as np
import xarray as xr
import os, sys, glob, time
from datetime import datetime
from netCDF4 import Dataset as nc_open


def rankdata_1d(arr):
    """Compute empirical CDF ranks for 1D array (0 to 1)."""
    n = len(arr)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    return ranks / (n + 1)


def is_coastal(lat, lon):
    """Check if a grid cell is in coastal region (Red Sea or Persian Gulf)."""
    # Red Sea coast: 16-30N, 34-44E (coastal + offshore strip)
    # Persian Gulf coast: 24-30N, 48-56E
    in_red_sea = (16 <= lat <= 30) and (34 <= lon <= 42)
    in_gulf = (24 <= lat <= 30) and (48 <= lon <= 56)
    return in_red_sea or in_gulf


def main():
    t0 = time.time()
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    indicators_dir = os.path.join(project_dir, "indicators")
    forecast_dir = os.path.join(project_dir, "forecast")

    # ── 1. Find files ──
    nc_files = sorted(glob.glob(os.path.join(indicators_dir, "saudi_indicators_*.nc")))
    ndays = len(nc_files)
    print(f"找到 {ndays} 天指标数据")

    # ── 2. Grid ──
    nc0 = nc_open(nc_files[0], "r")
    if "latitude" in nc0.variables:
        lat = nc0.variables["latitude"][:160].astype(np.float32)
        lon = nc0.variables["longitude"][:220].astype(np.float32)
    else:
        lat = nc0.variables["lat"][:160].astype(np.float32)
        lon = nc0.variables["lon"][:220].astype(np.float32)
    nc0.close()
    nlat, nlon = len(lat), len(lon)
    print(f"格点: {nlat} x {nlon} = {nlat*nlon}")

    # Identify coastal cells
    coastal_mask = np.zeros((nlat, nlon), dtype=bool)
    for i in range(nlat):
        for j in range(nlon):
            coastal_mask[i, j] = is_coastal(lat[i], lon[j])
    n_coastal = int(coastal_mask.sum())
    print(f"沿海格点: {n_coastal}/{nlat*nlon} ({n_coastal/(nlat*nlon)*100:.1f}%)")

    # ── 3. Read 4 indicators ──
    INDICATORS = ["sst_celsius", "rh2m", "t2m_c", "wind10_speed"]
    # direction: 1=higher→extreme, -1=lower→extreme
    DIRECTION = [1, 1, 1, -1]

    print(f"读取 {ndays} 天湿热指标: {INDICATORS} ...")
    data = {ind: np.full((ndays, nlat, nlon), np.nan, dtype=np.float32) for ind in INDICATORS}
    missing = {ind: 0 for ind in INDICATORS}
    dates = []

    for day_idx, fpath in enumerate(nc_files):
        date_str = os.path.basename(fpath).replace("saudi_indicators_", "").replace(".nc", "")
        dates.append(date_str)
        try:
            nc = nc_open(fpath, "r")
            for ind in INDICATORS:
                if ind in nc.variables:
                    p = nc.variables[ind][:]
                    while p.ndim > 2:
                        p = p[0] if p.shape[0] == 1 else p.mean(axis=0)
                    data[ind][day_idx] = p[:nlat, :nlon]
                else:
                    missing[ind] += 1
            nc.close()
        except Exception:
            for ind in INDICATORS:
                missing[ind] += 1
            continue
        if (day_idx + 1) % 100 == 0:
            print(f"  {day_idx+1}/{ndays} ({time.time()-t0:.0f}s)", end="\r")

    elapsed = time.time() - t0
    print(f"\n  读取完成 ({elapsed:.0f}s)")
    for ind in INDICATORS:
        coastal_vals = data[ind][:, coastal_mask]
        valid = coastal_vals[np.isfinite(coastal_vals)]
        print(f"  {ind}: {ndays - missing[ind]}/{ndays} 天, "
              f"沿海 range [{np.nanmin(data[ind][:, coastal_mask]):.2f}, "
              f"{np.nanmax(data[ind][:, coastal_mask]):.2f}]")

    # ── 4. Empirical Copula (coastal cells only) ──
    print("\n计算椰海湿热 Copula 联合概率...")
    joint_prob = np.full((ndays, nlat, nlon), np.nan, dtype=np.float32)

    for i in range(nlat):
        for j in range(nlon):
            if not coastal_mask[i, j]:
                continue

            ranks_list = []
            for ind, direction in zip(INDICATORS, DIRECTION):
                s = data[ind][:, i, j]
                valid = np.isfinite(s)
                if valid.sum() < 30:
                    ranks_list.append(None)
                    break

                s_dir = s * direction  # higher = more extreme
                r = np.full(len(s), np.nan)
                r[valid] = rankdata_1d(s_dir[valid])
                ranks_list.append(r)

            if any(r is None for r in ranks_list):
                continue

            # Gumbel copula: ALL must be extreme
            joint = np.min(ranks_list, axis=0)
            joint_prob[:, i, j] = joint.astype(np.float32)

        if (i + 1) % 40 == 0:
            print(f"  row {i+1}/{nlat} ({time.time()-t0:.0f}s)", end="\r")

    print(f"\n  Copula 完成 ({time.time()-t0:.0f}s)")

    # ── 5. Key locations check ──
    print(f"\n{'='*70}")
    print(f"  沿海湿热 Copula 联合概率 — 关键地点")
    print(f"{'='*70}")

    locations = {
        "Jeddah": (21.54, 39.17), "Dammam": (26.42, 50.10),
        "Jubail": (26.96, 49.57), "Yanbu": (24.09, 38.06),
    }
    check_dates = ["20250630", "20250715", "20250801", "20250115"]

    print(f"{'Location':<12s}", end="")
    for d in check_dates:
        print(f"{d:>12s}", end="")
    print(f"{' PeakDate':>12s} {'MaxVal':>8s}")
    print("-" * 75)

    for name, (clat, clon) in locations.items():
        i = int(np.argmin(np.abs(lat - clat)))
        j = int(np.argmin(np.abs(lon - clon)))
        print(f"{name:<12s}", end="")
        max_val = 0
        max_date = ""
        for d in check_dates:
            if d in dates:
                jp = joint_prob[dates.index(d), i, j]
                print(f"{jp:>11.3f}", end=" ") if np.isfinite(jp) else print(f"{'N/A':>12s}", end="")
            else:
                print(f"{' -':>12s}", end="")
        # Find peak joint prob date
        cell_jp = joint_prob[:, i, j]
        valid = np.isfinite(cell_jp)
        if valid.any():
            max_idx = np.nanargmax(cell_jp)
            max_val = cell_jp[max_idx]
            max_date = dates[max_idx]
        print(f" {max_date:>12s} {max_val:>7.3f}")

    # Overall distribution
    jp_valid = joint_prob[np.isfinite(joint_prob) & (joint_prob > 0)]
    print(f"\n  沿海湿热 Joint prob 分布:")
    for pct in [50, 75, 90, 95, 98, 99]:
        print(f"    P{pct}: {np.percentile(jp_valid, pct):.4f}")

    # ── 6. Save ──
    percentiles = [50, 75, 90, 95, 98, 99]
    joint_pct = np.zeros((len(percentiles), nlat, nlon), dtype=np.float32)
    for pi, pv in enumerate(percentiles):
        joint_pct[pi] = np.nanpercentile(joint_prob, pv, axis=0)

    # Per-indicator percentiles for on-the-fly computation
    ind_pct_data = {}
    for ind in INDICATORS:
        ipct = np.zeros((len(percentiles), nlat, nlon), dtype=np.float32)
        for pi, pv in enumerate(percentiles):
            ipct[pi] = np.nanpercentile(data[ind], pv, axis=0)
        ind_pct_data[ind] = ipct

    output_path = os.path.join(forecast_dir, "humid_heat_joint_climatology.nc")
    print(f"\n保存: {output_path}")

    ds_out = xr.Dataset(
        {
            "joint_prob_percentiles": (
                ["percentile", "lat", "lon"], joint_pct,
                {"description": "湿热联合概率百分位", "percentiles": str(percentiles)}),
            "joint_prob_mean": (["lat", "lon"],
                                np.nanmean(joint_prob, axis=0).astype(np.float32)),
            "joint_prob_max": (["lat", "lon"],
                               np.nanmax(joint_prob, axis=0).astype(np.float32)),
            "sst_pct": (["percentile", "lat", "lon"], ind_pct_data["sst_celsius"],
                        {"description": "SST 百分位 (°C)"}),
            "rh2m_pct": (["percentile", "lat", "lon"], ind_pct_data["rh2m"],
                         {"description": "rh2m 百分位 (%)"}),
            "t2m_pct": (["percentile", "lat", "lon"], ind_pct_data["t2m_c"],
                        {"description": "t2m 百分位 (°C)"}),
            "wind10_pct": (["percentile", "lat", "lon"], ind_pct_data["wind10_speed"],
                           {"description": "wind10_speed 百分位 (m/s, negated)"}),
            "coastal_mask": (["lat", "lon"], coastal_mask.astype(np.int8),
                             {"description": "1=coastal (Red Sea + Persian Gulf)"}),
        },
        coords={
            "lat": (["lat"], lat, {"units": "degrees_north"}),
            "lon": (["lon"], lon, {"units": "degrees_east"}),
            "percentile": (["percentile"], np.array(percentiles, dtype=np.int32)),
        },
        attrs={
            "description": "Saudi coastal humid heat joint probability (Copula)",
            "method": "Gumbel copula: P = min(F_sst, F_rh, F_t2m, F_1/wind)",
            "indicators": str(INDICATORS),
            "n_days": ndays,
            "n_coastal_cells": n_coastal,
            "created": datetime.now().isoformat(),
        }
    )
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars}
    ds_out.to_netcdf(output_path, encoding=encoding)
    ds_out.close()

    print(f"\n{'='*70}")
    print(f"  沿海湿热 Copula 完成 ({time.time()-t0:.0f}s)")
    print(f"{'='*70}")
    print(f"  方法: Empirical Copula (Gumbel)")
    print(f"  逻辑: P = min(F_sst, F_rh, F_t2m, F_1/wind)")
    print(f"  沿海格点: {n_coastal}")
    print(f"  输出: {output_path}")


if __name__ == "__main__":
    main()
