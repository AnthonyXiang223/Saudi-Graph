"""
沙尘暴联合概率计算 — Empirical Copula 方法

对每个格点的 365 天 ERA5 数据:
  1. 提取 4 个沙尘指标: wind10_speed, dewpoint_depression_c, rh2m, wind_shear_850_200
  2. 计算每个指标的经验 CDF (rank / (n+1)):
     - wind10_speed: 越大越极端 (右尾)
     - dewpoint_depression_c: 越大越极端 (右尾)
     - rh2m: 越小越极端, 取负号后 rank (右尾)
     - wind_shear_850_200: 越大越极端 (右尾)
  3. Joint prob = min(rank1, rank2, rank3, rank4)
     → Gumbel copula 逻辑: 只有 ALL FOUR 都极端时 joint 才极端
  4. 对 ground truth 沙尘事件日期验证联合概率

输出: forecast/dust_joint_climatology.nc

用法:
  python compute_dust_joint_prob.py
"""

import numpy as np
import xarray as xr
import os, sys, glob, time
from datetime import datetime
from netCDF4 import Dataset as nc_open


def main():
    t0 = time.time()
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    indicators_dir = os.path.join(project_dir, "indicators")
    forecast_dir = os.path.join(project_dir, "forecast")

    # ── 1. Find files ──
    nc_files = sorted(glob.glob(os.path.join(indicators_dir, "saudi_indicators_*.nc")))
    print(f"找到 {len(nc_files)} 天指标数据")
    ndays = len(nc_files)

    # ── 2. Grid metadata ──
    ds0 = xr.open_dataset(nc_files[0], decode_times=False)
    if "latitude" in ds0:
        lat = ds0["latitude"].values[:160].astype(np.float32)
        lon = ds0["longitude"].values[:220].astype(np.float32)
    else:
        lat = ds0["lat"].values[:160].astype(np.float32)
        lon = ds0["lon"].values[:220].astype(np.float32)
    ds0.close()

    nlat, nlon = len(lat), len(lon)
    ncells = nlat * nlon
    print(f"格点: {nlat} x {nlon} = {ncells}")

    # ── 3. Read 4 dust indicators for all 365 days ──
    INDICATORS = ["wind10_speed", "dewpoint_depression_c", "rh2m", "wind_shear_850_200"]
    # Direction: 1 = higher→extreme, -1 = lower→extreme (rank will be negated)
    DIRECTION = [1, 1, -1, 1]  # rh2m is negated because low humidity → extreme

    print(f"读取 {ndays} 天沙尘指标: {INDICATORS} ...")
    data = {ind: np.full((ndays, nlat, nlon), np.nan, dtype=np.float32) for ind in INDICATORS}
    missing = {ind: 0 for ind in INDICATORS}
    dates = []

    for day_idx, fpath in enumerate(nc_files):
        # Extract date from filename
        date_str = os.path.basename(fpath).replace("saudi_indicators_", "").replace(".nc", "")
        dates.append(date_str)

        try:
            nc = nc_open(fpath, "r")
            for ind in INDICATORS:
                if ind in nc.variables:
                    p = nc.variables[ind][:]
                    while p.ndim > 2:
                        p = p[0] if p.shape[0] == 1 else p[0]
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
        print(f"  {ind}: {ndays - missing[ind]}/{ndays} 天可用, "
              f"range [{np.nanmin(data[ind]):.2f}, {np.nanmax(data[ind]):.2f}]")

    # ── 4. Compute empirical CDF per cell → joint prob per day ──
    print("\n计算经验 Copula 联合概率...")
    joint_prob = np.full((ndays, nlat, nlon), np.nan, dtype=np.float32)

    for i in range(nlat):
        for j in range(nlon):
            # Get the 4 time series for this cell
            series = []
            for ind, direction in zip(INDICATORS, DIRECTION):
                s = data[ind][:, i, j]
                # Remove NaN
                valid = np.isfinite(s)
                if valid.sum() < 30:  # too few valid days
                    series.append(None)
                    continue

                # Directional adjustment: negate if higher→less extreme
                s_directed = s * direction  # now higher always = more extreme

                # Empirical CDF via ranking (0 to 1)
                ranks = np.full(len(s), np.nan)
                ranks[valid] = rankdata(s_directed[valid]) / (valid.sum() + 1)
                series.append(ranks)

            if any(s is None for s in series):
                # Not enough data for this cell
                continue

            # Gumbel copula: joint = min(rank1, rank2, rank3, rank4)
            # This means ALL four conditions must be simultaneously extreme
            # for the joint probability to be high
            joint = np.min(series, axis=0)
            joint_prob[:, i, j] = joint.astype(np.float32)

        if (i + 1) % 40 == 0:
            print(f"  row {i+1}/{nlat} ({time.time()-t0:.0f}s)", end="\r")

    print(f"\n  Copula 计算完成 ({time.time()-t0:.0f}s)")

    # ── 5. Ground truth verification ──
    print(f"\n{'='*70}")
    print(f"  Ground Truth 沙尘事件 — Copula 联合概率验证")
    print(f"{'='*70}")

    gt_events = [
        {"name": "May 4-5 Qassim Haboob", "dates": ["20250504", "20250505"],
         "region": "Qassim", "lat": 26.3, "lon": 44.0},
        {"name": "May 4-5 Riyadh", "dates": ["20250504", "20250505"],
         "region": "Riyadh", "lat": 24.7, "lon": 46.7},
        {"name": "May 16-19 Nationwide dust (N-Rafha)", "dates": ["20250516", "20250517", "20250518", "20250519"],
         "region": "Rafha", "lat": 29.6, "lon": 43.5},
        {"name": "May 16-19 Nationwide dust (Dammam)", "dates": ["20250516", "20250517", "20250518", "20250519"],
         "region": "Dammam", "lat": 26.4, "lon": 50.1},
        {"name": "Jun 30-Jul 5 East dust (Dammam)", "dates": ["20250630", "20250701", "20250702", "20250703", "20250704", "20250705"],
         "region": "Dammam", "lat": 26.4, "lon": 50.1},
        {"name": "Jun 30-Jul 5 East dust (Riyadh)", "dates": ["20250630", "20250701", "20250702", "20250703", "20250704", "20250705"],
         "region": "Riyadh", "lat": 24.7, "lon": 46.7},
    ]

    # Also check a few "quiet" dates for contrast
    quiet_dates = ["20250215", "20250920", "20251110"]  # Feb, Sep, Nov - quiet months

    print(f"{'Event':<40s} {'Date':>10s} {'Joint':>8s} {'Wnd':>7s} {'Dew':>7s} {'RH':>7s} {'Shr':>8s} {'Verdict':>10s}")
    print("-" * 105)

    for gt in gt_events:
        i = int(np.argmin(np.abs(lat - gt["lat"])))
        j = int(np.argmin(np.abs(lon - gt["lon"])))

        for date_str in gt["dates"]:
            day_idx = dates.index(date_str) if date_str in dates else -1
            if day_idx < 0:
                continue

            jp = joint_prob[day_idx, i, j]
            # Individual ranks
            ranks_str = ""
            for ind, direction in zip(INDICATORS, DIRECTION):
                s = data[ind][day_idx, i, j]
                # Quick rank estimate from log
                rank_val = np.nan
                if np.isfinite(s):
                    all_valid = data[ind][:, i, j]
                    all_valid = all_valid[np.isfinite(all_valid)]
                    if len(all_valid) >= 30:
                        s_dir = s * direction
                        all_dir = all_valid * direction
                        rank_val = (all_dir <= s_dir).sum() / (len(all_dir) + 1)
                ranks_str += f" {rank_val:.2f}" if np.isfinite(rank_val) else "   N/A"

            # Verdict
            if np.isfinite(jp):
                if jp > 0.90:
                    verdict = "EXTREME"
                elif jp > 0.75:
                    verdict = "HIGH"
                elif jp > 0.50:
                    verdict = "MODERATE"
                else:
                    verdict = "low"
            else:
                verdict = "NODATA"

            print(f"{gt['name']:<40s} {date_str:>10s} {jp:>7.3f} {ranks_str} {verdict:>10s}")

    # Quiet dates
    print(f"\n  Quiet period contrast:")
    for date_str in quiet_dates:
        if date_str not in dates:
            continue
        day_idx = dates.index(date_str)
        for name, clat, clon in [("Riyadh", 24.7, 46.7), ("Dammam", 26.4, 50.1), ("Qassim", 26.3, 44.0)]:
            i = int(np.argmin(np.abs(lat - clat)))
            j = int(np.argmin(np.abs(lon - clon)))
            jp = joint_prob[day_idx, i, j]
            print(f"  {date_str} {name:<10s}: joint={jp:.3f}")

    # ── 6. Save climatology ──
    # Compute percentile levels of joint_prob for reference
    print(f"\n  Joint prob 分布 (所有格点, 所有天):")
    jp_valid = joint_prob[np.isfinite(joint_prob)]
    for pct in [50, 75, 90, 95, 98, 99]:
        print(f"    P{pct}: {np.percentile(jp_valid, pct):.4f}")

    output_path = os.path.join(forecast_dir, "dust_joint_climatology.nc")
    print(f"\n保存: {output_path}")

    # Compute per-cell joint prob percentiles (for threshold reference)
    pct_levels = [50, 75, 90, 95, 98, 99]
    joint_pct = np.zeros((len(pct_levels), nlat, nlon), dtype=np.float32)
    for pi, pv in enumerate(pct_levels):
        joint_pct[pi] = np.nanpercentile(joint_prob, pv, axis=0)

    # Also compute per-indicator percentiles for on-the-fly rank estimation
    ind_pct_data = {}
    for ind in INDICATORS:
        ipct = np.zeros((len(pct_levels), nlat, nlon), dtype=np.float32)
        for pi, pv in enumerate(pct_levels):
            ipct[pi] = np.nanpercentile(data[ind], pv, axis=0)
        ind_pct_data[ind] = ipct

    ds_out = xr.Dataset(
        {
            "joint_prob_percentiles": (
                ["percentile", "lat", "lon"], joint_pct,
                {"description": "沙尘联合概率百分位 (Copula min-rank)",
                 "percentile_values": str(pct_levels)}),
            "joint_prob_mean": (["lat", "lon"],
                                np.nanmean(joint_prob, axis=0).astype(np.float32),
                                {"description": "年均联合概率"}),
            "joint_prob_max": (["lat", "lon"],
                               np.nanmax(joint_prob, axis=0).astype(np.float32),
                               {"description": "年最大联合概率"}),
            # Per-indicator percentiles for on-the-fly Copula computation
            "wind10_pct": (["percentile", "lat", "lon"], ind_pct_data["wind10_speed"],
                           {"description": "wind10_speed 百分位 (m/s)"}),
            "dewpoint_pct": (["percentile", "lat", "lon"], ind_pct_data["dewpoint_depression_c"],
                            {"description": "dewpoint_depression_c 百分位 (C)"}),
            "rh2m_pct": (["percentile", "lat", "lon"], ind_pct_data["rh2m"],
                         {"description": "rh2m 百分位 (%)"}),
            "shear_pct": (["percentile", "lat", "lon"], ind_pct_data["wind_shear_850_200"],
                          {"description": "wind_shear_850_200 百分位 (m/s)"}),
        },
        coords={
            "lat": (["lat"], lat, {"units": "degrees_north"}),
            "lon": (["lon"], lon, {"units": "degrees_east"}),
            "percentile": (["percentile"], np.array(pct_levels, dtype=np.int32)),
        },
        attrs={
            "description": "Saudi dust storm joint probability (Empirical Copula)",
            "method": "Gumbel copula via min-rank: P = min(F_wind, F_dewpoint, F_1/rh, F_shear)",
            "indicators": str(INDICATORS),
            "n_days": ndays,
            "created": datetime.now().isoformat(),
        }
    )
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars}
    ds_out.to_netcdf(output_path, encoding=encoding)
    ds_out.close()

    # ── 7. Summary ──
    print(f"\n{'='*70}")
    print(f"  沙尘联合概率计算完成 ({time.time()-t0:.0f}s)")
    print(f"{'='*70}")
    print(f"  方法: Empirical Copula (Gumbel)")
    print(f"  逻辑: P_dust = min(F_wind, F_dew, F_1/rh, F_shear)")
    print(f"  含义: 四个条件都极端时 joint 才极端")
    print(f"  输出: {output_path}")


def rankdata(arr):
    """rankdata without scipy dependency (for 1D arrays)."""
    n = len(arr)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    # Average ranks for ties
    # (simplified: just use order position, ties get sequential ranks)
    return ranks


if __name__ == "__main__":
    main()
