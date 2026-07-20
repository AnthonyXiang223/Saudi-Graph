"""
极端高温 GPD (Generalized Pareto Distribution) 计算

方法: Peaks-Over-Threshold (POT)
  - 阈值 = P90 (每个格点)
  - 对所有超过阈值的天拟合 GPD
  - 365 天即可稳定拟合（每个格点 ~36 天超过 P90）

GPD CDF: F(x|μ,σ,ξ) = 1 - (1 + ξ(x-μ)/σ)^(-1/ξ)
  - μ = 阈值 (P90)
  - σ = 尺度参数
  - ξ = 形状参数 (ξ>0 厚尾, ξ=0 指数尾, ξ<0 有界尾)

沙特的物理约束:
  - 极端高温应有厚尾 (ξ>0): 46-50°C 比正态推断的更常见
  - 沙漠格点系统性低估 2-4°C
  - 沿海格点受红海/波斯湾调节，尾部分布偏薄 (ξ<0)

输出: forecast/heat_gpd_climatology.nc

用法:
  python compute_heat_gpd.py
"""

import numpy as np
import xarray as xr
import os, sys, glob, time
from datetime import datetime
from netCDF4 import Dataset as nc_open
from scipy.stats import genpareto
import warnings
warnings.filterwarnings("ignore")


def fit_gpd_per_cell(tmax_series, threshold_pct=90):
    """
    Fit GPD to a single grid cell's daily maximum temperature series.

    POT approach:
    1. Empirical threshold = P90 of this cell's tmax
    2. Extract exceedances (days where tmax > threshold)
    3. Fit GPD to exceedances
    4. Return GPD parameters + empirical non-exceedance fraction

    Returns:
        (threshold, shape, scale, exceedance_rate, fit_quality, n_exceed, n_total)
    """
    valid = tmax_series[np.isfinite(tmax_series)]
    n_total = len(valid)

    if n_total < 100:
        return (0, 0, 1, 0, 'insufficient', 0, n_total)

    # Empirical threshold: P90
    threshold = float(np.percentile(valid, threshold_pct))
    if threshold < 30:  # Sanity check: summer temps in Saudi should be >30C
        threshold = 35.0  # fallback for unusual cells

    # Extract exceedances
    exceedances = valid[valid > threshold] - threshold
    n_exceed = len(exceedances)

    if n_exceed < 10:
        # Too few exceedances: use P95 as threshold instead
        threshold = float(np.percentile(valid, 95))
        exceedances = valid[valid > threshold] - threshold
        n_exceed = len(exceedances)

    if n_exceed < 5:
        return (threshold, 0, 1, 0, 'too_few', n_exceed, n_total)

    # Exceedance rate
    exceedance_rate = n_exceed / n_total

    # Fit GPD via MLE
    try:
        shape, loc, scale = genpareto.fit(exceedances, floc=0)
        if not np.isfinite(shape) or not np.isfinite(scale):
            raise ValueError("MLE failed")

        # Physical constraints for temperature in Saudi
        # ξ should be > -0.5 (otherwise MLE properties break)
        # ξ > 0.5 means extremely thick tail (very rare for temperature)
        shape = float(np.clip(shape, -0.4, 0.5))
        scale = float(np.clip(scale, 0.1, 20.0))

        quality = 'mle'
    except Exception:
        # Method of Moments fallback
        if len(exceedances) >= 5:
            mean_ex = np.mean(exceedances)
            var_ex = np.var(exceedances)
            if var_ex > 1e-6 and mean_ex > 0.1:
                shape_mom = 0.5 * ((mean_ex**2 / var_ex) - 1)
                scale_mom = 0.5 * mean_ex * ((mean_ex**2 / var_ex) + 1)
                shape = float(np.clip(shape_mom, -0.4, 0.5))
                scale = float(np.clip(scale_mom, 0.1, 20.0))
                quality = 'mom'
            else:
                return (threshold, 0, 1, exceedance_rate, 'constant', n_exceed, n_total)
        else:
            return (threshold, 0, 1, exceedance_rate, 'too_few', n_exceed, n_total)

    return (threshold, shape, scale, exceedance_rate, quality, n_exceed, n_total)


def gpd_probability(today_tmax, threshold, shape, scale, exceedance_rate):
    """
    Compute probability that tmax exceeds a given value using GPD.

    P(T > x) for x > threshold:
      P(T > x) = exceedance_rate × (1 + ξ(x-μ)/σ)^(-1/ξ)

    Returns: probability in [0, 1] or NaN if x <= threshold
    """
    if not np.isfinite(today_tmax) or threshold <= 0 or scale <= 0:
        return np.nan

    if today_tmax <= threshold:
        # Below threshold: return empirical CDF
        return 1.0 - exceedance_rate  # non-exceedance probability for below-threshold

    excess = today_tmax - threshold
    if excess <= 0:
        return 1.0 - exceedance_rate

    if abs(shape) < 0.001:
        # Exponential tail (ξ ≈ 0)
        prob = exceedance_rate * np.exp(-excess / scale)
    else:
        arg = 1.0 + shape * excess / scale
        if arg <= 0:
            # Upper bound reached (ξ < 0 case: finite upper endpoint)
            return 0.0
        prob = exceedance_rate * arg**(-1.0 / shape)

    return float(np.clip(prob, 0.0, 1.0))


def main():
    t0 = time.time()
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    indicators_dir = os.path.join(project_dir, "indicators")
    forecast_dir = os.path.join(project_dir, "forecast")

    # ── 1. Find files ──
    nc_files = sorted(glob.glob(os.path.join(indicators_dir, "saudi_indicators_*.nc")))
    ndays = len(nc_files)
    print(f"找到 {ndays} 天指标数据")

    # ── 2. Grid metadata ──
    nc0 = nc_open(nc_files[0], "r")
    if "latitude" in nc0.variables:
        lat = nc0.variables["latitude"][:160].astype(np.float32)
        lon = nc0.variables["longitude"][:220].astype(np.float32)
    else:
        lat = nc0.variables["lat"][:160].astype(np.float32)
        lon = nc0.variables["lon"][:220].astype(np.float32)
    nc0.close()
    nlat, nlon = len(lat), len(lon)
    ncells = nlat * nlon
    print(f"格点: {nlat} x {nlon} = {ncells}")

    # ── 3. Read all tmax ──
    print(f"读取 {ndays} 天 tmax_c...")
    tmax_all = np.zeros((ndays, nlat, nlon), dtype=np.float32)
    dates = []
    skipped = 0

    for day_idx, fpath in enumerate(nc_files):
        date_str = os.path.basename(fpath).replace("saudi_indicators_", "").replace(".nc", "")
        dates.append(date_str)
        try:
            nc = nc_open(fpath, "r")
            if "tmax_c" in nc.variables:
                p = nc.variables["tmax_c"][:]
            elif "t2m" in nc.variables:
                p = nc.variables["t2m"][:] - 273.15  # Convert K to C
            else:
                skipped += 1
                nc.close()
                continue
            nc.close()
            while p.ndim > 2:
                p = p[0] if p.shape[0] == 1 else p.max(axis=0)  # daily max
            tmax_all[day_idx] = p[:nlat, :nlon]
        except Exception:
            skipped += 1
            continue

        if (day_idx + 1) % 100 == 0:
            print(f"  {day_idx+1}/{ndays} ({time.time()-t0:.0f}s)", end="\r")

    print(f"\n  有效天数: {ndays - skipped}/{ndays}")
    print(f"  tmax 范围: {np.nanmin(tmax_all):.1f} - {np.nanmax(tmax_all):.1f} °C")

    # ── 4. Fit GPD per cell ──
    print("\n拟合 GPD (每格点 Peaks-Over-Threshold)...")
    gpd_threshold = np.full((nlat, nlon), np.nan, dtype=np.float32)
    gpd_shape = np.full((nlat, nlon), np.nan, dtype=np.float32)
    gpd_scale = np.full((nlat, nlon), np.nan, dtype=np.float32)
    gpd_exceed_rate = np.full((nlat, nlon), np.nan, dtype=np.float32)
    gpd_quality = np.full((nlat, nlon), "", dtype=object)
    gpd_n_exceed = np.full((nlat, nlon), 0, dtype=np.int16)

    quality_counts = {"mle": 0, "mom": 0, "too_few": 0, "insufficient": 0, "constant": 0}

    for i in range(nlat):
        for j in range(nlon):
            series = tmax_all[:, i, j]
            th, sh, sc, er, q, ne, nt = fit_gpd_per_cell(series)
            gpd_threshold[i, j] = th
            gpd_shape[i, j] = sh
            gpd_scale[i, j] = sc
            gpd_exceed_rate[i, j] = er
            gpd_quality[i, j] = q
            gpd_n_exceed[i, j] = ne
            quality_counts[q] += 1
        if (i + 1) % 40 == 0:
            print(f"  row {i+1}/{nlat} mle={quality_counts['mle']} "
                  f"mom={quality_counts['mom']} too_few={quality_counts['too_few']}", end="\r")

    print(f"\n  Done. mle={quality_counts['mle']} mom={quality_counts['mom']} "
          f"too_few={quality_counts['too_few']} insufficient={quality_counts['insufficient']}")

    # ── 5. Compute GPD probability for selected days (ground truth verification) ──
    print(f"\n{'='*70}")
    print(f"  Ground Truth — 极端高温 GPD 概率验证")
    print(f"{'='*70}")

    # Known extreme heat days from ground truth
    # Jun 30-Jul 5: East + Hijaz dust with concurrent extreme heat
    # Also check some regular summer days for contrast
    check_events = [
        {"name": "Jun 30 Riyadh (dust+heat)", "date": "20250630", "lat": 24.71, "lon": 46.68},
        {"name": "Jul 1 Riyadh (heat peak)",   "date": "20250701", "lat": 24.71, "lon": 46.68},
        {"name": "Jul 15 Riyadh (midsummer)",  "date": "20250715", "lat": 24.71, "lon": 46.68},
        {"name": "Aug 1 Riyadh (peak summer)", "date": "20250801", "lat": 24.71, "lon": 46.68},
        {"name": "Jun 30 Dammam (coastal)",    "date": "20250630", "lat": 26.42, "lon": 50.10},
        {"name": "Jul 1 Dammam",               "date": "20250701", "lat": 26.42, "lon": 50.10},
        {"name": "Jan 15 Riyadh (winter)",     "date": "20250115", "lat": 24.71, "lon": 46.68},
        {"name": "Apr 15 Riyadh (spring)",     "date": "20250415", "lat": 24.71, "lon": 46.68},
    ]

    print(f"{'Event':<32s} {'Date':>10s} {'Tmax':>6s} {'Thresh':>7s} {'Shape':>7s} "
          f"{'Scale':>7s} {'Ex.Rate':>8s} {'GPD P':>7s} {'Assessment':>14s}")
    print("-" * 110)

    for ev in check_events:
        i = int(np.argmin(np.abs(lat - ev["lat"])))
        j = int(np.argmin(np.abs(lon - ev["lon"])))

        # Get today's tmax
        day_idx = dates.index(ev["date"]) if ev["date"] in dates else -1
        if day_idx < 0:
            continue

        today_tmax = float(tmax_all[day_idx, i, j])
        th = float(gpd_threshold[i, j])
        sh = float(gpd_shape[i, j])
        sc = float(gpd_scale[i, j])
        er = float(gpd_exceed_rate[i, j])

        # GPD exceedance probability
        gpd_p = gpd_probability(today_tmax, th, sh, sc, er)

        if np.isfinite(gpd_p):
            if gpd_p < 0.01:
                assess = "EXTREME (>P99)"
            elif gpd_p < 0.05:
                assess = "VERY HIGH (>P95)"
            elif gpd_p < 0.10:
                assess = "HIGH (>P90)"
            elif gpd_p > 0.50:
                assess = "normal"
            else:
                assess = "moderate"
        else:
            assess = "N/A"

        print(f"{ev['name']:<32s} {ev['date']:>10s} {today_tmax:>5.1f}°C {th:>6.1f}°C "
              f"{sh:>7.3f} {sc:>6.2f} {er:>7.3f}  {gpd_p:>6.4f} {assess:>14s}")

    # ── 6. Per-cell summary stats ──
    print(f"\n  GPD shape 参数分布 (ξ):")
    sh_valid = gpd_shape[np.isfinite(gpd_shape) & (gpd_shape > -0.4)]
    for pct in [10, 25, 50, 75, 90]:
        print(f"    P{pct}: {np.percentile(sh_valid, pct):.4f}")
    n_thick_tail = int((sh_valid > 0.05).sum())
    n_thin_tail = int((sh_valid < -0.05).sum())
    n_exp_tail = int((abs(sh_valid) <= 0.05).sum())
    print(f"    厚尾(ξ>0.05): {n_thick_tail} 格点 — 极端值比正态更频繁")
    print(f"    指数尾(|ξ|≤0.05): {n_exp_tail} 格点")
    print(f"    薄尾(ξ<-0.05): {n_thin_tail} 格点 — 温度有实际上限(沿海)")

    # ── 7. Save ──
    output_path = os.path.join(forecast_dir, "heat_gpd_climatology.nc")
    print(f"\n保存: {output_path}")

    ds_out = xr.Dataset(
        {
            "gpd_threshold": (["lat", "lon"], gpd_threshold,
                              {"description": "GPD threshold (P90 tmax, °C)"}),
            "gpd_shape": (["lat", "lon"], gpd_shape,
                          {"description": "GPD shape parameter (ξ)"}),
            "gpd_scale": (["lat", "lon"], gpd_scale,
                          {"description": "GPD scale parameter (σ, °C)"}),
            "exceedance_rate": (["lat", "lon"], gpd_exceed_rate,
                                {"description": "Fraction of days exceeding threshold"}),
            "n_exceedances": (["lat", "lon"], gpd_n_exceed,
                              {"description": "Number of exceedance days"}),
            "tmax_mean": (["lat", "lon"],
                          np.nanmean(tmax_all, axis=0).astype(np.float32),
                          {"description": "日均最高温 (°C)"}),
            "tmax_max": (["lat", "lon"],
                         np.nanmax(tmax_all, axis=0).astype(np.float32),
                         {"description": "年最高温 (°C)"}),
        },
        coords={
            "lat": (["lat"], lat, {"units": "degrees_north"}),
            "lon": (["lon"], lon, {"units": "degrees_east"}),
        },
        attrs={
            "description": "Saudi extreme heat GPD (Peaks-Over-Threshold)",
            "method": "GPD with P90 threshold, MLE fitting",
            "n_days": ndays - skipped,
            "created": datetime.now().isoformat(),
        }
    )
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars}
    ds_out.to_netcdf(output_path, encoding=encoding)
    ds_out.close()

    # ── 8. Key locations summary ──
    print(f"\n{'='*70}")
    print(f"  极端高温 GPD 计算完成 ({time.time()-t0:.0f}s)")
    print(f"{'='*70}")
    locations = {
        "利雅得": (24.71, 46.68), "吉达": (21.54, 39.17),
        "达曼": (26.42, 50.10), "鲁布哈利": (20.0, 50.0),
    }
    for name, (clat, clon) in locations.items():
        i = int(np.argmin(np.abs(lat - clat)))
        j = int(np.argmin(np.abs(lon - clon)))
        print(f"  {name}: thresh={gpd_threshold[i,j]:.1f}°C ξ={gpd_shape[i,j]:.3f} "
              f"σ={gpd_scale[i,j]:.2f} max={np.nanmax(tmax_all[:,i,j]):.1f}°C "
              f"n_exceed={gpd_n_exceed[i,j]}")


if __name__ == "__main__":
    main()
