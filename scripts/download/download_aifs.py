"""
Download ECMWF IFS operational forecasts from AWS S3 (free, no key)
IFS = same model system as ERA5 → minimal systematic bias
→ Extract Saudi region → Compute ALL hazard indicators
"""
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "cfgrib", "xarray", "netcdf4", "scipy"])

import os, numpy as np, xarray as xr
from datetime import datetime, timedelta
import requests, json

# ── Config ──
S3_BASE = "https://ecmwf-forecasts.s3.amazonaws.com"
GRID_LAT = np.arange(16.0, 32.25, 0.25)
GRID_LON = np.arange(34.0, 56.25, 0.25)

# 验证: 4季各1天 × 多预报步
DATES = ["2025-01-15", "2025-04-15", "2025-07-01", "2025-10-15"]
STEPS  = [0, 12, 24]  # 分析场 + 正午 + 全天累积 (3步×120MB×4天=1.4GB)

OUTPUT_DIR = "aifs_forecasts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Download ──
def download_ifs_step(date_str, hour_str, step_h, target_dir):
    """Download one IFS GRIB2 file from AWS S3."""
    date_no_dash = date_str.replace("-", "")
    filename = f"{date_no_dash}{hour_str}0000-{step_h}h-oper-fc.grib2"
    url = f"{S3_BASE}/{date_no_dash}/{hour_str}z/ifs/0p25/oper/{filename}"
    local = os.path.join(target_dir, filename)

    if os.path.exists(local) and os.path.getsize(local) > 1000:
        return local

    print(f"    Downloading {filename}...")
    r = requests.get(url, stream=True, timeout=600)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    dl_mb = 0
    with open(local, 'wb') as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
            dl_mb += len(chunk)
    mb = os.path.getsize(local) / 1e6
    print(f"      {mb:.1f} MB done")
    return local

# ── 2. Extract Saudi region ──
def extract_saudi(grib_path, step_h=0):
    """Extract Saudi sub-region from IFS GRIB2 via multi-pass cfgrib.
    All variables interpolated to common 0.25deg Saudi grid.
    step_h: forecast step hour (0=analysis, 6/12/18/24=forecast)."""
    import os as _os

    # GRIB shortName → our variable name + unit conversion
    var_map = {
        "2t":   ("t2m",   lambda x: x - 273.15),
        "2d":   ("d2m",   lambda x: x - 273.15),
        "10u":  ("u10",   lambda x: x),
        "10v":  ("v10",   lambda x: x),
        "msl":  ("msl",   lambda x: x / 100.0),
        "sp":   ("sp",    lambda x: x / 100.0),
        "tcwv": ("tcwv",  lambda x: x),
        "tp":   ("tp",    lambda x: x * 1000.0),
        "skt":  ("skt",   lambda x: x - 273.15),  # skin temp → SST proxy over ocean
        "lsm":  ("lsm",   lambda x: x),            # land-sea mask (0=sea, 1=land)
        "mucape": ("cape", lambda x: x),           # most unstable CAPE
    }

    # Multi-pass: different typeOfLevel need separate cfgrib reads
    filters = [
        {"typeOfLevel": "surface", "step": step_h},
        {"typeOfLevel": "heightAboveGround", "level": 2, "step": step_h},
        {"typeOfLevel": "heightAboveGround", "level": 10, "step": step_h},
        {"typeOfLevel": "meanSea", "step": step_h},
        {"typeOfLevel": "entireAtmosphere", "step": step_h},
        {"typeOfLevel": "mostUnstableParcel", "step": step_h},
    ]

    result = {}
    for filt in filters:
        try:
            idx_path = grib_path + '.5b7b6.idx'
            if _os.path.exists(idx_path): _os.remove(idx_path)

            ds = xr.open_dataset(grib_path, engine="cfgrib",
                                 backend_kwargs={"filter_by_keys": filt, "errors": "ignore"})
            src_lat = ds["latitude"].values; src_lon = ds["longitude"].values

            for v in ds.data_vars:
                short = ds[v].attrs.get("GRIB_shortName", v)
                if short not in var_map:
                    continue
                our_name, convert = var_map[short]
                arr = ds[v].values
                if arr.ndim >= 2:
                    # Interpolate to target Saudi grid
                    if arr.ndim == 3:
                        arr = arr[0]
                    da = xr.DataArray(arr, dims=["lat","lon"],
                                      coords={"lat": src_lat, "lon": src_lon})
                    da_saudi = da.interp(lat=GRID_LAT, lon=GRID_LON, method="linear")
                    result[our_name] = convert(da_saudi.values).astype(np.float64)
            ds.close()
        except Exception:
            pass

    # ── Post-process: derive SST from skt over ocean ──
    if "skt" in result and "lsm" in result:
        ocean_mask = result["lsm"] < 0.5
        result["sst"] = result["skt"].copy()
        result["sst"][~ocean_mask] = np.nan
        # Clean up intermediate vars
        del result["lsm"]
        del result["skt"]

    return result, GRID_LAT.copy(), GRID_LON.copy()

# ── 3. Derive hazard indicators (matching operators.json) ──
def compute_indicators(v):
    """Compute all indicators needed by rules.json from raw IFS variables."""
    ind = {}
    # Raw pass-through
    for key in ["t2m", "tmax", "tmin", "d2m", "u10", "v10", "msl", "sp",
                "tcwv", "tp", "cp", "cape", "sst", "tcc"]:
        if key in v:
            ind[key] = v[key]

    # wind10_speed
    if "u10" in v and "v10" in v:
        ind["wind10_speed"] = np.sqrt(v["u10"]**2 + v["v10"]**2)

    # rh2m (Magnus formula)
    if "t2m" in v and "d2m" in v:
        es = 6.112 * np.exp(17.67 * v["t2m"] / (v["t2m"] + 243.5))
        e  = 6.112 * np.exp(17.67 * v["d2m"] / (v["d2m"] + 243.5))
        ind["rh2m"] = np.clip(100.0 * e / np.maximum(es, 0.001), 0, 100)

    # dewpoint_depression_c
    if "t2m" in v and "d2m" in v:
        ind["dewpoint_depression_c"] = v["t2m"] - v["d2m"]

    # vpd_kpa
    if "t2m" in v and "rh2m" in ind:
        es = 6.112 * np.exp(17.67 * v["t2m"] / (v["t2m"] + 243.5))
        ea = es * ind["rh2m"] / 100.0
        ind["vpd_kpa"] = np.maximum(0, (es - ea) / 10.0)

    # daily_precip_total
    if "tp" in v:
        ind["daily_precip_total"] = np.maximum(0, v["tp"])  # mm

    # Heat index (Rothfusz)
    if "t2m" in v and "rh2m" in ind:
        t = v["t2m"]; rh = ind["rh2m"]
        hi = t.copy()
        # Simple heat index: if T > 27°C and RH > 40%
        mask = (t > 27) & (rh > 40)
        hi[mask] = -8.7847 + 1.6114*t[mask] + 2.3385*rh[mask] \
                   - 0.1461*t[mask]*rh[mask] - 0.0123*t[mask]**2 \
                   - 0.0164*rh[mask]**2 + 0.00221*t[mask]**2*rh[mask] \
                   + 0.000725*t[mask]*rh[mask]**2 - 0.00000358*t[mask]**2*rh[mask]**2
        ind["heat_index_c"] = hi

    return ind

# ── 4. Probability gating (对接气候态文件) ──
def _interp_clim_to_grid(clim_arr, clim_lat, clim_lon, tgt_lat, tgt_lon):
    """Interpolate climatology (0.1deg) to target grid. Handles 2D and 3D arrays."""
    import numpy as _np
    if clim_arr.ndim == 3:
        result = []
        for i in range(clim_arr.shape[0]):
            da = xr.DataArray(clim_arr[i], dims=["lat","lon"],
                              coords={"lat": clim_lat, "lon": clim_lon})
            result.append(da.interp(lat=tgt_lat, lon=tgt_lon, method="linear").values)
        return _np.stack(result)
    else:
        da = xr.DataArray(clim_arr, dims=["lat","lon"],
                          coords={"lat": clim_lat, "lon": clim_lon})
        return da.interp(lat=tgt_lat, lon=tgt_lon, method="linear").values

def load_climatologies(tgt_lat, tgt_lon):
    """Load all 4 climatology files, interpolated to target grid."""
    from scipy.stats import gamma as gamma_dist
    cl = {}

    # 1. precip_climatology (Gamma 分布参数)
    ds = xr.open_dataset("forecast/precip_climatology.nc")
    cl["precip_gamma_shape"] = _interp_clim_to_grid(
        ds["gamma_shape"].values, ds["lat"].values, ds["lon"].values, tgt_lat, tgt_lon)
    cl["precip_gamma_scale"] = _interp_clim_to_grid(
        ds["gamma_scale"].values, ds["lat"].values, ds["lon"].values, tgt_lat, tgt_lon)
    ds.close()

    # 2. heat_gpd (GPD 极值分布参数)
    ds = xr.open_dataset("forecast/heat_gpd_climatology.nc")
    cl["gpd_threshold"] = _interp_clim_to_grid(
        ds["gpd_threshold"].values, ds["lat"].values, ds["lon"].values, tgt_lat, tgt_lon)
    cl["gpd_shape"] = _interp_clim_to_grid(
        ds["gpd_shape"].values, ds["lat"].values, ds["lon"].values, tgt_lat, tgt_lon)
    cl["gpd_scale"] = _interp_clim_to_grid(
        ds["gpd_scale"].values, ds["lat"].values, ds["lon"].values, tgt_lat, tgt_lon)
    cl["gpd_exc_rate"] = _interp_clim_to_grid(
        ds["exceedance_rate"].values, ds["lat"].values, ds["lon"].values, tgt_lat, tgt_lon)
    ds.close()

    # 3. dust_joint (Copula: 边际百分位表)
    ds = xr.open_dataset("forecast/dust_joint_climatology.nc")
    cl["dust_pct_vals"] = ds["percentile"].values  # the percentile levels
    cl["wind10_pct"] = _interp_clim_to_grid(
        ds["wind10_pct"].values, ds["lat"].values, ds["lon"].values, tgt_lat, tgt_lon)
    cl["dewpoint_pct"] = _interp_clim_to_grid(
        ds["dewpoint_pct"].values, ds["lat"].values, ds["lon"].values, tgt_lat, tgt_lon)
    cl["rh2m_pct"] = _interp_clim_to_grid(
        ds["rh2m_pct"].values, ds["lat"].values, ds["lon"].values, tgt_lat, tgt_lon)
    cl["shear_pct"] = _interp_clim_to_grid(
        ds["shear_pct"].values, ds["lat"].values, ds["lon"].values, tgt_lat, tgt_lon)
    ds.close()

    # 4. humid_heat_joint (Copula)
    ds = xr.open_dataset("forecast/humid_heat_joint_climatology.nc")
    cl["humid_pct_vals"] = ds["percentile"].values
    cl["h_sst_pct"] = _interp_clim_to_grid(
        ds["sst_pct"].values, ds["lat"].values, ds["lon"].values, tgt_lat, tgt_lon)
    cl["h_rh2m_pct"] = _interp_clim_to_grid(
        ds["rh2m_pct"].values, ds["lat"].values, ds["lon"].values, tgt_lat, tgt_lon)
    cl["h_t2m_pct"] = _interp_clim_to_grid(
        ds["t2m_pct"].values, ds["lat"].values, ds["lon"].values, tgt_lat, tgt_lon)
    cl["h_wind10_pct"] = _interp_clim_to_grid(
        ds["wind10_pct"].values, ds["lat"].values, ds["lon"].values, tgt_lat, tgt_lon)
    ds.close()

    return cl

def _lookup_percentile(value, pct_vals, pct_table):
    """给定值, 查百分位表得到经验CDF值 [0,1]。
    pct_table: (n_pct, nlat, nlon), pct_vals: (n_pct,)"""
    from scipy.interpolate import interp1d
    n_pct, ny, nx = pct_table.shape
    result = np.zeros((ny, nx), dtype=np.float64)
    for i in range(ny):
        for j in range(nx):
            pct_curve = pct_table[:, i, j]
            if np.all(np.isfinite(pct_curve)) and np.any(np.diff(pct_curve) > 0):
                f = interp1d(pct_curve, pct_vals, bounds_error=False,
                             fill_value=(pct_vals[0], pct_vals[-1]))
                result[i, j] = f(value[i, j])
            else:
                result[i, j] = 0.0
    return np.clip(result, 0.0, 1.0)

def compute_prob_indicators(ifs_indicators, climatologies):
    """计算概率化门控指标, 添加到 ifs_indicators 中."""
    from scipy.stats import gamma, genpareto
    ind = ifs_indicators.copy()
    cl = climatologies

    # 1. precip_percentile (Gamma CDF)
    if "daily_precip_total" in ind:
        shape = cl["precip_gamma_shape"]
        scale = cl["precip_gamma_scale"]
        valid = (shape > 0) & (scale > 0)
        pct = np.zeros_like(ind["daily_precip_total"])
        pct[valid] = gamma.cdf(ind["daily_precip_total"][valid],
                               a=shape[valid], scale=scale[valid]) * 100.0
        ind["precip_percentile"] = pct

    # 2. heat_gpd_prob (GPD exceedance)
    if "tmax" in ind:
        thresh = cl["gpd_threshold"]
        shape = cl["gpd_shape"]
        scale = cl["gpd_scale"]
        exc_rate = cl["gpd_exc_rate"]
        prob = np.ones_like(ind["tmax"])
        exceed = ind["tmax"] > thresh
        exc_val = ind["tmax"][exceed] - thresh[exceed]
        valid = (shape[exceed] != 0) & (scale[exceed] > 0)
        if valid.any():
            prob_exc = np.ones_like(exc_val)
            pos_shape = shape[exceed] > 0.001
            neg_shape = shape[exceed] < -0.001
            zero_shape = np.abs(shape[exceed]) <= 0.001
            if pos_shape.any():
                prob_exc[pos_shape] = (1 + shape[exceed][pos_shape] *
                                       exc_val[pos_shape] / scale[exceed][pos_shape]) ** (-1/shape[exceed][pos_shape])
            if neg_shape.any():
                prob_exc[neg_shape] = 1.0  # bounded upper tail, use exponential approx
            if zero_shape.any():
                prob_exc[zero_shape] = np.exp(-exc_val[zero_shape] / scale[exceed][zero_shape])
            prob[exceed] = prob_exc * exc_rate[exceed]
        ind["heat_gpd_prob"] = np.clip(prob, 0, 1)

    # 3. dust_joint_prob (Copula min)
    if all(v in ind for v in ["wind10_speed", "dewpoint_depression_c", "rh2m", "wind_shear_850_200"]):
        f_wind = _lookup_percentile(ind["wind10_speed"], cl["dust_pct_vals"], cl["wind10_pct"])
        f_dew  = _lookup_percentile(ind["dewpoint_depression_c"], cl["dust_pct_vals"], cl["dewpoint_pct"])
        f_rh   = _lookup_percentile(100 - ind["rh2m"], cl["dust_pct_vals"], cl["rh2m_pct"])
        f_shear = _lookup_percentile(ind["wind_shear_850_200"], cl["dust_pct_vals"], cl["shear_pct"])
        ind["dust_joint_prob"] = np.minimum(np.minimum(f_wind, f_dew),
                                            np.minimum(f_rh, f_shear))
    else:
        ind["dust_joint_prob"] = np.zeros_like(ind.get("t2m", np.zeros((65,89))))

    # 4. humid_heat_joint_prob (Copula min)
    if all(v in ind for v in ["sst", "rh2m", "t2m", "wind10_speed"]):
        f_sst   = _lookup_percentile(ind["sst"], cl["humid_pct_vals"], cl["h_sst_pct"])
        f_rh    = _lookup_percentile(ind["rh2m"], cl["humid_pct_vals"], cl["h_rh2m_pct"])
        f_t2m   = _lookup_percentile(ind["t2m"], cl["humid_pct_vals"], cl["h_t2m_pct"])
        f_wind  = _lookup_percentile(ind["wind10_speed"], cl["humid_pct_vals"], cl["h_wind10_pct"])
        ind["humid_heat_joint_prob"] = np.minimum(np.minimum(f_sst, f_rh),
                                                  np.minimum(f_t2m, f_wind))
    else:
        ind["humid_heat_joint_prob"] = np.zeros_like(ind.get("t2m", np.zeros((65,89))))

    # 派生变量
    if "t2m" in ind and "t2m_c" not in ind:
        ind["t2m_c"] = ind["t2m"]
    if "tmax" not in ind and "t2m" in ind:
        ind["tmax_c"] = ind["t2m"]  # fallback
    if "tmax" in ind:
        ind["tmax_c"] = ind["tmax"]
    if "t2m" in ind:
        t2m = ind["t2m"]
        ind["t2m_anomaly_c"] = t2m - np.nanmean(t2m)  # approximate

    return ind

# ── 5. Run hazard detection from rules.json ──
def detect_hazards(indicators, rules_path="schema/rules.json"):
    """Apply rules.json hazard detection to computed indicators."""
    with open(rules_path, encoding='utf-8') as f:
        rules = json.load(f)["rules"]

    results = {}
    for rule_group in rules:
        hazard_type = rule_group.get("hazard_type", rule_group.get("type", "unknown"))
        conditions = rule_group.get("conditions", [])
        severity_levels = rule_group.get("severity", [
            {"label": "low", "range": [0.0, 0.3]},
            {"label": "medium", "range": [0.3, 0.6]},
            {"label": "high", "range": [0.6, 0.8]},
            {"label": "extreme", "range": [0.8, 1.0]},
        ])

        ref_arr = list(indicators.values())[0]
        score = np.zeros(ref_arr.shape, dtype=np.float64)
        total_w = 0.0

        for cond in conditions:
            indicator_name = cond.get("indicator", cond.get("variable", ""))
            op = cond.get("op", ">=")
            threshold = cond.get("value", cond.get("threshold", 0))
            weight = cond.get("weight", 1.0)

            if indicator_name not in indicators:
                continue

            arr = indicators[indicator_name]
            if op == ">=": hit = arr >= threshold
            elif op == ">": hit = arr > threshold
            elif op == "<=": hit = arr <= threshold
            elif op == "<": hit = arr < threshold
            else: continue

            score += weight * hit.astype(np.float64)
            total_w += weight

        # Normalize
        if total_w > 0:
            score /= total_w

        # Apply region filter (coastal_humid_heat only for Red Sea + Persian Gulf)
        region_filter = rule_group.get("region_filter", {})
        if region_filter and "applies_to" in region_filter:
            y_arr = np.arange(16.0, 32.25, 0.25)[:, None]
            x_arr = np.arange(34.0, 56.25, 0.25)[None, :]
            region_mask = np.zeros(ref_arr.shape, dtype=bool)
            if "red_sea" in region_filter["applies_to"]:
                region_mask |= (y_arr >= 16) & (y_arr <= 28) & (x_arr >= 34) & (x_arr <= 43)
            if "persian_gulf" in region_filter["applies_to"]:
                region_mask |= (y_arr >= 24) & (y_arr <= 30) & (x_arr >= 48) & (x_arr <= 55)
            score[~region_mask] = 0.0

        max_score = float(score.max())
        trigger_pct = float(np.mean(score >= 0.3) * 100)

        # Determine severity
        severity = "none"
        for sev in severity_levels:
            lo, hi = sev["range"]
            if max_score >= lo:
                severity = sev["label"]

        results[hazard_type] = {
            "max_score": max_score,
            "triggered_pct": trigger_pct,
            "severity": severity,
            "coverage": f"{len([c for c in conditions if c['indicator'] in indicators])}/{len(conditions)}",
        }

    return results

# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # MAIN
    # ══════════════════════════════════════════════════════════════
    print("=" * 70)
    print("  ECMWF IFS → Saudi Hazard Detection Pipeline")
    print("  (Same model system as ERA5 — minimal systematic bias)")
    print("=" * 70)

    all_hazard_results = {}

    # ── 预加载气候态文件 (所有日期共用) ──
    print("\n加载气候态文件...")
    CLIM = load_climatologies(GRID_LAT, GRID_LON)
    print(f"  完成: precip/heat_gpd/dust_joint/humid_heat_joint")

    for date in DATES:
        print(f"\n{'='*50}")
        print(f"  {date}")
        print(f"{'='*50}")

        date_dir = os.path.join(OUTPUT_DIR, date.replace("-", ""))
        os.makedirs(date_dir, exist_ok=True)

        # ── Download & process multiple forecast steps ──
        step_indicators = {}  # step_h -> indicators dict
        step_hazards = {}     # step_h -> hazard results
        saudi_lat = saudi_lon = None

        for step_h in STEPS:
            print(f"\n  --- Step +{step_h}h ---")

            # Download
            try:
                grib_path = download_ifs_step(date, "00", step_h, date_dir)
            except Exception as e:
                print(f"    Download failed: {e}")
                continue

            # Extract Saudi
            try:
                raw_vars, s_lat, s_lon = extract_saudi(grib_path, step_h)
                if saudi_lat is None:
                    saudi_lat, saudi_lon = s_lat, s_lon
            except Exception as e:
                print(f"    Extract failed: {e}")
                continue

            # Compute indicators + probability gating
            indicators = compute_indicators(raw_vars)
            try:
                indicators = compute_prob_indicators(indicators, CLIM)
            except Exception as e:
                pass  # prob indicators optional per step

            step_indicators[step_h] = indicators

            # Quick stats
            tp_val = indicators.get("daily_precip_total", np.zeros(1))
            print(f"    t2m={indicators.get('t2m',np.zeros(1)).mean():.1f}C, "
                  f"tp_max={np.nanmax(tp_val):.1f}mm, "
                  f"precip_pct_max={np.nanmax(indicators.get('precip_percentile',np.zeros(1))):.0f}%")

            # Save per-step NetCDF
            step_out = os.path.join(date_dir, f"ifs_indicators_{date.replace('-','')}_{step_h}h.nc")
            ds_out = xr.Dataset()
            for v, arr in indicators.items():
                if arr.ndim == 2 and arr.shape == s_lat.shape + s_lon.shape:
                    ds_out[v] = xr.DataArray(arr, dims=["lat", "lon"],
                                             coords={"lat": s_lat, "lon": s_lon})
            ds_out.to_netcdf(step_out)

            # Hazard detection (per step)
            try:
                hazards = detect_hazards(indicators)
                step_hazards[step_h] = hazards
            except Exception as e:
                pass

        # ── Build combined daily indicators ──
        if len(step_indicators) >= 2:
            # Daily mean from all steps for temperature
            t2m_steps = np.stack([step_indicators[s]["t2m"] for s in STEPS if s in step_indicators and "t2m" in step_indicators[s]])
            if t2m_steps.shape[0] > 0:
                daily_mean_t2m = t2m_steps.mean(axis=0)
                daily_max_t2m  = t2m_steps.max(axis=0)
            # Accumulated precip = max across steps
            tp_steps = np.stack([step_indicators[s].get("daily_precip_total", np.zeros_like(daily_mean_t2m))
                                for s in STEPS if s in step_indicators])
            daily_max_tp = tp_steps.max(axis=0) if tp_steps.shape[0] > 0 else None

        # ── Summary ──
        print(f"\n  {'='*50}")
        print(f"  Summary: {date} ({len(step_indicators)}/{len(STEPS)} steps)")
        for step_h in sorted(step_hazards.keys()):
            print(f"\n  --- T+{step_h}h ---")
            for ht, result in step_hazards[step_h].items():
                print(f"    {ht}: {result.get('severity','?'):>10s} "
                      f"score={result['max_score']:.2f} trigger={result['triggered_pct']:.1f}%")

    print(f"\n{'='*70}")
    print(f"  Done. Hazard indicators saved to {OUTPUT_DIR}/")
    print(f"{'='*70}")
