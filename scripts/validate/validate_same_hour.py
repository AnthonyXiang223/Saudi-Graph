"""
Same-hour validation: Pangu/FCN forecast vs ERA5 at matching valid times.
Forecast init = 00Z, so T+6h=06Z, T+12h=12Z, T+18h=18Z.
ERA5 downloaded at exactly those times => apples-to-apples comparison.
"""
import numpy as np, xarray as xr, os, sys
from datetime import datetime, timedelta

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def rmse(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() == 0: return np.nan
    a64, b64 = a[valid].astype(np.float64), b[valid].astype(np.float64)
    return float(np.sqrt(np.nanmean((a64 - b64)**2)))

def mae(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() == 0: return np.nan
    return float(np.nanmean(np.abs(a[valid].astype(np.float64) - b[valid].astype(np.float64))))

def bias_fn(a, b):
    """a - b: positive = forecast too warm"""
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() == 0: return np.nan
    return float(np.nanmean(a[valid].astype(np.float64) - b[valid].astype(np.float64)))

def acc(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 10: return np.nan
    a_v = a[valid].astype(np.float64); b_v = b[valid].astype(np.float64)
    a_anom = a_v - np.nanmean(a_v); b_anom = b_v - np.nanmean(b_v)
    num = np.nansum(a_anom * b_anom)
    den = np.sqrt(np.nansum(a_anom**2) * np.nansum(b_anom**2))
    return float(num / den) if den > 0 else 0.0

def is_sane_temp(arr_kelvin):
    return (arr_kelvin > 150) & (arr_kelvin < 400)

FORECAST_DIR = "forecast"
ERA5_VAL_DIR = "era5_validation"
INIT_DATES = ["20250115", "20250415", "20250701", "20251015"]
SEASONS = {"01": "Winter", "04": "Spring", "07": "Summer", "10": "Autumn"}
VALID_TIMES = {"06:00": "T+6h", "12:00": "T+12h", "18:00": "T+18h"}

# ── Load forecast (same as before) ──
def load_forecast(path):
    ds = xr.open_dataset(path)
    arr = ds["t2m"].values
    lat = ds["lat"].values; lon = ds["lon"].values
    lead_h = np.arange(arr.shape[1]) * 6 + 6
    ds.close()
    arr_c = arr[0].astype(np.float64) - 273.15
    sane = is_sane_temp(arr[0])
    arr_c[~sane] = np.nan
    return arr_c, lat, lon, lead_h

# ── Load ERA5 validation ──
def load_era5_val(date_str, time_str):
    """Load ERA5 at specific valid time. Returns (t2m_celsius, lat, lon)."""
    time_compact = time_str.replace(":", "")
    path = os.path.join(ERA5_VAL_DIR, f"era5_sfc_{date_str}_{time_compact}Z.nc")
    if not os.path.exists(path):
        return None, None, None
    ds = xr.open_dataset(path)
    if "t2m" not in ds:
        ds.close(); return None, None, None
    arr_k = ds["t2m"].values  # Kelvin, (valid_time, lat, lon)
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    ds.close()
    arr_c = arr_k[0].astype(np.float64) - 273.15  # first valid_time
    return arr_c, lat, lon

def coarsen_to_grid(era_arr, era_lat, era_lon, target_lat, target_lon):
    """Interpolate ERA5 to forecast grid. Both arrays are north-to-south."""
    da = xr.DataArray(era_arr, dims=["latitude", "longitude"],
                      coords={"latitude": era_lat, "longitude": era_lon})
    return da.interp(latitude=target_lat, longitude=target_lon, method="linear").values

# ── Also load indicator data for comparison ──
def load_indicator(date_str):
    path = os.path.join("indicators", f"saudi_indicators_{date_str}.nc")
    if not os.path.exists(path): return None, None, None
    ds = xr.open_dataset(path)
    if "t2m_c" not in ds: ds.close(); return None, None, None
    arr = ds["t2m_c"].values.astype(np.float64)
    glat = ds["latitude"].values.astype(np.float64)
    glon = ds["longitude"].values.astype(np.float64)
    ds.close()
    return arr, glat, glon

# ══════════════════════════════════════════════════════════════
print("=" * 100)
print("  SAME-HOUR VALIDATION: Pangu6/FCNv1 vs ERA5 Instantaneous")
print("=" * 100)
print(f"  Comparing forecast at valid time against ERA5 at the same hour")
print()

all_results = []

for date_c in INIT_DATES:
    season = SEASONS[date_c[4:6]]

    pangu_path = os.path.join(FORECAST_DIR, "pangu_results", f"pangu_{date_c}.nc")
    fcn_path   = os.path.join(FORECAST_DIR, "fcn_results", f"fcn_{date_c}.nc")

    if not os.path.exists(pangu_path) or not os.path.exists(fcn_path):
        continue

    p_arr, p_lat, p_lon, p_lead = load_forecast(pangu_path)
    f_arr, f_lat, f_lon, f_lead = load_forecast(fcn_path)

    p_ok = np.isfinite(p_arr).any(axis=(1,2)).sum()
    f_ok = np.isfinite(f_arr).any(axis=(1,2)).sum()

    print(f"  {date_c} ({season})")
    print(f"    Pangu: {p_ok}/{p_arr.shape[0]} steps valid, "
          f"T2m range {np.nanmin(p_arr):.1f} ~ {np.nanmax(p_arr):.1f} C")
    print(f"    FCN:   {f_ok}/{f_arr.shape[0]} steps valid, "
          f"T2m range {np.nanmin(f_arr):.1f} ~ {np.nanmax(f_arr):.1f} C")

    for step in range(min(p_arr.shape[0], f_arr.shape[0])):
        lead_h = p_lead[step]

        # Map lead hour to ERA5 valid time
        if lead_h == 6:
            valid_time_str = "06:00"
        elif lead_h == 12:
            valid_time_str = "12:00"
        elif lead_h == 18:
            valid_time_str = "18:00"
        else:
            continue

        # Load same-hour ERA5
        era_arr, era_lat, era_lon = load_era5_val(date_c, valid_time_str)
        if era_arr is None:
            continue

        era_coarse = coarsen_to_grid(era_arr, era_lat, era_lon, p_lat, p_lon)

        p_slice = p_arr[step]
        f_slice = f_arr[step]
        p_finite = np.isfinite(p_slice)
        f_finite = np.isfinite(f_slice)

        # Pangu vs same-hour ERA5
        p_vs_e = {"rmse": np.nan, "mae": np.nan, "bias": np.nan, "acc": np.nan}
        if p_finite.any():
            p_vs_e["rmse"] = rmse(p_slice, era_coarse)
            p_vs_e["mae"]  = mae(p_slice, era_coarse)
            p_vs_e["bias"] = bias_fn(p_slice, era_coarse)
            p_vs_e["acc"]  = acc(p_slice, era_coarse)

        # FCN vs same-hour ERA5
        f_vs_e = {"rmse": np.nan, "mae": np.nan, "bias": np.nan, "acc": np.nan}
        if f_finite.any():
            f_vs_e["rmse"] = rmse(f_slice, era_coarse)
            f_vs_e["mae"]  = mae(f_slice, era_coarse)
            f_vs_e["bias"] = bias_fn(f_slice, era_coarse)
            f_vs_e["acc"]  = acc(f_slice, era_coarse)

        all_results.append({
            "init": date_c, "season": season, "lead_h": lead_h, "time": valid_time_str,
            "p_rmse": p_vs_e["rmse"], "p_mae": p_vs_e["mae"],
            "p_bias": p_vs_e["bias"], "p_acc": p_vs_e["acc"],
            "f_rmse": f_vs_e["rmse"], "f_mae": f_vs_e["mae"],
            "f_bias": f_vs_e["bias"], "f_acc": f_vs_e["acc"],
            "p_ok": bool(p_finite.any()), "f_ok": bool(f_finite.any()),
        })

# ══════════════════════════════════════════════════════════════
# Also compare against INDICATOR (daily avg) for the same dates/times
# to quantify the daily-mean vs instantaneous difference
print()
print("  --- Indicator (daily mean) vs same-hour ERA5 baseline ---")
indicator_comparison = []
for date_c in INIT_DATES:
    ind_arr, ind_lat, ind_lon = load_indicator(date_c)
    if ind_arr is None: continue
    for valid_time_str in ["06:00", "12:00", "18:00"]:
        era_arr, era_lat, era_lon = load_era5_val(date_c, valid_time_str)
        if era_arr is None: continue
        p_lat_sample = np.arange(16.0, 32.25, 0.25)
        p_lon_sample = np.arange(34.0, 56.25, 0.25)
        era_c = coarsen_to_grid(era_arr, era_lat, era_lon, p_lat_sample, p_lon_sample)
        ind_c = coarsen_to_grid(ind_arr, ind_lat, ind_lon, p_lat_sample, p_lon_sample)
        indicator_comparison.append({
            "date": date_c, "time": valid_time_str,
            "rmse": rmse(ind_c, era_c), "bias": bias_fn(ind_c, era_c),
        })

# ══════════════════════════════════════════════════════════════
# SUMMARIES
print(f"\n{'='*100}")
print(f"  RESULTS: Same-hour validation ({len(all_results)} forecast steps)")
print(f"{'='*100}")

p_valid = [r for r in all_results if r["p_ok"] and np.isfinite(r["p_rmse"])]
f_valid = [r for r in all_results if r["f_ok"] and np.isfinite(r["f_rmse"])]

print(f"\n  Pangu valid steps: {len(p_valid)}, FCN valid steps: {len(f_valid)}")

# ── Core metrics ──
print(f"\n  {'Metric':<22s} {'Pangu6':>10s} {'FCNv1':>10s} {'Note':>20s}")
print(f"  {'-'*65}")
if p_valid:
    p_rmse_avg = np.nanmean([r["p_rmse"] for r in p_valid])
    p_mae_avg  = np.nanmean([r["p_mae"] for r in p_valid])
    p_bias_avg = np.nanmean([r["p_bias"] for r in p_valid])
    p_acc_avg  = np.nanmean([r["p_acc"] for r in p_valid if np.isfinite(r["p_acc"])])
    print(f"  {'RMSE (C)':<22s} {p_rmse_avg:>10.3f} {'--':>10s} {'same-hour reference':>20s}")

if f_valid:
    f_rmse_avg = np.nanmean([r["f_rmse"] for r in f_valid])
    print(f"  {'RMSE (C)':<22s} {'--':>10s} {f_rmse_avg:>10.3f}")

if p_valid:
    print(f"  {'MAE (C)':<22s} {p_mae_avg:>10.3f} {'--':>10s}")
    print(f"  {'Bias (C)':<22s} {p_bias_avg:>10.3f} {'--':>10s}")
    print(f"  {'ACC':<22s} {p_acc_avg:>10.4f} {'--':>10s}")

# ── Per lead time ──
print(f"\n  {'='*65}")
print(f"  By Lead Time:")
for lh in [6, 12, 18]:
    p_at_lh = [r for r in p_valid if r["lead_h"] == lh]
    f_at_lh = [r for r in f_valid if r["lead_h"] == lh]
    print(f"\n  [T+{lh}h]")
    if p_at_lh:
        print(f"    Pangu: n={len(p_at_lh)}, RMSE={np.nanmean([r['p_rmse'] for r in p_at_lh]):.3f}+/-{np.nanstd([r['p_rmse'] for r in p_at_lh]):.3f}, "
              f"Bias={np.nanmean([r['p_bias'] for r in p_at_lh]):.3f}, ACC={np.nanmean([r['p_acc'] for r in p_at_lh]):.3f}")
    if f_at_lh:
        print(f"    FCN:   n={len(f_at_lh)}, RMSE={np.nanmean([r['f_rmse'] for r in f_at_lh]):.3f}+/-{np.nanstd([r['f_rmse'] for r in f_at_lh]):.3f}, "
              f"Bias={np.nanmean([r['f_bias'] for r in f_at_lh]):.3f}, ACC={np.nanmean([r['f_acc'] for r in f_at_lh]):.3f}")

# ── Per season ──
print(f"\n  {'='*65}")
print(f"  By Season:")
for season in ["Winter", "Spring", "Summer", "Autumn"]:
    s_p = [r for r in p_valid if r["season"] == season]
    s_f = [r for r in f_valid if r["season"] == season]
    if s_p:
        print(f"\n  [{season}]")
        print(f"    Pangu: n={len(s_p)}, RMSE={np.nanmean([r['p_rmse'] for r in s_p]):.3f}, "
              f"Bias={np.nanmean([r['p_bias'] for r in s_p]):.3f}, "
              f"ACC={np.nanmean([r['p_acc'] for r in s_p]):.3f}")
        if s_f:
            print(f"    FCN:   n={len(s_f)}, RMSE={np.nanmean([r['f_rmse'] for r in s_f]):.3f}, "
                  f"Bias={np.nanmean([r['f_bias'] for r in s_f]):.3f}")

# ── Indicator daily-mean bias ──
print(f"\n{'='*100}")
print(f"  INDICATOR DAILY-MEAN BIAS vs Same-hour ERA5")
print(f"{'='*100}")
if indicator_comparison:
    by_time = {}
    for r in indicator_comparison:
        t = r["time"]
        if t not in by_time: by_time[t] = []
        by_time[t].append(r)
    print(f"\n  {'Valid Time':<12s} {'RMSE':>8s} {'Bias':>8s}  {'Meaning'}")
    print(f"  {'-'*55}")
    for t in ["06:00", "12:00", "18:00"]:
        if t in by_time:
            items = by_time[t]
            avg_rmse = np.nanmean([r["rmse"] for r in items])
            avg_bias = np.nanmean([r["bias"] for r in items])
            if avg_bias < -1:
                meaning = "daily_mean COLDER than this hour"
            elif avg_bias > 1:
                meaning = "daily_mean WARMER than this hour"
            else:
                meaning = "daily_mean ~ this hour"
            print(f"  {t:<12s} {avg_rmse:>8.3f} {avg_bias:>8.3f}  {meaning}")

# ── Detailed table ──
print(f"\n{'='*100}")
print(f"  DETAILED: Forecast vs Same-hour ERA5 (T2m)")
print(f"{'='*100}")
print(f"  {'Init':<10s} {'Lead':>5s} {'ERA5@':>6s} {'P-RMSE':>8s} {'P-Bias':>8s} {'P-ACC':>7s} {'F-RMSE':>8s} {'F-Bias':>8s} {'Status'}")
print(f"  {'-'*85}")
for r in all_results:
    def f(x): return f"{x:.2f}" if np.isfinite(x) else "  N/A  "
    def fa(x): return f"{x:.3f}" if np.isfinite(x) else "  N/A "
    status = "both" if (r["p_ok"] and r["f_ok"]) else ("P only" if r["p_ok"] else "F only")
    print(f"  {r['init']:<10s} T+{r['lead_h']:>2.0f}h {r['time']:>6s} "
          f"{f(r['p_rmse']):>8s} {f(r['p_bias']):>8s} {fa(r['p_acc']):>7s} "
          f"{f(r['f_rmse']):>8s} {f(r['f_bias']):>8s} {status}")

print()
print("NOTE: This is apples-to-apples -- both forecast and ERA5 are INSTANTANEOUS at the same hour.")
print("The previous ~4.2C daily-mean comparison conflated model error with diurnal cycle mismatch.")
