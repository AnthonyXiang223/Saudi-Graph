"""
Pangu6 vs FCNv1 vs ERA5 Indicators -- 定量差异分析
"""
import numpy as np, xarray as xr, os, sys
from datetime import datetime, timedelta

# Fix Windows GBK encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def rmse(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() == 0: return np.nan
    # Use float64 to avoid overflow
    a64, b64 = a[valid].astype(np.float64), b[valid].astype(np.float64)
    return float(np.sqrt(np.nanmean((a64 - b64)**2)))

def mae(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() == 0: return np.nan
    return float(np.nanmean(np.abs(a[valid].astype(np.float64) - b[valid].astype(np.float64))))

def bias_fn(a, b):
    """a - b: pos=overestimate, neg=underestimate"""
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() == 0: return np.nan
    return float(np.nanmean(a[valid].astype(np.float64) - b[valid].astype(np.float64)))

def acc(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 10: return np.nan
    a_v = a[valid].astype(np.float64)
    b_v = b[valid].astype(np.float64)
    a_anom = a_v - np.nanmean(a_v)
    b_anom = b_v - np.nanmean(b_v)
    num = np.nansum(a_anom * b_anom)
    den = np.sqrt(np.nansum(a_anom**2) * np.nansum(b_anom**2))
    return float(num / den) if den > 0 else 0.0

def is_sane_temp(arr_kelvin):
    """Check if temperature values are physically reasonable (173K to 373K = -100C to 100C)"""
    return (arr_kelvin > 150) & (arr_kelvin < 400)

FORECAST_DIR = "forecast"
INDICATOR_DIR = "indicators"

COMMON_DATES = ["20250115", "20250415", "20250701", "20251015"]
SEASONS = {"01": "Winter", "04": "Spring", "07": "Summer", "10": "Autumn"}

def load_forecast(path):
    """Load forecast NC, returns (t2m_celsius, lat, lon, lead_hours)"""
    ds = xr.open_dataset(path)
    arr = ds["t2m"].values  # Kelvin, (1, nsteps, nlat, nlon)
    lat = ds["lat"].values
    lon = ds["lon"].values
    lead_h = np.arange(arr.shape[1]) * 6 + 6
    ds.close()
    arr_k = arr[0].astype(np.float64)  # Remove time dim, convert to float64
    arr_c = arr_k - 273.15  # K -> C
    # Mark garbage values as NaN
    sane = is_sane_temp(arr_k)
    arr_c[~sane] = np.nan
    return arr_c, lat, lon, lead_h

def load_indicator(date_str):
    """Load indicator t2m_c. CRITICAL: uses 'latitude'/'longitude' dims (north-to-south),
    NOT 'lat'/'lon' (which are south-to-north!)."""
    path = os.path.join(INDICATOR_DIR, f"saudi_indicators_{date_str}.nc")
    if not os.path.exists(path):
        return None, None, None
    ds = xr.open_dataset(path)
    if "t2m_c" not in ds:
        ds.close(); return None, None, None
    # t2m_c has dims ('latitude', 'longitude') -- use THOSE coordinates
    arr = ds["t2m_c"].values.astype(np.float64)  # (160, 220) north-to-south
    glat = ds["latitude"].values.astype(np.float64)  # 31.9 -> 16.0 (north-to-south)
    glon = ds["longitude"].values.astype(np.float64) # 34.0 -> 55.9
    ds.close()
    return arr, glat, glon

def coarsen_to_grid(era_arr, era_lat, era_lon, target_lat, target_lon):
    """ERA5 0.1deg -> forecast 0.25deg via bilinear interpolation.
    era_arr is north-to-south (matching era_lat which descends)."""
    da = xr.DataArray(era_arr, dims=["latitude", "longitude"],
                      coords={"latitude": era_lat, "longitude": era_lon})
    return da.interp(latitude=target_lat, longitude=target_lon, method="linear").values

# ── Main Analysis ──
all_results = []

print("=" * 90)
print("  Pangu6 vs FCNv1 vs ERA5 Indicators -- 2m Temperature Comparison")
print("=" * 90)
print(f"  Forecast grid: 0.25deg (65x89), Indicator grid: 0.1deg (160x220)")
print(f"  Cross-comparison uses bilinear interpolation to coarsen indicators")
print()

for date_c in COMMON_DATES:
    season = SEASONS[date_c[4:6]]
    init_date = datetime.strptime(date_c, "%Y%m%d")

    pangu_path = os.path.join(FORECAST_DIR, "pangu_results", f"pangu_{date_c}.nc")
    fcn_path   = os.path.join(FORECAST_DIR, "fcn_results", f"fcn_{date_c}.nc")

    if not os.path.exists(pangu_path) or not os.path.exists(fcn_path):
        continue

    p_arr, p_lat, p_lon, p_lead = load_forecast(pangu_path)
    f_arr, f_lat, f_lon, f_lead = load_forecast(fcn_path)

    p_valid_steps = np.isfinite(p_arr).any(axis=(1,2)).sum()
    f_valid_steps = np.isfinite(f_arr).any(axis=(1,2)).sum()
    p_total_cells = np.isfinite(p_arr).sum()
    f_total_cells = np.isfinite(f_arr).sum()

    print(f"  {date_c} ({season}) -- init: {init_date.strftime('%Y-%m-%d')}")
    p_min, p_max = np.nanmin(p_arr), np.nanmax(p_arr)
    f_min, f_max = np.nanmin(f_arr), np.nanmax(f_arr)
    print(f"    Pangu: {p_arr.shape[0]} steps, {p_valid_steps} valid, "
          f"T2m range {p_min:.1f} ~ {p_max:.1f} C, {p_total_cells} sane cells")
    print(f"    FCN:   {f_arr.shape[0]} steps, {f_valid_steps} valid, "
          f"T2m range {f_min:.1f} ~ {f_max:.1f} C, {f_total_cells} sane cells")

    if f_valid_steps == 0:
        print(f"    *** FCN all garbage (unreasonable temps) -- skipping vs ERA5")
    if p_valid_steps == 0:
        print(f"    *** Pangu all garbage (unreasonable temps) -- skipping vs ERA5")

    n_steps = min(p_arr.shape[0], f_arr.shape[0])
    for step in range(n_steps):
        lead_h = p_lead[step]
        target_date = init_date + timedelta(hours=int(lead_h))
        target_str = target_date.strftime("%Y%m%d")

        era_arr, i_lat, i_lon = load_indicator(target_str)
        if era_arr is None:
            continue

        era_coarse = coarsen_to_grid(era_arr, i_lat, i_lon, p_lat, p_lon)

        p_slice = p_arr[step]
        f_slice = f_arr[step]

        p_finite = np.isfinite(p_slice)
        f_finite = np.isfinite(f_slice)

        # Pangu vs ERA5
        if p_finite.any():
            p_vs_e = {"rmse": rmse(p_slice, era_coarse), "mae": mae(p_slice, era_coarse),
                      "bias": bias_fn(p_slice, era_coarse), "acc": acc(p_slice, era_coarse)}
        else:
            p_vs_e = {"rmse": np.nan, "mae": np.nan, "bias": np.nan, "acc": np.nan}

        # FCN vs ERA5
        if f_finite.any():
            f_vs_e = {"rmse": rmse(f_slice, era_coarse), "mae": mae(f_slice, era_coarse),
                      "bias": bias_fn(f_slice, era_coarse), "acc": acc(f_slice, era_coarse)}
        else:
            f_vs_e = {"rmse": np.nan, "mae": np.nan, "bias": np.nan, "acc": np.nan}

        # Pangu vs FCN direct
        both_finite = p_finite & f_finite
        if both_finite.any():
            pf_rmse_val = rmse(p_slice, f_slice)
            pf_bias_val = bias_fn(p_slice, f_slice)
        else:
            pf_rmse_val, pf_bias_val = np.nan, np.nan

        all_results.append({
            "init": date_c, "season": season, "lead_h": lead_h, "target": target_str,
            "p_rmse": p_vs_e["rmse"], "p_mae": p_vs_e["mae"],
            "p_bias": p_vs_e["bias"], "p_acc": p_vs_e["acc"],
            "f_rmse": f_vs_e["rmse"], "f_mae": f_vs_e["mae"],
            "f_bias": f_vs_e["bias"], "f_acc": f_vs_e["acc"],
            "pf_rmse": pf_rmse_val, "pf_bias": pf_bias_val,
            "p_ok": bool(p_finite.any()), "f_ok": bool(f_finite.any()),
        })

# ── Summary ──
print(f"\n{'='*90}")
print(f"  SUMMARY")
print(f"{'='*90}")

p_all = [r for r in all_results if r["p_ok"] and np.isfinite(r["p_rmse"])]
f_all = [r for r in all_results if r["f_ok"] and np.isfinite(r["f_rmse"])]
pf_both = [r for r in all_results if r["p_ok"] and r["f_ok"] and np.isfinite(r["pf_rmse"])]

print(f"\n  Total forecast steps: {len(all_results)}")
print(f"  Pangu valid pairs: {len(p_all)}")
print(f"  FCN valid pairs:   {len(f_all)}")
print(f"  Both valid:        {len(pf_both)}")

if p_all and f_all:
    print(f"\n  {'Metric':<20s} {'Pangu6':>10s} {'FCNv1':>10s} {'Winner':>10s}")
    print(f"  {'-'*50}")
    p_rmse_avg = np.nanmean([r["p_rmse"] for r in p_all])
    f_rmse_avg = np.nanmean([r["f_rmse"] for r in f_all])
    print(f"  {'RMSE (C)':<20s} {p_rmse_avg:>10.3f} {f_rmse_avg:>10.3f} {'Pangu' if p_rmse_avg < f_rmse_avg else 'FCN':>10s}")

    p_mae_avg = np.nanmean([r["p_mae"] for r in p_all])
    f_mae_avg = np.nanmean([r["f_mae"] for r in f_all])
    print(f"  {'MAE (C)':<20s} {p_mae_avg:>10.3f} {f_mae_avg:>10.3f}")

    p_bias_avg = np.nanmean([r["p_bias"] for r in p_all])
    f_bias_avg = np.nanmean([r["f_bias"] for r in f_all])
    print(f"  {'Bias (C)':<20s} {p_bias_avg:>10.3f} {f_bias_avg:>10.3f}")

    p_acc_vals = [r["p_acc"] for r in p_all if np.isfinite(r["p_acc"])]
    f_acc_vals = [r["f_acc"] for r in f_all if np.isfinite(r["f_acc"])]
    p_acc_avg = np.nanmean(p_acc_vals) if p_acc_vals else np.nan
    f_acc_avg = np.nanmean(f_acc_vals) if f_acc_vals else np.nan
    print(f"  {'ACC':<20s} {p_acc_avg:>10.4f} {f_acc_avg:>10.4f}")

    # By season
    print(f"\n  {'='*50}")
    print(f"  By Season:")
    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        s_p = [r for r in p_all if r["season"] == season]
        s_f = [r for r in f_all if r["season"] == season]
        if s_p or s_f:
            print(f"\n  [{season}]")
            if s_p:
                vals = [r["p_rmse"] for r in s_p]
                biases = [r["p_bias"] for r in s_p]
                print(f"    Pangu: n={len(s_p)}, RMSE={np.nanmean(vals):.3f}+/-{np.nanstd(vals):.3f}, Bias={np.nanmean(biases):.3f}")
            if s_f:
                vals = [r["f_rmse"] for r in s_f]
                biases = [r["f_bias"] for r in s_f]
                print(f"    FCN:   n={len(s_f)}, RMSE={np.nanmean(vals):.3f}+/-{np.nanstd(vals):.3f}, Bias={np.nanmean(biases):.3f}")

    # Pangu vs FCN direct
    if pf_both:
        pf_rmse_avg = np.nanmean([r["pf_rmse"] for r in pf_both])
        pf_bias_avg = np.nanmean([r["pf_bias"] for r in pf_both])
        print(f"\n  Pangu vs FCN direct difference ({len(pf_both)} pairs):")
        print(f"    RMSE: {pf_rmse_avg:.4f} C")
        print(f"    Bias (P-F): {pf_bias_avg:.4f} C")

# ── Detailed table ──
print(f"\n{'='*90}")
print(f"  Detailed Step-by-Step (T2m)")
print(f"{'='*90}")
print(f"  {'Init':<10s} {'T+':>4s} {'Target':<10s} {'P-RMSE':>8s} {'P-Bias':>8s} {'F-RMSE':>8s} {'F-Bias':>8s} {'P-F RMSE':>9s}  {'Status'}")
print(f"  {'-'*90}")
for r in all_results:
    def fmt(x):
        return f"{x:.2f}" if np.isfinite(x) else "  N/A  "
    status = ""
    if r["p_ok"] and r["f_ok"]:
        status = "both OK"
    elif r["p_ok"]:
        status = "Pangu only"
    elif r["f_ok"]:
        status = "FCN only"
    else:
        status = "ALL BAD"
    print(f"  {r['init']:<10s} {r['lead_h']:>3.0f}h {r['target']:<10s} "
          f"{fmt(r['p_rmse']):>8s} {fmt(r['p_bias']):>8s} "
          f"{fmt(r['f_rmse']):>8s} {fmt(r['f_bias']):>8s} "
          f"{fmt(r['pf_rmse']):>9s}  {status}")

# ── ERA5 ground truth stats ──
print(f"\n{'='*90}")
print(f"  ERA5 Indicator Ground Truth")
print(f"{'='*90}")
for date_c in COMMON_DATES:
    era_arr, i_lat, i_lon = load_indicator(date_c)
    if era_arr is not None:
        print(f"  {date_c}: {era_arr.shape}, "
              f"min={np.nanmin(era_arr):.1f}C, max={np.nanmax(era_arr):.1f}C, "
              f"mean={np.nanmean(era_arr):.2f}C, std={np.nanstd(era_arr):.2f}C")

print(f"\nNote: Negative bias = forecast colder than ERA5")
print(f"FCN 20250415/20250701 have all garbage values (massive overflow) -- need re-run")
print(f"Pangu all 4 dates are clean -- more robust than FCN v1")
