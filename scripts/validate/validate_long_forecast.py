"""
验证 GPU 长预报: Pangu6/FCN3 7天预测 vs ERA5
支持 28步 (0-7天) 预报输出
"""
import numpy as np, xarray as xr, os, sys
from datetime import datetime, timedelta

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def rmse(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() == 0: return np.nan
    return float(np.sqrt(np.nanmean((a[valid].astype(np.float64) - b[valid].astype(np.float64))**2)))

def mae(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() == 0: return np.nan
    return float(np.nanmean(np.abs(a[valid].astype(np.float64) - b[valid].astype(np.float64))))

def bias_fn(a, b):
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

def is_sane_temp(arr_k):
    return (arr_k > 150) & (arr_k < 400)

def load_forecast(path, tag="t2m"):
    ds = xr.open_dataset(path)
    arr = ds[tag].values
    lat = ds["lat"].values; lon = ds["lon"].values
    lead_h = np.arange(arr.shape[1]) * 6 + 6
    ds.close()
    arr_c = arr[0].astype(np.float64) - 273.15
    sane = is_sane_temp(arr[0])
    arr_c[~sane] = np.nan
    return arr_c, lat, lon, lead_h

def load_indicator(date_str):
    path = os.path.join("indicators", f"saudi_indicators_{date_str}.nc")
    if not os.path.exists(path): return None, None, None
    ds = xr.open_dataset(path)
    if "t2m_c" not in ds: ds.close(); return None, None, None
    arr = ds["t2m_c"].values.astype(np.float64)
    glat = ds["latitude"].values.astype(np.float64)
    glon = ds["longitude"].values.astype(np.float64)
    ds.close(); return arr, glat, glon

def load_era5_val(date_str, time_str):
    time_compact = time_str.replace(":", "")
    path = os.path.join("era5_validation", f"era5_sfc_{date_str}_{time_compact}Z.nc")
    if not os.path.exists(path): return None, None, None
    ds = xr.open_dataset(path)
    if "t2m" not in ds: ds.close(); return None, None, None
    arr_c = ds["t2m"].values[0].astype(np.float64) - 273.15
    lat = ds["latitude"].values; lon = ds["longitude"].values
    ds.close(); return arr_c, lat, lon

def coarsen(era_arr, era_lat, era_lon, tgt_lat, tgt_lon):
    da = xr.DataArray(era_arr, dims=["latitude","longitude"],
                      coords={"latitude":era_lat,"longitude":era_lon})
    return da.interp(latitude=tgt_lat, longitude=tgt_lon, method="linear").values

SEASONS = {"01":"冬","04":"春","07":"夏","10":"秋"}
FORECAST_DIR = "forecast"

# ══════════════════════════════════════════════════════════════════
# PART 1: 短预报 (T+6h/12h/18h) vs 同时次 ERA5 — 已有下载数据
# ══════════════════════════════════════════════════════════════════
print("=" * 100)
print("  PART 1: Same-hour validation (T+6h/12h/18h vs ERA5)")
print("=" * 100)

INIT_DATES = ["20250115", "20250415", "20250701", "20251015"]

# Pangu short forecast
for date_c in INIT_DATES:
    path = os.path.join(FORECAST_DIR, "pangu_results", f"pangu_{date_c}.nc")
    if not os.path.exists(path): continue
    p_arr, p_lat, p_lon, p_lead = load_forecast(path)
    print(f"\n  {date_c}: Pangu {p_arr.shape[0]} steps")

    for step in range(p_arr.shape[0]):
        lh = p_lead[step]
        valid_t = f"{int(lh):02d}:00"
        era_arr, era_lat, era_lon = load_era5_val(date_c, valid_t)
        if era_arr is None: continue
        era_c = coarsen(era_arr, era_lat, era_lon, p_lat, p_lon)
        p = p_arr[step]
        if np.isfinite(p).any():
            print(f"    T+{lh:.0f}h vs ERA5 {valid_t}Z: "
                  f"RMSE={rmse(p,era_c):.2f}C, Bias={bias_fn(p,era_c):.2f}C, ACC={acc(p,era_c):.3f}")

# ══════════════════════════════════════════════════════════════════
# PART 2: 预报 vs Indicator 日均值 — 按天聚合
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print(f"  PART 2: Forecast vs Indicator (daily mean) — daily aggregation")
print(f"{'='*100}")

for date_c in INIT_DATES:
    path = os.path.join(FORECAST_DIR, "pangu_results", f"pangu_{date_c}.nc")
    if not os.path.exists(path): continue
    p_arr, p_lat, p_lon, p_lead = load_forecast(path)
    init_date = datetime.strptime(date_c, "%Y%m%d")

    print(f"\n  {date_c} (init 00Z): {p_arr.shape[0]} steps, {p_lead[-1]/24:.0f}天预报")

    # 按预报天数分组，取日平均 vs indicator
    for day_offset in range(0, int(p_lead[-1] / 24) + 1):
        day_steps = [s for s in range(p_arr.shape[0])
                     if day_offset * 24 < p_lead[s] <= (day_offset + 1) * 24]
        if not day_steps: continue

        target_date = init_date + timedelta(days=day_offset)
        target_str = target_date.strftime("%Y%m%d")

        ind_arr, ind_lat, ind_lon = load_indicator(target_str)
        if ind_arr is None:
            print(f"    Day+{day_offset} ({target_str}): no indicator")
            continue

        # 预报日平均
        fc_day = np.nanmean(p_arr[day_steps], axis=0)

        # Indicator 粗化到预报网格
        ind_c = coarsen(ind_arr, ind_lat, ind_lon, p_lat, p_lon)

        r = rmse(fc_day, ind_c)
        b = bias_fn(fc_day, ind_c)
        a = acc(fc_day, ind_c)
        print(f"    Day+{day_offset} ({target_str}): "
              f"RMSE={r:.2f}C, Bias={b:.2f}C, ACC={a:.3f} "
              f"[{len(day_steps)}步平均]")

# ══════════════════════════════════════════════════════════════════
# PART 3: RMSE 随预报时效增长曲线
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print(f"  PART 3: Error growth with lead time (vs indicator daily mean)")
print(f"{'='*100}")

growth_data = {}
for date_c in INIT_DATES:
    path = os.path.join(FORECAST_DIR, "pangu_results", f"pangu_{date_c}.nc")
    if not os.path.exists(path): continue
    p_arr, p_lat, p_lon, p_lead = load_forecast(path)
    init_date = datetime.strptime(date_c, "%Y%m%d")

    for step in range(p_arr.shape[0]):
        lh = p_lead[step]
        target_date = init_date + timedelta(hours=int(lh))
        target_str = target_date.strftime("%Y%m%d")

        ind_arr, ind_lat, ind_lon = load_indicator(target_str)
        if ind_arr is None: continue

        ind_c = coarsen(ind_arr, ind_lat, ind_lon, p_lat, p_lon)
        p_slice = p_arr[step]
        if not np.isfinite(p_slice).any(): continue

        r = rmse(p_slice, ind_c)
        key = int(lh)
        if key not in growth_data:
            growth_data[key] = {"rmse":[], "bias":[], "acc":[]}
        growth_data[key]["rmse"].append(r)
        growth_data[key]["bias"].append(bias_fn(p_slice, ind_c))
        growth_data[key]["acc"].append(acc(p_slice, ind_c))

print(f"\n  {'Lead':<8s} {'n':>4s} {'RMSE':>8s} {'Bias':>8s} {'ACC':>8s}")
print(f"  {'-'*40}")
for lh in sorted(growth_data.keys()):
    d = growth_data[lh]
    print(f"  {lh:>4.0f}h   {len(d['rmse']):>4d}  {np.nanmean(d['rmse']):>8.3f}  "
          f"{np.nanmean(d['bias']):>8.3f}  {np.nanmean(d['acc']):>8.3f}")

print(f"\n  NOTE: vs indicator = vs daily mean, NOT same-hour.")
print(f"  For same-hour validation, see PART 1 above (only T+6/12/18h available).")
