"""
FCN GPU 结果 vs ERA5 全面验证 — 多变量 × 7天预报
对比: t2m, 风速, 湿度, 气压, 500hPa位势高度
"""
import numpy as np, xarray as xr, os, sys
from datetime import datetime, timedelta

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── 指标 ──
def rmse(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() == 0: return np.nan
    a64, b64 = a[valid].astype(np.float64), b[valid].astype(np.float64)
    return float(np.sqrt(np.nanmean((a64 - b64)**2)))

def bias(a, b):
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

# ── 数据加载 ──
def load_fcst(path):
    ds = xr.open_dataset(path)
    data = {v: ds[v].values[0] for v in ds.data_vars}  # (steps, lat, lon)
    lat = ds["lat"].values; lon = ds["lon"].values
    lead_h = np.arange(data["t2m"].shape[0]) * 6 + 6
    ds.close()
    # t2m K->C
    data["t2m"] = data["t2m"] - 273.15
    # 风速从 u/v 分量
    if "u10m" in data and "v10m" in data:
        data["wind10_speed"] = np.sqrt(data["u10m"]**2 + data["v10m"]**2)
    return data, lat, lon, lead_h

def load_indicator(date_str):
    path = os.path.join("indicators", f"saudi_indicators_{date_str}.nc")
    if not os.path.exists(path): return {}
    ds = xr.open_dataset(path)
    result = {}
    # 所有可比的 2D 变量
    want = {
        "t2m_c": "t2m",
        "wind10_speed": "wind10_speed",
        "rh2m": "rh2m",
        "sh2m": "sh2m",
        "dewpoint_depression_c": "dewpoint_depression",
        "vpd_kpa": "vpd",
        "total_cloud_cover": "cloud_cover",
        "daily_precip_total": "precip",
    }
    for ind_var, short in want.items():
        if ind_var in ds:
            result[short] = ds[ind_var].values.astype(np.float64)
    # 经纬度
    if "latitude" in ds:
        result["lat"] = ds["latitude"].values.astype(np.float64)
        result["lon"] = ds["longitude"].values.astype(np.float64)
    ds.close()
    return result

def coarsen(arr, src_lat, src_lon, tgt_lat, tgt_lon):
    da = xr.DataArray(arr, dims=["latitude","longitude"],
                      coords={"latitude": src_lat, "longitude": src_lon})
    return da.interp(latitude=tgt_lat, longitude=tgt_lon, method="linear").values

# ── 主验证 ──
DATES = ["20250115", "20250415", "20250701", "20251015"]
SEASONS = {"01": "Winter", "04": "Spring", "07": "Summer", "10": "Autumn"}
FCST_DIR = "forecast/fcn_gpu_results"

# 要对比的变量
VARS = {
    "t2m":         {"unit": "C",    "desc": "2m temperature"},
    "wind10_speed": {"unit": "m/s", "desc": "10m wind speed"},
    "msl":         {"unit": "Pa",   "desc": "MSLP"},
}

# 累积所有结果
all_results = []

for date_c in DATES:
    season = SEASONS[date_c[4:6]]
    path = os.path.join(FCST_DIR, f"fcn_gpu_{date_c}.nc")
    if not os.path.exists(path):
        print(f"  {date_c}: 文件不存在")
        continue

    fcst, fc_lat, fc_lon, lead_h = load_fcst(path)
    init_date = datetime.strptime(date_c, "%Y%m%d")
    n_steps = len(lead_h)

    print(f"\n  {date_c} ({season}): {n_steps}步, {lead_h[-1]/24:.1f}天预报")
    print(f"    T2m: {np.nanmin(fcst['t2m']):.1f} ~ {np.nanmax(fcst['t2m']):.1f} C")

    # ── 按天聚合 vs Indicator ──
    for day_offset in range(0, int(lead_h[-1] / 24) + 1):
        day_steps = [s for s in range(n_steps)
                     if day_offset * 24 < lead_h[s] <= (day_offset + 1) * 24]
        if not day_steps:
            continue

        target_date = init_date + timedelta(days=day_offset)
        target_str = target_date.strftime("%Y%m%d")
        ind = load_indicator(target_str)
        if not ind or "t2m" not in ind:
            continue

        for var_key, var_info in VARS.items():
            if var_key not in fcst:
                continue
            if var_key not in ind and var_key != "msl" and var_key != "wind10_speed":
                continue

            # 预报日平均
            fc_day = np.nanmean(fcst[var_key][day_steps], axis=0)

            # Indicator 粗化
            if var_key in ind:
                ind_c = coarsen(ind[var_key], ind["lat"], ind["lon"], fc_lat, fc_lon)
            else:
                continue

            if not np.isfinite(fc_day).any() or not np.isfinite(ind_c).any():
                continue

            r = rmse(fc_day, ind_c)
            b = bias(fc_day, ind_c)
            a = acc(fc_day, ind_c)

            all_results.append({
                "init": date_c, "season": season, "day": day_offset,
                "var": var_key, "rmse": r, "bias": b, "acc": a,
            })

# ── 汇总 ──
print(f"\n{'='*90}")
print(f"  MULTI-VARIABLE VALIDATION: FCN GPU 7-day forecast vs ERA5 Indicators")
print(f"  (Daily mean of 4 forecast steps vs daily indicator)")
print(f"{'='*90}")

# 按变量汇总
from collections import defaultdict
by_var = defaultdict(list)
for r in all_results:
    by_var[r["var"]].append(r)

print(f"\n  {'Variable':<22s} {'n':>5s} {'RMSE':>8s} {'Bias':>8s} {'ACC':>7s}  Notes")
print(f"  {'-'*65}")
for var_key in VARS:
    items = by_var[var_key]
    if not items:
        continue
    n = len(items)
    avg_r = np.nanmean([r["rmse"] for r in items])
    avg_b = np.nanmean([r["bias"] for r in items])
    avg_a = np.nanmean([r["acc"] for r in items if np.isfinite(r["acc"])])
    unit = VARS[var_key]["unit"]
    desc = VARS[var_key]["desc"]
    notes = ""
    if var_key == "t2m":
        notes = "(daily mean vs indicator daily mean)"
    elif var_key == "wind10_speed":
        notes = "(from u10m,v10m forecast)"
    elif var_key == "msl":
        notes = "(instantaneous)"
    print(f"  {desc:<22s} {n:>5d} {avg_r:>7.2f}{unit:>1s} {avg_b:>7.2f}{unit:>1s} {avg_a:>7.3f}  {notes}")

# ── t2m 按预报天数分解 ──
print(f"\n{'='*90}")
print(f"  T2m RMSE by Forecast Day (daily mean forecast vs daily mean indicator)")
print(f"{'='*90}")
t2m_by_day = defaultdict(list)
for r in all_results:
    if r["var"] == "t2m":
        t2m_by_day[r["day"]].append(r)

print(f"\n  {'Day':<6s} {'n':>4s} {'RMSE':>8s} {'Bias':>8s} {'ACC':>7s}")
print(f"  {'-'*40}")
for day in sorted(t2m_by_day.keys()):
    items = t2m_by_day[day]
    print(f"  Day+{day:<3d} {len(items):>4d} {np.nanmean([r['rmse'] for r in items]):>7.2f}C "
          f"{np.nanmean([r['bias'] for r in items]):>7.2f}C "
          f"{np.nanmean([r['acc'] for r in items]):>7.3f}")

# ── t2m 按季节 × 天数 ──
print(f"\n{'='*90}")
print(f"  T2m per Season — Bias evolution over 7 days")
print(f"{'='*90}")
for season in ["Winter", "Spring", "Summer", "Autumn"]:
    print(f"\n  [{season}]")
    for day in range(8):
        items = [r for r in all_results if r["var"] == "t2m" and r["season"] == season and r["day"] == day]
        if items:
            r = items[0]
            print(f"    Day+{day}: RMSE={r['rmse']:.2f}C, Bias={r['bias']:.2f}C, ACC={r['acc']:.3f}")

# ── 总结 ──
print(f"\n{'='*90}")
t2m_items = by_var.get("t2m", [])
wind_items = by_var.get("wind10_speed", [])
if t2m_items:
    print(f"  T2m (每日平均):      RMSE={np.nanmean([r['rmse'] for r in t2m_items]):.2f}C, "
          f"ACC={np.nanmean([r['acc'] for r in t2m_items]):.3f} "
          f"({len(t2m_items)} 天×季对)")
if wind_items:
    print(f"  10m风速 (每日平均):  RMSE={np.nanmean([r['rmse'] for r in wind_items]):.2f}m/s, "
          f"ACC={np.nanmean([r['acc'] for r in wind_items]):.3f} "
          f"({len(wind_items)} 天×季对)")
print(f"\n  Note: 预报4步日平均 vs Indicator日均值; FCN v1 变量数={len(VARS)}")
print(f"  之前只给气温是因为 indicator 中 t2m_c 最直接可比，")
print(f"  风速需要从 u/v 分量合成，气压没有日均 indicator。")
