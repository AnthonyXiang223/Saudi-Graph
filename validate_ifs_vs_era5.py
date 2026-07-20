"""
IFS forecast vs ERA5 indicator — hazard detection comparison
"""
import os, sys, json, numpy as np, xarray as xr
from datetime import datetime, timedelta
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
IFS_DIR = os.path.join(PROJECT_DIR, "aifs_forecasts")
IND_DIR  = os.path.join(PROJECT_DIR, "indicators")

# ── 加载 rules ──
with open(os.path.join(PROJECT_DIR, "schema", "rules.json"), encoding='utf-8') as f:
    rules_data = json.load(f)

def run_detection(indicators, hazard_types=None):
    """Run rules.json detection, return {hazard_type: {max_score, trigger_pct}}."""
    if hazard_types is None:
        hazard_types = ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]
    results = {}
    for rule in rules_data["rules"]:
        ht = rule["hazard_type"]
        if ht not in hazard_types: continue
        available = [c for c in rule["conditions"] if c["indicator"] in indicators]
        if len(available) < 2:
            results[ht] = {"max_score": 0, "trigger_pct": 0, "n_conds": len(available)}
            continue

        ref = list(indicators.values())[0]
        score = np.zeros(ref.shape, dtype=np.float64)
        total_w = 0
        for c in available:
            arr = indicators[c["indicator"]]
            op, th, w = c["op"], c["value"], c["weight"]
            if op == ">=": hit = arr >= th
            elif op == ">": hit = arr > th
            elif op == "<=": hit = arr <= th
            elif op == "<": hit = arr < th
            else: continue
            score += w * hit.astype(np.float64)
            total_w += w
        if total_w > 0: score /= total_w
        results[ht] = {
            "max_score": float(np.nanmax(score)),
            "trigger_pct": float(np.mean(score >= 0.3) * 100),
            "n_conds": len(available),
        }
    return results

# ── 加载 IFS ──
def load_ifs(date_str, step_h):
    path = os.path.join(IFS_DIR, date_str, f"ifs_indicators_{date_str}_{step_h}h.nc")
    if not os.path.exists(path): return None
    ds = xr.open_dataset(path)
    ind = {v: ds[v].values.astype(np.float64) for v in ds.data_vars}
    ds.close()
    return ind

# ── 加载 ERA5 ──
def load_era5(date_str):
    path = os.path.join(IND_DIR, f"saudi_indicators_{date_str}.nc")
    if not os.path.exists(path): return None
    ds = xr.open_dataset(path)
    ind = {}
    # Extract relevant variables (match operator.json IDs)
    for v in ds.data_vars:
        if v in ["t2m_c", "wind10_speed", "dewpoint_depression_c", "rh2m",
                 "vpd_kpa", "tmax_c", "t2m_anomaly_c", "daily_precip_total",
                 "daily_precip_anomaly", "cape", "pwat", "ivt_convergence",
                 "sst_celsius", "wind_shear_850_200", "flash_flood_risk",
                 "heatwave_day_flag", "precip_percentile", "heat_gpd_prob",
                 "dust_joint_prob", "humid_heat_joint_prob",
                 "ds10_max_1h", "ds10_max_30min"]:
            arr = ds[v].values.astype(np.float64)
            if arr.ndim > 2: arr = arr[0]
            ind[v] = arr
    ds.close()
    # Interpolate ERA5 (0.1deg) to IFS grid (0.25deg) — using lat/lon coordinates
    # ERA5 indicator uses 'latitude'/'longitude' coords
    era_lat = ds["latitude"].values if "latitude" in ds.coords else None
    era_lon = ds["longitude"].values if "longitude" in ds.coords else None
    # Re-open to get coords
    ds2 = xr.open_dataset(path)
    elat = ds2["latitude"].values.astype(np.float64)
    elon = ds2["longitude"].values.astype(np.float64)
    elon2 = ds2["lon"].values.astype(np.float64) if "lon" in ds2.coords else elon
    ds2.close()
    # IFS grid
    ifs_lat = np.arange(16.0, 32.25, 0.25)
    ifs_lon = np.arange(34.0, 56.25, 0.25)
    # Interpolate to IFS grid
    ind_ifs = {}
    for v, arr in ind.items():
        ny, nx = arr.shape
        # Pick correct coords based on data shape
        if ny == 160 and nx == 220:
            use_lat, use_lon = elat, elon  # latitude/longitude
        elif ny == 160 and nx == 221:
            use_lat, use_lon = elat, elon2  # lat/lon (different lon)
        else:
            continue
        da = xr.DataArray(arr, dims=["lat","lon"],
                          coords={"lat": use_lat[:ny], "lon": use_lon[:nx]})
        interp = da.interp(lat=ifs_lat, lon=ifs_lon, method="linear").values
        ind_ifs[v] = interp
    return ind_ifs

# ── Compare ──
INIT_DATES = ["20250115", "20250415", "20250701", "20251015"]

print("=" * 80)
print("  IFS Forecast vs ERA5 Indicator — Hazard Detection Comparison")
print("=" * 80)

all_comparisons = []

for init_date in INIT_DATES:
    init_dt = datetime.strptime(init_date, "%Y%m%d")

    # IFS steps available
    for step_h, valid_offset_h in [(0, 0), (12, 0), (24, 1)]:
        valid_dt = init_dt + timedelta(hours=valid_offset_h * 24)
        valid_str = valid_dt.strftime("%Y%m%d")

        ifs = load_ifs(init_date, step_h)
        era = load_era5(valid_str)
        if ifs is None or era is None:
            continue

        ifs_haz = run_detection(ifs)
        era_haz = run_detection(era)

        print(f"\n  {init_date} T+{step_h}h → valid {valid_str}")
        print(f"  {'Hazard':<22s} {'IFS score':>10s} {'IFS trig':>8s} {'ERA5 score':>10s} {'ERA5 trig':>8s} {'Match':>8s}")
        print(f"  {'-'*70}")

        for ht in ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]:
            is_h = ifs_haz.get(ht, {})
            er_h = era_haz.get(ht, {})
            is_score = is_h.get("max_score", 0)
            er_score = er_h.get("max_score", 0)
            is_trig  = is_h.get("trigger_pct", 0)
            er_trig  = er_h.get("trigger_pct", 0)
            # Match: both triggered or both not triggered
            if is_trig > 0 and er_trig > 0: match = "HIT"
            elif is_trig == 0 and er_trig == 0: match = "MISS"
            elif is_trig > 0 and er_trig == 0: match = "FP"  # false positive
            else: match = "FN"  # false negative
            print(f"  {ht:<22s} {is_score:>10.3f} {is_trig:>7.1f}% {er_score:>10.3f} {er_trig:>7.1f}% {match:>8s}")

            all_comparisons.append({
                "init": init_date, "step": step_h, "valid": valid_str,
                "hazard": ht,
                "ifs_score": is_score, "era5_score": er_score,
                "ifs_trig": is_trig, "era5_trig": er_trig,
            })

# ── Summary ──
print(f"\n{'='*80}")
print("  Summary: IFS vs ERA5 hit rate by hazard type")
print(f"{'='*80}")

by_hazard = defaultdict(lambda: {"hit": 0, "miss": 0, "fp": 0, "fn": 0, "total": 0})
for c in all_comparisons:
    ht = c["hazard"]
    by_hazard[ht]["total"] += 1
    if c["ifs_trig"] > 0 and c["era5_trig"] > 0: by_hazard[ht]["hit"] += 1
    elif c["ifs_trig"] == 0 and c["era5_trig"] == 0: by_hazard[ht]["miss"] += 1
    elif c["ifs_trig"] > 0 and c["era5_trig"] == 0: by_hazard[ht]["fp"] += 1
    else: by_hazard[ht]["fn"] += 1

print(f"\n  {'Hazard':<22s} {'HIT':>6s} {'MISS':>6s} {'FP':>6s} {'FN':>6s} {'命中':>8s}")
print(f"  {'-'*60}")
for ht in ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]:
    d = by_hazard[ht]
    hit_rate = d["hit"] / max(d["total"], 1) * 100
    print(f"  {ht:<22s} {d['hit']:>6d} {d['miss']:>6d} {d['fp']:>6d} {d['fn']:>6d} {hit_rate:>7.1f}%")

print(f"\n  HIT = IFS和ERA5都触发  MISS = 都不触发")
print(f"  FP  = IFS触发但ERA5不触发 (IFS过度预警)")
print(f"  FN  = IFS不触发但ERA5触发 (IFS漏报)")
