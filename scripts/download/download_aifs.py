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

# 验证: 4季各1天, 只用分析场 (0h) 做变量覆盖测试
DATES = ["2025-01-15", "2025-04-15", "2025-07-01", "2025-10-15"]

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
def extract_saudi(grib_path):
    """Extract Saudi sub-region from IFS GRIB2, return dict of numpy arrays."""
    ds = xr.open_dataset(grib_path, engine="cfgrib",
                         backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface"}})

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    lat_mask = (lat >= GRID_LAT[0] - 0.5) & (lat <= GRID_LAT[-1] + 0.5)
    lon_mask = (lon >= GRID_LON[0] - 0.5) & (lon <= GRID_LON[-1] + 0.5)

    # GRIB shortName → our variable name + unit conversion
    var_map = {
        "2t":   ("t2m",   lambda x: x - 273.15),       # K → °C
        "2d":   ("d2m",   lambda x: x - 273.15),       # K → °C
        "10u":  ("u10",   lambda x: x),
        "10v":  ("v10",   lambda x: x),
        "msl":  ("msl",   lambda x: x / 100.0),         # Pa → hPa
        "sp":   ("sp",    lambda x: x / 100.0),          # Pa → hPa
        "tcwv": ("tcwv",  lambda x: x),
        "tp":   ("tp",    lambda x: x * 1000.0),         # m → mm
        "cp":   ("cp",    lambda x: x * 1000.0),         # m → mm
        "sst":  ("sst",   lambda x: x - 273.15),         # K → °C
        "cape": ("cape",  lambda x: x),
        "mx2t": ("tmax",  lambda x: x - 273.15),         # K → °C
        "mn2t": ("tmin",  lambda x: x - 273.15),         # K → °C
        "tcc":  ("tcc",   lambda x: x),                   # total cloud cover
    }

    result = {}
    for grib_name, (our_name, convert) in var_map.items():
        if grib_name in ds:
            arr = ds[grib_name].values
            if arr.ndim >= 2:
                arr = arr[..., lat_mask, :][..., :, lon_mask]
            if arr.ndim == 3:
                arr = arr[0]  # remove singleton time dim
            result[our_name] = convert(arr).astype(np.float64)

    ds.close()
    saudi_lat = lat[lat_mask]; saudi_lon = lon[lon_mask]
    return result, saudi_lat, saudi_lon

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

# ── 4. Run hazard detection from rules.json ──
def detect_hazards(indicators, rules_path="schema/rules.json"):
    """Apply rules.json hazard detection to computed indicators."""
    with open(rules_path) as f:
        rules = json.load(f)

    results = {}
    for rule_group in rules:
        hazard_type = rule_group.get("hazard_type", rule_group.get("type", "unknown"))
        conditions = rule_group.get("conditions", [])

        score = np.zeros_like(indicators.get("t2m", np.zeros((65, 89))))
        triggered = np.zeros_like(score, dtype=bool)

        for cond in conditions:
            indicator_name = cond.get("indicator", cond.get("variable", ""))
            op = cond.get("op", ">=")
            threshold = cond.get("value", cond.get("threshold", 0))
            weight = cond.get("weight", 1.0)

            if indicator_name not in indicators:
                continue

            arr = indicators[indicator_name]
            if op == ">=":
                hit = arr >= threshold
            elif op == ">":
                hit = arr > threshold
            elif op == "<=":
                hit = arr <= threshold
            elif op == "<":
                hit = arr < threshold
            else:
                continue

            score = score + weight * hit.astype(np.float64)
            if weight > 0.3:  # 关键条件触发
                triggered = triggered | hit

        results[hazard_type] = {
            "score": score,
            "triggered_pct": float(triggered.mean() * 100),
            "max_score": float(score.max()),
        }

    return results

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("  ECMWF IFS → Saudi Hazard Detection Pipeline")
print("  (Same model system as ERA5 — minimal systematic bias)")
print("=" * 70)

all_hazard_results = {}

for date in DATES:
    print(f"\n{'='*50}")
    print(f"  {date}")
    print(f"{'='*50}")

    date_dir = os.path.join(OUTPUT_DIR, date.replace("-", ""))
    os.makedirs(date_dir, exist_ok=True)

    # Download analysis (0h) — we just need the variable coverage verified
    try:
        grib_path = download_ifs_step(date, "00", 0, date_dir)
    except Exception as e:
        print(f"  Download failed: {e}")
        continue

    # Extract Saudi
    try:
        raw_vars, saudi_lat, saudi_lon = extract_saudi(grib_path)
    except Exception as e:
        print(f"  Extract failed: {e}")
        continue

    # Compute indicators
    indicators = compute_indicators(raw_vars)

    # Run rules.json hazard detection
    try:
        hazards = detect_hazards(indicators)
        all_hazard_results[date] = hazards
    except Exception as e:
        print(f"  Hazard detection failed: {e}")
        continue

    # ── Print summary ──
    print(f"\n  Raw variables extracted ({len(raw_vars)}):")
    for v, arr in sorted(raw_vars.items()):
        if np.isfinite(arr).any():
            print(f"    {v:<6s}: {np.nanmin(arr):.1f} ~ {np.nanmax(arr):.1f}")

    print(f"\n  Computed indicators ({len(indicators)}):")
    for v in ["t2m", "d2m", "wind10_speed", "rh2m", "dewpoint_depression_c",
              "vpd_kpa", "daily_precip_total", "tcwv", "cape", "sst", "heat_index_c"]:
        if v in indicators:
            arr = indicators[v]
            if np.isfinite(arr).any():
                print(f"    {v:<25s}: {np.nanmin(arr):.1f} ~ {np.nanmax(arr):.1f}")

    print(f"\n  Hazard detection:")
    for hazard_type, result in hazards.items():
        print(f"    {hazard_type}: triggered={result['triggered_pct']:.1f}%, "
              f"max_score={result['max_score']:.1f}")

    # Save NetCDF
    out_path = os.path.join(date_dir, f"ifs_indicators_{date.replace('-','')}.nc")
    ds_out = xr.Dataset()
    for v, arr in indicators.items():
        if arr.ndim == 2 and arr.shape == saudi_lat.shape + saudi_lon.shape:
            ds_out[v] = xr.DataArray(arr, dims=["lat", "lon"],
                                     coords={"lat": saudi_lat, "lon": saudi_lon})
    ds_out.to_netcdf(out_path)
    print(f"\n  Saved → {out_path}")

print(f"\n{'='*70}")
print(f"  Done. Hazard indicators saved to {OUTPUT_DIR}/")
print(f"{'='*70}")
