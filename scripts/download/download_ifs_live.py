"""
Download TODAY's IFS forecast from ECMWF Open Data (free, no key)
→ Extract Saudi → Compute indicators → Hazard detection → Save to aifs_forecasts/
"""
import subprocess, sys, os
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ecmwf-opendata", "cfgrib", "xarray", "netcdf4", "scipy"])

import numpy as np, xarray as xr, json
from datetime import datetime, timedelta
from collections import OrderedDict
from ecmwf.opendata import Client

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # scripts/download/ -> project root
sys.path.insert(0, SCRIPT_DIR)  # to import download_aifs
OUTPUT_DIR = os.path.join(PROJECT_DIR, "aifs_forecasts")

# ── Config ──
STEPS = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168]  # 7天
GRID_LAT = np.arange(16.0, 32.25, 0.25)
GRID_LON = np.arange(34.0, 56.25, 0.25)

# Import extraction & detection from download_aifs
from download_aifs import (
    extract_saudi, compute_indicators, compute_prob_indicators,
    detect_hazards, load_climatologies
)

# ── 1. Get latest date ──
client = Client(source="ecmwf")
latest = client.latest()  # datetime
print(f"Latest IFS run: {latest}")
date_str = latest.strftime("%Y-%m-%d")
date_c   = latest.strftime("%Y%m%d")

date_dir = os.path.join(OUTPUT_DIR, date_c)
os.makedirs(date_dir, exist_ok=True)

# ── 2. Load climatologies once ──
print("Loading climatology files...")
CLIM = load_climatologies(GRID_LAT, GRID_LON)

# ── 3. Download & process each step ──
print(f"\n{'='*60}")
print(f"  Processing {date_str}")
print(f"{'='*60}")

for step_h in STEPS:
    print(f"\n  --- Step +{step_h}h ---")
    grib_path = os.path.join(date_dir, f"{date_c}000000-{step_h}h-oper-fc.grib2")

    # Download from ECMWF Open Data
    if not os.path.exists(grib_path) or os.path.getsize(grib_path) < 1000:
        try:
            print(f"    Downloading...")
            client.retrieve(
                date=date_str,
                time=0,
                step=step_h,
                param=["2t","2d","10u","10v","msl","sp","tcwv","tp","skt","lsm","mucape"],
                target=grib_path,
            )
            print(f"    {os.path.getsize(grib_path)/1e6:.1f} MB")
        except Exception as e:
            print(f"    Download failed: {e}")
            continue

    # Extract
    try:
        raw_vars, s_lat, s_lon = extract_saudi(grib_path, step_h)
    except Exception as e:
        print(f"    Extract failed: {e}")
        continue

    # Compute indicators
    indicators = compute_indicators(raw_vars)
    try:
        indicators = compute_prob_indicators(indicators, CLIM)
    except Exception:
        pass

    # Summary
    tp_val = indicators.get("daily_precip_total", np.zeros(1))
    print(f"    t2m={indicators.get('t2m',np.zeros(1)).mean():.1f}C, "
          f"tp_max={np.nanmax(tp_val):.1f}mm, "
          f"precip_pct_max={np.nanmax(indicators.get('precip_percentile',np.zeros(1))):.0f}%")

    # Hazard detection
    try:
        hazards = detect_hazards(indicators)
        for ht, result in hazards.items():
            if result["triggered_pct"] > 0:
                print(f"    {ht}: {result['severity']} (score={result['max_score']:.2f}, "
                      f"trigger={result['triggered_pct']:.1f}%)")
    except Exception:
        pass

    # Save NetCDF
    step_out = os.path.join(date_dir, f"ifs_indicators_{date_c}_{step_h}h.nc")
    ds_out = xr.Dataset()
    for v, arr in indicators.items():
        if arr.ndim == 2 and arr.shape == s_lat.shape + s_lon.shape:
            ds_out[v] = xr.DataArray(arr, dims=["lat", "lon"],
                                     coords={"lat": s_lat, "lon": s_lon})
    ds_out.to_netcdf(step_out)

print(f"\nDone → {date_dir}/")
