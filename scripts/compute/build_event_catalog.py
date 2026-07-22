"""
Build historical event catalog from 365 ERA5 days.

Runs EventDetector over all saudi_indicators_YYYYMMDD.nc files and serializes
results into lightweight queryable artifacts:

  kg_data/event_catalog.parquet  — per-event rows (date, hazard_type, severity, etc.)
  kg_data/grid_event_counts.npz  — grid cell × month × hazard statistics

Once built, ALL KG-forecast calibration queries run against these files —
zero ERA5 NetCDF loading at runtime.

Usage:
  python scripts/compute/build_event_catalog.py              # full 365-day build
  python scripts/compute/build_event_catalog.py --days 30    # last 30 days only (test)
  python scripts/compute/build_event_catalog.py --date 20250701  # single day
"""

import os, sys, json, glob, time, argparse
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)

from kg.datalayer import DataLayer
from kg.event_detector import EventDetector


def main():
    parser = argparse.ArgumentParser(description="Build KG event catalog from ERA5 indicators")
    parser.add_argument("--days", type=int, default=0, help="Process only last N days (0=all)")
    parser.add_argument("--date", type=str, default=None, help="Process single date YYYYMMDD")
    parser.add_argument("--output-dir", type=str, default="kg_data", help="Output directory")
    args = parser.parse_args()

    # ── Paths ──
    indicators_dir = os.path.join(PROJECT_DIR, "indicators")
    output_dir = os.path.join(PROJECT_DIR, args.output_dir)
    rules_path = os.path.join(PROJECT_DIR, "schema", "rules.json")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load rules ──
    with open(rules_path, "r", encoding="utf-8") as f:
        rules_data = json.load(f)
    rules = rules_data["rules"]

    # ── Find indicator files ──
    nc_files = sorted(glob.glob(os.path.join(indicators_dir, "saudi_indicators_*.nc")))
    if not nc_files:
        print(f"ERROR: No indicator files found in {indicators_dir}")
        return

    dates_all = [os.path.basename(f).replace("saudi_indicators_", "").replace(".nc", "")
                 for f in nc_files]

    if args.date:
        if args.date in dates_all:
            dates_all = [args.date]
        else:
            print(f"ERROR: Date {args.date} not found in indicators/")
            return

    if args.days > 0:
        dates_all = dates_all[-args.days:]

    ndays = len(dates_all)
    print(f"Building event catalog: {ndays} days ({dates_all[0]} → {dates_all[-1]})")

    # ── Init detector ──
    datalayer = DataLayer(indicators_dir)
    detector = EventDetector(rules, datalayer)

    # ── Process all days ──
    all_events = []  # list of dicts for parquet
    grid_event_count = np.zeros((160, 220, 4, 12), dtype=np.int16)   # [lat, lon, hazard, month]
    grid_severity_sum = np.zeros((160, 220, 4, 12), dtype=np.float32)

    hazard_idx = {"flash_flood": 0, "extreme_heat": 1, "dust_storm": 2, "coastal_humid_heat": 3}
    severities = {"none": 0, "low": 1, "caution": 1, "medium": 2, "moderate": 2,
                  "warning": 2, "high": 3, "alert": 3, "extreme": 4, "emergency": 4, "severe": 4}

    t_start = time.time()
    n_events_total = 0
    n_failures = 0

    for day_idx, date_str in enumerate(dates_all):
        try:
            events = detector.detect_events(date_str)
        except Exception as e:
            n_failures += 1
            if n_failures <= 3:
                print(f"  [{day_idx+1}/{ndays}] {date_str} FAILED: {e}")
            continue

        month = int(date_str[4:6]) - 1  # 0-based month index

        for evt in events:
            n_events_total += 1
            all_events.append({
                "date": date_str,
                "hazard_type": evt.hazard_type,
                "severity": evt.severity,
                "severity_score": round(evt.severity_score, 4),
                "confidence": round(evt.confidence, 4),
                "centroid_lat": round(evt.centroid_lat, 3),
                "centroid_lon": round(evt.centroid_lon, 3),
                "area_km2": round(evt.area_km2, 1),
                "n_cells": len(evt.affected_cells),
                "peak_risk": round(evt.peak_risk, 4),
                "region": evt.region,
                "cluster_id": evt.cluster_id,
            })

            # Update grid statistics
            hi = hazard_idx.get(evt.hazard_type, -1)
            sev_val = severities.get(evt.severity, 1)
            for lat_i, lon_i in evt.affected_cells:
                if 0 <= lat_i < 160 and 0 <= lon_i < 220:
                    grid_event_count[lat_i, lon_i, hi, month] += 1
                    grid_severity_sum[lat_i, lon_i, hi, month] += sev_val

        # Progress
        elapsed = time.time() - t_start
        rate = (day_idx + 1) / elapsed if elapsed > 0 else 0
        eta = (ndays - day_idx - 1) / rate if rate > 0 else 0
        if (day_idx + 1) % 30 == 0 or day_idx == ndays - 1:
            print(f"  [{day_idx+1}/{ndays}] {date_str} — "
                  f"{n_events_total} events so far, {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    # ── Compute derived statistics ──
    # Count available days per month (for frequency denominator)
    days_per_month = np.zeros(12, dtype=np.int16)
    for d in dates_all:
        days_per_month[int(d[4:6]) - 1] += 1

    # cell_event_frequency = P(event day | month at this cell)
    # Smooth with a tiny prior to avoid division by zero
    cell_event_frequency = np.zeros((160, 220, 4, 12), dtype=np.float32)
    mean_severity = np.zeros((160, 220, 4, 12), dtype=np.float32)
    for m in range(12):
        if days_per_month[m] > 0:
            cell_event_frequency[:, :, :, m] = (
                grid_event_count[:, :, :, m].astype(np.float32) / days_per_month[m]
            )
            denom = np.maximum(grid_event_count[:, :, :, m], 1)
            mean_severity[:, :, :, m] = grid_severity_sum[:, :, :, m] / denom

    # ── Save ──
    # Parquet event catalog
    parquet_path = os.path.join(output_dir, "event_catalog.parquet")
    try:
        import pandas as pd
        df = pd.DataFrame(all_events)
        df.to_parquet(parquet_path, index=False)
        print(f"\nSaved {len(df)} events to {parquet_path}")
    except ImportError:
        # Fallback: save as JSON lines
        jsonl_path = os.path.join(output_dir, "event_catalog.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for evt in all_events:
                f.write(json.dumps(evt, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(all_events)} events to {jsonl_path} (pandas not available)")

    # NPZ grid statistics
    npz_path = os.path.join(output_dir, "grid_event_counts.npz")
    np.savez_compressed(
        npz_path,
        event_count=grid_event_count,
        severity_sum=grid_severity_sum,
        cell_event_frequency=cell_event_frequency,
        mean_severity=mean_severity,
        days_per_month=days_per_month,
        hazard_order=np.array(["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]),
        rules_hash=hash(json.dumps(rules_data["rules"], sort_keys=True)),
    )
    print(f"Saved grid statistics to {npz_path}")

    # ── Validation ──
    print(f"\n{'='*60}")
    print(f"Build complete: {elapsed:.0f}s total, {ndays} days, {n_events_total} events")
    if n_failures:
        print(f"WARNING: {n_failures} days failed to process")
    print(f"Output: {output_dir}/")
    print(f"  event_catalog.parquet  — {len(all_events)} event rows")
    print(f"  grid_event_counts.npz  — {160}x{220}x4x12 statistics")

    # Quick sanity checks
    print(f"\nSanity checks:")
    for htype, hi in hazard_idx.items():
        total_events = int(grid_event_count[:, :, hi, :].sum())
        top_month = int(np.argmax(grid_event_count[:, :, hi, :].sum(axis=(0, 1))))
        print(f"  {htype:20s}: {total_events:5d} total events, peak month={top_month+1}")


if __name__ == "__main__":
    main()
