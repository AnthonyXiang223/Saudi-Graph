"""
Scan all 365 NetCDF files to compute per-indicator availability statistics.

Outputs: writes availability data back to operators.json.
"""

import os
import re
import json
import sys
import numpy as np
import xarray as xr
from collections import defaultdict
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')


def scan_indicators(data_dir: str, operators_path: str) -> dict:
    """
    Scan all NetCDF files and compute availability for each indicator.

    Args:
        data_dir: Path to indicators directory
        operators_path: Path to operators.json (read + write back)

    Returns:
        dict: {indicator_id: {"effective_days": N, "total_days": M}}
    """
    files = sorted([
        f for f in os.listdir(data_dir)
        if f.startswith("saudi_indicators_") and f.endswith(".nc")
    ])

    total_days = len(files)
    print(f"Scanning {total_days} files from {data_dir}...")

    # Track how many days each indicator has non-NaN data
    indicator_days = defaultdict(int)
    file_dates = []

    for i, fname in enumerate(files):
        date_match = re.search(r'saudi_indicators_(\d{8})\.nc', fname)
        date_str = date_match.group(1) if date_match else fname
        file_dates.append(date_str)

        ds = xr.open_dataset(os.path.join(data_dir, fname))
        var_names = [v for v in ds.variables if v not in ("latitude", "longitude", "time", "lat", "lon")]

        for var in var_names:
            data = ds[var].values
            # Count as "effective" if at least one grid cell has valid data
            if data.size > 0 and np.any(~np.isnan(data)):
                indicator_days[var] += 1

        ds.close()

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{total_days} files scanned...")

    print(f"  Complete: {total_days} files scanned")

    # Build availability dict
    availability = {}
    for indicator, days in sorted(indicator_days.items()):
        availability[indicator] = {
            "effective_days": days,
            "total_days": total_days,
            "coverage_pct": round(days / total_days * 100, 1),
        }

    # Report
    full_coverage = sum(1 for v in availability.values() if v["effective_days"] == total_days)
    partial_coverage = sum(1 for v in availability.values() if 0 < v["effective_days"] < total_days)
    missing = sum(1 for v in availability.values() if v["effective_days"] == 0)

    print(f"\nResults:")
    print(f"  Total indicators found in NetCDF: {len(availability)}")
    print(f"  Full coverage ({total_days}/{total_days} days): {full_coverage}")
    print(f"  Partial coverage: {partial_coverage}")
    print(f"  Zero data: {missing}")

    if partial_coverage > 0:
        print(f"\n  Partial coverage indicators:")
        for ind, v in sorted(availability.items()):
            if 0 < v["effective_days"] < total_days:
                print(f"    {ind}: {v['effective_days']}/{total_days} ({v['coverage_pct']}%)")

    # Write back to operators.json
    if os.path.exists(operators_path):
        with open(operators_path, "r", encoding="utf-8") as f:
            ops_data = json.load(f)

        updated = 0
        for op in ops_data.get("operators", []):
            ind_id = op["id"]
            if ind_id in availability:
                op["availability"] = availability[ind_id]
                updated += 1

        with open(operators_path, "w", encoding="utf-8") as f:
            json.dump(ops_data, f, indent=2, ensure_ascii=False)

        print(f"\n  Updated {updated} operators with availability data in {operators_path}")

    # Also check which operators.json indicators are NOT in NetCDF
    with open(operators_path, "r", encoding="utf-8") as f:
        ops_data = json.load(f)

    nc_vars = set(availability.keys())
    json_ids = {op["id"] for op in ops_data.get("operators", [])}
    missing_in_nc = json_ids - nc_vars
    extra_in_nc = nc_vars - json_ids

    if missing_in_nc:
        print(f"\n  WARNING: {len(missing_in_nc)} indicators in operators.json NOT found in NetCDF:")
        for ind in sorted(missing_in_nc):
            print(f"    - {ind}")

    if extra_in_nc:
        print(f"\n  NOTE: {len(extra_in_nc)} variables in NetCDF NOT in operators.json:")
        for ind in sorted(extra_in_nc):
            print(f"    - {ind}")

    return availability


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "indicators")
    operators_path = os.path.join(project_dir, "schema", "operators.json")

    if not os.path.isdir(data_dir):
        print(f"ERROR: data directory not found: {data_dir}")
        sys.exit(1)

    availability = scan_indicators(data_dir, operators_path)
