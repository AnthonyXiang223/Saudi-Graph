"""
FCN 预报准确率验证 — 与 2025 年 ERA5 实测数据逐日比对
用法:
  1. WSL2 中跑: python run_fcn.py --init 2025-07-01 --days 7
     生成 forecast/fcn_forecast.nc
  2. Windows 中跑: python validate_fcn.py
"""

import numpy as np
import xarray as xr
import os, sys, json, glob
from datetime import datetime, timedelta

FORECAST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forecast")
INDICATORS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indicators")

# FCN variable → ERA5 indicator mapping
# (FCN raw var, is_vector, ERA5 indicator)
VAR_MAP = [
    ("t2m",       False, "t2m_c",          "°C",   lambda x: x - 273.15, "2米气温"),
    ("wind_speed", True,  "wind10_speed",   "m/s",  None,                 "10米风速"),
    ("tcwv",      False, "pwat",            "mm",   None,                 "柱水汽总量"),
    ("sp",        False, "surface_pressure", "Pa",  None,                 "地表气压"),
]

# Derived indicators to also validate
DERIVED_MAP = [
    ("dewpoint_depression", "dewpoint_depression_c", "°C", "露点差"),
    ("rh2m_estimate",       "rh2m",                 "%",  "相对湿度"),
    ("precip_proxy",        "daily_precip_total",   "mm", "日降水量"),
]


def load_fcn_data(nc_path: str):
    """Load FCN forecast and extract surface variables at each lead time."""
    f = xr.open_dataset(nc_path)
    init_time = str(f["time"].values[0])[:19]
    lead_times = f["lead_time"].values / 3_600_000_000_000  # ns → hours

    fcn = {}
    for lt_idx, lt_h in enumerate(lead_times):
        date_str = _forecast_date(init_time, lt_h)
        fcn[date_str] = {}

        # Direct variables
        if "t2m" in f.variables:
            fcn[date_str]["t2m"] = f["t2m"].values[0, lt_idx, :, :]
        if "u10m" in f.variables and "v10m" in f.variables:
            u = f["u10m"].values[0, lt_idx, :, :]
            v = f["v10m"].values[0, lt_idx, :, :]
            fcn[date_str]["wind_speed"] = np.sqrt(u**2 + v**2)
            fcn[date_str]["wind_direction"] = (np.arctan2(-u, -v) * 180/np.pi) % 360
        if "tcwv" in f.variables:
            fcn[date_str]["tcwv"] = f["tcwv"].values[0, lt_idx, :, :]
        if "sp" in f.variables:
            fcn[date_str]["sp"] = f["sp"].values[0, lt_idx, :, :]

        # Derived: dewpoint depression at 850hPa
        if all(v in f.variables for v in ["t850", "r850"]):
            t8 = f["t850"].values[0, lt_idx, :, :]
            r8 = f["r850"].values[0, lt_idx, :, :]
            t8c = t8 - 273.15
            es = 6.112 * np.exp(17.67 * t8c / (t8c + 243.5))
            e  = es * np.clip(r8, 0.1, 100.0) / 100.0
            ln_e = np.log(np.maximum(e / 6.112, 1e-10))
            td8c = 243.5 * ln_e / (17.67 - ln_e)
            fcn[date_str]["dewpoint_depression"] = t8c - td8c
            fcn[date_str]["rh2m_estimate"] = r8

        # Derived: precipitation proxy (MFC-based)
        if all(v in f.variables for v in ["tcwv", "t850", "r850", "u850", "v850"]):
            fcn[date_str]["precip_proxy"] = _compute_precip_proxy(f, lt_idx)

    f.close()
    return fcn, init_time


def _forecast_date(init_str, lead_h):
    """Compute YYYYMMDD from init time + lead hours."""
    t0 = datetime.fromisoformat(init_str.replace("T", " ").split(".")[0])
    return (t0 + timedelta(hours=int(lead_h))).strftime("%Y%m%d")


def _compute_precip_proxy(f, lt_idx):
    """Replicate the MFC-based precipitation proxy from agent_tools.py."""
    tcwv = f["tcwv"].values[0, lt_idx, :, :]
    r850 = f["r850"].values[0, lt_idx, :, :]
    t850 = f["t850"].values[0, lt_idx, :, :]
    u850 = f["u850"].values[0, lt_idx, :, :]
    v850 = f["v850"].values[0, lt_idx, :, :]
    lat = f["lat"].values

    t850_c = t850 - 273.15
    es = 6.112 * np.exp(17.67 * t850_c / (t850_c + 243.5))
    e  = es * np.clip(r850, 0.1, 100.0) / 100.0
    q850 = 0.622 * e / (850.0 - 0.378 * e)

    F_u = q850 * u850
    F_v = q850 * v850

    R = 6371000.0
    lat_rad = np.deg2rad(lat)
    dlat_deg = 0.25; dlon_deg = 0.25
    m_per_deg_lat = np.deg2rad(dlat_deg) * R
    m_per_deg_lon = np.deg2rad(dlon_deg) * R * np.cos(lat_rad)

    dFu_dlon = np.gradient(F_u, dlon_deg, axis=1)
    dFv_dlat = np.gradient(F_v, dlat_deg, axis=0)
    dFu_dx = dFu_dlon / m_per_deg_lon[:, np.newaxis]
    dFv_dy = dFv_dlat / m_per_deg_lat
    MFC = -(dFu_dx + dFv_dy)
    MFC_pos = np.maximum(MFC, 0.0)
    sat_factor = (r850 / 100.0) ** 2
    return np.clip(tcwv * MFC_pos * 86400.0 * sat_factor * 5.0, 0.0, 200.0)


def load_era5_data(date_str: str, indicator: str):
    """Load one indicator from ERA5 NetCDF file."""
    nc_path = os.path.join(INDICATORS_DIR, f"saudi_indicators_{date_str}.nc")
    if not os.path.exists(nc_path):
        return None
    ds = xr.open_dataset(nc_path)
    if indicator in ds.data_vars:
        data = ds[indicator].values
    elif indicator in ds.variables:
        data = ds[indicator].values
    else:
        data = None
    ds.close()
    return data


def compare(fcn_val, era5_val):
    """Compute error metrics between FCN and ERA5 arrays."""
    if fcn_val is None or era5_val is None:
        return None

    # Ensure same shape
    if fcn_val.shape != era5_val.shape:
        era5_val = _crop_to_match(era5_val, fcn_val.shape)

    valid = np.isfinite(fcn_val) & np.isfinite(era5_val)
    if valid.sum() < 100:
        return None

    fc = fcn_val[valid]
    er = era5_val[valid]

    bias = float(np.mean(fc - er))
    rmse = float(np.sqrt(np.mean((fc - er)**2)))
    mae  = float(np.mean(np.abs(fc - er)))
    corr = float(np.corrcoef(fc, er)[0, 1]) if len(fc) > 1 else 0

    return {"bias": round(bias, 3), "rmse": round(rmse, 3),
            "mae": round(mae, 3), "corr": round(corr, 3),
            "n_valid": int(valid.sum())}


def _crop_to_match(arr, target_shape):
    """Crop array to match target shape (for lat/lon mismatch)."""
    if arr.ndim == 2:
        return arr[:target_shape[0], :target_shape[1]]
    return arr


def run_validation(fcn_nc: str = None):
    """Main validation routine."""
    if fcn_nc is None:
        fcn_nc = os.path.join(FORECAST_DIR, "fcn_forecast.nc")

    if not os.path.exists(fcn_nc):
        print(f"FCN 预报文件不存在: {fcn_nc}")
        print("先在 WSL2 中运行: python run_fcn.py --init 2025-07-01 --days 7")
        return

    print("=" * 65)
    print("  FCN 预报准确率验证")
    print("=" * 65)

    # 1. Load FCN forecast
    print("\n1. 加载 FCN 预报...")
    fcn_data, init_time = load_fcn_data(fcn_nc)
    dates = sorted(fcn_data.keys())
    print(f"   初始化: {init_time}")
    print(f"   预报日期: {dates[0]} ~ {dates[-1]} ({len(dates)} 天)")

    # 2. Compare each date
    print("\n2. 逐日比对 (FCN vs ERA5 再分析)...")
    all_results = {}

    for var_id, is_vec, era_ind, unit, transform, label in VAR_MAP:
        print(f"\n--- {label} ({var_id} → {era_ind}) [{unit}] ---")
        daily = []
        for date_str in dates:
            fcn_val = fcn_data[date_str].get(var_id)
            if fcn_val is not None and transform:
                fcn_val = transform(fcn_val)
            era_val = load_era5_data(date_str, era_ind)
            metrics = compare(fcn_val, era_val)
            if metrics:
                daily.append({"date": date_str, **metrics})
                print(f"  {date_str}: bias={metrics['bias']:+.2f}{unit} "
                      f"rmse={metrics['rmse']:.2f} corr={metrics['corr']:.2f}")

        if daily:
            avg_bias = np.mean([d["bias"] for d in daily])
            avg_rmse = np.mean([d["rmse"] for d in daily])
            avg_corr = np.mean([d["corr"] for d in daily])
            all_results[label] = {
                "avg_bias": round(avg_bias, 3),
                "avg_rmse": round(avg_rmse, 3),
                "avg_corr": round(avg_corr, 3),
                "n_days": len(daily),
            }

    # 3. Summary
    print("\n" + "=" * 65)
    print("  验证总结")
    print("=" * 65)
    print(f"{'指标':<12s} {'天数':<6s} {'平均偏差':<10s} {'RMSE':<10s} {'相关系数':<8s}")
    print("-" * 50)
    for label, m in all_results.items():
        print(f"{label:<12s} {m['n_days']:<6d} {m['avg_bias']:+.3f}     "
              f"{m['avg_rmse']:.3f}      {m['avg_corr']:.3f}")

    # 4. Save JSON
    out_path = os.path.join(FORECAST_DIR, "validation_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "fc_init_time": init_time,
            "validation_dates": dates,
            "summary": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {out_path}")


if __name__ == "__main__":
    run_validation()
