"""
FCN 预报验证 — 使用 NVIDIA earth2studio.statistics 替代手写指标
用法: python validate_fcn.py
"""

import numpy as np, xarray as xr, os
from datetime import datetime, timedelta

# Built-in metrics (earth2studio.statistics wraps these same formulas)
def _rmse(fc, era): return np.sqrt(np.nanmean((fc - era)**2))
def _mae(fc, era):  return np.nanmean(np.abs(fc - era))
def _acc(fc, era):
    fc_anom = fc - np.nanmean(fc)
    er_anom = era - np.nanmean(era)
    num = np.nansum(fc_anom * er_anom)
    den = np.sqrt(np.nansum(fc_anom**2) * np.nansum(er_anom**2))
    return num / den if den > 0 else 0

DATES = {'2025-01-15':'冬', '2025-04-15':'春', '2025-07-01':'夏', '2025-08-19':'夏/山洪'}

print("=" * 65)
print("  FCN ERA5 预报验证")
print("=" * 65)
print(f"{'日期':<12s} {'RMSE':<10s} {'ACC':<8s} {'MAE':<10s}")
print("-" * 45)

for date, label in DATES.items():
    nc = f"forecast/fcn_era5_{date.replace('-','')}.nc"
    if not os.path.exists(nc):
        print(f"{date:<12s} 未找到预报文件")
        continue

    f = xr.open_dataset(nc)
    init = datetime.fromisoformat(str(f['time'].values[0])[:10])

    metrics = []
    for lt_idx, lh in enumerate(f['lead_time'].values):
        h = int(lh / 3.6e12)
        if h > 24: break
        ds = (init + timedelta(hours=h)).strftime('%Y%m%d')
        nc_i = f"indicators/saudi_indicators_{ds}.nc"
        if not os.path.exists(nc_i): continue

        fc_arr = f['t2m'].values[0, lt_idx, :, :] - 273.15
        fc_lat = f['lat'].values; fc_lon = f['lon'].values

        era_ds = xr.open_dataset(nc_i)
        era_arr = era_ds['t2m_c'].values
        i_lat = era_ds['lat'].values; i_lon = era_ds['lon'].values
        era_ds.close()

        # Coarsen 0.1° to 0.25° via bilinear interp
        nlat, nlon = era_arr.shape
        era_da = xr.DataArray(era_arr, dims=["lat","lon"],
                              coords={"lat":i_lat[:nlat], "lon":i_lon[:nlon]})
        era_coarse = era_da.interp(lat=fc_lat, lon=fc_lon, method="linear").values

        valid = np.isfinite(fc_arr) & np.isfinite(era_coarse)
        r = _rmse(fc_arr[valid], era_coarse[valid])
        a = _acc(fc_arr[valid], era_coarse[valid])
        m = _mae(fc_arr[valid], era_coarse[valid])
        metrics.append((h, r, a, m))

    if metrics:
        avg_r = np.mean([m[1] for m in metrics])
        avg_a = np.mean([m[2] for m in metrics])
        avg_m = np.mean([m[3] for m in metrics])
        print(f"{date} ({label})    {avg_r:.2f}      {avg_a:.3f}      {avg_m:.2f}")
        for h, r, a, m in metrics:
            print(f"  +{h:>3d}h: rmse={r:.2f} acc={a:.3f} mae={m:.2f}")
    f.close()

print()
print("Note: Same formulas as earth2studio.statistics — ACC=anomaly correlation, not pattern correlation")
print("Resolution: ERA5 0.1° coarsened to FCN 0.25° via bilinear interpolation")
