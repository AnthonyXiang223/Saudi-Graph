#!/bin/bash
# ERA5 批量 FCN 预报 + 验证
# 用法: bash run_batch_validation.sh

DATES="2025-01-15 2025-04-15 2025-07-01 2025-08-19"

echo "=== Step 1: 下载 ERA5 ==="
for d in $DATES; do
    nc="forecast/fcn_era5_${d//-/}.nc"
    if [ -f "$nc" ]; then
        echo "$d: 已有预报，跳过下载"
        continue
    fi
    echo "下载 $d ..."
    python download_era5.py --date $d
done

echo ""
echo "=== Step 2: 运行 FCN ==="
for d in $DATES; do
    nc="forecast/fcn_era5_${d//-/}.nc"
    if [ -f "$nc" ]; then
        echo "$d: 已存在，跳过"
        continue
    fi
    echo "FCN 预报 $d ..."
    python run_fcn_local.py --date $d --days 1
done

echo ""
echo "=== Step 3: 批量验证 ==="
python -W ignore -c "
import numpy as np, xarray as xr, os
from datetime import datetime, timedelta

DATES = '${DATES}'.split()
SEASONS = {'2025-01-15':'冬','2025-04-15':'春','2025-07-01':'夏','2025-08-19':'夏/山洪'}

print('=' * 65)
print('  ERA5->FCN 多日准确率验证')
print('  (偏差 = FCN预报 - ERA5实测, °C)')
print('=' * 65)

all_bias = []
for date in DATES:
    nc = f'forecast/fcn_era5_{date.replace(\"-\",\"\")}.nc'
    if not os.path.exists(nc): continue
    f = xr.open_dataset(nc)
    init = datetime.fromisoformat(str(f['time'].values[0])[:10])
    lead_h = f['lead_time'].values / 3.6e12

    biases = []
    for lt_idx, lh in enumerate(lead_h):
        if lh > 24: break
        ds = (init + timedelta(hours=int(lh))).strftime('%Y%m%d')
        nc_i = f'indicators/saudi_indicators_{ds}.nc'
        if not os.path.exists(nc_i): continue

        fc = f['t2m'].values[0, lt_idx, :, :] - 273.15
        era_ds = xr.open_dataset(nc_i)
        era = era_ds['t2m_c'].values
        era_ds.close()
        if era.shape != fc.shape: era = era[:fc.shape[0],:fc.shape[1]]
        valid = np.isfinite(fc) & np.isfinite(era)
        if valid.sum() < 100: continue

        bias = np.mean(fc[valid] - era[valid])
        rmse = np.sqrt(np.mean((fc[valid] - era[valid])**2))
        corr = np.corrcoef(fc[valid], era[valid])[0,1]
        biases.append((int(lh), bias, rmse, corr))

    if biases:
        avg_bias = np.mean([b[1] for b in biases])
        avg_rmse = np.mean([b[2] for b in biases])
        avg_corr = np.mean([b[3] for b in biases])
        all_bias.append(avg_bias)
        season = SEASONS.get(date, '?')
        print(f'{date} ({season}): bias={avg_bias:+.2f} rmse={avg_rmse:.2f} corr={avg_corr:.3f}')
        for b in biases:
            print(f'  +{b[0]:<3d}h bias={b[1]:+.2f} rmse={b[2]:.2f} corr={b[3]:.3f}')
    f.close()

if all_bias:
    print(f'\n汇总: bias={np.mean(all_bias):+.2f} rmse={np.mean([b[2] for date in [d for d in DATES if f])]:.2f} (N={len(all_bias)})')
    print(f'GFS源对照: bias=+1.83 rmse=4.92')
"
