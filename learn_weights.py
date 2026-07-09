"""
从 365 天 NetCDF 数据中学习规则权重 — 使用已知极端日期 vs 静默日期作为标签。
"""
import sys, os, json, re
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import xarray as xr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

PROJECT = r"f:\Saudi"
DATA_DIR = os.path.join(PROJECT, "indicators")
SCHEMA_DIR = os.path.join(PROJECT, "schema")

# ── 已知极端日期（报告第四章） ──
EXTREME_DATES = {
    "flash_flood": ["20250819", "20250820", "20250821", "20250822", "20250823"],
    "extreme_heat": ["20250717", "20250725", "20250726"],
}
QUIET_DATES = ["20250101", "20250105", "20250110", "20250115", "20250120", "20250125",
               "20250201", "20250205", "20250210", "20250215"]

# ── 配置 ──
HAZARD_CONFIG = {
    "flash_flood": {
        "indicators": ["daily_precip_total", "cape", "wind10_speed",
                        "ds10_max_1h", "ivt_convergence", "pwat"],
        "current_weights": [0.32, 0.24, 0.12, 0.18, 0.14, 0.00],
    },
    "extreme_heat": {
        # 只用因果指标，不用后果指标 (heat_index_c 是 T+RH 算出来的结果，不是原因)
        "indicators": ["tmax_c", "t2m_anomaly_c", "heatwave_duration_days",
                        "vpd_kpa", "dewpoint_depression_c", "rh2m"],
        "current_weights": [0.35, 0.20, 0.25, 0.12, 0.00, 0.08],
    },
}

STRIDE = 5  # 每 5 个格点取 1 个

def load_day_data(date_str, indicators):
    """加载一天的数据，返回 (features, grid_shape) 或 None"""
    path = os.path.join(DATA_DIR, f"saudi_indicators_{date_str}.nc")
    if not os.path.exists(path):
        return None

    ds = xr.open_dataset(path)
    arrays = []
    for ind in indicators:
        if ind not in ds.variables:
            ds.close(); return None
        data = ds[ind].values
        if data.ndim >= 3:
            data = data.mean(axis=0) if data.shape[0] > 1 else data[0]
            while data.ndim > 2: data = data[0]
        arrays.append(data)

    if "latitude" in ds.dims:
        nlat, nlon = len(ds["latitude"]), len(ds["longitude"])
    else:
        nlat, nlon = len(ds["lat"]), len(ds["lon"])
    ds.close()

    feat_2d = np.stack(arrays, axis=-1)  # (nlat, nlon, n_features)
    return feat_2d, nlat, nlon


# ── 收集样本 ──
print("收集极端 vs 静默样本...")
X = {h: [] for h in HAZARD_CONFIG}
y = {h: [] for h in HAZARD_CONFIG}

for hazard, config in HAZARD_CONFIG.items():
    indicators = config["indicators"]
    extreme_dates = EXTREME_DATES[hazard]

    # 极端日期样本 (label=1)
    for date_str in extreme_dates:
        result = load_day_data(date_str, indicators)
        if result is None:
            continue
        feat_2d, nlat, nlon = result
        feat_sampled = feat_2d[::STRIDE, ::STRIDE, :].reshape(-1, len(indicators))
        X[hazard].append(feat_sampled)
        y[hazard].append(np.ones(len(feat_sampled), dtype=int))

    # 静默日期样本 (label=0)
    for date_str in QUIET_DATES:
        result = load_day_data(date_str, indicators)
        if result is None:
            continue
        feat_2d, nlat, nlon = result
        feat_sampled = feat_2d[::STRIDE, ::STRIDE, :].reshape(-1, len(indicators))
        X[hazard].append(feat_sampled)
        y[hazard].append(np.zeros(len(feat_sampled), dtype=int))

    if X[hazard]:
        X[hazard] = np.vstack(X[hazard])
        y[hazard] = np.concatenate(y[hazard])
        # 剔除 NaN
        valid = ~np.isnan(X[hazard]).any(axis=1)
        X[hazard] = X[hazard][valid]
        y[hazard] = y[hazard][valid]
        print(f"  {hazard}: {len(X[hazard])} 样本, 正例={y[hazard].sum()} ({y[hazard].sum()/len(y[hazard])*100:.1f}%)")
    else:
        print(f"  {hazard}: 无有效样本")


# ── 学习权重 ──
print("\n===== 逻辑回归权重学习 =====\n")
learned_weights = {}

for hazard, config in HAZARD_CONFIG.items():
    if hazard not in X or len(X[hazard]) == 0:
        print(f"  {hazard}: 跳过")
        continue

    X_data = X[hazard]
    y_data = y[hazard]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)

    model = LogisticRegression(penalty='l1', solver='saga', C=0.5,
                                max_iter=2000, class_weight='balanced', random_state=42)
    model.fit(X_scaled, y_data)
    y_pred = model.predict(X_scaled)
    f1 = f1_score(y_data, y_pred, zero_division=0)

    coef = model.coef_[0]
    coef_abs = np.abs(coef)
    learned = coef_abs / coef_abs.sum() if coef_abs.sum() > 0 else np.ones(len(config["indicators"])) / len(config["indicators"])
    learned_weights[hazard] = learned.tolist()

    print(f"\n  {hazard} (F1={f1:.4f}, indicators={len(config['indicators'])}, "
          f"learned={len(learned)}, manual={len(config['current_weights'])})")
    print(f"  {'指标':30s} {'手动权重':>8s} {'学习权重':>8s} {'方向':>4s}")
    print(f"  {'─'*55}")
    for i, ind in enumerate(config["indicators"]):
        manual = config["current_weights"][i]
        lw = round(learned[i], 4)
        direction = "+" if coef[i] > 0 else "-" if coef[i] < 0 else "0"
        diff = " ⚠" if abs(manual - lw) > 0.10 else ""
        print(f"  {ind:30s} {manual:>8.3f} {lw:>8.4f} {direction:>4s}{diff}")


# ── 更新 rules.json ──
print(f"\n===== 更新 rules.json =====")

with open(os.path.join(SCHEMA_DIR, "rules.json"), "r", encoding="utf-8") as f:
    rules_data = json.load(f)

MAP_HAZARD = {"flash_flood": "flash_flood", "extreme_heat": "extreme_heat"}
updated = 0
for rule in rules_data["rules"]:
    htype = rule["hazard_type"]
    if htype not in learned_weights:
        continue
    learned = learned_weights[htype]
    for i, cond in enumerate(rule["conditions"]):
        if i < len(learned):
            old_w = cond["weight"]
            new_w = max(0.05, round(learned[i], 3))  # 最小 0.05，避免归零
            cond["weight"] = new_w
            diff = f"({old_w} → {new_w})"
            print(f"  {rule['id']}/{cond['indicator']}: {diff}")
            updated += 1

rules_data["meta"]["weight_source"] = "logistic_regression_L1_on_known_extreme_vs_quiet_dates"
rules_data["meta"]["last_updated"] = "2026-07-09"
rules_data["meta"]["note"] = "标签: 报告已知极端日期(20250819-23/20250717-25)为1, 静默日期(Jan-Feb)为0. 特征: 连续值, 标准化后 L1 正则逻辑回归. 样本: 每5格点采样."

with open(os.path.join(SCHEMA_DIR, "rules.json"), "w", encoding="utf-8") as f:
    json.dump(rules_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ 已更新 {updated} 个条件权重")
print(f"✅ 来源: {rules_data['meta']['weight_source']}")
