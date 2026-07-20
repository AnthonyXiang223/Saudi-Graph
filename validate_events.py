"""
验证 ERA5 indicator 灾害检测 vs 2025年真实事件
"""
import json, os, sys, numpy as np, xarray as xr
from datetime import datetime, timedelta
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── 2025年沙特真实灾害事件 ──
DUST_EVENTS = [
    ("2025-05-04", "2025-05-05", "卡西姆/利雅得巨型哈布尘暴"),
    ("2025-05-16", "2025-05-19", "全年最强持续沙尘(全国)"),
    ("2025-06-30", "2025-07-05", "东部/汉志持续沙尘"),
]

FLOOD_EVENTS = [
    ("2025-01-06", "2025-01-07", "麦加/吉达特大洪水"),
    ("2025-03-06", "2025-03-07", "哈伊勒/布赖代山洪"),
    ("2025-08-14", "2025-08-14", "塔伊夫冰雹洪水"),
    ("2025-08-27", "2025-08-28", "阿西尔/吉赞/纳季兰大范围山洪"),
    ("2025-12-09", "2025-12-10", "吉达历史性特大洪水(179mm/6h)"),
]

# ── 加载 rules.json ──
with open("schema/rules.json", encoding='utf-8') as f:
    rules_data = json.load(f)

# 找到 dust_storm 和 flash_flood 的条件
def get_rule_conditions(hazard_type):
    for rule in rules_data["rules"]:
        if rule["hazard_type"] == hazard_type:
            return rule["conditions"]
    return []

dust_conditions = get_rule_conditions("dust_storm")
flood_conditions = get_rule_conditions("flash_flood")

print("Dust storm conditions:")
for c in dust_conditions:
    print(f"  {c['indicator']} {c['op']} {c['value']} (weight={c['weight']})")
print("\nFlash flood conditions:")
for c in flood_conditions:
    print(f"  {c['indicator']} {c['op']} {c['value']} (weight={c['weight']})")

# ── 加载 indicator 数据 ──
def load_indicator(date_str, wanted_vars):
    path = f"indicators/saudi_indicators_{date_str}.nc"
    if not os.path.exists(path): return None
    ds = xr.open_dataset(path)
    result = {}
    for v in wanted_vars:
        if v in ds:
            arr = ds[v].values
            if arr.ndim > 2: arr = arr[0]  # first time step if multi-time
            result[v] = arr.astype(np.float64)
    ds.close()
    return result

def compute_dust_score(indicators, conditions):
    """计算沙尘暴得分 (0=无风险, 1=最高)"""
    score = np.zeros_like(list(indicators.values())[0])
    triggered = False
    for c in conditions:
        ind = c["indicator"]
        if ind not in indicators: continue
        arr = indicators[ind]
        op = c["op"]; th = c["value"]; w = c["weight"]
        if op == ">=": hit = arr >= th
        elif op == ">": hit = arr > th
        elif op == "<=": hit = arr <= th
        elif op == "<": hit = arr < th
        else: continue
        score += w * hit.astype(float)
    return score

def compute_flood_score(indicators, conditions):
    """计算山洪得分"""
    score = np.zeros_like(list(indicators.values())[0])
    for c in conditions:
        ind = c["indicator"]
        if ind not in indicators: continue
        arr = indicators[ind]
        op = c["op"]; th = c["value"]; w = c["weight"]
        if op == ">=": hit = arr >= th
        elif op == ">": hit = arr > th
        elif op == "<=": hit = arr <= th
        elif op == "<": hit = arr < th
        else: continue
        score += w * hit.astype(float)
    return score

# ── 验证沙尘暴 ──
print(f"\n{'='*70}")
print("  DUST STORM VALIDATION")
print(f"{'='*70}")

dust_wanted = set(c["indicator"] for c in dust_conditions)
for start, end, desc in DUST_EVENTS:
    print(f"\n  [{start} ~ {end}] {desc}")
    dates = []
    d = datetime.strptime(start, "%Y-%m-%d")
    while d <= datetime.strptime(end, "%Y-%m-%d"):
        dates.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)

    max_scores = []
    all_triggers = []
    for date_c in dates:
        ind = load_indicator(date_c, dust_wanted)
        if ind is None:
            print(f"    {date_c}: 无数据")
            continue
        score = compute_dust_score(ind, dust_conditions)
        max_score = float(score.max())
        pct_triggered = float((score >= 0.3).mean() * 100)  # >30% 总分视为触发
        max_scores.append(max_score)
        all_triggers.append(pct_triggered)

        # Per-indicator breakdown
        details = []
        for c in dust_conditions:
            if c["indicator"] in ind:
                arr = ind[c["indicator"]]
                details.append(f"{c['indicator']}={np.nanmean(arr):.1f}")
        print(f"    {date_c}: max_score={max_score:.2f}, trigger={pct_triggered:.1f}% | " + " | ".join(details))

    if max_scores:
        print(f"    → 期间最高分={max(max_scores):.2f}, 平均触发={np.mean(all_triggers):.1f}%")

# ── 验证山洪 ──
print(f"\n{'='*70}")
print("  FLASH FLOOD VALIDATION")
print(f"{'='*70}")

flood_wanted = set(c["indicator"] for c in flood_conditions)
for start, end, desc in FLOOD_EVENTS:
    print(f"\n  [{start} ~ {end}] {desc}")
    dates = []
    d = datetime.strptime(start, "%Y-%m-%d")
    while d <= datetime.strptime(end, "%Y-%m-%d"):
        dates.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)

    max_scores = []
    for date_c in dates:
        ind = load_indicator(date_c, flood_wanted)
        if ind is None:
            print(f"    {date_c}: 无数据")
            continue
        score = compute_flood_score(ind, flood_conditions)
        max_score = float(score.max())
        pct_triggered = float((score >= 0.3).mean() * 100)
        max_scores.append(max_score)

        details = []
        for c in flood_conditions:
            if c["indicator"] in ind:
                arr = ind[c["indicator"]]
                details.append(f"{c['indicator']}={np.nanmean(arr):.1f}")
        print(f"    {date_c}: max_score={max_score:.2f}, trigger={pct_triggered:.1f}% | " + " | ".join(details))

    if max_scores:
        print(f"    → 期间最高分={max(max_scores):.2f}")
