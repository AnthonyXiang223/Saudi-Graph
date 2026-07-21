"""
IFS Pipeline integration for agent_tools.py
Loads pre-computed IFS indicator NetCDF → runs hazard detection
"""
import os, json, numpy as np, xarray as xr

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
IFS_DIR = os.path.join(PROJECT_DIR, "aifs_forecasts")
SCHEMA_DIR = os.path.join(PROJECT_DIR, "schema")

def list_ifs_dates():
    """List available IFS forecast dates."""
    if not os.path.exists(IFS_DIR):
        return []
    dates = []
    for d in sorted(os.listdir(IFS_DIR)):
        subdir = os.path.join(IFS_DIR, d)
        if not os.path.isdir(subdir):
            continue
        # Match both old (no step suffix) and new (with step suffix) patterns
        has_data = any(
            f.startswith("ifs_indicators_") and f.endswith(".nc")
            for f in os.listdir(subdir)
        )
        if has_data:
            dates.append(d)
    return dates

def load_indicators_ifs(date_str, forecast_day=0):
    """Load IFS indicators from pre-computed NetCDF.

    Args:
        date_str: 'YYYYMMDD' init date
        forecast_day: day offset (0=analysis, 1=+24h, etc.)

    Returns: {"indicators": dict, "lat": array, "lon": array, "missing": list}
    """
    # Try new pattern (with step suffix) first, then old pattern
    step_h = forecast_day * 24
    nc_path = os.path.join(IFS_DIR, date_str, f"ifs_indicators_{date_str}_{step_h}h.nc")
    if not os.path.exists(nc_path):
        # Fallback: old pattern without step suffix (analysis only)
        nc_path = os.path.join(IFS_DIR, date_str, f"ifs_indicators_{date_str}.nc")
    if not os.path.exists(nc_path):
        # List available steps and pick closest
        subdir = os.path.join(IFS_DIR, date_str)
        if os.path.isdir(subdir):
            available = [f for f in os.listdir(subdir)
                        if f.startswith("ifs_indicators_") and f.endswith(".nc")]
            if available:
                import logging
                logging.getLogger('mazu.ifs').warning(
                    'Exact forecast hour %dh not found for %s, falling back to %s',
                    step_h, date_str, available[0])
                nc_path = os.path.join(subdir, available[0])
            else:
                return None
        else:
            return None

    ds = xr.open_dataset(nc_path)
    lat = ds["lat"].values
    lon = ds["lon"].values

    ind = {}
    for v in ds.data_vars:
        arr = ds[v].values
        if arr.ndim > 2:
            arr = arr[0]  # take first time/step
        ind[v] = arr.astype(np.float64)

    # ── Fill missing derived indicators ──
    # pwat = tcwv (total column water vapour ≈ precipitable water)
    if "tcwv" in ind and "pwat" not in ind:
        ind["pwat"] = ind["tcwv"]
    # t2m_c = t2m (same thing)
    if "t2m" in ind and "t2m_c" not in ind:
        ind["t2m_c"] = ind["t2m"]
    # tmax_c: use t2m for analysis field, or derive from steps
    if "t2m" in ind and "tmax_c" not in ind:
        ind["tmax_c"] = ind["t2m"]
    # sst_celsius: from sst if available, else from skt over ocean
    if "sst" in ind and "sst_celsius" not in ind:
        ind["sst_celsius"] = ind["sst"]
    # t2m_anomaly_c: approximate from field mean
    if "t2m" in ind and "t2m_anomaly_c" not in ind:
        ind["t2m_anomaly_c"] = ind["t2m"] - np.nanmean(ind["t2m"])
    # heatwave_day_flag: derived
    if "tmax_c" in ind and "heatwave_day_flag" not in ind:
        ind["heatwave_day_flag"] = (ind["tmax_c"] >= 40).astype(np.float64)
    # flash_flood_risk: derived (simplified 0-5 score)
    if "flash_flood_risk" not in ind:
        ff_risk = np.zeros_like(ind.get("t2m", np.zeros((65,89))))
        # +1 for daily_precip > 10mm, +1 for precip > 5mm, +1 for pwat > 30, +1 for pwat > 20, +1 for precip_percentile > 80
        if "daily_precip_total" in ind:
            ff_risk += (ind["daily_precip_total"] >= 10).astype(float)
        if "pwat" in ind:
            ff_risk += (ind["pwat"] >= 30).astype(float)
        if "rh2m" in ind:
            ff_risk += (ind["rh2m"] >= 70).astype(float)
        ind["flash_flood_risk"] = ff_risk

    # ── Probability gating indicators (from climatology) ──
    from scipy.stats import gamma as gamma_dist
    # precip_percentile
    if "precip_percentile" not in ind and "daily_precip_total" in ind:
        clim_path = os.path.join(PROJECT_DIR, "forecast", "precip_climatology.nc")
        if os.path.exists(clim_path):
            cds = xr.open_dataset(clim_path)
            # Interpolate to IFS grid
            shape_da = xr.DataArray(cds["gamma_shape"].values, dims=["lat","lon"],
                                    coords={"lat": cds["lat"].values, "lon": cds["lon"].values})
            scale_da = xr.DataArray(cds["gamma_scale"].values, dims=["lat","lon"],
                                    coords={"lat": cds["lat"].values, "lon": cds["lon"].values})
            shape_if = shape_da.interp(lat=lat, lon=lon, method="linear").values
            scale_if = scale_da.interp(lat=lat, lon=lon, method="linear").values
            valid = (shape_if > 0) & (scale_if > 0)
            pct = np.zeros_like(ind["daily_precip_total"])
            pct[valid] = gamma_dist.cdf(ind["daily_precip_total"][valid],
                                        a=shape_if[valid], scale=scale_if[valid]) * 100.0
            ind["precip_percentile"] = pct
            cds.close()

    # ── heat_gpd_prob (GPD extreme value) ──
    if "heat_gpd_prob" not in ind and "tmax_c" in ind:
        clim_path = os.path.join(PROJECT_DIR, "forecast", "heat_gpd_climatology.nc")
        if os.path.exists(clim_path):
            cds = xr.open_dataset(clim_path)
            def _interp(var_name):
                da = xr.DataArray(cds[var_name].values, dims=["lat","lon"],
                                  coords={"lat": cds["lat"].values, "lon": cds["lon"].values})
                return da.interp(lat=lat, lon=lon, method="linear").values
            thresh = _interp("gpd_threshold")
            shape  = _interp("gpd_shape")
            scale  = _interp("gpd_scale")
            exc_r  = _interp("exceedance_rate")
            prob = np.ones_like(ind["tmax_c"])
            exceed = ind["tmax_c"] > thresh
            if exceed.any():
                exc_val = ind["tmax_c"][exceed] - thresh[exceed]
                pe = np.exp(-exc_val / np.maximum(scale[exceed], 1.0))
                prob[exceed] = np.clip(pe * exc_r[exceed], 0, 1)
            ind["heat_gpd_prob"] = prob
            cds.close()

    # ── dust_joint_prob (Copula min) ──
    if "dust_joint_prob" not in ind and "wind10_speed" in ind and "dewpoint_depression_c" in ind:
        clim_path = os.path.join(PROJECT_DIR, "forecast", "dust_joint_climatology.nc")
        if os.path.exists(clim_path):
            cds = xr.open_dataset(clim_path)
            def _interp(var_name):
                da = xr.DataArray(cds[var_name].values, dims=["percentile","lat","lon"],
                                  coords={"percentile": cds["percentile"].values,
                                          "lat": cds["lat"].values, "lon": cds["lon"].values})
                return da.interp(lat=lat, lon=lon, method="linear").values  # (n_pct, ny, nx)
            from scipy.interpolate import interp1d
            wind_pct  = _interp("wind10_pct")
            dew_pct   = _interp("dewpoint_pct")
            rh_pct    = _interp("rh2m_pct")
            pct_vals  = cds["percentile"].values
            n_pct, ny, nx = wind_pct.shape
            f_wind = np.zeros((ny, nx)); f_dew = np.zeros_like(f_wind)
            f_rh = np.zeros_like(f_wind)
            for i in range(ny):
                for j in range(nx):
                    def _lookup(val, curve):
                        try:
                            f = interp1d(curve[:,i,j], pct_vals, bounds_error=False,
                                         fill_value=(pct_vals[0], pct_vals[-1]))
                            return float(np.clip(f(val[i,j]), 0, 1))
                        except: return 0.0
                    f_wind[i,j] = _lookup(ind["wind10_speed"], wind_pct)
                    f_dew[i,j]  = _lookup(ind["dewpoint_depression_c"], dew_pct)
                    f_rh[i,j]   = _lookup(100 - ind["rh2m"], rh_pct)
            ind["dust_joint_prob"] = np.minimum(np.minimum(f_wind, f_dew), f_rh)
            cds.close()

    # ── humid_heat_joint_prob (Copula min, coastal only) ──
    if "humid_heat_joint_prob" not in ind and "t2m" in ind and "rh2m" in ind:
        clim_path = os.path.join(PROJECT_DIR, "forecast", "humid_heat_joint_climatology.nc")
        if os.path.exists(clim_path):
            cds = xr.open_dataset(clim_path)
            def _interp(var_name):
                da = xr.DataArray(cds[var_name].values, dims=["percentile","lat","lon"],
                                  coords={"percentile": cds["percentile"].values,
                                          "lat": cds["lat"].values, "lon": cds["lon"].values})
                return da.interp(lat=lat, lon=lon, method="linear").values
            from scipy.interpolate import interp1d
            rh_pct_h  = _interp("rh2m_pct")
            t2m_pct_h = _interp("t2m_pct")
            pct_vals  = cds["percentile"].values
            n_pct, ny, nx = rh_pct_h.shape
            f_rh_h = np.zeros((ny, nx)); f_t2m_h = np.zeros_like(f_rh_h)
            for i in range(ny):
                for j in range(nx):
                    def _lookup(val, curve):
                        try:
                            f = interp1d(curve[:,i,j], pct_vals, bounds_error=False,
                                         fill_value=(pct_vals[0], pct_vals[-1]))
                            return float(np.clip(f(val[i,j]), 0, 1))
                        except: return 0.0
                    f_rh_h[i,j]  = _lookup(ind["rh2m"], rh_pct_h)
                    f_t2m_h[i,j] = _lookup(ind["t2m"], t2m_pct_h)
            ind["humid_heat_joint_prob"] = np.minimum(f_rh_h, f_t2m_h)
            cds.close()

    ds.close()

    # Check what's missing vs rules.json requirements
    with open(os.path.join(SCHEMA_DIR, "rules.json"), encoding="utf-8") as f:
        rules_data = json.load(f)

    all_needed = set()
    for rule in rules_data["rules"]:
        for cond in rule["conditions"]:
            all_needed.add(cond["indicator"])

    missing = sorted(all_needed - set(ind.keys()))

    return {
        "indicators": ind,
        "lat": lat,
        "lon": lon,
        "missing": missing,
        "lead_time_h": forecast_day * 24,
    }

def _eval_condition(arr, op, th):
    """Evaluate a single condition, returning a boolean mask. NaN-safe."""
    if arr is None:
        return np.zeros(1, dtype=bool)  # fallback
    valid = ~np.isnan(arr)
    result = np.zeros(arr.shape, dtype=bool)
    if op == ">=":
        result[valid] = arr[valid] >= th
    elif op == ">":
        result[valid] = arr[valid] > th
    elif op == "<=":
        result[valid] = arr[valid] <= th
    elif op == "<":
        result[valid] = arr[valid] < th
    return result


def _get_coastal_mask(lat_vals, lon_vals):
    """Return boolean mask for Red Sea + Persian Gulf coastal grid cells."""
    mask = np.zeros((len(lat_vals), len(lon_vals)), dtype=bool)
    for i in range(len(lat_vals)):
        for j in range(len(lon_vals)):
            lat, lon = lat_vals[i], lon_vals[j]
            in_red_sea = (16 <= lat <= 30) and (34 <= lon <= 42)
            in_gulf = (24 <= lat <= 30) and (48 <= lon <= 56)
            mask[i, j] = in_red_sea or in_gulf
    return mask


def detect_ifs_hazards(ifs_data, hazard_types=None):
    """Run hazard detection on IFS indicators with proper primary-gate logic.

    Key improvements over v1:
    - PRIMARY GATE: if primary condition NOT met, score × 0.25 (suppression)
    - PROBABILISTIC BYPASS: if probabilistic gate triggers, suppresses the primary-gate penalty
    - COASTAL FILTER: coastal_humid_heat only evaluated on Red Sea / Persian Gulf cells
    - NaN-safe: NaN indicator values treated as "condition not met"

    Args:
        ifs_data: return value from load_indicators_ifs()
        hazard_types: list of hazard types, None = all

    Returns: list of hazard results
    """
    import numpy as np

    with open(os.path.join(SCHEMA_DIR, "rules.json"), encoding="utf-8") as f:
        rules_data = json.load(f)

    if hazard_types is None:
        hazard_types = ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]

    indicators = ifs_data["indicators"]
    lat_vals = ifs_data["lat"]
    lon_vals = ifs_data["lon"]
    ref_shape = list(indicators.values())[0].shape
    coastal_mask = _get_coastal_mask(lat_vals, lon_vals)

    results = []
    for htype in hazard_types:
        rule = next((r for r in rules_data["rules"] if r["hazard_type"] == htype), None)
        if not rule:
            continue

        # Separate conditions by role
        primary_cond = None
        prob_gate_cond = None
        regular_conds = []

        for cond in rule["conditions"]:
            role = cond.get("role", "")
            if role == "derived_gate" or cond.get("primary"):
                primary_cond = cond
            elif role == "probabilistic_gate":
                prob_gate_cond = cond
            else:
                regular_conds.append(cond)

        # Check availability
        available_conds = [c for c in rule["conditions"] if c["indicator"] in indicators]
        unavailable_conds = [c["indicator"] for c in rule["conditions"] if c["indicator"] not in indicators]

        # Also check primary/prob gate availability
        primary_available = primary_cond and primary_cond["indicator"] in indicators
        prob_gate_available = prob_gate_cond and prob_gate_cond["indicator"] in indicators

        if len(available_conds) < 2:
            results.append({
                "hazard_type": htype,
                "detected": False,
                "coverage": f"{len(available_conds)}/{len(rule['conditions'])}",
                "reason": "insufficient_indicators",
            })
            continue

        # ── Compute weighted score ──
        score = np.zeros(ref_shape, dtype=float)
        total_w = 0.0
        triggered = []

        for cond in available_conds:
            arr = indicators[cond["indicator"]]
            op = cond["op"]
            th = cond["value"]
            w = cond.get("weight", 1.0)

            hit = _eval_condition(arr, op, th)
            score += w * hit.astype(float)
            total_w += w
            if w > 0.2:
                triggered.append(cond["indicator"])

        if total_w > 0:
            score /= total_w

        # ── Primary gate + Probabilistic bypass ──
        primary_hit = None
        prob_gate_hit = None

        if primary_available:
            arr = indicators[primary_cond["indicator"]]
            primary_hit = _eval_condition(arr, primary_cond["op"], primary_cond["value"])

        if prob_gate_available:
            arr = indicators[prob_gate_cond["indicator"]]
            prob_gate_hit = _eval_condition(arr, prob_gate_cond["op"], prob_gate_cond["value"])

        if primary_hit is not None:
            # Cells where primary is NOT met: suppress by ×0.25
            # UNLESS the probabilistic gate triggers (bypass)
            not_primary = ~primary_hit
            if prob_gate_hit is not None:
                # Prob gate bypass: don't suppress cells where prob gate triggers
                bypass = prob_gate_hit
                suppress = not_primary & (~bypass)
            else:
                suppress = not_primary
            score[suppress] *= 0.25

        # ── Region filter (coastal humid heat only) ──
        if htype == "coastal_humid_heat":
            score[~coastal_mask] = 0.0

        # ── Severity thresholds ──
        thresholds = rule.get("severity", [
            {"label": "low", "range": [0.0, 0.3]},
            {"label": "medium", "range": [0.3, 0.6]},
            {"label": "high", "range": [0.6, 0.8]},
            {"label": "extreme", "range": [0.8, 1.0]},
        ])

        max_level = "none"
        max_score_val = float(score.max()) if score.size > 0 else 0.0
        for sev in thresholds:
            lo, hi = sev["range"]
            if max_score_val >= lo:
                max_level = sev["label"]

        n_cells = int(np.sum(score >= 0.3))
        n_total = int(np.sum(coastal_mask)) if htype == "coastal_humid_heat" else score.size
        pct_cells = float(n_cells / n_total * 100) if n_total > 0 else 0.0

        # Hotspots
        if n_cells > 0:
            hotspot_idx = np.unravel_index(np.argmax(score), score.shape)
            hotspot_lat = float(lat_vals[hotspot_idx[0]])
            hotspot_lon = float(lon_vals[hotspot_idx[1]])

            from scipy import ndimage
            mask = score >= 0.3
            labeled, n_clusters = ndimage.label(mask)
            cluster_sizes = [int(np.sum(labeled == i)) for i in range(1, n_clusters + 1)]
        else:
            hotspot_lat = None
            hotspot_lon = None
            n_clusters = 0
            cluster_sizes = []

        hotspot_region = _identify_region(hotspot_lat, hotspot_lon) if hotspot_lat else None

        # Gate detail for transparency
        gate_detail = {}
        if primary_hit is not None:
            gate_detail["primary_met_cells"] = int(np.sum(primary_hit))
            gate_detail["primary_met_pct"] = round(float(np.mean(primary_hit) * 100), 1)
        if prob_gate_hit is not None:
            gate_detail["prob_gate_met_cells"] = int(np.sum(prob_gate_hit))
            gate_detail["prob_gate_met_pct"] = round(float(np.mean(prob_gate_hit) * 100), 1)

        results.append({
            "hazard_type": htype,
            "detected": n_cells > 0,
            "severity": max_level,
            "max_score": round(float(score.max()), 4),
            "mean_score": round(float(score.mean()), 4),
            "cells_triggered": n_cells,
            "total_cells": n_total,
            "triggered_pct": round(pct_cells, 1),
            "n_clusters": n_clusters,
            "cluster_sizes": cluster_sizes[:5],
            "hotspot": f"{hotspot_lat:.1f}N, {hotspot_lon:.1f}E" if hotspot_lat else None,
            "hotspot_region": hotspot_region,
            "coverage": f"{len(available_conds)}/{len(rule['conditions'])}",
            "unavailable": unavailable_conds,
            "primary_triggers": triggered,
            "gate_detail": gate_detail,
        })

    return results


def evaluate_city_hazards(city_name, city_info, ifs_data, hazard_types=None):
    """Evaluate hazards at a specific city's grid point (NOT regional hotspot).

    Args:
        city_name: e.g. 'mecca'
        city_info: dict with 'lat', 'lon', 'label', 'region'
        ifs_data: return value from load_indicators_ifs()
        hazard_types: list of hazard types, None = all

    Returns: dict with per-hazard results at the city's grid cell
    """
    import numpy as np

    with open(os.path.join(SCHEMA_DIR, "rules.json"), encoding="utf-8") as f:
        rules_data = json.load(f)

    if hazard_types is None:
        hazard_types = ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]

    indicators = ifs_data["indicators"]
    lat_vals = ifs_data["lat"]
    lon_vals = ifs_data["lon"]
    coastal_mask = _get_coastal_mask(lat_vals, lon_vals)

    # Find nearest grid point
    lat_idx = int(np.argmin(np.abs(lat_vals - city_info["lat"])))
    lon_idx = int(np.argmin(np.abs(lon_vals - city_info["lon"])))
    grid_lat = float(lat_vals[lat_idx])
    grid_lon = float(lon_vals[lon_idx])
    is_coastal = bool(coastal_mask[lat_idx, lon_idx])

    city_results = {
        "city": city_name,
        "label": city_info.get("label", city_name),
        "grid_point": f"{grid_lat:.2f}N, {grid_lon:.2f}E",
        "is_coastal": is_coastal,
        "lead_time_h": ifs_data.get("lead_time_h", "?"),
        "hazards": {},
    }

    for htype in hazard_types:
        rule = next((r for r in rules_data["rules"] if r["hazard_type"] == htype), None)
        if not rule:
            continue

        # Separate conditions by role (same logic as detect_ifs_hazards)
        primary_cond = None
        prob_gate_cond = None

        for cond in rule["conditions"]:
            role = cond.get("role", "")
            if role == "derived_gate" or cond.get("primary"):
                primary_cond = cond
            elif role == "probabilistic_gate":
                prob_gate_cond = cond

        # Evaluate each condition at this grid cell
        cond_results = []
        score = 0.0
        total_w = 0.0

        for cond in rule["conditions"]:
            ind_name = cond["indicator"]
            if ind_name not in indicators:
                cond_results.append({
                    "indicator": ind_name,
                    "value": None,
                    "threshold": f"{cond['op']} {cond['value']}",
                    "met": None,
                    "reason": "unavailable",
                    "weight": cond.get("weight", 0),
                })
                continue

            val = indicators[ind_name][lat_idx, lon_idx]
            if hasattr(val, "item"):
                val = val.item()
            op = cond["op"]
            th = cond["value"]
            w = cond.get("weight", 1.0)

            if val is None or (isinstance(val, float) and np.isnan(val)):
                met = False
                reason = "NaN"
            else:
                met = _eval_condition_at_point(val, op, th)
                reason = "met" if met else "not_met"

            if met:
                score += w
            total_w += w

            cond_results.append({
                "indicator": ind_name,
                "value": round(val, 2) if isinstance(val, (int, float)) and not (isinstance(val, float) and np.isnan(val)) else None,
                "threshold": f"{op} {th}",
                "met": met,
                "weight": w,
            })

        # Normalize
        if total_w > 0:
            score /= total_w

        # Primary gate logic at point
        primary_met = True  # default if no primary
        prob_gate_met = False

        if primary_cond and primary_cond["indicator"] in indicators:
            pv = indicators[primary_cond["indicator"]][lat_idx, lon_idx]
            if hasattr(pv, "item"):
                pv = pv.item()
            if pv is None or (isinstance(pv, float) and np.isnan(pv)):
                primary_met = False
            else:
                primary_met = _eval_condition_at_point(pv, primary_cond["op"], primary_cond["value"])

        if prob_gate_cond and prob_gate_cond["indicator"] in indicators:
            gv = indicators[prob_gate_cond["indicator"]][lat_idx, lon_idx]
            if hasattr(gv, "item"):
                gv = gv.item()
            if gv is not None and not (isinstance(gv, float) and np.isnan(gv)):
                prob_gate_met = _eval_condition_at_point(gv, prob_gate_cond["op"], prob_gate_cond["value"])

        if not primary_met and not prob_gate_met:
            score *= 0.25

        # Region filter for coastal humid heat
        if htype == "coastal_humid_heat" and not is_coastal:
            score = 0.0

        # Severity
        thresholds = rule.get("severity", [])
        severity = "none"
        for sev in thresholds:
            lo, hi = sev["range"]
            if score >= lo:
                severity = sev["label"]

        city_results["hazards"][htype] = {
            "score": round(score, 4),
            "severity": severity,
            "conditions": cond_results,
            "primary_met": primary_met,
            "prob_gate_met": prob_gate_met,
        }

    return city_results


def _eval_condition_at_point(val, op, th):
    """Evaluate a single condition at a scalar value."""
    if op == ">=":
        return val >= th
    elif op == ">":
        return val > th
    elif op == "<=":
        return val <= th
    elif op == "<":
        return val < th
    return False


def _identify_region(lat, lon):
    """Identify Saudi region from lat/lon."""
    if 24 <= lat <= 32 and 36 <= lon <= 51:
        return "北部/中部内陆"
    elif 21 <= lat < 24 and 39 <= lon <= 55:
        return "中部(利雅得地区)"
    elif 16 <= lat < 21 and 42 <= lon <= 55:
        return "西南部(阿西尔/吉赞)"
    elif 16 <= lat < 23 and 34 <= lon < 42:
        return "红海沿岸(吉达/麦加)"
    elif 23 <= lat < 30 and 34 <= lon < 40:
        return "西北部(塔布克)"
    elif 26 <= lat <= 32 and 48 <= lon <= 55:
        return "东部省(达曼/波斯湾)"
    elif 24 <= lat <= 32 and 40 <= lon < 48:
        return "中北部(卡西姆/哈伊勒)"
    return "未知区域"


def build_ifs_forecast_report(hazards, ifs_data, forecast_day, location=None):
    """Build a structured report from IFS hazard detection results."""
    output = {
        "forecast_source": "ECMWF IFS (AWS S3, 0.25deg)",
        "forecast_day": forecast_day,
        "lead_time_h": ifs_data.get("lead_time_h", "?"),
        "available_indicators": sorted(ifs_data["indicators"].keys()),
        "missing_indicators": ifs_data.get("missing", []),
        "hazards": hazards,
        "ifs_coverage": f"{len(ifs_data['indicators'])} indicators loaded",
    }

    if location:
        for h in hazards:
            if h.get("detected"):
                h["location_note"] = f"关注地点 {location} 在{h.get('hotspot_region', '附近')}区域"

    return output
