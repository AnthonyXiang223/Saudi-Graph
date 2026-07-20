"""
Event detection engine for Saudi extreme event knowledge graph.

Reads pre-computed indicators from the data layer, applies weighted detection
rules from the KG, performs connected-component labeling, and generates Event nodes.

Does NOT re-compute indicators — that's compute_indicators.py's job.
Does read flash_flood_risk (basic layer) and applies weighted upgrade logic.
"""

import numpy as np
from scipy.ndimage import label as connected_components
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime


@dataclass
class Event:
    """A detected extreme event instance (one per connected component)."""
    event_id: str
    date: str
    hazard_type: str
    severity: str
    severity_score: float
    confidence: float
    cluster_id: int
    affected_cells: List[Tuple[int, int]]  # (lat_idx, lon_idx)
    area_km2: float
    peak_risk: float
    centroid_lat: float
    centroid_lon: float
    region: str
    trigger_details: List[Dict] = field(default_factory=list)


class EventDetector:
    """
    Applies detection rules to data and generates Event nodes.

    Reads base indicators (including flash_flood_risk from compute_indicators.py),
    then applies weighted scoring, connected-component labeling, severity grading,
    and fallback strategies for missing indicators.
    """

    # Supported comparison operators
    COMPARATORS = {
        ">=": lambda a, b: a >= b,
        ">":  lambda a, b: a > b,
        "<=": lambda a, b: a <= b,
        "<":  lambda a, b: a < b,
        "==": lambda a, b: np.isclose(a, b),
        "!=": lambda a, b: ~np.isclose(a, b),
    }

    def __init__(self, rules, datalayer):
        """
        Args:
            rules: list of rule dicts from rules.json
            datalayer: DataLayer instance (for loading xarray data)
        """
        self.rules = rules
        self.data = datalayer
        self.events: List[Event] = []
        self._event_counter = 0
        self._climatology = None  # Lazy-loaded precip climatology
        self._dust_climatology = None  # Lazy-loaded dust copula climatology
        self._heat_climatology = None  # Lazy-loaded heat GPD climatology
        self._humid_climatology = None  # Lazy-loaded humid heat copula

    def detect_events(self, date_str: str,
                      hazard_types: Optional[List[str]] = None) -> List[Event]:
        """
        Main entry point: detect extreme events on a given date.

        Args:
            date_str: Date string "YYYYMMDD" or "YYYY-MM-DD"
            hazard_types: List of hazard types to check, or None for all

        Returns:
            List of Event objects detected
        """
        date_str = date_str.replace("-", "")
        ds = self.data.load_day(date_str)
        events = []

        if hazard_types is None:
            hazard_types = ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]

        for htype in hazard_types:
            rule = self._get_rule(htype)
            if rule is None:
                continue
            detected = self._apply_rule(date_str, ds, rule)
            events.extend(detected)

        self.events = events
        return events

    def _get_rule(self, hazard_type: str) -> Optional[Dict]:
        """Look up the detection rule for a hazard type from rules.json."""
        for rule in self.rules:
            if rule.get("hazard_type") == hazard_type:
                return rule
        return None

    def _apply_rule(self, date_str: str, ds, rule: Dict) -> List[Event]:
        """
        Apply a single detection rule to the dataset.

        Steps:
        1. Extract each condition's indicator data from the xarray Dataset
        2. Handle missing indicators via fallback strategy
        3. Compute weighted risk score per grid cell
        4. Threshold and label connected components
        5. Generate Event objects for qualifying clusters
        """
        conditions = rule.get("conditions", [])
        connectivity = rule.get("connectivity", 4)
        min_cluster_size = rule.get("min_cluster_size", 3)
        severity_levels = rule.get("severity", [])
        fallback = rule.get("fallback", {})

        lat = ds["latitude"].values if "latitude" in ds else ds["lat"].values
        lon = ds["longitude"].values if "longitude" in ds else ds["lon"].values
        nlat, nlon = len(lat), len(lon)

        # Step 1-2: Evaluate each condition
        risk_score = np.zeros((nlat, nlon))
        total_weight = 0.0
        confidence = 1.0
        trigger_details = []
        primary_mask = None  # Gate: primary condition must be met
        prob_gate_mask = None  # Probabilistic gate: bypass primary gate when triggered

        for cond in conditions:
            indicator = cond["indicator"]
            comp_op = cond.get("condition", cond.get("op"))
            threshold = cond["value"]
            weight = cond.get("weight", 0.0)
            is_primary = cond.get("primary", False)
            role = cond.get("role", "")

            # ── Handle humid_heat_joint_prob (Copula probabilistic gate) ──
            if indicator == "humid_heat_joint_prob":
                hh_data = self._compute_humid_heat_joint(ds, lat, lon, nlat, nlon)
                if hh_data is not None:
                    mask = self._eval_condition(hh_data, comp_op, threshold)
                    prob_gate_mask = (prob_gate_mask | mask) if prob_gate_mask is not None else mask
                    trigger_details.append({
                        "indicator": indicator,
                        "condition": f"{comp_op} {threshold}",
                        "weight": weight,
                        "primary": False,
                        "role": role,
                        "cells_triggered": int(mask.sum()),
                        "peak_value": float(np.nanmin(hh_data)) if np.any(~np.isnan(hh_data)) else None,
                        "status": "evaluated",
                        "note": "湿热Copula联合概率: P=min(F_sst,F_rh,F_t2m,F_1/wind), 仅沿海格点有效"
                    })
                    risk_score += weight * mask.astype(float)
                    total_weight += weight
                else:
                    trigger_details.append({
                        "indicator": indicator,
                        "status": "missing",
                        "action": "humid_heat_joint_climatology.nc not found"
                    })
                continue

            # ── Handle heat_gpd_prob (GPD probabilistic gate) ──
            if indicator == "heat_gpd_prob":
                gpd_data = self._compute_heat_gpd(ds, lat, lon, nlat, nlon)
                if gpd_data is not None:
                    # heat_gpd_prob uses "<=" (P<=0.05 means extreme)
                    mask = self._eval_condition(gpd_data, comp_op, threshold)
                    prob_gate_mask = (prob_gate_mask | mask) if prob_gate_mask is not None else mask
                    trigger_details.append({
                        "indicator": indicator,
                        "condition": f"{comp_op} {threshold}",
                        "weight": weight,
                        "primary": False,
                        "role": role,
                        "cells_triggered": int(mask.sum()),
                        "peak_value": float(np.nanmin(gpd_data)) if np.any(~np.isnan(gpd_data)) else None,
                        "status": "evaluated",
                        "note": "GPD极端高温概率: P<=0.05触发, P<=0.01极端。基于Peaks-Over-Threshold模型。"
                    })
                    risk_score += weight * mask.astype(float)
                    total_weight += weight
                else:
                    trigger_details.append({
                        "indicator": indicator,
                        "status": "missing",
                        "action": "heat_gpd_climatology.nc not found, GPD gate disabled"
                    })
                continue

            # ── Handle dust_joint_prob (Copula probabilistic gate) ──
            if indicator == "dust_joint_prob":
                joint_data = self._compute_dust_joint_prob(ds, lat, lon, nlat, nlon)
                if joint_data is not None:
                    mask = self._eval_condition(joint_data, comp_op, threshold)
                    prob_gate_mask = (prob_gate_mask | mask) if prob_gate_mask is not None else mask
                    trigger_details.append({
                        "indicator": indicator,
                        "condition": f"{comp_op} {threshold}",
                        "weight": weight,
                        "primary": False,
                        "role": role,
                        "cells_triggered": int(mask.sum()),
                        "peak_value": float(np.nanmax(joint_data)) if np.any(~np.isnan(joint_data)) else None,
                        "status": "evaluated",
                        "note": "Copula联合概率: P=min(F_wind,F_dew,F_1/rh,F_shear), 触发时绕过wind10_speed门控"
                    })
                    risk_score += weight * mask.astype(float)
                    total_weight += weight
                else:
                    trigger_details.append({
                        "indicator": indicator,
                        "status": "missing",
                        "action": "dust_joint_climatology.nc not found, copula gate disabled"
                    })
                continue

            # ── Handle precip_percentile (probabilistic gate) ──
            if indicator == "precip_percentile":
                pct_data = self._compute_precip_percentile(ds, lat, lon, nlat, nlon)
                if pct_data is not None:
                    mask = self._eval_condition(pct_data, comp_op, threshold)
                    prob_gate_mask = mask
                    trigger_details.append({
                        "indicator": indicator,
                        "condition": f"{comp_op} {threshold}",
                        "weight": weight,
                        "primary": False,
                        "role": role,
                        "cells_triggered": int(mask.sum()),
                        "peak_value": float(np.nanmax(pct_data)) if np.any(~np.isnan(pct_data)) else None,
                        "status": "evaluated",
                        "note": "概率化门控: 触发时绕过flash_flood_risk门控"
                    })
                    # Also contribute to risk score
                    risk_score += weight * mask.astype(float)
                    total_weight += weight
                else:
                    trigger_details.append({
                        "indicator": indicator,
                        "status": "missing",
                        "action": "precip_climatology.nc not found, probabilistic gate disabled"
                    })
                continue

            # Check if indicator is available in the dataset
            if indicator not in ds.variables:
                if indicator in fallback.get("if_missing", []):
                    strategy = fallback.get("strategy", "drop_condition")
                    if strategy == "drop_condition":
                        confidence -= fallback.get("confidence_penalty", 0.1)
                        trigger_details.append({
                            "indicator": indicator,
                            "status": "missing",
                            "action": f"dropped (fallback), confidence penalty applied"
                        })
                        continue
                    elif strategy == "use_alternative":
                        alt = fallback.get("alternative", {}).get(indicator)
                        if alt and alt in ds.variables:
                            indicator = alt
                        else:
                            continue
                else:
                    trigger_details.append({
                        "indicator": indicator,
                        "status": "missing",
                        "action": "skipped, no fallback"
                    })
                    continue

            # Get data and evaluate condition
            data = ds[indicator].values
            # Handle multi-dimensional data: take first time step or mean
            if data.ndim >= 3:
                # SST has (time, lat, lon) — take mean over time
                data = data.mean(axis=0) if data.shape[0] > 1 else data[0]
                while data.ndim > 2:
                    data = data[0]  # fallback squeeze
            # Handle shape mismatch (SST uses lat/lon dims that may differ)
            if data.shape != (nlat, nlon):
                data = data[:nlat, :nlon]

            mask = self._eval_condition(data, comp_op, threshold)

            # Weighted contribution
            risk_score += weight * mask.astype(float)
            total_weight += weight

            # Track primary mask for final gating
            if is_primary:
                if primary_mask is None:
                    primary_mask = mask
                else:
                    primary_mask = primary_mask | mask

            # Record for explainability
            trigger_details.append({
                "indicator": indicator,
                "condition": f"{comp_op} {threshold}",
                "weight": weight,
                "primary": is_primary,
                "cells_triggered": int(mask.sum()),
                "peak_value": float(np.nanmax(data)) if np.any(~np.isnan(data)) else None,
                "status": "evaluated"
            })

        # Final gate: if primary condition not met, cap risk at 0.25
        # BUT: cells where probabilistic gate triggered are treated as gate-passed
        if primary_mask is not None:
            # Merge probabilistic gate into primary: either traditional gate OR prob gate passes
            effective_gate = primary_mask
            if prob_gate_mask is not None:
                effective_gate = primary_mask | prob_gate_mask
            risk_score = np.where(effective_gate, risk_score, risk_score * 0.25)
            risk_score = np.clip(risk_score, 0, 1.0)

        # Normalize risk score
        if total_weight > 0:
            risk_score = risk_score / total_weight
        else:
            return []  # No valid conditions

        # Step 4: Threshold for binary mask
        # Use the SECOND severity level's lower bound as minimum (skip "none" level)
        # If only one level, use its lower bound; otherwise use level[1].lo
        if len(severity_levels) >= 2:
            min_severity = severity_levels[1]["range"][0]
        elif len(severity_levels) >= 1:
            min_severity = severity_levels[0]["range"][0]
        else:
            min_severity = 0.3
        mask = risk_score >= min_severity

        # Step 5: Connected component labeling
        if connectivity == 4:
            structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        else:  # 8-connectivity
            structure = np.ones((3, 3))

        clusters, n_clusters = connected_components(mask, structure=structure)

        # Step 6: Generate Event objects
        events = []
        for cid in range(1, n_clusters + 1):
            ys, xs = np.where(clusters == cid)
            if len(ys) < min_cluster_size:
                continue

            self._event_counter += 1
            cluster_risk = risk_score[ys, xs]
            severity, severity_score = self._assign_severity(cluster_risk.max(), severity_levels)

            # Approximate area: 0.1 deg * 111 km/deg at equator
            # More accurate: lat-dependent
            centroid_lat = lat[ys].mean()
            km_per_deg_lat = 111.32
            km_per_deg_lon = 111.32 * np.cos(np.radians(centroid_lat))
            cell_area_km2 = 0.1 * km_per_deg_lat * 0.1 * km_per_deg_lon
            area_km2 = len(ys) * cell_area_km2

            region = self._infer_region(centroid_lat, lon[xs].mean())

            event = Event(
                event_id=f"event_{date_str}_{rule['hazard_type']}_{self._event_counter:03d}",
                date=date_str,
                hazard_type=rule["hazard_type"],
                severity=severity,
                severity_score=round(severity_score, 3),
                confidence=round(confidence, 3),
                cluster_id=cid,
                affected_cells=list(zip(ys.tolist(), xs.tolist())),
                area_km2=round(area_km2, 1),
                peak_risk=round(float(cluster_risk.max()), 3),
                centroid_lat=round(float(centroid_lat), 2),
                centroid_lon=round(float(lon[xs].mean()), 2),
                region=region,
                trigger_details=trigger_details,
            )
            events.append(event)

        return events

    def _compute_humid_heat_joint(self, ds, lat, lon, nlat, nlon):
        """
        Compute coastal humid heat joint probability via Copula.

        P = min(F_sst, F_rh, F_t2m, F_1/wind)
        where F_1/wind is the rank of negative wind speed (low wind → extreme).
        Only meaningful for coastal cells (Red Sea + Persian Gulf).

        Returns:
            2D array of joint probability (0-1), lower = more extreme, or None.
        """
        import os

        if self._humid_climatology is None:
            clim_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "forecast", "humid_heat_joint_climatology.nc")
            if not os.path.exists(clim_path):
                return None
            import xarray as xr
            self._humid_climatology = xr.open_dataset(clim_path)

        # Indicator → (clim_var, direction)
        ind_map = {
            "sst_celsius":  ("sst_pct",  1),
            "rh2m":         ("rh2m_pct",  1),
            "t2m_c":        ("t2m_pct",   1),
            "wind10_speed": ("wind10_pct", -1),
        }

        # Check availability
        available = []
        for ind, (clim_var, direc) in ind_map.items():
            if ind in ds.variables and clim_var in self._humid_climatology:
                available.append((ind, clim_var, direc))

        if len(available) < 3:
            return None

        clim_pcts = self._humid_climatology["percentile"].values
        pct_vals = clim_pcts / 100.0

        nlat_c, nlon_c = nlat, nlon
        ranks = []

        for ind, clim_var, direc in available:
            today = ds[ind].values
            while today.ndim > 2:
                today = today[0] if today.shape[0] == 1 else today.mean(axis=0)
            today = today[:nlat, :nlon].astype(float)
            if direc == -1:
                today = -today

            cell_pcts = self._humid_climatology[clim_var].values[:, :nlat, :nlon]
            if direc == -1:
                cell_pcts = -cell_pcts

            rank = np.full((nlat_c, nlon_c), np.nan, dtype=float)
            for i in range(nlat_c):
                for j in range(nlon_c):
                    tv = today[i, j]
                    cp = cell_pcts[:, i, j]
                    if not np.isfinite(tv) or not np.all(np.isfinite(cp)):
                        continue
                    idx = np.searchsorted(cp, tv)
                    if idx <= 0:
                        rank[i, j] = 0.01
                    elif idx >= len(cp):
                        rank[i, j] = 0.99
                    else:
                        lo_p, hi_p = pct_vals[idx-1], pct_vals[idx]
                        lo_v, hi_v = cp[idx-1], cp[idx]
                        frac = (tv - lo_v) / (hi_v - lo_v) if hi_v > lo_v else 0.5
                        rank[i, j] = lo_p + frac * (hi_p - lo_p)

            ranks.append(np.clip(rank, 0.0, 1.0))

        if len(ranks) < 3:
            return None

        return np.min(ranks, axis=0)


    def _compute_heat_gpd(self, ds, lat, lon, nlat, nlon):
        """
        Compute extreme heat GPD exceedance probability for each grid cell.

        P = P(T > today_tmax) using the GPD Peaks-Over-Threshold model.
        Small P means more extreme (P=0.01 = 1% chance of exceeding this tmax).
        Uses the per-cell GPD parameters stored in heat_gpd_climatology.nc.

        Returns:
            2D numpy array of exceedance probabilities (0-1), or None if unavailable.
        """
        import os

        if self._heat_climatology is None:
            clim_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "forecast", "heat_gpd_climatology.nc")
            if not os.path.exists(clim_path):
                return None
            import xarray as xr
            self._heat_climatology = xr.open_dataset(clim_path)

        # Get today's tmax
        tmax_today = None
        for var_name in ["tmax_c", "t2m_c"]:
            if var_name in ds.variables:
                v = ds[var_name].values
                while v.ndim > 2:
                    v = v[0] if v.shape[0] == 1 else v.max(axis=0)
                tmax_today = v[:nlat, :nlon].astype(float)
                break

        if tmax_today is None:
            return None

        # Get GPD parameters
        threshold = self._heat_climatology["gpd_threshold"].values[:nlat, :nlon]
        shape = self._heat_climatology["gpd_shape"].values[:nlat, :nlon]
        scale = self._heat_climatology["gpd_scale"].values[:nlat, :nlon]
        exceed_rate = self._heat_climatology["exceedance_rate"].values[:nlat, :nlon]

        # Compute GPD exceedance probability per cell
        prob = np.full((nlat, nlon), np.nan, dtype=float)

        for i in range(nlat):
            for j in range(nlon):
                t = tmax_today[i, j]
                u = threshold[i, j]
                s = scale[i, j]
                xi = shape[i, j]
                er = exceed_rate[i, j]

                if not (np.isfinite(t) and np.isfinite(u) and u > 0
                        and np.isfinite(s) and s > 0):
                    continue

                if t <= u:
                    prob[i, j] = 1.0 - er  # below threshold → non-exceedance
                else:
                    excess = t - u
                    if abs(xi) < 0.001:
                        p = er * np.exp(-excess / s)
                    else:
                        arg = 1.0 + xi * excess / s
                        if arg <= 0:
                            p = 0.0  # beyond upper bound
                        else:
                            p = er * arg**(-1.0 / xi)
                    prob[i, j] = float(np.clip(p, 0.0, 1.0))

        return prob.astype(float)


    def _compute_dust_joint_prob(self, ds, lat, lon, nlat, nlon):
        """
        Compute dust storm joint probability via Empirical Copula (Gumbel).

        P_dust = min(F_wind, F_dew, F_1/rh, F_shear)

        Estimates each indicator's empirical CDF rank by interpolating
        today's value against the per-cell indicator percentiles stored
        in dust_joint_climatology.nc.

        Returns:
            2D numpy array of joint probability (0-1), or None if unavailable.
        """
        import os

        # Lazy-load dust copula climatology
        if self._dust_climatology is None:
            clim_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "forecast", "dust_joint_climatology.nc")
            if not os.path.exists(clim_path):
                return None
            import xarray as xr
            self._dust_climatology = xr.open_dataset(clim_path)

        # Map: indicator → (clim_var_name, direction)
        # direction 1 = higher→more extreme, -1 = lower→more extreme
        ind_map = {
            "wind10_speed":          ("wind10_pct",  1),
            "dewpoint_depression_c": ("dewpoint_pct", 1),
            "rh2m":                  ("rh2m_pct",    -1),
            "wind_shear_850_200":    ("shear_pct",   1),
        }

        # Check which indicators are available today
        available = []
        for ind, (clim_var, direc) in ind_map.items():
            if ind in ds.variables and clim_var in self._dust_climatology:
                available.append((ind, clim_var, direc))

        if len(available) < 3:  # need at least 3 of 4 indicators
            return None

        # Reference percentiles from climatology
        clim_pcts = self._dust_climatology["percentile"].values  # [50, 75, 90, 95, 98, 99]
        pct_vals = clim_pcts / 100.0  # Convert to 0-1 scale

        nlat_c, nlon_c = nlat, nlon
        joint_prob = np.full((nlat_c, nlon_c), np.nan, dtype=float)

        # For each available indicator, compute rank
        ranks = []
        for ind, clim_var, direc in available:
            if ind not in ds.variables or clim_var not in self._dust_climatology:
                continue

            # Today's value
            today = ds[ind].values
            while today.ndim > 2:
                today = today[0] if today.shape[0] == 1 else today.mean(axis=0)
            today = today[:nlat, :nlon].astype(float)

            if direc == -1:
                today = -today  # negate so higher = more extreme

            # Per-cell percentiles from climatology
            cell_pcts = self._dust_climatology[clim_var].values[:, :nlat, :nlon]  # (n_pct, lat, lon)
            # Also negate if needed
            if direc == -1:
                cell_pcts = -cell_pcts

            # Estimate rank by linear interpolation between stored percentiles
            rank = np.full((nlat_c, nlon_c), np.nan, dtype=float)
            for i in range(nlat_c):
                for j in range(nlon_c):
                    tv = today[i, j]
                    cp = cell_pcts[:, i, j]
                    if not np.isfinite(tv) or not np.all(np.isfinite(cp)):
                        continue
                    # Find where today falls among percentiles
                    idx = np.searchsorted(cp, tv)
                    if idx <= 0:
                        rank[i, j] = 0.01  # below min percentile
                    elif idx >= len(cp):
                        rank[i, j] = 0.99  # above max percentile
                    else:
                        # Linear interpolation
                        lo_pct = pct_vals[idx - 1]
                        hi_pct = pct_vals[idx]
                        lo_val = cp[idx - 1]
                        hi_val = cp[idx]
                        if hi_val > lo_val:
                            frac = (tv - lo_val) / (hi_val - lo_val)
                        else:
                            frac = 0.5
                        rank[i, j] = lo_pct + frac * (hi_pct - lo_pct)

            ranks.append(np.clip(rank, 0.0, 1.0))

        if len(ranks) < 3:
            return None

        # Gumbel copula: joint = min of all ranks
        joint_prob = np.min(ranks, axis=0)

        return joint_prob.astype(float)


    def _compute_precip_percentile(self, ds, lat, lon, nlat, nlon):
        """
        Compute precipitation percentile for each grid cell based on climatology.

        Loads precip_climatology.nc (lazy, cached) and compares today's
        daily_precip_total against the per-cell Gamma-fitted percentile distribution.

        Returns:
            2D numpy array of percentile values (0-100), or None if unavailable.
        """
        import os

        # Lazy-load climatology
        if self._climatology is None:
            clim_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "forecast", "precip_climatology.nc")
            if not os.path.exists(clim_path):
                return None
            import xarray as xr
            self._climatology = xr.open_dataset(clim_path)

        # Get today's precipitation
        precip_today = None
        for var_name in ["daily_precip_total", "precip_proxy"]:
            if var_name in ds.variables:
                precip_today = ds[var_name].values
                break

        if precip_today is None:
            return None

        # Squeeze to 2D
        while precip_today.ndim > 2:
            precip_today = precip_today[0] if precip_today.shape[0] == 1 else precip_today.mean(axis=0)
        precip_today = precip_today[:nlat, :nlon]

        # Use method-of-moments Gamma CDF to compute percentile
        # Percentile = P(X <= today_precip | shape, scale)
        shape = self._climatology["gamma_shape"].values[:nlat, :nlon]
        scale = self._climatology["gamma_scale"].values[:nlat, :nlon]

        from scipy.stats import gamma as gamma_dist
        percentile = np.zeros((nlat, nlon), dtype=np.float32)

        # Vectorized: compute CDF for all cells with valid Gamma params
        valid = np.isfinite(shape) & np.isfinite(scale) & (shape > 0) & (scale > 0)
        if valid.any():
            percentile[valid] = gamma_dist.cdf(
                np.maximum(precip_today[valid], 0.001),
                shape[valid],
                scale=scale[valid]
            ) * 100.0  # Convert to 0-100 scale

        # For dry cells (no Gamma fit), use empirical: if precip > 0, it's >P90
        dry_cells = ~valid
        if dry_cells.any():
            # Check empirical P90 from climatology
            p90 = self._climatology["precip_percentiles"].values[2, :nlat, :nlon]  # P90
            percentile[dry_cells] = np.where(
                precip_today[dry_cells] >= p90[dry_cells], 95.0, 50.0)

        return np.clip(percentile, 0.0, 100.0)


    def _eval_condition(self, data: np.ndarray, op_str: str, threshold: float) -> np.ndarray:
        """Evaluate a comparison on numpy array, handling NaN."""
        if op_str not in self.COMPARATORS:
            raise ValueError(f"Unknown comparison operator: {op_str}")
        cmp_fn = self.COMPARATORS[op_str]
        result = cmp_fn(data, threshold)
        # NaN-safe: treat NaN as False
        result = np.where(np.isnan(result), False, result)
        return result.astype(bool)

    def _assign_severity(self, score: float, severity_levels: List[Dict]) -> Tuple[str, float]:
        """Map a risk score to a severity label."""
        for level in severity_levels:
            lo, hi = level["range"]
            if lo <= score <= hi:
                return level["label"], score
        return "unknown", score

    def _infer_region(self, lat: float, lon: float) -> str:
        """Infer which geographic region a point belongs to."""
        regions = {
            "red_sea":       (16.0, 30.0, 34.0, 44.0),
            "persian_gulf":  (24.0, 30.0, 48.0, 56.0),
            "north_saudi":   (26.0, 32.0, 34.0, 56.0),
            "central_saudi": (21.0, 26.0, 34.0, 56.0),
            "south_saudi":   (16.0, 21.0, 34.0, 56.0),
        }
        matches = []
        for name, (lat_min, lat_max, lon_min, lon_max) in regions.items():
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                matches.append(name)
        return ", ".join(matches) if matches else "saudi_bbox"

    def explain(self, event: Event) -> str:
        """Generate a human-readable explanation of an event."""
        lines = [
            f"{'='*60}",
            f"事件 #{event.event_id} | {event.date} | {event.region}",
            f"灾害类型: {event.hazard_type}",
            f"严重度: {event.severity} (评分: {event.severity_score}, 置信度: {event.confidence})",
            f"受影响格点: {len(event.affected_cells)} 个 (~{event.area_km2} km²)",
            f"质心: {event.centroid_lat}°N, {event.centroid_lon}°E",
            f"峰值风险分: {event.peak_risk}",
            f"",
            f"触发条件:",
        ]
        for detail in event.trigger_details:
            status_icon = "✓" if detail["status"] == "evaluated" else "✗"
            if detail["status"] == "evaluated":
                lines.append(
                    f"  {status_icon} {detail['indicator']}: "
                    f"{detail['condition']} (触发格点: {detail['cells_triggered']}, "
                    f"峰值: {detail['peak_value']}, 权重: {detail['weight']})"
                )
            elif detail["status"] == "missing":
                lines.append(
                    f"  {status_icon} {detail['indicator']}: 数据缺失 — {detail['action']}"
                )
        lines.append(f"{'='*60}")
        return "\n".join(lines)
