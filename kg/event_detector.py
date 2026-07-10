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

        for cond in conditions:
            indicator = cond["indicator"]
            comp_op = cond.get("condition", cond.get("op"))
            threshold = cond["value"]
            weight = cond.get("weight", 0.0)
            is_primary = cond.get("primary", False)

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

        # Final gate: if primary condition not met, cap risk at 0.2 (below all severity thresholds)
        if primary_mask is not None:
            risk_score = np.where(primary_mask, risk_score, risk_score * 0.25)
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
