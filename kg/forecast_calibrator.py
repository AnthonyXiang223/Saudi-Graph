"""
ForecastCalibrator — KG historical event catalog bridge to IFS forecast detection.

Uses pre-computed event catalog (from build_event_catalog.py) to calibrate
forecast detection results with historical context. Zero ERA5 NetCDF dependency
at runtime — all data comes from the catalog files.

Usage:
    calibrator = ForecastCalibrator()

    # City-level calibration
    result = calibrator.calibrate_city_confidence(
        city_hazard_result, date_str="20260721", city_info={...}
    )

    # Regional reliability assessment
    result = calibrator.assess_forecast_reliability(
        ifs_hazards, date_str="20260721"
    )
"""

import os, json, logging
import numpy as np

log = logging.getLogger("mazu.calibrator")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HAZARD_LABELS = {
    "flash_flood": "山洪",
    "extreme_heat": "极端高温",
    "dust_storm": "沙尘强风",
    "coastal_humid_heat": "沿海湿热",
}

HAZARD_IDX = {"flash_flood": 0, "extreme_heat": 1, "dust_storm": 2, "coastal_humid_heat": 3}

SEVERITY_RANK = {
    "none": 0, "low": 1, "caution": 1, "medium": 2, "moderate": 2,
    "warning": 2, "high": 3, "alert": 3, "extreme": 4, "emergency": 4, "severe": 4,
}


class ForecastCalibrator:
    """Bridge between KG historical catalog and IFS forecast detection."""

    def __init__(self, catalog_dir: str = None):
        if catalog_dir is None:
            catalog_dir = os.path.join(PROJECT_DIR, "kg_data")
        self._catalog_dir = catalog_dir
        self._events_df = None
        self._grid_stats = None
        self._hazard_order = None
        self._loaded = False

    # ── Lazy loading ──────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._loaded:
            return
        parquet_path = os.path.join(self._catalog_dir, "event_catalog.parquet")
        npz_path = os.path.join(self._catalog_dir, "grid_event_counts.npz")

        if not os.path.exists(parquet_path) and not os.path.exists(npz_path):
            log.warning("Event catalog not found — calibration disabled. "
                        "Run: python scripts/compute/build_event_catalog.py")
            self._loaded = True
            return

        if os.path.exists(parquet_path):
            try:
                import pandas as pd
                self._events_df = pd.read_parquet(parquet_path)
                log.info("Loaded %d events from %s", len(self._events_df), parquet_path)
            except ImportError:
                log.warning("pandas not available, parquet loading skipped")
                self._events_df = None

        if os.path.exists(npz_path):
            self._grid_stats = dict(np.load(npz_path, allow_pickle=True))
            self._hazard_order = self._grid_stats.get("hazard_order", None)
            log.info("Loaded grid stats from %s", npz_path)

        self._loaded = True

    @property
    def available(self) -> bool:
        """Whether calibration data is available."""
        self._ensure_loaded()
        return self._grid_stats is not None

    # ── Grid mapping ──────────────────────────────────────────────

    @staticmethod
    def _grid_to_indices(lat_val, lon_val, nlat=160, nlon=220,
                         lat_min=16.0, lat_max=31.9, lon_min=34.0, lon_max=55.9):
        """Convert lat/lon to ERA5 grid indices.

        ERA5 grid: lat DESCENDING (31.9→16.0 north-to-south), lon ASCENDING.
        """
        # Latitude: descending (north=0, south=159)
        lat_idx = int(np.clip((lat_max - lat_val) / (lat_max - lat_min) * nlat, 0, nlat - 1))
        # Longitude: ascending (west=0, east=219)
        lon_idx = int(np.clip((lon_val - lon_min) / (lon_max - lon_min) * nlon, 0, nlon - 1))
        return lat_idx, lon_idx

    # ── Core calibration methods ──────────────────────────────────

    def get_monthly_base_rate(self, hazard_type: str, lat: float, lon: float,
                              month: int) -> float:
        """
        Historical event frequency for a location and month.

        Returns P(event | this cell, this month) — how often does this
        hazard type trigger at this location in this calendar month?
        """
        self._ensure_loaded()
        if self._grid_stats is None:
            return -1.0

        hi = HAZARD_IDX.get(hazard_type, -1)
        if hi < 0:
            return -1.0

        lat_idx, lon_idx = self._grid_to_indices(lat, lon)
        freq = self._grid_stats["cell_event_frequency"]
        if lat_idx >= freq.shape[0] or lon_idx >= freq.shape[1]:
            # Use 3x3 smoothed estimate for robustness
            li0 = max(0, lat_idx - 1)
            li1 = min(freq.shape[0], lat_idx + 2)
            lj0 = max(0, lon_idx - 1)
            lj1 = min(freq.shape[1], lon_idx + 2)
            val = float(np.mean(freq[li0:li1, lj0:lj1, hi, month]))
        else:
            val = float(freq[lat_idx, lon_idx, hi, month])

        return round(val, 4)

    def get_historical_severity_percentile(self, hazard_type: str, lat: float,
                                           lon: float, month: int,
                                           forecast_score: float) -> float:
        """
        Where does the forecast score rank among historical event-day scores?

        Returns percentile (0-100). 85 means 85% of historical event days
        had LOWER scores than this forecast.
        """
        self._ensure_loaded()
        if self._events_df is None:
            return -1.0

        hi = HAZARD_IDX.get(hazard_type)
        if hi is None:
            return -1.0

        lat_idx, lon_idx = self._grid_to_indices(lat, lon)

        # Filter events: same hazard, nearby location, same calendar month
        df = self._events_df
        mask = (
            (df["hazard_type"] == hazard_type) &
            (df["date"].str[4:6].astype(int) == month + 1) &
            (np.abs(df["centroid_lat"] - lat) < 1.5) &
            (np.abs(df["centroid_lon"] - lon) < 1.5)
        )
        scores = df.loc[mask, "severity_score"].values
        if len(scores) == 0:
            return -1.0

        pct = float(np.mean(scores <= forecast_score) * 100)
        return round(pct, 1)

    def get_historical_analogs(self, hazard_type: str, lat: float, lon: float,
                               date_str: str, n_analogs: int = 5,
                               days_window: int = 15) -> dict:
        """
        Find most similar historical dates for a given location and hazard.

        Searches within +/- days_window of the target calendar day across all
        365 days, returns dates ranked by severity similarity.
        """
        self._ensure_loaded()
        if self._events_df is None:
            return {"analogs": [], "note": "Event catalog not available"}

        # Parse target month/day
        target_month = int(date_str[4:6])
        target_day = int(date_str[6:8])

        df = self._events_df
        # Filter by hazard type and nearby location
        mask = (
            (df["hazard_type"] == hazard_type) &
            (np.abs(df["centroid_lat"] - lat) < 2.0) &
            (np.abs(df["centroid_lon"] - lon) < 2.0)
        )
        candidates = df[mask].copy()

        if len(candidates) == 0:
            return {"analogs": [], "note": f"No historical {hazard_type} events within 2deg of ({lat:.1f}N, {lon:.1f}E)"}

        # Score calendar proximity
        candidates["month"] = candidates["date"].str[4:6].astype(int)
        candidates["day"] = candidates["date"].str[6:8].astype(int)
        candidates["day_of_year"] = (
            candidates["month"] * 30 + candidates["day"]
        )
        target_doy = target_month * 30 + target_day
        candidates["calendar_dist"] = np.minimum(
            np.abs(candidates["day_of_year"] - target_doy),
            365 - np.abs(candidates["day_of_year"] - target_doy)
        )

        # Within window?
        in_window = candidates["calendar_dist"] <= days_window
        window_candidates = candidates[in_window].nsmallest(
            min(n_analogs * 2, len(candidates)), "calendar_dist"
        )

        if len(window_candidates) < n_analogs:
            # Expand: take closest outside window
            extra = candidates[~in_window].nsmallest(
                n_analogs - len(window_candidates), "calendar_dist"
            )
            window_candidates = (
                pd.concat([window_candidates, extra])
                if hasattr(pd, "concat") else window_candidates
            )

        top = window_candidates.head(n_analogs)

        analogs = []
        for _, row in top.iterrows():
            analogs.append({
                "date": row["date"],
                "severity": row["severity"],
                "severity_score": float(row["severity_score"]),
                "calendar_distance_days": int(row["calendar_dist"]),
            })

        # Summary statistics
        all_window_scores = candidates[candidates["calendar_dist"] <= days_window]["severity_score"]
        event_rate = len(window_candidates) / max(days_window * 2, 1)

        return {
            "analogs": analogs,
            "mean_severity": round(float(all_window_scores.mean()), 3) if len(all_window_scores) > 0 else None,
            "max_severity": round(float(all_window_scores.max()), 3) if len(all_window_scores) > 0 else None,
            "event_rate": round(event_rate, 3),
            "n_window_events": len(window_candidates),
        }

    # ── High-level calibration API ────────────────────────────────

    def calibrate_city_confidence(self, city_hazard_result: dict,
                                  date_str: str, city_info: dict) -> dict:
        """
        Augment get_city_hazards result with KG historical calibration.

        For each hazard type, adds:
        - historical_base_rate: how often this hazard triggers in this month
        - calibrated_confidence: high/medium/low
        - severity_percentile: where forecast score ranks historically
        - calibration_note: human-readable summary
        """
        self._ensure_loaded()

        month = int(date_str[4:6]) - 1
        lat = city_info.get("lat", 0)
        lon = city_info.get("lon", 0)

        calibrated = dict(city_hazard_result)
        calibrated_hazards = {}

        for htype, hdata in city_hazard_result.get("hazards", {}).items():
            ch = dict(hdata)

            if self.available:
                base_rate = self.get_monthly_base_rate(htype, lat, lon, month)
                score_pct = self.get_historical_severity_percentile(
                    htype, lat, lon, month, hdata.get("score", 0)
                )

                ch["historical_base_rate"] = base_rate
                ch["severity_percentile"] = score_pct

                # Confidence logic
                forecast_sev = hdata.get("severity", "none")
                forecast_score = hdata.get("score", 0)

                if base_rate < 0:
                    confidence = "unknown"
                    note = "KG事件目录不可用"
                elif base_rate < 0.02:
                    # Very rare event for this month — flag as unusual
                    if SEVERITY_RANK.get(forecast_sev, 0) >= 3:
                        confidence = "low"
                        note = (f"该月历史基准触发率仅{base_rate*100:.1f}%。"
                                f"当前预报严重度{forecast_sev}缺少充分历史先例，建议保守报告。")
                    else:
                        confidence = "high"
                        note = (f"该月历史触发率极低({base_rate*100:.1f}%)，"
                                f"本次未触发，与历史模式一致。")
                elif base_rate < 0.15:
                    confidence = "medium"
                    note = (f"该月历史基准触发率{base_rate*100:.1f}%。"
                            f"当前评分处于历史P{score_pct:.0f}水平。")
                else:
                    # Frequent event — high confidence in detection
                    confidence = "high"
                    note = (f"该月历史基准触发率{base_rate*100:.1f}%（季节性常态）。"
                            f"当前评分处于历史P{score_pct:.0f}水平。")

                ch["calibrated_confidence"] = confidence
                ch["calibration_note"] = note
            else:
                ch["calibrated_confidence"] = "unknown"
                ch["calibration_note"] = "KG事件目录未构建。运行 build_event_catalog.py 启用校准。"
                ch["historical_base_rate"] = -1
                ch["severity_percentile"] = -1

            calibrated_hazards[htype] = ch

        calibrated["hazards"] = calibrated_hazards
        calibrated["_kg_calibrated"] = self.available
        return calibrated

    def assess_forecast_reliability(self, ifs_hazards: list,
                                    date_str: str) -> dict:
        """
        Global reliability assessment across all detected IFS hazards.

        Checks each detected hazard against historical patterns and flags
        any that lack historical precedent.
        """
        self._ensure_loaded()

        if not self.available:
            return {"overall_reliability": "unknown",
                    "note": "KG事件目录不可用，无法进行可靠性评估"}

        month = int(date_str[4:6]) - 1
        per_hazard = {}
        warnings = []

        for h in ifs_hazards:
            htype = h.get("hazard_type", "")
            if not htype:
                continue

            has_precedent = False
            precedent_detail = None

            if h.get("detected") and h.get("hotspot"):
                # Parse hotspot lat/lon
                hs = h["hotspot"]
                try:
                    hs_lat = float(hs.split("N,")[0].strip())
                    hs_lon = float(hs.split("N,")[1].replace("E", "").strip())
                except (ValueError, IndexError):
                    hs_lat, hs_lon = 24.0, 45.0  # fallback central Saudi

                base_rate = self.get_monthly_base_rate(htype, hs_lat, hs_lon, month)
                if base_rate > 0.01:
                    has_precedent = True
                    precedent_detail = (
                        f"该区域{month+1}月历史基准触发率{base_rate*100:.1f}%"
                    )
                else:
                    precedent_detail = (
                        f"该区域{month+1}月历史触发率{base_rate*100:.1f}%，"
                        f"缺少充分历史先例"
                    )
                    warnings.append(
                        f"{HAZARD_LABELS.get(htype, htype)}检测在"
                        f"{h.get('hotspot_region', '未知区域')}缺少历史先例"
                    )

            per_hazard[htype] = {
                "consistent_with_history": has_precedent if h.get("detected") else True,
                "historical_precedent": precedent_detail,
            }

        overall = "high" if len(warnings) == 0 else ("medium" if len(warnings) <= 1 else "low")
        return {
            "overall_reliability": overall,
            "per_hazard": per_hazard,
            "warnings": warnings,
            "recommendations": [
                f"⚠ {w}" for w in warnings
            ] if warnings else ["所有检测结果与历史模式吻合，置信度高"],
        }
