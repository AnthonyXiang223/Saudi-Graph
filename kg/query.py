"""
Unified query interface for the Saudi extreme event knowledge graph.

Three-layer query architecture:
1. Knowledge layer — pure graph traversal (no data access)
2. Data layer — xarray spatiotemporal queries
3. Joint layer — KG-driven data queries + event detection + explainability
"""

import os
from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd

from .ontology import KnowledgeGraph
from .datalayer import DataLayer
from .event_detector import EventDetector, Event


class QueryEngine:
    """
    Unified query interface for the Saudi extreme event KG.

    Usage:
        qe = QueryEngine(schema_dir="schema", data_dir="indicators")
        qe.init()

        # Knowledge queries
        qe.get_hazard_indicators("flash_flood")
        qe.get_indicator_chain("heatwave_duration_days")

        # Data queries
        qe.get_timeseries(24.7, 46.7, "2025-07-01", "2025-07-31", ["tmax_c"])

        # Joint queries
        events = qe.detect_events("2025-08-19", ["flash_flood"])
        print(qe.explain(events[0]))
    """

    def __init__(self, schema_dir: str = "schema", data_dir: str = "indicators"):
        self.schema_dir = schema_dir
        self.data_dir = data_dir
        self.kg: Optional[KnowledgeGraph] = None
        self.data: Optional[DataLayer] = None
        self.detector: Optional[EventDetector] = None

    def init(self):
        """Initialize all three layers. Must be called before queries."""
        self.kg = KnowledgeGraph(self.schema_dir)
        self.kg.build()

        self.data = DataLayer(self.data_dir)

        self.detector = EventDetector(self.kg, self.data)

        return self

    # ═══════════════════════════════════════════════════════════
    # Layer 1: Knowledge Graph Queries (pure graph traversal)
    # ═══════════════════════════════════════════════════════════

    def get_hazard_indicators(self, hazard_type: str) -> List[Dict]:
        """Get all indicators that contribute to a hazard type."""
        return self.kg.get_hazard_indicators(hazard_type)

    def get_indicator_chain(self, indicator_id: str) -> List[Dict]:
        """Get the full derivation chain of an indicator back to data sources."""
        return self.kg.get_indicator_chain(indicator_id)

    def get_operator(self, indicator_id: str) -> Optional[Dict]:
        """Get the full operator definition including DAG."""
        return self.kg.get_operator(indicator_id)

    def get_co_occurring(self, indicator_id: str) -> List[str]:
        """Get indicators that co-occur with this one."""
        return self.kg.get_co_occurring(indicator_id)

    def get_indicators_by_source(self, source: str) -> List[str]:
        """Get all indicators from a specific data source."""
        indicators = []
        for node, data in self.kg.graph.nodes(data=True):
            if data.get("type") == "Indicator" and data.get("source") == source:
                indicators.append(node)
        return sorted(indicators)

    def kg_summary(self) -> Dict:
        """Get graph summary statistics."""
        return self.kg.summary()

    # ═══════════════════════════════════════════════════════════
    # Layer 2: Data Layer Queries (xarray)
    # ═══════════════════════════════════════════════════════════

    def get_timeseries(self, lat: float, lon: float,
                       start_date: str, end_date: str,
                       variables: List[str]) -> pd.DataFrame:
        """Extract a time series for a specific location."""
        return self.data.get_timeseries(lat, lon, start_date, end_date, variables)

    def get_spatial_snapshot(self, date_str: str, variable: str) -> np.ndarray:
        """Get a 2D spatial map of a variable on a given date."""
        return self.data.get_spatial_snapshot(date_str, variable)

    def get_multi_snapshot(self, date_str: str,
                           variables: List[str]) -> Dict[str, np.ndarray]:
        """Get multiple variables on the same date."""
        return self.data.get_multi_variable_snapshot(date_str, variables)

    def cache_info(self) -> Dict:
        """Get data cache statistics."""
        return self.data.cache_info()

    # ═══════════════════════════════════════════════════════════
    # Layer 3: Joint Queries (KG + Data + Detection)
    # ═══════════════════════════════════════════════════════════

    def detect_events(self, date_str: str,
                      hazard_types: Optional[List[str]] = None) -> List[Event]:
        """
        Detect extreme events on a given date.

        The KG provides the rules → data layer provides the numbers →
        event_detector applies weighted scoring + connected components.

        Args:
            date_str: "YYYYMMDD" or "YYYY-MM-DD"
            hazard_types: e.g. ["flash_flood", "extreme_heat"], or None for all

        Returns:
            List of Event objects
        """
        return self.detector.detect_events(date_str, hazard_types)

    def explain(self, event: Event) -> str:
        """
        Generate a human-readable explanation of why an event was detected.

        This is the key interface for Agent scenarios — provides full
        audit trail of which indicators fired, which thresholds were met,
        and what confidence penalties (if any) were applied.
        """
        return self.detector.explain(event)

    def assess_risk(self, date_str: str,
                    region: Optional[str] = None) -> Dict:
        """
        Comprehensive risk assessment for a date, optionally filtered by region.

        Returns a dict with per-hazard-type summaries.
        """
        events = self.detect_events(date_str)

        if region:
            events = [e for e in events if region in e.region]

        assessment = {
            "date": date_str,
            "region": region or "all",
            "total_events": len(events),
            "by_hazard": {},
            "max_severity": "none",
            "max_severity_score": 0.0,
        }

        for e in events:
            if e.hazard_type not in assessment["by_hazard"]:
                assessment["by_hazard"][e.hazard_type] = {
                    "count": 0,
                    "max_severity": "none",
                    "total_area_km2": 0.0,
                    "total_affected_cells": 0,
                }
            h = assessment["by_hazard"][e.hazard_type]
            h["count"] += 1
            h["total_area_km2"] += e.area_km2
            h["total_affected_cells"] += len(e.affected_cells)
            if e.severity_score > h.get("max_score", 0):
                h["max_severity"] = e.severity
                h["max_score"] = e.severity_score

            if e.severity_score > assessment["max_severity_score"]:
                assessment["max_severity"] = e.severity
                assessment["max_severity_score"] = e.severity_score

        return assessment

    # ═══════════════════════════════════════════════════════════
    # Convenience
    # ═══════════════════════════════════════════════════════════

    def validate(self) -> List[str]:
        """Validate the knowledge graph integrity."""
        return self.kg.validate()


# ── CLI entry point ──
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Saudi Extreme Event KG Query Engine")
    parser.add_argument("--date", default="2025-08-19", help="Date (YYYYMMDD)")
    parser.add_argument("--hazard", default="flash_flood",
                        choices=["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat", "all"],
                        help="Hazard type")
    parser.add_argument("--explain", action="store_true", help="Print event explanations")
    parser.add_argument("--validate", action="store_true", help="Validate KG before querying")
    parser.add_argument("--summary", action="store_true", help="Print KG summary")

    args = parser.parse_args()

    # Determine paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    schema_dir = os.path.join(project_dir, "schema")
    data_dir = os.path.join(project_dir, "indicators")

    print(f"Initializing query engine...")
    qe = QueryEngine(schema_dir, data_dir)
    qe.init()
    print("Done.\n")

    if args.summary:
        summary = qe.kg_summary()
        print("Knowledge Graph Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print()

    if args.validate:
        issues = qe.validate()
        if issues:
            print(f"Validation issues ({len(issues)}):")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Validation: OK (no issues found)")
        print()

    if args.hazard == "all":
        hazard_types = None
    else:
        hazard_types = [args.hazard]

    print(f"Detecting events for {args.date}...")
    events = qe.detect_events(args.date, hazard_types)
    print(f"Found {len(events)} event(s)\n")

    if args.explain:
        for event in events:
            print(qe.explain(event))
            print()
    else:
        for event in events:
            print(f"  {event.event_id}: {event.hazard_type} | {event.severity} "
                  f"(score={event.severity_score}, conf={event.confidence}) "
                  f"| {event.region} | {len(event.affected_cells)} cells")
