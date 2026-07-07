"""
Verification script for the Saudi Extreme Event Knowledge Graph.

Tests:
1. KG construction and validation
2. Event detection on known extreme dates
3. Negative test on a quiet date
4. Operator chain integrity
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from kg.query import QueryEngine

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_DIR = os.path.join(PROJECT_DIR, "schema")
DATA_DIR = os.path.join(PROJECT_DIR, "indicators")


def test_1_kg_construction():
    """Test KG builds and validates cleanly."""
    print("=" * 60)
    print("TEST 1: KG Construction & Validation")
    print("=" * 60)

    qe = QueryEngine(SCHEMA_DIR, DATA_DIR)
    qe.init()

    summary = qe.kg_summary()
    print(f"Nodes: {summary.get('total_nodes', '?')}")
    print(f"Edges: {summary.get('total_edges', '?')}")

    # Break down by node type
    node_types = {}
    edge_types = {}
    for node, data in qe.kg.graph.nodes(data=True):
        nt = data.get("type", "unknown")
        node_types[nt] = node_types.get(nt, 0) + 1
    for u, v, data in qe.kg.graph.edges(data=True):
        et = data.get("relationship", "unknown")
        edge_types[et] = edge_types.get(et, 0) + 1

    print("Nodes by type:")
    for nt, count in sorted(node_types.items()):
        print(f"  {nt}: {count}")
    print("Edges by type:")
    for et, count in sorted(edge_types.items()):
        print(f"  {et}: {count}")

    issues = qe.validate()
    if issues:
        print(f"\nVALIDATION ISSUES ({len(issues)}):")
        for issue in issues:
            print(f"  [ISSUE] {issue}")
        return False
    else:
        print("\nValidation: PASSED (no issues)")
        return True


def test_2_flash_flood():
    """Test flash flood detection on 2025-08-19 (known extreme date)."""
    print("\n" + "=" * 60)
    print("TEST 2: Flash Flood Detection — 2025-08-19")
    print("=" * 60)

    qe = QueryEngine(SCHEMA_DIR, DATA_DIR)
    qe.init()

    events = qe.detect_events("2025-08-19", ["flash_flood"])
    print(f"Events detected: {len(events)}")

    if events:
        for e in events[:3]:
            print(f"\n  {e.event_id}: severity={e.severity} score={e.severity_score} "
                  f"area={e.area_km2}km² cells={len(e.affected_cells)}")
        print("\n" + qe.explain(events[0]))
        return len(events) > 0
    else:
        print("  WARNING: No flash flood events detected on a known extreme date!")
        print("  Expected: at least 1 event (report shows flash_flood_risk max=4 on this date)")
        return False


def test_3_extreme_heat():
    """Test extreme heat detection on 2025-07-25 (known extreme date)."""
    print("\n" + "=" * 60)
    print("TEST 3: Extreme Heat Detection — 2025-07-25")
    print("=" * 60)

    qe = QueryEngine(SCHEMA_DIR, DATA_DIR)
    qe.init()

    events = qe.detect_events("2025-07-25", ["extreme_heat"])
    print(f"Events detected: {len(events)}")

    if events:
        for e in events[:3]:
            print(f"\n  {e.event_id}: severity={e.severity} score={e.severity_score} "
                  f"area={e.area_km2}km² cells={len(e.affected_cells)}")
        print("\n" + qe.explain(events[0]))
        return len(events) > 0
    else:
        print("  WARNING: No extreme heat events detected!")
        print("  Expected: at least 1 event (report shows tmax_c max=53.75°C on this date)")
        return False


def test_4_quiet_day():
    """Negative test: verify no false positives on a quiet day."""
    print("\n" + "=" * 60)
    print("TEST 4: Negative Test (Quiet Day) — 2025-01-15")
    print("=" * 60)

    qe = QueryEngine(SCHEMA_DIR, DATA_DIR)
    qe.init()

    # Check all hazard types
    all_events = qe.detect_events("2025-01-15", None)
    high_severity = [e for e in all_events if e.severity_score >= 0.3]
    print(f"Total events: {len(all_events)}")
    print(f"High severity (score >= 0.3): {len(high_severity)}")

    if high_severity:
        print("  Events:")
        for e in high_severity:
            print(f"    {e.event_id}: {e.hazard_type} severity={e.severity} score={e.severity_score}")

    # A quiet day should have few or no high-severity events
    return len(high_severity) <= 2  # Allow a couple low-confidence detections


def test_5_operator_chain():
    """Test operator chain traversal."""
    print("\n" + "=" * 60)
    print("TEST 5: Operator Chain Integrity")
    print("=" * 60)

    qe = QueryEngine(SCHEMA_DIR, DATA_DIR)
    qe.init()

    # Test indicator chains for key indicators
    test_indicators = [
        "heatwave_duration_days",
        "flash_flood_risk",
        "t2m_anomaly_c",
        "ivt_convergence",
    ]

    all_ok = True
    for ind_id in test_indicators:
        chain = qe.get_indicator_chain(ind_id)
        if chain and chain.get("indicators"):
            indicators = chain["indicators"]
            sources = chain["sources"]
            edges = chain["edges"]
            print(f"  {ind_id}: {len(indicators)} indicators in chain, {len(sources)} data sources, {len(edges)} derived_from edges")
            for src in sources:
                print(f"    → source: {src}")
        else:
            print(f"  {ind_id}: EMPTY CHAIN — indicator not found in KG!")
            all_ok = False

    return all_ok


def main():
    print("Saudi Extreme Event Knowledge Graph — Verification")
    print(f"Project: {PROJECT_DIR}")
    print(f"Schema: {SCHEMA_DIR}")
    print(f"Data: {DATA_DIR}")
    print()

    results = {}

    try:
        results["1_kg_construction"] = test_1_kg_construction()
    except Exception as e:
        print(f"TEST 1 FAILED with exception: {e}")
        import traceback; traceback.print_exc()
        results["1_kg_construction"] = False

    try:
        results["2_flash_flood"] = test_2_flash_flood()
    except Exception as e:
        print(f"TEST 2 FAILED with exception: {e}")
        import traceback; traceback.print_exc()
        results["2_flash_flood"] = False

    try:
        results["3_extreme_heat"] = test_3_extreme_heat()
    except Exception as e:
        print(f"TEST 3 FAILED with exception: {e}")
        import traceback; traceback.print_exc()
        results["3_extreme_heat"] = False

    try:
        results["4_quiet_day"] = test_4_quiet_day()
    except Exception as e:
        print(f"TEST 4 FAILED with exception: {e}")
        import traceback; traceback.print_exc()
        results["4_quiet_day"] = False

    try:
        results["5_operator_chain"] = test_5_operator_chain()
    except Exception as e:
        print(f"TEST 5 FAILED with exception: {e}")
        import traceback; traceback.print_exc()
        results["5_operator_chain"] = False

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for test, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {test}")
    print(f"\n{passed}/{total} tests passed")
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
