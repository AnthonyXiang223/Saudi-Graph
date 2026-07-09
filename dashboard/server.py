"""
Dashboard server for Saudi Extreme Event Knowledge Graph.
Flask backend — now with SPARQL/DMDO endpoints alongside original API.
"""

import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, render_template, send_from_directory

app = Flask(__name__)
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DASHBOARD_DIR)
SCHEMA_DIR = os.path.join(PROJECT_DIR, "schema")
DATA_DIR = os.path.join(PROJECT_DIR, "indicators")

# ── Init both backends ──
print("Initializing networkx KG...")
from kg.query import QueryEngine
qe = QueryEngine(SCHEMA_DIR, DATA_DIR)
qe.init()

print("Initializing DMDO RDF graph...")
from kg.owl import SaudiDMDOConverter, SPARQLQueries
converter = SaudiDMDOConverter(SCHEMA_DIR, DATA_DIR)
converter.build_graph()
sq = SPARQLQueries(converter)
print(f"Ready. networkx: {qe.kg_summary()['total_nodes']} nodes, RDF: {len(converter.graph)} triples")


# ═══════════════════════════════════════════
# Static
# ═══════════════════════════════════════════
@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(os.path.join(DASHBOARD_DIR, "static"), filename)


# ═══════════════════════════════════════════
# Frontend
# ═══════════════════════════════════════════
@app.route("/")
def index():
    """DMDO-SPARQL powered dashboard."""
    return render_template("index_sparql.html")


# ═══════════════════════════════════════════
# SPARQL API — Knowledge layer
# ═══════════════════════════════════════════

@app.route("/api/sparql/summary")
def api_sparql_summary():
    """Full graph as nodes + edges (same format as original, from RDF)."""
    g = converter.graph
    nodes = []
    edges = []
    seen_nodes = set()

    # Extract Saudi-specific nodes (skip DMDO internal classes/axioms + Grid/Observation nodes)
    PREFIX = "https://mazu.cma/saudi#"
    SKIP = ["Grid/", "Observation"]
    for s, p, o in g.triples((None, None, None)):
        s_str = str(s)
        if PREFIX in s_str and s_str not in seen_nodes:
            # Skip ephemeral Grid and Observation nodes
            if any(x in s_str for x in SKIP):
                continue
            seen_nodes.add(s_str)
            node_id = s_str.split("#")[-1]
            # Determine type
            ntype = "Indicator"
            if "HazardType/" in s_str:
                ntype = "HazardType"
            elif "DataSource/" in s_str:
                ntype = "DataSource"
            elif "Region/" in s_str:
                ntype = "Region"
            elif "Hazard/" in s_str:
                ntype = "Rule"
            elif "Event/" in s_str:
                ntype = "Event"

            nodes.append({
                "id": node_id,
                "type": ntype,
                "group": ntype,
                "label": node_id.split("/")[-1],
                "title": node_id,
            })

    # Edges from key relationships
    REL_MAP = {
        "http://www.w3.org/ns/prov#wasDerivedFrom": "derived_from",
        "https://mazu.cma/saudi#coOccursWith": "co_occurs_with",
        "http://purl.org/disaster/deo#hasHazardProperty": "contributes_to",
        "http://purl.org/disaster/deo#hazardType": "detects",
        "http://purl.org/disaster/deo#possiblyCauses": "causes",
        "https://mazu.cma/saudi#hasCondition": "has_condition",
    }

    for s, p, o in g.triples((None, None, None)):
        rel = REL_MAP.get(str(p))
        s_str, o_str = str(s), str(o)
        # Skip edges involving Grid/Observation nodes
        if any(x in s_str for x in SKIP) or any(x in o_str for x in SKIP):
            continue
        if rel and PREFIX in s_str and PREFIX in o_str:
            edges.append({
                "from": str(s).split("#")[-1],
                "to": str(o).split("#")[-1],
                "label": rel,
                "arrows": "to,from" if rel == "co_occurs_with" else "to",
            })

    return jsonify({
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "triples": len(g),
        "nodes": nodes,
        "edges": edges,
    })


@app.route("/api/sparql/indicator/<indicator_id>")
def api_sparql_indicator(indicator_id):
    """Indicator detail from SPARQL."""
    detail = sq.indicator_detail(indicator_id)
    chain = sq.indicator_chain(indicator_id)
    co = sq.co_occurring(indicator_id)
    return jsonify({
        "detail": detail,
        "chain": [r.get("derived", "") for r in chain],
        "co_occurring": co,
    })


@app.route("/api/sparql/hazard/<hazard_type>")
def api_sparql_hazard(hazard_type):
    """Hazard indicators from SPARQL."""
    results = sq.hazard_indicators(hazard_type)
    indicators = []
    for r in results:
        name = str(r.get("indicator", "")).split("/")[-1].rstrip(">")
        desc = str(r.get("description", ""))
        indicators.append({"id": name, "description": desc})
    return jsonify({"hazard_type": hazard_type, "indicators": indicators})


@app.route("/api/sparql/chain/<indicator_id>")
def api_sparql_chain(indicator_id):
    """Derivation chain from SPARQL."""
    chain = sq.indicator_chain(indicator_id)
    return jsonify([{
        "derived": str(r.get("derived", "")).split("/")[-1].rstrip(">"),
        "source": str(r.get("source", "")).split("/")[-1].rstrip(">"),
    } for r in chain])


@app.route("/api/sparql/search")
def api_sparql_search():
    """Search indicators by keyword."""
    q = request.args.get("q", "")
    if not q:
        return jsonify([])
    results = sq.search_indicators(q)
    return jsonify([{
        "id": str(r.get("indicator", "")).split("/")[-1].rstrip(">"),
        "description": str(r.get("description", "")),
    } for r in results])


@app.route("/api/sparql/rule/<rule_id>")
def api_sparql_rule(rule_id):
    """Full rule detail: conditions, severity, fallback from RDF."""
    g = converter.graph
    SAUDI = "https://mazu.cma/saudi#"
    DEO = "http://purl.org/disaster/deo#"

    from rdflib import URIRef, RDF, RDFS
    rule_uri = URIRef(f"{SAUDI}Hazard/{rule_id}")

    conditions = []
    severity_levels = []
    fallback = None
    hazard_type = ""

    for s, p, o in g.triples((rule_uri, None, None)):
        if str(p) == str(DEO + "hazardType"):
            hazard_type = str(o).split("#")[-1] if "#" in str(o) else str(o).split("/")[-1]
        elif str(p) == f"{SAUDI}hasCondition":
            cond = {}
            for _, cp, co in g.triples((o, None, None)):
                key = str(cp).split("#")[-1]
                val = str(co)
                if "float" in val or "boolean" in val:
                    val = val.split("^")[0].strip('"')
                cond[key] = val
            conditions.append(cond)
        elif str(p) == f"{SAUDI}hasSeverityLevel":
            sev = {}
            for _, sp, so in g.triples((o, None, None)):
                key = str(sp).split("#")[-1]
                val = str(so)
                if "float" in val:
                    val = float(val.split("^")[0].strip('"'))
                sev[key] = val
            severity_levels.append(sev)
        elif str(p) == f"{SAUDI}hasFallback":
            fb = {}
            for _, fp, fo in g.triples((o, None, None)):
                key = str(fp).split("#")[-1]
                val = str(fo)
                if "float" in val:
                    val = float(val.split("^")[0].strip('"'))
                fb[key] = val
            fallback = fb

    return jsonify({
        "rule_id": rule_id,
        "hazard_type": hazard_type,
        "conditions": conditions,
        "severity_levels": severity_levels,
        "fallback": fallback,
    })


@app.route("/api/sparql/events")
def api_sparql_events():
    """Query all events in RDF graph."""
    results = sq.compare_hazard_severity("flash_flood")
    return jsonify([{
        "event": str(r.get("event", "")).split("/")[-1].rstrip(">"),
        "date": str(r.get("date", "")),
        "severity": str(r.get("severity", "")),
        "area": str(r.get("area", "")),
    } for r in results])


# ═══════════════════════════════════════════
# SPARQL API — SOSA Observation layer
# ═══════════════════════════════════════════

@app.route("/api/sparql/observations/create", methods=["POST"])
def api_sparql_create_observations():
    """Create SOSA Observation triples from NetCDF data.
    Body: {date, indicator_ids, threshold_filter}
    """
    body = request.get_json() or {}
    date_str = body.get("date", "2025-08-19")
    indicator_ids = body.get("indicator_ids", None)
    threshold_filter = body.get("threshold_filter", None)

    if threshold_filter:
        threshold_filter = {k: tuple(v) for k, v in threshold_filter.items()}

    n = converter.add_observations(
        date_str=date_str,
        indicator_ids=indicator_ids,
        threshold_filter=threshold_filter
    )
    return jsonify({
        "date": date_str,
        "observations_created": n,
        "total_triples": len(converter.graph)
    })


@app.route("/api/sparql/observations/query")
def api_sparql_query_observations():
    """Query SOSA Observations by indicator + date + min value."""
    indicator_id = request.args.get("indicator", "")
    date_str = request.args.get("date", "")
    min_value = request.args.get("min_value", None)

    if not indicator_id:
        return jsonify({"error": "indicator required"}), 400

    if min_value:
        min_value = float(min_value)

    results = sq.observation_by_indicator(indicator_id, date_str or None, min_value)
    return jsonify(list(results))


@app.route("/api/sparql/observations/stats")
def api_sparql_observation_stats():
    """Get observation summary stats for a date."""
    date_str = request.args.get("date", "2025-08-19")
    results = sq.observations_summary(date_str)
    return jsonify(list(results))


# ═══════════════════════════════════════════
# SPARQL API — GeoSPARQL spatial queries (P1)
# ═══════════════════════════════════════════

@app.route("/api/sparql/geospatial/radius")
def api_geospatial_radius():
    """Spatial search: find observations within radius using Python-side filtering."""
    import math
    ind = request.args.get("indicator", "tmax_c")
    lat = float(request.args.get("lat", 24.7))
    lon = float(request.args.get("lon", 46.7))
    radius_km = float(request.args.get("radius", 100))
    date_str = request.args.get("date", None)
    min_val = request.args.get("min_value", None)
    if min_val:
        min_val = float(min_val)

    # Query observations from RDF (basic SPARQL, no GeoSPARQL extension)
    from rdflib import URIRef, Literal
    g = converter.graph
    SAUDI_NS = "https://mazu.cma/saudi#"
    ind_uri = URIRef(f"{SAUDI_NS}Indicator/{ind}")

    results = []
    for s, p, o in g.triples((None, SOSA.observedProperty, ind_uri)):
        obs_node = s
        val = None
        obs_date = None
        obs_lat = None
        obs_lon = None
        for _, pp, oo in g.triples((obs_node, None, None)):
            if pp == SOSA.hasSimpleResult:
                val = float(oo)
            elif pp == SOSA.resultTime:
                obs_date = str(oo)
            elif pp == SOSA.hasFeatureOfInterest:
                for _, gp, go in g.triples((oo, GEO_F.hasGeometry, None)):
                    for _, gpp, gowkt in g.triples((go, GEO_F.asWKT, None)):
                        wkt = str(gowkt)
                        if wkt.startswith("POINT("):
                            parts = wkt[6:-1].split()
                            if len(parts) >= 2:
                                try:
                                    obs_lon = float(parts[0])
                                    obs_lat = float(parts[1])
                                except ValueError:
                                    pass
        if val is None or obs_lat is None:
            continue
        if date_str and obs_date and date_str not in obs_date:
            continue
        if min_val is not None and val <= min_val:
            continue
        # Haversine distance
        dlat = math.radians(obs_lat - lat)
        dlon = math.radians(obs_lon - lon)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat))*math.cos(math.radians(obs_lat))*math.sin(dlon/2)**2
        dist = 2*6371*math.asin(math.sqrt(a))
        if dist <= radius_km:
            results.append({"lat": round(obs_lat,4), "lon": round(obs_lon,4), "value": round(val,2), "date": obs_date, "distance_km": round(dist,1)})
    results.sort(key=lambda r: r.get("distance_km", 999))
    return jsonify(results[:100])


@app.route("/api/sparql/geospatial/intersects")
def api_geospatial_intersects():
    """GeoSPARQL: find events intersecting a region."""
    region = request.args.get("region", "red_sea")
    results = sq.events_intersecting_region(region)
    return jsonify(list(results))


@app.route("/api/sparql/geospatial/aggregate")
def api_geospatial_aggregate():
    """GeoSPARQL: aggregate observations by region."""
    ind = request.args.get("indicator", "daily_precip_total")
    date = request.args.get("date", "2025-08-19")
    region = request.args.get("region", None)
    results = sq.spatial_aggregation(ind, date, region)
    return jsonify(list(results))


# ═══════════════════════════════════════════
# SPARQL API — OWL-Time temporal queries (P2)
# ═══════════════════════════════════════════

@app.route("/api/sparql/temporal/sequence")
def api_temporal_sequence():
    """OWL-Time: cascading event chains."""
    ht = request.args.get("hazard_type", None)
    results = sq.temporal_event_sequence(ht)
    return jsonify(list(results))


@app.route("/api/sparql/temporal/timeline")
def api_temporal_timeline():
    """OWL-Time: event timeline with dates and severities."""
    start = request.args.get("start", None)
    end = request.args.get("end", None)
    results = sq.event_timeline(start, end)
    return jsonify(list(results))


@app.route("/api/sparql/temporal/cascade/<event_id>")
def api_temporal_cascade(event_id):
    """OWL-Time: trace cascading chain from an event."""
    results = sq.cascading_hazard_chain(event_id)
    return jsonify(list(results))


# ═══════════════════════════════════════════
# SPARQL API — PROV-O provenance queries (P2)
# ═══════════════════════════════════════════

@app.route("/api/sparql/provenance/indicator/<indicator_id>")
def api_provenance_indicator(indicator_id):
    """PROV-O: full provenance chain for an indicator."""
    results = sq.provenance_chain(indicator_id)
    return jsonify(list(results))


@app.route("/api/sparql/provenance/event/<event_id>")
def api_provenance_event(event_id):
    """PROV-O: how an event was generated."""
    results = sq.provenance_event(event_id)
    return jsonify(list(results))


# ═══════════════════════════════════════════
# Original API — Event detection (needs live NetCDF)
# ═══════════════════════════════════════════

@app.route("/api/detect", methods=["POST"])
def api_detect():
    body = request.get_json() or {}
    date_str = body.get("date", "2025-08-19")
    hazard_type = body.get("hazard_type", None)
    hazard_types = [hazard_type] if hazard_type else None
    events = qe.detect_events(date_str, hazard_types)
    results = []
    for e in events:
        converter.add_event(e)  # Also write to RDF
        results.append({
            "event_id": e.event_id, "date": e.date,
            "hazard_type": e.hazard_type, "severity": e.severity,
            "severity_score": e.severity_score, "confidence": e.confidence,
            "area_km2": e.area_km2, "cells_count": len(e.affected_cells),
            "centroid_lat": e.centroid_lat, "centroid_lon": e.centroid_lon,
            "region": e.region, "peak_risk": e.peak_risk,
            "trigger_details": e.trigger_details,
        })
    return jsonify({"date": date_str, "events": results, "rdf_triples": len(converter.graph)})


@app.route("/api/kg/summary")
def api_kg_summary():
    return jsonify(qe.kg_summary())


if __name__ == "__main__":
    print("\nDashboard: http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
