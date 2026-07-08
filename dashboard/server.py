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

    # Extract Saudi-specific nodes (skip DMDO internal classes/axioms)
    PREFIX = "https://mazu.cma/saudi#"
    for s, p, o in g.triples((None, None, None)):
        s_str = str(s)
        if PREFIX in s_str and s_str not in seen_nodes:
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
        if rel and PREFIX in str(s) and PREFIX in str(o):
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
