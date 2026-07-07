"""
Dashboard server for Saudi Extreme Event Knowledge Graph.
Flask backend serving KG queries, event detection, and indicator computation.
"""

import json
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, render_template, send_from_directory
from kg.query import QueryEngine

app = Flask(__name__)
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))

# Init once at startup
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_DIR = os.path.join(PROJECT_DIR, "schema")
DATA_DIR = os.path.join(PROJECT_DIR, "indicators")

print("Initializing QueryEngine...")
qe = QueryEngine(SCHEMA_DIR, DATA_DIR)
qe.init()
print("Ready.")


# ── KG Knowledge Endpoints ──

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/kg/summary")
def api_kg_summary():
    """Get knowledge graph summary statistics."""
    summary = qe.kg_summary()
    # Add node/edge lists for visualization
    nodes = []
    edges = []
    for node, data in qe.kg.graph.nodes(data=True):
        nodes.append({
            "id": node,
            "type": data.get("type", "unknown"),
            "label": node,
            "group": data.get("type", "unknown"),
            "title": _format_node_title(node, data),
            **{k: v for k, v in data.items() if k not in ("dag", "conditions")}
        })
    for u, v, data in qe.kg.graph.edges(data=True):
        edges.append({
            "from": u,
            "to": v,
            "label": data.get("relationship", ""),
            "arrows": "to",
        })

    summary["nodes"] = nodes
    summary["edges"] = edges
    return jsonify(summary)


@app.route("/api/kg/indicator/<indicator_id>")
def api_indicator(indicator_id):
    """Get full indicator details including operator."""
    op = qe.get_operator(indicator_id)
    chain = qe.get_indicator_chain(indicator_id)
    co = qe.get_co_occurring(indicator_id)
    return jsonify({
        "operator": op,
        "chain": chain,
        "co_occurring": co,
    })


@app.route("/api/kg/hazard/<hazard_type>")
def api_hazard_indicators(hazard_type):
    """Get indicators contributing to a hazard type."""
    indicators = qe.get_hazard_indicators(hazard_type)
    details = []
    for ind_id in indicators:
        op = qe.get_operator(ind_id)
        details.append({
            "id": ind_id,
            "description": op.get("description", "") if op else "",
            "unit": op.get("output_unit", "") if op else "",
            "source": op.get("source", "") if op else "",
        })
    return jsonify({"hazard_type": hazard_type, "indicators": details})


@app.route("/api/kg/chain/<indicator_id>")
def api_chain(indicator_id):
    """Get the full derivation chain for an indicator."""
    chain = qe.get_indicator_chain(indicator_id)
    # Enrich with descriptions
    for ind_id in chain.get("indicators", []):
        op = qe.get_operator(ind_id)
        chain[f"_desc_{ind_id}"] = op.get("description", "") if op else ""
    return jsonify(chain)


@app.route("/api/kg/search")
def api_search():
    """Search indicators by keyword."""
    q = request.args.get("q", "").lower()
    if not q:
        return jsonify([])
    results = []
    for node, data in qe.kg.graph.nodes(data=True):
        if data.get("type") == "Indicator":
            desc = data.get("description", "").lower()
            if q in node.lower() or q in desc:
                results.append({
                    "id": node,
                    "description": data.get("description", ""),
                    "category": data.get("category", ""),
                    "unit": data.get("output_unit", ""),
                })
    return jsonify(results[:20])


# ── Event Detection Endpoints ──

@app.route("/api/detect", methods=["POST"])
def api_detect():
    """Run event detection for a date and hazard type."""
    body = request.get_json() or {}
    date_str = body.get("date", "2025-08-19")
    hazard_type = body.get("hazard_type", None)
    if hazard_type:
        hazard_types = [hazard_type]
    else:
        hazard_types = None

    events = qe.detect_events(date_str, hazard_types)
    results = []
    for e in events:
        results.append({
            "event_id": e.event_id,
            "date": e.date,
            "hazard_type": e.hazard_type,
            "severity": e.severity,
            "severity_score": e.severity_score,
            "confidence": e.confidence,
            "area_km2": e.area_km2,
            "cells_count": len(e.affected_cells),
            "centroid_lat": e.centroid_lat,
            "centroid_lon": e.centroid_lon,
            "region": e.region,
            "peak_risk": e.peak_risk,
            "trigger_details": e.trigger_details,
        })
    return jsonify({"date": date_str, "events": results})


# ── Indicator Computation Endpoint ──

@app.route("/api/compute", methods=["POST"])
def api_compute():
    """Evaluate an indicator's DAG with sample context values."""
    from kg.operators import evaluate_dag, OPS

    body = request.get_json() or {}
    indicator_id = body.get("indicator_id", "")
    context = body.get("context", {})

    op = qe.get_operator(indicator_id)
    if not op:
        return jsonify({"error": f"Indicator not found: {indicator_id}"}), 404

    dag = op.get("dag")
    if not dag:
        return jsonify({"error": "No DAG defined for this indicator"}), 400

    try:
        # Resolve inputs from context or from chain
        if not context:
            # Try to compute from leaf inputs
            chain = qe.get_indicator_chain(indicator_id)
            for src in chain.get("sources", []):
                if src not in context:
                    context[src] = None  # Placeholder

        result = evaluate_dag(dag, context)
        return jsonify({
            "indicator_id": indicator_id,
            "expression": op.get("expression", ""),
            "dag": dag,
            "context": context,
            "result": result,
        })
    except Exception as e:
        return jsonify({"error": str(e), "indicator_id": indicator_id}), 400


# ── Data Layer Endpoints ──

@app.route("/api/data/timeseries")
def api_timeseries():
    """Get a time series for a location."""
    lat = float(request.args.get("lat", 24.7))
    lon = float(request.args.get("lon", 46.7))
    start = request.args.get("start", "2025-07-01")
    end = request.args.get("end", "2025-07-31")
    variables = request.args.get("variables", "tmax_c,t2m_c").split(",")

    try:
        df = qe.get_timeseries(lat, lon, start, end, variables)
        return jsonify({
            "lat": lat, "lon": lon,
            "nearest_lat": df.attrs.get("nearest_lat", lat),
            "nearest_lon": df.attrs.get("nearest_lon", lon),
            "data": df.reset_index().to_dict(orient="records"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/data/snapshot")
def api_snapshot():
    """Get a spatial snapshot for a variable on a date."""
    date_str = request.args.get("date", "2025-08-19")
    variable = request.args.get("variable", "daily_precip_total")

    try:
        data, lats, lons = qe.data.get_spatial_snapshot(date_str, variable)
        # Return as compressed grid data
        return jsonify({
            "date": date_str,
            "variable": variable,
            "shape": list(data.shape),
            "lat_range": [float(lats.min()), float(lats.max())],
            "lon_range": [float(lons.min()), float(lons.max())],
            "min": float(data.nanmin()) if hasattr(data, 'nanmin') else float(data.min()),
            "max": float(data.nanmax()) if hasattr(data, 'nanmax') else float(data.max()),
            "data": data.tolist(),
            "lats": lats.tolist(),
            "lons": lons.tolist(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Helpers ──

def _format_node_title(node_id, data):
    """Format a node's tooltip title."""
    ntype = data.get("type", "")
    if ntype == "Indicator":
        desc = data.get("description", "")
        unit = data.get("output_unit", "")
        source = data.get("source", "")
        return f"{node_id}\n{desc}\nUnit: {unit}\nSource: {source}"
    elif ntype == "HazardType":
        return f"{node_id}\n{data.get('display_name', '')}\n{data.get('description', '')}"
    elif ntype == "DataSource":
        return f"{node_id}\n{data.get('full_name', '')}\n{data.get('product', '')}"
    return node_id


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(os.path.join(DASHBOARD_DIR, "static"), filename)


if __name__ == "__main__":
    print("\nDashboard: http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
