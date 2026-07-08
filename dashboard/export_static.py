"""
Export all KG data as static JSON files for GitHub Pages hosting.
Removes the need for a Flask backend — the dashboard becomes a pure static site.
"""

import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kg.ontology import KnowledgeGraph

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_DIR = os.path.join(PROJECT_DIR, "schema")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "api")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Build KG
kg = KnowledgeGraph(SCHEMA_DIR)
kg.build()
summary = kg.summary()

# ── Export 1: Full graph data ──
nodes = []
for node, data in kg.graph.nodes(data=True):
    nodes.append({
        "id": node,
        "type": data.get("type", "unknown"),
        "label": node,
        "group": data.get("type", "unknown"),
        "title": f"{node}\n{data.get('description','')}".strip(),
        "description": data.get("description", ""),
        "output_unit": data.get("output_unit", ""),
        "source": data.get("source", ""),
        "category": data.get("category", ""),
        "expression": data.get("expression", ""),
        "availability": data.get("availability"),
        "limitations": data.get("limitations"),
        "display_name": data.get("display_name", ""),
        "lat_min": data.get("lat_min"),
        "lat_max": data.get("lat_max"),
        "lon_min": data.get("lon_min"),
        "lon_max": data.get("lon_max"),
        "hazard_type": data.get("hazard_type", ""),
        "conditions": data.get("conditions", []),
        "severity": data.get("severity", []),
        "fallback": data.get("fallback"),
        "connectivity": data.get("connectivity"),
        "min_cluster_size": data.get("min_cluster_size"),
        "full_name": data.get("full_name", ""),
        "product": data.get("product", ""),
        "resolution": data.get("resolution", ""),
    })

edges = []
for u, v, data in kg.graph.edges(data=True):
    edges.append({
        "from": u,
        "to": v,
        "label": data.get("relationship", ""),
        "arrows": "to" if data.get("relationship") != "co_occurs_with" else "to,from",
    })

graph_data = {
    "total_nodes": summary["total_nodes"],
    "total_edges": summary["total_edges"],
    "nodes": nodes,
    "edges": edges,
}
with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(graph_data, f, ensure_ascii=False)
print(f"Exported: summary.json ({len(nodes)} nodes, {len(edges)} edges)")

# ── Export 2: Per-indicator detail files ──
for node_id, data in kg.graph.nodes(data=True):
    if data.get("type") != "Indicator":
        continue

    op = kg.get_operator(node_id) or {}
    chain = kg.get_indicator_chain(node_id)
    co = kg.get_co_occurring(node_id)

    detail = {
        "operator": op,
        "chain": chain,
        "co_occurring": co,
    }
    with open(os.path.join(OUTPUT_DIR, f"indicator_{node_id}.json"), "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False)

print(f"Exported: {len([n for n,d in kg.graph.nodes(data=True) if d.get('type')=='Indicator'])} indicator detail files")

# ── Export 3: Per-hazard indicator lists ──
for htype in ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]:
    indicators = kg.get_hazard_indicators(htype)
    details = []
    for ind_id in indicators:
        op = kg.get_operator(ind_id)
        details.append({
            "id": ind_id,
            "description": op.get("description", "") if op else "",
            "unit": op.get("output_unit", "") if op else "",
            "source": op.get("source", "") if op else "",
        })
    with open(os.path.join(OUTPUT_DIR, f"hazard_{htype}.json"), "w", encoding="utf-8") as f:
        json.dump({"hazard_type": htype, "indicators": details}, f, ensure_ascii=False)
    print(f"Exported: hazard_{htype}.json ({len(details)} indicators)")

# ── Export 4: Search index ──
search_data = []
for node, data in kg.graph.nodes(data=True):
    if data.get("type") == "Indicator":
        search_data.append({
            "id": node,
            "description": data.get("description", ""),
            "category": data.get("category", ""),
            "unit": data.get("output_unit", ""),
        })
with open(os.path.join(OUTPUT_DIR, "search.json"), "w", encoding="utf-8") as f:
    json.dump(search_data, f, ensure_ascii=False)
print(f"Exported: search.json ({len(search_data)} indicators)")

print(f"\nDone! All files in: {OUTPUT_DIR}")
print("Copy dashboard/static/ to your GitHub Pages repo.")
