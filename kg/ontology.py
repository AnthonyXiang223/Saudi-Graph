"""
Knowledge Graph builder for the Saudi extreme event early warning system.

Builds a networkx.MultiDiGraph from schema files (ontology.json, operators.json,
rules.json) and provides graph query/traversal functions.

Architecture:
    ~120 nodes, ~250 edges (MultiDiGraph supports multiple relationships
    between the same node pair — e.g. an indicator may be both derived_from
    and co_occurs_with another).
    5 node types: Indicator, DataSource, HazardType, Region, Event
    7 relationship types: derived_from, co_occurs_with, sourced_from,
                           contributes_to, detects, instance_of, located_in
"""

import json
import os

import networkx as nx

# ---------------------------------------------------------------------------
# Region definitions  (lat_min, lat_max, lon_min, lon_max)
# ---------------------------------------------------------------------------
_REGION_BBOXES = {
    "saudi_bbox":     (16, 32, 34, 56),
    "red_sea":        (16, 30, 34, 44),
    "persian_gulf":   (24, 30, 48, 56),
    "north_saudi":    (26, 32, 34, 56),
    "central_saudi":  (21, 26, 34, 56),
    "south_saudi":    (16, 21, 34, 56),
}

# DataSource metadata used when building DataSource nodes
_DATA_SOURCE_META = {
    "DS1": {
        "full_name": "ERA5 Monthly Reanalysis",
        "product": "ERA5",
        "resolution": "0.25 deg",
        "temporal_coverage": "1940-present",
    },
    "DS2": {
        "full_name": "ERA5 Daily Reanalysis",
        "product": "ERA5",
        "resolution": "0.25 deg",
        "temporal_coverage": "1940-present",
    },
    "DS4": {
        "full_name": "ERA5 Extremes / Aggregated Fields",
        "product": "ERA5",
        "resolution": "0.25 deg",
        "temporal_coverage": "1940-present",
    },
    "DS8": {
        "full_name": "GHCN-Daily Station Climatology",
        "product": "GHCN-Daily",
        "resolution": "station (nearest-neighbour mapped)",
        "temporal_coverage": "1991-2020",
    },
    "DS10": {
        "full_name": "GPM IMERG V07 Satellite Precipitation",
        "product": "GPM IMERG V07",
        "resolution": "0.1 deg",
        "temporal_coverage": "2000-present",
    },
    "SST": {
        "full_name": "OSTIA Sea Surface Temperature",
        "product": "OSTIA",
        "resolution": "0.05 deg",
        "temporal_coverage": "2007-present",
    },
}

# HazardType metadata
_HAZARD_TYPE_META = {
    "flash_flood": {
        "display_name": "Flash Flood",
        "description": (
            "Rapid-onset flooding driven by intense convective precipitation, "
            "high CAPE, strong IVT convergence, and saturated soils."
        ),
        "typical_season": "Oct-Apr",
    },
    "extreme_heat": {
        "display_name": "Extreme Heat",
        "description": (
            "Prolonged periods of abnormally high temperatures exceeding "
            "climatological and absolute thresholds."
        ),
        "typical_season": "May-Sep",
    },
    "dust_storm": {
        "display_name": "Dust Storm",
        "description": (
            "High winds, low humidity, and deep vertical wind shear "
            "lifting dust over arid surfaces."
        ),
        "typical_season": "Mar-Aug",
    },
    "coastal_humid_heat": {
        "display_name": "Coastal Humid Heat",
        "description": (
            "Combined high temperature and humidity along the Red Sea and "
            "Persian Gulf coasts, driven by warm SST and onshore moisture."
        ),
        "typical_season": "Jun-Oct",
    },
}


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------
def infer_region(lat, lon):
    """Return a list of Region ids whose bounding box contains (*lat*, *lon*).

    Parameters
    ----------
    lat : float
        Latitude in degrees north.
    lon : float
        Longitude in degrees east.

    Returns
    -------
    list[str]
        Region ids that contain the point.  May be empty if the point falls
        outside every defined region.
    """
    matches = []
    for region_id, (lat_min, lat_max, lon_min, lon_max) in _REGION_BBOXES.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            matches.append(region_id)
    return matches


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------
class KnowledgeGraph:
    """Build, validate, and query the Saudi extreme-event knowledge graph.

    Uses a ``networkx.MultiDiGraph`` so that the same pair of nodes may carry
    more than one relationship (e.g. ``derived_from`` and ``co_occurs_with``).
    """

    def __init__(self, schema_dir):
        """Load schema files from *schema_dir*.

        Parameters
        ----------
        schema_dir : str
            Path to the directory containing ``ontology.json``,
            ``operators.json`` and ``rules.json``.
        """
        self.schema_dir = schema_dir
        self.ontology = self._load_json("ontology.json")
        self.operators_data = self._load_json("operators.json")
        self.rules_data = self._load_json("rules.json")
        self.graph = nx.MultiDiGraph()
        self._op_by_id = {}
        self._validation_issues = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_json(self, filename):
        path = os.path.join(self.schema_dir, filename)
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _parse_sources(source_str):
        """Split a compound source string like ``"DS2+DS8"`` into individual
        DataSource ids."""
        if source_str is None:
            return []
        return [s.strip() for s in source_str.split("+") if s.strip()]

    @staticmethod
    def _edge_has_relationship(graph, u, v, relationship):
        """Return ``True`` if *any* edge from *u* to *v* carries
        *relationship*."""
        if not graph.has_edge(u, v):
            return False
        for _key, edata in graph[u][v].items():
            if edata.get("relationship") == relationship:
                return True
        return False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self):
        """Construct the full ``networkx.MultiDiGraph`` from the loaded schema
        data.

        Node types created
        ------------------
        - **Indicator**  (from operators.json)
        - **DataSource** (DS1, DS2, DS4, DS8, DS10, SST)
        - **HazardType** (flash_flood, extreme_heat, dust_storm,
          coastal_humid_heat)
        - **Region**     (saudi_bbox, red_sea, persian_gulf, north_saudi,
          central_saudi, south_saudi)

        Edge types created
        ------------------
        - ``derived_from``:  Indicator -> Indicator  (operator inputs)
        - ``co_occurs_with``: Indicator <-> Indicator  (bidirectional)
        - ``sourced_from``:  Indicator -> DataSource
        - ``contributes_to``: Indicator -> HazardType  (rule conditions)
        - ``detects``:       Rule -> HazardType
        """
        # ---- DataSource nodes --------------------------------------------
        for ds_id, meta in _DATA_SOURCE_META.items():
            self.graph.add_node(ds_id, type="DataSource", **meta)

        # ---- Indicator nodes ---------------------------------------------
        self._op_by_id = {}
        for op in self.operators_data.get("operators", []):
            oid = op["id"]
            self._op_by_id[oid] = op
            props = {
                "type": "Indicator",
                "description": op.get("description", ""),
                "category": op.get("category", ""),
                "expression": op.get("expression", ""),
                "dag": op.get("dag"),
                "output_unit": op.get("output_unit", ""),
                "source": op.get("source", ""),
                "availability": op.get("availability"),
                "limitations": op.get("limitations"),
            }
            self.graph.add_node(oid, **props)

        # ---- HazardType nodes --------------------------------------------
        for ht_id, meta in _HAZARD_TYPE_META.items():
            self.graph.add_node(ht_id, type="HazardType", **meta)

        # ---- Region nodes ------------------------------------------------
        for region_id, (lat_min, lat_max, lon_min, lon_max) in _REGION_BBOXES.items():
            self.graph.add_node(
                region_id,
                type="Region",
                display_name=region_id.replace("_", " ").title(),
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
            )

        # ---- derived_from edges ------------------------------------------
        for op in self.operators_data.get("operators", []):
            for inp_id in op.get("inputs", []):
                if self.graph.has_node(inp_id):
                    self.graph.add_edge(
                        inp_id, op["id"], relationship="derived_from"
                    )

        # ---- co_occurs_with edges (bidirectional) ------------------------
        seen_co = set()
        for op in self.operators_data.get("operators", []):
            for co_id in op.get("co_occurs_with", []):
                pair = tuple(sorted([op["id"], co_id]))
                if pair in seen_co:
                    continue
                seen_co.add(pair)
                self.graph.add_edge(
                    op["id"], co_id, relationship="co_occurs_with"
                )
                self.graph.add_edge(
                    co_id, op["id"], relationship="co_occurs_with"
                )

        # ---- sourced_from edges ------------------------------------------
        for op in self.operators_data.get("operators", []):
            for src in self._parse_sources(op.get("source", "")):
                if self.graph.has_node(src):
                    self.graph.add_edge(
                        op["id"], src, relationship="sourced_from"
                    )

        # ---- Rule nodes + contributes_to + detects -----------------------
        for rule in self.rules_data.get("rules", []):
            rule_id = rule["id"]
            hazard_type = rule["hazard_type"]

            self.graph.add_node(
                rule_id,
                type="Rule",
                hazard_type=hazard_type,
                conditions=rule.get("conditions", []),
                connectivity=rule.get("connectivity"),
                min_cluster_size=rule.get("min_cluster_size"),
                severity=rule.get("severity", []),
                fallback=rule.get("fallback"),
                region_filter=rule.get("region_filter"),
            )

            # detects: Rule -> HazardType
            if self.graph.has_node(hazard_type):
                self.graph.add_edge(
                    rule_id, hazard_type, relationship="detects"
                )

            # contributes_to: each condition indicator -> HazardType
            for cond in rule.get("conditions", []):
                indicator_id = cond.get("indicator", "")
                if indicator_id and self.graph.has_node(indicator_id):
                    # Avoid duplicate contributes_to edges for the same pair
                    if not self._edge_has_relationship(
                        self.graph, indicator_id, hazard_type, "contributes_to"
                    ):
                        self.graph.add_edge(
                            indicator_id,
                            hazard_type,
                            relationship="contributes_to",
                        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self):
        """Run all integrity checks on the graph.

        Returns
        -------
        list[str]
            Human-readable descriptions of every issue found.  An empty list
            means the graph is valid.
        """
        issues = []

        # -- 1. Orphan nodes (exclude Region — connected only at runtime) --
        _runtime_only_types = {"Region"}
        for node_id in self.graph.nodes():
            if self.graph.degree(node_id) == 0:
                ntype = self.graph.nodes[node_id].get("type", "")
                if ntype not in _runtime_only_types:
                    issues.append(
                        f"Orphan node: '{node_id}' (type={ntype}) has no edges."
                    )

        # -- 2. Edge targets exist ----------------------------------------
        for u, v, data in self.graph.edges(data=True, keys=False):
            if not self.graph.has_node(v):
                rel = data.get("relationship", "?")
                issues.append(
                    f"Edge target missing: '{u}' -[{rel}]-> '{v}' "
                    f"(target node '{v}' does not exist)"
                )

        # -- 3. No cycles in derived_from subgraph ------------------------
        derived_edges = [
            (u, v)
            for u, v, d in self.graph.edges(data=True, keys=False)
            if d.get("relationship") == "derived_from"
        ]
        if derived_edges:
            derived_subgraph = nx.DiGraph()
            derived_subgraph.add_nodes_from(self.graph.nodes())
            derived_subgraph.add_edges_from(derived_edges)
            if not nx.is_directed_acyclic_graph(derived_subgraph):
                try:
                    cycle = nx.find_cycle(derived_subgraph, orientation="original")
                    cycle_str = " -> ".join(u for u, _v, _d in cycle)
                    issues.append(
                        f"Cycle detected in derived_from chain: {cycle_str}"
                    )
                except nx.NetworkXNoCycle:
                    issues.append(
                        "Cycle detected in derived_from chain "
                        "(could not extract path)."
                    )

        # -- 4. Every operator id has a matching Indicator node -----------
        for op in self.operators_data.get("operators", []):
            oid = op["id"]
            if oid not in self.graph:
                issues.append(
                    f"Operator id '{oid}' has no corresponding graph node."
                )
            else:
                node_data = self.graph.nodes[oid]
                if node_data.get("type") != "Indicator":
                    issues.append(
                        f"Node '{oid}' exists but has type "
                        f"'{node_data.get('type')}', expected 'Indicator'."
                    )

        # -- 5. Operator inputs reference real indicators -----------------
        for op in self.operators_data.get("operators", []):
            for inp_id in op.get("inputs", []):
                if inp_id not in self.graph:
                    issues.append(
                        f"Operator '{op['id']}' references unknown input "
                        f"'{inp_id}'."
                    )

        # -- 6. co_occurs_with references are real indicators -------------
        for op in self.operators_data.get("operators", []):
            for co_id in op.get("co_occurs_with", []):
                if co_id not in self.graph:
                    issues.append(
                        f"Operator '{op['id']}' co_occurs_with unknown "
                        f"indicator '{co_id}'."
                    )

        # -- 7. Rule conditions reference real indicators -----------------
        for rule in self.rules_data.get("rules", []):
            for cond in rule.get("conditions", []):
                ind_id = cond.get("indicator", "")
                if ind_id and ind_id not in self.graph:
                    issues.append(
                        f"Rule '{rule['id']}' references unknown indicator "
                        f"'{ind_id}'."
                    )

        # -- 8. sourced_from targets a DataSource -------------------------
        for u, v, d in self.graph.edges(data=True, keys=False):
            if d.get("relationship") == "sourced_from":
                tgt = self.graph.nodes.get(v, {})
                if tgt.get("type") != "DataSource":
                    issues.append(
                        f"sourced_from edge '{u}' -> '{v}' does not point "
                        f"to a DataSource node (type={tgt.get('type')})."
                    )

        # -- 9. contributes_to targets a HazardType -----------------------
        for u, v, d in self.graph.edges(data=True, keys=False):
            if d.get("relationship") == "contributes_to":
                tgt = self.graph.nodes.get(v, {})
                if tgt.get("type") != "HazardType":
                    issues.append(
                        f"contributes_to edge '{u}' -> '{v}' does not point "
                        f"to a HazardType node (type={tgt.get('type')})."
                    )

        # -- 10. Missing DataSource references -----------------------------
        for op in self.operators_data.get("operators", []):
            oid = op["id"]
            srcs = self._parse_sources(op.get("source", ""))
            for src in srcs:
                if src not in self.graph:
                    issues.append(
                        f"Indicator '{oid}' references unknown "
                        f"DataSource '{src}'."
                    )

        self._validation_issues = issues
        return issues

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------
    def get_hazard_indicators(self, hazard_type):
        """Return the list of indicator ids that contribute to *hazard_type*.

        Parameters
        ----------
        hazard_type : str
            E.g. ``"flash_flood"``.

        Returns
        -------
        list[str]
            Indicator node ids (sorted).
        """
        indicators = set()
        for u, v, d in self.graph.edges(data=True, keys=False):
            if d.get("relationship") == "contributes_to" and v == hazard_type:
                if self.graph.nodes[u].get("type") == "Indicator":
                    indicators.add(u)
        return sorted(indicators)

    def get_indicator_chain(self, indicator_id):
        """Return the full derivation chain from *indicator_id* backwards via
        ``derived_from`` edges to leaf DataSource nodes.

        The traversal follows ``derived_from`` edges in reverse (predecessors)
        until it reaches indicators with no further ``derived_from``
        predecessors.  For those leaf indicators it then follows
        ``sourced_from`` edges to collect DataSource nodes.

        Parameters
        ----------
        indicator_id : str
            Starting indicator node id.

        Returns
        -------
        dict
            Keys:
            - ``"indicators"``: list[str] – all indicator ids in the chain
              (includes *indicator_id*).
            - ``"sources"``: list[str] – DataSource ids feeding the leaf
              indicators.
            - ``"edges"``: list[tuple[str, str]] – ``(from, to)``
              ``derived_from`` edges traversed.
        """
        if indicator_id not in self.graph:
            return {"indicators": [], "sources": [], "edges": []}

        visited_indicators = set()
        derived_edges = []
        data_sources = set()
        stack = [indicator_id]

        while stack:
            current = stack.pop()
            if current in visited_indicators:
                continue
            visited_indicators.add(current)

            # Walk backward along derived_from
            predecessors = []
            for pred in self.graph.predecessors(current):
                if self._edge_has_relationship(
                    self.graph, pred, current, "derived_from"
                ):
                    predecessors.append(pred)

            if predecessors:
                for pred in predecessors:
                    derived_edges.append((pred, current))
                    if pred not in visited_indicators:
                        stack.append(pred)
            else:
                # Leaf indicator: follow sourced_from to DataSource
                for _, ds, d in self.graph.out_edges(current, data=True, keys=False):
                    if d.get("relationship") == "sourced_from":
                        if self.graph.nodes[ds].get("type") == "DataSource":
                            data_sources.add(ds)

        return {
            "indicators": sorted(visited_indicators),
            "sources": sorted(data_sources),
            "edges": sorted(derived_edges),
        }

    def get_operator(self, indicator_id):
        """Return the full operator definition for *indicator_id* as stored in
        ``operators.json``.

        Parameters
        ----------
        indicator_id : str
            Indicator node id.

        Returns
        -------
        dict or None
            The operator object, or ``None`` if not found.
        """
        return self._op_by_id.get(indicator_id)

    def get_co_occurring(self, indicator_id):
        """Return indicator ids connected to *indicator_id* via
        ``co_occurs_with`` edges.

        Parameters
        ----------
        indicator_id : str
            Indicator node id.

        Returns
        -------
        list[str]
            Co-occurring indicator ids (sorted).
        """
        if indicator_id not in self.graph:
            return []
        co = []
        for _, neighbor, d in self.graph.out_edges(
            indicator_id, data=True, keys=False
        ):
            if d.get("relationship") == "co_occurs_with":
                if self.graph.nodes[neighbor].get("type") == "Indicator":
                    co.append(neighbor)
        return sorted(co)

    def summary(self):
        """Return a summary dictionary with node and edge counts by type.

        Returns
        -------
        dict
            Keys:
            - ``"nodes"``: dict[str, int]  node-type -> count
            - ``"edges"``: dict[str, int]  relationship -> count
            - ``"total_nodes"``: int
            - ``"total_edges"``: int
        """
        node_counts = {}
        for _, data in self.graph.nodes(data=True):
            ntype = data.get("type", "Unknown")
            node_counts[ntype] = node_counts.get(ntype, 0) + 1

        edge_counts = {}
        for _u, _v, data in self.graph.edges(data=True, keys=False):
            rel = data.get("relationship", "Unknown")
            edge_counts[rel] = edge_counts.get(rel, 0) + 1

        return {
            "nodes": dict(sorted(node_counts.items())),
            "edges": dict(sorted(edge_counts.items())),
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
        }


# ---------------------------------------------------------------------------
# Convenience: build-and-validate when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Default schema directory is relative to this file's location
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    schema_dir = os.path.join(base_dir, "schema")

    kg = KnowledgeGraph(schema_dir)
    kg.build()

    issues = kg.validate()
    if issues:
        print(f"VALIDATION: {len(issues)} issue(s) found:")
        for i, issue in enumerate(issues, 1):
            print(f"  [{i}] {issue}")
    else:
        print("VALIDATION: All checks passed.")

    s = kg.summary()
    print(f"\nGraph summary:")
    print(f"  Nodes: {s['total_nodes']}  ({s['nodes']})")
    print(f"  Edges: {s['total_edges']}  ({s['edges']})")

    # Quick smoke-tests
    for ht in ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]:
        inds = kg.get_hazard_indicators(ht)
        print(f"\n  {ht}: {len(inds)} indicator(s) contribute")

    # Test indicator chain for a composite indicator
    chain = kg.get_indicator_chain("flash_flood_risk")
    print(
        f"\n  flash_flood_risk chain: {len(chain['indicators'])} indicators, "
        f"{len(chain['sources'])} sources, "
        f"{len(chain['edges'])} derived_from edges"
    )

    # Test co-occurring
    co = kg.get_co_occurring("daily_precip_total")
    print(f"  daily_precip_total co-occurs with: {co}")

    # Test region inference
    print("\nRegion inference:")
    for pt in [(24.7, 46.7), (18.2, 40.1), (27.5, 50.2), (30.1, 35.5)]:
        regions = infer_region(*pt)
        print(f"    ({pt[0]}, {pt[1]}) -> {regions}")
