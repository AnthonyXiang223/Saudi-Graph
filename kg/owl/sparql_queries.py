"""
SPARQL query equivalents for the Saudi KG — DMDO-aligned.
Each function returns a SPARQL query string + a Python runner.

Usage:
    from kg.owl.sparql_queries import SPARQLQueries
    sq = SPARQLQueries(converter)
    results = sq.hazard_indicators("flash_flood")
"""

from rdflib import Graph, Literal
from typing import List, Dict

SAUDI = "https://mazu.cma/saudi#"
DEO = "http://purl.org/disaster/deo#"
SOSA = "http://www.w3.org/ns/sosa/"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
PROV = "http://www.w3.org/ns/prov#"
TIME = "http://www.w3.org/2006/time#"
XSD = "http://www.w3.org/2001/XMLSchema#"
GEO = "http://www.opengis.net/ont/geosparql#"
DPO = "http://purl.org/disaster/dpo#"
QUDT = "http://qudt.org/schema/qudt#"

PREFIXES = f"""
PREFIX deo: <{DEO}>
PREFIX saudi: <{SAUDI}>
PREFIX sosa: <{SOSA}>
PREFIX rdfs: <{RDFS}>
PREFIX prov: <{PROV}>
PREFIX time: <{TIME}>
PREFIX xsd: <{XSD}>
PREFIX geo: <{GEO}>
PREFIX dpo: <{DPO}>
PREFIX qudt: <{QUDT}>
"""


def _uri(kind: str, name: str) -> str:
    """Build a full URI: <https://mazu.cma/saudi#Kind/name>"""
    return f"<{SAUDI}{kind}/{name}>"


class SPARQLQueries:
    """Collection of SPARQL queries that mirror the networkx query layer."""

    def __init__(self, converter):
        self.converter = converter
        self.graph: Graph = converter.graph

    def _run(self, query: str) -> List[Dict]:
        """Execute SPARQL and return list of dicts."""
        results = self.graph.query(query)
        vars = [str(v) for v in results.vars]
        return [{v: str(row[v]) if row[v] is not None else None for v in vars} for row in results]

    # ═══════════════════════════════════════════════════════════
    # Layer 1: Knowledge queries
    # ═══════════════════════════════════════════════════════════

    def hazard_indicators(self, hazard_type: str) -> List[Dict]:
        ht_uri = _uri("HazardType", hazard_type)
        q = PREFIXES + f"""
        SELECT ?indicator ?label ?description WHERE {{
            ?hazard deo:hazardType {ht_uri} ;
                    deo:hasHazardProperty ?indicator .
            ?indicator rdfs:label ?label .
            OPTIONAL {{ ?indicator rdfs:comment ?description . }}
        }}
        """
        return self._run(q)

    def indicator_chain(self, indicator_id: str) -> List[Dict]:
        ind_uri = _uri("Indicator", indicator_id)
        q = PREFIXES + f"""
        SELECT ?derived ?source WHERE {{
            ?derived prov:wasDerivedFrom+ {ind_uri} .
            ?derived prov:wasDerivedFrom ?source .
        }}
        """
        return self._run(q)

    def co_occurring(self, indicator_id: str) -> List[str]:
        ind_uri = _uri("Indicator", indicator_id)
        q = PREFIXES + f"""
        SELECT DISTINCT ?co WHERE {{
            {{ {ind_uri} saudi:coOccursWith ?co . }}
            UNION
            {{ ?co saudi:coOccursWith {ind_uri} . }}
        }}
        """
        results = self._run(q)
        return [r["co"].split("/")[-1].rstrip(">") for r in results if r.get("co")]

    def indicator_detail(self, indicator_id: str) -> Dict:
        ind_uri = _uri("Indicator", indicator_id)
        q = PREFIXES + f"""
        SELECT ?label ?description ?expression ?dag ?source ?effDays ?totalDays WHERE {{
            {ind_uri} rdfs:label ?label ;
                      rdfs:comment ?description .
            OPTIONAL {{ {ind_uri} saudi:expression ?expression . }}
            OPTIONAL {{ {ind_uri} saudi:hasDAG ?dag . }}
            OPTIONAL {{ {ind_uri} prov:wasDerivedFrom ?source . }}
            OPTIONAL {{ {ind_uri} saudi:effectiveDays ?effDays . }}
            OPTIONAL {{ {ind_uri} saudi:totalDays ?totalDays . }}
        }}
        """
        results = self._run(q)
        return results[0] if results else {}

    def search_indicators(self, keyword: str) -> List[Dict]:
        kw = keyword.lower()
        q = PREFIXES + f"""
        SELECT ?indicator ?label ?description WHERE {{
            ?indicator a sosa:ObservableProperty ;
                       rdfs:label ?label ;
                       rdfs:comment ?description .
            FILTER(CONTAINS(LCASE(?label), "{kw}") || CONTAINS(LCASE(?description), "{kw}"))
        }}
        LIMIT 20
        """
        return self._run(q)

    # ═══════════════════════════════════════════════════════════
    # Layer 2: Hazard + Event chain queries
    # ═══════════════════════════════════════════════════════════

    def hazard_to_disaster_chain(self, hazard_type: str) -> List[Dict]:
        ht_uri = _uri("HazardType", hazard_type)
        q = PREFIXES + f"""
        SELECT ?hazard ?disaster ?severity WHERE {{
            ?hazard deo:hazardType {ht_uri} .
            ?hazard deo:possiblyCauses ?disaster .
            ?disaster a deo:Disaster .
            OPTIONAL {{ ?disaster dpo:Severity ?severity . }}
        }}
        """
        return self._run(q)

    def events_in_region(self, region_id: str, date_start: str = None, date_end: str = None) -> List[Dict]:
        reg_uri = _uri("Region", region_id)
        date_filter = ""
        if date_start:
            date_filter += f'FILTER(?date >= "{date_start}"^^xsd:date)\n'
        if date_end:
            date_filter += f'FILTER(?date <= "{date_end}"^^xsd:date)\n'

        q = PREFIXES + f"""
        SELECT ?event ?label ?hazardType ?severity ?date ?area WHERE {{
            ?event a deo:Disaster ;
                   rdfs:label ?label ;
                   deo:hazardType ?hazardType ;
                   deo:hasTemporalScope ?t ;
                   geo:hasGeometry {reg_uri} .
            ?t time:inXSDDate ?date .
            OPTIONAL {{ ?event dpo:Severity ?severity . }}
            OPTIONAL {{ ?event saudi:areaKm2 ?area . }}
            {date_filter}
        }}
        ORDER BY DESC(?severity)
        """
        return self._run(q)

    def what_hazards_affect_region(self, region_id: str) -> List[Dict]:
        reg_uri = _uri("Region", region_id)
        q = PREFIXES + f"""
        SELECT DISTINCT ?hazardType WHERE {{
            ?disaster a deo:Disaster ;
                      deo:hazardType ?hazardType ;
                      geo:hasGeometry {reg_uri} .
        }}
        """
        return self._run(q)

    def which_indicators_drove_event(self, event_id: str) -> List[Dict]:
        ev_uri = _uri("Event", event_id)
        q = PREFIXES + f"""
        SELECT ?indicator ?value WHERE {{
            {ev_uri} saudi:hasTriggerObservation ?obs .
            ?obs sosa:observedProperty ?indicator ;
                 sosa:hasSimpleResult ?value .
        }}
        """
        return self._run(q)

    def compare_hazard_severity(self, hazard_type: str) -> List[Dict]:
        ht_uri = _uri("HazardType", hazard_type)
        q = PREFIXES + f"""
        SELECT ?event ?date ?severity ?area WHERE {{
            ?disaster a deo:Disaster ;
                      deo:hazardType {ht_uri} ;
                      dpo:Severity ?severity ;
                      deo:hasTemporalScope ?t ;
                      saudi:areaKm2 ?area .
            ?t time:inXSDDate ?date .
        }}
        ORDER BY DESC(?severity)
        """
        return self._run(q)


    # ═══════════════════════════════════════════════════════════
    # Layer 4: SOSA Observation queries (NetCDF values in RDF)
    # ═══════════════════════════════════════════════════════════

    def observation_by_indicator(self, indicator_id: str, date_str: str = None,
                                  min_value: float = None) -> List[Dict]:
        """Query all SOSA Observations for an indicator, optionally filtered."""
        ind_uri = _uri("Indicator", indicator_id)
        date_filter = ""
        val_filter = ""
        if date_str:
            date_clean = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}" if len(date_str) == 8 else date_str
            date_filter = f'FILTER(?date = "{date_clean}"^^xsd:date)'
        if min_value is not None:
            val_filter = f"FILTER(?value > {min_value})"

        q = PREFIXES + f"""
        SELECT ?lat ?lon ?value ?date WHERE {{
            ?obs sosa:observedProperty {ind_uri} ;
                 sosa:hasSimpleResult ?value ;
                 sosa:resultTime ?date ;
                 sosa:hasFeatureOfInterest ?grid .
            ?grid geo:hasGeometry ?geom .
            BIND(STRBEFORE(STRAFTER(STR(?geom), "POINT("), " ") AS ?lon)
            BIND(STRBEFORE(STRAFTER(STR(?geom), " "), ")") AS ?lat)
            {date_filter}
            {val_filter}
        }}
        ORDER BY DESC(?value)
        """
        return self._run(q)

    def observations_summary(self, date_str: str, indicator_ids: list = None) -> Dict:
        """Get observation count and stats for a date."""
        date_clean = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}" if len(date_str) == 8 else date_str

        if indicator_ids is None:
            # Count all
            q = PREFIXES + f"""
            SELECT (COUNT(?obs) as ?count) WHERE {{
                ?obs sosa:resultTime "{date_clean}"^^xsd:date .
            }}
            """
            results = self._run(q)
            return {"date": date_str, "total_observations": int(results[0]["count"]) if results else 0}

        q = PREFIXES + f"""
        SELECT ?indicator (COUNT(?obs) as ?count) (MAX(?value) as ?maxVal) (MIN(?value) as ?minVal) WHERE {{
            ?obs sosa:observedProperty ?indicator ;
                 sosa:hasSimpleResult ?value ;
                 sosa:resultTime "{date_clean}"^^xsd:date .
        }}
        GROUP BY ?indicator
        ORDER BY DESC(?count)
        """
        return self._run(q)

    def compare_indicator_dates(self, indicator_id: str, dates: list) -> List[Dict]:
        """Compare an indicator's observation stats across multiple dates."""
        ind_uri = _uri("Indicator", indicator_id)
        date_values = " ".join([f'"{d}"^^xsd:date' for d in dates])

        q = PREFIXES + f"""
        SELECT ?date (COUNT(?obs) as ?count) (MAX(?value) as ?max) (AVG(?value) as ?avg) WHERE {{
            ?obs sosa:observedProperty {ind_uri} ;
                 sosa:hasSimpleResult ?value ;
                 sosa:resultTime ?date .
            FILTER(?date IN ({date_values}))
        }}
        GROUP BY ?date
        ORDER BY ?date
        """
        return self._run(q)


    # ═══════════════════════════════════════════════════════════
    # Layer 5: GeoSPARQL spatial queries (P1)
    # ═══════════════════════════════════════════════════════════

    def observations_within_radius(self, indicator_id: str, center_lat: float,
                                    center_lon: float, radius_km: float,
                                    date_str: str = None, min_value: float = None) -> List[Dict]:
        """
        GeoSPARQL: find observations within radius of a point.
        Uses OGC GeoSPARQL 1.1 geof:distance function.

        Args:
            indicator_id: e.g. "tmax_c"
            center_lat, center_lon: center point in decimal degrees
            radius_km: search radius in kilometers
        """
        ind_uri = _uri("Indicator", indicator_id)
        deg_radius = radius_km / 111.32  # ~degrees at equator

        date_filter = ""
        val_filter = ""
        if date_str:
            date_clean = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}" if len(date_str) == 8 else date_str
            date_filter = f'FILTER(?date = "{date_clean}"^^xsd:date)'
        if min_value is not None:
            val_filter = f"FILTER(?val > {min_value})"

        q = PREFIXES + f"""
        SELECT ?lat ?lon ?val ?date WHERE {{
            ?obs sosa:observedProperty {ind_uri} ;
                 sosa:hasResult ?result ;
                 sosa:resultTime ?date ;
                 sosa:hasFeatureOfInterest ?foi .
            ?result qudt:numericValue ?val .
            ?foi geof:hasGeometry ?geom .
            ?geom geof:asWKT ?wkt .

            # GeoSPARQL spatial filter
            BIND(geof:distance(
                ?wkt,
                "POINT({center_lon} {center_lat})"^^geof:wktLiteral,
                <http://www.opengis.net/def/uom/OGC/1.0/degree>
            ) AS ?dist)
            FILTER(?dist < {deg_radius})

            BIND(STRBEFORE(STRAFTER(STR(?wkt), "POINT("), " ") AS ?lonStr)
            BIND(STRBEFORE(STRAFTER(STR(?wkt), " "), ")") AS ?latStr)
            BIND(xsd:float(?lonStr) AS ?lon)
            BIND(xsd:float(?latStr) AS ?lat)
            {date_filter}
            {val_filter}
        }}
        ORDER BY ?dist
        LIMIT 100
        """
        return self._run(q)

    def events_intersecting_region(self, region_id: str) -> List[Dict]:
        """
        GeoSPARQL: find events whose spatial extent intersects a region.
        Uses geof:sfIntersects for topological spatial query.
        """
        reg_uri = _uri("Region", region_id)
        q = PREFIXES + f"""
        SELECT ?event ?hazardType ?severity WHERE {{
            {reg_uri} geof:hasGeometry ?regGeom .
            ?regGeom geof:asWKT ?regWKT .

            ?event a deo:Disaster ;
                   deo:hazardType ?hazardType ;
                   dpo:Severity ?severity ;
                   saudi:centroidLat ?lat ;
                   saudi:centroidLon ?lon .

            # Check if event centroid falls within region polygon
            BIND(STRDT(CONCAT("POINT(", STR(?lon), " ", STR(?lat), ")"), geof:wktLiteral) AS ?eventWKT)
            FILTER(geof:sfIntersects(?eventWKT, ?regWKT))
        }}
        ORDER BY DESC(?severity)
        """
        return self._run(q)

    def spatial_aggregation(self, indicator_id: str, date_str: str,
                            region_id: str = None) -> List[Dict]:
        """
        GeoSPARQL: aggregate observations by region.
        Returns count, max, avg for an indicator within a region.
        """
        ind_uri = _uri("Indicator", indicator_id)
        date_clean = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}" if len(date_str) == 8 else date_str

        region_filter = ""
        if region_id:
            reg_uri = _uri("Region", region_id)
            region_filter = f"""
            {reg_uri} geof:hasGeometry ?regGeom .
            ?regGeom geof:asWKT ?regWKT .
            FILTER(geof:sfIntersects(?wkt, ?regWKT))
            """

        q = PREFIXES + f"""
        SELECT (COUNT(?obs) AS ?count) (MAX(?val) AS ?maxVal) (AVG(?val) AS ?avgVal) WHERE {{
            ?obs sosa:observedProperty {ind_uri} ;
                 sosa:hasResult ?result ;
                 sosa:resultTime "{date_clean}"^^xsd:date ;
                 sosa:hasFeatureOfInterest ?foi .
            ?result qudt:numericValue ?val .
            ?foi geof:hasGeometry ?geom .
            ?geom geof:asWKT ?wkt .
            {region_filter}
        }}
        """
        return self._run(q)

    def neighborhood_query(self, indicator_id: str, date_str: str,
                           hotspot_threshold: float) -> List[Dict]:
        """
        Find spatial clusters of observations exceeding a threshold.
        Returns adjacent grid cells that all exceed the threshold — implicit GeoSPARQL clustering.
        """
        ind_uri = _uri("Indicator", indicator_id)
        date_clean = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}" if len(date_str) == 8 else date_str

        q = PREFIXES + f"""
        SELECT ?lat ?lon ?val WHERE {{
            ?obs sosa:observedProperty {ind_uri} ;
                 sosa:hasResult ?result ;
                 sosa:resultTime "{date_clean}"^^xsd:date ;
                 sosa:hasFeatureOfInterest ?foi .
            ?result qudt:numericValue ?val .
            ?foi geof:hasGeometry ?geom .
            ?geom geof:asWKT ?wkt .
            FILTER(?val >= {hotspot_threshold})
            BIND(STRBEFORE(STRAFTER(STR(?wkt), "POINT("), " ") AS ?lonStr)
            BIND(STRBEFORE(STRAFTER(STR(?wkt), " "), ")") AS ?latStr)
            BIND(xsd:float(?lonStr) AS ?lon)
            BIND(xsd:float(?latStr) AS ?lat)
        }}
        ORDER BY DESC(?val)
        LIMIT 200
        """
        return self._run(q)


def demo():
    """Run a quick demo of SPARQL queries against the Saudi KG."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from kg.owl.to_rdf import SaudiDMDOConverter

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    schema_dir = os.path.join(project_dir, "schema")
    data_dir = os.path.join(project_dir, "indicators")

    print("Building RDF graph...")
    converter = SaudiDMDOConverter(schema_dir, data_dir)
    converter.build_graph()

    sq = SPARQLQueries(converter)

    print("\n=== Q1: Flash flood indicators (SPARQL) ===")
    for r in sq.hazard_indicators("flash_flood"):
        print(f"  {r.get('indicator','').split('/')[-1].rstrip('>')}: {r.get('description','')}")

    print("\n=== Q2: vpd_kpa co-occurring (SPARQL) ===")
    for r in sq.co_occurring("vpd_kpa"):
        print(f"  {r}")

    print("\n=== Q3: Search 'heat' (SPARQL) ===")
    for r in sq.search_indicators("heat"):
        print(f"  {r.get('indicator','').split('/')[-1].rstrip('>')}")

    print("\n=== Serialize to Turtle ===")
    out_path = os.path.join(project_dir, "saudi_kg.ttl")
    converter.serialize(out_path)
    print(f"Done. Graph: {len(converter.graph)} triples")


if __name__ == "__main__":
    demo()
