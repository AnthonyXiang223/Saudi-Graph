"""
Convert Saudi operators.json + ontology.json + detected Events into RDF triples
aligned with KWG's DMDO (Disaster Management Domain Ontology) and SOSA/SSN.

Output: a single rdflib.Graph containing all triples, serializable to Turtle/OWL.
"""

import json
import os
from datetime import datetime
from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, XSD, OWL, TIME, PROV, SOSA, GEO
from typing import Optional

# ── Namespace prefixes ──
SAUDI = Namespace("https://mazu.cma/saudi#")
DEO = Namespace("http://purl.org/disaster/deo#")
DPO = Namespace("http://purl.org/disaster/dpo#")
HIP = Namespace("http://purl.org/disaster/hip#")
QUDT = Namespace("http://qudt.org/schema/qudt#")
GEO_F = Namespace("http://www.opengis.net/ont/geosparql#")
SSN = Namespace("http://www.w3.org/ns/ssn/")
SYS = Namespace("http://www.ontology-of-units-of-measure.org/resource/om-2/system/")
SF = Namespace("http://www.opengis.net/ont/sf#")

NAMESPACES = {
    "saudi": SAUDI, "deo": DEO, "dpo": DPO, "hip": HIP,
    "sosa": SOSA, "ssn": SSN, "geo": GEO, "geof": GEO_F,
    "sf": SF, "time": TIME, "prov": PROV, "qudt": QUDT,
    "owl": OWL, "rdfs": RDFS,
}


class SaudiDMDOConverter:
    """
    Convert the Saudi extreme event KG schema + data into DMDO-aligned RDF.

    Usage:
        converter = SaudiDMDOConverter("schema/", "indicators/")
        converter.build_graph()
        converter.graph.serialize("saudi_kg.ttl", format="turtle")
    """

    def __init__(self, schema_dir: str, data_dir: str):
        self.schema_dir = schema_dir
        self.data_dir = data_dir
        self.graph = Graph()

        # Bind namespaces
        for prefix, ns in NAMESPACES.items():
            self.graph.bind(prefix, ns)

        # Load DMDO OWL files for reference
        self._load_dmdo()

        # Load our JSON schemas
        self._load_schemas()

    def _load_dmdo(self):
        """Load DMDO ontology files as reference triples."""
        dmdo_dir = os.path.join(os.path.dirname(__file__), "dmdo", "modules")
        event_ttl = os.path.join(dmdo_dir, "disaster-event-module", "disaster-event-module-generalized.ttl")
        props_ttl = os.path.join(dmdo_dir, "disaster-properties-module", "disaster-properties-ontology.ttl")
        hip_ttl = os.path.join(os.path.dirname(dmdo_dir), "undrr-isc-hazard-classification.ttl")

        for ttl_file in [event_ttl, props_ttl, hip_ttl]:
            if os.path.exists(ttl_file):
                try:
                    self.graph.parse(ttl_file, format="turtle")
                except Exception:
                    pass  # Skip if parsing fails for some sub-module

    def _load_schemas(self):
        """Load our operator and ontology definitions."""
        with open(os.path.join(self.schema_dir, "operators.json"), "r", encoding="utf-8") as f:
            self.operators = json.load(f)["operators"]
        with open(os.path.join(self.schema_dir, "ontology.json"), "r", encoding="utf-8") as f:
            self.ontology = json.load(f)
        with open(os.path.join(self.schema_dir, "rules.json"), "r", encoding="utf-8") as f:
            self.rules = json.load(f)["rules"]

        self._op_by_id = {op["id"]: op for op in self.operators}

    # ═══════════════════════════════════════════════════════════
    # Build graph
    # ═══════════════════════════════════════════════════════════

    def build_graph(self):
        """Populate the RDF graph with all entities."""
        self._add_data_sources()
        self._add_hazard_types()
        self._add_regions()
        self._add_indicators_as_observable_properties()
        self._add_rules_as_hazards()
        self._add_operator_chains()
        print(f"Graph built: {len(self.graph)} triples")
        return self.graph

    # ── DataSource → prov:Entity ──
    def _add_data_sources(self):
        """Create DataSource entities + SOSA/SSN Sensor/Platform/Procedure modeling."""
        sources = {
            "DS1": ("ERA5 Monthly Reanalysis", "0.25 deg", "1940-present",
                    "ECMWF", "Atmospheric model + observational reanalysis", "Monthly"),
            "DS2": ("ERA5 Daily Reanalysis", "0.25 deg", "1940-present",
                    "ECMWF", "Atmospheric model + observational reanalysis", "Daily"),
            "DS4": ("ERA5 Extremes", "0.25 deg", "1940-present",
                    "ECMWF", "ERA5 aggregated daily max/min fields", "Daily"),
            "DS8": ("GHCN-Daily Station Climatology", "station", "1991-2020",
                    "NOAA NCEI", "Global Historical Climatology Network station normals", "Daily (climatology)"),
            "DS10": ("GPM IMERG V07", "0.1 deg", "2000-present",
                    "NASA/JAXA", "GPM satellite constellation — microwave + IR precipitation retrieval", "30-min"),
            "SST": ("OSTIA SST", "0.05 deg", "2007-present",
                    "UK Met Office", "Operational Sea Surface Temperature and Ice Analysis — satellite + in-situ", "Daily"),
        }
        for src_id, (name, res, coverage, platform, proc_desc, freq) in sources.items():
            # ── DataSource entity (prov:Entity) ──
            ds_uri = SAUDI[f"DataSource/{src_id}"]
            self.graph.add((ds_uri, RDF.type, PROV.Entity))
            self.graph.add((ds_uri, RDFS.label, Literal(src_id)))
            self.graph.add((ds_uri, RDFS.comment, Literal(f"{name}, {res}, {coverage}")))

            # ── SOSA/SSN Sensor ──
            sensor_uri = SAUDI[f"Sensor/{src_id}"]
            self.graph.add((sensor_uri, RDF.type, SOSA.Sensor))
            self.graph.add((sensor_uri, RDF.type, SSN.Sensor))
            self.graph.add((sensor_uri, RDFS.label, Literal(f"{name}")))
            self.graph.add((sensor_uri, SSN.frequency, Literal(freq)))
            self.graph.add((sensor_uri, RDFS.comment, Literal(proc_desc)))

            # ── SSN Platform ──
            platform_uri = SAUDI[f"Platform/{platform.replace(' ', '_')}"]
            self.graph.add((platform_uri, RDF.type, SSN.System))
            self.graph.add((platform_uri, RDFS.label, Literal(platform)))
            self.graph.add((platform_uri, SSN.hasSubSystem, sensor_uri))

            # ── SOSA Procedure (observation method) ──
            proc_uri = SAUDI[f"Procedure/{src_id}"]
            self.graph.add((proc_uri, RDF.type, SOSA.Procedure))
            self.graph.add((proc_uri, RDFS.label, Literal(f"{src_id} observation procedure")))
            self.graph.add((proc_uri, RDFS.comment, Literal(proc_desc)))

            # ── Link DataSource ↔ Sensor/Procedure ──
            self.graph.add((ds_uri, SAUDI.hasSensor, sensor_uri))
            self.graph.add((ds_uri, SAUDI.hasProcedure, proc_uri))

    # ── HazardType → hip:SpecificHazard ──
    def _add_hazard_types(self):
        hazard_map = {
            "flash_flood": ("Flash Flood", "HIP/FL"),
            "extreme_heat": ("Extreme Heat", "HIP/EH"),
            "dust_storm": ("Dust Storm", "HIP/DS"),
            "coastal_humid_heat": ("Coastal Humid Heat", "HIP/CH"),
        }
        for ht_id, (label, hip_code) in hazard_map.items():
            uri = SAUDI[f"HazardType/{ht_id}"]
            self.graph.add((uri, RDF.type, HIP.SpecificHazard))
            self.graph.add((uri, RDF.type, DEO.HazardType))
            self.graph.add((uri, RDFS.label, Literal(label)))

    # ── Region → geo:Feature ──
    def _add_regions(self):
        regions = {
            "saudi_bbox": (16.0, 32.0, 34.0, 56.0),
            "red_sea": (16.0, 30.0, 34.0, 44.0),
            "persian_gulf": (24.0, 30.0, 48.0, 56.0),
            "north_saudi": (26.0, 32.0, 34.0, 56.0),
            "central_saudi": (21.0, 26.0, 34.0, 56.0),
            "south_saudi": (16.0, 21.0, 34.0, 56.0),
        }
        for reg_id, (lat_min, lat_max, lon_min, lon_max) in regions.items():
            uri = SAUDI[f"Region/{reg_id}"]
            self.graph.add((uri, RDF.type, GEO_F.Feature))
            self.graph.add((uri, RDFS.label, Literal(reg_id)))
            # Bounding box as WKT
            wkt = f"POLYGON(({lon_min} {lat_min}, {lon_max} {lat_min}, {lon_max} {lat_max}, {lon_min} {lat_max}, {lon_min} {lat_min}))"
            self.graph.add((uri, GEO_F.asWKT, Literal(wkt, datatype=GEO_F.wktLiteral)))

    # ── Indicator → dpo:ObservableProperty / sosa:ObservableProperty ──
    def _add_indicators_as_observable_properties(self):
        for op in self.operators:
            ind_id = op["id"]
            uri = SAUDI[f"Indicator/{ind_id}"]

            # Each indicator is a SOSA ObservableProperty
            self.graph.add((uri, RDF.type, SOSA.ObservableProperty))
            self.graph.add((uri, RDF.type, DPO.Intensity))  # Indicators measure hazard intensity
            self.graph.add((uri, RDFS.label, Literal(ind_id)))
            self.graph.add((uri, RDFS.comment, Literal(op.get("description", ""))))

            # Unit via QUDT
            unit_map = {
                "mm": QUDT.MilliM,
                "mm/day": QUDT["MilliM-PER-DAY"],
                "degC": QUDT.DegreeCelsius,
                "%": QUDT.Percent,
                "m s-1": QUDT["M-PER-SEC"],
                "W m-2": QUDT["W-PER-M2"],
                "J kg-1": QUDT["J-PER-KiloGM"],
                "kg m-2": QUDT["KiloGM-PER-M2"],
                "kg m-1 s-1": QUDT["KiloGM-PER-M-SEC"],
                "kg m-2 s-1": QUDT["KiloGM-PER-M2-SEC"],
                "Pa": QUDT.Pascal,
                "gpm": QUDT["M"],
                "K": QUDT.Kelvin,
                "N m-2": QUDT["N-PER-M2"],
                "kPa": QUDT.KiloPA,
                "steps": None,
                "1": QUDT.UNITLESS,
                "flag": None,
                "score": None,
                "days": QUDT.Day,
                "km": QUDT.KiloM,
                "s-1": QUDT["PER-SEC"],
                "kg kg-1": QUDT["KiloGM-PER-KiloGM"],
            }
            unit = op.get("output_unit", "")
            if unit in unit_map and unit_map[unit] is not None:
                self.graph.add((uri, QUDT.unit, unit_map[unit]))

            # Data source provenance
            src = op.get("source", "")
            if src:
                for s in src.replace("+", " ").split():
                    s = s.strip()
                    if s:
                        src_uri = SAUDI[f"DataSource/{s}"]
                        self.graph.add((uri, PROV.wasDerivedFrom, src_uri))

            # Availability metadata
            avail = op.get("availability", {})
            if avail:
                eff_days = avail.get("effective_days")
                if eff_days is not None:
                    self.graph.add((uri, SAUDI.effectiveDays, Literal(eff_days, datatype=XSD.integer)))
                    self.graph.add((uri, SAUDI.totalDays, Literal(avail.get("total_days", 365), datatype=XSD.integer)))

            # DAG expression (unique feature — store as custom property)
            expr = op.get("expression", "")
            if expr:
                self.graph.add((uri, SAUDI.expression, Literal(expr)))
            dag = op.get("dag")
            if dag:
                dag_str = json.dumps(dag, ensure_ascii=False)
                self.graph.add((uri, SAUDI.hasDAG, Literal(dag_str)))

    # ── Rules → deo:Hazard ──
    def _add_rules_as_hazards(self):
        for rule in self.rules:
            rule_id = rule["id"]
            hazard_type = rule["hazard_type"]
            hazard_uri = SAUDI[f"Hazard/{rule_id}"]
            ht_uri = SAUDI[f"HazardType/{hazard_type}"]

            # Rule as a deo:Hazard (potential threat)
            self.graph.add((hazard_uri, RDF.type, DEO.Hazard))
            self.graph.add((hazard_uri, DEO.hazardType, ht_uri))
            self.graph.add((hazard_uri, RDFS.label, Literal(rule_id)))

            # Each condition as a hazard property
            for cond in rule.get("conditions", []):
                ind_id = cond.get("indicator", "")
                ind_uri = SAUDI[f"Indicator/{ind_id}"]
                weight = cond.get("weight", 0.0)

                # Link hazard to its observable property
                self.graph.add((hazard_uri, DEO.hasHazardProperty, ind_uri))

                # Weight as a reified condition node
                cond_node = BNode()
                self.graph.add((cond_node, RDF.type, SAUDI.DetectionCondition))
                self.graph.add((cond_node, SAUDI.indicator, ind_uri))
                self.graph.add((cond_node, SAUDI.threshold, Literal(cond.get("value", 0), datatype=XSD.float)))
                self.graph.add((cond_node, SAUDI.comparisonOp, Literal(cond.get("op", cond.get("condition", ">=")))))
                self.graph.add((cond_node, SAUDI.weight, Literal(weight, datatype=XSD.float)))
                self.graph.add((cond_node, SAUDI.isPrimary, Literal(cond.get("primary", False), datatype=XSD.boolean)))
                self.graph.add((hazard_uri, SAUDI.hasCondition, cond_node))

            # Severity levels
            for sev in rule.get("severity", []):
                sev_node = BNode()
                self.graph.add((sev_node, RDF.type, SAUDI.SeverityLevel))
                self.graph.add((sev_node, SAUDI.severityLabel, Literal(sev["label"])))
                self.graph.add((sev_node, SAUDI.lowerBound, Literal(sev["range"][0], datatype=XSD.float)))
                self.graph.add((sev_node, SAUDI.upperBound, Literal(sev["range"][1], datatype=XSD.float)))
                self.graph.add((hazard_uri, SAUDI.hasSeverityLevel, sev_node))

            # Fallback
            fb = rule.get("fallback")
            if fb:
                fb_node = BNode()
                self.graph.add((fb_node, RDF.type, SAUDI.FallbackStrategy))
                self.graph.add((fb_node, SAUDI.missingIndicator, Literal(fb.get("missing_indicator", ""))))
                self.graph.add((fb_node, SAUDI.strategy, Literal(fb.get("strategy", ""))))
                self.graph.add((fb_node, SAUDI.confidencePenalty, Literal(fb.get("confidence_penalty", 0), datatype=XSD.float)))
                self.graph.add((hazard_uri, SAUDI.hasFallback, fb_node))

    # ── Operator chains → derived_from as SOSA observation derivation ──
    def _add_operator_chains(self):
        for op in self.operators:
            out_uri = SAUDI[f"Indicator/{op['id']}"]
            for inp_id in op.get("inputs", []):
                inp_uri = SAUDI[f"Indicator/{inp_id}"]
                # derived_from: output ← input
                self.graph.add((out_uri, PROV.wasDerivedFrom, inp_uri))

            # co_occurs_with
            for co_id in op.get("co_occurs_with", []):
                co_uri = SAUDI[f"Indicator/{co_id}"]
                self.graph.add((out_uri, SAUDI.coOccursWith, co_uri))

    # ═══════════════════════════════════════════════════════════
    # Observation layer — NetCDF values → SOSA Observation instances
    # ═══════════════════════════════════════════════════════════

    def add_observations(self, date_str: str, indicator_ids: list = None,
                         lat_idx: list = None, lon_idx: list = None,
                         threshold_filter: dict = None) -> int:
        """
        Create SOSA Observation triples from NetCDF data.

        Selectively instantiates observations to avoid blowing up the graph
        (35,200 grids × 365 days × 91 indicators = 1.2 billion observations).

        Args:
            date_str: Date string "YYYYMMDD" or "YYYY-MM-DD"
            indicator_ids: List of indicator IDs, or None for all
            lat_idx: List of latitude indices to include, or None for all
            lon_idx: List of longitude indices to include, or None for all
            threshold_filter: Dict {indicator_id: (op, value)} to only create
                              observations that exceed a threshold.
                              e.g. {"tmax_c": (">=", 45), "daily_precip_total": (">=", 10)}

        Returns:
            Number of observation triples created
        """
        import xarray as xr
        import numpy as np
        import os

        date_clean = date_str.replace("-", "")
        nc_path = os.path.join(self.data_dir, f"saudi_indicators_{date_clean}.nc")

        if not os.path.exists(nc_path):
            raise FileNotFoundError(f"NetCDF not found: {nc_path}")

        ds = xr.open_dataset(nc_path)

        # Resolve lat/lon
        if "latitude" in ds.dims:
            lats = ds["latitude"].values
            lons = ds["longitude"].values
        elif "lat" in ds.dims:
            lats = ds["lat"].values
            lons = ds["lon"].values
        else:
            ds.close()
            raise KeyError("Cannot find lat/lon dimensions")

        # Resolve indicator list
        if indicator_ids is None:
            indicator_ids = [op["id"] for op in self._op_by_id.values()]
        available = [i for i in indicator_ids if i in ds.variables]

        # Resolve grid indices
        if lat_idx is None:
            lat_idx = list(range(len(lats)))
        if lon_idx is None:
            lon_idx = list(range(len(lons)))

        # Build threshold filters
        filters = {}
        if threshold_filter:
            for ind_id, (op_str, val) in threshold_filter.items():
                filters[ind_id] = (op_str, val)

        cmp_fns = {
            ">=": lambda a, b: a >= b,
            ">":  lambda a, b: a > b,
            "<=": lambda a, b: a <= b,
            "<":  lambda a, b: a < b,
            "==": lambda a, b: np.isclose(a, b),
        }

        count = 0
        formatted_date = f"{date_clean[:4]}-{date_clean[4:6]}-{date_clean[6:8]}"

        for ind_id in available:
            data = ds[ind_id].values

            # Handle multi-dimensional data
            if data.ndim >= 3:
                data = data.mean(axis=0) if data.shape[0] > 1 else data[0]
                while data.ndim > 2:
                    data = data[0]

            ind_uri = SAUDI[f"Indicator/{ind_id}"]

            has_filter = ind_id in filters
            cmp_fn = None
            threshold_val = None
            if has_filter:
                op_str, threshold_val = filters[ind_id]
                cmp_fn = cmp_fns.get(op_str)

            for i in lat_idx:
                if i >= data.shape[0]:
                    continue
                for j in lon_idx:
                    if j >= data.shape[1]:
                        continue
                    val = data[i, j]
                    if np.isnan(val):
                        continue

                    # Apply threshold filter if present
                    if has_filter and cmp_fn is not None:
                        if not cmp_fn(val, threshold_val):
                            continue

                    # ── SOSA/SSN complete Observation ──
                    obs_node = BNode()
                    lat = float(lats[i])
                    lon_val = float(lons[j])

                    # FeatureOfInterest: GeoSPARQL geo:Feature with sf:Point geometry
                    foi_uri = SAUDI[f"Grid/{lat:.2f}_{lon_val:.2f}"]
                    geom_node = BNode()
                    wkt = f"POINT({lon_val} {lat})"

                    self.graph.add((foi_uri, RDF.type, GEO_F.Feature))
                    self.graph.add((foi_uri, RDF.type, SF.Point))
                    self.graph.add((foi_uri, RDFS.label, Literal(f"Grid({lat:.2f}N,{lon_val:.2f}E)")))
                    self.graph.add((foi_uri, GEO_F.hasGeometry, geom_node))
                    self.graph.add((geom_node, RDF.type, SF.Point))
                    self.graph.add((geom_node, GEO_F.asWKT, Literal(wkt, datatype=GEO_F.wktLiteral)))
                    self.graph.add((geom_node, GEO_F.asGML, Literal(f"<gml:Point srsName='EPSG:4326'><gml:pos>{lat} {lon_val}</gml:pos></gml:Point>")))

                    # Observation core
                    self.graph.add((obs_node, RDF.type, SOSA.Observation))
                    self.graph.add((obs_node, SOSA.observedProperty, ind_uri))
                    self.graph.add((obs_node, SOSA.hasFeatureOfInterest, foi_uri))
                    self.graph.add((obs_node, SOSA.resultTime, Literal(formatted_date, datatype=XSD.date)))
                    self.graph.add((obs_node, SOSA.phenomenonTime, Literal(formatted_date, datatype=XSD.date)))

                    # Structured Result (SOSA Result class)
                    result_node = BNode()
                    self.graph.add((result_node, RDF.type, SOSA.Result))
                    self.graph.add((result_node, QUDT.numericValue, Literal(float(val), datatype=XSD.float)))
                    # Map unit
                    op = self._op_by_id.get(ind_id, {})
                    unit_str = op.get("output_unit", "")
                    unit_map = {"degC": QUDT.DegreeCelsius, "mm": QUDT.MilliM, "m s-1": QUDT["M-PER-SEC"],
                                "W m-2": QUDT["W-PER-M2"], "J kg-1": QUDT["J-PER-KiloGM"], "%": QUDT.Percent,
                                "kg m-2": QUDT["KiloGM-PER-M2"], "Pa s-1": QUDT["PA-PER-SEC"],
                                "kg m-1 s-1": QUDT["KiloGM-PER-M-SEC"], "kPa": QUDT.KiloPA,
                                "gpm": QUDT["M"], "K": QUDT.Kelvin, "days": QUDT.Day, "km": QUDT.KiloM,
                                "s-1": QUDT["PER-SEC"], "steps": None, "score": None, "flag": None,
                                "kg m-2 s-1": QUDT["KiloGM-PER-M2-SEC"]}
                    if unit_str in unit_map and unit_map[unit_str] is not None:
                        self.graph.add((result_node, QUDT.unit, unit_map[unit_str]))
                    self.graph.add((obs_node, SOSA.hasResult, result_node))
                    self.graph.add((obs_node, SOSA.hasSimpleResult, Literal(float(val), datatype=XSD.float)))

                    # Link to Sensor + Procedure (SSN)
                    src = op.get("source", "")
                    if src:
                        for s in src.replace("+", " ").split():
                            s = s.strip()
                            if s:
                                sensor_uri = SAUDI[f"Sensor/{s}"]
                                proc_uri = SAUDI[f"Procedure/{s}"]
                                if (sensor_uri, RDF.type, SOSA.Sensor) in self.graph:
                                    self.graph.add((obs_node, SOSA.madeBySensor, sensor_uri))
                                if (proc_uri, RDF.type, SOSA.Procedure) in self.graph:
                                    self.graph.add((obs_node, SOSA.usedProcedure, proc_uri))

                    count += 1

        ds.close()
        print(f"Created {count} observations for {date_clean} ({len(available)} indicators, "
              f"{len(lat_idx)}×{len(lon_idx)} grid)")
        return count

    # ═══════════════════════════════════════════════════════════
    # Event detection → deo:Disaster instances
    # ═══════════════════════════════════════════════════════════

    def add_event(self, event) -> URIRef:
        """
        Add a detected Event as a deo:Disaster instance.

        Args:
            event: Event dataclass from event_detector.py

        Returns:
            URIRef of the created disaster node
        """
        disaster_uri = SAUDI[f"Event/{event.event_id}"]
        hazard_type_uri = SAUDI[f"HazardType/{event.hazard_type}"]

        self.graph.add((disaster_uri, RDF.type, DEO.Disaster))
        self.graph.add((disaster_uri, DEO.hazardType, hazard_type_uri))
        self.graph.add((disaster_uri, RDFS.label, Literal(event.event_id)))

        # Temporal scope
        time_node = BNode()
        self.graph.add((time_node, RDF.type, TIME.Instant))
        date_str = event.date
        formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}" if len(date_str) == 8 else date_str
        self.graph.add((time_node, TIME.inXSDDate, Literal(formatted, datatype=XSD.date)))
        self.graph.add((disaster_uri, DEO.hasTemporalScope, time_node))

        # Spatial scope (centroid)
        self.graph.add((disaster_uri, SAUDI.centroidLat, Literal(event.centroid_lat, datatype=XSD.float)))
        self.graph.add((disaster_uri, SAUDI.centroidLon, Literal(event.centroid_lon, datatype=XSD.float)))
        self.graph.add((disaster_uri, SAUDI.areaKm2, Literal(event.area_km2, datatype=XSD.float)))

        # Severity
        self.graph.add((disaster_uri, DPO.Severity, Literal(event.severity_score, datatype=XSD.float)))

        # Region
        region_uri = SAUDI[f"Region/{event.region.split(',')[0].strip().replace(' ', '_')}"]
        if (region_uri, RDF.type, GEO_F.Feature) in self.graph:
            self.graph.add((disaster_uri, GEO_F.hasGeometry, region_uri))

        # Causal link from hazard rule to disaster
        rule_uri = SAUDI[f"Hazard/flash_flood_weighted"]  # Link to the relevant rule
        if (rule_uri, RDF.type, DEO.Hazard) in self.graph:
            self.graph.add((rule_uri, DEO.possiblyCauses, disaster_uri))

        # Trigger details as observations
        for detail in event.trigger_details:
            if detail.get("status") == "evaluated":
                obs_node = BNode()
                ind_uri = SAUDI[f"Indicator/{detail['indicator']}"]
                self.graph.add((obs_node, RDF.type, SOSA.Observation))
                self.graph.add((obs_node, SOSA.observedProperty, ind_uri))
                if detail.get("peak_value") is not None:
                    self.graph.add((obs_node, SOSA.hasSimpleResult, Literal(detail["peak_value"], datatype=XSD.float)))
                self.graph.add((disaster_uri, SAUDI.hasTriggerObservation, obs_node))

        return disaster_uri

    # ═══════════════════════════════════════════════════════════
    # SPARQL query helpers
    # ═══════════════════════════════════════════════════════════

    def query(self, sparql: str):
        """Execute a SPARQL query against the graph."""
        return self.graph.query(sparql)

    def get_hazard_properties(self, hazard_type: str):
        """DMDO equivalent of get_hazard_indicators()."""
        q = f"""
        PREFIX deo: <http://purl.org/disaster/deo#>
        PREFIX saudi: <https://mazu.cma/saudi#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?indicator ?label ?description WHERE {{
            ?hazard deo:hazardType saudi:HazardType/{hazard_type} ;
                    deo:hasHazardProperty ?indicator .
            ?indicator rdfs:label ?label ;
                       rdfs:comment ?description .
        }}
        """
        return list(self.graph.query(q))

    def get_indicator_chain(self, indicator_id: str):
        """DMDO equivalent of get_indicator_chain(). Uses PROV derivation."""
        q = f"""
        PREFIX prov: <http://www.w3.org/ns/prov#>
        PREFIX saudi: <https://mazu.cma/saudi#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?source ?target WHERE {{
            ?target prov:wasDerivedFrom+ saudi:Indicator/{indicator_id} .
            ?target prov:wasDerivedFrom ?source .
        }}
        """
        return list(self.graph.query(q))

    def get_co_occurring(self, indicator_id: str):
        """DMDO equivalent of get_co_occurring()."""
        q = f"""
        PREFIX saudi: <https://mazu.cma/saudi#>

        SELECT ?co WHERE {{
            {{ saudi:Indicator/{indicator_id} saudi:coOccursWith ?co . }}
            UNION
            {{ ?co saudi:coOccursWith saudi:Indicator/{indicator_id} . }}
        }}
        """
        return list(self.graph.query(q))

    def get_events_by_date(self, date_str: str):
        """Query all disasters on a given date."""
        formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}" if len(date_str) == 8 else date_str
        q = f"""
        PREFIX deo: <http://purl.org/disaster/deo#>
        PREFIX saudi: <https://mazu.cma/saudi#>
        PREFIX time: <http://www.w3.org/2006/time#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?event ?label ?hazardType ?severity ?lat ?lon ?area WHERE {{
            ?event a deo:Disaster ;
                   rdfs:label ?label ;
                   deo:hazardType ?hazardType ;
                   deo:hasTemporalScope ?t .
            ?t time:inXSDDate "{formatted}"^^xsd:date .
            OPTIONAL {{ ?event deo:severity ?severity . }}
            OPTIONAL {{ ?event saudi:centroidLat ?lat . }}
            OPTIONAL {{ ?event saudi:centroidLon ?lon . }}
            OPTIONAL {{ ?event saudi:areaKm2 ?area . }}
        }}
        """
        return list(self.graph.query(q))

    def serialize(self, path: str, format: str = "turtle"):
        """Write graph to file."""
        self.graph.serialize(destination=path, format=format)
        print(f"Serialized {len(self.graph)} triples to {path}")
