"""
Saudi KG → DMDO OWL module.

Provides:
- SaudiDMDOConverter: operators.json + rules.json → RDF triples
- SPARQLQueries: SPARQL equivalents of the networkx query layer
"""

from .to_rdf import SaudiDMDOConverter
from .sparql_queries import SPARQLQueries
