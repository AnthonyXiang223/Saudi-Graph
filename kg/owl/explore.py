"""
Interactive SPARQL explorer for the Saudi DMDO Knowledge Graph.
Prints the graph structure organized by entity type.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.stdout.reconfigure(encoding='utf-8')

from kg.owl import SaudiDMDOConverter, SPARQLQueries

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCHEMA = os.path.join(PROJECT, "schema")
DATA = os.path.join(PROJECT, "indicators")


def explore():
    print("Building DMDO-aligned RDF graph...")
    converter = SaudiDMDOConverter(SCHEMA, DATA)
    converter.build_graph()
    sq = SPARQLQueries(converter)
    g = converter.graph

    # ── 1. Overview ──
    print("=" * 65)
    print("DMDO 沙特极端事件知识图谱 — 总览")
    print("=" * 65)
    print(f"三元组总数: {len(g):,}")

    # Count by type
    type_counts = {}
    for s, p, o in g.triples((None, None, None)):
        if str(p) == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
            t = str(o).split("#")[-1]
            type_counts[t] = type_counts.get(t, 0) + 1

    print("\n实体类型分布:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")

    # ── 2. HazardTypes ──
    print("\n" + "-" * 65)
    print("灾害类型 (hip:SpecificHazard / deo:HazardType)")
    print("-" * 65)
    q = """
    PREFIX hip: <http://purl.org/disaster/hip#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?hazard ?label WHERE {
        ?hazard a hip:SpecificHazard ; rdfs:label ?label .
    }
    """
    for r in g.query(q):
        name = str(r['hazard']).split('/')[-1]
        print(f"  {name} → {r['label']}")

    # ── 3. DataSources ──
    print("\n" + "-" * 65)
    print("数据源 (prov:Entity)")
    print("-" * 65)
    q = """
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?src ?comment WHERE {
        ?src a prov:Entity ; rdfs:comment ?comment .
    }
    """
    for r in g.query(q):
        name = str(r['src']).split('/')[-1]
        print(f"  {name} → {r['comment']}")

    # ── 4. Regions ──
    print("\n" + "-" * 65)
    print("地理分区 (geo:Feature)")
    print("-" * 65)
    q = """
    PREFIX geo: <http://www.opengis.net/ont/geosparql#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?region WHERE {
        ?region a geo:Feature ; rdfs:label ?label .
    }
    """
    for r in g.query(q):
        name = str(r['region']).split('/')[-1]
        print(f"  {name}")

    # ── 5. Detection Rules (deo:Hazard) ──
    print("\n" + "-" * 65)
    print("检测规则 (deo:Hazard)")
    print("-" * 65)
    q = """
    PREFIX deo: <http://purl.org/disaster/deo#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX saudi: <https://mazu.cma/saudi#>
    SELECT ?hazard ?htype ?condCount WHERE {
        ?hazard a deo:Hazard ; rdfs:label ?label ; deo:hazardType ?htype .
        OPTIONAL {
            SELECT ?hazard (COUNT(?cond) as ?condCount) WHERE {
                ?hazard saudi:hasCondition ?cond .
            } GROUP BY ?hazard
        }
    }
    """
    for r in g.query(q):
        name = str(r['hazard']).split('/')[-1]
        ht = str(r['htype']).split('/')[-1]
        cc = r.get('condCount', '?')
        print(f"  {name} → 检测 {ht} ({cc} 个条件)")

    # ── 6. Key indicators by category ──
    print("\n" + "-" * 65)
    print("指标样本 (sosa:ObservableProperty / dpo:Intensity)")
    print("-" * 65)
    categories = {}
    for op in converter._op_by_id.values():
        cat = op.get("category", "other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(op["id"])

    for cat, ids in sorted(categories.items()):
        print(f"\n  [{cat}] ({len(ids)} 个):")
        print(f"    {', '.join(ids[:5])}{'...' if len(ids) > 5 else ''}")

    # ── 7. Derived_from chains ──
    print("\n" + "-" * 65)
    print("算子依赖链 (prov:wasDerivedFrom) — 前 10 条")
    print("-" * 65)
    q = """
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX saudi: <https://mazu.cma/saudi#>
    SELECT ?derived ?source WHERE {
        ?derived prov:wasDerivedFrom ?source .
    }
    LIMIT 10
    """
    for r in g.query(q):
        d = str(r['derived']).split('Indicator/')[-1].rstrip('>')
        s = str(r['source']).split('Indicator/')[-1].rstrip('>')
        print(f"  {d} ← 派生自 ← {s}")

    # ── 8. Try SPARQL queries ──
    print("\n" + "=" * 65)
    print("SPARQL 查询测试")
    print("=" * 65)

    print("\nQ: 山洪依赖哪些指标?")
    for r in sq.hazard_indicators("flash_flood"):
        name = str(r.get('indicator', '')).split('/')[-1].rstrip('>')
        desc = str(r.get('description', ''))
        print(f"  {name}: {desc}")

    print("\nQ: vpd_kpa 和谁联合解释?")
    for r in sq.co_occurring("vpd_kpa"):
        print(f"  {r}")

    print("\nQ: 搜索 'precip'?")
    for r in sq.search_indicators("precip"):
        name = str(r.get('indicator', '')).split('/')[-1].rstrip('>')
        desc = str(r.get('description', ''))
        print(f"  {name}: {desc}")

    print("\n" + "=" * 65)
    print("导出文件: saudi_kg.ttl")
    print("用文本编辑器打开可读 RDF，或用 Protégé / GraphDB 加载")
    print("=" * 65)


if __name__ == "__main__":
    explore()
