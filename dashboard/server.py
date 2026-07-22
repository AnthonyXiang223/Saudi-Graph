"""
Dashboard server for Saudi Extreme Event Knowledge Graph.
Flask backend — now with SPARQL/DMDO endpoints alongside original API.
"""

import json, sys, os, re, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, render_template, send_from_directory
from rdflib.namespace import SOSA, GEO
from rdflib import Namespace
GEO_F = Namespace("http://www.opengis.net/ont/geosparql#")

app = Flask(__name__)
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DASHBOARD_DIR)
SCHEMA_DIR = os.path.join(PROJECT_DIR, "schema")
DATA_DIR = os.path.join(PROJECT_DIR, "indicators")

# ── Init KWG-based DMDO RDF graph ──
print("Initializing DMDO RDF graph...")
from kg.owl import SaudiDMDOConverter, SPARQLQueries
converter = SaudiDMDOConverter(SCHEMA_DIR, DATA_DIR)
converter.build_graph()
sq = SPARQLQueries(converter)

# Load rules.json for event detection
import json
with open(os.path.join(SCHEMA_DIR, "rules.json"), "r", encoding="utf-8") as f:
    _rules_data = json.load(f)

print(f"Ready. RDF: {len(converter.graph)} triples, Rules: {len(_rules_data['rules'])} loaded")


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
    """DMDO-SPARQL knowledge graph dashboard."""
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
    SKIP = ["Grid/", "Observation", "Region/", "Sensor/", "Platform/", "Procedure/", "NetCDF/"]
    for s, p, o in g.triples((None, None, None)):
        s_str = str(s)
        if PREFIX in s_str and s_str not in seen_nodes:
            # Skip ephemeral Grid, Observation, and orphaned nodes (no tracked edges in REL_MAP)
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

            # Use full path as ID to avoid duplicates (e.g., DataSource/DS1 vs Sensor/DS1)
            clean_id = node_id  # Keep type prefix for uniqueness
            nodes.append({
                "id": clean_id,
                "type": ntype,
                "group": ntype,
                "label": clean_id.split("/")[-1] if len(clean_id.split("/")[-1]) <= 30 else clean_id.split("/")[-1][:28]+"…",
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
    from kg.datalayer import DataLayer
    from kg.event_detector import EventDetector
    body = request.get_json() or {}
    date_str = body.get("date", "2025-08-19")
    hazard_type = body.get("hazard_type", None)
    hazard_types = [hazard_type] if hazard_type else None
    detector = EventDetector(_rules_data["rules"], DataLayer(DATA_DIR))
    events = detector.detect_events(date_str, hazard_types)
    results = []
    for e in events:
        converter.add_event(e)
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
    """Redirect to SPARQL summary (old compatibility endpoint)."""
    return jsonify({"triples": len(converter.graph), "indicators": len(converter._op_by_id), "rules": len(_rules_data["rules"])})


# ═══════════════════════════════════════════
# Chat API — ReAct loop with DeepSeek + tools
# ═══════════════════════════════════════════

# Lazy imports for chat
_chat_imports = None

def _get_chat_imports():
    global _chat_imports
    if _chat_imports is None:
        from openai import OpenAI
        from agent_tools import TOOLS as AGENT_TOOLS, dispatch_tool, smart_truncate
        _chat_imports = (OpenAI, AGENT_TOOLS, dispatch_tool, smart_truncate)
    return _chat_imports


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Proxy chat requests to DeepSeek with ReAct tool-use loop."""
    body = request.get_json() or {}
    messages = body.get("messages", [])
    lang = body.get("lang", "en")

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    # API key
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        # Try reading from .env
        env_path = os.path.join(PROJECT_DIR, ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == "DEEPSEEK_API_KEY":
                            api_key = v.strip()
                            break
    if not api_key:
        return jsonify({"error": "DeepSeek API key not configured"}), 503

    try:
        OpenAI, AGENT_TOOLS, dispatch_tool, smart_truncate = _get_chat_imports()

        client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )


        # Build system prompt (same comprehensive prompt as Streamlit app.py)
        import datetime as _dt
        today = _dt.date.today()

        # Auto-detect latest IFS date
        ifs_init = today
        ifs_dir = os.path.join(PROJECT_DIR, "aifs_forecasts")
        if os.path.isdir(ifs_dir):
            ifs_dates = sorted([d for d in os.listdir(ifs_dir)
                              if os.path.isdir(os.path.join(ifs_dir, d)) and re.match(r'^\d{8}', d)])
            if ifs_dates:
                try:
                    ifs_init = _dt.date.fromisoformat(ifs_dates[-1])
                except Exception:
                    pass
        ifs_offset = max((today - ifs_init).days, 0)

        # Language instruction
        lang_instr_map = {
            "zh-CN": "- **Automatic language detection**: Reply in the SAME language as the user's question.\n- 用户用中文提问 → 你用中文回答。\n- User asks in English → reply in English.\n- إذا سأل المستخدم بالعربية → أجب بالعربية。",
            "ar": "- **Automatic language detection**: Reply in the SAME language as the user's question.\n- إذا سأل المستخدم بالعربية → أجب بالعربية。\n- User asks in English → reply in English.",
            "en": "Reply in English. Be concise and professional.",
        }
        lang_instr = lang_instr_map.get(lang, "Reply in English.")

        system_prompt = f"""你是 MAZU 多灾种早期预警系统的气象分析助手，服务沙特阿拉伯气象预警业务。
You are the meteorological analysis assistant of the MAZU multi-hazard early warning system.

══════════════════════════════════════
语言 / Language / اللغة
══════════════════════════════════════
{lang_instr}

══════════════════════════════════════
时间上下文
══════════════════════════════════════
- 当前日期：{today.isoformat()}。用户说的"今天""明天"以此为准。
- 历史数据：2025 年全年 ERA5 再分析（365 天 NetCDF，35,200 格点，约100 个指标）。
- 预报数据：ECMWF IFS 全球预报(0.25°)，初始化于 {ifs_init.isoformat()}，forecast_day={ifs_offset} = 今天，明天 = {ifs_offset+1}，以此类推，最多覆盖初始化后 7 天。

══════════════════════════════════════
能力边界
══════════════════════════════════════

## 你能回答的问题（调用对应工具即可获取答案）
- 四类灾害（极端高温、沙尘强风、山洪、沿海湿热）的未来 7 天风险检测 → detect_future_events
- 2025 年任意日期的历史极端事件回顾 → detect_extreme_events
- 91 个气象指标的物理定义、公式、推导链、数据来源 → query_indicator_* 系列
- 4 条检测规则的完整条件、权重、角色 → query_rule_detail
- 指定坐标周边半径内的指标观测 → query_observations_nearby

## 你无法回答的问题（直接说明能力不足，不要绕弯）
- 任何涉及"实况""实测""实时""当前此刻"的问题 — 你只有 2025 年再分析和 IFS 预报，没有实时观测
- 卫星反演、雷达回波、土壤墒情、大气能见度 — 系统没有接入这些数据
- 2 小时短临预报、30 天长期预测 — 超出 IFS 预报范围
- 概率百分比、精确起止时间、能见度米数 — 系统只输出风险评分和严重度等级

══════════════════════════════════════
研判规则（踩坑纠错 + 沙特本地校准）
══════════════════════════════════════

**规则 1：高温不只看温度数值**
- 沙特 7 月沙漠格点 44-48℃ 是常态，不要一看到就报"极端"。
- 异常判断看三件：是否超过气候态 +5℃？露点差是否 >20℃？连续几天？
- ERA5 格点值在沙漠区域系统性偏低 2-4℃，输出时提及这个偏差，但不要说"已订正"。

**规则 2：沙尘热点不重合 ≠ 目标区域安全**
- 检测到的沙尘热点在阿曼湾时，不要直接判"港区不受影响"。
- 同一气团 + 同一干燥背景下，沙尘可沿 Shamal 方向传播。

**规则 3：KG 物理一致性得分低 ≠ 模型不可靠**
- 沙特干季（5-10 月）可降水量与 IVT 辐合弱相关或负相关，这是气候常态。

**规则 4：山洪未触发 ≠ 无对流**
- 山洪检测覆盖有限，未触发不说明安全。
- 必须单独检查日降水量 ≥10mm 的格点数量和位置。

══════════════════════════════════════
输出规范（强制遵守）
══════════════════════════════════════

**通用原则（强制）：**
- 结论前置——第一句必须是问题直接回答，不许以"根据系统数据""通过调用工具"铺垫。
- 技术细节放末尾——不打断阅读流。
- 表格只在多指标/多地对比时使用，单个结果不堆表格。

**禁止行为**：
- 禁止编造方法名称、算法流程、数据处理步骤
- 禁止说"仅供参考""建议进一步核实"等搪塞话
- 禁止把"你不具备的能力"说成"建议查询 XX 数据"
- 禁止前后矛盾：如果工具 A 和工具 B 结果冲突，明确指出冲突

## 沙特地理
- 红海沿岸：16-30°N, 34-44°E，吉达、延布。对流由阿西尔山脉地形触发。
- 波斯湾沿岸：24-30°N, 48-56°E，达曼、朱拜勒。受沙马风（Shamal）控制。
- 利雅得：24.7°N, 46.7°E，中部沙漠。
- 鲁布哈利沙漠（Empty Quarter）：17-23°N, 45-56°E，世界最大连续沙体。"""

        full_messages = [{"role": "system", "content": system_prompt}] + messages

        tool_calls_log = []
        final_content = ""

        for turn in range(5):
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=full_messages,
                tools=AGENT_TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                full_messages.append(msg)
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    result = dispatch_tool(name, args)
                    result = smart_truncate(result, name, max_chars=2000)
                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                    # For detect tools, keep full result; otherwise trim
                    if name in ("detect_future_events", "detect_extreme_events"):
                        result_for_ui = result
                    else:
                        result_for_ui = result[:2000]
                    tool_calls_log.append({
                        "name": name,
                        "args": args,
                        "result": result_for_ui,
                    })
            else:
                final_content = msg.content or ""
                break
        else:
            full_messages.append({
                "role": "user",
                "content": "Summarize briefly based on the tool results above.",
            })
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=full_messages,
            )
            final_content = response.choices[0].message.content or ""

        return jsonify({
            "content": final_content,
            "tool_calls": tool_calls_log if tool_calls_log else None,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════
# Service status
# ═══════════════════════════════════════════

@app.route("/api/service/ifs-status")
def api_ifs_status():
    """Check if IFS forecast data is available."""
    ifs_dir = os.path.join(PROJECT_DIR, "aifs_forecasts")
    available = False
    if os.path.isdir(ifs_dir):
        import re
        dirs = [d for d in os.listdir(ifs_dir)
                if os.path.isdir(os.path.join(ifs_dir, d)) and re.match(r'^\d{8}', d)]
        available = len(dirs) > 0
    return jsonify({"available": available})


if __name__ == "__main__":
    print("\nDashboard: http://127.0.0.1:5000")
    print("Unified UI: http://127.0.0.1:5000/unified")
    app.run(debug=True, host="0.0.0.0", port=5000)
