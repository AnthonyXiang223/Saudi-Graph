"""
MAZU Unified API Server — FastAPI backend
Serves: REST API + SSE streaming + static frontend
Launch: python server.py  →  http://127.0.0.1:8000
"""
import json, os, sys, logging, traceback
from datetime import datetime

# ── Bootstrap: load .env ──
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    print("[ERROR] DEEPSEEK_API_KEY not set in .env")
    sys.exit(1)

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from agent_tools import TOOLS, dispatch_tool, smart_truncate
from context_manager import ContextManager
from session_manager import SessionManager
from agent import SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [api] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("mazu.server")

# ── Clients ──
llm = OpenAI(api_key=API_KEY, base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
sm = SessionManager()

# ── KG init ──
kg_loaded = False
sq = None
try:
    from kg.owl import SaudiDMDOConverter, SPARQLQueries
    SCHEMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema")
    INDICATORS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indicators")
    converter = SaudiDMDOConverter(SCHEMA_DIR, INDICATORS_DIR)
    converter.build_graph()
    sq = SPARQLQueries(converter)
    kg_loaded = True
    log.info("KG initialized: %d triples", len(converter.graph))
except Exception as e:
    log.warning("KG not loaded: %s", e)

app = FastAPI(title="MAZU API", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ══════════════════════════════════════════════════════════════
# KG API (from dashboard/server.py)
# ══════════════════════════════════════════════════════════════
@app.get("/api/kg/summary")
def kg_summary():
    if not kg_loaded or sq is None:
        return JSONResponse({"error": "KG not loaded"}, status_code=503)
    g = converter.graph
    nodes, edges, seen = [], [], set()
    PREFIX = "https://mazu.cma/saudi#"
    SKIP = ["Grid/", "Observation", "Region/", "Sensor/", "Platform/", "Procedure/", "NetCDF/"]
    REL_MAP = {
        "http://www.w3.org/ns/prov#wasDerivedFrom": "derived_from",
        "https://mazu.cma/saudi#coOccursWith": "co_occurs_with",
        "http://purl.org/disaster/deo#hasHazardProperty": "contributes_to",
        "http://purl.org/disaster/deo#hazardType": "detects",
        "http://purl.org/disaster/deo#possiblyCauses": "causes",
        "https://mazu.cma/saudi#hasCondition": "has_condition",
    }
    for s, p, o in g.triples((None, None, None)):
        s_str = str(s)
        if PREFIX in s_str and s_str not in seen:
            if any(x in s_str for x in SKIP): continue
            seen.add(s_str)
            node_id = s_str.split("#")[-1]
            ntype = "Indicator"
            for t in ["HazardType", "DataSource", "Region", "Hazard", "Event"]:
                if f"{t}/" in s_str: ntype = t if t != "Hazard" else "Rule"; break
            nodes.append({"id": node_id, "type": ntype, "group": ntype, "label": node_id.rsplit("/", 1)[-1]})
    for s, p, o in g.triples((None, None, None)):
        rel = REL_MAP.get(str(p))
        s_str, o_str = str(s), str(o)
        if any(x in s_str for x in SKIP) or any(x in o_str for x in SKIP): continue
        if rel and PREFIX in s_str and PREFIX in o_str:
            edges.append({"from": s_str.split("#")[-1], "to": o_str.split("#")[-1], "label": rel})
    return {"total_nodes": len(nodes), "total_edges": len(edges), "triples": len(g), "nodes": nodes, "edges": edges}

@app.get("/api/kg/indicator/{indicator_id}")
def kg_indicator(indicator_id: str):
    if not kg_loaded or sq is None: return JSONResponse({"error": "KG not loaded"}, status_code=503)
    return JSONResponse({"detail": sq.indicator_detail(indicator_id), "chain": [r.get("derived", "") for r in sq.indicator_chain(indicator_id)], "co_occurring": sq.co_occurring(indicator_id)})

@app.get("/api/kg/hazard/{hazard_type}")
def kg_hazard(hazard_type: str):
    if not kg_loaded or sq is None: return JSONResponse({"error": "KG not loaded"}, status_code=503)
    results = sq.hazard_indicators(hazard_type)
    return JSONResponse({"hazard_type": hazard_type, "indicators": [{"id": str(r.get("indicator","")).split("/")[-1].rstrip(">"), "description": str(r.get("description",""))} for r in results]})

@app.get("/api/kg/chain/{indicator_id}")
def kg_chain(indicator_id: str):
    if not kg_loaded or sq is None: return JSONResponse({"error": "KG not loaded"}, status_code=503)
    chain = sq.indicator_chain(indicator_id)
    return JSONResponse([{"derived": str(r.get("derived","")).split("/")[-1].rstrip(">"), "source": str(r.get("source","")).split("/")[-1].rstrip(">")} for r in chain])

@app.get("/api/kg/search")
def kg_search(q: str = ""):
    if not kg_loaded or sq is None: return JSONResponse({"error": "KG not loaded"}, status_code=503)
    if not q: return JSONResponse([])
    return JSONResponse([{"id": str(r.get("indicator","")).split("/")[-1].rstrip(">"), "description": str(r.get("description",""))} for r in sq.search_indicators(q)])


# ══════════════════════════════════════════════════════════════
# Agent Chat API
# ══════════════════════════════════════════════════════════════
@app.post("/api/chat")
async def api_chat(request: Request):
    body = await request.json()
    user_input = body.get("message", "").strip()
    session_id = body.get("session_id", "")
    if not user_input: raise HTTPException(400, "message required")

    messages, session_id = _get_or_create_session(session_id)
    messages.append({"role": "user", "content": user_input})

    # Language enforcement: inject a strong system message to force reply language
    lang = _detect_lang(user_input)
    if lang != 'zh':
        messages.insert(-1, {"role": "system", "content": f"USER'S LANGUAGE IS {lang.upper()}. You MUST reply in {lang}. DO NOT use Chinese. The user wrote in {lang}, reply in {lang}."})

    ctx = ContextManager(max_turns=6)
    messages = ctx.trim(messages)

    for turn in range(5):
        log.info("[Turn %d] %d msgs, ~%d tokens", turn+1, len(messages), 0)
        resp = llm.chat.completions.create(model="deepseek-chat", messages=messages, tools=TOOLS, tool_choice="auto")
        msg = resp.choices[0].message

        if msg.tool_calls:
            log.info("[Turn %d] → %d tool call(s)", turn+1, len(msg.tool_calls))
            messages.append(msg)
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                result = smart_truncate(dispatch_tool(name, args), name)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            content = msg.content or ""
            messages.append({"role": "assistant", "content": content})
            sm.save_messages(session_id, messages, [])
            return {"session_id": session_id, "reply": content}

    messages.append({"role": "user", "content": "Based on the tool results above, give me a concise summary in English."})
    summary = llm.chat.completions.create(model="deepseek-chat", messages=messages).choices[0].message.content or ""
    messages.append({"role": "assistant", "content": summary})
    sm.save_messages(session_id, messages, [])
    return {"session_id": session_id, "reply": summary}


@app.post("/api/chat/stream")
async def api_chat_stream(request: Request):
    body = await request.json()
    user_input = body.get("message", "").strip()
    session_id = body.get("session_id", "")
    if not user_input: raise HTTPException(400, "message required")

    messages, session_id = _get_or_create_session(session_id)
    messages.append({"role": "user", "content": user_input})

    # Language enforcement
    lang = _detect_lang(user_input)
    if lang != 'zh':
        messages.insert(-1, {"role": "system", "content": f"USER'S LANGUAGE IS {lang.upper()}. You MUST reply in {lang}. DO NOT use Chinese. The user wrote in {lang}, reply in {lang}."})

    ctx = ContextManager(max_turns=6)
    messages = ctx.trim(messages)

    async def generate():
        try:
            for turn in range(5):
                resp = llm.chat.completions.create(model="deepseek-chat", messages=messages, tools=TOOLS, tool_choice="auto", stream=True)

                content_buf = ""
                tool_calls_buf: dict[int, dict] = {}
                for chunk in resp:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content_buf += delta.content
                        yield f"data: {json.dumps({'type': 'text', 'content': delta.content})}\n\n"
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_calls_buf:
                                tool_calls_buf[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                            e = tool_calls_buf[idx]
                            if tc_delta.id: e["id"] += tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name: e["function"]["name"] += tc_delta.function.name
                                if tc_delta.function.arguments: e["function"]["arguments"] += tc_delta.function.arguments

                if tool_calls_buf:
                    tcs = [tool_calls_buf[i] for i in sorted(tool_calls_buf)]
                    yield f"data: {json.dumps({'type': 'tool_calls', 'calls': [{'name': tc['function']['name'], 'args': tc['function']['arguments'][:200]} for tc in tcs]})}\n\n"
                    tc_objs = [{"id": tc["id"], "type": tc["type"], "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}} for tc in tcs]
                    messages.append({"role": "assistant", "content": content_buf or None, "tool_calls": tc_objs})
                    for tc in tcs:
                        result = smart_truncate(dispatch_tool(tc["function"]["name"], json.loads(tc["function"]["arguments"])), tc["function"]["name"])
                        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                    continue

                messages.append({"role": "assistant", "content": content_buf})
                sm.save_messages(session_id, messages, [])
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            summary = llm.chat.completions.create(model="deepseek-chat", messages=messages).choices[0].message.content or ""
            messages.append({"role": "assistant", "content": summary})
            sm.save_messages(session_id, messages, [])
            yield f"data: {json.dumps({'type': 'text', 'content': summary})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            log.error("Stream error: %s", traceback.format_exc()[-300:])
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ══════════════════════════════════════════════════════════════
# Session API
# ══════════════════════════════════════════════════════════════
@app.post("/api/session/new")
def session_new(): return {"session_id": sm.create_session()}

@app.get("/api/session/list")
def session_list(): return sm.list_sessions()

@app.get("/api/session/{sid}")
def session_get(sid: str):
    data = sm.get_session(sid)
    if data is None: raise HTTPException(404, "session not found")
    return {"session_id": sid, "messages": data["messages"]}

@app.delete("/api/session/{sid}")
def session_delete(sid: str):
    sm.delete_session(sid)
    return {"ok": True}


# ══════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════
def _detect_lang(text: str) -> str:
    """Detect language of user input. Returns 'en', 'zh', or 'ar'."""
    arabic = sum(1 for c in text if '؀' <= c <= 'ۿ')
    chinese = sum(1 for c in text if '一' <= c <= '鿿')
    if arabic > chinese and arabic > 3: return 'ar'
    if chinese > arabic and chinese > 3: return 'zh'
    return 'en'

def _get_or_create_session(session_id: str):
    if session_id:
        data = sm.get_session(session_id)
        if data:
            return [{"role": "system", "content": SYSTEM_PROMPT}] + data["messages"], session_id
    sid = sm.create_session()
    return [{"role": "system", "content": SYSTEM_PROMPT}], sid


# ══════════════════════════════════════════════════════════════
# Static files (must be last — mounts after API routes)
# ══════════════════════════════════════════════════════════════
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(os.path.join(STATIC_DIR, "css"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "js"), exist_ok=True)

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    print(f"\n  MAZU API → http://127.0.0.1:8000")
    print(f"  Frontend → http://127.0.0.1:8000/\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
