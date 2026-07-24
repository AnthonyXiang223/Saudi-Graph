"""
MAZU Agent API Server — Flask REST + SSE streaming wrapper around the ReAct agent.
Launch: python agent_server.py  →  http://127.0.0.1:5001
"""
import json, os, sys, logging, traceback
from flask import Flask, request, jsonify, Response, stream_with_context

# ── Bootstrap: load API key & DeepSeek client ──
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

from openai import OpenAI
from agent_tools import TOOLS, dispatch_tool, smart_truncate
from context_manager import ContextManager, estimate_messages_tokens
from session_manager import SessionManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [api] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("mazu.api")

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    log.error("DEEPSEEK_API_KEY not set")
    sys.exit(1)

client = OpenAI(api_key=API_KEY, base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))

# ── Shared state ──
from agent import SYSTEM_PROMPT  # reuse agent.py's full prompt
sm = SessionManager()
ctx_factory = lambda: ContextManager(max_turns=6)

app = Flask(__name__)


# ═══════════════════════════════════════════════════════
# API endpoints
# ═══════════════════════════════════════════════════════

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Non-streaming chat. Body: {"message": "...", "session_id": "..."}"""
    body = request.get_json(force=True)
    user_input = body.get("message", "").strip()
    session_id = body.get("session_id", "")

    if not user_input:
        return jsonify({"error": "message is required"}), 400

    # Restore or create session
    messages, session_id = _get_or_create_session(session_id)
    messages.append({"role": "user", "content": user_input})

    try:
        result = _run_react_loop(messages)
    except Exception as e:
        log.error("ReAct loop failed: %s", traceback.format_exc()[-300:])
        return jsonify({"error": str(e)}), 500

    # Persist
    sm.save_messages(session_id, messages, [])
    return jsonify({"session_id": session_id, "reply": result})


@app.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    """SSE streaming chat. Body: {"message": "...", "session_id": "..."}"""
    body = request.get_json(force=True)
    user_input = body.get("message", "").strip()
    session_id = body.get("session_id", "")

    if not user_input:
        return jsonify({"error": "message is required"}), 400

    messages, session_id = _get_or_create_session(session_id)
    messages.append({"role": "user", "content": user_input})

    def generate():
        try:
            ctx = ctx_factory()
            messages_local = ctx.trim(messages)

            for turn in range(5):
                log.info("[stream Turn %d] %d messages, ~%d tokens",
                         turn + 1, len(messages_local), estimate_messages_tokens(messages_local))

                response = client.chat.completions.create(
                    model="deepseek-chat", messages=messages_local,
                    tools=TOOLS, tool_choice="auto", stream=True,
                )

                # Collect stream
                content_buf = ""
                tool_calls_buf: dict[int, dict] = {}

                for chunk in response:
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
                    tool_calls = [tool_calls_buf[i] for i in sorted(tool_calls_buf)]
                    yield f"data: {json.dumps({'type': 'tool_calls', 'calls': [{'name': tc['function']['name'], 'args': tc['function']['arguments'][:200]} for tc in tool_calls]})}\n\n"

                    # Append assistant + tool results to messages
                    tc_objs = []
                    for tc in tool_calls:
                        tc_objs.append({
                            "id": tc["id"], "type": tc["type"],
                            "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]},
                        })
                    messages_local.append({"role": "assistant", "content": content_buf or None, "tool_calls": tc_objs})

                    for tc in tool_calls:
                        name = tc["function"]["name"]
                        args = json.loads(tc["function"]["arguments"])
                        result = dispatch_tool(name, args)
                        result = smart_truncate(result, name)
                        messages_local.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

                    messages = messages_local  # sync back
                    continue

                # Final text answer
                messages.append({"role": "assistant", "content": content_buf})
                sm.save_messages(session_id, messages, [])
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            # Exceeded 5 turns
            summary = client.chat.completions.create(
                model="deepseek-chat", messages=messages_local,
            ).choices[0].message.content
            messages.append({"role": "assistant", "content": summary})
            sm.save_messages(session_id, messages, [])
            yield f"data: {json.dumps({'type': 'text', 'content': summary})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            log.error("Stream error: %s", traceback.format_exc()[-300:])
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/api/session/new", methods=["POST"])
def api_session_new():
    sid = sm.create_session()
    return jsonify({"session_id": sid})


@app.route("/api/session/list", methods=["GET"])
def api_session_list():
    return jsonify(sm.list_sessions())


@app.route("/api/session/<sid>", methods=["GET"])
def api_session_get(sid):
    data = sm.get_session(sid)
    if data is None:
        return jsonify({"error": "session not found"}), 404
    return jsonify({"session_id": sid, "messages": data["messages"]})


@app.route("/api/session/<sid>", methods=["DELETE"])
def api_session_delete(sid):
    sm.delete_session(sid)
    return jsonify({"ok": True})


# ═══════════════════════════════════════════════════════
# Internals
# ═══════════════════════════════════════════════════════

def _get_or_create_session(session_id: str):
    """Restore existing session or create new. Returns (messages, session_id)."""
    if session_id:
        data = sm.get_session(session_id)
        if data:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + data["messages"]
            return msgs, session_id
    sid = sm.create_session()
    return [{"role": "system", "content": SYSTEM_PROMPT}], sid


def _run_react_loop(messages: list) -> str:
    """Non-streaming ReAct loop. Returns final assistant text."""
    ctx = ctx_factory()
    messages = ctx.trim(messages)

    for turn in range(5):
        log.info("[Turn %d] %d messages, ~%d tokens",
                 turn + 1, len(messages), estimate_messages_tokens(messages))

        response = client.chat.completions.create(
            model="deepseek-chat", messages=messages,
            tools=TOOLS, tool_choice="auto",
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            log.info("[Turn %d] → %d tool call(s)", turn + 1, len(msg.tool_calls))
            messages.append(msg)

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                result = dispatch_tool(name, args)
                result = smart_truncate(result, name)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                log.info("[Turn %d] %s → %d chars", turn + 1, name, len(result))
        else:
            content = msg.content or ""
            messages.append({"role": "assistant", "content": content})
            return content

    # Force summary
    messages.append({"role": "user", "content": "Based on the tool results above, give me a concise summary."})
    response = client.chat.completions.create(model="deepseek-chat", messages=messages)
    return response.choices[0].message.content or ""


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"MAZU Agent API starting on http://127.0.0.1:5001")
    print(f"  POST /api/chat           — non-streaming chat")
    print(f"  POST /api/chat/stream    — SSE streaming chat")
    print(f"  POST /api/session/new    — create session")
    print(f"  GET  /api/session/list   — list sessions")
    print(f"  GET  /api/session/<id>   — get session history")
    print(f"  DELETE /api/session/<id> — delete session")
    app.run(debug=False, host="0.0.0.0", port=5001)
