"""
Context Window Manager + Streaming utilities for MAZU Agent.

Usage:
    from context_manager import ContextManager, stream_chat_completion

    ctx = ContextManager(max_turns=6)
    messages = ctx.trim(messages)  # before each API call
"""

import json
import logging

logger = logging.getLogger("mazu.context")

# ── Token estimation (rough: 1 Chinese char ≈ 1.5 tokens, 1 English word ≈ 1.3 tokens) ──

def estimate_tokens(text: str) -> int:
    """Rough token count for mixed Chinese/English text."""
    if not text:
        return 0
    chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 1.5 + other_chars * 0.3)


def estimate_messages_tokens(messages: list) -> int:
    """Estimate total tokens in a message list."""
    total = 0
    for msg in messages:
        content = msg.get("content", "") or ""
        total += estimate_tokens(content)
        # Tool calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                total += estimate_tokens(tc.function.arguments or "")
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                total += estimate_tokens(tc.get("function", {}).get("arguments", ""))
    return total


# ══════════════════════════════════════════════════════════════
# Message normalisation + integrity helpers
# ══════════════════════════════════════════════════════════════

def _normalise_msg(msg) -> dict:
    """Convert a message (dict or Pydantic SDK object) to a plain dict."""
    if isinstance(msg, dict):
        return msg
    result = {
        "role": getattr(msg, "role", ""),
        "content": getattr(msg, "content", None),
    }
    tcs = getattr(msg, "tool_calls", None)
    if tcs:
        result["tool_calls"] = [
            {
                "id": getattr(tc, "id", ""),
                "type": getattr(tc, "type", "function"),
                "function": {
                    "name": getattr(getattr(tc, "function", None), "name", ""),
                    "arguments": getattr(getattr(tc, "function", None), "arguments", ""),
                },
            }
            for tc in tcs
        ]
    tc_id = getattr(msg, "tool_call_id", None)
    if tc_id:
        result["tool_call_id"] = tc_id
    return result


def _repair_orphaned_tools(messages: list) -> list:
    """Remove orphaned ``role: tool`` messages that lack a preceding
    ``role: assistant`` with ``tool_calls``.

    Orphans can arise when a turn-splitting bug (now fixed) incorrectly
    separated tool_calls from their tool_results, or when a session is
    restored from a corrupted state.  The safest repair is to drop the
    orphaned tool message entirely.
    """
    cleaned = []
    pending_tool_ids = set()

    for m_raw in messages:
        m = _normalise_msg(m_raw)  # ensure plain dict
        role = m.get("role", "")
        has_tc = bool(m.get("tool_calls"))

        if role == "assistant":
            if has_tc:
                pending_tool_ids = {
                    tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                    for tc in (m.get("tool_calls") or [])
                }
            else:
                pending_tool_ids = set()
            cleaned.append(m)

        elif role == "tool":
            tid = m.get("tool_call_id", "")
            if tid in pending_tool_ids:
                cleaned.append(m)
                pending_tool_ids.discard(tid)  # each id used once
            else:
                # Orphan: drop it
                logger = logging.getLogger("mazu.context")
                logger.warning(
                    "Dropping orphaned tool message (no matching tool_calls): %s",
                    tid or "?",
                )

        else:
            # user / system: reset tracking
            pending_tool_ids = set()
            cleaned.append(m)

    return cleaned


# ══════════════════════════════════════════════════════════════
# Context Manager
# ══════════════════════════════════════════════════════════════

class ContextManager:
    """Sliding-window context manager with LLM summarization.

    Strategy:
      1. System prompt always stays as messages[0] (never trimmed).
      2. Keep the last *max_turns* complete user→assistant exchanges verbatim.
      3. Older turns are LLM-summarised into a compact system-style message
         inserted right after the main system prompt.
      4. Individual tool results are capped at *max_tool_result_chars*.
      5. If total estimated tokens exceed *max_tokens*, aggressive trimming
         is applied even within the recent window.

    A "turn" = one user message + all assistant/tool messages that follow
    until (and including) the assistant's final text response.
    """

    def __init__(
        self,
        max_turns: int = 6,
        max_tool_result_chars: int = 3000,
        max_tokens: int = 32000,
        summarise_callback=None,
    ):
        """
        Args:
            max_turns: number of recent turns to keep verbatim.
            max_tool_result_chars: cap per-tool-result content length.
            max_tokens: soft cap on total estimated tokens.
            summarise_callback: async fn(messages) -> str for LLM summarisation.
                                If None, a simple extraction is used (no API call).
        """
        self.max_turns = max_turns
        self.max_tool_result_chars = max_tool_result_chars
        self.max_tokens = max_tokens
        self._summarise_fn = summarise_callback

    def trim(self, messages: list) -> list:
        """Return a trimmed copy of *messages*.

        Never mutates the original list.  Always returns a new list so
        the caller can decide whether to replace its reference.
        """
        if len(messages) <= 1:
            return list(messages)

        # ── 0. Normalise: convert any Pydantic SDK objects to plain dicts ──
        messages = [_normalise_msg(m) for m in messages]

        # ── 1. Cap tool results ──
        msgs = []
        for m in messages:
            if m.get("role") == "tool":
                content = m.get("content", "")
                if isinstance(content, str) and len(content) > self.max_tool_result_chars:
                    m = dict(m)
                    m["content"] = content[:self.max_tool_result_chars] + "\n...(truncated)"
            msgs.append(m)

        # ── 2. Split into turns ──
        turns = self._split_turns(msgs)
        if len(turns) <= self.max_turns:
            return _repair_orphaned_tools(msgs)  # early return — still validate

        # ── 3. Summarise old turns, keep recent ones ──
        old_turns = turns[:-self.max_turns]
        recent_turns = turns[-self.max_turns:]

        summary = self._build_summary(old_turns)

        # Rebuild: system + summary + recent
        system = msgs[0]
        result = [system]
        if summary:
            result.append({
                "role": "system",
                "content": (
                    "[历史对话摘要 — 以下是对早前对话的自动压缩，"
                    "保留关键信息供后续回答参考]\n\n" + summary
                ),
            })
        for turn in recent_turns:
            result.extend(turn)

        # ── 4. Hard token cap (aggressive) ──
        est = estimate_messages_tokens(result)
        if est > self.max_tokens:
            # Drop the oldest full turn until under cap
            while len(recent_turns) > 2 and estimate_messages_tokens(result) > self.max_tokens:
                recent_turns = recent_turns[1:]
                result = [system]
                if summary:
                    result.append({
                        "role": "system",
                        "content": "[历史对话摘要]\n\n" + summary,
                    })
                for turn in recent_turns:
                    result.extend(turn)

        # ── 5. Safety: repair orphaned tool messages ──
        result = _repair_orphaned_tools(result)

        return result

    # ── internal helpers ──

    @staticmethod
    def _split_turns(messages: list) -> list:
        """Split message list into conversational turns.

        A turn ends when the assistant sends a *final* text response —
        i.e. a message that has text content but does NOT have pending
        tool_calls.  Assistant messages with tool_calls (even if they
        also contain a brief text preamble) are NOT turn boundaries,
        because their tool_result children must stay in the same turn.
        """
        body = messages[1:]  # exclude system prompt
        turns = []
        current = []
        for msg in body:
            current.append(msg)
            # Normalise access: dict vs. SDK-object (both appear in session state)
            role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            has_tool_calls = bool(
                msg.get("tool_calls") if isinstance(msg, dict)
                else getattr(msg, "tool_calls", None)
            )
            # Turn boundary: assistant text response WITHOUT pending tool calls
            if role == "assistant" and content and not has_tool_calls:
                turns.append(current)
                current = []
        if current:
            # trailing messages (shouldn't happen, but be safe)
            if turns:
                turns[-1].extend(current)
            else:
                turns.append(current)
        return turns

    @staticmethod
    def _build_summary(turns: list) -> str:
        """Extract key information from old turns without an extra API call.

        Returns a compact bullet-point summary in Chinese.
        """
        lines = []
        for ti, turn in enumerate(turns, 1):
            user_q = ""
            tool_names = []
            final_answer = ""

            for msg in turn:
                role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
                content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)

                if role == "user":
                    user_q = (content or "")[:200]
                elif role == "assistant":
                    # Check for tool calls (dict and SDK-object paths)
                    tcs = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
                    if tcs:
                        for tc in tcs:
                            if isinstance(tc, dict):
                                name = tc.get("function", {}).get("name", "?")
                            else:
                                name = getattr(getattr(tc, "function", None), "name", "?")
                            tool_names.append(name)
                    # Text content
                    if isinstance(content, str) and content.strip():
                        final_answer = content[:150]

            if user_q:
                lines.append(f"问: {user_q}")
            if tool_names:
                lines.append(f"  调用工具: {', '.join(tool_names[:5])}")
            if final_answer:
                lines.append(f"  结论: {final_answer}...")

        if not lines:
            return "(无历史对话)"

        return "## 历史对话摘要\n\n" + "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# Streaming helpers
# ══════════════════════════════════════════════════════════════

class StreamingToolParser:
    """Accumulates streaming chunks and detects tool_calls vs content.

    Usage::

        parser = StreamingToolParser()
        for chunk in stream:
            text = parser.feed(chunk)
            if text:
                yield text  # print / write to UI
        final = parser.result
        # final.content → str | None
        # final.tool_calls → list[dict] | None
    """

    def __init__(self):
        self._content = ""
        self._tool_calls: dict[int, dict] = {}  # index → {id, function.name, function.arguments}
        self._finish_reason = None

    def feed(self, chunk) -> str | None:
        """Feed one streaming chunk. Returns new display text or None."""
        delta = chunk.choices[0].delta
        fr = chunk.choices[0].finish_reason
        if fr:
            self._finish_reason = fr

        new_text = None
        if delta.content:
            self._content += delta.content
            new_text = delta.content

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in self._tool_calls:
                    self._tool_calls[idx] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                entry = self._tool_calls[idx]
                if tc_delta.id:
                    entry["id"] += tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        entry["function"]["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        entry["function"]["arguments"] += tc_delta.function.arguments

        return new_text

    @property
    def has_tool_calls(self) -> bool:
        return len(self._tool_calls) > 0

    @property
    def content(self) -> str:
        return self._content

    @property
    def tool_calls(self) -> list[dict]:
        """Return tool calls sorted by index."""
        return [self._tool_calls[i] for i in sorted(self._tool_calls)]

    @property
    def finish_reason(self) -> str | None:
        return self._finish_reason


class StreamMessage:
    """Lightweight container for a streamed message result."""
    def __init__(self, content, tool_calls, finish_reason):
        self.content = content
        self.tool_calls = tool_calls
        self.finish_reason = finish_reason


def stream_chat_completion(client, model, messages, tools=None, tool_choice=None):
    """Generator that yields (text_chunk_or_StreamMessage) tuples.

    Yields:
        str              — a piece of streaming text content (print immediately).
        StreamMessage     — sentinel at end of stream, carrying final state.

    Usage::

        for item in stream_chat_completion(client, model, messages, tools):
            if isinstance(item, str):
                print(item, end="", flush=True)
            else:
                msg = item  # StreamMessage with .content / .tool_calls
    """
    kwargs = dict(model=model, messages=messages, stream=True)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice or "auto"

    response = client.chat.completions.create(**kwargs)

    parser = StreamingToolParser()
    for chunk in response:
        text = parser.feed(chunk)
        if text:
            yield text

    # Build final StreamMessage
    tool_calls = None
    if parser.has_tool_calls:
        tool_calls = [
            type('ToolCall', (), {
                'id': tc['id'],
                'type': tc['type'],
                'function': type('Function', (), {
                    'name': tc['function']['name'],
                    'arguments': tc['function']['arguments'],
                }),
            })()
            for tc in parser.tool_calls
        ]

    yield StreamMessage(
        content=parser.content or None,
        tool_calls=tool_calls,
        finish_reason=parser.finish_reason,
    )
