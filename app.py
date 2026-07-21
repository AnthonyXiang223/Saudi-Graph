"""
MAZU Agent — Streamlit Web 应用
启动: streamlit run app.py
需要: DeepSeek API key (.env) + Flask dashboard (python dashboard/server.py)
"""

import streamlit as st
import json
import os
import sys
import logging
from context_manager import ContextManager, stream_chat_completion
from session_manager import SessionManager

# ── Page config ──
st.set_page_config(
    page_title="MAZU 沙特极端天气预警",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load API key ──
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# ── Load indicator name mapping ──
def _load_indicator_names():
    """Load {indicator_id: chinese_description} from operators.json."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "schema", "operators.json"), "r", encoding="utf-8") as f:
            ops = json.load(f)
        return {o["id"]: o["description"] for o in ops.get("operators", [])}
    except Exception:
        return {}

INDICATOR_NAMES = _load_indicator_names()

def _indicator_label(ind_id: str) -> str:
    """Get Chinese display name for an indicator ID."""
    name = INDICATOR_NAMES.get(ind_id, "")
    if name:
        return f"{name}（{ind_id}）"
    return ind_id

# ── Build dynamic system prompt ──
import datetime as _dt

def _build_system_prompt():
    today = _dt.date.today()

    # Auto-detect latest IFS date
    ifs_init = today
    project_dir = os.path.dirname(os.path.abspath(__file__))
    ifs_dir = os.path.join(project_dir, "aifs_forecasts")
    if os.path.isdir(ifs_dir):
        import re as _re
        ifs_dates = sorted([d for d in os.listdir(ifs_dir)
                          if os.path.isdir(os.path.join(ifs_dir, d)) and _re.match(r'^\d{8}', d)])
        if ifs_dates:
            try:
                ifs_init = _dt.date.fromisoformat(ifs_dates[-1])
            except Exception:
                pass
    ifs_offset = max((today - ifs_init).days, 0)

    return f"""你是 MAZU 多灾种早期预警系统的气象分析助手，服务沙特阿拉伯气象预警业务。

══════════════════════════════════════
时间上下文
══════════════════════════════════════
- 当前日期：{today.isoformat()}。用户说的"今天""明天"以此为准。
- 历史数据：2025 年全年 ERA5 再分析（365 天 NetCDF，35,200 格点，~100 个指标）。
- 预报数据：ECMWF IFS 全球预报(0.25°)，初始化于 {ifs_init.isoformat()}，forecast_day={ifs_offset} = 今天，明天 = {ifs_offset+1}，以此类推，最多覆盖初始化后 7 天。
- IFS 输出 0.25°×0.25° 全球网格，6 小时间隔。含完整大气变量（温度、湿度、风、降水、CAPE、海温等），变量覆盖远优于 FCN。

══════════════════════════════════════
能力边界（严格区分"能"与"不能"）
══════════════════════════════════════

## 你能回答的问题（调用对应工具即可获取答案）
- 四类灾害（极端高温、沙尘强风、山洪、沿海湿热）的未来 7 天风险检测 → detect_future_events
- 2025 年任意日期的历史极端事件回顾 → detect_extreme_events
- 91 个气象指标的物理定义、公式、推导链、数据来源 → query_indicator_* 系列
- 4 条检测规则的完整条件、权重、角色 → query_rule_detail
- 指定坐标周边半径内的指标观测 → query_observations_nearby
- 区域、时间线、级联事件、溯源查询 → 对应的 GeoSPARQL/OWL-Time/PROV-O 工具

## 你完全无法回答的问题（直接说明能力不足，不要绕弯）
- 任何涉及"实况""实测""实时""当前此刻"的问题 — 你只有 2025 年再分析和 IFS 预报，没有实时观测
- 卫星反演、雷达回波、土壤墒情、大气能见度 — 系统没有接入这些数据
- 2 小时短临预报、30 天长期预测、干旱演变、复合灾害叠加 — 超出 IFS 预报范围
- 概率百分比、精确起止时间、能见度米数 — 系统只输出风险评分和严重度等级
- 沙漠站点订正、卫星数据修正、数据融合推演 — 系统不具备这些算法
- 行业影响量化（减产百分比、经济损失、通航风险评估）— 系统没有接入行业模型
- 预警准确率、漏误报统计、复盘分析 — 系统没有运行业务化指标
- 双语/多语报告、热力图导出 — 系统不支持

**应对策略**：当被问及上述问题时，这样回答：
"当前系统不具备 [XX] 能力。以下基于 IFS 网格预报（或 ERA5 再分析），从 [已有数据的方面] 给出可用的分析："
然后立即给出已有数据能支撑的部分，不要先说一堆"我做不到"再给结论。

## 你可以部分回答的问题（需要拆解 + 诚实标注）
以下问题类型超出系统部分能力，但你仍可以从已有数据中提取有用信息：

| 用户问的是 | 你能做的是 | 必须标注的局限 |
|---|---|---|
| 沙漠无观测区温度 | 给出 IFS 格点预报温度值 | "IFS 网格预报值，沙漠区域再分析格点值通常低估地表实际温度 2-4℃" |
| 红海对流信号 | 检测山洪风险 + 查看日降水量格点触发情况 | "基于 IFS 预报场，非卫星实测" |
| 港区 72h 高温/沙尘 | 逐日调用 detect_future_events 检测 | 每天检测是独立的，不构成时间序列推演 |
| 干旱发展趋势 | 检查连续多日的日降水量和露点差 | "IFS 最多覆盖 7 天，无法做 10 天以上干旱趋势" |
| 区域差异化分析 | 分别对不同区域调用空间搜索 + 逐区域检测 | "阈值来自 rules.json，未做区域自适应校准" |
| 行业影响评估 | 基于检测结果 + 沙特气候常识，给出定性业务建议 | "定性分析，非量化行业模型输出" |

══════════════════════════════════════
研判规则（踩坑纠错 + 沙特本地校准）
══════════════════════════════════════

以下不是你本来就会的气象常识，而是实测中发现的错误模式。每条对应一个已被证实的误判场景。

**规则 1：高温不只看温度数值**
- 沙特 7 月沙漠格点 44-48℃ 是常态，不要一看到就报"极端"。
- 异常判断看三件：是否超过气候态 +5℃？露点差是否 >20℃？连续几天？
- ERA5 格点值在沙漠区域系统性偏低 2-4℃，输出时提及这个偏差，但不要说"已订正"。

**规则 2：沙尘热点不重合 ≠ 目标区域安全**
- 检测到的沙尘热点在阿曼湾时，不要直接判"港区不受影响"。
- 同一气团 + 同一干燥背景下，沙尘可沿 Shamal 方向传播。需检查目标区域的风向和干燥条件。
- 露点差 ≥20 且 RH <25% → 地表已满足起沙条件，即使当前风速未触发阈值，应标注为"潜在风险"而非"无风险"。

**规则 3：KG 物理一致性得分低 ≠ 模型不可靠**
- 沙特干季（5-10 月）可降水量与 IVT 辐合弱相关或负相关，这是气候常态。
- 不要因为相关系数低就说"存在不确定性"或"物理不一致"。它只是干季特征，不代表预报质量差。

**规则 4：山洪未触发 ≠ 无对流**
- 山洪检测覆盖有限（缺 CAPE、缺 flash_flood_risk），未触发不说明安全。
- 必须单独检查日降水量 ≥10mm 的格点数量和位置，这些才是有意义的对流信号。
- 红海沿岸的对流触发是地形抬升（阿西尔山脉），不是大尺度锋面系统。

══════════════════════════════════════
输出规范（强制遵守）
══════════════════════════════════════

**回答结构**（根据问题类型选择对应结构，禁止混用）：

### 类型 A：预警/检测类（"明天会不会有极端高温""本周沙尘风险如何"）
1. **结论先行**：一句话回答"有/没有，什么级别"。
2. **关键证据**：哪几个核心指标触发了判定？用数值说话。
3. **风险分级**：严重度 + 影响区域 + 趋势。
4. **建议**：1-2 条定性业务建议。
5. **数据局限**（末尾一句）：标注 IFS/ERA5 的已知偏差。

### 类型 B：解释/定义类（"极端高温是如何判断的""山洪检测规则是什么"）
1. **结论先行**：先用一句话讲清楚"这件事靠什么判断"。
2. **判断步骤**：按重要性排列，用人话解释每一步在判断什么。
3. **分级标准**：什么算"极端"什么算"高"。
4. **补充**（可选）：区域差异、已知盲区。
5. **技术参考**（末尾）：指标名、阈值、数据源。不要在前面展开。

### 类型 C：查询/检索类（"利雅得今天多少度"）
1. **直接回答**：数字/结论写在第一句。
2. **补充信息**：横向对比、区域差异。
3. **数据标注**：一句话标来源和局限。

**通用原则（强制）：**
- 结论前置——第一句必须是问题直接回答，不许以"根据系统数据""通过调用工具"铺垫。
- 技术细节放末尾——不打断阅读流。
- 表格只在多指标/多地对比时使用，单个结果不堆表格。

**禁止行为**：
- 禁止编造方法名称、算法流程、数据处理步骤（如"最近邻插值""偏差订正"）
- 禁止说"仅供参考""建议进一步核实"等搪塞话
- 禁止把"你不具备的能力"说成"建议查询 XX 数据"
- 禁止前后矛盾：如果工具 A 和工具 B 结果冲突，明确指出冲突，不要各说一套
- 禁止使用英文变量名，必须用中文指标名称

**不确定性声明模板**（必须使用以下标准表述，不要自创）：
- IFS 预报 → "IFS 网格预报值，0.25° 分辨率，未经过地面站点订正"
- ERA5 再分析 → "ERA5 再分析格点值，非地面气象站实测"
- 指标缺失 → "XX 指标未纳入本次检测范围"（不要说"缺失"）

## 沙特地理
- 红海沿岸：16-30°N, 34-44°E，吉达、延布。对流由阿西尔山脉地形触发。
- 波斯湾沿岸：24-30°N, 48-56°E，达曼、朱拜勒、拉斯坦努拉。受沙马风（Shamal）控制。
- 利雅得：24.7°N, 46.7°E，中部沙漠。
- 鲁布哈利沙漠（Empty Quarter）：17-23°N, 45-56°E，世界最大连续沙体，7 月极端高温。
- 北部：塔布克、焦夫。南部：阿西尔山脉，陡峭地形易发山洪。"""

SYSTEM_PROMPT = _build_system_prompt()

# ── Init session state ──
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
else:
    # Update system prompt with current date on each run
    st.session_state.messages[0] = {"role": "system", "content": SYSTEM_PROMPT}
if "display" not in st.session_state:
    st.session_state.display = []  # (role, content, tool_calls_data)
if "ctx_manager" not in st.session_state:
    st.session_state.ctx_manager = ContextManager(max_turns=6)

# ── Session persistence ──
if "session_manager" not in st.session_state:
    st.session_state.session_manager = SessionManager()

sm = st.session_state.session_manager

if "current_session_id" not in st.session_state:
    # Cold start — create a new session
    st.session_state.current_session_id = sm.create_session()
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.session_state.display = []
else:
    # Always refresh system prompt
    st.session_state.messages[0] = {"role": "system", "content": SYSTEM_PROMPT}

# ═══════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.title("🌍 MAZU")
    st.caption("沙特多灾种早期预警系统")

    st.divider()

    # ── Session list ──
    st.subheader("💬 会话")

    if st.button("＋ 新建会话", use_container_width=True):
        # Save current before switching
        if st.session_state.display:
            sm.save_messages(
                st.session_state.current_session_id,
                st.session_state.messages,
                st.session_state.display,
            )
        st.session_state.current_session_id = sm.create_session()
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.session_state.display = []
        st.rerun()

    sessions = sm.list_sessions()
    for s in sessions:
        is_active = s["id"] == st.session_state.current_session_id
        prefix = "▸ " if is_active else "  "
        label = f"{prefix}{s['title'] or '新会话'}"

        c1, c2 = st.columns([9, 1])
        with c1:
            if st.button(
                label,
                key=f"sess_{s['id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if s["id"] != st.session_state.current_session_id:
                    # Save current session
                    if st.session_state.display:
                        sm.save_messages(
                            st.session_state.current_session_id,
                            st.session_state.messages,
                            st.session_state.display,
                        )
                    # Load selected session
                    data = sm.get_session(s["id"])
                    if data:
                        st.session_state.current_session_id = s["id"]
                        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}] + data["messages"]
                        st.session_state.display = data["display"]
                    st.rerun()
        with c2:
            if st.button("✕", key=f"del_{s['id']}", help="删除会话"):
                sm.delete_session(s["id"])
                if s["id"] == st.session_state.current_session_id:
                    # Deleted current — create new
                    st.session_state.current_session_id = sm.create_session()
                    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                    st.session_state.display = []
                st.rerun()

    st.divider()
    st.title("🌍 MAZU")
    st.caption("沙特多灾种早期预警系统")

    st.divider()

    st.subheader("服务状态")

    import requests
    kg_online = False
    try:
        r = requests.get("http://127.0.0.1:5000/api/sparql/summary", timeout=2)
        kg_online = True
        st.success("知识图谱 · 在线")
    except Exception:
        st.error("知识图谱 · 离线")
        st.caption("请运行 `python dashboard/server.py`")

    import os as _os
    ifs_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "aifs_forecasts")
    ifs_available = _os.path.isdir(ifs_dir) and any(
        _os.path.isdir(_os.path.join(ifs_dir, d))
        for d in _os.listdir(ifs_dir) if d[:1] != '.'
    )
    if ifs_available:
        st.success("IFS 预报 · 就绪")
    else:
        st.warning("IFS 预报 · 未就绪")
        st.caption("请运行 IFS 下载脚本获取预报数据")

    st.divider()

    if kg_online:
        st.link_button(
            "🔗 打开知识图谱仪表盘",
            "http://127.0.0.1:5000/",
            use_container_width=True,
        )

    st.divider()
    st.caption("DeepSeek-V3 + ECMWF IFS + KWG")


# ═══════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════

SEVERITY_COLORS = {
    "extreme": "#dc3545",
    "emergency": "#dc3545",
    "severe": "#dc3545",
    "high": "#fd7e14",
    "alert": "#fd7e14",
    "medium": "#ffc107",
    "warning": "#ffc107",
    "moderate": "#ffc107",
    "low": "#28a745",
    "caution": "#28a745",
}

HAZARD_LABELS = {
    "flash_flood": "🌊 山洪",
    "extreme_heat": "🔥 极端高温",
    "dust_storm": "💨 沙尘强风",
    "coastal_humid_heat": "🏖️ 沿海湿热",
}


def render_hazard_card(h: dict):
    """Render a single hazard detection result as a card."""
    sev = h.get("severity", "low")
    color = SEVERITY_COLORS.get(sev, "#6c757d")
    label = HAZARD_LABELS.get(h.get("hazard_type", ""), h.get("hazard_type", "?"))

    with st.container(border=True):
        cols = st.columns([1, 4])
        with cols[0]:
            st.markdown(f"### {label}")
        with cols[1]:
            if h.get("detected"):
                st.markdown(
                    f"<span style='background:{color};color:white;padding:2px 10px;"
                    f"border-radius:10px;font-weight:bold'>{sev.upper()}</span> "
                    f"得分 **{h.get('max_risk_score', 0):.3f}** | "
                    f"覆盖 {h.get('coverage', '?')} | "
                    f"热点 ({h.get('hotspot_lat', '?')}N, {h.get('hotspot_lon', '?')}E)",
                    unsafe_allow_html=True,
                )
            else:
                reason = h.get("reason", "指标不足")
                st.caption(f"⚠ 未检出 — {reason}")

        if h.get("triggered_conditions"):
            cols2 = st.columns(len(h["triggered_conditions"]))
            for i, tc in enumerate(h["triggered_conditions"]):
                label = _indicator_label(tc["indicator"])
                with cols2[i]:
                    st.metric(
                        label,
                        f"{tc['peak_value']}",
                        delta=f"{tc['cells_triggered']} 格点触发",
                    )
                    st.caption(tc["condition"])

        if h.get("unavailable_indicators"):
            missing_cn = [_indicator_label(m) for m in h["unavailable_indicators"]]
            st.caption(f"⚠ 暂不可用: {', '.join(missing_cn)}")


def render_detection_results(data: dict):
    """Render detect_future_events output."""
    if "error" in data:
        st.error(f"预报失败: {data['error']}")
        if "hint" in data:
            st.info(data["hint"])
        return

    # Header
    st.markdown(f"**{data.get('forecast_source', 'IFS')}** · "
                f"Day +{data.get('forecast_day', '?')} · "
                f"Lead {data.get('lead_time_h', '?')}h")

    # Indicators
    with st.expander(f"📊 可用指标 ({len(data.get('available_indicators', []))})"):
        cols = st.columns(2)
        for i, ind in enumerate(data.get("available_indicators", [])):
            cols[i % 2].caption(f"• {_indicator_label(ind)}")
        if data.get("missing_indicators"):
            missing_cn = [_indicator_label(m) for m in data['missing_indicators']]
            st.caption(f"⚠ 暂不可用: {', '.join(missing_cn)}")

    # Hazard cards
    for h in data.get("hazards", []):
        render_hazard_card(h)

    # KG consistency
    if "kg_physical_consistency" in data:
        with st.expander("🔬 KG 物理一致性验证"):
            for htype, check in data["kg_physical_consistency"].items():
                score = check.get("physical_consistency_score")
                assessment = check.get("assessment", "")
                icon = "✅" if score and score >= 0.5 else "⚠️" if score else "❌"
                st.markdown(f"{icon} **{HAZARD_LABELS.get(htype, htype)}**: "
                           f"{assessment} (score={score}, {check.get('checks_passed', '?')})")
                for d in check.get("details", [])[:3]:
                    if "correlation" in d:
                        st.caption(f"  {d['variables']}: r={d['correlation']} "
                                  f"[{d['expected_sign']}] → {'✓' if d.get('coherent') else '✗'}")

    # Synthesis
    if "synthesis" in data:
        s = data["synthesis"]
        color = {"high": "green", "medium": "orange", "low": "red"}.get(s.get("confidence", ""), "grey")
        st.markdown(f"**综合结论** ({color}): {s['verdict']}")
        st.info(s.get("recommendation", ""))


def render_tool_result(tool_name: str, result_str: str):
    """Render a tool call result based on tool type."""
    try:
        data = json.loads(result_str)
    except Exception:
        st.code(result_str[:2000], language="json")
        return

    if tool_name == "detect_future_events":
        render_detection_results(data)
    elif tool_name == "detect_extreme_events":
        for event in data.get("events", data.get("results", []))[:5]:
            sev = event.get("severity", "?")
            color = SEVERITY_COLORS.get(sev, "#6c757d")
            st.markdown(
                f"<span style='background:{color};color:white;padding:2px 8px;"
                f"border-radius:8px'>{sev}</span> "
                f"{HAZARD_LABELS.get(event.get('hazard_type', ''), '')} "
                f"热点评分 {event.get('max_risk_score', '?')}",
                unsafe_allow_html=True,
            )
    elif tool_name in ("query_hazard_indicators", "query_rule_detail",
                       "query_indicator_detail", "query_indicator_chain",
                       "query_provenance"):
        # KG query results — show compact
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list) and len(v) > 0:
                    st.markdown(f"**{k}** ({len(v)} items)")
                    for item in v[:5]:
                        if isinstance(item, dict):
                            st.caption(json.dumps(item, ensure_ascii=False)[:200])
                elif isinstance(v, (str, int, float)):
                    st.caption(f"{k}: {str(v)[:200]}")
        else:
            st.json(data)
    else:
        # Default: compact JSON
        result_str_short = json.dumps(data, ensure_ascii=False)
        if len(result_str_short) > 1500:
            st.code(result_str_short[:1500] + "\n...(truncated)", language="json")
        else:
            st.json(data)


# ═══════════════════════════════════════════════════════
# Main Chat Interface
# ═══════════════════════════════════════════════════════

st.title("MAZU 沙特极端天气预警助手")
st.caption("基于 KnowWhereGraph DMDO-OWL + ECMWF IFS + DeepSeek-V3")

# Display chat history
for entry in st.session_state.display:
    role = entry["role"]
    content = entry["content"]
    tool_calls_data = entry.get("tool_calls")

    with st.chat_message(role):
        if content:
            st.markdown(content)

        # Show tool calls if any
        if tool_calls_data:
            for tc in tool_calls_data:
                with st.expander(f"🔧 {tc['name']}", expanded=False):
                    if tc.get("args"):
                        st.caption(f"参数: `{json.dumps(tc['args'], ensure_ascii=False)}`")
                    if tc.get("result") is not None:
                        render_tool_result(tc["name"], tc["result"])

# Chat input
if not API_KEY:
    st.warning("请在侧边栏输入 DeepSeek API Key")
else:
    if prompt := st.chat_input("请输入你的问题，例如: 明天沙特会有什么极端天气风险？"):
        # Add user message
        st.session_state.display.append({"role": "user", "content": prompt, "tool_calls": None})
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # ── Context window management ──
        ctx = st.session_state.ctx_manager
        st.session_state.messages = ctx.trim(st.session_state.messages)

        # Run agent
        with st.chat_message("assistant"):
            from openai import OpenAI

            client = OpenAI(
                api_key=API_KEY,
                base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            )

            from agent_tools import TOOLS as AGENT_TOOLS, dispatch_tool, smart_truncate

            # ★ ReAct loop — 一次流式调用同时处理 tool_calls 和文本
            display_tool_calls = []
            final_content = ""
            max_turns = 5
            status_placeholder = st.empty()

            for turn in range(max_turns):
                status_placeholder.caption(f"⏳ 分析中... (第 {turn+1}/{max_turns} 轮)")

                # ★ 单次流式调用 — 不先调 non-streaming 再调 streaming
                collected = ""
                stream_msg = None

                placeholder = st.empty()
                for item in stream_chat_completion(
                    client, "deepseek-chat", st.session_state.messages, AGENT_TOOLS
                ):
                    if isinstance(item, str):
                        collected += item
                        placeholder.markdown(collected + "▌")
                    else:
                        stream_msg = item

                if stream_msg is None:
                    continue

                # ── 工具调用 ──
                if stream_msg.tool_calls:
                    status_placeholder.caption("🔧 调用工具中...")
                    # 清除中间文本 placeholder
                    placeholder.empty()

                    # 构造 assistant message
                    tc_dicts = []
                    for tc in stream_msg.tool_calls:
                        tc_dicts.append({
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        })
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": stream_msg.content,
                        "tool_calls": tc_dicts,
                    })

                    for tc in stream_msg.tool_calls:
                        name = tc.function.name
                        args = json.loads(tc.function.arguments)
                        result = dispatch_tool(name, args)
                        result = smart_truncate(result, name, max_chars=3000)

                        st.session_state.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
                        display_tool_calls.append({
                            "name": name,
                            "args": args,
                            "result": result,
                        })

                        # 即时展示工具调用
                        with st.expander(f"🔧 {name}", expanded=False):
                            st.caption(f"参数: `{json.dumps(args, ensure_ascii=False)}`")
                            render_tool_result(name, result)

                # ── 最终回答（文本已在流式中打印）──
                else:
                    placeholder.markdown(collected)
                    final_content = stream_msg.content or collected
                    status_placeholder.empty()
                    st.session_state.messages.append(
                        {"role": "assistant", "content": final_content}
                    )
                    break

            else:
                # Exceeded max turns — streaming fallback
                status_placeholder.caption("💬 总结中...")
                st.session_state.messages.append({
                    "role": "user",
                    "content": "请基于上述工具返回的结果，给我一个简洁的总结。",
                })
                stream = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=st.session_state.messages,
                    stream=True,
                )
                placeholder = st.empty()
                full_text = ""
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_text += delta.content
                        placeholder.markdown(full_text + "▌")
                placeholder.markdown(full_text)
                final_content = full_text
                status_placeholder.empty()

            # Save to display
            st.session_state.display.append({
                "role": "assistant",
                "content": final_content,
                "tool_calls": display_tool_calls if display_tool_calls else None,
            })

            # Persist session to disk
            sm.save_messages(
                st.session_state.current_session_id,
                st.session_state.messages,
                st.session_state.display,
            )

        st.rerun()
