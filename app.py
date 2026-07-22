"""
MAZU Agent — Streamlit Web UI
Launch: streamlit run app.py
Requires: DeepSeek API key (.env) + Flask dashboard (python dashboard/server.py)
"""

import streamlit as st
import json
import os
from context_manager import ContextManager
from session_manager import SessionManager

# ── Page config ──
st.set_page_config(
    page_title="MAZU · Saudi Multi-Hazard Early Warning",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════
# Custom CSS — taste-skill: dark scientific theme, amber accent
# ═══════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Root overrides ── */
.stApp { background: #0c0f14; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: #13171f;
  border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .stMarkdown h1 {
  font-size: 1.2rem !important;
  font-weight: 700 !important;
  letter-spacing: -0.01em;
  color: #e6edf3 !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .st-caption {
  color: #8b949e !important;
  font-size: 0.8rem !important;
}

/* ── Sidebar buttons ── */
[data-testid="stSidebar"] .stButton button {
  background: #191e28;
  border: 1px solid rgba(255,255,255,0.09);
  color: #e6edf3;
  border-radius: 6px;
  font-size: 0.82rem;
  font-weight: 500;
  transition: all 0.2s ease;
}
[data-testid="stSidebar"] .stButton button:hover {
  background: #1f2532;
  border-color: rgba(212,168,83,0.25);
  color: #d4a853;
}
[data-testid="stSidebar"] .stButton button[kind="primary"] {
  background: rgba(212,168,83,0.12);
  border-color: rgba(212,168,83,0.20);
}

/* ── Sidebar dividers ── */
[data-testid="stSidebar"] hr {
  border-color: rgba(255,255,255,0.06);
  margin: 0.8rem 0;
}

/* ── Sidebar expanders / status ── */
[data-testid="stSidebar"] .stAlert {
  background: transparent !important;
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 6px;
  font-size: 0.78rem;
}

/* ── Title / header ── */
.stApp header[data-testid="stHeader"] {
  background: transparent;
}
.main-header { display: none; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
  padding: 0.6rem 0.4rem;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
  font-size: 0.92rem;
  line-height: 1.65;
  color: #e6edf3;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3 {
  color: #e6edf3 !important;
  font-weight: 700;
  letter-spacing: -0.01em;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3 {
  font-size: 1.05rem !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] table {
  font-size: 0.82rem;
  border-collapse: collapse;
  width: 100%;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th {
  background: #191e28;
  color: #d4a853;
  font-weight: 600;
  padding: 6px 12px;
  border-bottom: 1px solid rgba(255,255,255,0.09);
  text-align: left;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] td {
  padding: 5px 12px;
  border-bottom: 1px solid rgba(255,255,255,0.05);
  color: #c0c7cf;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {
  font-family: 'SF Mono', 'Cascadia Code', 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  background: #11151d;
  padding: 1px 6px;
  border-radius: 3px;
  color: #6b9fd4;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
  background: #11151d;
  border: 1px solid rgba(255,255,255,0.09);
  border-radius: 8px;
  color: #e6edf3;
  font-size: 0.9rem;
  padding: 10px 14px;
}
[data-testid="stChatInput"] textarea:focus {
  border-color: #d4a853;
  box-shadow: none;
}
[data-testid="stChatInput"] textarea::placeholder {
  color: #555d68;
}

/* ── Expander (tool calls) ── */
[data-testid="stExpander"] {
  background: #11151d;
  border: 1px solid rgba(255,255,255,0.06) !important;
  border-radius: 6px !important;
  margin: 6px 0;
  font-size: 0.8rem;
}
[data-testid="stExpander"] details summary {
  color: #8b949e;
  font-family: 'SF Mono', 'Cascadia Code', 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  padding: 6px 10px;
}
[data-testid="stExpander"] details summary:hover {
  color: #c0c7cf;
}

/* ── Container cards (hazard cards) ── */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: #13171f;
  border: 1px solid rgba(255,255,255,0.06) !important;
  border-radius: 8px !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stMarkdownContainer"] h3) {
  border-left: 3px solid rgba(212,168,83,0.4) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
  background: #11151d;
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 6px;
  padding: 8px;
}
[data-testid="stMetric"] label {
  color: #8b949e !important;
  font-size: 0.7rem !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: #e6edf3 !important;
  font-size: 1.3rem !important;
  font-weight: 700 !important;
  font-family: 'SF Mono', 'Cascadia Code', 'JetBrains Mono', monospace;
}
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
  font-size: 0.72rem;
}

/* ── Buttons (main area) ── */
.stButton button {
  background: #191e28;
  border: 1px solid rgba(255,255,255,0.09);
  color: #e6edf3;
  border-radius: 6px;
  font-weight: 500;
  font-size: 0.82rem;
  transition: all 0.2s ease;
}
.stButton button:hover {
  background: #1f2532;
  border-color: rgba(212,168,83,0.25);
  color: #d4a853;
}
.stButton button[kind="primary"] {
  background: #d4a853;
  border-color: #d4a853;
  color: #0c0f14;
  font-weight: 600;
}
.stButton button[kind="primary"]:hover {
  background: #e0b95e;
  color: #0c0f14;
}

/* ── Link buttons ── */
.stLinkButton a {
  border-radius: 6px !important;
  font-size: 0.8rem !important;
  border: 1px solid rgba(255,255,255,0.09) !important;
  background: #191e28 !important;
  color: #8b949e !important;
}
.stLinkButton a:hover {
  border-color: rgba(212,168,83,0.25) !important;
  color: #d4a853 !important;
}

/* ── Warning / Info / Success boxes ── */
.stAlert {
  border-radius: 6px;
  font-size: 0.84rem;
  border: 1px solid rgba(255,255,255,0.06);
}
div[data-testid="stAlert"][kind="warning"] {
  background: rgba(212,131,58,0.08);
  border-color: rgba(212,131,58,0.2);
}
div[data-testid="stAlert"][kind="info"] {
  background: rgba(107,159,212,0.08);
  border-color: rgba(107,159,212,0.2);
}
div[data-testid="stAlert"][kind="success"] {
  background: rgba(90,158,111,0.08);
  border-color: rgba(90,158,111,0.2);
}
div[data-testid="stAlert"][kind="error"] {
  background: rgba(224,85,96,0.08);
  border-color: rgba(224,85,96,0.2);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.06); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.13); }

/* ── Caption / small text ── */
.st-caption { color: #555d68 !important; font-size: 0.75rem !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
  color: #8b949e;
  font-size: 0.82rem;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
  color: #d4a853;
}

/* ── Selectbox / inputs ── */
.stSelectbox [data-baseweb="select"] > div {
  background: #11151d;
  border-color: rgba(255,255,255,0.09);
  border-radius: 6px;
}

/* ── Divider ── */
hr {
  border-color: rgba(255,255,255,0.06) !important;
  margin: 0.6rem 0 !important;
}

/* ═══════════════════════════════════════════════
   MOTION — taste-skill MOTION_INTENSITY: 6
   ═══════════════════════════════════════════════ */

/* ── Ambient background orbs ── */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  background:
    radial-gradient(circle at 20% 30%, rgba(212,168,83,0.025) 0%, transparent 50%),
    radial-gradient(circle at 75% 60%, rgba(107,159,212,0.018) 0%, transparent 50%);
  animation: ambientShift 18s ease-in-out infinite alternate;
}
@keyframes ambientShift {
  0%   { opacity: 0.6; }
  50%  { opacity: 1; }
  100% { opacity: 0.6; }
}

/* ── Chat message entry ── */
[data-testid="stChatMessage"] {
  animation: msgEnter 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
}
[data-testid="stChatMessage"]:nth-child(1) { animation-delay: 0.02s; }
[data-testid="stChatMessage"]:nth-child(2) { animation-delay: 0.06s; }
[data-testid="stChatMessage"]:nth-child(3) { animation-delay: 0.10s; }
[data-testid="stChatMessage"]:nth-child(4) { animation-delay: 0.14s; }
[data-testid="stChatMessage"]:nth-child(5) { animation-delay: 0.18s; }
[data-testid="stChatMessage"]:nth-child(6) { animation-delay: 0.22s; }
[data-testid="stChatMessage"]:nth-child(7) { animation-delay: 0.26s; }
[data-testid="stChatMessage"]:nth-child(8) { animation-delay: 0.30s; }
@keyframes msgEnter {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ── Button hover physics ── */
.stButton button {
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.stButton button:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}
.stButton button:active {
  transform: scale(0.97) !important;
  transition-duration: 0.08s !important;
}
.stButton button[kind="primary"]:hover {
  box-shadow: 0 4px 20px rgba(212,168,83,0.2);
}

/* ── Link button hover ── */
.stLinkButton a {
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.stLinkButton a:hover {
  transform: translateY(-1px);
}

/* ── Card hover lift ── */
[data-testid="stVerticalBlockBorderWrapper"] {
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.25);
  border-color: rgba(212,168,83,0.15) !important;
}

/* ── Metric card hover ── */
[data-testid="stMetric"] {
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}
[data-testid="stMetric"]:hover {
  transform: translateY(-1px);
  border-color: rgba(212,168,83,0.2);
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

/* ── Expander hover ── */
[data-testid="stExpander"] {
  transition: all 0.25s ease;
}
[data-testid="stExpander"]:hover {
  border-color: rgba(255,255,255,0.12) !important;
  transform: translateX(2px);
}
[data-testid="stExpander"] details summary {
  transition: color 0.2s ease;
}
[data-testid="stExpander"] details summary .st-emotion-cache-1aegexp {
  transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}

/* ── Chat input focus glow ── */
[data-testid="stChatInput"] textarea {
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}
[data-testid="stChatInput"] textarea:focus {
  box-shadow: 0 0 0 2px rgba(212,168,83,0.25);
}

/* ── Sidebar button hover slide ── */
[data-testid="stSidebar"] .stButton button {
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
[data-testid="stSidebar"] .stButton button:hover {
  transform: translateX(3px);
}
[data-testid="stSidebar"] .stButton button:active {
  transform: scale(0.97) !important;
}

/* ── Status dot pulse ── */
[data-testid="stSidebar"] .stAlert [kind="success"] {
  animation: statusPulse 3s ease-in-out infinite;
}
@keyframes statusPulse {
  0%, 100% { border-color: rgba(90,158,111,0.15); }
  50%      { border-color: rgba(90,158,111,0.35); }
}

/* ── Tab underline slide ── */
.stTabs [data-baseweb="tab"] {
  transition: color 0.25s ease;
}
.stTabs [data-baseweb="tab"]::after {
  content: '';
  display: block;
  width: 0;
  height: 2px;
  background: #d4a853;
  transition: width 0.3s cubic-bezier(0.16, 1, 0.3, 1);
  margin-top: 2px;
}
.stTabs [data-baseweb="tab"][aria-selected="true"]::after {
  width: 100%;
}

/* ── Table row hover ── */
[data-testid="stChatMessage"] table tbody tr {
  transition: background 0.2s ease;
}
[data-testid="stChatMessage"] table tbody tr:hover {
  background: rgba(212,168,83,0.06);
}

/* ── Severity badge shimmer (inline spans with background color) ── */
[data-testid="stChatMessage"] span[style*="background"] {
  transition: all 0.3s ease;
}

/* ═══════════════════════════════════════════
   REDUCED MOTION
   ═══════════════════════════════════════════ */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
  .stApp::before { display: none; }
  [data-testid="stChatMessage"] { animation: none; opacity: 1; transform: none; }
}
</style>
""", unsafe_allow_html=True)

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
    """Load {indicator_id: description} from operators.json."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "schema", "operators.json"), "r", encoding="utf-8") as f:
            ops = json.load(f)
        return {o["id"]: o["description"] for o in ops.get("operators", [])}
    except Exception:
        return {}

INDICATOR_NAMES = _load_indicator_names()

def _indicator_label(ind_id: str) -> str:
    """Get display name for an indicator ID."""
    name = INDICATOR_NAMES.get(ind_id, "")
    if name:
        return f"{name} ({ind_id})"
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
You are the meteorological analysis assistant of the MAZU multi-hazard early warning system.

══════════════════════════════════════
语言 / Language / اللغة
══════════════════════════════════════
- **Automatic language detection**: Reply in the SAME language as the user's question.
- 用户用中文提问 → 你用中文回答。
- User asks in English → reply in English.
- إذا سأل المستخدم بالعربية → أجب بالعربية.
- All meteorological indicators, severity levels, and hazard names should be translated to the reply language.

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

### 类型 B：解释/定义类（"极端高温是如何判断的""山洪检测规则是什么"）
1. **结论先行**：先用一句话讲清楚"这件事靠什么判断"。
2. **判断步骤**：按重要性排列，用人话解释每一步在判断什么。
3. **分级标准**：什么算"极端"什么算"高"。
4. **补充**（可选）：区域差异、已知盲区。
5. **技术参考**（末尾）：指标名、阈值、数据源。不要在前面展开。

### 类型 C：查询/检索类（"利雅得今天多少度"）
1. **直接回答**：数字/结论写在第一句。
2. **补充信息**：横向对比、区域差异。

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
    # Cold start — clean up empty sessions, create a new one
    for s in sm.list_sessions():
        if s["message_count"] == 0:
            sm.delete_session(s["id"])
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
    st.markdown("### 🌍 MAZU")
    st.caption("Saudi Multi-Hazard Early Warning System")

    st.divider()

    # ── Session list ──
    st.caption("SESSIONS")

    if st.button("+ New Session", use_container_width=True):
        # Save current before switching (skip if empty)
        if st.session_state.display:
            sm.save_messages(
                st.session_state.current_session_id,
                st.session_state.messages,
                st.session_state.display,
            )
        else:
            # Don't keep empty sessions
            sm.delete_session(st.session_state.current_session_id)
        st.session_state.current_session_id = sm.create_session()
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.session_state.display = []
        st.rerun()

    sessions = [s for s in sm.list_sessions() if s["message_count"] > 0]
    for s in sessions:
        is_active = s["id"] == st.session_state.current_session_id
        prefix = "▸ " if is_active else "  "
        label = f"{prefix}{s['title'] or 'New Session'}"

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
            if st.button("✕", key=f"del_{s['id']}", help="Delete session"):
                sm.delete_session(s["id"])
                if s["id"] == st.session_state.current_session_id:
                    # Deleted current — create new
                    st.session_state.current_session_id = sm.create_session()
                    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                    st.session_state.display = []
                st.rerun()

    st.divider()

    st.caption("SERVICE STATUS")

    import requests
    kg_online = False
    try:
        r = requests.get("http://127.0.0.1:5000/api/sparql/summary", timeout=2)
        kg_online = True
        st.success("KG Dashboard · Online")
    except Exception:
        st.error("KG Dashboard · Offline")
        st.caption("Run `python dashboard/server.py`")

    import os as _os
    ifs_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "aifs_forecasts")
    ifs_available = _os.path.isdir(ifs_dir) and any(
        _os.path.isdir(_os.path.join(ifs_dir, d))
        for d in _os.listdir(ifs_dir) if d[:1] != '.'
    )
    if ifs_available:
        st.success("IFS Forecasts · Ready")
    else:
        st.warning("IFS Forecasts · Not Ready")
        st.caption("Run IFS download script")

    st.divider()

    if kg_online:
        st.link_button(
            "Knowledge Graph →",
            "http://127.0.0.1:5000/",
            use_container_width=True,
        )

    st.divider()


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
    "flash_flood": "🌊 Flash Flood",
    "extreme_heat": "🔥 Extreme Heat",
    "dust_storm": "💨 Dust Storm",
    "coastal_humid_heat": "🏖️ Coastal Humid Heat",
}


def _render_triggered_conditions(conditions: list):
    """FCN-style: triggered_conditions with indicator/peak_value/cells_triggered."""
    cols = st.columns(len(conditions))
    for i, tc in enumerate(conditions):
        label = _indicator_label(tc["indicator"])
        with cols[i]:
            st.metric(
                label,
                f"{tc['peak_value']}",
                delta=f"{tc['cells_triggered']} cells triggered",
            )
            st.caption(tc.get("condition", ""))


def _render_ifs_triggers(h: dict):
    """IFS-style: primary_triggers + gate_detail."""
    trigs = h.get("primary_triggers", [])
    gate = h.get("gate_detail", {})
    n = len(trigs)
    if n == 0:
        return
    cols = st.columns(min(n, 4))
    for i, ind_id in enumerate(trigs[:4]):
        label = _indicator_label(ind_id)
        with cols[i % 4]:
            st.metric(label, "✓ Triggered")
    if gate:
        pm = gate.get("primary_met_pct", 0)
        pg = gate.get("prob_gate_met_pct", 0)
        st.caption(f"Primary Gate {pm}% · Prob Gate {pg}%")
    if h.get("unavailable"):
        st.caption(f"Unavailable: {', '.join(h['unavailable'])}")


def render_hazard_card(h: dict):
    """Render a single hazard detection result as a card."""
    sev = h.get("severity", "low")
    color = SEVERITY_COLORS.get(sev, "#6c757d")
    label = HAZARD_LABELS.get(h.get("hazard_type", ""), h.get("hazard_type", "?"))

    score = h.get("max_risk_score") or h.get("max_score") or 0
    hotspot = h.get("hotspot") or f"{h.get('hotspot_lat', '?')}N, {h.get('hotspot_lon', '?')}E"
    coverage = h.get("coverage") or "?"
    triggered_pct = h.get("triggered_pct")

    with st.container(border=True):
        cols = st.columns([1, 4])
        with cols[0]:
            st.markdown(f"### {label}")
        with cols[1]:
            if h.get("detected"):
                extra = f" ({triggered_pct}% 格点)" if triggered_pct is not None else ""
                st.markdown(
                    f"<span style='background:{color};color:white;padding:2px 10px;"
                    f"border-radius:10px;font-weight:bold'>{sev.upper()}</span> "
                    f"得分 **{score:.3f}** | "
                    f"覆盖 {coverage} | "
                    f"热点 {hotspot}{extra}",
                    unsafe_allow_html=True,
                )
            else:
                reason = h.get("reason", "指标不足")
                st.caption(f"⚠ Not detected — {reason}")

        if h.get("triggered_conditions"):
            # FCN-style: triggered_conditions list
            _render_triggered_conditions(h["triggered_conditions"])
        elif h.get("primary_triggers"):
            # IFS-style: primary_triggers list of indicator IDs
            _render_ifs_triggers(h)
        elif h.get("unavailable"):
            missing = h["unavailable"]
            st.caption(f"⚠ Unavailable: {', '.join(missing)}")


def render_detection_results(data: dict):
    """Render detect_future_events output."""
    if "error" in data:
        st.error(f"Forecast failed: {data['error']}")
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
        st.markdown(f"**Overall Assessment** ({color}): {s['verdict']}")
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

st.markdown("## MAZU · Saudi Multi-Hazard Early Warning")
# Display chat history
for entry in st.session_state.display:
    role = entry["role"]
    content = entry["content"]
    tool_calls_data = entry.get("tool_calls")

    with st.chat_message(role):
        if content:
            st.markdown(content)

        # Tool calls are processed silently in the background

# Chat input
if not API_KEY:
    st.warning("Enter your DeepSeek API Key in the sidebar")
else:
    if prompt := st.chat_input("Ask a question, e.g.: What extreme weather risks does Saudi Arabia face tomorrow?"):
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

            # ReAct loop — non-streaming
            display_tool_calls = []
            final_content = ""
            max_turns = 5
            status_placeholder = st.empty()

            for turn in range(max_turns):
                status_placeholder.caption(f"⏳ 分析中... (第 {turn+1}/{max_turns} 轮)")

                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=st.session_state.messages,
                    tools=AGENT_TOOLS,
                    tool_choice="auto",
                )
                msg = response.choices[0].message

                # ── 工具调用 ──
                if msg.tool_calls:
                    status_placeholder.caption("🔧 调用工具中...")

                    st.session_state.messages.append(msg)

                    for tc in msg.tool_calls:
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

                        # Tool runs silently; results feed into the final answer

                # ── 最终回答 ──
                else:
                    final_content = msg.content or ""
                    status_placeholder.empty()
                    st.markdown(final_content)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": final_content}
                    )
                    break

            else:
                # Exceeded max turns — fallback
                status_placeholder.caption("💬 总结中...")
                st.session_state.messages.append({
                    "role": "user",
                    "content": "请基于上述工具返回的结果，给我一个简洁的总结。",
                })
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=st.session_state.messages,
                )
                final_content = response.choices[0].message.content or ""
                st.session_state.messages.append(
                    {"role": "assistant", "content": final_content}
                )
                st.markdown(final_content)
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
