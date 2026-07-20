"""
MAZU Agent — 沙特极端天气预警助手
DeepSeek-V3 + KWG KG Tool Calling

Setup:
    1. 复制 .env.example 为 .env，填入你的 DeepSeek API key
    2. python agent.py          → 命令行模式
    3. streamlit run app.py     → Web 界面
"""

import json
import os
import sys
import logging
from openai import OpenAI
from agent_tools import TOOLS, dispatch_tool
from context_manager import ContextManager

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mazu.agent")

# ── 加载 API key ──
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    print("请设置 DEEPSEEK_API_KEY 环境变量")
    print("方式: 复制 .env.example 为 .env，填入你的 key")
    sys.exit(1)

# ── 初始化 DeepSeek 客户端 ──
client = OpenAI(
    api_key=API_KEY,
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

# ── 系统 Prompt（自动获取日期，注入 KG 知识上下文） ──
import datetime as _dt

def _build_system_prompt():
    today = _dt.date.today()
    fcn_init = today
    fcn_nc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forecast", "fcn_forecast.nc")
    if os.path.exists(fcn_nc):
        try:
            import xarray as xr
            with xr.open_dataset(fcn_nc) as f:
                fcn_time = str(f["time"].values[0])[:10]
                fcn_init = _dt.date.fromisoformat(fcn_time)
        except Exception:
            pass
    fcn_offset = max((today - fcn_init).days, 0)

    return f"""你是 MAZU 多灾种早期预警系统的气象分析助手，服务沙特阿拉伯气象预警业务。

══════════════════════════════════════
时间上下文
══════════════════════════════════════
- 当前日期：{today.isoformat()}。用户说的"今天""明天"以此为准。
- 历史数据：2025 年全年 ERA5 再分析（365 天 NetCDF，35,200 格点，~100 个指标）。
- 预报数据：FCN 初始化于 {fcn_init.isoformat()}，forecast_day={fcn_offset} = 今天，明天 = {fcn_offset+1}，以此类推，最多覆盖初始化后 7 天。
- FCN 输出 0.25°×0.25° 网格（65×89 沙特区域），6 小时间隔。无站点级订正，无水汽通量、CAPE、海温等输出。

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
- 任何涉及"实况""实测""实时""当前此刻"的问题 — 你只有 2025 年再分析和 FCN 预报，没有实时观测
- 卫星反演、雷达回波、土壤墒情、大气能见度 — 系统没有接入这些数据
- 2 小时短临预报、30 天长期预测、干旱演变、复合灾害叠加 — 超出 FCN 预报范围
- 概率百分比、精确起止时间、能见度米数 — 系统只输出风险评分和严重度等级
- 沙漠站点订正、卫星数据修正、数据融合推演 — 系统不具备这些算法
- 行业影响量化（减产百分比、经济损失、通航风险评估）— 系统没有接入行业模型
- 预警准确率、漏误报统计、复盘分析 — 系统没有运行业务化指标
- 双语/多语报告、热力图导出 — 系统不支持

**应对策略**：当被问及上述问题时，这样回答：
"当前系统不具备 [XX] 能力。以下基于 FCN 网格预报（或 ERA5 再分析），从 [已有数据的方面] 给出可用的分析："
然后立即给出已有数据能支撑的部分，不要先说一堆"我做不到"再给结论。

## 你可以部分回答的问题（需要拆解 + 诚实标注）
以下问题类型超出系统部分能力，但你仍可以从已有数据中提取有用信息：

| 用户问的是 | 你能做的是 | 必须标注的局限 |
|---|---|---|
| 沙漠无观测区温度 | 给出 FCN 格点预报温度值 | "FCN 网格预报值，沙漠区域再分析格点值通常低估地表实际温度 2-4℃" |
| 红海对流信号 | 检测山洪风险 + 查看日降水量格点触发情况 | "基于 FCN 预报场，非卫星实测" |
| 港区 72h 高温/沙尘 | 逐日调用 detect_future_events 检测 | 每天检测是独立的，不构成时间序列推演 |
| 干旱发展趋势 | 检查连续多日的日降水量和露点差 | "FCN 最多覆盖 7 天，无法做 10 天以上干旱趋势" |
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

**回答结构**（所有问题，不只是预警类）：
1. **数据来源**：一句话说明用了什么数据、什么工具、什么局限。
2. **分析结果**：工具返回的关键数值和你的解读。
3. **气象机理**：基于上述沙特气候知识解释，不要套通用气象逻辑。
4. **业务含义**：对具体行业/区域的定性影响。如果没有行业信息，说最可能的受影响对象。

**禁止行为**：
- 禁止编造方法名称、算法流程、数据处理步骤（如"最近邻插值""偏差订正"）
- 禁止说"仅供参考""建议进一步核实"等搪塞话
- 禁止把"你不具备的能力"说成"建议查询 XX 数据"
- 禁止前后矛盾：如果工具 A 和工具 B 结果冲突，明确指出冲突，不要各说一套
- 禁止使用英文变量名，必须用中文指标名称

**不确定性声明模板**（必须使用以下标准表述，不要自创）：
- FCN 预报 → "FCN 网格预报值，0.25° 分辨率，未经过地面站点订正"
- ERA5 再分析 → "ERA5 再分析格点值，非地面气象站实测"
- 指标缺失 → "XX 指标未纳入本次检测范围"（不要说"缺失"）

## 沙特地理
- 红海沿岸：16-30°N, 34-44°E，吉达、延布。对流由阿西尔山脉地形触发。
- 波斯湾沿岸：24-30°N, 48-56°E，达曼、朱拜勒、拉斯坦努拉。受沙马风（Shamal）控制。
- 利雅得：24.7°N, 46.7°E，中部沙漠。
- 鲁布哈利沙漠（Empty Quarter）：17-23°N, 45-56°E，世界最大连续沙体，7 月极端高温。
- 北部：塔布克、焦夫。南部：阿西尔山脉，陡峭地形易发山洪。"""

SYSTEM_PROMPT = _build_system_prompt()


# ═══════════════════════════════════════════════════════
# Agent 主循环
# ═══════════════════════════════════════════════════════

def chat():
    print("=" * 55)
    print("  MAZU 沙特极端天气预警助手")
    print("  DeepSeek-V3 + KWG KG | 流式输出 | 滑动窗口记忆")
    print("  输入 'quit' 退出, 'tools' 查看可用工具")
    print("=" * 55)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    ctx = ContextManager(max_turns=6)

    while True:
        try:
            user_input = input("\n👉 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("再见。")
            break
        if user_input.lower() == "tools":
            for t in TOOLS:
                print(f"  🔧 {t['function']['name']}: {t['function']['description'][:80]}")
            print(f"  共 {len(TOOLS)} 个工具")
            continue

        messages.append({"role": "user", "content": user_input})

        # ── Context window management ──
        messages = ctx.trim(messages)

        # ── ReAct 循环（最多 5 轮工具调用） ──
        for turn in range(5):
            # 非流式调用 → 快速判断是否需要工具
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            # ── 模型直接回答（不需要工具）→ 流式输出最终答案 ──
            if not msg.tool_calls:
                print(f"\n🤖 Agent: ", end="", flush=True)
                # 重新以流式调用，获得逐 token 输出
                stream = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    stream=True,
                )
                full_content = ""
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_content += delta.content
                        print(delta.content, end="", flush=True)
                print()  # trailing newline
                messages.append({"role": "assistant", "content": full_content})
                break

            # ── 模型要调用工具 ──
            print(f"\n  🔄 调用工具中...")
            messages.append(msg)

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                print(f"    📡 {name}({json.dumps(args, ensure_ascii=False)})")

                result = dispatch_tool(name, args)
                # 截断过长结果
                if len(result) > 3000:
                    result = result[:3000] + "\n...(truncated)"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        else:
            # 超过 5 轮，强制总结
            print("\n🤖 Agent: 分析轮次较多，让我总结一下...")
            messages.append({"role": "user", "content": "请基于上述工具返回的结果，给我一个简洁的总结。"})
            # 流式输出总结
            stream = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=True,
            )
            full_content = ""
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_content += delta.content
                    print(delta.content, end="", flush=True)
            print()
            messages.append({"role": "assistant", "content": full_content})


if __name__ == "__main__":
    chat()
