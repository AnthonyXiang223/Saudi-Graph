"""
MAZU Agent — 沙特极端天气预警助手
DeepSeek-V3 + KWG KG Tool Calling

Setup:
    1. 复制 .env.example 为 .env，填入你的 DeepSeek API key
    2. python agent.py
"""

import json
import os
import sys
from openai import OpenAI
from agent_tools import TOOLS, dispatch_tool

# ── 加载 API key ──
# 方式1: 从 .env 文件
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

# 方式2: 直接设置环境变量
# export DEEPSEEK_API_KEY=sk-xxx

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

# ── 系统 Prompt（注入 KG 知识上下文） ──
SYSTEM_PROMPT = """你是 MAZU 多灾种早期预警系统的首席分析助手，服务于沙特阿拉伯气象预警中心。

## 你的能力
你挂载了一个基于 KnowWhereGraph (DMDO-OWL) 的知识图谱，包含：
- 91 个极端气象指标（来自 ERA5 再分析、GPM IMERG 卫星降水、OSTIA 海温等）
- 6 个数据源：DS1(ERA5月值)、DS2(ERA5日值)、DS4(ERA5极值)、DS8(GHCN气候态)、DS10(GPM卫星)、SST(OSTIA海温)
- 4 种灾害检测规则：山洪(flash_flood)、极端高温(extreme_heat)、沙尘强风(dust_storm)、沿海湿热(coastal_humid_heat)
- 每条规则区分因果指标(提前信号)和并发指标(实况确认)，配有 primary gate 门控和 fallback 降级策略
- 空间查询由 GeoSPARQL 支持，时间推理由 OWL-Time 支持，数据溯源由 PROV-O 支持

## 预测能力
你可以调用 ECMWF AIFS 数值预报数据。当用户问"明天/未来几天会不会有..."时：
- 调用 detect_future_events(forecast_day=N) 进行预报检测
- forecast_day=1 是明天，=2 是后天，最多 7 天
- 预报存在不确定性，回答时应注明"基于 AIFS 预报"

## 工作原则
1. 收到预警相关问题时，先调用工具查询知识图谱，再基于工具返回的事实给出回答。绝不编造数据。
2. 当需要空间定位时（"红海沿岸""利雅得周边"），使用空间查询工具。
3. 当需要查看时间序列或级联事件时，使用时间线工具。
4. 当被问到"某个指标是什么/怎么算的"时，使用指标详情工具，展示公式和数据来源。
5. 回答用简洁的中文，引用工具返回的具体数值。如果是预警场景，明确说明严重度等级和受影响区域。
6. 如果工具返回空结果或错误，如实告知用户，不要自行猜测。

## 沙特地理常识（可不经工具直接使用）
- 红海沿岸（red_sea）：沙特西部沿海，16-30°N, 34-44°E。主要城市：吉达(Jeddah)、延布(Yanbu)
- 波斯湾沿岸（persian_gulf）：沙特东部沿海，24-30°N, 48-56°E。主要城市：达曼(Dammam)、朱拜勒(Jubail)
- 利雅得(Riyadh)：24.7°N, 46.7°E，首都，中部沙漠
- 北部(north_saudi)：26-32°N，包括塔布克(Tabuk)、焦夫(Al Jawf)
- 南部(south_saudi)：16-21°N，包括阿西尔山脉(Asir)，地形陡峭易发山洪"""

# ── Agent 主循环 ──
def chat():
    print("=" * 55)
    print("  MAZU 沙特极端天气预警助手")
    print("  DeepSeek-V3 + KWG KG (12 tools)")
    print("  输入 'quit' 退出, 'tools' 查看可用工具")
    print("=" * 55)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input("\n👉 你: ").strip()
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

        # ── ReAct 循环（最多 5 轮工具调用） ──
        for turn in range(5):
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            # 模型直接回答（不需要工具）
            if not msg.tool_calls:
                print(f"\n🤖 Agent: {msg.content}")
                messages.append({"role": "assistant", "content": msg.content})
                break

            # 模型要调用工具
            print(f"\n  🔄 调用工具中...")
            messages.append(msg)

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                print(f"    📡 {name}({json.dumps(args, ensure_ascii=False)})")

                result = dispatch_tool(name, args)
                # 截断过长结果
                if len(result) > 4000:
                    result = result[:4000] + "\n...(truncated)"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        else:
            # 超过 5 轮，强制总结
            print("\n🤖 Agent: 分析轮次较多，让我总结一下...")
            messages.append({"role": "user", "content": "请基于上述工具返回的结果，给我一个简洁的总结。"})
            response = client.chat.completions.create(
                model="deepseek-chat", messages=messages
            )
            print(f"\n🤖 Agent: {response.choices[0].message.content}")


if __name__ == "__main__":
    chat()
