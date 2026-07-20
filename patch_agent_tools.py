"""Patch agent_tools.py to add IFS tools."""
import re

with open('agent_tools.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find insertion point: last tool before TOOLS list close
old_marker = '    },\n\n]\n\n\n# ============================================Tool Dispatch'

ifs_tools = '''    },

    # ═══════════════════════════════════════════════════
    # Event Detection — IFS (ECMWF downloaded forecast)
    # ═══════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "detect_ifs_forecast",
            "description": "基于 ECMWF IFS 全球预报(0.25deg)检测未来灾害风险。IFS有完整大气变量(温度/湿度/风/降水/CAPE/SST等14+指标)，覆盖全部四种灾害。支持概率化门控(Gamma/GPD/Copula气候态)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "IFS 初始日期 YYYYMMDD。留空则用最近可用日期。"},
                    "forecast_day": {"type": "integer", "description": "预报天数偏移(0=分析场, 1=明天...最多7天)，默认0"},
                    "location": {"type": "string", "description": "关注地点，如利雅得,红海沿岸,吉达，可选"},
                    "hazard_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]},
                        "description": "要检测的灾害类型列表，不传则检测全部四种"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_ifs_dates",
            "description": "列出所有可用的 IFS 预报日期(aifs_forecasts/目录)。",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },

]

# ============================================Tool Dispatch'''

content = content.replace(old_marker, ifs_tools)

with open('agent_tools.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done: IFS tools inserted into TOOLS list')

# Also add dispatch entries
with open('agent_tools.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Insert dispatch for detect_ifs_forecast
old_dispatch = "            elif tool_name == \"detect_future_events\":\n                return self._detect_from_forecast(arguments)\n"
ifs_dispatch = "            elif tool_name == \"detect_ifs_forecast\":\n                return self._detect_from_ifs(arguments)\n" + old_dispatch
content = content.replace(old_dispatch, ifs_dispatch)

# Insert dispatch for list_ifs_dates
old_list = "            elif tool_name == \"lookup_city\":\n                return self._lookup_city_tool(arguments)\n"
ifs_list = "            elif tool_name == \"list_ifs_dates\":\n                return self._list_ifs_dates(arguments)\n" + old_list
content = content.replace(old_list, ifs_list)

with open('agent_tools.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done: dispatch entries added')

# Add the implementation methods
with open('agent_tools.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add _detect_from_ifs method before _detect_sequence
old_method = "    def _detect_sequence(self, args: dict) -> str:"
ifs_method = '''    def _detect_from_ifs(self, args: dict) -> str:
        """IFS forecast-based hazard detection."""
        if not HAS_IFS:
            return json.dumps({"error": "IFS pipeline not installed", "hint": "pip install ifs_pipeline"}, ensure_ascii=False)

        from ifs_pipeline import load_indicators_ifs, detect_ifs_hazards, build_ifs_forecast_report, list_ifs_dates

        date = args.get("date")
        forecast_day = args.get("forecast_day", 0)
        hazard_types = args.get("hazard_types")
        location = args.get("location")

        if hazard_types is None:
            hazard_types = ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]

        # Auto-select date if not provided
        if date is None:
            dates = list_ifs_dates()
            if not dates:
                return json.dumps({"error": "没有可用的IFS预报数据", "hint": "运行 python scripts/download/download_aifs.py 下载"}, ensure_ascii=False)
            date = dates[-1]  # latest

        ifs_data = load_indicators_ifs(date, forecast_day)
        if ifs_data is None:
            return json.dumps({"error": f"IFS数据不存在: {date}", "available": list_ifs_dates()}, ensure_ascii=False)

        hazards = detect_ifs_hazards(ifs_data, hazard_types)
        report = build_ifs_forecast_report(hazards, ifs_data, forecast_day, location)

        return json.dumps(report, ensure_ascii=False, indent=2)

    def _list_ifs_dates(self, args: dict) -> str:
        """List available IFS dates."""
        if not HAS_IFS:
            return json.dumps({"error": "IFS pipeline not installed"}, ensure_ascii=False)
        from ifs_pipeline import list_ifs_dates
        dates = list_ifs_dates()
        return json.dumps({"available_dates": dates, "count": len(dates), "source": "aifs_forecasts/"}, ensure_ascii=False, indent=2)

    def _detect_sequence(self, args: dict) -> str:'''

content = content.replace(old_method, ifs_method)

with open('agent_tools.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done: IFS methods added')
