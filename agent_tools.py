"""
MAZU Agent Tool Definitions — OpenAI-compatible Function Calling schema.
Wraps the KWG-based DMDO-OWL KG API into LLM-callable tools.

Usage:
    from agent_tools import TOOLS, dispatch_tool

    # In your LLM function calling loop:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto"
    )
    # ... handle response, call dispatch_tool() for each tool call
"""

import json
import logging
import os
import sys
import time as _time
import traceback as _traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

log = logging.getLogger("mazu.tools")

# IFS forecast pipeline integration
try:
    from ifs_pipeline import load_indicators_ifs, detect_ifs_hazards, build_ifs_forecast_report
    HAS_IFS = True
except ImportError:
    HAS_IFS = False

# ── Tool Schema Definitions (OpenAI format) ──

TOOLS = [
    # ═══════════════════════════════════════════════════
    # Knowledge Graph Core Queries
    # ═══════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "query_hazard_indicators",
            "description": "查询某种灾害类型依赖的所有气象指标。返回指标ID、描述和单位。适用于'山洪需要哪些条件''极端高温依赖什么指标'等问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "hazard_type": {
                        "type": "string",
                        "enum": ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"],
                        "description": "灾害类型: flash_flood(山洪), extreme_heat(极端高温), dust_storm(沙尘强风), coastal_humid_heat(沿海湿热)"
                    }
                },
                "required": ["hazard_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_indicator_detail",
            "description": "查询一个气象指标的完整信息：物理含义、计算公式、可执行DAG、数据来源、联合解释指标、局限性说明。适用于'vpd_kpa是什么''cape怎么算的'等问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "indicator_id": {
                        "type": "string",
                        "description": "指标ID，如 tmax_c, cape, daily_precip_total, vpd_kpa 等（共91个）"
                    }
                },
                "required": ["indicator_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_indicator_chain",
            "description": "追溯一个指标从原始数据源开始的完整推导链。返回每一层计算步骤和最终数据来源（DS1/DS2/DS4/DS8/DS10/SST）。适用于'heatwave_duration_days是怎么一步步算出来的'等问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "indicator_id": {
                        "type": "string",
                        "description": "指标ID"
                    }
                },
                "required": ["indicator_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_indicators",
            "description": "按关键词搜索气象指标。返回匹配的指标ID和描述。适用于'有哪些降水相关的指标''搜索和温度有关的变量'等问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词，中英文均可（如 'precip', '降水', 'heat', '高温'）"
                    }
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_rule_detail",
            "description": "查询某条检测规则的完整条件列表，包括每个条件的因果角色（causal因果/提前量, concurrent并发/实况, derived_gate衍生/门控）、权重、阈值、primary gate和fallback降级策略。适用于'山洪检测规则是什么条件''高温预警怎么判断的'等问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_id": {
                        "type": "string",
                        "description": "规则ID: flash_flood_weighted, extreme_heat_weighted, dust_storm_weighted, coastal_humid_heat_weighted"
                    }
                },
                "required": ["rule_id"]
            }
        }
    },

    # ═══════════════════════════════════════════════════
    # GeoSPARQL Spatial Queries
    # ═══════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "query_observations_nearby",
            "description": "在指定坐标的半径范围内，搜索某指标超过阈值的观测值。返回每个格点的经纬度、数值和距离。适用于'利雅得周边100km内哪里超过45°C''红海沿岸哪些地方降水超过10mm'等问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "indicator": {
                        "type": "string",
                        "description": "指标ID，如 tmax_c, daily_precip_total"
                    },
                    "lat": {"type": "number", "description": "中心点纬度（度）"},
                    "lon": {"type": "number", "description": "中心点经度（度）"},
                    "radius_km": {"type": "number", "description": "搜索半径（公里）"},
                    "date": {"type": "string", "description": "日期 YYYYMMDD，可选"},
                    "min_value": {"type": "number", "description": "最小值过滤，可选"}
                },
                "required": ["indicator", "lat", "lon", "radius_km"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_events_in_region",
            "description": "查询某个地理区域内发生的所有极端灾害事件。返回事件ID、灾害类型、严重度和面积。适用于'红海有哪些山洪事件''波斯湾沿岸发生过什么灾害'等问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "enum": ["red_sea", "persian_gulf", "north_saudi", "central_saudi", "south_saudi", "saudi_bbox"],
                        "description": "地理区域ID"
                    }
                },
                "required": ["region"]
            }
        }
    },

    # ═══════════════════════════════════════════════════
    # OWL-Time Temporal Queries
    # ═══════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "query_event_timeline",
            "description": "查询指定时间范围内按时间排序的所有灾害事件。返回每个事件的日期、类型、严重度。适用于'2025年8月发生了哪些极端事件'等问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "开始日期 YYYYMMDD"},
                    "end_date": {"type": "string", "description": "结束日期 YYYYMMDD"}
                },
                "required": ["start_date", "end_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_cascading_chain",
            "description": "追踪一个灾害事件的级联影响链：查询它引发了哪些次生事件（通过possiblyCauses/time:before关系）。适用于'这场山洪是否引发了后续灾害'等问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "string",
                        "description": "事件ID，如 event_20250819_flash_flood_001"
                    }
                },
                "required": ["event_id"]
            }
        }
    },

    # ═══════════════════════════════════════════════════
    # PROV-O Provenance Queries
    # ═══════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "query_provenance",
            "description": "查询指标的完整数据溯源链：它来自哪个数据源（ERA5/GPM/OSTIA）、由哪个传感器观测、通过什么步骤推导得出。适用于't2m_anomaly_c的数据来源是什么'等问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "indicator_id": {"type": "string", "description": "指标ID"}
                },
                "required": ["indicator_id"]
            }
        }
    },

    # ═══════════════════════════════════════════════════
    # Event Detection — Historical (ERA5 NetCDF)
    # ═══════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "detect_extreme_events",
            "description": "运行极端事件检测引擎，分析历史日期的灾害事件。基于ERA5再分析数据(indicators/目录)。适用于'2025-08-19发生了什么'等回顾分析。",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "日期 YYYYMMDD，如 20250819"},
                    "hazard_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]},
                        "description": "要检测的灾害类型列表，不传则检测全部四种"
                    }
                },
                "required": ["date"]
            }
        }
    },

    # ═══════════════════════════════════════════════════
    # Event Detection — Forecast (IFS global)
    # ═══════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "detect_future_events",
            "description": "基于 ECMWF IFS 全球预报(0.25°)检测未来几天的极端灾害风险。IFS 有完整大气变量覆盖全部四种灾害。适用于'明天会有山洪吗''未来3天利雅得会有极端高温吗'等预报问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "forecast_day": {
                        "type": "integer",
                        "description": "预报天数偏移（1=明天, 2=后天...最多7天），默认1"
                    },
                    "location": {
                        "type": "string",
                        "description": "关注地点，如'利雅得''红海沿岸''吉达'，可选"
                    },
                    "hazard_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]},
                        "description": "要检测的灾害类型列表，不传则检测全部四种"
                    }
                },
                "required": ["forecast_day"]
            }
        }
    },

    # ═══════════════════════════════════════════════════
    # Multi-Day Sequence Detection
    # ═══════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "detect_forecast_sequence",
            "description": "批量检测未来连续多天的灾害风险趋势。一次调用返回1-7天的逐日检测结果和趋势分析，避免多次调用。适用于'未来72小时趋势''本周风险变化'等问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_day": {"type": "integer", "description": "起始预报天数，默认1"},
                    "end_day": {"type": "integer", "description": "结束预报天数，默认3，最多7"},
                    "hazard_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]},
                        "description": "灾害类型列表，不传则检测全部四种"
                    }
                },
                "required": []
            }
        }
    },

    # ═══════════════════════════════════════════════════
    # Raw Indicator Value Query (validation / accuracy check)
    # ═══════════════════════════════════════════════════
    # City Weather (city → nearest grid cell → all indicators)
    # ═══════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "get_city_weather",
            "description": "获取指定城市在指定预报日的所有气象指标格点值。根据城市坐标找到最近的IFS网格点，返回温度、湿度、风、降水等可用指标的实际数值。适用于'利雅得今天多少度''吉达的湿度是多少'等问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，中文或英文，如 利雅得、吉达、达曼"},
                    "forecast_day": {"type": "integer", "description": "预报天数偏移，1=明天，默认1"}
                },
                "required": ["city"]
            }
        }
    },

    # ═══════════════════════════════════════════════════
    # City Lookup (Saudi city → coordinates)
    # ═══════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "lookup_city",
            "description": "根据城市名称查找沙特城市的经纬度坐标和所属区域。支持中文名（吉达、利雅得）和英文名（Jeddah、Riyadh）。适用于'利雅得今天的天气怎么样'等需要城市定位的问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，中文或英文"}
                },
                "required": ["city"]
            }
        }
    },

    # ═══════════════════════════════════════════════════
    # Composite Multi-Hazard Risk
    # ═══════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "detect_composite_risk",
            "description": "检测指定日期的复合灾害叠加风险。同时运行四种灾害检测，计算多灾种叠加评分和热点空间重叠度。适用于'有没有复合灾害风险''高温和沙尘会不会同时来'等问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "forecast_day": {"type": "integer", "description": "预报天数偏移，1=明天"},
                },
                "required": ["forecast_day"]
            }
        }
    },

    # ═══════════════════════════════════════════════════
    # Historical Comparison
    # ═══════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "compare_with_history",
            "description": "将当前 IFS 预报的灾害风险与2025年同期历史检测结果进行对比，评估今年相对于去年的异常程度。适用于'比去年热吗''今年沙尘风险是不是更高'等问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "forecast_day": {"type": "integer", "description": "预报天数偏移"},
                    "hazard_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]},
                        "description": "灾害类型列表"
                    }
                },
                "required": ["forecast_day"]
            }
        }
    },

    # ═══════════════════════════════════════════════════
    # RAG Knowledge Base Search
    # ═══════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "搜索 MAZU 知识库，检索气象指标的物理定义、检测规则条件、沙特气候特征、模型局限性等技术知识。返回相关文档片段及来源。适用于'这个指标是什么意思''检测规则的条件权重是多少''沙特沙漠气候有什么特征'等需要查阅文档的问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询，建议用中文关键词或自然语言问题，如 'tmax_c 怎么算的' '山洪检测条件权重' 'Shamal风的气候特征'"
                    },
                    "k": {
                        "type": "integer",
                        "description": "返回结果数量，默认 3，最多 8"
                    }
                },
                "required": ["query"]
            }
        }
    },

    # ═══════════════════════════════════════════════════
    # Event Detection - IFS (ECMWF downloaded forecast)
    # ═══════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "detect_ifs_forecast",
            "description": "基于 ECMWF IFS 全球预报(0.25deg)检测未来灾害风险。IFS有完整大气变量覆盖全部四种灾害。",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "IFS 初始日期 YYYYMMDD。留空则用最近可用日期。"},
                    "forecast_day": {"type": "integer", "description": "预报天数偏移(0=分析场, 1=明天), 默认0"},
                    "location": {"type": "string", "description": "关注地点, 可选"},
                    "hazard_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]},
                        "description": "灾害类型列表, 不传则检测全部"
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
    {
        "type": "function",
        "function": {
            "name": "get_city_hazards",
            "description": "【城市天气查询首选】基于 IFS 预报对指定城市进行格点级灾害检测，返回四类灾害的严重度和逐条件判定结果。直接给出权威检测结论。当用户问某个城市今天天气或有没有灾害风险时，应优先调用此工具而非 get_city_weather。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名(中文或英文), 如 '麦加'/'mecca', '吉达'/'jeddah', '利雅得'/'riyadh'"},
                    "date": {"type": "string", "description": "IFS 初始日期 YYYYMMDD。留空则用最近可用日期。"},
                    "hour": {"type": "integer", "description": "预报时次(0/12/24/...), 默认12(下午,最热时段)。0=午夜分析场。"},
                    "hazard_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]},
                        "description": "灾害类型列表, 不传则检测全部四种"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_calibrated_city_hazards",
            "description": "【KG增强版城市灾害查询 - 推荐首选】基于IFS预报检测城市灾害风险，并用KG历史事件目录校准置信度。返回四类灾害的严重度+历史基准触发率+校准置信度(high/medium/low)。比get_city_hazards多一层KG历史验证。当用户问城市天气或灾害风险时优先使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名(中文或英文), 如 '麦加'/'mecca'"},
                    "date": {"type": "string", "description": "IFS初始日期 YYYYMMDD。留空则用最近可用日期。"},
                    "hour": {"type": "integer", "description": "预报时次(0/12/24/...), 默认12(下午,最热时段)"},
                    "hazard_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]},
                        "description": "灾害类型列表, 不传则检测全部四种"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "assess_forecast_reliability",
            "description": "用KG历史事件目录校验IFS全国区域预报检测的可靠性。对比当前预报与365天历史事件模式，返回校准置信度和异常标记。适用于detect_ifs_forecast后的全域可靠性验证。",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "IFS初始日期 YYYYMMDD。留空则用最近可用日期。"},
                    "forecast_day": {"type": "integer", "description": "预报天数偏移(0=分析场), 默认0"},
                    "hazard_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]},
                        "description": "灾害类型列表"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_historical_analogs",
            "description": "在KG历史事件目录中查找与指定条件最相似的过去日期。返回类似日期的灾害发生情况和条件相似度。适用于'历史上类似情况下发生了什么'等回溯验证问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "hazard_type": {
                        "type": "string",
                        "enum": ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"],
                        "description": "灾害类型"
                    },
                    "lat": {"type": "number", "description": "纬度"},
                    "lon": {"type": "number", "description": "经度"},
                    "date": {"type": "string", "description": "目标日期 YYYYMMDD"},
                    "n_analogs": {"type": "integer", "description": "返回数量, 默认5"}
                },
                "required": ["hazard_type", "lat", "lon", "date"]
            }
        }
    },

]


# ═══════════════════════════════════════════════════════
# Tool Dispatch — routes tool calls to KG backend
# ═══════════════════════════════════════════════════════

class ToolDispatcher:
    """Dispatches LLM function calls to the KG API endpoints."""

    def __init__(self, api_base="http://127.0.0.1:5000"):
        self.api_base = api_base
        import requests
        self.requests = requests

    def dispatch(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call and return result as JSON string."""
        t0 = _time.perf_counter()
        args_preview = json.dumps(arguments, ensure_ascii=False)
        if len(args_preview) > 200:
            args_preview = args_preview[:200] + "…"
        log.info("→ %s(%s)", tool_name, args_preview)

        try:
            if tool_name == "query_hazard_indicators":
                result = self._get(f"/api/sparql/hazard/{arguments['hazard_type']}")

            elif tool_name == "query_indicator_detail":
                result = self._get(f"/api/sparql/indicator/{arguments['indicator_id']}")

            elif tool_name == "query_indicator_chain":
                result = self._get(f"/api/sparql/chain/{arguments['indicator_id']}")

            elif tool_name == "search_indicators":
                result = self._get(f"/api/sparql/search?q={arguments['keyword']}")

            elif tool_name == "query_rule_detail":
                result = self._get(f"/api/sparql/rule/{arguments['rule_id']}")

            elif tool_name == "query_observations_nearby":
                params = {k: v for k, v in arguments.items() if v is not None}
                qs = "&".join(f"{k}={v}" for k, v in params.items())
                result = self._get(f"/api/sparql/geospatial/radius?{qs}")

            elif tool_name == "query_events_in_region":
                result = self._get(f"/api/sparql/geospatial/intersects?region={arguments['region']}")

            elif tool_name == "query_event_timeline":
                sd = arguments.get("start_date", "")
                ed = arguments.get("end_date", "")
                result = self._get(f"/api/sparql/temporal/timeline?start={sd}&end={ed}")

            elif tool_name == "query_cascading_chain":
                result = self._get(f"/api/sparql/temporal/cascade/{arguments['event_id']}")

            elif tool_name == "query_provenance":
                result = self._get(f"/api/sparql/provenance/indicator/{arguments['indicator_id']}")

            elif tool_name == "detect_extreme_events":
                body = {"date": arguments["date"]}
                if "hazard_types" in arguments:
                    body["hazard_type"] = arguments["hazard_types"][0] if arguments["hazard_types"] else None
                r = self.requests.post(f"{self.api_base}/api/detect", json=body)
                result = json.dumps(r.json(), ensure_ascii=False, indent=2)

            elif tool_name == "detect_ifs_forecast":
                result = self._detect_from_ifs(arguments)
            elif tool_name == "detect_future_events":
                result = self._detect_from_ifs(arguments)

            elif tool_name == "detect_forecast_sequence":
                result = self._detect_sequence(arguments)

            elif tool_name == "compare_with_history":
                result = self._compare_with_history(arguments)

            elif tool_name == "search_knowledge_base":
                result = self._search_kb(arguments)

            elif tool_name == "get_city_weather":
                result = self._get_city_weather(arguments)

            elif tool_name == "list_ifs_dates":
                result = self._list_ifs_dates(arguments)
            elif tool_name == "get_city_hazards":
                result = self._get_city_hazards(arguments)
            elif tool_name == "get_calibrated_city_hazards":
                result = self._get_calibrated_city_hazards(arguments)
            elif tool_name == "assess_forecast_reliability":
                result = self._assess_forecast_reliability(arguments)
            elif tool_name == "query_historical_analogs":
                result = self._query_historical_analogs_tool(arguments)
            elif tool_name == "lookup_city":
                result = self._lookup_city_tool(arguments)

            elif tool_name == "detect_composite_risk":
                result = self._detect_composite_risk(arguments)

            else:
                result = json.dumps({"error": f"Unknown tool: {tool_name}"})

            elapsed_ms = (_time.perf_counter() - t0) * 1000
            log.info("← %s  ok  %d ms  %d chars  preview: %.120s",
                     tool_name, int(elapsed_ms), len(result),
                     result.replace("\n", " ").strip())
            return result

        except Exception as e:
            elapsed_ms = (_time.perf_counter() - t0) * 1000
            log.error("← %s  FAIL  %d ms  %s: %s\n%s",
                      tool_name, int(elapsed_ms),
                      type(e).__name__, str(e)[:200],
                      _traceback.format_exc()[-500:])
            return json.dumps({"error": str(e), "tool": tool_name})

    # ═══════════════════════════════════════════════════════
    # IFS forecast indicator computation (legacy FCN loader — not used in IFS path)
    # ═══════════════════════════════════════════════════════

    def _load_indicators_fcn(self, nc_path: str, forecast_day: int) -> dict:
        """[DEPRECATED] Load FCN NetCDF → indicator arrays. No longer used; IFS is the primary data source. Selects best lead_time for forecast_day."""
        import numpy as np, xarray as xr

        ds = xr.open_dataset(nc_path)
        lat = ds['lat'].values
        lon = ds['lon'].values
        ind = {}
        missing = []

        # Pick lead_time index closest to forecast_day * 24 h
        lead_hours = ds['lead_time'].values / 3_600_000_000_000  # ns → hours
        target_h = forecast_day * 24
        lt_idx = int(np.argmin(np.abs(lead_hours - target_h)))
        actual_h = int(lead_hours[lt_idx])

        def _pick(arr):
            """Extract 2D slice from (time, lead_time, lat, lon) array."""
            if arr.ndim >= 3:
                return arr[0, lt_idx, :, :]
            return arr

        if 't2m' in ds.variables:
            ind['t2m_c'] = _pick(ds['t2m'].values) - 273.15
            ind['tmax_c'] = ind['t2m_c']  # forecast 2m temp ≈ daily max
        else:
            missing.append('t2m_c')
            missing.append('tmax_c')

        if 'u10m' in ds.variables and 'v10m' in ds.variables:
            u10 = _pick(ds['u10m'].values)
            v10 = _pick(ds['v10m'].values)
            ind['wind10_speed'] = np.sqrt(u10**2 + v10**2)
            # Meteorological wind direction (direction wind comes FROM)
            ind['wind_direction'] = (np.arctan2(-u10, -v10) * 180.0 / np.pi) % 360
        else:
            missing.append('wind10_speed')
            missing.append('wind_direction')

        if 'tcwv' in ds.variables:
            ind['pwat'] = _pick(ds['tcwv'].values)
        else:
            missing.append('pwat')

        if 'sp' in ds.variables:
            ind['sp'] = _pick(ds['sp'].values)

        # Wind shear 850-200 hPa
        if all(v in ds.variables for v in ['u850', 'v850', 'u250', 'v250']):
            du = _pick(ds['u850'].values) - _pick(ds['u250'].values)
            dv = _pick(ds['v850'].values) - _pick(ds['v250'].values)
            ind['wind_shear_850_200'] = np.sqrt(du**2 + dv**2)
        else:
            missing.append('wind_shear_850_200')

        # ═══════════════════════════════════════════════════
        # Moisture & Precipitation Proxy (FCN atmospheric state — legacy, not used in IFS path)
        #
        # FCN has no direct precipitation output (legacy — IFS has full variables)
        # physically constrained by: water vapor × convergence × saturation.
        # We derive a proxy from:
        #   tcwv  → total column water vapor (kg/m² ≈ mm)
        #   r850  → RH at 850hPa (%)
        #   t850  → T at 850hPa (K)
        #   u850, v850 → horizontal wind at 850hPa (m/s)
        #
        # 1. q850 = specific humidity from t850 + r850 (Magnus formula)
        # 2. Moisture flux: F_u = q850·u850, F_v = q850·v850
        # 3. Moisture flux convergence: MFC = -∇·(F_u, F_v)  [1/s]
        # 4. Precip proxy: tcwv × max(0, MFC) × 86400s × (r850/100)²
        #    → physical units: kg/m²/day ≈ mm/day
        # ═══════════════════════════════════════════════════

        has_full_moisture = all(v in ds.variables for v in ['r850', 't850', 'u850', 'v850'])
        has_basic_moisture = all(v in ds.variables for v in ['r850', 't850'])
        has_tcwv = 'tcwv' in ds.variables

        if has_full_moisture:
            r850_val = _pick(ds['r850'].values)
            t850_val = _pick(ds['t850'].values)
            u850_val = _pick(ds['u850'].values)
            v850_val = _pick(ds['v850'].values)

            # -- Specific humidity at 850hPa (kg/kg) --
            t850_c = t850_val - 273.15
            es_850 = 6.112 * np.exp(17.67 * t850_c / (t850_c + 243.5))  # hPa
            e_850  = es_850 * np.clip(r850_val, 0.1, 100.0) / 100.0
            q850   = 0.622 * e_850 / (850.0 - 0.378 * e_850)  # kg/kg
            ind['specific_humidity_850'] = q850 * 1000.0  # g/kg

            # -- Dewpoint depression at 850hPa --
            ln_e = np.log(np.maximum(e_850 / 6.112, 1e-10))
            td850_c = 243.5 * ln_e / (17.67 - ln_e)
            ind['dewpoint_depression_c'] = t850_c - td850_c
            ind['rh2m'] = r850_val  # proxy

            # -- Moisture flux convergence --
            F_u = q850 * u850_val  # kg/kg · m/s
            F_v = q850 * v850_val

            # Grid spacing in physical meters
            R = 6371000.0
            lat_rad = np.deg2rad(lat)
            dlat_deg = 0.25
            dlon_deg = 0.25
            m_per_deg_lat = np.deg2rad(dlat_deg) * R  # ~27830 m, scalar
            m_per_deg_lon = np.deg2rad(dlon_deg) * R * np.cos(lat_rad)  # (nlat,) array

            dFu_dlon = np.gradient(F_u, dlon_deg, axis=1)  # per degree lon
            dFv_dlat = np.gradient(F_v, dlat_deg, axis=0)  # per degree lat
            dFu_dx = dFu_dlon / m_per_deg_lon[:, np.newaxis]  # per meter
            dFv_dy = dFv_dlat / m_per_deg_lat               # per meter
            MFC = -(dFu_dx + dFv_dy)  # moisture flux convergence [1/s]
            MFC_pos = np.maximum(MFC, 0.0)

            # IVT convergence (scaled for rule threshold compatibility)
            ind['ivt_convergence'] = MFC * 1e6  # scale to ~0.001-0.1 range

            # -- Precipitation proxy (mm/day) --
            if has_tcwv:
                tcwv_val = _pick(ds['tcwv'].values)
                sat_factor = (r850_val / 100.0) ** 2
                # Physical basis: precip = tcwv × MFC × time × saturation
                # The 5x calibration accounts for convergence above 850hPa
                # (850hPa layer captures ~20% of column-integrated MFC)
                precip_proxy = tcwv_val * MFC_pos * 86400.0 * sat_factor * 5.0
                ind['daily_precip_total'] = np.clip(precip_proxy, 0.0, 200.0)
            else:
                missing.append('daily_precip_total')

        elif has_basic_moisture:
            # Fallback: moisture vars exist but t850/u850/v850 missing
            r850_val = _pick(ds['r850'].values)
            t850_val = _pick(ds['t850'].values)
            t850_c = t850_val - 273.15
            es = 6.112 * np.exp(17.67 * t850_c / (t850_c + 243.5))
            e  = es * np.clip(r850_val, 0.1, 100.0) / 100.0
            ln_e = np.log(np.maximum(e / 6.112, 1e-10))
            td850_c = 243.5 * ln_e / (17.67 - ln_e)
            ind['dewpoint_depression_c'] = t850_c - td850_c
            ind['rh2m'] = r850_val
            missing.extend(['daily_precip_total', 'ivt_convergence'])
        else:
            missing.extend(['dewpoint_depression_c', 'rh2m',
                           'daily_precip_total', 'ivt_convergence'])

        ds.close()

        return {
            "indicators": ind, "lat": lat, "lon": lon, "missing": missing,
            "lead_time_h": actual_h, "lead_time_idx": lt_idx,
        }

    @staticmethod
    def _build_region_threshold_map(lat_vals, lon_vals, region_calib, htype, ind_id, base_th):
        """Build a 2D threshold map with per-cell region calibration applied."""
        import numpy as np
        th_map = np.full((len(lat_vals), len(lon_vals)), base_th, dtype=float)

        if region_calib is None:
            return th_map

        for rid, rdata in region_calib.get("regions", {}).items():
            rlat = rdata["lat"]
            rlon = rdata["lon"]
            cal = rdata.get("calibration", {}).get(htype, {})
            offset = cal.get(ind_id, {}).get("threshold_offset", 0)
            if offset == 0:
                continue
            mask_lat = (lat_vals >= rlat[0]) & (lat_vals <= rlat[1])
            mask_lon = (lon_vals >= rlon[0]) & (lon_vals <= rlon[1])
            for i in np.where(mask_lat)[0]:
                for j in np.where(mask_lon)[0]:
                    th_map[i, j] = base_th + offset
        return th_map

    @staticmethod
    @staticmethod
    def _get_coastal_mask(lat_vals, lon_vals):
        """Return boolean mask for Red Sea + Persian Gulf coastal grid cells."""
        import numpy as np
        nlat, nlon = len(lat_vals), len(lon_vals)
        mask = np.zeros((nlat, nlon), dtype=bool)
        for i in range(nlat):
            for j in range(nlon):
                lat, lon = lat_vals[i], lon_vals[j]
                in_red_sea = (16 <= lat <= 30) and (34 <= lon <= 42)
                in_gulf = (24 <= lat <= 30) and (48 <= lon <= 56)
                mask[i, j] = in_red_sea or in_gulf
        return mask

    @staticmethod
    def _run_hazard_detection(indicators: dict, lat_vals, lon_vals,
                               rules_data: dict, hazard_types: list,
                               region_calib: dict = None,
                               coastal_mask=None) -> list:
        """Core detection: weighted scoring + primary gate + prob gate bypass + region calibration.

        2026-07-21 fixes:
        - Probabilistic gate bypass: prob gate trigger overrides primary gate penalty
        - Coastal filter: coastal_humid_heat only evaluated on Red Sea / Persian Gulf cells
        """
        import numpy as np

        # Build coastal mask lazily (once per call, shared across hazard types)
        if coastal_mask is None:
            coastal_mask = ToolDispatcher._get_coastal_mask(lat_vals, lon_vals)

        results = []
        for htype in hazard_types:
            rule = next((r for r in rules_data["rules"] if r["hazard_type"] == htype), None)
            if not rule:
                continue

            available_conds = []
            unavailable_conds = []
            for cond in rule["conditions"]:
                if cond["indicator"] in indicators:
                    available_conds.append(cond)
                else:
                    unavailable_conds.append(cond["indicator"])

            if len(available_conds) < 2:
                results.append({
                    "hazard_type": htype,
                    "detected": False,
                    "coverage": f"{len(available_conds)}/{len(rule['conditions'])}",
                    "reason": f"insufficient_indicators",
                })
                continue

            ref_arr = list(indicators.values())[0]
            score = np.zeros(ref_arr.shape, dtype=float)
            total_w = 0.0
            all_conds = len(rule["conditions"])  # total conditions in rule
            available_n = len(available_conds)
            triggered = []

            for cond in available_conds:
                ind_id = cond["indicator"]
                op = cond.get("op", cond.get("condition", ">="))
                base_th = cond["value"]
                w = cond["weight"]
                data = indicators[ind_id]

                if data.shape != score.shape:
                    continue

                # Apply region calibration to threshold
                th_map = ToolDispatcher._build_region_threshold_map(
                    lat_vals, lon_vals, region_calib, htype, ind_id, base_th)

                if op in (">=", ">"):
                    # Intensity scaling: how far beyond threshold relative to threshold
                    # e.g. precip=50mm vs th=10mm → exceedance=4 → capped at 1.0
                    exceedance = np.clip((data - th_map) / (np.abs(th_map) * 0.5 + 1e-6), 0.0, 1.0)
                    hit = exceedance >= 0.0
                    contribution = w * exceedance  # partial score based on intensity
                elif op in ("<", "<="):
                    # For "less than" conditions, reverse: below threshold = higher score
                    margin = np.clip((th_map - data) / (np.abs(th_map) * 0.5 + 1e-6), 0.0, 1.0)
                    contribution = w * margin
                    hit = data <= th_map
                else:
                    continue

                score += contribution
                total_w += w
                n_hit = int((exceedance > 0.3).sum() if op in (">=", ">") else (margin > 0.3).sum())
                peak = float(np.nanmax(data))
                th_min, th_max = float(th_map.min()), float(th_map.max())
                th_str = f"{op} {base_th}" if th_min == th_max else f"{op} {th_min:.0f}-{th_max:.0f}(区域校准)"
                if n_hit > 0:
                    triggered.append({
                        "indicator": ind_id,
                        "condition": th_str,
                        "base_threshold": base_th,
                        "calibrated_range": [th_min, th_max] if th_min != th_max else None,
                        "cells_triggered": n_hit,
                        "peak_value": round(peak, 2),
                    })

            if total_w > 0:
                score = score / total_w

            # Coverage penalty: partially available indicators → cap max score
            coverage_ratio = available_n / all_conds if all_conds > 0 else 1.0
            if coverage_ratio < 1.0:
                # Soft penalty: score * sqrt(coverage) so 3/6 → 0.707 not 0.5
                coverage_penalty = np.sqrt(coverage_ratio)
                score = score * coverage_penalty

            # ── Primary gate + Probabilistic gate bypass ──
            primary_cond = next((c for c in rule["conditions"] if c.get("primary")), None)
            prob_gate_cond = next((c for c in rule["conditions"] if c.get("role") == "probabilistic_gate"), None)

            primary_hit = None
            prob_gate_hit = None

            if primary_cond and primary_cond["indicator"] in indicators:
                pdata = indicators[primary_cond["indicator"]]
                if pdata.shape == score.shape:
                    pth_map = ToolDispatcher._build_region_threshold_map(
                        lat_vals, lon_vals, region_calib, htype,
                        primary_cond["indicator"], primary_cond["value"])
                    primary_hit = pdata >= pth_map

            if prob_gate_cond and prob_gate_cond["indicator"] in indicators:
                gdata = indicators[prob_gate_cond["indicator"]]
                if gdata.shape == score.shape:
                    gth_map = ToolDispatcher._build_region_threshold_map(
                        lat_vals, lon_vals, region_calib, htype,
                        prob_gate_cond["indicator"], prob_gate_cond["value"])
                    # Handle both >= and <= operators for prob gate
                    gop = prob_gate_cond.get("op", ">=")
                    if gop in (">=", ">"):
                        prob_gate_hit = gdata >= gth_map
                    else:
                        prob_gate_hit = gdata <= gth_map

            if primary_hit is not None:
                # Cells where primary NOT met → suppress ×0.25
                # UNLESS prob gate triggers (bypass)
                not_primary = ~primary_hit
                if prob_gate_hit is not None:
                    bypass = prob_gate_hit
                    suppress = not_primary & (~bypass)
                else:
                    suppress = not_primary
                score[suppress] *= 0.25
            elif primary_cond:
                # Primary gate unavailable → confidence penalty
                score = score * 0.7

            # ── Coastal region filter ──
            if htype == "coastal_humid_heat":
                score[~coastal_mask] = 0.0

            max_score = float(np.nanmax(score))
            n_total = int(coastal_mask.sum()) if htype == "coastal_humid_heat" else score.size
            n_risky = int((score >= 0.2).sum())

            if np.isfinite(max_score):
                peak_idx = np.unravel_index(np.nanargmax(score), score.shape)
                risky_lat = float(lat_vals[peak_idx[0]])
                risky_lon = float(lon_vals[peak_idx[1]])
            else:
                risky_lat = risky_lon = 0.0

            sev = ("extreme" if max_score >= 0.6 else "high" if max_score >= 0.4
                   else "medium" if max_score >= 0.2 else "low")

            results.append({
                "hazard_type": htype,
                "detected": n_risky > 0,
                "max_risk_score": round(max_score, 3),
                "severity": sev,
                "grid_cells_at_risk": n_risky,
                "total_cells": n_total,
                "hotspot_lat": round(risky_lat, 1),
                "hotspot_lon": round(risky_lon, 1),
                "triggered_conditions": triggered,
                "unavailable_indicators": unavailable_conds,
                "coverage": f"{len(available_conds)}/{len(rule['conditions'])}",
                "region_calibrated": region_calib is not None,
            })

        return results

    @staticmethod
    @staticmethod
    def _fcns_physical_consistency_check(indicators: dict, operators: list) -> dict:
        """KG物理约束验证——检查FCN预报的内部物理自洽性。

        例如: 高温预报应伴随高露点差(干燥大气)和低相对湿度。
        返回每种灾害类型的物理一致性评分。
        """
        import numpy as np

        PHYSICS_RULES = {
            "extreme_heat": [
                ("t2m_c", "dewpoint_depression_c", "positive",
                 "高温→高露点差(干燥大气→夜间降温弱→热浪持续)"),
                ("t2m_c", "rh2m", "negative",
                 "高温→低湿(干热vs湿热区分)"),
            ],
            "flash_flood": [
                ("pwat", "daily_precip_total", "positive",
                 "高可降水量→高降水(水汽凝结)"),
                ("pwat", "ivt_convergence", "positive",
                 "高水汽含量→水汽辐合增强"),
            ],
            "dust_storm": [
                ("wind10_speed", "dewpoint_depression_c", "positive",
                 "强风+干燥→起尘"),
                ("dewpoint_depression_c", "rh2m", "negative",
                 "高露点差⇔低RH(等价干燥度指标)"),
            ],
            "coastal_humid_heat": [
                ("t2m_c", "rh2m", "positive",
                 "湿热=高温+高湿同时出现"),
                ("rh2m", "dewpoint_depression_c", "negative",
                 "高RH⇔低露点差(数学关联)"),
            ],
        }

        results = {}
        for htype, rules in PHYSICS_RULES.items():
            checks = []
            passed = 0
            total = 0
            for var_a, var_b, expected_sign, explanation in rules:
                if var_a not in indicators or var_b not in indicators:
                    continue
                total += 1
                a = indicators[var_a].ravel()
                b = indicators[var_b].ravel()
                valid = np.isfinite(a) & np.isfinite(b)
                if valid.sum() < 10:
                    checks.append({"variables": f"{var_a} vs {var_b}",
                                   "note": "insufficient valid data"})
                    continue
                corr = np.corrcoef(a[valid], b[valid])[0, 1]
                is_coherent = bool((expected_sign == "positive" and corr > 0) or
                                 (expected_sign == "negative" and corr < 0))
                checks.append({
                    "variables": f"{var_a} ↔ {var_b}",
                    "expected_sign": expected_sign,
                    "correlation": round(float(corr), 3),
                    "coherent": is_coherent,
                    "meaning": explanation,
                })
                if is_coherent:
                    passed += 1

            score = round(passed / total, 2) if total > 0 else None
            results[htype] = {
                "physical_consistency_score": score,
                "checks_passed": f"{passed}/{total}",
                "details": checks,
                "assessment": (
                    "物理自洽 ✓" if score and score >= 0.5
                    else "部分自洽 ⚠" if score and score > 0
                    else "无法评估" if score is None
                    else "物理不一致 ✗ — 建议查询 KG 交叉验证"
                ),
            }

        return results

    # ═══════════════════════════════════════════════════════
    # Main forecast detection — IFS-based (was FCN-only)
    # ═══════════════════════════════════════════════════════

    def _detect_from_forecast(self, args: dict) -> str:
        """[DEPRECATED] 纯 FCN 本地 GPU 预报 + KG 物理一致性验证。No longer called; _detect_from_ifs is the active path."""
        import numpy as np, os

        forecast_day = args.get("forecast_day", 1)
        hazard_types = args.get("hazard_types", None)
        location = args.get("location", None)

        if hazard_types is None:
            hazard_types = ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]

        project_dir = os.path.dirname(os.path.abspath(__file__))
        schema_dir = os.path.join(project_dir, "schema")
        forecast_dir = os.path.join(project_dir, "forecast")

        # ── 1. Load FCN forecast (DEPRECATED — IFS path is active) ──
        fcn_path = os.path.join(forecast_dir, "fcn_forecast.nc")
        if not os.path.exists(fcn_path):
            return json.dumps({
                "error": "FCN 预报文件不存在（已弃用，请使用 IFS）",
                "hint": "请在 WSL2 中运行: python run_fcn.py --days 7",
                "expected_path": fcn_path,
            }, ensure_ascii=False)

        fcn_data = self._load_indicators_fcn(fcn_path, forecast_day)

        # ── 2. Load rules + operators ──
        with open(os.path.join(schema_dir, "rules.json"), "r", encoding="utf-8") as f:
            rules_data = json.load(f)
        with open(os.path.join(schema_dir, "operators.json"), "r", encoding="utf-8") as f:
            ops_data = json.load(f)

        # Load region calibration
        region_calib = None
        calib_path = os.path.join(schema_dir, "region_calibration.json")
        if os.path.exists(calib_path):
            with open(calib_path, "r", encoding="utf-8") as f:
                region_calib = json.load(f)

        # ── 3. Run hazard detection (with region calibration) ──
        hazards = self._run_hazard_detection(
            fcn_data["indicators"], fcn_data["lat"], fcn_data["lon"],
            rules_data, hazard_types, region_calib)
        hazards = self._tag_hazards_with_region(hazards)

        # ── 3b. Post-processing: Shamal flag, isolated convection, region report ──
        post = self._post_process_detection(
            hazards, fcn_data["indicators"], fcn_data["lat"], fcn_data["lon"])

        # ── 4. KG physical consistency check ──
        kg_checks = self._fcns_physical_consistency_check(
            fcn_data["indicators"], ops_data.get("operators", []))

        # ── 5. Composite risk scoring ──
        composite = self._compute_composite_risk(
            hazards, fcn_data["indicators"], fcn_data["lat"], fcn_data["lon"],
            region_calib)

        # ── 6. Build output ──
        output = {
            "forecast_source": "ECMWF IFS 0.25deg",
            "forecast_day": forecast_day,
            "lead_time_h": fcn_data.get("lead_time_h", "?"),
            "available_indicators": sorted(fcn_data["indicators"].keys()),
            "missing_indicators": fcn_data.get("missing", []),
            "hazards": hazards,
            "composite_risk": composite,
            "region_report": post.get("region_report", {}),
            "shamal_flag": post.get("shamal_flag", False),
            "shamal_detail": post.get("shamal_detail"),
            "isolated_convection": post.get("isolated_convection"),
            "kg_physical_consistency": kg_checks,
            "region_calibrated": region_calib is not None,
        }

        # ── 6. Location context ──
        if location:
            for h in hazards:
                if h.get("detected"):
                    h["location_note"] = (
                        f"热点 ({h['hotspot_lat']}N, {h['hotspot_lon']}E) — "
                        f"使用 query_observations_nearby 在 {location} 周边精确搜索"
                    )

        # ── 7. Synthesis ──
        inconsistent = [htype for htype, c in kg_checks.items()
                       if c.get("assessment", "").startswith("物理不一致")]
        partial = [htype for htype, c in kg_checks.items()
                   if c.get("assessment", "").startswith("部分自洽")]
        if inconsistent:
            output["synthesis"] = {
                "verdict": f"⚠ {len(inconsistent)} 类灾害物理不一致: {inconsistent}",
                "confidence": "low",
                "recommendation": (
                    "IFS 预报中存在物理不一致的指标关系。建议: "
                    "1) 使用 query_indicator_detail 检查相关指标的物理含义; "
                    "2) 用 query_hazard_indicators 确认灾害依赖的完整指标链; "
                    "3) 结合 detect_extreme_events 查看历史相似日期作为参考。"
                ),
            }
        elif partial:
            output["synthesis"] = {
                "verdict": f"IFS 预报物理一致性可接受 ({len(partial)} 类部分自洽)",
                "confidence": "medium",
                "recommendation": (
                    "部分指标关系未完全满足物理预期，但整体可接受。"
                    "查询 KG 中的 co_occurs_with 关系可了解指标间的物理关联。"
                ),
            }
        else:
            output["synthesis"] = {
                "verdict": "IFS 预报物理一致性良好",
                "confidence": "high",
                "recommendation": (
                    "IFS 各指标间物理关系自洽。可使用 query_indicator_chain "
                    "追溯任意指标的计算链，或 query_rule_detail 查看检测规则详情。"
                ),
            }

        return json.dumps(output, ensure_ascii=False, indent=2)

    # ═══════════════════════════════════════════════════════
    # Region tagging
    # ═══════════════════════════════════════════════════════

    SAUDI_REGIONS = {
        "red_sea":        {"lat": (16.0, 30.0), "lon": (34.0, 44.0), "label": "红海沿岸"},
        "persian_gulf":   {"lat": (24.0, 30.0), "lon": (48.0, 56.0), "label": "波斯湾沿岸"},
        "empty_quarter":  {"lat": (17.0, 23.0), "lon": (45.0, 56.0), "label": "鲁布哈利沙漠"},
        "central_desert": {"lat": (21.0, 28.0), "lon": (40.0, 48.0), "label": "中部沙漠(利雅得)"},
        "north":          {"lat": (28.0, 32.0), "lon": (34.0, 48.0), "label": "北部(塔布克/焦夫)"},
        "south_asir":     {"lat": (16.0, 21.0), "lon": (40.0, 46.0), "label": "南部阿西尔山脉"},
    }

    SAUDI_CITIES = {
        # 西部红海沿岸（真正沿海，距海 <50km）
        "jeddah":    {"lat": 21.54, "lon": 39.17, "label": "吉达", "region": "red_sea"},
        "yanbu":     {"lat": 24.09, "lon": 38.06, "label": "延布", "region": "red_sea"},
        "rabigh":    {"lat": 22.80, "lon": 39.03, "label": "拉比格", "region": "red_sea"},
        # 西部内陆（Hijaz 山脉以东，距海 >50km，非沿海）
        "mecca":     {"lat": 21.39, "lon": 39.86, "label": "���", "region": "central_saudi"},
        "medina":    {"lat": 24.47, "lon": 39.61, "label": "���地那", "region": "central_saudi"},
        # 东部波斯湾沿岸
        "dammam":    {"lat": 26.42, "lon": 50.10, "label": "达曼", "region": "persian_gulf"},
        "jubail":    {"lat": 26.96, "lon": 49.57, "label": "朱拜勒", "region": "persian_gulf"},
        "dhahran":   {"lat": 26.30, "lon": 50.13, "label": "宰赫兰", "region": "persian_gulf"},
        "khobar":    {"lat": 26.29, "lon": 50.21, "label": "胡拜尔", "region": "persian_gulf"},
        "ras_tanura":{"lat": 26.70, "lon": 50.10, "label": "拉斯坦努拉", "region": "persian_gulf"},
        "qatif":     {"lat": 26.56, "lon": 49.99, "label": "盖提夫", "region": "persian_gulf"},
        # 中部
        "riyadh":    {"lat": 24.71, "lon": 46.68, "label": "利雅得", "region": "central_desert"},
        "buraidah":  {"lat": 26.33, "lon": 43.97, "label": "布赖代", "region": "central_desert"},
        "hail":      {"lat": 27.52, "lon": 41.69, "label": "哈伊勒", "region": "central_desert"},
        # 北部
        "tabuk":     {"lat": 28.40, "lon": 36.57, "label": "塔布克", "region": "north"},
        "aljawf":    {"lat": 29.50, "lon": 39.58, "label": "焦夫", "region": "north"},
        "arar":      {"lat": 30.98, "lon": 41.04, "label": "阿尔阿尔", "region": "north"},
        # 南部
        "abha":      {"lat": 18.22, "lon": 42.51, "label": "艾卜哈", "region": "south_asir"},
        "khamis":    {"lat": 18.30, "lon": 42.73, "label": "海米斯穆谢特", "region": "south_asir"},
        "jizan":     {"lat": 16.89, "lon": 42.55, "label": "吉赞", "region": "south_asir"},
        "najran":    {"lat": 17.49, "lon": 44.13, "label": "奈季兰", "region": "south_asir"},
        # 西南沙漠
        "sharorah":  {"lat": 17.49, "lon": 47.11, "label": "沙鲁拉", "region": "empty_quarter"},
    }

    @classmethod
    def _tag_region(cls, lat: float, lon: float) -> str:
        """Return region label for a coordinate."""
        for rid, r in cls.SAUDI_REGIONS.items():
            if r["lat"][0] <= lat <= r["lat"][1] and r["lon"][0] <= lon <= r["lon"][1]:
                return r["label"]
        return "未知区域"

    @classmethod
    def _lookup_city(cls, query: str) -> dict:
        """Find city by name (Chinese or English). Returns {lat, lon, label, region} or None."""
        q = query.strip().lower()
        # Direct match
        for cid, cdata in cls.SAUDI_CITIES.items():
            if q == cid or q == cdata["label"]:
                return cdata
        # Fuzzy match
        for cid, cdata in cls.SAUDI_CITIES.items():
            if q in cid or q in cdata["label"] or cid in q or cdata["label"] in q:
                return cdata
        return None

    @classmethod
    def _nearest_city(cls, lat: float, lon: float, max_dist_deg: float = 3.0) -> dict:
        """Find nearest city within max_dist_deg (approx 330km at equator)."""
        import numpy as np
        best = None
        best_dist = max_dist_deg
        for cid, cdata in cls.SAUDI_CITIES.items():
            d = np.sqrt((lat - cdata["lat"])**2 + (lon - cdata["lon"])**2)
            if d < best_dist:
                best_dist = d
                best = dict(cdata, distance_deg=round(float(d), 1))
        return best

    @classmethod
    def _tag_hazards_with_region(cls, hazards: list) -> list:
        """Add region field to each detected hazard."""
        for h in hazards:
            if h.get("detected") and "hotspot_lat" in h:
                h["region"] = cls._tag_region(h["hotspot_lat"], h["hotspot_lon"])
                city = cls._nearest_city(h["hotspot_lat"], h["hotspot_lon"])
                if city:
                    h["nearest_city"] = city["label"]
                    h["nearest_city_dist_deg"] = city["distance_deg"]
        return hazards

    # ═══════════════════════════════════════════════════════
    # detect_forecast_sequence — batch multi-day detection
    # ═══════════════════════════════════════════════════════

    def _detect_from_ifs(self, args: dict) -> str:
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

    def _get_city_hazards(self, args: dict) -> str:
        """City-level IFS hazard detection — authoritative severity per city grid cell."""
        if not HAS_IFS:
            return json.dumps({"error": "IFS pipeline not installed"}, ensure_ascii=False)

        import os, numpy as np, xarray as xr
        from ifs_pipeline import evaluate_city_hazards, list_ifs_dates

        city_name = args.get("city", "")
        hour = args.get("hour", 12)  # default 12Z = afternoon peak
        hazard_types = args.get("hazard_types")

        # Lookup city
        city = self._lookup_city(city_name)
        if not city:
            return json.dumps({
                "error": f"未找到城市: {city_name}",
                "hint": "使用 lookup_city 查看可用城市列表",
            }, ensure_ascii=False)

        # Find IFS date
        date = args.get("date")
        if date is None:
            dates = list_ifs_dates()
            if not dates:
                return json.dumps({"error": "没有可用的IFS预报数据"}, ensure_ascii=False)
            date = dates[-1]

        # Load the requested hour's indicator file
        project_dir = os.path.dirname(os.path.abspath(__file__))
        ifs_dir = os.path.join(project_dir, "aifs_forecasts", date)
        nc_path = os.path.join(ifs_dir, f"ifs_indicators_{date}_{hour}h.nc")

        # Fallback: try 0h if requested hour not available
        if not os.path.exists(nc_path):
            available = [f for f in os.listdir(ifs_dir)
                        if f.startswith("ifs_indicators_") and f.endswith(".nc")]
            if available:
                nc_path = os.path.join(ifs_dir, available[0])
                # Parse hour from filename
                import re
                m = re.search(r'(\d+)h\.nc', available[0])
                if m:
                    hour = int(m.group(1))
            else:
                return json.dumps({"error": f"IFS 指标数据不存在: {date}/{hour}h"}, ensure_ascii=False)

        ds = xr.open_dataset(nc_path)
        lat = ds["lat"].values
        lon = ds["lon"].values

        # Load all indicators
        ind = {}
        for v in ds.data_vars:
            arr = ds[v].values
            if arr.ndim > 2:
                arr = arr[0]
            ind[v] = arr.astype(np.float64)

        # Derived indicators (same as load_indicators_ifs)
        if "tcwv" in ind and "pwat" not in ind:
            ind["pwat"] = ind["tcwv"]
        if "t2m" in ind and "t2m_c" not in ind:
            ind["t2m_c"] = ind["t2m"]
        if "t2m" in ind and "tmax_c" not in ind:
            ind["tmax_c"] = ind["t2m"]
        if "sst" in ind and "sst_celsius" not in ind:
            ind["sst_celsius"] = ind["sst"]
        if "t2m" in ind and "t2m_anomaly_c" not in ind:
            ind["t2m_anomaly_c"] = ind["t2m"] - np.nanmean(ind["t2m"])
        if "tmax_c" in ind and "heatwave_day_flag" not in ind:
            ind["heatwave_day_flag"] = (ind["tmax_c"] >= 40).astype(np.float64)

        # flash_flood_risk
        if "flash_flood_risk" not in ind:
            ff_risk = np.zeros_like(ind.get("t2m", np.zeros((len(lat), len(lon)))))
            if "daily_precip_total" in ind:
                ff_risk += (ind["daily_precip_total"] >= 10).astype(float)
            if "pwat" in ind:
                ff_risk += (ind["pwat"] >= 30).astype(float)
            if "rh2m" in ind:
                ff_risk += (ind["rh2m"] >= 70).astype(float)
            ind["flash_flood_risk"] = ff_risk

        # heat_gpd_prob from climatology
        if "heat_gpd_prob" not in ind and "tmax_c" in ind:
            clim_path = os.path.join(project_dir, "forecast", "heat_gpd_climatology.nc")
            if os.path.exists(clim_path):
                cds = xr.open_dataset(clim_path)
                def _interp(var_name):
                    da = xr.DataArray(cds[var_name].values, dims=["lat", "lon"],
                                      coords={"lat": cds["lat"].values, "lon": cds["lon"].values})
                    return da.interp(lat=lat, lon=lon, method="linear").values
                thresh = _interp("gpd_threshold")
                scale = _interp("gpd_scale")
                exc_r = _interp("exceedance_rate")
                prob = np.ones_like(ind["tmax_c"])
                exceed = ind["tmax_c"] > thresh
                if exceed.any():
                    exc_val = ind["tmax_c"][exceed] - thresh[exceed]
                    pe = np.exp(-exc_val / np.maximum(scale[exceed], 1.0))
                    prob[exceed] = np.clip(pe * exc_r[exceed], 0, 1)
                ind["heat_gpd_prob"] = prob
                cds.close()

        ds.close()

        ifs_data = {
            "indicators": ind,
            "lat": lat,
            "lon": lon,
            "missing": [],
            "lead_time_h": hour,
        }

        # Run city-level evaluation
        result = evaluate_city_hazards(city_name, city, ifs_data, hazard_types)

        # Add context metadata
        result["forecast_source"] = f"ECMWF IFS 0.25deg, init {date}, +{hour}h"
        result["city_region"] = city.get("region", "unknown")
        result["note"] = "权威检测结论。severity 为最终判定，严禁 LLM 用先验知识覆盖。"

        return json.dumps(result, ensure_ascii=False, indent=2)

    def _get_calibrated_city_hazards(self, args: dict) -> str:
        """KG-calibrated city hazard detection — wraps _get_city_hazards + ForecastCalibrator."""
        # 1. Run base city detection
        raw = self._get_city_hazards(args)
        base_result = json.loads(raw)

        if "error" in base_result:
            return raw  # pass through errors

        # 2. Get city info and date
        city_name = args.get("city", "")
        city = self._lookup_city(city_name)
        if not city:
            return raw

        date = args.get("date")
        if date is None:
            from ifs_pipeline import list_ifs_dates
            dates = list_ifs_dates()
            date = dates[-1] if dates else "20260720"

        # 3. Run KG calibration
        try:
            from kg.forecast_calibrator import ForecastCalibrator
            calibrator = ForecastCalibrator()
            calibrated = calibrator.calibrate_city_confidence(base_result, date, city)
            calibrated["note"] = "KG校准增强版。含历史基准触发率和校准置信度。severity为权威检测结论。"
        except Exception as e:
            base_result["_kg_error"] = str(e)
            base_result["note"] = "KG校准失败，返回未校准结果。" + base_result.get("note", "")
            return json.dumps(base_result, ensure_ascii=False, indent=2)

        return json.dumps(calibrated, ensure_ascii=False, indent=2)

    def _assess_forecast_reliability(self, args: dict) -> str:
        """Run IFS detection + KG reliability assessment."""
        # 1. Run IFS detection
        raw = self._detect_from_ifs(args)
        fc_data = json.loads(raw)

        if "error" in fc_data:
            return raw

        date = args.get("date")
        if date is None:
            from ifs_pipeline import list_ifs_dates
            dates = list_ifs_dates()
            date = dates[-1] if dates else "20260720"

        # 2. Run KG reliability assessment
        try:
            from kg.forecast_calibrator import ForecastCalibrator
            calibrator = ForecastCalibrator()
            reliability = calibrator.assess_forecast_reliability(
                fc_data.get("hazards", []), date)
            fc_data["kg_reliability"] = reliability
        except Exception as e:
            fc_data["kg_reliability"] = {"error": str(e)}

        return json.dumps(fc_data, ensure_ascii=False, indent=2)

    def _query_historical_analogs_tool(self, args: dict) -> str:
        """Query KG for historical analog dates."""
        hazard_type = args.get("hazard_type")
        lat = args.get("lat")
        lon = args.get("lon")
        date = args.get("date")
        n_analogs = args.get("n_analogs", 5)

        try:
            from kg.forecast_calibrator import ForecastCalibrator, HAZARD_LABELS
            calibrator = ForecastCalibrator()
            analogs = calibrator.get_historical_analogs(
                hazard_type, lat, lon, date, n_analogs)
            analogs["hazard_type"] = hazard_type
            analogs["hazard_label"] = HAZARD_LABELS.get(hazard_type, hazard_type)
            analogs["query_location"] = f"({lat}N, {lon}E)"
            analogs["target_date"] = date
        except Exception as e:
            return json.dumps({"error": str(e), "hint": "确保已运行 build_event_catalog.py"}, ensure_ascii=False)

        return json.dumps(analogs, ensure_ascii=False, indent=2)

    def _detect_sequence(self, args: dict) -> str:
        """Batch detection across multiple forecast days with trend analysis."""
        import os

        start_day = args.get("start_day", 1)
        end_day = min(args.get("end_day", 3), 7)
        hazard_types = args.get("hazard_types")
        if start_day > end_day:
            start_day, end_day = end_day, start_day

        days = list(range(start_day, end_day + 1))
        daily_results = []

        for day in days:
            day_args = {"forecast_day": day}
            if hazard_types:
                day_args["hazard_types"] = hazard_types
            raw = self._detect_from_ifs(day_args)
            try:
                daily_results.append({"forecast_day": day, "data": json.loads(raw)})
            except Exception:
                daily_results.append({"forecast_day": day, "error": "解析失败"})

        # Trend analysis
        trends = {}
        if hazard_types is None:
            hazard_types = ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]

        for htype in hazard_types:
            scores = []
            sevs = []
            for dr in daily_results:
                data = dr.get("data", {})
                hazards = data.get("hazards", [])
                h = next((r for r in hazards if r["hazard_type"] == htype), {})
                if h.get("detected"):
                    scores.append(h.get("max_risk_score", 0))
                    sevs.append(h.get("severity", "?"))
                else:
                    scores.append(0)
                    sevs.append("未检出")

            # Trend direction
            if len(scores) >= 2:
                first_half = sum(scores[:len(scores)//2]) / max(len(scores)//2, 1)
                second_half = sum(scores[len(scores)//2:]) / max(len(scores) - len(scores)//2, 1)
                diff = second_half - first_half
                if diff > 0.1:
                    direction = "↑ 加剧"
                elif diff < -0.1:
                    direction = "↓ 减弱"
                elif all(s == 0 for s in scores):
                    direction = "— 未检出"
                else:
                    direction = "→ 持续"
            else:
                direction = "— 单日无法判断趋势"

            trends[htype] = {
                "daily_scores": {f"day_{d}": s for d, s in zip(days, scores)},
                "daily_severities": {f"day_{d}": s for d, s in zip(days, sevs)},
                "trend": direction,
                "peak_day": f"day_{days[scores.index(max(scores))]}" if max(scores) > 0 else None,
                "peak_score": round(max(scores), 3),
            }

        return json.dumps({
            "forecast_days": f"day {start_day}-{end_day}",
            "days_scanned": len(days),
            "daily_results": [
                {
                    "forecast_day": dr["forecast_day"],
                    "hazards": dr.get("data", {}).get("hazards", []),
                    "kg_consistency": dr.get("data", {}).get("kg_physical_consistency", {}),
                }
                for dr in daily_results
            ],
            "trend_analysis": trends,
        }, ensure_ascii=False, indent=2)

    # ═══════════════════════════════════════════════════════
    # compare_with_history — forecast vs 2025 same date
    # ═══════════════════════════════════════════════════════

    def _compare_with_history(self, args: dict) -> str:
        """Compare IFS forecast with 2025 same-date historical detection."""
        import os, datetime

        forecast_day = args.get("forecast_day", 1)
        hazard_types = args.get("hazard_types")
        if hazard_types is None:
            hazard_types = ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]

        # 1. Run forecast detection
        fc_raw = self._detect_from_ifs({"forecast_day": forecast_day, "hazard_types": hazard_types})
        try:
            fc_data = json.loads(fc_raw)
        except Exception:
            fc_data = {}

        # 2. Compute corresponding 2025 date
        today = datetime.date.today()
        # Get IFS init date (latest available)
        ifs_dates = list_ifs_dates() if HAS_IFS else []
        if ifs_dates:
            ifs_init = datetime.date.fromisoformat(ifs_dates[-1])
        else:
            ifs_init = today

        target_date = ifs_init + datetime.timedelta(days=forecast_day)
        year_ago = target_date.replace(year=2025)
        date_str = year_ago.strftime("%Y%m%d")

        # 3. Run historical detection
        hist_raw = self.dispatch("detect_extreme_events", {"date": date_str})
        try:
            hist_data = json.loads(hist_raw)
        except Exception:
            hist_data = {"error": "2025 年历史检测失败"}

        # 4. Per-hazard comparison
        comparison = []
        fc_hazards = fc_data.get("hazards", [])
        hist_events = hist_data.get("events", hist_data.get("results", []))

        for htype in hazard_types:
            fc_h = next((r for r in fc_hazards if r["hazard_type"] == htype), {})
            hist_h = next((r for r in hist_events if r.get("hazard_type") == htype), {})

            fc_score = fc_h.get("max_risk_score", 0)
            hist_score = hist_h.get("max_risk_score", 0)
            fc_sev = fc_h.get("severity", "N/A")
            hist_sev = hist_h.get("severity", "N/A")

            diff = round(fc_score - hist_score, 3)
            if diff > 0.15:
                anomaly = "↑ 显著高于去年同期"
            elif diff < -0.15:
                anomaly = "↓ 显著低于去年同期"
            elif abs(diff) <= 0.05:
                anomaly = "≈ 与去年同期持平"
            else:
                anomaly = "→ 与去年同期略有差异"

            comparison.append({
                "hazard_type": htype,
                "fc_score_2026": fc_score,
                "hist_score_2025": hist_score,
                "fc_severity": fc_sev,
                "hist_severity": hist_sev,
                "score_diff": diff,
                "anomaly_assessment": anomaly,
                "fc_hotspot": fc_h.get("hotspot_lat", "?"),
                "hist_hotspot": hist_h.get("hotspot_lat", "?"),
            })

        return json.dumps({
            "forecast_date": target_date.isoformat(),
            "historical_date": date_str,
            "comparison": comparison,
            "note": f"对比 2026 年 IFS 预报 vs 2025 年 ERA5 再分析检测。注意：数据源不同（预报 vs 再分析），对比结果反映相对异常程度，非严格同源对比。",
        }, ensure_ascii=False, indent=2)

    def _get_city_weather(self, args: dict) -> str:
        """Read all IFS indicators at the nearest grid cell to a city."""
        import os, numpy as np

        if not HAS_IFS:
            return json.dumps({"error": "IFS pipeline not installed"}, ensure_ascii=False)

        from ifs_pipeline import load_indicators_ifs, list_ifs_dates

        city_name = args.get("city", "")
        forecast_day = args.get("forecast_day", 0)
        city = self._lookup_city(city_name)

        if not city:
            return json.dumps({
                "error": f"未找到城市: {city_name}",
                "hint": "使用 lookup_city 查看可用城市列表",
            }, ensure_ascii=False)

        # Auto-select latest IFS date
        dates = list_ifs_dates()
        if not dates:
            return json.dumps({"error": "没有可用的IFS预报数据"}, ensure_ascii=False)
        date = dates[-1]

        ifs_data = load_indicators_ifs(date, forecast_day)
        if ifs_data is None:
            return json.dumps({"error": f"IFS数据不存在: {date}"}, ensure_ascii=False)

        ind = ifs_data["indicators"]
        lat = ifs_data["lat"]
        lon = ifs_data["lon"]

        # Nearest grid cell
        d2 = np.sqrt((lat[:, None] - city["lat"])**2 +
                     (lon[None, :] - city["lon"])**2)
        ni, nj = np.unravel_index(np.argmin(d2), d2.shape)
        dist_km = round(float(d2[ni, nj] * 111.0), 1)

        # Read values
        values = {}
        for ind_id, arr in sorted(ind.items()):
            values[ind_id] = round(float(arr[ni, nj]), 2)

        # Group
        cats = {
            "温度": ["t2m_c", "tmax_c"],
            "湿度": ["rh2m", "dewpoint_depression_c", "specific_humidity_850"],
            "风": ["wind10_speed", "wind_direction", "wind_shear_850_200"],
            "降水": ["daily_precip_total", "pwat", "ivt_convergence"],
            "气压": ["sp"],
        }
        grouped = {}
        for cat, ids in cats.items():
            grouped[cat] = {i: values[i] for i in ids if i in values}
        other = {k: v for k, v in values.items() if k not in sum(cats.values(), [])}
        if other:
            grouped["其他"] = other

        return json.dumps({
            "city": city["label"],
            "city_coords": f"({city['lat']}N, {city['lon']}E)",
            "nearest_grid": f"({float(lat[ni]):.1f}N, {float(lon[nj]):.1f}E)",
            "grid_distance_km": dist_km,
            "forecast_day": forecast_day,
            "lead_time_h": ifs_data.get("lead_time_h", "?"),
            "values": grouped,
            "note": "IFS 网格预报值，0.25°分辨率，未经过地面站点订正",
        }, ensure_ascii=False, indent=2)

    def _lookup_city_tool(self, args: dict) -> str:
        """Lookup city coordinates for the agent."""
        city_name = args.get("city", "")
        city = self._lookup_city(city_name)
        if city:
            return json.dumps({
                "found": True,
                "city": city["label"],
                "lat": city["lat"],
                "lon": city["lon"],
                "region": self.SAUDI_REGIONS.get(city.get("region", ""), {}).get("label", ""),
                "hint": f"使用 get_city_weather 或 query_observations_nearby 查询该城市数据。",
            }, ensure_ascii=False, indent=2)
        all_cities = [f'{c["label"]}({cid})' for cid, c in self.SAUDI_CITIES.items()]
        return json.dumps({
            "found": False,
            "query": city_name,
            "available_cities": all_cities,
        }, ensure_ascii=False, indent=2)

    # ═══════════════════════════════════════════════════════
    # Post-processing: Shamal flag, region report, isolated convection
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _post_process_detection(hazards: list, indicators: dict,
                                 lat_vals, lon_vals) -> dict:
        """Enrich detection results with Shamal wind, region grouping, and
        isolated convection signals."""
        import numpy as np

        result = {"shamal_flag": False, "region_report": {}, "isolated_convection": None}

        # ── 1. Shamal wind flag for dust_storm ──
        dust = next((h for h in hazards if h["hazard_type"] == "dust_storm"), None)
        wind_dir = indicators.get("wind_direction")
        wind_spd = indicators.get("wind10_speed")

        if dust and dust.get("detected") and wind_dir is not None:
            # Shamal = wind from NW (300-360°) or N (345-15°)
            nw_mask = (wind_dir >= 285) & (wind_dir <= 360)
            n_mask  = (wind_dir >= 345) | (wind_dir <= 15)

            shamal_cells = int((nw_mask | n_mask).sum())
            total_cells = wind_dir.size
            shamal_fraction = shamal_cells / total_cells if total_cells > 0 else 0

            # Check if Shamal wind overlaps with dust risk regions
            dust_lat = dust.get("hotspot_lat", 0)
            dust_lon = dust.get("hotspot_lon", 0)
            near_gulf = (24 <= dust_lat <= 30) and (48 <= dust_lon <= 56)

            if shamal_fraction > 0.15 or (near_gulf and shamal_cells > 0):
                result["shamal_flag"] = True
                result["shamal_detail"] = {
                    "shamal_cells": shamal_cells,
                    "shamal_fraction": round(shamal_fraction, 2),
                    "assessment": (
                        "Shamal 风活跃，NW-N 方向风占 {:.0%}。"
                        "沙尘可沿西北→东南方向传播至波斯湾沿岸港区。"
                    ).format(shamal_fraction),
                }

        # ── 2. Region-grouped risk report ──
        region_summary = {}
        for h in hazards:
            if not h.get("detected"):
                continue
            region = h.get("region", "未知区域")
            if region not in region_summary:
                region_summary[region] = []
            region_summary[region].append({
                "hazard_type": h["hazard_type"],
                "severity": h.get("severity"),
                "score": h.get("max_risk_score"),
                "triggered_count": len(h.get("triggered_conditions", [])),
            })

        # Sort regions by risk level
        sev_order = {"extreme": 4, "high": 3, "medium": 2, "low": 1}
        for region in region_summary:
            region_summary[region].sort(
                key=lambda x: sev_order.get(x["severity"], 0), reverse=True)

        result["region_report"] = {
            "regions_affected": len(region_summary),
            "by_region": region_summary,
        }

        # ── 3. Isolated convection signals ──
        ff = next((h for h in hazards if h["hazard_type"] == "flash_flood"), None)
        precip = indicators.get("daily_precip_total")

        if precip is not None:
            heavy_cells = int((precip >= 10).sum())
            precip_max = float(np.nanmax(precip))

            if heavy_cells > 0:
                peak_idx = np.unravel_index(np.nanargmax(precip), precip.shape)
                conv_lat = float(lat_vals[peak_idx[0]])
                conv_lon = float(lon_vals[peak_idx[1]])

                # Tag region for convection
                conv_region = ToolDispatcher._tag_region(conv_lat, conv_lon)

                result["isolated_convection"] = {
                    "heavy_precip_cells": heavy_cells,
                    "max_precip_mm": round(precip_max, 1),
                    "hotspot_lat": round(conv_lat, 1),
                    "hotspot_lon": round(conv_lon, 1),
                    "region": conv_region,
                    "flash_flood_triggered": ff.get("detected", False) if ff else False,
                    "assessment": (
                        f"检测到 {heavy_cells} 个格点日降水量 ≥10mm"
                        f"（最大 {precip_max:.1f}mm，位于 {conv_region} "
                        f"{conv_lat:.1f}N, {conv_lon:.1f}E）。"
                    ),
                }

                if ff and not ff.get("detected"):
                    result["isolated_convection"]["assessment"] += (
                        "山洪综合检测未触发，但这些孤立强降水格点存在局地对流风险，"
                        "尤其在阿西尔山脉陡峭地形区域可能引发局地山洪。"
                    )

        return result

    # ═══════════════════════════════════════════════════════
    # Composite risk scoring
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _compute_composite_risk(hazards: list, indicators: dict,
                                 lat_vals, lon_vals, region_calib: dict = None) -> dict:
        """Compute multi-hazard composite risk from individual hazard scores."""
        import numpy as np

        detected_hazards = [h for h in hazards if h.get("detected")]
        if len(detected_hazards) < 2:
            return {
                "composite_level": "无复合风险",
                "overlapping_hazards": len(detected_hazards),
                "note": "仅单一灾害触发或未检出，不存在复合叠加",
            }

        # Find overlap regions (grid cells where multiple hazards trigger)
        # We use the hotspot proximity as a coarse proxy
        overlap_pairs = []
        composite_rules = region_calib.get("composite_risk", {}).get("rules", []) if region_calib else []
        rule_map = {tuple(sorted(r["hazards"])): r for r in composite_rules}

        # Check for overlapping hazards
        hazard_names = sorted([h["hazard_type"] for h in detected_hazards])
        hazards_key = tuple(hazard_names)

        multiplier = 1.0
        rule_note = ""
        if hazards_key in rule_map:
            r = rule_map[hazards_key]
            multiplier = r["multiplier"]
            rule_note = r["note"]
        else:
            # Check partial matches
            for pair_key, r in rule_map.items():
                if all(h in hazard_names for h in pair_key):
                    multiplier = max(multiplier, r["multiplier"])
                    rule_note = r["note"]

        # Compute composite score
        scores = [h["max_risk_score"] for h in detected_hazards]
        avg_score = sum(scores) / len(scores)
        composite_score = min(avg_score * multiplier, 1.0)

        # Hotspot proximity analysis
        hotspots = [(h["hotspot_lat"], h["hotspot_lon"], h["hazard_type"])
                     for h in detected_hazards if "hotspot_lat" in h]
        proximity_warning = ""
        if len(hotspots) >= 2:
            for i in range(len(hotspots)):
                for j in range(i+1, len(hotspots)):
                    dlat = hotspots[i][0] - hotspots[j][0]
                    dlon = hotspots[i][1] - hotspots[j][1]
                    dist_deg = np.sqrt(dlat**2 + dlon**2)
                    if dist_deg < 5:  # within ~500km
                        overlap_pairs.append({
                            "hazard_a": hotspots[i][2],
                            "hazard_b": hotspots[j][2],
                            "distance_deg": round(float(dist_deg), 1),
                        })
        if overlap_pairs:
            proximity_warning = f"{len(overlap_pairs)} 对灾害热点相邻（<5°），存在空间叠加风险"

        composite_level = (
            "extreme" if composite_score >= 0.8
            else "high" if composite_score >= 0.6
            else "medium" if composite_score >= 0.4
            else "low"
        )

        return {
            "composite_level": composite_level,
            "composite_score": round(composite_score, 3),
            "multiplier": multiplier,
            "multiplier_rule": rule_note,
            "overlapping_hazards": len(detected_hazards),
            "hazard_types": [h["hazard_type"] for h in detected_hazards],
            "hazard_scores": {h["hazard_type"]: h["max_risk_score"] for h in detected_hazards},
            "hotspot_overlaps": overlap_pairs,
            "proximity_warning": proximity_warning,
        }

    def _detect_composite_risk(self, args: dict) -> str:
        """Detect composite multi-hazard risk for a forecast day."""
        import os
        forecast_day = args.get("forecast_day", 1)
        hazard_types = ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]

        # Run full detection with region calibration
        result = self._detect_from_ifs({
            "forecast_day": forecast_day,
            "hazard_types": hazard_types,
        })
        data = json.loads(result)

        composite = data.get("composite_risk", {})
        if composite.get("overlapping_hazards", 0) < 2:
            # Also check if single hazard with region calibration gives useful output
            calibrated = data.get("region_calibrated", False)

        return json.dumps({
            "forecast_day": forecast_day,
            "region_calibrated": data.get("region_calibrated", False),
            "individual_hazards": [
                {
                    "hazard_type": h["hazard_type"],
                    "severity": h.get("severity"),
                    "score": h.get("max_risk_score"),
                    "region": h.get("region"),
                    "triggered_count": len(h.get("triggered_conditions", [])),
                }
                for h in data.get("hazards", [])
            ],
            "composite_risk": composite,
            "analysis": (
                f"复合风险等级: {composite.get('composite_level', 'N/A')} "
                f"(评分 {composite.get('composite_score', 0):.3f})。"
                f"触发灾害: {composite.get('hazard_types', [])}。"
                f"{composite.get('multiplier_rule', '')}"
                f"{composite.get('proximity_warning', '')}"
            ),
        }, ensure_ascii=False, indent=2)

    # ═══════════════════════════════════════════════════════
    # RAG Knowledge Base Search
    # ═══════════════════════════════════════════════════════

    def _search_kb(self, args: dict) -> str:
        """Search the local ChromaDB knowledge base."""
        import os

        query = args.get("query", "")
        k = min(args.get("k", 3), 8)

        if not query.strip():
            return json.dumps({"error": "query 不能为空"}, ensure_ascii=False)

        project_dir = os.path.dirname(os.path.abspath(__file__))
        kb_dir = os.path.join(project_dir, "kb")

        if not os.path.isdir(kb_dir):
            return json.dumps({
                "error": "知识库未构建",
                "hint": "运行 python build_kb.py 构建知识库",
                "kb_path": kb_dir,
            }, ensure_ascii=False)

        try:
            from langchain_community.vectorstores import Chroma
            from langchain_community.embeddings import HuggingFaceEmbeddings

            embed_model = os.environ.get(
                "MAZU_EMBED_MODEL",
                "BAAI/bge-small-zh-v1.5",
            )

            embeddings = HuggingFaceEmbeddings(
                model_name=embed_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            vectorstore = Chroma(
                persist_directory=kb_dir,
                embedding_function=embeddings,
                collection_name="mazu_knowledge",
            )

            results = vectorstore.similarity_search_with_score(query, k=k)

            hits = []
            for doc, score in results:
                hits.append({
                    "content": doc.page_content[:800],
                    "source": doc.metadata.get("source", "?"),
                    "type": doc.metadata.get("type", "?"),
                    "relevance": round(float(1.0 / (1.0 + score)), 4),  # L2 → 相似度
                })

            return json.dumps({
                "query": query,
                "total_hits": len(hits),
                "results": hits,
            }, ensure_ascii=False, indent=2)

        except ImportError as e:
            return json.dumps({
                "error": f"缺少依赖: {e}",
                "hint": "pip install langchain langchain-community chromadb sentence-transformers",
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"检索失败: {str(e)}"}, ensure_ascii=False)

    def _get(self, path):
        r = self.requests.get(f"{self.api_base}{path}")
        return json.dumps(r.json(), ensure_ascii=False, indent=2)


# ── Smart tool-result truncation ──

def _trunc_str(s: str, n: int = 60) -> str:
    """Truncate a long string value, keeping it parseable."""
    if len(s) <= n:
        return s
    return s[:n] + "…"


def smart_truncate(result: str, tool_name: str = "", max_chars: int = 3000) -> str:
    """Intelligently truncate a tool result, keeping JSON structure valid.

    Different from ``result[:max_chars]`` which can cut JSON mid-key and
    destroy parseability.  This function parses the result, downsamples
    large lists / long strings, and re-serialises, so the truncated
    output is still valid JSON the LLM can read.

    For non-JSON results we break at the last paragraph boundary.
    """
    if len(result) <= max_chars:
        return result

    # ── Non-JSON: paragraph-boundary truncation ──
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        truncated = result[:max_chars]
        last_nl = truncated.rfind("\n")
        if last_nl > max_chars * 0.5:
            truncated = truncated[:last_nl]
        omitted = len(result) - len(truncated)
        return truncated + f"\n…(省略 {omitted} 字符)"

    # ── JSON result: structure-aware trimming ──
    _TOOLS_DETECTION = {
        "detect_future_events", "detect_extreme_events", "detect_ifs_forecast",
    }
    _TOOLS_SEQUENCE = {"detect_forecast_sequence"}
    _TOOLS_KB = {"search_knowledge_base"}

    if tool_name in _TOOLS_DETECTION:
        keep = {}
        for k in ("forecast_source", "forecast_day", "lead_time_h", "date",
                   "region_calibrated", "synthesis"):
            if k in data:
                keep[k] = data[k]
        hazards = data.get("hazards", data.get("events", []))
        keep["hazards"] = []
        for h in hazards:
            summary = {
                "hazard_type": h.get("hazard_type"),
                "detected": h.get("detected"),
                "severity": h.get("severity"),
                "max_risk_score": h.get("max_risk_score"),
                "coverage": h.get("coverage"),
                "hotspot": f"{h.get('hotspot_lat', '?')}N, {h.get('hotspot_lon', '?')}E",
            }
            triggers = h.get("triggered_conditions", [])
            # Keep top 3 only (each trigger with truncated strings)
            summary["top_triggers"] = [
                {"indicator": _trunc_str(t.get("indicator", ""), 60),
                 "cells": t.get("cells_triggered"),
                 "peak": t.get("peak_value"),
                 "condition": _trunc_str(t.get("condition", ""), 50)}
                for t in triggers[:3]
            ]
            if len(triggers) > 3:
                summary["_triggers_omitted"] = len(triggers) - 3
            keep["hazards"].append(summary)
        keep["_truncated"] = True

    elif tool_name in _TOOLS_SEQUENCE:
        keep = {}
        for k in ("start_day", "end_day", "forecast_source", "trend"):
            if k in data:
                keep[k] = data[k]
        days = data.get("daily_results", [])
        keep["daily_summary"] = []
        for d in days[:3]:  # max 3 days
            keep["daily_summary"].append({
                "forecast_day": d.get("forecast_day"),
                "lead_time_h": d.get("lead_time_h"),
                "hazards": [
                    {"type": h.get("hazard_type"), "sev": h.get("severity"),
                     "score": h.get("max_risk_score")}
                    for h in d.get("hazards", [])
                ],
            })
        if len(days) > 3:
            keep["_days_omitted"] = len(days) - 3
        keep["_truncated"] = True

    elif tool_name in _TOOLS_KB:
        results = data.get("results", [])
        keep = {
            "query": data.get("query"),
            "total_hits": data.get("total_hits"),
            "results": [
                {**r, "content": r.get("content", "")[:200]}
                for r in results[:3]
            ],
            "_truncated": True,
        }

    else:
        # Generic: downsample large lists / long strings / large dicts
        keep = {}
        for k, v in data.items():
            if isinstance(v, list) and len(v) > 5:
                keep[k] = v[:5]
                keep[f"{k}_count"] = len(v)
            elif isinstance(v, str) and len(v) > 500:
                keep[k] = v[:500] + "…"
            elif isinstance(v, dict):
                if len(v) > 10:
                    sub = dict(list(v.items())[:10])
                    sub["_total_keys"] = len(v)
                    keep[k] = sub
                else:
                    sub = {}
                    for sk, sv in v.items():
                        if isinstance(sv, list) and len(sv) > 5:
                            sub[sk] = sv[:3]
                            sub[f"{sk}_count"] = len(sv)
                        elif isinstance(sv, str) and len(sv) > 500:
                            sub[sk] = sv[:500] + "…"
                        else:
                            sub[sk] = sv
                    keep[k] = sub
            else:
                keep[k] = v
        keep["_truncated"] = True

    # ── Safe re-serialisation ──
    # 1. Try compact (no indent) — fastest path to fit
    out = json.dumps(keep, ensure_ascii=False, indent=None, separators=(",", ":"))
    if len(out) <= max_chars:
        # Compact fits — re-serialise with indent for readability if it still fits
        pretty = json.dumps(keep, ensure_ascii=False, indent=2)
        if len(pretty) <= max_chars:
            return pretty
        return out

    # 2. Compact still too long: progressively remove items until it fits
    if tool_name in _TOOLS_DETECTION and "hazards" in keep:
        while len(keep["hazards"]) > 1:
            keep["hazards"].pop()
            keep["_hazards_omitted"] = keep.get("_hazards_omitted", 0) + 1
            out = json.dumps(keep, ensure_ascii=False, indent=None, separators=(",", ":"))
            if len(out) <= max_chars:
                return out
        # Last resort: drop triggers too
        for h in keep["hazards"]:
            while h.get("top_triggers") and len(h["top_triggers"]) > 1:
                h["top_triggers"].pop()
                h["_triggers_omitted"] = h.get("_triggers_omitted", 0) + 1
                out = json.dumps(keep, ensure_ascii=False, indent=None, separators=(",", ":"))
                if len(out) <= max_chars:
                    return out

    # Generic progressive removal
    if isinstance(keep, dict):
        list_keys = [k for k, v in keep.items() if isinstance(v, list)]
        for lk in list_keys:
            while len(keep[lk]) > 1:
                keep[lk].pop()
                out = json.dumps(keep, ensure_ascii=False, indent=None, separators=(",", ":"))
                if len(out) <= max_chars:
                    return out

    # 3. Absolute last resort: minimal valid JSON with error metadata
    return json.dumps(
        {"_error": "result_too_large", "_original_chars": len(result),
         "_truncated_to": max_chars, "_tool": tool_name},
        ensure_ascii=False,
    )


# ── Convenience: dispatch function (stateless, for simple loops) ──

_dispatcher = None

def dispatch_tool(tool_name: str, arguments: dict) -> str:
    """Stateless wrapper — call from your LLM tool-handling loop."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = ToolDispatcher()
    return _dispatcher.dispatch(tool_name, arguments)


# ── Demo ──
if __name__ == "__main__":
    print("MAZU Agent Tools — Schema Preview:")
    for t in TOOLS:
        print(f"  {t['function']['name']}: {t['function']['description'][:60]}...")
    print(f"\nTotal: {len(TOOLS)} tools ready for Function Calling")
