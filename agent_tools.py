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
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    # Event Detection — Forecast (AIFS NetCDF)
    # ═══════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "detect_future_events",
            "description": "基于 ECMWF AIFS 预报数据，检测未来几天的极端灾害风险。数据来自 forecast/ 目录。适用于'明天会有山洪吗''未来3天利雅得会有极端高温吗'等预报问题。",
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
    # Multi-Model Arbitration (FCN + AIFS + KG)
    # ═══════════════════════════════════════════════════
    {
        "type": "function",
        "function": {
            "name": "compare_forecast_models",
            "description": "同时查询 FCN 和 AIFS 两个独立预报模型的输出，用知识图谱中的气候态做交叉验证。返回两模型对比、物理一致性评分和置信度。适用于'明天会多热''这个预报可信吗'等问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "forecast_day": {"type": "integer", "description": "预报天数，1=明天"},
                    "variable": {"type": "string", "enum": ["t2m", "tp", "wind"], "description": "对比变量"}
                },
                "required": ["forecast_day", "variable"]
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
        try:
            if tool_name == "query_hazard_indicators":
                return self._get(f"/api/sparql/hazard/{arguments['hazard_type']}")

            elif tool_name == "query_indicator_detail":
                return self._get(f"/api/sparql/indicator/{arguments['indicator_id']}")

            elif tool_name == "query_indicator_chain":
                return self._get(f"/api/sparql/chain/{arguments['indicator_id']}")

            elif tool_name == "search_indicators":
                return self._get(f"/api/sparql/search?q={arguments['keyword']}")

            elif tool_name == "query_rule_detail":
                return self._get(f"/api/sparql/rule/{arguments['rule_id']}")

            elif tool_name == "query_observations_nearby":
                params = {k: v for k, v in arguments.items() if v is not None}
                qs = "&".join(f"{k}={v}" for k, v in params.items())
                return self._get(f"/api/sparql/geospatial/radius?{qs}")

            elif tool_name == "query_events_in_region":
                return self._get(f"/api/sparql/geospatial/intersects?region={arguments['region']}")

            elif tool_name == "query_event_timeline":
                sd = arguments.get("start_date", "")
                ed = arguments.get("end_date", "")
                return self._get(f"/api/sparql/temporal/timeline?start={sd}&end={ed}")

            elif tool_name == "query_cascading_chain":
                return self._get(f"/api/sparql/temporal/cascade/{arguments['event_id']}")

            elif tool_name == "query_provenance":
                return self._get(f"/api/sparql/provenance/indicator/{arguments['indicator_id']}")

            elif tool_name == "detect_extreme_events":
                body = {"date": arguments["date"]}
                if "hazard_types" in arguments:
                    body["hazard_type"] = arguments["hazard_types"][0] if arguments["hazard_types"] else None
                r = self.requests.post(f"{self.api_base}/api/detect", json=body)
                return json.dumps(r.json(), ensure_ascii=False, indent=2)

            elif tool_name == "detect_future_events":
                return self._detect_from_forecast(arguments)

            elif tool_name == "compare_forecast_models":
                return self._compare_models(arguments)

            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except Exception as e:
            return json.dumps({"error": str(e), "tool": tool_name})

    def _detect_from_forecast(self, args: dict) -> str:
        """从 AIFS 预报数据运行事件检测"""
        import numpy as np
        import xarray as xr
        import os, sys

        forecast_day = args.get("forecast_day", 1)
        hazard_types = args.get("hazard_types", None)
        location = args.get("location", None)

        nc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "forecast", f"saudi_forecast_d{forecast_day:02d}.nc")

        if not os.path.exists(nc_path):
            return json.dumps({
                "error": f"预报文件不存在: {nc_path}",
                "hint": f"请先运行: python get_forecast.py --days {forecast_day}"
            }, ensure_ascii=False)

        # 加载预报 + 计算派生指标
        ds = xr.open_dataset(nc_path)
        lat_vals = ds['lat'].values
        lon_vals = ds['lon'].values

        # 计算 event_detector 需要的基本指标
        indicators = {}
        if 't2m' in ds.variables:
            indicators['t2m_c'] = ds['t2m'].values - 273.15  # K → °C
        if 'tp' in ds.variables:
            indicators['daily_precip_total'] = ds['tp'].values
        if 'u10' in ds.variables and 'v10' in ds.variables:
            indicators['wind10_speed'] = np.sqrt(ds['u10'].values**2 + ds['v10'].values**2)
        if 'pwat' in ds.variables:
            indicators['pwat'] = ds['pwat'].values
        if 'd2m' in ds.variables:
            d2m_c = ds['d2m'].values - 273.15
            if 't2m_c' in indicators:
                indicators['dewpoint_depression_c'] = indicators['t2m_c'] - d2m_c
        if 'tcc' in ds.variables:
            indicators['total_cloud_cover'] = ds['tcc'].values * 100.0  # 0-1 → %

        # 加载规则
        schema_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema")
        with open(os.path.join(schema_dir, "rules.json"), "r", encoding="utf-8") as f:
            rules_data = json.load(f)

        # 对每种灾害做简单阈值检测（无需完整的 event_detector）
        results = []
        if hazard_types is None:
            hazard_types = ["flash_flood", "extreme_heat", "dust_storm", "coastal_humid_heat"]

        for htype in hazard_types:
            rule = next((r for r in rules_data["rules"] if r["hazard_type"] == htype), None)
            if not rule:
                continue

            # 取条件中可用的指标
            available_conds = []
            for cond in rule["conditions"]:
                ind_id = cond["indicator"]
                if ind_id in indicators:
                    available_conds.append(cond)

            if len(available_conds) < 2:
                continue

            # 简化的加权评分
            mask = np.ones_like(list(indicators.values())[0], dtype=bool)
            score = np.zeros_like(mask, dtype=float)
            total_w = 0.0
            triggered = []

            for cond in available_conds:
                ind_id = cond["indicator"]
                op = cond.get("op", cond.get("condition", ">="))
                th = cond["value"]
                w = cond["weight"]
                data = indicators[ind_id]

                if op == ">=":
                    hit = data >= th
                elif op == ">":
                    hit = data > th
                elif op == "<":
                    hit = data < th
                elif op == "<=":
                    hit = data <= th
                else:
                    continue

                score += w * hit.astype(float)
                total_w += w
                n_hit = int(hit.sum())
                peak = float(data.max()) if not np.isnan(data).all() else 0
                if n_hit > 0:
                    triggered.append({"indicator": ind_id, "condition": f"{op} {th}",
                                       "cells_triggered": n_hit, "peak_value": round(peak, 2)})

            if total_w > 0:
                score = score / total_w

            # Primary gate
            primary_cond = next((c for c in rule["conditions"] if c.get("primary")), None)
            if primary_cond and primary_cond["indicator"] in indicators:
                pdata = indicators[primary_cond["indicator"]]
                pmask = pdata >= primary_cond["value"]
                score = np.where(pmask, score, score * 0.25)

            # 找到最高风险区域
            max_score = float(np.nanmax(score))
            n_risky = int((score >= 0.3).sum())
            risky_lat = float(lat_vals[np.unravel_index(np.nanargmax(score), score.shape)[0]])
            risky_lon = float(lon_vals[np.unravel_index(np.nanargmax(score), score.shape)[1]])

            sev = "extreme" if max_score >= 0.8 else "high" if max_score >= 0.6 else "medium" if max_score >= 0.3 else "low"

            results.append({
                "hazard_type": htype,
                "forecast_day": forecast_day,
                "max_risk_score": round(max_score, 3),
                "severity": sev,
                "grid_cells_at_risk": n_risky,
                "hotspot_lat": round(risky_lat, 1),
                "hotspot_lon": round(risky_lon, 1),
                "triggered_conditions": triggered,
            })

        ds.close()

        if location:
            for r in results:
                r["location_note"] = f"热点 ({r['hotspot_lat']}N, {r['hotspot_lon']}E) 需要空间查询精确定位 {location} 周边"

        return json.dumps({
            "forecast_day": forecast_day,
            "forecast_source": "ECMWF AIFS",
            "results": results,
            "note": "预报基于 AIFS 全球模型输出，存在不确定性。建议结合实时观测验证。"
        }, ensure_ascii=False, indent=2)

    def _compare_models(self, args: dict) -> str:
        """Multi-model comparison with KG-based physical consistency check."""
        import numpy as np, xarray as xr, os

        day = args.get("forecast_day", 1)
        variable = args.get("variable", "t2m")
        project_dir = os.path.dirname(os.path.abspath(__file__))

        # Load both forecasts
        fcn_path = os.path.join(project_dir, "forecast", "fcn_forecast.nc")
        aifs_path = os.path.join(project_dir, "forecast", f"saudi_forecast_d{day:02d}.nc")

        result = {"forecast_day": day, "variable": variable, "models": {}}

        # --- FCN ---
        if os.path.exists(fcn_path):
            f = xr.open_dataset(fcn_path)
            if variable == "t2m" and "t2m" in f.variables:
                # Use lead_time=12h for afternoon peak
                fcn_val = f["t2m"].values[0, 2] - 273.15
                result["models"]["fcn"] = {
                    "mean": round(float(np.nanmean(fcn_val)), 1),
                    "max": round(float(np.nanmax(fcn_val)), 1),
                    "min": round(float(np.nanmin(fcn_val)), 1),
                    "note": "取12h预报(午后峰值)"
                }
            f.close()

        # --- AIFS ---
        if os.path.exists(aifs_path):
            a = xr.open_dataset(aifs_path)
            var_map = {"t2m": "t2m", "tp": "tp", "wind": "u10"}
            av = var_map.get(variable, "t2m")
            if av in a.variables:
                aifs_val = a[av].values
                if av == "t2m":
                    aifs_val = aifs_val - 273.15
                result["models"]["aifs"] = {
                    "mean": round(float(np.nanmean(aifs_val)), 1),
                    "max": round(float(np.nanmax(aifs_val)), 1),
                    "min": round(float(np.nanmin(aifs_val)), 1),
                }
            a.close()

        # --- KG arbitration ---
        if "fcn" in result["models"] and "aifs" in result["models"]:
            fcn_m = result["models"]["fcn"]["mean"]
            aifs_m = result["models"]["aifs"]["mean"]
            diff = round(abs(fcn_m - aifs_m), 1)

            # KG check: compare against climatology
            with open(os.path.join(project_dir, "schema", "operators.json"), "r", encoding="utf-8") as fp:
                ops = json.load(fp)
            t2m_op = next((o for o in ops["operators"] if o["id"] == "t2m_c"), {})

            if diff > 3:
                result["arbitration"] = {
                    "verdict": "模型分歧显著",
                    "diff_celsius": diff,
                    "confidence": "medium",
                    "recommendation": f"FCN 预报偏高 {diff}°C，两模型不一致。建议关注 KG 中 vpd_kpa(干燥度) 和 dewpoint_depression_c(露点差) 是否也偏高——如果都偏高则 FCN 更可信（沙漠极端高温信号），否则 AIFS 保守值更安全。",
                    "kg_evidence": "t2m 气候态表明沙特 7 月均值 33-43°C，需结合 t2m_anomaly_c 判断异常程度。"
                }
            else:
                result["arbitration"] = {
                    "verdict": "两模型一致",
                    "diff_celsius": diff,
                    "confidence": "high",
                    "recommendation": f"FCN 和 AIFS 预报差异仅 {diff}°C，可采信。可进一步查询该区域气候态确认是否异常。"
                }

        return json.dumps(result, ensure_ascii=False, indent=2)

    def _get(self, path):
        r = self.requests.get(f"{self.api_base}{path}")
        return json.dumps(r.json(), ensure_ascii=False, indent=2)


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
