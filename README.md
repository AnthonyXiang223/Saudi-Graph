# MAZU 沙特多灾种早期预警智能体

MAZU 系列早期预警系统 — 沙特阿拉伯本地化多灾种智能预警算法。基于 KnowWhereGraph (DMDO-OWL) 知识图谱 + NVIDIA FourCastNet AI 预报 + DeepSeek-V3 LLM Agent。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 DeepSeek API Key
```

### 3. 准备数据

**知识图谱数据**（约 5GB，365 个 NetCDF 文件）：

将指标文件放入 `indicators/` 目录：
```
indicators/
├── saudi_indicators_20250101.nc
├── saudi_indicators_20250102.nc
├── ...
└── saudi_indicators_20251231.nc
```

**预报数据**（在 WSL2 中生成）：

```bash
# WSL2 Ubuntu: conda activate earth2
cd /mnt/f/Saudi
python run_fcn.py --days 7
# 输出: forecast/fcn_forecast.nc (~3MB)
```

### 4. 启动系统

```bash
# 终端 1: 知识图谱 API
python dashboard/server.py

# 终端 2: Agent Web 界面
python server.py
```

打开 **http://127.0.0.1:8501** 开始使用。

### 5. 命令行模式（可选）

```bash
python agent.py
```

## 项目结构

```
schema/                          # 知识定义
├── ontology.json                # 5 节点类型 + 7 边类型
├── operators.json               # 91 指标 DAG 表达式 + 数据来源
├── rules.json                   # 4 灾害检测规则（权重/因果角色/fallback）
└── region_calibration.json      # 六大区域阈值偏移量

kg/                              # 知识图谱引擎
├── owl/
│   ├── to_rdf.py                # RDF 三元组构建 (7,801 triples)
│   └── sparql_queries.py        # 10 类 SPARQL 查询
├── event_detector.py            # 历史事件检测（加权 + 连通域）
└── datalayer.py                 # xarray + LRU cache

dashboard/                       # Web 服务
├── server.py                    # Flask API（12 个端点）
└── templates/
    └── index_sparql.html        # KG 可视化界面

agent.py                         # DeepSeek-V3 Agent (CLI)
agent_tools.py                   # 15 个 Function Calling 工具 + FCN 预报引擎
server.py                        # FastAPI Web UI + API 服务
run_fcn.py                       # FourCastNet 预报脚本 (WSL2)
learn_weights.py                 # L1 逻辑回归权重学习
```

## 能力矩阵（15 个工具）

### 知识查询
| 工具 | 说明 |
|---|---|
| `query_hazard_indicators` | 查询灾害依赖的指标列表 |
| `query_indicator_detail` | 指标物理定义 + DAG + 数据源 |
| `query_indicator_chain` | 指标推导链追溯 |
| `search_indicators` | 关键词搜索指标 |
| `query_rule_detail` | 检测规则完整条件 |

### 时空推理
| 工具 | 说明 |
|---|---|
| `query_observations_nearby` | GeoSPARQL 半径空间搜索 |
| `query_events_in_region` | 区域历史事件查询 |
| `query_event_timeline` | OWL-Time 时间线 |
| `query_cascading_chain` | 灾害级联链 |
| `query_provenance` | PROV-O 数据溯源 |

### 事件检测
| 工具 | 说明 |
|---|---|
| `detect_extreme_events` | 历史事件检测 (ERA5) |
| `detect_future_events` | FCN 单日预报 + KG 验证 + 区域标注 |
| `detect_forecast_sequence` | 批量 1-7 天趋势分析 |
| `detect_composite_risk` | 复合灾害叠加评分 |
| `compare_with_history` | 2025 年同期对比 |

## 四类灾害

| 灾害 | 检测条件 | 关键机理 |
|---|---|---|
| 山洪 | 降水 + 水汽辐合 + 可降水量 | 红海地形抬升对流 |
| 极端高温 | 温度 + 露点差 + 持续时间 | 区域校准阈值 |
| 沙尘强风 | 风速 + 干燥度 + Shamal 风向 | 波斯湾传播路径 |
| 沿海湿热 | 温度 + 湿度 + 海温 | 红海/波斯湾沿岸 |

## 区域校准

六大地理分区独立阈值：鲁布哈利沙漠高温 47°C 触发、红海沿岸 40°C、阿西尔山脉 36°C。详见 `schema/region_calibration.json`。

## 技术栈

Python · DeepSeek-V3 · FourCastNet · Streamlit · Flask · rdflib · xarray · NumPy · SciPy
