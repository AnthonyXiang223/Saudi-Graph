# Saudi Extreme Event Knowledge Graph

MAZU 多灾种早期预警系统 — 沙特极端气候事件知识图谱。

基于 91 个专家定义的极端事件指标算子，构建可嵌入 MAZU 系统的轻量化知识图谱。支持山洪、极端高温、沙尘强风、沿海湿热四类灾害的智能检测与可解释推理。

## 架构

```
schema/          ← 知识定义（本体 + 91个算子 + 4条检测规则）
kg/              ← 引擎（networkx 图 + DAG 解释器 + xarray 数据层 + 事件检测）
dashboard/       ← Web 交互界面（Flask + vis-network）
verify.py        ← 5项集成验证
```

## 快速开始

### 1. 安装依赖

```bash
pip install networkx xarray netCDF4 numpy pandas scipy flask
```

### 2. 准备数据

将 365 个 NetCDF 指标文件放入 `indicators/` 目录：

```
indicators/
├── saudi_indicators_20250101.nc
├── saudi_indicators_20250102.nc
├── ...
└── saudi_indicators_20251231.nc
```

> 数据文件约 5GB，不包含在 Git 仓库中。文件命名格式：`saudi_indicators_YYYYMMDD.nc`

### 3. 运行验证

```bash
python verify.py
```

预期输出：

```
[PASS] 1_kg_construction
[PASS] 2_flash_flood       # 2025-08-19 山洪检测
[PASS] 3_extreme_heat       # 2025-07-25 高温检测
[PASS] 4_quiet_day          # 静默日零误报
[PASS] 5_operator_chain     # 算子链完整性
5/5 tests passed
```

### 4. 启动仪表盘

```bash
python dashboard/server.py
```

打开 **http://127.0.0.1:5000**，交互式可视化知识图谱：

- 点击任意节点 → 查看关系 / 计算公式 / DAG / 推导链
- 搜索框 → 快速定位指标
- 底部检测面板 → 输入日期运行事件检测

### 5. 命令行查询

```bash
python -m kg.query --date 2025-08-19 --hazard flash_flood --explain
```

## 项目结构

| 文件 | 说明 |
|---|---|
| `schema/ontology.json` | 5 类节点 + 7 类关系定义 |
| `schema/operators.json` | 91 个指标的物理公式 + 可执行 DAG |
| `schema/rules.json` | 4 类灾害检测规则（加权 + fallback + 连通域） |
| `kg/ontology.py` | networkx 图构建，111 节点，428 边 |
| `kg/operators.py` | OPS 词汇表（16 操作符）+ DAG 解释器 |
| `kg/datalayer.py` | xarray 封装 + LRU 缓存 (maxsize=7) |
| `kg/event_detector.py` | 加权评分 + primary gate + 连通域 + fallback |
| `kg/query.py` | 三层查询入口（知识 / 数据 / 联合） |

## 知识图谱统计

| 指标 | 值 |
|---|---|
| 总节点 | 111 |
| Indicator 节点 | 91 |
| DataSource 节点 | 6 (DS1/DS2/DS4/DS8/DS10/SST) |
| HazardType 节点 | 4 |
| Rule 节点 | 4 |
| Region 节点 | 6 |
| 总边数 | 428 |
| derived_from | 42 |
| co_occurs_with | 264 |
| sourced_from | 101 |
| contributes_to | 17 |
| detects | 4 |

## 四类灾害检测

| 灾害 | 条件数 | Primary Gate | Fallback |
|---|---|---|---|
| 山洪 (flash_flood) | 5 | flash_flood_risk ≥ 3 | ds10_max_1h 缺失 → 降权重 |
| 极端高温 (extreme_heat) | 4 | heatwave_day_flag ≥ 1 | 无 |
| 沙尘强风 (dust_storm) | 4 | wind10_speed ≥ 12 m/s | wind_shear_850_200 缺失 → 降权重 |
| 沿海湿热 (coastal_humid_heat) | 4 | sst_celsius ≥ 30°C | 无 |

## 验证结果

| 测试 | 日期 | 结果 |
|---|---|---|
| 山洪检测 | 2025-08-19 | ✅ 9 事件，最大集群 57 格点，红海沿岸 |
| 极端高温 | 2025-07-25 | ✅ 16 事件，最大 2050 格点 (~218K km²) |
| 静默日 | 2025-01-15 | ✅ 0 事件 — 零误报 |
| 算子链 | — | ✅ 全部可追溯到数据源 |
| KG 验证 | — | ✅ 无孤立节点，无循环依赖 |

## 技术栈

networkx · xarray · numpy · scipy · Flask · vis-network · NetCDF4

## 相关文档

- [沙特极端事件知识图谱构建方案](沙特极端事件知识图谱构建方案.md)
- [指标计算与分布洞察报告](saudi_extreme_event_indicator_report.md)
