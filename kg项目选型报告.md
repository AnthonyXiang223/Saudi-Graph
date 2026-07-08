# 可替换为沙特极端气象灾害预警的知识图谱项目分析

> 目标：找到 GitHub 可用的、使用 NetCDF 结构化气象数据的、实体属性方便用 Saudi 91个指标替换的知识图谱项目

---

## 一、筛选结果总览

六轮搜索后，锁定 **6 个项目/框架**——3 个可 GitHub 直接 fork，3 个论文但架构可直接复用：

| # | 项目 | 类型 | GitHub | NetCDF | 替换难度 |
|---|---|---|---|---|---|
| 1 | **DOMINO-SEE** | 空间极端事件网络检测 | ✅ [PREP-NexT/DOMINO-SEE](https://github.com/PREP-NexT/DOMINO-SEE) | ✅ xarray/NetCDF 原生 | ⭐ 低 |
| 2 | **GeoOutageKG** | 极端天气停电多模态 KG | ✅ [purl.org/geoutagekg](https://purl.org/geoutagekg) | ✅ xarray + BlackMarblePy | ⭐⭐ 中 |
| 3 | **HadUKGrid Agent** | 气候 NetCDF → OWL KG | ✅ [cambridge-cares/TheWorldAvatar](https://github.com/cambridge-cares/TheWorldAvatar) | ✅ NetCDF Agent | ⭐ 低 |
| 4 | **Storm Surge ST-KG** | 风暴潮时空知识图谱 | ❌ 论文 (IJGI 2024) | — | ⭐⭐⭐ 高 |
| 5 | **Typhoon KG** | 台风灾害五层知识模型 | ❌ 论文 (JGIS 2023) | — | ⭐⭐⭐ 高 |
| 6 | **NOAA Zarr/Dask/KG/LLM** | NOAA 气候 KG 查询栈 | ❌ 演讲 (FOSS4G) | ✅ Zarr→KG | ⭐⭐ 中 |

---

## 二、TOP 3 深度分析（可直接 fork 的项目）

### 2.1 DOMINO-SEE ⭐⭐⭐⭐⭐ — **最推荐**

| 维度 | 详情 |
|---|---|
| **GitHub** | [PREP-NexT/DOMINO-SEE](https://github.com/PREP-NexT/DOMINO-SEE) |
| **许可证** | GPL 2.0（可 fork、修改、商用） |
| **论文** | *Nature Water* (2025) — 全球空间同步水文极端事件 |
| **语言** | Python 3.9+ |
| **核心依赖** | xarray, netCDF4, dask, numpy, scipy, numba |

**架构**：

```
NetCDF 网格数据 (lat × lon × time)
         │
         ▼
  [eventorize]  二值化: 连续时间序列 → 0/1 事件序列
         │         (替换点: 把 hardcoded 阈值换成你的 91 个指标 DAG)
         │
         ▼
  [eca/es]  事件重合分析 / 事件同步
         │    统计检验 → 哪些格点的事件在时间上同步
         │
         ▼
  [network]  邻接矩阵 → 空间 event 网络
         │   节点 = 格点, 边 = 显著同步事件
         │
         ▼
  输出: 级联极端事件的时空传播网络
```

**为什么最适合你**：

| 你的项目 | DOMINO-SEE | 直接替换方案 |
|---|---|---|
| `indicators/*.nc` 365日文件 | `xarray.open_dataset()` | 读你自己的文件路径即可 |
| 91 个指标变量 | `eventorize.get_event()` 阈值化 | 把阈值逻辑换成你 `operators.json` 的 DAG |
| `flash_flood_risk` | ECA 事件网络节点 | 用你的风险分做节点值 |
| 沙特 bbox 35,200 格点 | Fekete 格点约 1-10 万 | 直接用你的 160×220 网格 |
| 空间连通域 (scipy.ndimage.label) | `network.get_link_from_confidence()` | 互补：你检测格点簇 → DOMINO-SEE 检测跨格点传播 |

**替换的具体步骤**：

```python
# DOMINO-SEE 原始: 用分位数阈值
events = get_event(data, method='percentile', th_frac=0.95)

# 替换为你的: 用 operators.json 的 DAG 阈值
from kg.operators import evaluate_dag
events = evaluate_dag(operator["dag"], context)
```

---

### 2.2 HadUKGrid Agent ⭐⭐⭐⭐ — **最接近你的架构**

| 维度 | 详情 |
|---|---|
| **GitHub** | [cambridge-cares/TheWorldAvatar](https://github.com/cambridge-cares/TheWorldAvatar) (GasGrid/HadUKGrid 模块) |
| **许可证** | MIT |
| **论文** | Cambridge CARES 数字孪生项目 |

**架构**：

```
OWL 气候本体 (climate_ontology/)
         │
         ├── 定义: ClimateMeasurement → 关联到 → ONS Region
         ├── 定义: Temperature, Precipitation, Wind 等气候变量属性
         │
         ▼
  [hadukgrid_inputs_agent.py]  ← Python Agent
         │
         ├── 打开 NetCDF 文件
         ├── 解释气候变量
         ├── 关联每个 output polygon 到一组网格点
         ├── 聚合温度/降水值
         │
         ▼
  输出: RDF 三元组 → GraphDB/SPARQL
```

**为什么适合**：

| 你的项目 | HadUKGrid | 直接替换方案 |
|---|---|---|
| `schema/ontology.json` | `climate_ontology/` OWL 文件 | 把你的 5 类节点 + 7 类关系转成 OWL |
| `kg/datalayer.py` 加载 NetCDF | `hadukgrid_inputs_agent.py` | 改路径和变量名即可 |
| 91 个指标 | ClimateMeasurement 属性 | 替换为你的 indicator ID 列表 |
| 沙特 bbox | UK ONS 统计区 | 换成你的 6 个 Region |

**你的项目的天然适配**：你的 `Region` 节点（red_sea, persian_gulf 等）和 HadUKGrid 的 ONS 区域节点是同构的——都是空间聚合单元。把 OWL 本体换成你的 ontology.json，NetCDF Agent 换成你的 datalayer.py，管道直接通。

---

### 2.3 GeoOutageKG ⭐⭐⭐ — **最成熟的 OWL2 本体方案**

| 维度 | 详情 |
|---|---|
| **GitHub** | [purl.org/geoutagekg](https://purl.org/geoutagekg) |
| **论文** | ISWC 2025 |
| **数据规模** | 10.6M 停电记录 + 300K 卫星图像 + 15K 停电地图 |

**架构**：

```
数据采集: NASA Black Marble (VNP46A2) → 夜间灯光
          EAGLE-I → 县级停电时间序列
              │
              ▼
  [数据 FAIR 化] → xarray DataArray 处理 → 异常检测
              │
              ▼
  GeoOutageOnto (OWL2)
    ├── 对齐: GEOSatDB, DBpedia, Ontology for Media Resources
    ├── 实体: County, Hurricane, PowerOutage, SatelliteImage
    ├── 关系: causedBy, locatedIn, observedAt, hasSeverity
    │
              ▼
  序列化: Turtle (RDF 1.2) → GraphDB + SPARQL endpoint
```

**为什么值得参考**：

1. **OWL2 本体工程最成熟**：对齐了多个国际标准本体，你的项目如果要对接 MAZU 标准，这个模式最规范
2. **多模态融合**：卫星图像 + 时间序列 + 事件记录 → 三条数据管道融合到一个 KG。你的项目有 DS10（卫星降水）+ DS2（再分析）融合需求
3. **极端天气驱动**：用例就是飓风 Ian/Milton 的停电分析，和你的极端事件预警是同类型的"天气 → 影响"建模

**替换方案**：

| GeoOutageKG | 你的项目 | 替换 |
|---|---|---|
| `County` 实体 | `Region` 实体 | 换成 red_sea, persian_gulf... |
| `PowerOutage` 事件 | `Event` 实体 | 换成 flash_flood, extreme_heat... |
| `Hurricane` 触发器 | `Rule` 节点 | 换成你的 4 条规则 |
| `satelliteImage` | NetCDF 网格 | 换成你的 35,200 格点 |
| EAGLE-I 时间序列 | NetCDF 日数据 | 换成 indicators/*.nc |

---

## 三、论文项目（无 GitHub，但架构可复用）

### 3.4 Storm Surge ST-KG — 时空网格编码方案

| 维度 | 详情 |
|---|---|
| **论文** | *Multi-Scale Spatio-Temporal Knowledge Graph for Storm Surge Hazards*, IJGI 2024 |
| **技术栈** | Python + Neo4j + GeoSOT 编码 |
| **规模** | 2.1M+ 洪水格点节点 |

**核心创新 — GeoSOT 时空编码**：

```
lat=24.7°N, lon=46.7°E, time=2025-08-19T14:00
          │
          ▼
  GeoSOT 编码: "G001310322-2-3-5..."  ← 一个字符串同时编码空间+时间
          │
          ▼
  KG 节点: FloodGrid_{GeoSOT_string}
          属性: flooded_depth, flooded_level, time_code
```

**这对你项目的价值**：如果你想把 35,200 个格点 × 365 天 ≈ 1285 万个时空切片都编进 KG，GeoSOT 是最省存储的方案。一个 32 字符字符串替代 (lat, lon, date) 三个字段。

### 3.5 Typhoon KG — 五层知识表示模型

| 维度 | 详情 |
|---|---|
| **论文** | *顾及时空过程的台风灾害事件知识图谱*, JGIS 2023 |
| **技术栈** | Python + Neo4j + LSTM NER |
| **规模** | 204 轨迹记录 + 2,566 灾情记录 |

**五层模型**：

```
概念层 (Concept)   → "台风"、"暴雨"、"洪涝"  ← 你的 4 类 HazardType
对象层 (Object)    → "台风烟花"、"福建省"     ← 你的 Region
状态层 (State)     → "2021-07-25T08:00 风速45m/s" ← 你的每日 NetCDF 快照
特征层 (Feature)   → 最大风速、累计降水、路径   ← 你的 91 个 Indicator
关系层 (Relation)  → 导致、发生于、先于、加剧   ← 你的 7 类关系
```

**和你的项目天然对应**：

| 台风 KG 层 | 你的项目 |
|---|---|
| 概念层 | `HazardType` (4 个) |
| 对象层 | `Region` (6 个) + `Event` (动态) |
| 状态层 | 每日 NetCDF 网格快照（数据层） |
| 特征层 | `Indicator` (91 个) |
| 关系层 | 7 类 edge type |

---

## 四、综合对比：哪个最适合你

| 评判维度 | DOMINO-SEE | HadUKGrid | GeoOutageKG | Storm Surge | Typhoon KG |
|---|---|---|---|---|---|
| **GitHub 可 fork** | ✅ | ✅ | ✅ | ❌ | ❌ |
| **NetCDF 原生** | ✅ xarray | ✅ Agent | ❌ (图像) | ❌ | ❌ |
| **指标替换容易度** | ⭐⭐⭐ 低（改阈值函数） | ⭐⭐⭐ 低（改变量名） | ⭐⭐ 中 | ⭐⭐⭐ 高 | ⭐⭐⭐ 高 |
| **实体属性替换** | ⭐⭐ 中 | ⭐⭐⭐ 低 | ⭐⭐ 中 | ⭐ 极低 | ⭐ 极低 |
| **知识图谱标准** | 自定义 | OWL | OWL2 | Neo4j | Neo4j |
| **与你架构相似度** | ⭐⭐⭐ 事件检测 | ⭐⭐⭐ 本体+Agent | ⭐⭐ 多模态 | ⭐ 高定制 | ⭐⭐ 五层对应 |
| **可运行 demo** | ✅ Jupyter | ✅ Python Agent | ✅ Turtle+SPARQL | ❌ | ❌ |
| **轻量化** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ (GraphDB) | ⭐⭐ (Neo4j) | ⭐⭐ (Neo4j) |

---

## 五、推荐选型

### 首推：DOMINO-SEE（fork 替换阈值逻辑）

```
DOMINO-SEE
  ├── 保留: xarray 加载 + ECA 网络分析 + dask 并行
  ├── 替换: eventorize 阈值函数 → 你的 operators.json DAG
  ├── 替换: 输入数据 → indicators/*.nc
  └── 新增: 你的 4 类灾害规则 + explain() 审计
```

最轻量、最直接、可运行 demo 马上就出。

### 其次：HadUKGrid（OWL 本体 + NetCDF Agent 模板）

```
HadUKGrid
  ├── 保留: OWL ontology 模板 + NetCDF Agent 结构
  ├── 替换: climate_ontology → ontology.json → to_owl.py
  ├── 替换: UK 区域 → 沙特 6 个 Region
  └── 新增: 你的 DAG 解释器 + Event 检测
```

适合后续 MAZU 对接时做标准化的 OWL 输出。

### 参考：Typhoon KG 五层模型（架构对齐参考）

不 fork 代码，只参考它的五层架构来整理你现有的 5 类节点是否有遗漏——比如你的"状态层"目前在数据层（xarray），但 Event 节点生成后也可以编码进 KG。

---

## 六、三个项目的替换映射表

你的 report 里 91 个指标 → 各项目的替换：

| 你的指标 | DOMINO-SEE | HadUKGrid | GeoOutageKG |
|---|---|---|---|
| `daily_precip_total` | `get_event(data, th=10mm)` | `ClimateMeasurement.precipitation` | — |
| `tmax_c` | `get_event(data, th=45°C)` | `ClimateMeasurement.temperature` | — |
| `cape` | `get_event(data, th=1000 J/kg)` | `ClimateMeasurement.cape` | — |
| `flash_flood_risk` | 网络节点值 | 空间聚合结果 | `DisasterEvent.severity` |
| `heatwave_duration_days` | ECA 时间窗口 | 时间序列聚合 | `Event.duration` |
| `sst_celsius` | 多层级节点 | `ClimateMeasurement.sst` | — |
| Region (red_sea) | 空间子图 | `Region` OWL 类 | `County` 类 |
| Event | ECA 同步事件 | `DisasterEvent` | `PowerOutage` 事件 |
