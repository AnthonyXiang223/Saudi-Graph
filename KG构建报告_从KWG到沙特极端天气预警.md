# 从 KnowWhereGraph 到沙特极端天气预警知识图谱

## 构建过程、架构适配与差异化设计

---

## 摘要

本文档阐述沙特极端事件知识图谱的完整构建过程。以 KnowWhereGraph (KWG) 的 DMDO 灾害本体为参考框架，结合沙特热带沙漠气候特征和 MAZU 早期预警系统的工程约束，通过裁剪灾后属性、嵌入可执行算子引擎（DAG）、引入阈值过滤的 SOSA 观测实例化流程，以及补充沙漠气候特有的联合解释指标网络，构建了一个轻量化、可嵌入、准实时的多灾种预警知识图谱。最终图谱包含 7801 个 RDF 三元组，覆盖 91 个气象指标、6 个数据源、4 类灾害和 4 条加权检测规则，通过了 SPARQL 知识查询和事件检测的集成验证。

---

## 一、KWG 参考架构：DMDO 本体的三层结构

KnowWhereGraph 是全球最大的公开地理知识图谱（290 亿三元组），其灾害管理领域本体 DMDO（Disaster Management Domain Ontology）采用三层模块化设计：

### 1.1 灾害事件模块（Hazard Event Module）

定义灾害全生命周期的核心实体类：

```
deo:Event                          # 所有事件的抽象基类
  ├── deo:Hazard                   # 潜在威胁（尚未造成实际损害）
  ├── deo:Disaster                 # 已造成实际损害的事件
  └── deo:DisasterImpact           # 灾害造成的后果（伤亡、经济损失）

deo:HazardType                     # 灾害分类标签（对接 HIP 灾害分类体系）
deo:ElementAtRisk                  # 承灾体（人口、建筑、基础设施、农田）
```

核心关系包括 `possiblyCauses`（事件因果链）、`affects`（威胁→承灾体）、`hasHazardProperty`（灾害类型的可观测属性）。

### 1.2 灾害属性模块（DPO — Disaster Properties Ontology）

将"风险"拆分为 6 个独立可观测属性：

```
dpo:Intensity       # 灾害强度（如日降水 254mm）
dpo:Severity        # 后果严重度（如受影响面积 1600 km²）
dpo:LevelOfExposure # 承灾体暴露度
dpo:Vulnerability   # 承灾体脆弱性
dpo:Capacity        # 应对能力（灾后救援资源）
dpo:Resilience      # 恢复能力（灾后恢复速度）
```

所有属性继承自 `sosa:ObservableProperty`，确保可观测、可测量、可对比。

### 1.3 SOSA/SSN 观测模型

所有数值数据采用 W3C 标准传感器观测本体建模：

```
FeatureOfInterest (利雅得格点 24.7°N, 46.7°E)
       │
       ▼
Observation (2025-08-19 14:00 UTC)
       ├── observedProperty: dpo:Intensity  # 观测什么
       ├── hasSimpleResult: 45.2 mm         # 观测值
       ├── madeBySensor: GPM IMERG          # 传感器来源
       └── resultTime: 2025-08-19           # 观测时间
```

---

## 二、沙特场景的适配挑战

直接部署 KWG/DMDO 面临三个根本性矛盾：

### 2.1 气候场景不匹配

| 维度 | KWG 原始设计 | 沙特场景 |
|---|---|---|
| 典型灾害 | 飓风、野火、烟羽、滑坡（温带/亚热带） | 山洪、极端高温干旱、沙尘暴、沿海湿热（热带沙漠） |
| 关键指标 | NDVI 植被指数、雪盖、海冰 | 降水距平率、VPD（饱和水汽压差）、IVT（水汽输送）、热浪持续天数 |
| 数据特征 | 多源遥感（MODIS、NASA Black Marble）| ERA5 再分析 + GPM IMERG 卫星降水 + OSTIA 海温 |
| 空间粒度 | S2 全球网格（Level 13 ≈ 20 km²）| 0.1° 规则网格 ≈ 11 km（35,200 格点） |

### 2.2 系统定位不匹配

| 维度 | KWG | MAZU 早期预警 |
|---|---|---|
| 灾害阶段 | 全周期（减灾→备灾→响应→恢复） | **仅预警**（灾前 + 灾中实时） |
| 查询模式 | SPARQL endpoint，离线知识检索 | 准实时事件检测 + 可解释推理 |
| 运维约束 | GraphDB 集群，专业运维 | 轻量化，可嵌入，单机部署 |
| 数据时效 | 月度更新 | 每日更新 |

### 2.3 数据可用性约束

沙特作为发展中国家目标区域，存在以下数据限制：
- DS10（GPM IMERG 卫星降水）Q4（10-12 月）缺测
- DS8（GHCN 气候态）在沙特 bbox 内仅 7 个降水站点、11 个最高温站点
- 无本地承灾体数据库（人口、建筑、农田矢量数据）

---

## 三、架构适配：五个关键修改

### 3.1 裁剪灾后属性：从"全周期"收缩到"预警阶段"

DMDO 的 6 个风险属性中，我们**保留了前 2 个，裁剪了后 4 个**：

```
保留:
  dpo:Intensity  → 91 个气象指标（如 daily_precip_total = 254mm）
  dpo:Severity   → 加权风险分（event_detector 输出 0-1）

裁剪（对早期预警无直接贡献）:
  dpo:LevelOfExposure  → 需要承灾体数据（当前不可用）
  dpo:Vulnerability    → 需要社会经济数据（超出气象数据范围）
  dpo:Capacity         → 灾后救援资源（不属于预警阶段）
  dpo:Resilience       → 灾后恢复能力（不属于预警阶段）
```

**理由**：MAZU 是"早期预警"系统，不是"灾害管理"系统。预测"灾后多久能恢复"超出了当前气象数据和模型的能力范围。专注做好 Intensity + Severity → 综合 DisasterRisk 即足够支撑预警决策。

### 3.2 嵌入 DAG 可执行算子：打破 DMDO 的"静态本体"局限

**DMDO 的根本缺陷**：它是静态本体，只能记录"降水 = 254mm"，无法表达"这个 254mm 是怎么算出来的"。

**我们的创新**：每个 `sosa:ObservableProperty`（指标）同时携带三层表达：

| 字段 | 消费者 | 内容 |
|---|---|---|
| `description` | LLM / Agent | "日累计总降水，来自 DS2 的 tp 变量" |
| `expression` | 人类开发者 | "tp" |
| `dag` | DAG 解释器 | `{"op":"var","name":"tp"}` |

复杂指标如 `vpd_kpa` 的 DAG 包含 5 层操作节点：

```
vpd_kpa = 0.6108 × exp(17.27 × T / (T + 237.3)) × (1 - RH/100)
         ──┬──    ───┬───    ──┬───    ─────┬─────
          mul       exp       div           sub
```

**16 个基础操作符**（add, sub, mul, div, sqrt, sqr, pow, max, min, abs, threshold, where, and_, or_, 以及气象专用 vpd_formula, heat_index_formula, sum_flags）构成完整的可执行词汇表。这使得图谱不仅是知识存储，更是计算推理的基础设施。

### 3.3 知识层与数据层分离：适配 35,200 × 365 的时空体量

KWG 将所有空间实体（S2 Cell）都编入图谱，但我们的数据体量是 35,200 格点 × 365 天 × 91 指标 = 约 12 亿个时空切片。全部编入 RDF 会导致图谱爆炸。

**我们的双层架构**：

```
知识层 (networkx + rdflib, ~7801 triples)
  └── 只存 Indicator、DataSource、HazardType、Rule、Region、Event 的知识关系

数据层 (xarray + NetCDF)
  └── 管所有时空栅格数据，查询通过 xarray.where() 执行
```

两层之间通过 Indicator.id = NetCDF variable name 桥接。运行时，知识层告诉引擎"查什么"，数据层执行查询并返回结果。

### 3.4 阈值过滤的 SOSA 观测实例化

全量将 12 亿个时空切片写入 RDF 不可行。我们采用**按需实例化 + 阈值过滤**策略：

```python
# 只将超过预警阈值的格点写入 RDF
converter.add_observations(
    date_str="2025-08-19",
    indicator_ids=["tmax_c", "daily_precip_total"],
    threshold_filter={"tmax_c": (">=", 45), "daily_precip_total": (">=", 10)}
)
# 结果: 4065 条观测（全量 7 万+ 条中的预警相关子集）
```

### 3.5 沙特特有的联合解释网络

沙漠气候下，单一指标容易误判：

- 日总降水 50mm 但分布在 12 小时 → 不危险
- 日总降水 30mm 但集中在 1 小时 → 危险（山洪）

因此 91 个指标间构建了 218 条 `co_occurs_with`（联合解释）双向边。例如 `daily_precip_total` 必须与 `ds10_max_1h`（1 小时最大降水）和 `ds10_rainy_steps`（有雨持续时间）联合解释才有预警意义。

---

## 四、最终图谱结构

### 4.1 规模统计

| 指标 | 值 |
|---|---|
| RDF 三元组总数 | 7,801 |
| Indicator 节点 | 91（sosa:ObservableProperty） |
| DataSource 节点 | 6（DS1/DS2/DS4/DS8/DS10/SST） |
| HazardType 节点 | 4（山洪/极端高温/沙尘强风/沿海湿热） |
| Rule 节点 | 4（每条一种检测规则，含 primary gate） |
| Region 节点 | 6（沙特全域/红海/波斯湾/北部/中部/南部） |
| co_occurs_with 边 | 218 |
| derived_from 边 | 143 |
| contributes_to 边 | 17 |
| detects 边 | 4 |

### 4.2 四类灾害检测规则

| 灾害 | 条件数 | Primary Gate | Fallback 策略 |
|---|---|---|---|
| 山洪 | 5 | flash_flood_risk ≥ 3 | ds10_max_1h 缺失 → 跳过该条件，置信度 -0.15 |
| 极端高温 | 4 | heatwave_day_flag ≥ 1 | 无 |
| 沙尘强风 | 4 | wind10_speed ≥ 12 m/s | wind_shear_850_200 缺失 → 跳过，置信度 -0.10 |
| 沿海湿热 | 4 | sst_celsius ≥ 30°C | 无 |

每条规则支持 primary gate（主条件门控）机制——主条件不满足的格点风险分压制 75%，大幅降低误报率。

### 4.3 与 KWG/DMDO 的映射关系

| KWG DMDO | 本图谱 | 适配说明 |
|---|---|---|
| `deo:Hazard` | 4 个 Rule 节点 | 预警规则 → 潜在威胁 |
| `deo:Disaster` | 动态 Event 节点 | 检测输出 → 实际事件 |
| `dpo:Intensity` | 91 个 Indicator | 气象极值指标 |
| `dpo:Severity` | severity_score | 加权风险分 |
| `sosa:ObservableProperty` | 91 个 Indicator | 每个指标同时是 SOSA 属性 |
| `sosa:Observation` | 按需创建 | 阈值过滤，避免 12 亿条爆炸 |
| `dpo:LevelOfExposure` | — | **裁剪**（缺承灾体数据） |
| `dpo:Vulnerability` | — | **裁剪**（超出预警范围） |
| `dpo:Capacity` | — | **裁剪**（超出预警范围） |
| `dpo:Resilience` | — | **裁剪**（超出预警范围） |

---

## 五、验证结果

| 测试 | 日期 | 结果 |
|---|---|---|
| KG 构建与验证 | — | 7801 triples，无孤立节点，无循环依赖 |
| 山洪检测 | 2025-08-19 | 9 个事件，最大 6722 km²（红海沿岸），符合报告记录 |
| 极端高温 | 2025-07-25 | 16 个事件，最大 218,491 km²（tmax=53.75°C） |
| 静默日（反例） | 2025-01-15 | 0 事件 — 零误报 |
| 算子链完整性 | — | 全部 91 个指标的 derived_from 链可追溯到 DataSource |
| 降级逻辑 | Q4 缺测期 | fallback 策略正常执行，检测不中断 |
| SPARQL 知识查询 | — | 山洪指标、推导链、联合解释、关键词搜索全通 |
| SOSA Observation 时空查询 | 2025-08-19 | tmax≥45°C → 2622 条，每条带 lat/lon/值/日期 |

---

## 六、与 KWG 的差异总结

| 维度 | KWG | 本图谱 | 修改原因 |
|---|---|---|---|
| 理论框架 | DMDO 全周期本体 | DMDO 裁剪版（仅预警） | MAZU 定位：早期预警 ≠ 灾害管理 |
| 计算能力 | 静态本体 | 静态本体 + DAG 可执行引擎 | 气象预警需要动态计算，不能只存结果 |
| 数据架构 | 全 RDF 存储（S2 Cell 全部编入） | 知识层(7801 triples) + 数据层(xarray) 分离 | 12 亿时空切片无法全入图 |
| 观测存储 | 全部预先计算 | 阈值过滤按需实例化 | 预警只需关注异常值 |
| 指标网络 | 独立属性 | 91 节点 + 218 条联合解释边 | 沙漠气候单指标易误判 |
| 部署模型 | GraphDB 集群 + SPARQL endpoint | Flask + networkx + rdflib | MAZU 要求轻量化嵌入 |
| 数据特征 | 多源遥感 + 社会人口 | ERA5 再分析 + GPM + OSTIA | 沙特可用的全球数据集 |
