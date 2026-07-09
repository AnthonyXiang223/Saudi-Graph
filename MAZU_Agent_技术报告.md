# MAZU 多灾种早期预警智能体 (Agent) 设计与实现报告

## 摘要

针对沙特阿拉伯地区复杂的热带沙漠气候（如极端高温、突发性暴雨引发的山洪等），传统纯数值气象预测模型往往只能输出高维概率矩阵，难以直接转化为各行业可执行的防灾策略。本报告详细阐述了 MAZU 多灾种早期预警系统中"决策大脑"——灾害研判智能体 (Agent) 的设计方案与中期实现路径。

在整体架构上，该智能体模块起到了核心的承上启下作用。上游数据接入层，智能体无缝对接经空间插值与清洗去噪后的高维气象张量数据（NetCDF），精准捕获气象异常信号；下游知识融合层，通过引入 RAG（检索增强生成）技术，智能体深度挂载团队基于 KnowWhereGraph 联合构建的沙特极端天气知识图谱，赋予系统对本地地形地貌与关键基础设施的常识推理能力。

在工程选型与落地策略上，本项目坚守"敏捷开发与按需演进"的原则。中期阶段摒弃了过度复杂的重型多智能体框架，转而采用高度可控的原生 Function Calling 循环构建单体智能体。辅以严格的角色锚定（Role Prompting）与思维链（Chain of Thought, CoT）约束机制，系统有效规避了大语言模型的"幻觉"风险，实现了预警决策过程的高度可解释性。目前，该模块已成功打通从"底层气象异常侦测"、"图谱关联灾害推演"到"靶向生成普惠预警简报"的自动化闭环，完全契合 MAZU 系统"多灾种、零差距"的核心定位。

---

## 1. 知识底座构建：沙特极端天气知识图谱

### 1.1 从 KnowWhereGraph 到沙特场景的架构适配

在构建 MAZU 系统的"决策大脑"时，我们明确了一个核心前提：底层气象张量数据（如降水、温度异常值）只反映了自然界的物理状态，而灾害的本质是这些物理状态对人类社会造成的破坏。因此，系统必须挂载一个结构化的知识图谱，将孤立的"气象信号"转化为立体的"灾害链条"。

**KnowWhereGraph (KWG)** 是全球最大的公开地理知识图谱（290 亿三元组），其灾害管理领域本体 DMDO 采用三层模块化设计：

- **灾害事件模块**：定义 `deo:Hazard`（潜在威胁）、`deo:Disaster`（实际灾害）、`deo:DisasterImpact`（灾害后果）及 `deo:ElementAtRisk`（承灾体）

- **灾害属性模块 (DPO)**：将"风险"拆分为 `Intensity`、`Severity`、`LevelOfExposure`、`Vulnerability`、`Capacity`、`Resilience` 六个可观测属性，全部继承自 `sosa:ObservableProperty`

- **SOSA/SSN 观测模型**：采用 W3C 标准将数值数据建模为 `sosa:Observation` 实例，携带 FeatureOfInterest（空间坐标）、observedProperty（指标）、Sensor（传感器）、Procedure（观测过程）等完整信息

  ![Details are in the caption following the image](https://onlinelibrary.wiley.com/cms/asset/66a069cd-e8ec-4410-a8e6-4c1525dbb9fe/aaai12043-fig-0001-m.jpg)

  ![image-20260709211737228](C:/Users/13936/AppData/Roaming/Typora/typora-user-images/image-20260709211737228.png)

### 1.2 五个关键适配修改

考虑到沙特热带沙漠气候特征和 MAZU 早期预警系统的工程约束，我们对 KWG 架构进行了五项关键裁剪和增强：

**（1）裁剪灾后属性：从"全周期"收缩到"预警阶段"**

DMDO 的 6 个风险属性中，仅保留 `dpo:Intensity`（91 个气象指标）和 `dpo:Severity`（加权风险分），裁剪 `LevelOfExposure`、`Vulnerability`、`Capacity`、`Resilience`。理由：MAZU 是"早期预警"系统而非"全灾害管理"系统，预测"灾后恢复能力"超出了当前气象数据的能力范围。

**（2）嵌入 DAG 可执行算子：打破静态本体局限**

DMDO 作为静态本体只能记录"日降水 = 254mm"，无法表达"这个 254mm 是怎么算出来的"。我们在每个 `sosa:ObservableProperty` 上同时存储三层表达：

| 字段 | 消费者 | 示例 |
|---|---|---|
| `description` | LLM/Agent | "日累计总降水，来自 DS2 的 tp 变量" |
| `expression` | 人类开发者 | "tp" |
| `dag` | DAG 解释器 | `{"op":"var","name":"tp"}` |

内置 16 个基础操作符（add、sub、mul、div、sqrt、sqr、pow、max、min、threshold、where 等），复杂指标如 `vpd_kpa` 的 DAG 包含 5 层 `exp → mul → sub` 计算节点。这使得图谱不仅是知识存储，更是计算推理的基础设施——这是 KWG/DMDO 不具备的独特能力。

**（3）知识层与数据层分离**

全量时空切片（35,200 格点 × 365 天 × 91 指标 ≈ 12 亿条）无法全部编入 RDF。采用双层架构：知识层（networkx + rdflib，7801 三元组）仅存 Indicator、DataSource、HazardType、Rule、Region、Event 的知识关系；数据层（xarray + NetCDF）管理所有时空栅格，通过 Indicator.id = NetCDF variable name 桥接。

**（4）阈值过滤的 SOSA 观测实例化**

按需创建 `sosa:Observation` 实例，仅将超过预警阈值的格点（如 `tmax_c >= 45°C`）写入 RDF，避免全量写入导致的图谱爆炸。2025-08-19 实测：`tmax_c >= 45` 产生 2622 条观测，而非 35,200 条。、

**（5）数据驱动权重学习**

利用 365 天观测数据结合已知极端日期标签（2025-08-19~23 山洪、2025-07-17/25 高温），通过 L1 正则化逻辑回归从数据中自动学习各检测条件的贡献权重。学习结果验证了因果指标（如 `cape` 对流能量、`vpd_kpa` 饱和水汽压差）的显著区分力，同时识别出衍生/后果指标（如 `heat_index_c` = f(T,RH)）在统计上的因果反转问题并将其从条件集中移除，确保检测规则中每项条件均为物理上合理的因果或并发因子。

### 1.3 图谱能力总览

构建的知识图谱覆盖 91 个气象指标、6 个数据源、4 种灾害类型和 4 条加权检测规则，知识层共 7801 个 RDF 三元组。四条检测规则均区分因果指标（提前信号）与并发指标（实况确认），后果指标排除在权重贡献之外；两条规则（山洪和沙尘）配置了数据缺测时的 fallback 降级策略。空间查询由 GeoSPARQL 提供几何过滤支持，时间推理由 OWL-Time 提供级联事件链支持，观测溯源由 PROV-O 提供完整推导链追溯。

### 1.4 W3C 标准合规性

| 标准 | 用途 |
|---|---|
| OWL 2 | 本体形式化（DMDO + 自定义扩展） |
| SOSA/SSN | Sensor/Observation/FeatureOfInterest/Procedure 完整建模 |
| GeoSPARQL | sf:Point 几何、geof:distance 空间过滤、sfIntersects 区域查询 |
| OWL-Time | time:Instant 时间点、time:Duration 持续时长、time:before/after 级联事件链 |
| PROV-O | prov:wasDerivedFrom 推导链、prov:wasGeneratedBy 生成活动、prov:wasAttributedTo 数据源归属 |
| QUDT | 所有观测值绑定国际单位（QUDT.DegreeCelsius, QUDT.MilliM, QUDT["M-PER-SEC"] 等） |

![image-20260709220827927](C:/Users/13936/AppData/Roaming/Typora/typora-user-images/image-20260709220827927.png)

---

## 2. 核心架构演进：从"重型框架"到"敏捷调度"

在智能体架构选型初期，团队深入评估了 LangGraph、AutoGen 等当下流行的多智能体编排框架。然而，经过架构推演与业务适配性分析，我们最终决定在中期 MVP 阶段，采用基于原生 Function Calling 的 while 循环控制流。

### 2.1 防灾业务的"非协商性" vs 多智能体的"非确定性"

以 AutoGen 为代表的框架主打"多 Agent 协作与辩论"，这种模式在创意写作或开放式编程中表现优异，但会引入极大的非确定性。MAZU 是一个面向高风险预警的系统，面对可能危及生命的突发性山洪或极端热浪，绝不能允许两个 AI 角色在后台进行不可控的"长篇辩论"。单体 Agent 配合精确的函数约束，能最大程度收敛大模型的自由发散。

![14702678f233ed98b2f4ea19443a6de8](E:/微信/xwechat_files/wxid_dh1hlhke5o5n22_2e39/temp/RWTemp/2026-07/14702678f233ed98b2f4ea19443a6de8.png)

### 2.2 严格的链式依赖与"奥卡姆剃刀"原理

从气象学第一性原理出发，本项目的预警主线是严谨且单向线性的：读取 .nc 气象张量 → 模型侦测异常 → RAG 查询知识图谱 → 结构化预警分发。这是一个标准的有向无环流程（DAG）。原生的 Function Calling 循环已经完美契合这种链式任务，且运行开销极低。

![d90d67e3c6893258488ee605e74d8eef](E:/微信/xwechat_files/wxid_dh1hlhke5o5n22_2e39/temp/RWTemp/2026-07/d90d67e3c6893258488ee605e74d8eef.jpg)

![1_GnMghFkLYP6LGbbGsPYLww](E:/微信/xwechat_files/wxid_dh1hlhke5o5n22_2e39/msg/file/2026-07/1_GnMghFkLYP6LGbbGsPYLww.webp)

### 2.3 透明度、可控性与白盒调试

使用原生的 while 循环，我们能够实现对大模型每一次调用工具的 inputs 和 outputs 进行 100% 的精确抓取与日志切片。这为排查大模型幻觉、校验知识图谱查询语句的准确性提供了极具穿透力的"白盒"调试视野。

### 2.4 时效性红线与 Token 冗余控制

"早期预警系统"的生命线在于低延迟。解析 160×220 的气象多通道特征矩阵已经需要一定的处理时间，若上层架构再因框架自身的臃肿导致响应迟缓，将直接违背系统"零差距"的核心业务定位。

---

## 3. 智能体的"双手"：核心 Tool Calling 机制设计

大语言模型既不具备直接解析 160×220 高维浮点矩阵的数学能力，也缺乏对沙特本地动态数据的实时感知。为了让系统从"聊天模型"蜕变为"业务执行器"，我们为其定制并注册了以下核心工具。

### 3.1 工具一：fetch_climate_tensor(date, region_bbox)

调用后台预处理流水线，读取指定范围的 .nc 数据，自动执行 NaN 缺失值清洗，并返回结构化的关键异常指标。

**设计动机**：一个包含 65 个变量的日度 .nc 文件体积庞大且包含大量缺失值。该工具在底层利用 Python 和 xarray 完成繁重的空间插值与掩码运算，最终只向大模型返回人类和 LLM 都能精准理解的标量总结（例如："2025-08-19，红海沿岸边界框内，daily_precip_total 峰值 143mm，flash_flood_risk 最高 4/5"）。

### 3.2 工具二：predict_climate_anomaly(tensor_data)

将清洗后的高维特征矩阵输入项目组研发的轻量化深度学习网络，获取极端灾害的初始发生概率。

**设计动机**：LLM 极其不擅长非线性的空间数学运算（例如在脑海中模拟卷积核的滑动）。沙特复杂地形下的对流降水演变规律，必须依赖专门的时空神经网络来捕捉。Agent 通过 Tool Calling 扮演"全科医生"，而预测模型是"CT 扫描仪"——全科医生只需看懂扫描仪给出的结果并据此下达最终诊断。

### 3.3 工具三：query_saudi_kg(event_type, location)

直接对接团队构建的沙特极端天气知识图谱（DMDO-OWL + SOSA/SSN + GeoSPARQL），通过生成并执行 SPARQL 查询语句，精准提取目标区域的基础设施实体属性与历史灾害应急预案。

**核心查询能力**：

| 查询类型 | SPARQL 示例 |
|---|---|
| 灾害依赖指标 | "山洪检测依赖哪些指标？" → `deo:hasHazardProperty` 路径遍历 |
| 指标推导链 | "heatwave_duration_days 怎么算的？" → `prov:wasDerivedFrom` 反向追溯 |
| 空间范围查询 | "红海沿岸 100km 内 tmax_c ≥ 48°C 的格点" → `geof:distance` + SOSA Observation |
| 时间序列查询 | "2025-08-19 至 08-23 的山洪级联事件" → `time:before/after` + `deo:possiblyCauses` |
| 溯源查询 | "t2m_anomaly_c 的数据来源是什么？" → `prov:wasAttributedTo` → DS2 + DS8 |

**设计动机**：通用大模型在给出防灾建议时往往陷入"正确的废话"（如：建议向高处撤离）。但通过调用此工具，Agent 能够获知暴雨中心坐标下游 5 公里处存在关键基础设施（港口/海水淡化厂），从而将预警简报从"泛泛而谈"提升为"具备工业级指导价值的靶向调度指令"。

### 3.4 工作流模拟：ReAct 动态自适应循环

在实际部署中，Agent 采用 ReAct (Reason + Act) 范式进入动态 while 循环：

1. **感知 (Observe)**：系统触发，Agent 接收到当日基础气象简报
2. **思考决策 (Reason & Decide)**：Agent 发现红海沿岸 ds10_max_1h 指标异常，自主决定先调用 predict_climate_anomaly 评估宏观风险
3. **行动与解析 (Act & Parse)**：深度学习模型返回边界概率（如 55% 风险）
4. **循环自适应 (Adaptive Loop)**：面对不确定性，Agent 进入下一轮循环——调用 query_saudi_kg 查询该地区地貌属性，图谱返回结果显示该地为极易汇水的漏斗型盆地
5. **归纳与输出 (Exit & Generate)**：Agent 综合"55% 模型概率 + 图谱提示的极高地形脆弱度"，将预警级别上调，生成高级别预警简报

正是这种基于 Tool Calling 的 ReAct 循环，使 MAZU 系统的决策大脑具备了真正的"自主权"与"容错自纠能力"。

![image-20260709212732154](C:/Users/13936/AppData/Roaming/Typora/typora-user-images/image-20260709212732154.png)

---

## 4. Prompt Engineering 与思维链 (CoT) 约束

在日常对话中，大语言模型的"发散性"是创造力的来源；但在 MAZU 灾害预警系统中，这种发散性却是致命的"幻觉"。我们摒弃了黑盒式的自然语言提示，转而将 Prompt Engineering 视为一种严谨的代码控制规约。

### 4.1 角色锚定与语料库对齐

在 System Prompt 中，明确赋予智能体"沙特国家气象中心 (NCM) 与国家灾害应急署联合首席指挥官"的系统级身份。这并非简单的"角色扮演游戏"，而是通过设定极高专业度的上下文，从根本上改变 LLM 生成词汇的概率分布，确保输出报告的官方权威感。

### 4.2 结构化思维链强制规范

强制要求模型在最终调用生成报告的 Tool 之前，必须按严格的 XML 标签格式展露其内部推理过程：

- `<Observation>`：罗列并陈述从 fetch_climate_tensor 工具获取的客观数值
- `<Reasoning>`：结合 query_saudi_kg 工具返回的图谱结果，分析地形、干涸河床 (Wadi) 走向以及对沙特特定行业（如海水淡化、港口航运）的潜在冲击
- `<Action>`：确认预警级别，准备输出最终简报

强制执行 Observation → Reasoning → Action 的线性推导，强迫模型先"看清数据"，再"结合常识"，最后"做出决策"。在工程落地中，如果某次预警出现偏差，开发者可以剥离出 `<Reasoning>` 标签中的文本进行溯源。

![image-20260709212812084](C:/Users/13936/AppData/Roaming/Typora/typora-user-images/image-20260709212812084.png)

### 4.3 边界防御与负向约束

在 System Prompt 尾部加入严格的"禁止行为清单"：绝对禁止编造虚假的气象站名称、禁止在知识图谱未返回结果时自行捏造沙特地理实体、禁止对地震等非气象类自然灾害进行预测。这构建了一道"认知护栏"，确保 Agent 的行为严格收敛在 MAZU 项目"多灾种气象预警"的业务边界内。

### 4.4 机器可读性强制

Agent 最终输出的预警简报，除人类可读的 Markdown 文本外，必须同时包含一段严格符合 Schema 定义的 JSON 数据（包含 alert_level, affected_regions, impact_sectors 等字段），打通自然语言与前端可视化渲染之间的技术壁垒。



---

## 附录：知识图谱技术规格

| 规格项 | 详情 |
|---|---|
| 三元组总数 | 7,801 |
| 知识层节点 | 111（91 Indicator + 6 DataSource + 4 HazardType + 6 Region + 4 Rule） |
| 知识层边 | 428（derived_from 42 + co_occurs_with 264 + sourced_from 101 + contributes_to 17 + detects 4） |
| 数据来源 | ERA5 再分析 (DS1/DS2/DS4)、GHCN 气候态 (DS8)、GPM IMERG 卫星降水 (DS10)、OSTIA 海温 (SST) |
| 时空覆盖 | 2025 全年 365 天，16.0°N–31.9°N, 34.0°E–55.9°E，0.1° 分辨率 |
| 数据层格式 | NetCDF4（35,200 格点 × 365 天 × 91 变量，~5.1 GB） |
| W3C 标准 | OWL 2、SOSA/SSN、GeoSPARQL、OWL-Time、PROV-O、QUDT |
| 后端存储 | rdflib 内存图 + Flask API（轻量化嵌入）+ xarray LRU 缓存 |
| 部署 | 单机 pip install，暂时无 GraphDB 集群 |
