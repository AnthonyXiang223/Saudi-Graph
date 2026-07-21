# MAZU 概率化门控（Probabilistic Gating）方法论

## 概述

MAZU 系统的四类灾害检测（极端高温、沙尘强风、山洪、沿海湿热）均采用 **"加权评分 + 概率化门控"** 双层判定架构：

- **加权评分层**：多个气象指标按权重累加，生成原始风险分数
- **概率化门控层**：一个基于极值统计/联合分布的指标，作为"二次校验"防止常规条件下误报

概率化门控的核心思想是：**仅当气象条件的极端程度在统计上显著偏离气候态时，才允许触发高级别预警**。它解决的痛点是沙特干热气候下"每天都很热/干/风大"——仅靠固定阈值（如 tmax≥42°C）会在夏季产生大量虚警。

---

## 四个概率化门控指标

### 1. `heat_gpd_prob` — 极端高温门控

| 属性 | 值 |
|---|---|
| **方法** | GPD POT（Peak-Over-Threshold，极值理论） |
| **输入** | `tmax_c`（日最高气温） |
| **气候态** | `heat_gpd_climatology.nc`（2025 年全年 tmax 经验分布） |
| **公式** | `P(T>tmax) = exp(-(tmax - threshold) / scale) × exceedance_rate`（仅当 tmax > threshold 时计算；否则 prob=1.0） |
| **判据** | 值越小越极端。`<= 0.05` 表示当前气温在气候态中属于 top 5% 极端事件 |
| **检测规则中的权重** | 0.30 |
| **检测规则中的角色** | `probabilistic_gate`（非 primary gate） |

**物理含义**：GPD 分布拟合 tmax 气候态的尾部（超过阈值的部分），计算当前 tmax 在尾部中的超越概率。与传统"高于气候态平均值 +X°C"的 anomaly 方法相比，GPD 更精确地刻画了极端事件的尾部行为，避免了沙漠区域夏季常态高温的误报。

**局限性**：基于 2025 年仅一年数据拟合 GPD，参数估计方差较大。沙漠格点的 GPD 阈值可能偏高。`<= 0.05` 门控保证低虚警率但可能漏报边界极端事件。

---

### 2. `precip_percentile` — 山洪门控

| 属性 | 值 |
|---|---|
| **方法** | Gamma CDF（百分位法） |
| **输入** | `daily_precip_total`（日降水量） |
| **气候态** | `precip_climatology.nc`（2025 年全年日降水经验分布） |
| **公式** | `P(X <= daily_precip_total \| Gamma(shape, scale)) × 100` |
| **判据** | 百分位越高越极端。`>= 80` 表示日降水超过气候态 80% 的值（即 top 20% 降水日） |
| **检测规则中的权重** | 0.30 |
| **检测规则中的角色** | `probabilistic_gate` |

**物理含义**：沙特大部分区域年降水极少（<50mm），使用绝对阈值（如"日降水 >=10mm"）在干旱区过于严苛。Gamma 分布拟合降水的正偏态分布特征，百分位法使检测对干旱/半干旱区的弱降水信号更敏感——即使绝对降水量不高，只要在本地气候态中属于极端，就能触发门控。

**局限性**：干旱区 P95 可能仅为 0-2mm，百分位分辨率低。山地区域建议使用 P80 而非 P95。Gamma 分布在极低降水区拟合可能不稳定。仅基于 2025 年单年数据，年际变率未体现。

---

### 3. `dust_joint_prob` — 沙尘强风门控

| 属性 | 值 |
|---|---|
| **方法** | Empirical Copula min（经验 Copula 最小算子，4 变量） |
| **输入** | `wind10_speed`（10m 风速）、`dewpoint_depression_c`（露点差）、`rh2m`（相对湿度，取反 `1-RH`）、`wind_shear_850_200`（高低空风切变） |
| **气候态** | `dust_joint_climatology.nc`（2025 年全年 4 变量经验分布） |
| **公式** | `P_dust = min(F_wind, F_dew, F_rh_rev, F_shear)`，其中 `F_*` 为各变量在气候态中的经验 CDF |
| **判据** | 值越高表示四个条件同时越极端。`>= 0.50` 触发门控（意味着所有四个条件都超过各自气候态中位数） |
| **检测规则中的权重** | 0.30 |
| **检测规则中的角色** | `probabilistic_gate` |

**物理含义**：沙尘暴的形成需要四个条件同时满足——强风（输送沙尘）、干燥大气（露点差大、RH 低）、强风切变（将沙尘抬升至高空）。单一条件达标不够，必须同时极端。Copula min 算子实现"取最短板"逻辑：四个 CDF 值的最小值决定了联合概率——任何一个条件不达标，整体概率就被压制。这避免了 Shamal 风季每天报沙尘的问题。

**局限性**：Copula min 假设 4 变量独立，忽略交互效应。`>= 0.50` 门控抑制 primary gate（wind10_speed 的 0.35 权重）。Haboob 型对流沙尘暴（CAPE 触发）不在此门控范围内。仅基于 2025 年单年数据。

---

### 4. `humid_heat_joint_prob` — 沿海湿热门控

| 属性 | 值 |
|---|---|
| **方法** | Empirical Copula min（经验 Copula 最小算子，4 变量） |
| **输入** | `sst_celsius`（海表温度）、`rh2m`（相对湿度）、`t2m_c`（2m 气温）、`wind10_speed`（10m 风速，取反 `1/wind`） |
| **气候态** | `humid_heat_joint_climatology.nc`（2025 年全年 4 变量经验分布） |
| **公式** | `P_humid = min(F_sst, F_rh, F_t2m, F_1/wind)`，其中 `F_*` 为各变量在气候态中的经验 CDF |
| **判据** | 值越**小**越极端。`<= 0.05` 触发门控（所有四个条件同时处于各自气候态的极端区间） |
| **检测规则中的权重** | 0.30 |
| **检测规则中的角色** | `probabilistic_gate` |

**物理含义**：沿海湿热需同时满足高海温（水汽源）、高湿度（RH≥60% 区分干热/湿热）、高温、低风速（静风使体感更闷热）。Copula min 同样实现"四条件同时极端"的判定。与沙尘门控不同，此处使用 `<= 0.05` 的低概率判据（而非 `>= 0.50` 的高概率），因为湿热事件在海湾地区相对罕见，需要更严格的"同时极端"标准。

**局限性**：仅对沿海有效（红海/波斯湾沿岸），内陆无 SST 数据则指标无意义。`<= 0.05` 门槛可能偏高，导致弱湿热事件漏报。1/wind 对静风天零风速附近不稳定。仅基于 2025 年单年数据。

---

## 概率化门控在检测规则中的集成方式

以极端高温 (`extreme_heat`) 为例，`rules.json` 中的检测规则：

```
primary gate:  heatwave_day_flag (weight=0.30) → 主门控，必须触发
prob_gate:     heat_gpd_prob    (weight=0.30) → 概率化门控，提供统计置信度
causal:        vpd_kpa          (weight=0.25) → 因果关系指标
causal:        t2m_anomaly_c    (weight=0.20) → 因果关系指标
concurrent:    tmax_c           (weight=0.15) → 并发实况指标
causal:        dewpoint_depression_c (weight=0.10) → 因果关系指标
```

**双层判定流程**：
1. 所有可用条件按权重加权评分：`score = Σ(条件触发程度 × 权重) / Σ(权重)`
2. Primary gate（`heatwave_day_flag`）未触发 → 整体严重度强制降级
3. 概率化门控（`heat_gpd_prob`）权重 0.30，在加权评分中贡献最大——这意味着即使其他指标达标，若统计上不够极端，总分也会被压下

**为什么所有概率化门控权重都是 0.30**：
- 它在加权总分中的角色是"统计校验器"，不是主要驱动
- 0.30 意味着它有足够的权重来抑制常规条件下的误报，但又不会大到独自决定检测结果
- Primary gate（如 `heatwave_day_flag`、`wind10_speed`、`sst_celsius`）才是"一票否决/通过"的开关
- 概率化门控是"加分项"——极端程度越高（probability 越低/越高），该项对总分的贡献越大

---

## 四种灾害的权重分布对比

| 灾害 | Primary Gate（权重） | Prob Gate（权重） | 核心 causal 指标 |
|---|---|---|---|
| 极端高温 | heatwave_day_flag (0.30) | heat_gpd_prob (0.30) | vpd_kpa (0.25), t2m_anomaly_c (0.20) |
| 沙尘强风 | wind10_speed (0.35) | dust_joint_prob (0.30) | dewpoint_depression_c (0.25), rh2m (0.25) |
| 山洪 | flash_flood_risk (0.30) | precip_percentile (0.30) | cape (0.25), ds10_max (0.15×2) |
| 沿海湿热 | sst_celsius (0.35) | humid_heat_joint_prob (0.30) | rh2m (0.35), t2m_c (0.20) |

每个规则的设计遵循同一套架构：**primary gate 控制准入 + probabilistic gate 控制置信度 + causal/concurrent 提供多维证据**。但各灾害的物理驱动机制不同，primary gate 的选择和权重分配随之不同。

---

## 参考文献

- Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer. — GPD POT 方法的理论基础
- Nelsen, R. B. (2006). *An Introduction to Copulas*. Springer. — Copula 联合概率的理论基础
- Wilks, D. S. (2011). *Statistical Methods in the Atmospheric Sciences*. Academic Press. — Gamma 分布拟合降水、经验 CDF 的方法参考
- 四个指标在 `schema/operators.json` 中的完整定义（含 DAG、公式、参数、局限性）
- 四个检测规则在 `schema/rules.json` 中的完整定义（含条件、权重、角色、严重度分级）
