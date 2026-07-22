# 沙特极端事件指标计算方法与分布洞察报告

## 1. 报告范围

本报告说明 `compute_indicators.py` 中已实现的沙特区域极端事件指标，包括每个指标的物理含义、输入来源、计算方法、单位解释，以及基于 2025 年已计算结果的初步分布洞察。

分析对象：

- 指标脚本：`compute_indicators.py`
- 已裁剪区域数据：`/Volumes/E/气象数据/saudi_region_output`
- DS8 气候态数据：`/Volumes/E/气象数据/8_SURF_CLI_GLB_1991_2020`
- 已计算指标输出：`/Volumes/E/气象数据/saudi_region_output/indicators/saudi_indicators_YYYYMMDD.nc`
- 时间范围：2025-01-01 至 2025-12-31，共 365 个日文件
- 单文件指标数：加入 DS8 异常类指标并重算后为 26-91 个数据变量，取决于当日可用的 DS2、DS4、DS10、多层分析变量和日最高温资料
- 空间范围：裁剪 bbox 为 16.0N-32.0N，34.0E-56.0E；指标文件实际网格为 16.0N-31.9N，34.0E-55.9E，约 0.1 度分辨率

本报告中的年度分布统计以“每日区域均值”为主，辅以全网格最大值。这样可以同时反映区域背景变化和局地极端信号。

## 2. 指标体系总览

指标体系面向沙特高影响天气，重点覆盖四类灾害机理：

| 灾害/场景 | 关键指标组 | 解释重点 |
|---|---|---|
| 暴雨和山洪 | 降水、降水距平、对流降水比例、CAPE、IVT、低层风、水汽辐合、短时卫星降水、`flash_flood_risk` | 判断是否具备强水汽、强对流和强降水叠加条件 |
| 极端高温和热健康风险 | 2 米气温距平、最高温距平、热浪持续天数、热指数、VPD、辐射收支、热胁迫代理指标、SST | 判断热力背景、干热程度、持续性、夜间降温能力和沿海热湿环境 |
| 沙尘和强风背景 | 10 米风、850/925 hPa 风、风切变、地形重力波应力、地表动量通量 | 判断近地面扬尘条件和天气系统动力背景 |
| 沿海和水汽输送 | SST、红海/波斯湾海温、IVT、PWAT、低层水汽输送 | 判断红海和波斯湾对降水、湿热和沿海风险的背景贡献 |

DS8 已确认为 1991-2020 全球站点气候态 normals，而不是格点气候态。脚本将 DS8 中落在沙特 bbox 内的站点逐日 normals 最近邻映射到指标网格，并写入站点距离指标。当前可用站点覆盖为：逐日 PRE 7 个站点、逐日 TAVG 45 个站点、逐日 TMAX 11 个站点，逐日 TMIN 在 bbox 内没有有效站点。

脚本同时记录了仍不能可靠计算的高级指标：

- LCL/LFC/EL：需要完整垂直热力廓线。
- SPI：需要多年降水时间序列或分布参数；DS8 只有均值 normals，不能单独支撑正式 SPI。
- 500 hPa 位势高度距平和副高强度：需要压力层位势高度多年气候态；DS8 是地面站点气候态，不能支撑该项。
- PET/ET0：需要经过验证的气压、辐射、风、温湿联合流程。

## 3. 指标含义与计算方法

### 3.1 月尺度地表和能量指标

这些指标来自 DS1 月尺度产品，并在每天计算时按对应月份写入日指标文件。因此它们反映月背景，不代表逐日变化。降水类月指标使用 `MONTH_ACC`，辐射、热通量和风应力等月背景指标使用 `MONTH_AVG`。

| 指标 | 含义 | 计算方法 | 单位 | 使用解释 |
|---|---|---|---|---|
| `monthly_precip_total` | 月累计总降水 | `tp` from `MONTH_ACC` | mm | 月尺度降水总量主指标 |
| `monthly_precip_mmday` | 月平均降水强度 | `tp / days_in_month` | mm/day | 将月累计降水折算为日平均强度 |
| `monthly_convective_precip` | 月累计对流降水 | `acpcp` from `MONTH_ACC` | mm | 月尺度对流降水贡献 |
| `monthly_large_scale_precip` | 月累计非对流降水 | `ncpcp` from `MONTH_ACC` | mm | 月尺度层状/大尺度降水贡献 |
| `monthly_convective_precip_ratio` | 月对流降水占比 | `acpcp / tp` | 1 | 越高说明降水更偏对流性，需注意 `tp` 近零时不稳定 |
| `monthly_bowen_ratio` | 月 Bowen 比 | `avg_ishf / avg_slhtf` | 1 | 感热相对潜热越高，地表越干热；分母小会放大异常 |
| `monthly_sw_net` | 月净短波辐射 | `sdswrf - suswrf` | W m-2 | 表示地表吸收的太阳短波能量 |
| `monthly_lw_net` | 月净长波辐射 | `sdlwrf - sulwrf` | W m-2 | 通常为负，表示地表长波净损失 |
| `monthly_net_radiation` | 月净辐射 | `monthly_sw_net + monthly_lw_net` | W m-2 | 地表能量收支背景 |
| `monthly_heat_stress_index` | 月地表热胁迫代理 | `avg_ishf + (1 - avg_al/100) * sdswrf` | W m-2 | 结合感热通量和吸收短波，越高越偏热胁迫 |
| `monthly_wind_stress_mag` | 月地表风应力强度 | `sqrt(avg_utaua^2 + avg_vtaua^2)` | N m-2 | 反映地表动量交换和大尺度风背景 |
| `monthly_orographic_stress` | 月地形重力波应力 | `sqrt(iegwss^2 + ingwss^2)` | N m-2 | 表征地形扰动和山地动力作用 |
| `monthly_uvb_flux` | 月 UV-B 下行通量 | `duvb` | W m-2 | 健康风险和晴空辐射背景 |
| `monthly_uvb_clear_ratio` | 晴空/全天 UV-B 比 | `cduvb / duvb` | 1 | 衡量云和气溶胶对 UV-B 的调制 |

### 3.2 云量指标

| 指标 | 含义 | 计算方法 | 单位 | 使用解释 |
|---|---|---|---|---|
| `total_cloud_cover` | 总云量 | `tcc` | % | 可辅助解释辐射、降水和日温变化 |
| `low_cloud_cover` | 低云量 | `tcc_lowCloudLayer_0_0` | % | 与近地层湿度、低云和沿海天气有关 |
| `middle_cloud_cover` | 中云量 | `tcc_middleCloudLayer_0_0` | % | 与层状降水和中层湿度有关 |
| `high_cloud_cover` | 高云量 | `tcc_highCloudLayer_0_0` | % | 与高层云、对流砧云和大尺度环流有关 |

### 3.3 日尺度降水、辐射和地表能量指标

| 指标 | 含义 | 计算方法 | 单位 | 使用解释 |
|---|---|---|---|---|
| `precip_mmday` | 日平均降水率换算 | `prate * 86400` | mm/day | 来自日平均降水率，目前有效日很少，不宜作为主降水指标 |
| `convective_precip_ratio` | 日对流降水占比 | `cpr / prate` | 1 | 表示降水对流性，需和总降水共同解释 |
| `daily_precip_total` | 日累计总降水 | `tp` | mm | 当前最可靠的日降水主指标 |
| `daily_convective_precip` | 日累计对流降水 | `acpcp` | mm | 表征强对流贡献 |
| `daily_large_scale_precip` | 日累计非对流降水 | `ncpcp` | mm | 表征层状或大尺度降水贡献 |
| `bowen_ratio` | 日 Bowen 比 | `avg_ishf / avg_slhtf` | 1 | 极端值较多，建议加分母阈值后再用于建模 |
| `sw_net` | 日净短波辐射 | `sdswrf - suswrf` | W m-2 | 白天太阳加热背景 |
| `lw_net` | 日净长波辐射 | `sdlwrf - sulwrf` | W m-2 | 夜间和地表长波冷却背景 |
| `net_radiation` | 日净辐射 | `sw_net + lw_net` | W m-2 | 地表可用能量 |
| `heat_stress_index` | 日地表热胁迫代理 | `avg_ishf + (1 - avg_al/100) * sdswrf` | W m-2 | 越高代表吸收太阳能和感热通量越强 |

### 3.4 日尺度近地面热湿和风指标

| 指标 | 含义 | 计算方法 | 单位 | 使用解释 |
|---|---|---|---|---|
| `t2m_c` | 2 米气温 | `t2m - 273.15` | degC | 日平均热背景 |
| `d2m_c` | 2 米露点温度 | `d2m - 273.15` | degC | 近地面水汽含量代理 |
| `dewpoint_depression_c` | 露点差 | `t2m_c - d2m_c` | degC | 越大表示越干燥，蒸发潜力越强 |
| `rh2m` | 2 米相对湿度 | `r2` | % | 湿热和云降水环境解释变量 |
| `vpd_kpa` | 饱和水汽压差 | `es(T) * (1 - RH/100)` | kPa | 越高代表空气越干、蒸散需求越强 |
| `heat_index_c` | 热指数 | Rothfusz 回归，低温/低湿时退回气温 | degC | 健康热风险指标，湿热条件下高于气温 |
| `sh2m` | 2 米比湿 | `sh2` | kg kg-1 | 近地面绝对水汽 |
| `apparent_temp_c` | 体感温度 | 清洗后 `aptmp - 273.15` | degC | 体感热风险辅助指标，已在派生前屏蔽填充值 |
| `wind10_speed` | 10 米风速 | `sqrt(u10^2 + v10^2)` | m s-1 | 近地面强风、沙尘和热输送背景 |
| `tmax_c` | 日最高 2 米气温 | `tmax - 273.15` | degC | 高温日筛查主指标 |
| `tmin_c` | 日最低 2 米气温 | `tmin - 273.15` | degC | 夜间热胁迫和持续高温 |
| `diurnal_temp_range_c` | 日较差 | `tmax_c - tmin_c` | degC | 反映昼夜热力差异和夜间冷却能力 |
| `qmax_2m` | 日最大 2 米比湿 | `qmax` | kg kg-1 | 湿热上限 |
| `qmin_2m` | 日最小 2 米比湿 | `qmin` | kg kg-1 | 干燥背景下限 |

### 3.5 DS8 气候态距平和热浪持续指标

这些指标来自 DS8 1991-2020 全球站点气候态 normals。脚本只使用落在当前沙特 bbox 内的站点，将逐日 normals 用最近邻方式映射到指标网格，并同时写入最近站距离。由于 DS8 不是格点气候态，距平指标更适合作为预警业务中的异常背景参考，不应解释为精细格点气候真值。

| 指标 | 含义 | 计算方法 | 单位 | 使用解释 |
|---|---|---|---|---|
| `daily_precip_climatology` | 1991-2020 逐日降水气候态 | DS8 `PRE/MDAY` 最近站逐日 normal | mm/day | 站点气候态映射到网格后的降水背景 |
| `daily_precip_climatology_station_distance_km` | 降水气候态最近站距离 | 网格点到所用 DS8 PRE 站的球面距离 | km | 距离越大，局地解释不确定性越高 |
| `daily_precip_anomaly` | 日降水距平 | `daily_precip_total - daily_precip_climatology` | mm | 正值表示高于 1991-2020 同日站点气候态 |
| `daily_precip_anomaly_ratio` | 日降水距平率 | `(daily_precip_total - climatology) / climatology` | 1 | 干旱区分母很小时会放大，解释时需设降水气候态阈值 |
| `t2m_climatology_c` | 1991-2020 日平均气温气候态 | DS8 `TAVG/MDAY` 最近站逐日 normal | degC | 与 `t2m_c` 对比使用 |
| `t2m_climatology_station_distance_km` | TAVG 气候态最近站距离 | 网格点到所用 DS8 TAVG 站的球面距离 | km | 用于评估温度距平空间代表性 |
| `t2m_anomaly_c` | 2 米气温距平 | `t2m_c - t2m_climatology_c` | degC | 正值表示日平均气温偏暖 |
| `tmax_climatology_c` | 1991-2020 日最高温气候态 | DS8 `TMAX/MDAY` 最近站逐日 normal | degC | 与 `tmax_c` 对比使用 |
| `tmax_climatology_station_distance_km` | TMAX 气候态最近站距离 | 网格点到所用 DS8 TMAX 站的球面距离 | km | TMAX 站点较少，必须和距平一起查看 |
| `tmax_anomaly_c` | 日最高温距平 | `tmax_c - tmax_climatology_c` | degC | 正值表示最高温偏暖 |
| `heatwave_day_flag` | 热浪日标志 | `tmax_c >= max(40 degC, tmax_climatology_c + 5 degC)` | flag | 同时要求绝对高温和相对气候态偏热 |
| `heatwave_duration_days` | 热浪持续天数 | 对 `heatwave_day_flag >= 1` 逐格逐日连续计数 | days | 反映热浪持续性；缺少 TMAX 的日期不回写 |

`spi_status` 和 `geopotential_height500_anomaly_status` 写在每个 NetCDF 文件属性中，当前均为 `not_computed`。这不是漏算，而是为了避免把缺少历史分布的降水距平伪装成 SPI，或用地面气候态伪装成 500 hPa 位势高度距平。

### 3.6 对流稳定度、地形和气压指标

| 指标 | 含义 | 计算方法 | 单位 | 使用解释 |
|---|---|---|---|---|
| `cape` | 对流有效位能 | 若源变量带额外层维度，先对非水平维取最大值 | J kg-1 | 越高代表潜在对流能量越强，重算后为二维水平网格 |
| `cin` | 对流抑制能量 | 若源变量带额外层维度，先对非水平维取最小值 | J kg-1 | 越负代表抑制越强，接近 0 时更容易触发对流 |
| `surface_lifted_index` | 地面抬升指数 | 若源变量带额外层维度，先对非水平维取最小值 | K | 越低越不稳定 |
| `best_lifted_index` | 最佳四层抬升指数 | 若源变量带额外层维度，先对非水平维取最小值 | K | 越低越有利于强对流 |
| `surface_pressure` | 地面气压 | `sp` | Pa | 低压和天气系统背景 |
| `orography` | 地形高度 | `orog` | m | 静态地形，用于解释山地抬升和降水空间分布 |

### 3.7 多层动力和水汽输送指标

| 指标 | 含义 | 计算方法 | 单位 | 使用解释 |
|---|---|---|---|---|
| `pwat` | 可降水量 | `pwat` | kg m-2 | 柱总水汽，暴雨背景核心指标 |
| `ivt_u` | 整层水汽输送纬向分量 | `1/g * integral(q*u dp)` | kg m-1 s-1 | 正值偏东向输送 |
| `ivt_v` | 整层水汽输送经向分量 | `1/g * integral(q*v dp)` | kg m-1 s-1 | 正值偏北向输送 |
| `ivt` | 整层水汽输送强度 | `sqrt(ivt_u^2 + ivt_v^2)` | kg m-1 s-1 | 强水汽通道识别 |
| `ivt_divergence` | 水汽输送散度 | `d(ivt_u)/dx + d(ivt_v)/dy` | kg m-2 s-1 | 正值代表水汽发散 |
| `ivt_convergence` | 水汽输送辐合 | `-ivt_divergence` | kg m-2 s-1 | 正值代表水汽汇聚，有利降水 |
| `wind925_speed` | 925 hPa 风速 | `sqrt(u925^2 + v925^2)` | m s-1 | 低层输送和近地层动力 |
| `moisture_transport925` | 925 hPa 水汽输送强度 | `q925 * wind925_speed` | m s-1 | 低层水汽输送代理 |
| `wind850_speed` | 850 hPa 风速 | `sqrt(u850^2 + v850^2)` | m s-1 | 低空急流和水汽输送 |
| `moisture_transport850` | 850 hPa 水汽输送强度 | `q850 * wind850_speed` | m s-1 | 850 hPa 水汽输送代理 |
| `jet300_speed` | 300 hPa 风速 | `sqrt(u300^2 + v300^2)` | m s-1 | 高空急流背景 |
| `jet200_speed` | 200 hPa 风速 | `sqrt(u200^2 + v200^2)` | m s-1 | 副热带急流和高空动力 |
| `wind_shear_850_300` | 850-300 hPa 垂直风切变 | `sqrt((u300-u850)^2 + (v300-v850)^2)` | m s-1 | 强对流组织化条件 |
| `wind_shear_850_200` | 850-200 hPa 垂直风切变 | `sqrt((u200-u850)^2 + (v200-v850)^2)` | m s-1 | 深层风切变 |
| `relative_vorticity850` | 850 hPa 相对涡度 | `dv/dx - du/dy` | s-1 | 低层旋转和天气扰动 |
| `divergence850` | 850 hPa 水平散度 | `du/dx + dv/dy` | s-1 | 正值发散，负值辐合 |
| `absolute_vorticity850` | 850 hPa 绝对涡度 | `absv` | s-1 | 综合地转和相对旋转背景 |
| `omega700` | 700 hPa 垂直速度 | `omega at 700 hPa` | Pa s-1 | 负值表示上升运动 |
| `omega500` | 500 hPa 垂直速度 | `omega at 500 hPa` | Pa s-1 | 中层上升/下沉运动 |
| `geopotential_height500` | 500 hPa 位势高度 | `gh500` | gpm | 中层环流和脊槽背景 |

### 3.8 海温和卫星降水指标

| 指标 | 含义 | 计算方法 | 单位 | 使用解释 |
|---|---|---|---|---|
| `sst_celsius` | 海表温度 | `analysed_sst`，源值大于 200 时转为 `K - 273.15` | degC | 红海和波斯湾热湿背景 |
| `ds10_daily_total` | DS10 卫星日降水 | `daily_total`，最近邻对齐到指标网格 | mm | 高频卫星降水日聚合，可与 DS2 日累计降水交叉验证 |
| `ds10_max_30min` | DS10 30 分钟最大降水 | `max_30min`，最近邻对齐到指标网格 | mm | 短历时强降水核心指标 |
| `ds10_max_1h` | DS10 1 小时最大降水 | `max_1h`，最近邻对齐到指标网格 | mm | 山洪短历时降水核心候选，并已纳入 `flash_flood_risk` |
| `ds10_max_3h` | DS10 3 小时最大降水 | `max_3h` | mm | 短时累计降水 |
| `ds10_max_6h` | DS10 6 小时最大降水 | `max_6h` | mm | 半日尺度强降水 |
| `ds10_rainy_steps` | DS10 有雨时次数 | `rainy_steps` | steps | 表示日内降水持续性 |
| `ds10_ds2_precip_diff` | DS10 与 DS2 日降水差值 | `ds10_daily_total - daily_precip_total` | mm | 正值表示 DS10 高于 DS2 |
| `ds10_ds2_precip_ratio` | DS10/DS2 日降水比值 | `ds10_daily_total / daily_precip_total` | 1 | 用于识别两套降水估计偏差 |
| `ds10_ds2_heavy_rain_overlap` | DS10 与 DS2 强降水重叠 | 两者均 `>= 10 mm` 时为 1 | flag | 用于筛选两套数据共同支持的强降水网格 |

### 3.9 综合山洪筛查指标

| 指标 | 含义 | 计算方法 | 单位 | 使用解释 |
|---|---|---|---|---|
| `flash_flood_risk` | 山洪初筛分数 | 可用条件阈值标志求和 | score | 分数越高表示更多山洪触发因子同时满足 |

当前阈值项包括：

- `daily_precip_total >= 10 mm`，若无该指标则用 `precip_mmday >= 10 mm/day`。
- `convective_precip_ratio >= 0.5`。
- `cape >= 1000 J kg-1`。
- `wind10_speed >= 10 m s-1`。
- `ds10_max_1h >= 10 mm`。

DS10 指标已修正并重算，`ds10_max_1h >= 10 mm` 现在会实际参与 `flash_flood_risk`。`cape` 也已从带额外层维度的源变量折算为二维水平网格，因此 `cape >= 1000 J kg-1` 现在会实际参与评分。风险评分只使用二维水平网格指标，避免带垂直层的变量把评分广播成三维。

当 DS10 缺测时，`daily_precip_total` 可以作为日总降水替代参考，但不能替代 `ds10_max_30min`、`ds10_max_1h` 等短历时峰值；此时应将短时强降水项解释为不可用，而不是用日累计降水硬填。

## 4. 2025 年指标分布洞察

### 4.1 数据覆盖

| 指标类别 | 有效覆盖 |
|---|---:|
| 指标文件 | 365 天 |
| 日累计降水 | 363 天 |
| 2 米气温 | 362 天 |
| 日最高/最低温 | 363 天 |
| CAPE 和多层动力指标 | 约 291-293 天 |
| SST | 365 天 |
| DS10 卫星降水指标 NetCDF | 273 天有效；有效日每个 DS10 降水变量均为 35200 个网格 |
| DS1 月降水指标 | 已按 `MONTH_ACC tp/acpcp/ncpcp` 重跑，365 个日指标文件均有月降水背景；`monthly_precip_mmday` 每日文件有效网格数为 35200 |
| DS8 气候态状态 | 365 天均自动发现并使用 `/Volumes/E/气象数据/8_SURF_CLI_GLB_1991_2020` |
| 降水距平 | 363 天；与 `daily_precip_total` 覆盖一致 |
| TAVG/TMAX 温度距平 | `t2m_anomaly_c` 362 天，`tmax_anomaly_c` 363 天 |
| 热浪持续天数 | 363 天；缺少 TMAX 的日期不回写 |
| SPI 和 500 hPa 高度距平 | 365 天均在属性中标记为 `not_computed`，原因分别是缺少多年降水分布和压力层气候态 |

### 4.2 核心指标年度分布表

下表按每日区域均值统计，`max_grid` 为全年所有网格中的最大值和发生日期。

| 指标 | 有效天数 | 日区域均值 | 日区域均值 95 分位 | 最大日区域均值 | max_grid |
|---|---:|---:|---:|---:|---|
| `daily_precip_total` | 363 | 0.23 mm | 1.01 mm | 4.06 mm | 254.9 mm @ 20250823 |
| `daily_precip_climatology` | 363 | 0.55 mm | 1.79 mm | 5.47 mm | 16.8 mm @ 20250112 |
| `daily_precip_anomaly` | 363 | -0.33 mm | 0.45 mm | 3.66 mm | 254.9 mm @ 20250823 |
| `daily_convective_precip` | 363 | 0.16 mm | 0.73 mm | 1.65 mm | 236.5 mm @ 20250814 |
| `daily_large_scale_precip` | 363 | 0.07 mm | 0.31 mm | 2.60 mm | 116.0 mm @ 20251216 |
| `t2m_c` | 362 | 27.09 degC | 35.11 degC | 35.98 degC | 45.90 degC @ 20250717 |
| `t2m_anomaly_c` | 362 | 0.89 degC | 2.95 degC | 4.27 degC | 17.77 degC @ 20250403 |
| `tmax_c` | 363 | 31.92 degC | 40.45 degC | 41.39 degC | 53.75 degC @ 20250725 |
| `tmax_anomaly_c` | 363 | -0.13 degC | 2.24 degC | 4.97 degC | 21.92 degC @ 20250223 |
| `heatwave_day_flag` | 363 | 0.050 | 0.192 | 0.303 | 1 @ 20250329 |
| `heatwave_duration_days` | 363 | 0.35 days | 1.61 days | 3.18 days | 72 days @ 20250816 |
| `tmin_c` | 363 | 21.94 degC | 29.30 degC | 30.14 degC | 41.11 degC @ 20250717 |
| `diurnal_temp_range_c` | 362 | 9.99 degC | 11.73 degC | 12.66 degC | 25.69 degC @ 20250620 |
| `rh2m` | 364 | 32.54 % | 48.02 % | 57.31 % | 100.0 % @ 20251228 |
| `vpd_kpa` | 362 | 2.76 kPa | 4.46 kPa | 4.87 kPa | 9.42 kPa @ 20250717 |
| `heat_index_c` | 362 | 27.67 degC | 36.55 degC | 37.57 degC | 54.67 degC @ 20250816 |
| `wind10_speed` | 364 | 3.52 m/s | 4.53 m/s | 5.58 m/s | 20.73 m/s @ 20250726 |
| `cape` | 293 | 228 J/kg | 597 J/kg | 752 J/kg | 5196 J/kg @ 20250831 |
| `pwat` | 292 | 17.98 kg/m2 | 27.51 kg/m2 | 33.94 kg/m2 | 68.63 kg/m2 @ 20250820 |
| `ivt` | 291 | 88.57 kg/m/s | 139.36 kg/m/s | 170.60 kg/m/s | 728.40 kg/m/s @ 20250819 |
| `wind850_speed` | 291 | 5.23 m/s | 6.95 m/s | 7.91 m/s | 26.75 m/s @ 20250820 |
| `jet200_speed` | 292 | 21.50 m/s | 40.88 m/s | 55.94 m/s | 85.84 m/s @ 20251228 |
| `wind_shear_850_200` | 291 | 22.98 m/s | 40.81 m/s | 51.55 m/s | 83.33 m/s @ 20251228 |
| `omega500` | 292 | 0.015 Pa/s | 0.055 Pa/s | 0.145 Pa/s | 3.38 Pa/s @ 20250415 |
| `geopotential_height500` | 293 | 5868 gpm | 5912 gpm | 5926 gpm | 5970 gpm @ 20250814 |
| `sst_celsius` | 365 | 27.78 degC | 31.58 degC | 32.03 degC | 36.90 degC @ 20250822 |
| `flash_flood_risk` | 365 | 0.079 | 0.244 | 0.286 | 4 @ 20250819 |

### 4.3 降水和山洪信号

2025 年沙特区域平均降水整体偏低，日累计降水区域均值仅约 0.23 mm，95 分位约 1.01 mm。这符合沙特多数区域干旱少雨的气候背景。但局地极端仍然明显，`daily_precip_total` 全网格最大值达到 254.9 mm，发生在 2025-08-23。对流降水最大网格值达到 236.5 mm，发生在 2025-08-14，说明局地强对流可以远远高于区域均值。

DS1 月降水背景已用 `MONTH_ACC` 重新计算，`monthly_precip_mmday` 全年 365 个日指标文件均有效，按日文件统计的区域均值约 0.0076 mm/day，最高月背景为 202512，区域均值约 0.0255 mm/day。旧字段 `precip_mmday` 仍来自 `MONTH_AVG prate * 86400`，有效覆盖很少，不应作为月降水主指标。

新增的 `daily_precip_anomaly` 显示，按 DS8 站点气候态映射后的背景，2025 年多数日期区域平均略偏干，日区域均值全年平均约 -0.33 mm。需要注意，DS8 逐日 PRE 在沙特 bbox 内只有 7 个有效站点，降水气候态最近站距离区域均值约 770 km，最远超过 1700 km。因此降水距平适合做“是否异常偏湿/偏干”的背景参考，局地精细预警仍应优先看 `daily_precip_total`、DS10 短历时降水和水汽/对流条件。

`flash_flood_risk` 的区域均值仍然偏低，全年平均约 0.079，最大日区域均值约 0.286，发生在强水汽输送和强降水过程附近；网格最大分数达到 4，发生在 2025-08-19。分数升高主要来自 CAPE 已被正确折算为二维并参与评分。这说明该风险分数更适合做空间网格筛查，而不是只看区域平均。后续可将其与流域、城市、山谷和道路承灾体叠加，提取局地风险热点。

### 4.4 高温和干热信号

`t2m_c` 日区域均值全年平均约 27.09 degC，95 分位约 35.11 degC，最大日区域均值为 35.98 degC。日最高温 `tmax_c` 的最大日区域均值达到 41.39 degC，局地最大值达到 53.75 degC，发生在 2025-07-25。`tmin_c` 局地最大值也达到 41.11 degC，发生在 2025-07-17，提示夜间热胁迫值得关注。

`vpd_kpa` 重算后已在派生前屏蔽填充值和明显非物理输入，区域均值约 2.75 kPa，95 分位约 4.46 kPa，最大局地值约 9.42 kPa，表现出强干热蒸发需求。`heat_index_c` 的局地最大值达到 54.67 degC，发生在 2025-08-16，说明沿海或湿热区域的体感热风险可能显著高于单纯气温。

与 DS8 气候态相比，`t2m_anomaly_c` 的区域平均全年约 +0.89 degC，95 分位约 +2.95 degC；`tmax_anomaly_c` 区域平均约 -0.13 degC，但 95 分位约 +2.24 degC，说明最高温异常更偏向阶段性和局地性。`heatwave_day_flag` 年平均区域占比约 5.0%，`heatwave_duration_days` 的局地最大连续天数为 72 天，发生在 2025-08-16 的某些网格。这类持续性指标比单日最高温更适合进入预警业务，因为它直接表达承灾体累积暴露。

### 4.5 对流、水汽输送和动力背景

`cape` 有效覆盖 293 天，重算后已统一为二维水平网格。区域均值约 228 J/kg，95 分位约 597 J/kg，但局地最大值达到 5196 J/kg，发生在 2025-08-31。这说明多数日期区域平均对流能量不高，但局地强不稳定环境存在。

`pwat` 区域均值约 17.98 kg/m2，95 分位约 27.51 kg/m2，局地最大值 68.63 kg/m2，发生在 2025-08-20。`ivt` 区域均值约 88.57 kg/m/s，95 分位约 139.36 kg/m/s，局地最大值 728.40 kg/m/s，发生在 2025-08-19。2025-08-19 至 2025-08-23 附近同时出现较强 IVT、局地水汽辐合和强降水，是值得进一步复盘的水汽输送型降水过程。

高空动力方面，`jet200_speed` 和 `wind_shear_850_200` 在 2025-12-28 前后达到全年高值，最大日区域均值分别约 55.94 m/s 和 51.55 m/s。这类冬季强切变和高空急流背景，对沙特北部冷季降水、沙尘和锋面系统有解释价值。

### 4.6 海温背景

`sst_celsius` 全年有效，区域均值约 27.78 degC，95 分位约 31.58 degC，最大日区域均值约 32.03 degC，局地最大值 36.90 degC，发生在 2025-08-22。红海和波斯湾周边高海温可为沿海湿热、低层水汽和局地强降水提供背景条件。脚本还在 NetCDF 属性中保存了红海和波斯湾区域 SST 均值/最大值，后续报告可以进一步按海区拆分。

## 5. 数据质量和解释限制

### 5.1 DS1 月降水来源修正

此前月降水覆盖异常的原因不是 DS1 裁剪坏了，而是旧脚本使用了 `MONTH_AVG` 中多数月份缺测的 `prate/cpr`；与此同时，裁剪后的 DS1 `MONTH_ACC` 文件中 `tp/acpcp/ncpcp` 在 12 个月都有有效值。

`compute_indicators.py` 已修正为加载 `ds1_acc`，并用 `tp/acpcp/ncpcp` 计算 `monthly_precip_total`、`monthly_precip_mmday`、`monthly_convective_precip`、`monthly_large_scale_precip` 和 `monthly_convective_precip_ratio`。当前 `/Volumes/E/气象数据/saudi_region_output/indicators` 已完成重跑，365 个日指标文件中的 `monthly_precip_total` 和 `monthly_precip_mmday` 均有效，月降水相关指标每个文件恢复为 35200 个有效网格。

### 5.2 DS10 卫星降水网格对齐和交叉验证

此前 DS10 指标写入 NetCDF 后全 NaN，根因是 DS10 NPZ 网格相对 DS2/指标网格存在半个格点偏移，且纬度方向不同；xarray 合并时按坐标精确对齐，导致 DS10 数组被对齐为空。`compute_indicators.py` 已改为将 DS10 日聚合结果最近邻重采样到指标网格，并在写入前保持 `latitude/longitude` 坐标一致。

当前重算后的指标中，`ds10_daily_total`、`ds10_max_30min`、`ds10_max_1h`、`ds10_max_3h`、`ds10_max_6h`、`ds10_rainy_steps` 均有 273 天有效，有效日每个变量均有 35200 个网格。脚本还新增 `ds10_ds2_precip_diff`、`ds10_ds2_precip_ratio` 和 `ds10_ds2_heavy_rain_overlap`，用于把 DS10 卫星日降水与 DS2 日累计降水做交叉验证。`flash_flood_risk` 已纳入 `ds10_max_1h`，并限制为二维水平网格评分。

DS10 缺测日期可以用 DS2 `daily_precip_total` 作为日总降水替代参考；但 DS2 日累计不能替代 DS10 的短历时最大降水。因此 202510-202512 这类 DS10 缺测时段，山洪评分中的短时强降水项应视为缺测，而不是用日降水强行填补。

### 5.3 派生热湿指标的填充值污染修正

旧版 `vpd_kpa` 和 `apparent_temp_c` 在 2025-01-02、2025-01-30 出现明显异常：

| 日期 | 异常指标 | 表现 |
|---|---|---|
| 20250102 | `vpd_kpa` | 日区域均值约 1.09e7 kPa |
| 20250102 | `apparent_temp_c` | 日区域均值约 -5.37e8 degC |
| 20250130 | `vpd_kpa` | 日区域均值约 1.16e7 kPa |
| 20250130 | `apparent_temp_c` | 日区域均值约 -5.37e8 degC |

这些异常不符合物理意义，根因是派生计算前没有先清理 GRIB 缺测填充值。`compute_indicators.py` 现已在近地面温度、湿度、体感温度、比湿和风速等输入变量上先屏蔽 `NaN`、`inf`、超大填充值和明显非物理范围，再计算 `vpd_kpa`、`heat_index_c`、`apparent_temp_c` 等派生指标。正式重算后，`t2m_c/vpd_kpa/heat_index_c/apparent_temp_c` 的物理范围异常天数均为 0。

### 5.4 CAPE 二维化和风险评分修正

真实指标文件中，部分 `cape` 源变量带 `pressureFromGroundLayer` 这类额外维度。旧版为避免把 `flash_flood_risk` 广播成三维，风险评分只接受二维变量，因此 CAPE 项没有实际参与评分。

当前脚本已将 `cape` 在非水平维度上取最大值并折算为 `latitude, longitude` 二维网格；`cin` 和 lifted index 类变量也按物理解释折算为二维。正式重算后，`cape` 293 天有效且维度为二维，`flash_flood_risk` 365 天有效，网格最高分由旧版 3 更新为 4。

### 5.5 比值类指标需要分母阈值

`monthly_bowen_ratio`、`bowen_ratio`、`convective_precip_ratio` 这类比值指标在分母接近 0 时容易出现极端值。脚本已用 `abs(denominator) > 1e-12` 做初步保护，但对于气象解释和建模，建议额外设置物理阈值，例如潜热通量绝对值过小时不解释 Bowen 比。

### 5.6 DS8 气候态的代表性限制

数据 8 是多年气候态数据，具体为 1991-2020 全球地面站点 normals。它可以用于构造异常背景，但不是与 DS2 同分辨率的格点气候场。当前脚本采用最近邻站点映射，且将最近站距离写入 NetCDF，因此使用距平时应同时检查距离指标。

沙特 bbox 内 DS8 逐日站点覆盖并不均匀：PRE 只有 7 个有效站点，TMAX 约 11 个有效站点，TAVG 约 45 个有效站点，TMIN 没有可用站点。因此目前新增的是降水距平、TAVG/TMAX 温度距平和基于 TMAX 的热浪持续天数，没有新增 TMIN 距平或夜间热浪持续指标。

`daily_precip_anomaly_ratio` 在干旱区尤其容易被很小的气候态降水分母放大。它适合做异常提示或排序辅助，不应单独作为预警阈值。正式业务阈值建议优先使用 `daily_precip_anomaly`、`daily_precip_total`、DS10 短历时降水和水汽/对流条件组合。

### 5.7 SPI 和 500 hPa 高度距平未计算的原因

SPI 不是“当前降水减去平均降水”。正式 SPI 需要多年降水时间序列拟合概率分布，或至少需要可复现的分布参数。DS8 只提供 1991-2020 的均值 normals，缺少方差、分布形状和逐年序列，因此本轮没有伪计算 SPI，并在 365 个指标文件属性中写入 `spi_status=not_computed`。

500 hPa 高度距平需要压力层位势高度的多年气候态或历史序列。DS8 是地面站点气候态，不包含 500 hPa 位势高度，因此本轮只保留现有 `geopotential_height500` 当前场，并在文件属性中写入 `geopotential_height500_anomaly_status=not_computed`。

## 6. 后续建议

1. 为比值类指标增加更有物理含义的分母阈值，例如低降水量时不解释对流降水比例、潜热通量过小时不解释 Bowen 比。
2. 将 `flash_flood_risk` 从简单阈值计数升级为可配置权重模型，区分降水、对流、水汽、地形和承灾体暴露。
3. 为 SST 增加红海、波斯湾分区时间序列指标，而不仅保存在属性 JSON 中。
4. 若要计算正式 SPI，需补充 1991-2020 或更长时段的逐月/逐日降水历史序列，并按 1/3/6/12 个月尺度拟合分布。
5. 若要计算 500 hPa 高度距平，需补充压力层位势高度的多年气候态或历史再分析基线。
6. 若要提升距平空间代表性，建议引入格点化气候态或对 DS8 站点 normals 做经地形约束的空间插值，而不是只用最近邻站点。
