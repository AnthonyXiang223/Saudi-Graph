"""Generate operators.json with all 91 indicators from the report."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

operators = []
def op(id, desc, cat, inputs, expr, dag, unit, src, co=None, avail_eff=None, avail_tot=365, limits=None):
    operators.append({
        "id": id, "description": desc, "category": cat,
        "inputs": inputs, "expression": expr, "dag": dag,
        "output_unit": unit, "source": src,
        "co_occurs_with": co or [],
        "availability": {"effective_days": avail_eff, "total_days": avail_tot},
        "limitations": limits
    })

# ====== 3.1 Monthly Surface & Energy (DS1) - 14 vars ======
op("monthly_precip_total", "月累计总降水，来自 DS1 MONTH_ACC", "monthly_surface_energy",
   [], "tp (MONTH_ACC)", {"op":"var","name":"tp"}, "mm", "DS1",
   ["monthly_precip_mmday","monthly_convective_precip","monthly_large_scale_precip"], 365, 365, "月背景，不反映逐日变化")

op("monthly_precip_mmday", "月平均降水强度，月累计除以当月天数", "monthly_surface_energy",
   ["monthly_precip_total"], "monthly_precip_total / days_in_month",
   {"op":"div","left":{"op":"var","name":"monthly_precip_total"},"right":"days_in_month"},
   "mm/day", "DS1", ["monthly_precip_total"], 365, 365, "days_in_month 为外部参数(28/29/30/31)")

op("monthly_convective_precip", "月累计对流降水，来自 DS1 MONTH_ACC acpcp", "monthly_surface_energy",
   [], "acpcp (MONTH_ACC)", {"op":"var","name":"acpcp"}, "mm", "DS1",
   ["monthly_large_scale_precip","monthly_convective_precip_ratio"])

op("monthly_large_scale_precip", "月累计非对流(层状/大尺度)降水，来自 DS1 MONTH_ACC ncpcp", "monthly_surface_energy",
   [], "ncpcp (MONTH_ACC)", {"op":"var","name":"ncpcp"}, "mm", "DS1",
   ["monthly_convective_precip"])

op("monthly_convective_precip_ratio", "月对流降水占比，越高说明降水更偏对流性", "monthly_surface_energy",
   ["monthly_convective_precip","monthly_precip_total"],
   "monthly_convective_precip / monthly_precip_total",
   {"op":"div","left":{"op":"var","name":"monthly_convective_precip"},"right":{"op":"var","name":"monthly_precip_total"}},
   "1", "DS1", ["monthly_convective_precip"], limits="分母 monthly_precip_total 近零时不稳定的比值")

op("monthly_bowen_ratio", "月 Bowen 比，感热相对潜热越高地表越干热", "monthly_surface_energy",
   [], "avg_ishf / avg_slhtf",
   {"op":"div","left":{"op":"var","name":"avg_ishf"},"right":{"op":"var","name":"avg_slhtf"}},
   "1", "DS1", ["monthly_heat_stress_index"], limits="分母潜热通量小时会放大异常，建议加阈值")

op("monthly_sw_net", "月净短波辐射，地表吸收的太阳短波能量", "monthly_surface_energy",
   [], "sdswrf - suswrf",
   {"op":"sub","left":{"op":"var","name":"sdswrf"},"right":{"op":"var","name":"suswrf"}},
   "W m-2", "DS1", ["monthly_lw_net","monthly_net_radiation"])

op("monthly_lw_net", "月净长波辐射，通常为负表示地表长波净损失", "monthly_surface_energy",
   [], "sdlwrf - sulwrf",
   {"op":"sub","left":{"op":"var","name":"sdlwrf"},"right":{"op":"var","name":"sulwrf"}},
   "W m-2", "DS1", ["monthly_sw_net","monthly_net_radiation"])

op("monthly_net_radiation", "月净辐射，地表能量收支背景", "monthly_surface_energy",
   ["monthly_sw_net","monthly_lw_net"], "monthly_sw_net + monthly_lw_net",
   {"op":"add","left":{"op":"var","name":"monthly_sw_net"},"right":{"op":"var","name":"monthly_lw_net"}},
   "W m-2", "DS1", ["monthly_sw_net","monthly_lw_net"])

op("monthly_heat_stress_index", "月地表热胁迫代理，结合感热通量和吸收短波", "monthly_surface_energy",
   [], "avg_ishf + (1 - avg_al/100) * sdswrf",
   {"op":"add","left":{"op":"var","name":"avg_ishf"},"right":{"op":"mul","left":{"op":"sub","left":1,"right":{"op":"div","left":{"op":"var","name":"avg_al"},"right":100}},"right":{"op":"var","name":"sdswrf"}}},
   "W m-2", "DS1", ["monthly_bowen_ratio"])

op("monthly_wind_stress_mag", "月地表风应力强度，反映地表动量交换", "monthly_surface_energy",
   [], "sqrt(avg_utaua^2 + avg_vtaua^2)",
   {"op":"sqrt","left":{"op":"add","left":{"op":"sqr","left":{"op":"var","name":"avg_utaua"}},"right":{"op":"sqr","left":{"op":"var","name":"avg_vtaua"}}}},
   "N m-2", "DS1", ["monthly_orographic_stress"])

op("monthly_orographic_stress", "月地形重力波应力，表征地形扰动和山地动力作用", "monthly_surface_energy",
   [], "sqrt(iegwss^2 + ingwss^2)",
   {"op":"sqrt","left":{"op":"add","left":{"op":"sqr","left":{"op":"var","name":"iegwss"}},"right":{"op":"sqr","left":{"op":"var","name":"ingwss"}}}},
   "N m-2", "DS1", ["monthly_wind_stress_mag"])

op("monthly_uvb_flux", "月 UV-B 下行通量，健康风险和晴空辐射背景", "monthly_surface_energy",
   [], "duvb", {"op":"var","name":"duvb"}, "W m-2", "DS1", ["monthly_uvb_clear_ratio"])

op("monthly_uvb_clear_ratio", "晴空/全天 UV-B 比，衡量云和气溶胶对 UV-B 的调制", "monthly_surface_energy",
   [], "cduvb / duvb",
   {"op":"div","left":{"op":"var","name":"cduvb"},"right":{"op":"var","name":"duvb"}},
   "1", "DS1", ["monthly_uvb_flux"])

# ====== 3.2 Cloud (DS2) - 4 vars ======
op("total_cloud_cover", "总云量", "cloud",
   [], "tcc", {"op":"var","name":"tcc"}, "%", "DS2",
   ["low_cloud_cover","middle_cloud_cover","high_cloud_cover","daily_precip_total"])

op("low_cloud_cover", "低云量", "cloud",
   [], "tcc_lowCloudLayer_0_0", {"op":"var","name":"tcc_lowCloudLayer_0_0"}, "%", "DS2",
   ["total_cloud_cover","rh2m"])

op("middle_cloud_cover", "中云量", "cloud",
   [], "tcc_middleCloudLayer_0_0", {"op":"var","name":"tcc_middleCloudLayer_0_0"}, "%", "DS2",
   ["total_cloud_cover","daily_large_scale_precip"])

op("high_cloud_cover", "高云量", "cloud",
   [], "tcc_highCloudLayer_0_0", {"op":"var","name":"tcc_highCloudLayer_0_0"}, "%", "DS2",
   ["total_cloud_cover","daily_convective_precip","cape"])

# ====== 3.3 Daily Precip/Radiation/Energy (DS2) - 10 vars ======
op("precip_mmday", "日平均降水率换算，来自日平均降水率 prate * 86400", "daily_precip_energy",
   [], "prate * 86400",
   {"op":"mul","left":{"op":"var","name":"prate"},"right":86400},
   "mm/day", "DS2", ["daily_precip_total"], limits="目前有效日很少，不宜作为主降水指标")

op("convective_precip_ratio", "日对流降水占比，表示降水对流性", "daily_precip_energy",
   [], "cpr / prate",
   {"op":"div","left":{"op":"var","name":"cpr"},"right":{"op":"var","name":"prate"}},
   "1", "DS2", ["daily_convective_precip","daily_large_scale_precip","cape"],
   limits="需和总降水共同解释，分母 prate 近零时不稳定")

op("daily_precip_total", "日累计总降水，当前最可靠的日降水主指标", "daily_precip_energy",
   [], "tp", {"op":"var","name":"tp"}, "mm", "DS2",
   ["ds10_max_1h","daily_convective_precip","pwat","ivt_convergence","flash_flood_risk"], 363)

op("daily_convective_precip", "日累计对流降水，表征强对流贡献", "daily_precip_energy",
   [], "acpcp", {"op":"var","name":"acpcp"}, "mm", "DS2",
   ["daily_large_scale_precip","daily_precip_total","cape"], 363)

op("daily_large_scale_precip", "日累计非对流降水，表征层状或大尺度降水贡献", "daily_precip_energy",
   [], "ncpcp", {"op":"var","name":"ncpcp"}, "mm", "DS2",
   ["daily_convective_precip","daily_precip_total"], 363)

op("bowen_ratio", "日 Bowen 比，感热相对潜热", "daily_precip_energy",
   [], "avg_ishf / avg_slhtf",
   {"op":"div","left":{"op":"var","name":"avg_ishf"},"right":{"op":"var","name":"avg_slhtf"}},
   "1", "DS2", ["heat_stress_index","vpd_kpa"], limits="极端值较多，建议加分母阈值后再用于建模")

op("sw_net", "日净短波辐射，白天太阳加热背景", "daily_precip_energy",
   [], "sdswrf - suswrf",
   {"op":"sub","left":{"op":"var","name":"sdswrf"},"right":{"op":"var","name":"suswrf"}},
   "W m-2", "DS2", ["lw_net","net_radiation"])

op("lw_net", "日净长波辐射，夜间和地表长波冷却背景", "daily_precip_energy",
   [], "sdlwrf - sulwrf",
   {"op":"sub","left":{"op":"var","name":"sdlwrf"},"right":{"op":"var","name":"sulwrf"}},
   "W m-2", "DS2", ["sw_net","net_radiation"])

op("net_radiation", "日净辐射，地表可用能量", "daily_precip_energy",
   ["sw_net","lw_net"], "sw_net + lw_net",
   {"op":"add","left":{"op":"var","name":"sw_net"},"right":{"op":"var","name":"lw_net"}},
   "W m-2", "DS2", ["sw_net","lw_net"])

op("heat_stress_index", "日地表热胁迫代理，越高代表吸收太阳能和感热通量越强", "daily_precip_energy",
   [], "avg_ishf + (1 - avg_al/100) * sdswrf",
   {"op":"add","left":{"op":"var","name":"avg_ishf"},"right":{"op":"mul","left":{"op":"sub","left":1,"right":{"op":"div","left":{"op":"var","name":"avg_al"},"right":100}},"right":{"op":"var","name":"sdswrf"}}},
   "W m-2", "DS2", ["bowen_ratio","t2m_c","heat_index_c"])

# ====== 3.4 Surface Thermo-Hygro & Wind (DS2+DS4) - 14 vars ======
op("t2m_c", "2米气温", "surface_thermo_wind",
   [], "t2m - 273.15",
   {"op":"sub","left":{"op":"var","name":"t2m"},"right":273.15},
   "degC", "DS2", ["tmax_c","tmin_c","t2m_anomaly_c","heat_index_c","apparent_temp_c","vpd_kpa","diurnal_temp_range_c"], 362)

op("d2m_c", "2米露点温度", "surface_thermo_wind",
   [], "d2m - 273.15",
   {"op":"sub","left":{"op":"var","name":"d2m"},"right":273.15},
   "degC", "DS2", ["dewpoint_depression_c","rh2m","sh2m"])

op("dewpoint_depression_c", "露点差，越大越干燥蒸发潜力越强", "surface_thermo_wind",
   ["t2m_c","d2m_c"], "t2m_c - d2m_c",
   {"op":"sub","left":{"op":"var","name":"t2m_c"},"right":{"op":"var","name":"d2m_c"}},
   "degC", "DS2", ["rh2m","vpd_kpa","wind10_speed"])

op("rh2m", "2米相对湿度", "surface_thermo_wind",
   [], "r2", {"op":"var","name":"r2"}, "%", "DS2",
   ["d2m_c","dewpoint_depression_c","vpd_kpa","heat_index_c","sst_celsius"], 364)

op("vpd_kpa", "饱和水汽压差，越高空气越干蒸散需求越强", "surface_thermo_wind",
   ["t2m_c","rh2m"], "es(T) * (1 - RH/100)",
   {"op":"vpd_formula","t_var":"t2m_c","rh_var":"rh2m"},
   "kPa", "DS2", ["dewpoint_depression_c","heat_index_c","bowen_ratio"], 362,
   limits="派生前已屏蔽填充值和非物理输入")

op("heat_index_c", "热指数(Rothfusz回归)，低温/低湿时退回气温", "surface_thermo_wind",
   ["t2m_c","rh2m"], "Rothfusz_regression(t2m_c, rh2m)",
   {"op":"heat_index_formula","t_var":"t2m_c","rh_var":"rh2m"},
   "degC", "DS2", ["t2m_c","apparent_temp_c","tmax_c","sst_celsius"], 362,
   limits="健康热风险指标，湿热条件下高于气温")

op("sh2m", "2米比湿", "surface_thermo_wind",
   [], "sh2", {"op":"var","name":"sh2"}, "kg kg-1", "DS2", ["d2m_c","qmax_2m","qmin_2m"])

op("apparent_temp_c", "体感温度", "surface_thermo_wind",
   [], "aptmp - 273.15",
   {"op":"sub","left":{"op":"var","name":"aptmp"},"right":273.15},
   "degC", "DS2", ["heat_index_c","t2m_c"], limits="派生前已屏蔽填充值")

op("wind10_speed", "10米风速", "surface_thermo_wind",
   [], "sqrt(u10^2 + v10^2)",
   {"op":"sqrt","left":{"op":"add","left":{"op":"sqr","left":{"op":"var","name":"u10"}},"right":{"op":"sqr","left":{"op":"var","name":"v10"}}}},
   "m s-1", "DS2", ["dewpoint_depression_c","wind850_speed","wind925_speed","flash_flood_risk"], 364)

op("tmax_c", "日最高2米气温，高温日筛查主指标", "surface_thermo_wind",
   [], "tmax - 273.15",
   {"op":"sub","left":{"op":"var","name":"tmax"},"right":273.15},
   "degC", "DS4", ["t2m_c","tmin_c","diurnal_temp_range_c","tmax_anomaly_c","heatwave_day_flag","heat_index_c"], 363)

op("tmin_c", "日最低2米气温，夜间热胁迫参考", "surface_thermo_wind",
   [], "tmin - 273.15",
   {"op":"sub","left":{"op":"var","name":"tmin"},"right":273.15},
   "degC", "DS4", ["t2m_c","tmax_c","diurnal_temp_range_c"], 363)

op("diurnal_temp_range_c", "日较差，反映昼夜热力差异和夜间冷却能力", "surface_thermo_wind",
   ["tmax_c","tmin_c"], "tmax_c - tmin_c",
   {"op":"sub","left":{"op":"var","name":"tmax_c"},"right":{"op":"var","name":"tmin_c"}},
   "degC", "DS4", ["tmax_c","tmin_c","t2m_c"], 362)

op("qmax_2m", "日最大2米比湿，湿热上限", "surface_thermo_wind",
   [], "qmax", {"op":"var","name":"qmax"}, "kg kg-1", "DS4", ["sh2m","qmin_2m"])

op("qmin_2m", "日最小2米比湿，干燥背景下限", "surface_thermo_wind",
   [], "qmin", {"op":"var","name":"qmin"}, "kg kg-1", "DS4", ["sh2m","qmax_2m"])

# ====== 3.5 DS8 Anomaly & Heatwave - 12 vars ======
op("daily_precip_climatology", "1991-2020逐日降水气候态(DS8站点最近邻映射到网格)", "ds8_anomaly",
   [], "DS8 PRE/MDAY 最近站逐日 normal", {"op":"ds8_lookup","element":"PRE","stat":"MDAY"},
   "mm/day", "DS8", ["daily_precip_anomaly","daily_precip_anomaly_ratio","daily_precip_climatology_station_distance_km"], 363,
   limits="DS8为站点气候态非格点场，沙特bbox内仅7个PRE站点，最近邻映射")

op("daily_precip_climatology_station_distance_km", "降水气候态最近站距离，越大局地不确定性越高", "ds8_anomaly",
   [], "网格点到所用 DS8 PRE 站的球面距离", {"op":"ds8_station_distance","element":"PRE"},
   "km", "DS8", ["daily_precip_climatology"], 363)

op("daily_precip_anomaly", "日降水距平，正值表示高于1991-2020同日站点气候态", "ds8_anomaly",
   ["daily_precip_total","daily_precip_climatology"],
   "daily_precip_total - daily_precip_climatology",
   {"op":"sub","left":{"op":"var","name":"daily_precip_total"},"right":{"op":"var","name":"daily_precip_climatology"}},
   "mm", "DS2+DS8", ["daily_precip_total","daily_precip_anomaly_ratio"], 363)

op("daily_precip_anomaly_ratio", "日降水距平率，干旱区分母很小会放大", "ds8_anomaly",
   ["daily_precip_total","daily_precip_climatology"],
   "(daily_precip_total - climatology) / climatology",
   {"op":"div","left":{"op":"sub","left":{"op":"var","name":"daily_precip_total"},"right":{"op":"var","name":"daily_precip_climatology"}},"right":{"op":"var","name":"daily_precip_climatology"}},
   "1", "DS2+DS8", ["daily_precip_anomaly"], 363,
   limits="不适合单独作为预警阈值，建议优先使用 daily_precip_anomaly 和 daily_precip_total")

op("t2m_climatology_c", "1991-2020日平均气温气候态(DS8站点最近邻映射)", "ds8_anomaly",
   [], "DS8 TAVG/MDAY 最近站逐日 normal", {"op":"ds8_lookup","element":"TAVG","stat":"MDAY"},
   "degC", "DS8", ["t2m_anomaly_c","t2m_climatology_station_distance_km"], 362,
   limits="TAVG在bbox内约45个站点")

op("t2m_climatology_station_distance_km", "TAVG气候态最近站距离", "ds8_anomaly",
   [], "网格点到所用 DS8 TAVG 站的球面距离", {"op":"ds8_station_distance","element":"TAVG"},
   "km", "DS8", ["t2m_climatology_c"])

op("t2m_anomaly_c", "2米气温距平，正值表示日平均气温偏暖", "ds8_anomaly",
   ["t2m_c","t2m_climatology_c"], "t2m_c - t2m_climatology_c",
   {"op":"sub","left":{"op":"var","name":"t2m_c"},"right":{"op":"var","name":"t2m_climatology_c"}},
   "degC", "DS2+DS8", ["t2m_c","tmax_anomaly_c","heatwave_day_flag"], 362)

op("tmax_climatology_c", "1991-2020日最高温气候态(DS8站点最近邻映射)", "ds8_anomaly",
   [], "DS8 TMAX/MDAY 最近站逐日 normal", {"op":"ds8_lookup","element":"TMAX","stat":"MDAY"},
   "degC", "DS8", ["tmax_anomaly_c","tmax_climatology_station_distance_km"], 363,
   limits="TMAX站点较少，沙特bbox内仅约11个有效站点，必须和距平一起查看")

op("tmax_climatology_station_distance_km", "TMAX气候态最近站距离", "ds8_anomaly",
   [], "网格点到所用 DS8 TMAX 站的球面距离", {"op":"ds8_station_distance","element":"TMAX"},
   "km", "DS8", ["tmax_climatology_c"])

op("tmax_anomaly_c", "日最高温距平，正值表示最高温偏暖", "ds8_anomaly",
   ["tmax_c","tmax_climatology_c"], "tmax_c - tmax_climatology_c",
   {"op":"sub","left":{"op":"var","name":"tmax_c"},"right":{"op":"var","name":"tmax_climatology_c"}},
   "degC", "DS4+DS8", ["tmax_c","t2m_anomaly_c","heatwave_day_flag"], 363)

op("heatwave_day_flag", "热浪日标志", "ds8_anomaly",
   ["tmax_c","tmax_climatology_c"],
   "tmax_c >= max(40 degC, tmax_climatology_c + 5 degC)",
   {"op":"threshold","left":{"op":"var","name":"tmax_c"},"condition":">=","right":{"op":"max","left":40,"right":{"op":"add","left":{"op":"var","name":"tmax_climatology_c"},"right":5}}},
   "flag", "DS4+DS8", ["tmax_c","tmax_anomaly_c","heatwave_duration_days"], 363,
   limits="同时要求绝对高温(>=40度)和相对气候态偏热(+5度)")

op("heatwave_duration_days", "热浪持续天数", "ds8_anomaly",
   ["heatwave_day_flag"], "对 heatwave_day_flag>=1 逐格逐日连续计数",
   {"op":"consecutive_count","flag_var":"heatwave_day_flag"},
   "days", "DS4+DS8", ["heatwave_day_flag","tmax_c"], 363,
   limits="缺少TMAX的日期不回写，最长局地72天(2025-08-16)")

# ====== 3.6 Convective Stability & Topography (DS2) - 6 vars ======
op("cape", "对流有效位能，已折算为二维水平网格", "convective_stability",
   [], "若源变量带额外层维度，先对非水平维取最大值",
   {"op":"max_over_dim","var":"cape_raw","dim":"pressureFromGroundLayer"},
   "J kg-1", "DS2", ["cin","surface_lifted_index","best_lifted_index","daily_convective_precip","flash_flood_risk","pwat"], 293,
   limits="越高代表潜在对流能量越强，有效覆盖293天")

op("cin", "对流抑制能量，越负抑制越强", "convective_stability",
   [], "若源变量带额外层维度，先对非水平维取最小值",
   {"op":"min_over_dim","var":"cin_raw","dim":"pressureFromGroundLayer"},
   "J kg-1", "DS2", ["cape","surface_lifted_index"], 293,
   limits="接近0时更容易触发对流")

op("surface_lifted_index", "地面抬升指数，越低越不稳定", "convective_stability",
   [], "若源变量带额外层维度，先对非水平维取最小值",
   {"op":"min_over_dim","var":"surface_lifted_index_raw","dim":"pressureFromGroundLayer"},
   "K", "DS2", ["cape","best_lifted_index"])

op("best_lifted_index", "最佳四层抬升指数，越低越有利于强对流", "convective_stability",
   [], "若源变量带额外层维度，先对非水平维取最小值",
   {"op":"min_over_dim","var":"best_lifted_index_raw","dim":"pressureFromGroundLayer"},
   "K", "DS2", ["cape","surface_lifted_index"])

op("surface_pressure", "地面气压", "convective_stability",
   [], "sp", {"op":"var","name":"sp"}, "Pa", "DS2", ["geopotential_height500","wind10_speed"])

op("orography", "地形高度(静态)，用于解释山地抬升和降水空间分布", "convective_stability",
   [], "orog", {"op":"var","name":"orog"}, "m", "DS2", ["monthly_orographic_stress","surface_pressure"],
   limits="静态地形")

# ====== 3.7 Multi-level Dynamics & Moisture Transport (DS2) - 20 vars ======
op("pwat", "可降水量(柱总水汽)，暴雨背景核心指标", "multilevel_dynamics",
   [], "pwat", {"op":"var","name":"pwat"}, "kg m-2", "DS2",
   ["ivt","ivt_convergence","daily_precip_total","cape"], 292)

op("ivt_u", "整层水汽输送纬向分量，正值偏东向输送", "multilevel_dynamics",
   [], "1/g * integral(q*u dp)", {"op":"ivt_component","direction":"u"},
   "kg m-1 s-1", "DS2", ["ivt","ivt_v"], 291)

op("ivt_v", "整层水汽输送经向分量，正值偏北向输送", "multilevel_dynamics",
   [], "1/g * integral(q*v dp)", {"op":"ivt_component","direction":"v"},
   "kg m-1 s-1", "DS2", ["ivt","ivt_u"], 291)

op("ivt", "整层水汽输送强度，强水汽通道识别", "multilevel_dynamics",
   ["ivt_u","ivt_v"], "sqrt(ivt_u^2 + ivt_v^2)",
   {"op":"sqrt","left":{"op":"add","left":{"op":"sqr","left":{"op":"var","name":"ivt_u"}},"right":{"op":"sqr","left":{"op":"var","name":"ivt_v"}}}},
   "kg m-1 s-1", "DS2", ["ivt_u","ivt_v","pwat","ivt_convergence"], 291)

op("ivt_divergence", "水汽输送散度，正值代表水汽发散", "multilevel_dynamics",
   ["ivt_u","ivt_v"], "d(ivt_u)/dx + d(ivt_v)/dy",
   {"op":"spatial_divergence","u_var":"ivt_u","v_var":"ivt_v"},
   "kg m-2 s-1", "DS2", ["ivt_convergence"], 291)

op("ivt_convergence", "水汽输送辐合，正值代表水汽汇聚有利降水", "multilevel_dynamics",
   ["ivt_divergence"], "-ivt_divergence",
   {"op":"neg","left":{"op":"var","name":"ivt_divergence"}},
   "kg m-2 s-1", "DS2", ["ivt","ivt_divergence","daily_precip_total","pwat","flash_flood_risk"], 291)

op("wind925_speed", "925 hPa风速，低层输送和近地层动力", "multilevel_dynamics",
   [], "sqrt(u925^2 + v925^2)",
   {"op":"sqrt","left":{"op":"add","left":{"op":"sqr","left":{"op":"var","name":"u925"}},"right":{"op":"sqr","left":{"op":"var","name":"v925"}}}},
   "m s-1", "DS2", ["moisture_transport925","wind850_speed","wind10_speed"])

op("moisture_transport925", "925 hPa水汽输送强度，低层水汽输送代理", "multilevel_dynamics",
   [], "q925 * wind925_speed",
   {"op":"mul","left":{"op":"var","name":"q925"},"right":{"op":"var","name":"wind925_speed"}},
   "m s-1", "DS2", ["wind925_speed"])

op("wind850_speed", "850 hPa风速，低空急流和水汽输送", "multilevel_dynamics",
   [], "sqrt(u850^2 + v850^2)",
   {"op":"sqrt","left":{"op":"add","left":{"op":"sqr","left":{"op":"var","name":"u850"}},"right":{"op":"sqr","left":{"op":"var","name":"v850"}}}},
   "m s-1", "DS2", ["moisture_transport850","wind925_speed","wind_shear_850_300","wind_shear_850_200"], 291)

op("moisture_transport850", "850 hPa水汽输送强度", "multilevel_dynamics",
   [], "q850 * wind850_speed",
   {"op":"mul","left":{"op":"var","name":"q850"},"right":{"op":"var","name":"wind850_speed"}},
   "m s-1", "DS2", ["wind850_speed"])

op("jet300_speed", "300 hPa风速，高空急流背景", "multilevel_dynamics",
   [], "sqrt(u300^2 + v300^2)",
   {"op":"sqrt","left":{"op":"add","left":{"op":"sqr","left":{"op":"var","name":"u300"}},"right":{"op":"sqr","left":{"op":"var","name":"v300"}}}},
   "m s-1", "DS2", ["jet200_speed","wind_shear_850_300"])

op("jet200_speed", "200 hPa风速，副热带急流和高空动力", "multilevel_dynamics",
   [], "sqrt(u200^2 + v200^2)",
   {"op":"sqrt","left":{"op":"add","left":{"op":"sqr","left":{"op":"var","name":"u200"}},"right":{"op":"sqr","left":{"op":"var","name":"v200"}}}},
   "m s-1", "DS2", ["jet300_speed","wind_shear_850_200"], 292)

op("wind_shear_850_300", "850-300 hPa垂直风切变，强对流组织化条件", "multilevel_dynamics",
   [], "sqrt((u300-u850)^2 + (v300-v850)^2)",
   {"op":"sqrt","left":{"op":"add","left":{"op":"sqr","left":{"op":"sub","left":{"op":"var","name":"u300"},"right":{"op":"var","name":"u850"}}},"right":{"op":"sqr","left":{"op":"sub","left":{"op":"var","name":"v300"},"right":{"op":"var","name":"v850"}}}}},
   "m s-1", "DS2", ["wind_shear_850_200","cape"])

op("wind_shear_850_200", "850-200 hPa垂直风切变，深层风切变", "multilevel_dynamics",
   [], "sqrt((u200-u850)^2 + (v200-v850)^2)",
   {"op":"sqrt","left":{"op":"add","left":{"op":"sqr","left":{"op":"sub","left":{"op":"var","name":"u200"},"right":{"op":"var","name":"u850"}}},"right":{"op":"sqr","left":{"op":"sub","left":{"op":"var","name":"v200"},"right":{"op":"var","name":"v850"}}}}},
   "m s-1", "DS2", ["wind_shear_850_300","jet200_speed"], 291)

op("relative_vorticity850", "850 hPa相对涡度，低层旋转和天气扰动", "multilevel_dynamics",
   [], "dv/dx - du/dy", {"op":"spatial_vorticity","u_var":"u850","v_var":"v850"},
   "s-1", "DS2", ["absolute_vorticity850","divergence850"])

op("divergence850", "850 hPa水平散度，正值发散负值辐合", "multilevel_dynamics",
   [], "du/dx + dv/dy", {"op":"spatial_divergence","u_var":"u850","v_var":"v850"},
   "s-1", "DS2", ["relative_vorticity850"])

op("absolute_vorticity850", "850 hPa绝对涡度，综合地转和相对旋转背景", "multilevel_dynamics",
   [], "absv", {"op":"var","name":"absv"}, "s-1", "DS2", ["relative_vorticity850"])

op("omega700", "700 hPa垂直速度，负值表示上升运动", "multilevel_dynamics",
   [], "omega at 700 hPa", {"op":"var","name":"omega700_raw"},
   "Pa s-1", "DS2", ["omega500","cape"])

op("omega500", "500 hPa垂直速度，中层上升/下沉运动", "multilevel_dynamics",
   [], "omega at 500 hPa", {"op":"var","name":"omega500_raw"},
   "Pa s-1", "DS2", ["omega700","geopotential_height500"], 292)

op("geopotential_height500", "500 hPa位势高度，中层环流和脊槽背景", "multilevel_dynamics",
   [], "gh500", {"op":"var","name":"gh500"}, "gpm", "DS2", ["omega500","surface_pressure"], 293,
   limits="距平未计算：缺少压力层位势高度气候态(DS8为地面站点气候态)")

# ====== 3.8 SST & DS10 Satellite Precip - 10 vars ======
op("sst_celsius", "海表温度，红海和波斯湾热湿背景", "sst_ds10",
   [], "analysed_sst, 源值>200时转为K-273.15",
   {"op":"where","cond":{"op":"threshold","left":{"op":"var","name":"analysed_sst"},"condition":">","right":200},"true":{"op":"sub","left":{"op":"var","name":"analysed_sst"},"right":273.15},"false":{"op":"var","name":"analysed_sst"}},
   "degC", "SST", ["rh2m","heat_index_c"], 365,
   limits="红海和波斯湾高海温为沿海湿热提供背景条件")

op("ds10_daily_total", "DS10卫星日降水，最近邻对齐到指标网格", "sst_ds10",
   [], "daily_total, 最近邻重采样", {"op":"nearest_resample","var":"ds10_daily_total_raw"},
   "mm", "DS10", ["ds10_max_1h","ds10_max_3h","ds10_ds2_precip_diff","ds10_ds2_precip_ratio","daily_precip_total"], 273,
   limits="高频卫星降水日聚合，Q4(10-12月)缺测")

op("ds10_max_30min", "DS10 30分钟最大降水", "sst_ds10",
   [], "max_30min, 最近邻重采样", {"op":"nearest_resample","var":"ds10_max_30min_raw"},
   "mm", "DS10", ["ds10_max_1h","flash_flood_risk"], 273)

op("ds10_max_1h", "DS10 1小时最大降水，山洪短历时降水核心候选", "sst_ds10",
   [], "max_1h, 最近邻重采样", {"op":"nearest_resample","var":"ds10_max_1h_raw"},
   "mm", "DS10", ["ds10_max_30min","ds10_max_3h","flash_flood_risk","daily_precip_total"], 273,
   limits="已纳入 flash_flood_risk 评分")

op("ds10_max_3h", "DS10 3小时最大降水", "sst_ds10",
   [], "max_3h, 最近邻重采样", {"op":"nearest_resample","var":"ds10_max_3h_raw"},
   "mm", "DS10", ["ds10_max_1h","ds10_max_6h"], 273)

op("ds10_max_6h", "DS10 6小时最大降水", "sst_ds10",
   [], "max_6h, 最近邻重采样", {"op":"nearest_resample","var":"ds10_max_6h_raw"},
   "mm", "DS10", ["ds10_max_3h"], 273)

op("ds10_rainy_steps", "DS10有雨时次数，表示日内降水持续性", "sst_ds10",
   [], "rainy_steps, 最近邻重采样", {"op":"nearest_resample","var":"ds10_rainy_steps_raw"},
   "steps", "DS10", ["ds10_daily_total"], 273)

op("ds10_ds2_precip_diff", "DS10与DS2日降水差值，正值DS10偏高", "sst_ds10",
   ["ds10_daily_total","daily_precip_total"], "ds10_daily_total - daily_precip_total",
   {"op":"sub","left":{"op":"var","name":"ds10_daily_total"},"right":{"op":"var","name":"daily_precip_total"}},
   "mm", "DS10+DS2", ["ds10_ds2_precip_ratio","ds10_ds2_heavy_rain_overlap"], 273)

op("ds10_ds2_precip_ratio", "DS10/DS2日降水比值，识别两套降水估计偏差", "sst_ds10",
   ["ds10_daily_total","daily_precip_total"], "ds10_daily_total / daily_precip_total",
   {"op":"div","left":{"op":"var","name":"ds10_daily_total"},"right":{"op":"var","name":"daily_precip_total"}},
   "1", "DS10+DS2", ["ds10_ds2_precip_diff"], 273)

op("ds10_ds2_heavy_rain_overlap", "DS10与DS2强降水重叠标志", "sst_ds10",
   ["ds10_daily_total","daily_precip_total"], "两者均>=10mm时为1",
   {"op":"and_","left":{"op":"threshold","left":{"op":"var","name":"ds10_daily_total"},"condition":">=","right":10},"right":{"op":"threshold","left":{"op":"var","name":"daily_precip_total"},"condition":">=","right":10}},
   "flag", "DS10+DS2", ["ds10_ds2_precip_diff","daily_precip_total"], 273,
   limits="用于筛选两套数据共同支持的强降水网格")

# ====== 3.9 Composite - 1 var ======
op("flash_flood_risk", "山洪初筛分数(0-5)，可用条件阈值标志求和", "composite",
   ["daily_precip_total","convective_precip_ratio","cape","wind10_speed","ds10_max_1h"],
   "SUM of 5 threshold flags",
   {"op":"sum_flags","flags":[
     {"indicator":"daily_precip_total","op":">=","value":10},
     {"indicator":"convective_precip_ratio","op":">=","value":0.5},
     {"indicator":"cape","op":">=","value":1000},
     {"indicator":"wind10_speed","op":">=","value":10},
     {"indicator":"ds10_max_1h","op":">=","value":10}
   ]},
   "score", "DS2+DS10", ["daily_precip_total","cape","ds10_max_1h","ivt_convergence","pwat"], 365,
   limits="CAPE已折算二维后参与评分。DS10缺测时短时强降水项视为缺测。分数越高越多触发因子同时满足")

# ====== Write output ======
output = {
    "description": "沙特极端事件指标算子定义 — 物理公式编码，与业务阈值分离。共91个指标，覆盖9个类别，6个数据源。",
    "total_indicators": len(operators),
    "operators": operators
}

with open("schema/operators.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Written {len(operators)} operators to schema/operators.json")

# Summary by category
from collections import Counter
cats = Counter(o["category"] for o in operators)
for cat, count in sorted(cats.items()):
    srcs = set(o["source"] for o in operators if o["category"] == cat)
    has_inputs = sum(1 for o in operators if o["category"] == cat and o["inputs"])
    print(f"  {cat}: {count} vars (source: {', '.join(sorted(srcs))}, with_inputs: {has_inputs})")

# Validate all DAG ops
planned_ops = {"var","add","sub","mul","div","neg","sqrt","sqr","pow","max","min","abs",
               "threshold","where","and_","or_","vpd_formula","heat_index_formula",
               "max_over_dim","min_over_dim","ds8_lookup","ds8_station_distance",
               "consecutive_count","ivt_component","spatial_divergence","spatial_vorticity",
               "nearest_resample","sum_flags"}
used_ops = set()
def collect_ops(dag):
    if isinstance(dag, dict) and "op" in dag:
        used_ops.add(dag["op"])
        for v in dag.values():
            collect_ops(v)
    elif isinstance(dag, dict):
        for v in dag.values():
            collect_ops(v)
    elif isinstance(dag, list):
        for item in dag:
            collect_ops(item)
for o in operators:
    collect_ops(o["dag"])
unknown = used_ops - planned_ops
if unknown:
    print(f"\nWARNING: Unknown DAG ops: {unknown}")
else:
    print(f"\nAll {len(used_ops)} DAG ops in vocabulary: {sorted(used_ops)}")

# Print leaf vs non-leaf counts
leaf = [o for o in operators if not o["inputs"]]
derived = [o for o in operators if o["inputs"]]
print(f"\nLeaf indicators (no inputs): {len(leaf)}")
print(f"Derived indicators (with inputs): {len(derived)}")
