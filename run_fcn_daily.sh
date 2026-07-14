#!/bin/bash
# FCN 每日自动预报 — 在 WSL2 中配置 cron 或在 Windows 任务计划中调用
# 用法: wsl bash /mnt/f/Saudi/run_fcn_daily.sh

# 激活 conda 环境并运行预报
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate earth2

# 设置 HuggingFace 镜像（国内加速）
export HF_ENDPOINT=https://hf-mirror.com

# 运行 FCN，输出到 Windows 目录
cd /mnt/f/Saudi
python run_fcn.py --days 7

echo "FCN 预报完成: $(date)"
