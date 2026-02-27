#!/bin/bash
export HCCL_CONNECT_TIMEOUT=1800  # 将超时增加到 1800 秒（30分钟）
export HCCL_EXEC_TIMEOUT=1800
# 配置环境变量
export ASCEND_VISIBLE_DEVICES=4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
# export NCCL_P2P_LEVEL=NVL
export PYTHONUNBUFFERED=1
export WANDB_API_KEY=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
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=""

# 进入目录
# cd qwen_vl || exit

# 启动 DeepSpeed
# 不使用 nohup，日志会直接打在屏幕上
deepspeed --master_port=29555 \
    qwenvl_run.py args/qwen.yaml \
    --deepspeed \
    --deepspeed_config ds_config.json \
    2>&1 | tee qwenvl_output_2.txt