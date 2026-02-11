#!/bin/bash

# qwenvl_run_background.sh
# 在后台运行 Qwen-VL 模型的 DeepSpeed 脚本

# 设置环境变量
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

# 日志文件名称
LOG_FILE="qwenvl.log"
echo "日志将保存到: $LOG_FILE"
echo "进程将在后台运行"

# 检查必要的文件是否存在
if [ ! -f "qwenvl_run.py" ]; then
    echo "错误: qwenvl_run.py 文件不存在！"
    exit 1
fi

if [ ! -f "args/qwen.yaml" ]; then
    echo "错误: args/qwen.yaml 文件不存在！"
    exit 1
fi

if [ ! -f "ds_config.json" ]; then
    echo "警告: ds_config.json 文件不存在，DeepSpeed 可能无法正常工作"
fi

# 运行命令并在后台执行
echo "开始在后台运行 Qwen-VL 训练..."
echo "开始时间: $(date)"

nohup deepspeed --master_port 29501 \
    qwenvl_run.py args/qwen.yaml \
    --deepspeed \
    --deepspeed_config ds_config.json 2>&1 | tee "$LOG_FILE" &

# 获取进程ID
PID=$!
echo "进程 PID: $PID"
echo "使用以下命令查看日志: tail -f $LOG_FILE"
echo "使用以下命令杀死进程: kill $PID"

# # 保存进程ID到文件
# echo $PID > qwenvl.pid
# echo "进程ID已保存到: qwenvl.pid"