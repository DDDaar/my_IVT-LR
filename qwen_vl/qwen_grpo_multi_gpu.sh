#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash qwen_grpo_multi_gpu.sh [NUM_PROCESSES] [CONFIG_PATH] [MAIN_PROCESS_PORT] [LAUNCHER]
#
# Examples:
#   bash qwen_grpo_multi_gpu.sh
#   bash qwen_grpo_multi_gpu.sh 8 args/qwen_grpo.yaml 29531
#   bash qwen_grpo_multi_gpu.sh 8 args/qwen_grpo.yaml 29531 torchrun

find /home/ma-user/.cache/huggingface/datasets/ -type f -name "*of_00032.arrow" -mmin +300 -delete
export HF_DATASETS_CACHE="/home/ma-user/work/lbx/hf_data_cache"
find /home/ma-user/work/lbx/hf_data_cache -type f -name "*of_00032.arrow" -mmin +300 -delete
export WANDB_API_KEY=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
export HF_TOKEN=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
export HF_ENDPOINT=https://hf-mirror.com

NUM_PROCESSES="${1:-auto}"
CONFIG_PATH="${2:-args/qwen_grpo.yaml}"
MAIN_PROCESS_PORT="${3:-29531}"
LAUNCHER="${4:-auto}"   # auto | accelerate | torchrun
LOG_FILE="${LOG_FILE:-grpo_train_multi_gpu.log}"

# Optional environment hints:a
#   CUDA:   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#   ASCEND: export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#           export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "[ERROR] Config not found: ${CONFIG_PATH}"
  exit 1
fi

NPU_MODE=0
if [ -n "${ASCEND_VISIBLE_DEVICES:-}" ] || [ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
  NPU_MODE=1
fi
if python - <<'PY' >/dev/null 2>&1
import torch
ok = hasattr(torch, "npu")
if ok:
    try:
        ok = bool(torch.npu.is_available())
    except Exception:
        ok = False
raise SystemExit(0 if ok else 1)
PY
then
  NPU_MODE=1
fi

if [ "${NUM_PROCESSES}" = "auto" ]; then
  if [ -n "${ASCEND_VISIBLE_DEVICES:-}" ]; then
    NUM_PROCESSES="$(echo "${ASCEND_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
  elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_PROCESSES="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
  else
    NUM_PROCESSES=1
  fi
fi

if [ "${LAUNCHER}" = "auto" ]; then
  if [ "${NPU_MODE}" = "1" ]; then
    LAUNCHER="torchrun"
  else
    LAUNCHER="accelerate"
  fi
fi

echo "[INFO] Starting GRPO multi-GPU training"
echo "[INFO] npu_mode=${NPU_MODE}"
echo "[INFO] launcher=${LAUNCHER}"
echo "[INFO] num_processes=${NUM_PROCESSES}"
echo "[INFO] config=${CONFIG_PATH}"
echo "[INFO] main_process_port=${MAIN_PROCESS_PORT}"
echo "[INFO] log_file=${LOG_FILE}"

if [ "${LAUNCHER}" = "accelerate" ]; then
  if ! command -v accelerate >/dev/null 2>&1; then
    echo "[ERROR] 'accelerate' not found. Install with: pip install accelerate"
    exit 1
  fi
  accelerate launch \
    --multi_gpu \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    qwenvl_grpo_run.py "${CONFIG_PATH}" 2>&1 | tee "${LOG_FILE}"
elif [ "${LAUNCHER}" = "torchrun" ]; then
  if ! command -v torchrun >/dev/null 2>&1; then
    echo "[ERROR] 'torchrun' not found."
    exit 1
  fi
  # NPU distributed defaults.
  if [ "${NPU_MODE}" = "1" ]; then
    export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
    export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1800}"
    export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
  fi
  torchrun \
    --nproc_per_node "${NUM_PROCESSES}" \
    --master_port "${MAIN_PROCESS_PORT}" \
    qwenvl_grpo_run.py "${CONFIG_PATH}" 2>&1 | tee "${LOG_FILE}"
else
  echo "[ERROR] Unknown launcher: ${LAUNCHER}. Use auto/accelerate/torchrun."
  exit 1
fi
