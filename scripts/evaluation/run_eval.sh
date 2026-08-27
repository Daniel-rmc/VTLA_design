#!/bin/bash
# 双流VTLA模型评测脚本

set -e

# 配置
TASK="insert_HDMI"
TASK_CONFIG="demo"
NUM_SEEDS=10  # 快速验证：10 seeds；完整评测：100 seeds
START_SEED=1000000

# 路径
UNIVTAC_ROOT="/home/rmc/workspace/UniVTAC"
VTLA_ROOT="/home/rmc/workspace/VTLA_design"
RUN_DIR=$(ls -td ${VTLA_ROOT}/runs/dual_stream/dual_stream_${TASK}_* | head -1)
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"

# 检查checkpoint是否存在
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

# 创建policy符号链接
POLICY_DIR="${UNIVTAC_ROOT}/policy/DualStream"
mkdir -p ${POLICY_DIR}

# 复制policy文件
cp ${VTLA_ROOT}/policy_wrapper/dual_stream_policy.py ${POLICY_DIR}/__init__.py
cp ${VTLA_ROOT}/policy_wrapper/deploy.yml ${POLICY_DIR}/deploy.yml

# 创建train_config目录结构
TRAIN_CONFIG_DIR="${POLICY_DIR}/dual_stream_ckpt/dual_stream-${TASK}/demo-50/train_config"
mkdir -p ${TRAIN_CONFIG_DIR}

# 复制checkpoint和stats
cp ${CHECKPOINT_DIR}/dual_stream_best.ckpt ${TRAIN_CONFIG_DIR}/policy_best.ckpt
cp ${RUN_DIR}/config.json ${TRAIN_CONFIG_DIR}/config.json

# 检查并复制dataset stats
if [ -f "${RUN_DIR}/dataset_stats.pkl" ]; then
    cp ${RUN_DIR}/dataset_stats.pkl ${TRAIN_CONFIG_DIR}/
else
    echo "⚠️  Warning: dataset_stats.pkl not found, will need to create it"
fi

echo "============================================"
echo "Dual-Stream VTLA Evaluation"
echo "============================================"
echo "Task: ${TASK}"
echo "Run: $(basename ${RUN_DIR})"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Seeds: ${START_SEED} to $((START_SEED + NUM_SEEDS - 1))"
echo "============================================"
echo ""

# 设置环境变量
export TRAIN_CONFIG="train_config"

# 进入UniVTAC目录
cd ${UNIVTAC_ROOT}

# 运行评测
python scripts/eval_policy.py \
    ${TASK} \
    ${TASK_CONFIG} \
    DualStream/deploy.yml \
    --start_seed ${START_SEED} \
    --max_seed $((START_SEED + NUM_SEEDS)) \
    --device cuda:0

echo ""
echo "============================================"
echo "Evaluation Complete!"
echo "============================================"
echo "Results saved to: eval_result/DualStream/${TASK}/"
