#!/bin/bash
# VTLA训练启动脚本 - 使用tmux管理训练进程

set -e

# 配置
PROJECT_DIR="/home/rmc/workspace/VTLA_design"
CONDA_ENV="UniVTAC"
STAGE="${1:-stage2}"  # 默认stage2
SESSION_NAME="vtla_${STAGE}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}VTLA Training Launcher${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查tmux是否安装
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}Error: tmux is not installed${NC}"
    echo "Please install: sudo apt-get install tmux"
    exit 1
fi

# 检查tmux会话是否已存在
if tmux has-session -t ${SESSION_NAME} 2>/dev/null; then
    echo -e "${YELLOW}Warning: tmux session '${SESSION_NAME}' already exists${NC}"
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t ${SESSION_NAME}"
    echo "  2. Kill and restart: tmux kill-session -t ${SESSION_NAME}"
    read -p "Kill existing session and restart? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t ${SESSION_NAME}
        echo -e "${GREEN}Killed existing session${NC}"
    else
        echo -e "${YELLOW}Attaching to existing session...${NC}"
        tmux attach -t ${SESSION_NAME}
        exit 0
    fi
fi

# 创建日志目录
LOG_DIR="${PROJECT_DIR}/logs/${STAGE}"
CKPT_DIR="${PROJECT_DIR}/checkpoints/${STAGE}"
mkdir -p ${LOG_DIR} ${CKPT_DIR}

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

echo -e "${GREEN}Configuration:${NC}"
echo "  Stage: ${STAGE}"
echo "  Project Dir: ${PROJECT_DIR}"
echo "  Conda Env: ${CONDA_ENV}"
echo "  Log File: ${LOG_FILE}"
echo "  Checkpoint Dir: ${CKPT_DIR}"
echo ""

# 根据stage设置训练参数
if [ "${STAGE}" == "stage1" ]; then
    DATASET_DIR="/home/rmc/workspace/UniVTAC/data/grasp_classify/demo"
    BATCH_SIZE=16
    NUM_EPOCHS=100
    EXTRA_ARGS="--tactile_supervise rgb marker"
elif [ "${STAGE}" == "stage2" ]; then
    DATASET_DIR="/home/rmc/workspace/UniVTAC/data/grasp_classify/demo"
    BATCH_SIZE=4
    NUM_EPOCHS=500
    EXTRA_ARGS="--camera_names cam_high --chunk_size 50"
    # 如果有stage1 checkpoint，添加
    STAGE1_CKPT=$(ls ${PROJECT_DIR}/checkpoints/stage1/stage1_epoch_*.ckpt 2>/dev/null | tail -1)
    if [ -n "${STAGE1_CKPT}" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --stage1_ckpt ${STAGE1_CKPT}"
        echo -e "${GREEN}  Using Stage1 checkpoint: ${STAGE1_CKPT}${NC}"
    fi
elif [ "${STAGE}" == "stage3" ]; then
    DATASET_DIR="/home/rmc/workspace/UniVTAC/data/grasp_classify/demo"
    BATCH_SIZE=8
    NUM_EPOCHS=200
    EXTRA_ARGS="--camera_names cam_high --chunk_size 50"
    # 如果有stage2 checkpoint，添加
    STAGE2_CKPT=$(ls ${PROJECT_DIR}/checkpoints/stage2/stage2_epoch_*.ckpt 2>/dev/null | tail -1)
    if [ -n "${STAGE2_CKPT}" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --stage2_ckpt ${STAGE2_CKPT}"
        echo -e "${GREEN}  Using Stage2 checkpoint: ${STAGE2_CKPT}${NC}"
    else
        echo -e "${RED}  Warning: No Stage2 checkpoint found!${NC}"
    fi
else
    echo -e "${RED}Error: Invalid stage '${STAGE}'${NC}"
    echo "Valid stages: stage1, stage2, stage3"
    exit 1
fi

# 构建训练命令
TRAIN_CMD="cd ${PROJECT_DIR} && \
    conda activate ${CONDA_ENV} && \
    export PYTHONPATH=${PROJECT_DIR}:${PROJECT_DIR}/../UniVTAC:${PROJECT_DIR}/../UniVTAC/policy/ACT:\$PYTHONPATH && \
    python train_vtla.py \
        --stage ${STAGE} \
        --dataset_dir ${DATASET_DIR} \
        --tactile_names tac_left tac_right \
        --state_dim 14 \
        --batch_size ${BATCH_SIZE} \
        --num_epochs ${NUM_EPOCHS} \
        --ckpt_dir ${CKPT_DIR} \
        --device cuda:0 \
        --num_workers 2 \
        --save_freq 50 \
        ${EXTRA_ARGS} \
        2>&1 | tee ${LOG_FILE}"

echo -e "${YELLOW}Starting tmux session: ${SESSION_NAME}${NC}"
echo ""

# 创建tmux会话并运行训练
tmux new-session -d -s ${SESSION_NAME}
tmux send-keys -t ${SESSION_NAME} "${TRAIN_CMD}" C-m

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Training started successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Commands:"
echo -e "  ${YELLOW}Attach to session:${NC}  tmux attach -t ${SESSION_NAME}"
echo -e "  ${YELLOW}Detach (inside):${NC}   Ctrl+B, then D"
echo -e "  ${YELLOW}View log:${NC}          tail -f ${LOG_FILE}"
echo -e "  ${YELLOW}Kill session:${NC}      tmux kill-session -t ${SESSION_NAME}"
echo ""
echo -e "${GREEN}Log file: ${LOG_FILE}${NC}"
echo ""

# 询问是否立即attach
read -p "Attach to training session now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    tmux attach -t ${SESSION_NAME}
fi
