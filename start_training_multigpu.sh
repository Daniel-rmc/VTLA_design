#!/bin/bash
# VTLA多卡训练启动脚本 (支持4卡并行)

set -e

# 配置
PROJECT_DIR="/home/rmc/workspace/VTLA_design"
CONDA_ENV="UniVTAC"
STAGE="${1:-stage2}"
NUM_GPUS="${2:-4}"  # 默认使用4张卡
SESSION_NAME="vtla_${STAGE}_${NUM_GPUS}gpu"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}VTLA Multi-GPU Training Launcher${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查GPU数量
AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ ${NUM_GPUS} -gt ${AVAILABLE_GPUS} ]; then
    echo -e "${RED}Error: Requested ${NUM_GPUS} GPUs but only ${AVAILABLE_GPUS} available${NC}"
    exit 1
fi

echo -e "${BLUE}GPU Configuration:${NC}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# 检查tmux
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}Error: tmux is not installed${NC}"
    exit 1
fi

# 检查现有会话
if tmux has-session -t ${SESSION_NAME} 2>/dev/null; then
    echo -e "${YELLOW}Warning: tmux session '${SESSION_NAME}' already exists${NC}"
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

# 创建目录
LOG_DIR="${PROJECT_DIR}/logs/${STAGE}"
CKPT_DIR="${PROJECT_DIR}/checkpoints/${STAGE}"
mkdir -p ${LOG_DIR} ${CKPT_DIR}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${NUM_GPUS}gpu_${TIMESTAMP}.log"

echo -e "${GREEN}Configuration:${NC}"
echo "  Stage: ${STAGE}"
echo "  GPUs: ${NUM_GPUS}"
echo "  Project Dir: ${PROJECT_DIR}"
echo "  Conda Env: ${CONDA_ENV}"
echo "  Log File: ${LOG_FILE}"
echo "  Checkpoint Dir: ${CKPT_DIR}"
echo ""

# 根据stage和GPU数量设置参数
if [ "${STAGE}" == "stage1" ]; then
    DATASET_DIR="/home/rmc/workspace/UniVTAC/data/grasp_classify/demo"
    BATCH_SIZE=8  # 每卡8，总batch=32
    NUM_EPOCHS=100
    EXTRA_ARGS="--tactile_supervise rgb marker --state_dim 9"
elif [ "${STAGE}" == "stage2" ]; then
    DATASET_DIR="/home/rmc/workspace/UniVTAC/data/grasp_classify/demo"
    BATCH_SIZE=4  # 每卡4，总batch=16
    NUM_EPOCHS=500
    EXTRA_ARGS="--camera_names cam_high --chunk_size 50 --state_dim 9"

    # 查找stage1 checkpoint
    STAGE1_CKPT=$(ls ${PROJECT_DIR}/checkpoints/stage1/stage1_epoch_*.ckpt 2>/dev/null | tail -1)
    if [ -n "${STAGE1_CKPT}" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --stage1_ckpt ${STAGE1_CKPT}"
        echo -e "${GREEN}  Using Stage1 checkpoint: ${STAGE1_CKPT}${NC}"
    fi
elif [ "${STAGE}" == "stage3" ]; then
    DATASET_DIR="/home/rmc/workspace/UniVTAC/data/grasp_classify/demo"
    BATCH_SIZE=4  # 每卡4，总batch=16
    NUM_EPOCHS=200
    EXTRA_ARGS="--camera_names cam_high --chunk_size 50"

    # 查找stage2 checkpoint
    STAGE2_CKPT=$(ls ${PROJECT_DIR}/checkpoints/stage2/stage2_epoch_*.ckpt 2>/dev/null | tail -1)
    if [ -n "${STAGE2_CKPT}" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --stage2_ckpt ${STAGE2_CKPT}"
        echo -e "${GREEN}  Using Stage2 checkpoint: ${STAGE2_CKPT}${NC}"
    else
        echo -e "${RED}  Warning: No Stage2 checkpoint found!${NC}"
    fi
else
    echo -e "${RED}Error: Invalid stage '${STAGE}'${NC}"
    exit 1
fi

# 计算有效batch size
EFFECTIVE_BATCH=$((BATCH_SIZE * NUM_GPUS))
echo -e "${BLUE}Training Configuration:${NC}"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Effective batch size: ${EFFECTIVE_BATCH}"
echo "  Number of epochs: ${NUM_EPOCHS}"
echo "  Workers per GPU: 4"
echo ""

# 构建训练命令
TRAIN_CMD="cd ${PROJECT_DIR} && \
    conda activate ${CONDA_ENV} && \
    export PYTHONPATH=${PROJECT_DIR}:${PROJECT_DIR}/../UniVTAC:${PROJECT_DIR}/../UniVTAC/policy/ACT:\$PYTHONPATH && \
    export CUDA_VISIBLE_DEVICES=0,1,2,3 && \
    python train_vtla_multigpu.py \
        --stage ${STAGE} \
        --num_gpus ${NUM_GPUS} \
        --dataset_dir ${DATASET_DIR} \
        --tactile_names tac_left tac_right \
        --state_dim 14 \
        --batch_size ${BATCH_SIZE} \
        --num_epochs ${NUM_EPOCHS} \
        --ckpt_dir ${CKPT_DIR} \
        --num_workers 4 \
        --save_freq 50 \
        ${EXTRA_ARGS} \
        2>&1 | tee ${LOG_FILE}"

echo -e "${YELLOW}Starting tmux session: ${SESSION_NAME}${NC}"
echo ""

# 创建tmux会话
tmux new-session -d -s ${SESSION_NAME}
tmux send-keys -t ${SESSION_NAME} "${TRAIN_CMD}" C-m

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Multi-GPU Training Started!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}GPU Usage:${NC}"
echo "  Using ${NUM_GPUS} x NVIDIA L40S (46GB each)"
echo "  Total GPU memory: $((NUM_GPUS * 46))GB"
echo ""
echo -e "${BLUE}Commands:${NC}"
echo -e "  ${YELLOW}Attach to session:${NC}     tmux attach -t ${SESSION_NAME}"
echo -e "  ${YELLOW}Detach (inside):${NC}      Ctrl+B, then D"
echo -e "  ${YELLOW}View log:${NC}             tail -f ${LOG_FILE}"
echo -e "  ${YELLOW}Monitor GPU:${NC}          watch -n 1 nvidia-smi"
echo -e "  ${YELLOW}Check training:${NC}       ./check_training.sh"
echo -e "  ${YELLOW}Kill session:${NC}         tmux kill-session -t ${SESSION_NAME}"
echo ""
echo -e "${GREEN}Log file: ${LOG_FILE}${NC}"
echo ""

# 等待2秒查看GPU使用情况
sleep 2
echo -e "${BLUE}Initial GPU Status:${NC}"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
echo ""

# 询问是否attach
read -p "Attach to training session now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    tmux attach -t ${SESSION_NAME}
fi
