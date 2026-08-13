#!/bin/bash
# 双流VTLA训练启动脚本
# 用法: ./start_dual_stream_training.sh <task> <gpu_id>

set -e

# 默认参数
TASK=${1:-"insert_HDMI"}
GPU_ID=${2:-"0"}

# 路径配置
PROJECT_ROOT="/home/rmc/workspace/VTLA_design"
UNIVTAC_ROOT="/home/rmc/workspace/UniVTAC"
PYTHON="/home/rmc/miniconda/envs/UniVTAC/bin/python"

# 数据集路径
DATASET_DIR="${UNIVTAC_ROOT}/data/official/${TASK}/clean"
MANIFEST="${PROJECT_ROOT}/data_manifests/${TASK}_official_first50.json"

# 检查文件是否存在
if [ ! -d "$DATASET_DIR" ]; then
    echo "错误: 数据集目录不存在: $DATASET_DIR"
    exit 1
fi

if [ ! -f "$MANIFEST" ]; then
    echo "错误: Manifest文件不存在: $MANIFEST"
    exit 1
fi

# 相机和触觉配置（根据任务自动选择）
case $TASK in
    "insert_tube"|"lift_bottle")
        CAMERA_NAMES="cam_high cam_left_wrist"
        ;;
    *)
        CAMERA_NAMES="cam_high"
        ;;
esac

TACTILE_NAMES="tac_left tac_right"

echo "============================================"
echo "双流VTLA训练"
echo "============================================"
echo "任务: $TASK"
echo "GPU: $GPU_ID"
echo "数据集: $DATASET_DIR"
echo "相机: $CAMERA_NAMES"
echo "触觉: $TACTILE_NAMES"
echo "============================================"
echo ""

# 启动训练
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON ${PROJECT_ROOT}/scripts/training/train_dual_stream.py \
    --task $TASK \
    --dataset-dir $DATASET_DIR \
    --output-dir ${PROJECT_ROOT}/runs/dual_stream \
    --device cuda \
    \
    --batch-size 8 \
    --num-episodes 50 \
    --train-ratio 0.9 \
    --chunk-size 50 \
    --seed 42 \
    \
    --camera-names $CAMERA_NAMES \
    --tactile-names $TACTILE_NAMES \
    \
    --state-dim 14 \
    --hidden-dim 512 \
    --nheads 8 \
    --dim-feedforward 2048 \
    --enc-layers 4 \
    --dec-layers 6 \
    --dropout 0.1 \
    \
    --shared-encoder True \
    --shared-decoder False \
    --enable-cross-stream False \
    --fusion-type gated \
    --use-contact-routing False \
    --use-cvae True \
    --latent-dim 32 \
    \
    --tactile-backbone resnet34 \
    --tactile-latent-dim 512 \
    --pretrained-backbones True \
    \
    --backbone resnet18 \
    --position-embedding sine \
    \
    --kl-weight 10.0 \
    --pad-weight 1.0 \
    --l1-reduction valid_mean \
    --aux-vision-weight 0.0 \
    --aux-tactile-weight 0.0 \
    \
    --num-epochs 2000 \
    --lr 1e-4 \
    --lr-backbone 1e-5 \
    --lr-tactile 1e-5 \
    --weight-decay 1e-4 \
    --grad-clip 0.0 \
    \
    --use-scheduler False \
    --save-every 100

echo ""
echo "============================================"
echo "训练完成！"
echo "============================================"
