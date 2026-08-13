#!/bin/bash
# 快速查看训练状态和loss曲线

set -e

PROJECT_ROOT="/home/rmc/workspace/VTLA_design"

# 找到最新的run
RUN_DIR=$(ls -td ${PROJECT_ROOT}/runs/dual_stream/dual_stream_* 2>/dev/null | head -1)

if [ -z "$RUN_DIR" ]; then
    echo "❌ No training run found"
    exit 1
fi

echo "============================================"
echo "Training Status Check"
echo "============================================"
echo "Run: $(basename $RUN_DIR)"
echo ""

# 检查CSV日志是否存在
CSV_LOG="$RUN_DIR/training_log.csv"

if [ ! -f "$CSV_LOG" ]; then
    echo "⚠️  No training log found yet"
    echo "Training may still be initializing..."
    exit 0
fi

# 统计已训练的epoch数
TOTAL_EPOCHS=$(tail -n +2 "$CSV_LOG" | wc -l)
echo "📊 Epochs completed: $TOTAL_EPOCHS"

if [ $TOTAL_EPOCHS -eq 0 ]; then
    echo "⚠️  No epochs completed yet"
    exit 0
fi

# 显示最近5个epoch的loss
echo ""
echo "Recent Loss (last 5 epochs):"
echo "--------------------------------------------"
echo "Epoch | Train Loss | Train L1 | Train KL"
echo "--------------------------------------------"
tail -n 5 "$CSV_LOG" | awk -F',' '{printf "%5s | %10.4f | %8.4f | %8.4f\n", $1, $2, $3, $4}'
echo ""

# 显示最佳loss
echo "📈 Best Metrics:"
BEST_TRAIN_LOSS=$(tail -n +2 "$CSV_LOG" | awk -F',' '{print $2}' | sort -g | head -1)
BEST_L1=$(tail -n +2 "$CSV_LOG" | awk -F',' '{print $3}' | sort -g | head -1)
echo "  Best Train Loss: $BEST_TRAIN_LOSS"
echo "  Best L1 Loss:    $BEST_L1"
echo ""

# 检查TensorBoard
TB_DIR="$RUN_DIR/tensorboard"
if [ -d "$TB_DIR" ]; then
    echo "📊 TensorBoard available:"
    echo "  tensorboard --logdir=$TB_DIR"
    echo ""
fi

# 生成可视化图（如果有matplotlib）
if command -v python &> /dev/null; then
    echo "📊 Generating loss curves..."
    python ${PROJECT_ROOT}/scripts/analysis/plot_training_curves.py "$CSV_LOG" --output-dir "$RUN_DIR" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  ✓ Plots saved to: $RUN_DIR"
        echo "    - loss_curve.png"
        echo "    - l1_loss_curve.png"
        echo "    - kl_loss_curve.png"
        echo "    - training_summary.png"
    else
        echo "  ⚠️  Failed to generate plots (matplotlib may not be available)"
    fi
fi

echo ""
echo "============================================"
echo "Run directory: $RUN_DIR"
echo "============================================"
