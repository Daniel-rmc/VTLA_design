#!/bin/bash
# 快速查看训练状态和日志

PROJECT_DIR="/home/rmc/workspace/VTLA_design"

echo "========================================"
echo "VTLA Training Status"
echo "========================================"
echo ""

# 检查tmux会话
echo "Active tmux sessions:"
tmux ls 2>/dev/null | grep vtla || echo "  No VTLA training sessions found"
echo ""

# 显示最近的日志文件
echo "Recent log files:"
for stage in stage1 stage2 stage3; do
    LOG_DIR="${PROJECT_DIR}/logs/${stage}"
    if [ -d "${LOG_DIR}" ]; then
        LATEST_LOG=$(ls -t ${LOG_DIR}/*.log 2>/dev/null | head -1)
        if [ -n "${LATEST_LOG}" ]; then
            echo ""
            echo "=== ${stage} ==="
            echo "Log: ${LATEST_LOG}"
            echo "Size: $(du -h ${LATEST_LOG} | cut -f1)"
            echo "Last modified: $(stat -c %y ${LATEST_LOG} | cut -d'.' -f1)"
            echo ""
            echo "Last 10 lines:"
            tail -10 ${LATEST_LOG}
        fi
    fi
done

echo ""
echo "========================================"
echo "GPU Status:"
echo "========================================"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo "========================================"
echo "Commands:"
echo "  View live log:     tail -f <log_file>"
echo "  Attach to tmux:    tmux attach -t vtla_<stage>"
echo "  List checkpoints:  ls -lh ${PROJECT_DIR}/checkpoints/<stage>/"
echo "========================================"
