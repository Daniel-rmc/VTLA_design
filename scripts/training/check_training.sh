#!/usr/bin/env bash
# Show active sessions, recent run artifacts, and GPU state.

set -u
PROJECT_DIR="/home/rmc/workspace/VTLA_design"

echo "=== VTLA tmux sessions ==="
tmux list-panes -a -F '#{session_name}|#{pane_dead}|#{pane_dead_status}|#{pane_current_command}' 2>/dev/null |
    awk -F'|' '
        $1 ~ /^vtla_/ {
            exit_code = ($3 == "") ? "unknown" : $3
            state = ($2 == "1") ? "finished(exit=" exit_code ")" : "running"
            printf "%s: %s, command=%s\n", $1, state, $4
            found = 1
        }
        END {if (!found) print "No VTLA tmux sessions."}
    ' || echo "No VTLA tmux sessions."

echo ""
echo "=== Most recent runs ==="
for stage in stage1 stage2 stage3; do
    latest=$(ls -td "${PROJECT_DIR}"/runs/${stage}/* 2>/dev/null | head -n 1 || true)
    [[ -z "${latest}" ]] && continue
    echo "${stage}: ${latest}"
    [[ -f "${latest}/config.json" ]] && echo "  config: ${latest}/config.json"
    [[ -f "${latest}/exit_code" ]] && echo "  exit code: $(<"${latest}/exit_code")"
    if [[ -f "${latest}/train.log" ]]; then
        echo "  recent log:"
        tail -n 8 "${latest}/train.log" | sed 's/^/    /'
    fi
done

echo ""
echo "=== GPU status ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free,memory.total --format=csv

echo ""
echo "Use 'tail -f <run_dir>/train.log' for live output."
