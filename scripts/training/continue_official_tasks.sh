#!/usr/bin/env bash
# Fill a currently idle benchmark GPU, then continue the remaining task queues.

set -euo pipefail

PROJECT_DIR="/home/rmc/workspace/VTLA_design"
LAUNCHER="${PROJECT_DIR}/scripts/training/start_official_tasks.sh"
GROUP_DIR="${1:-}"

if [[ -z "${GROUP_DIR}" || ! -d "${GROUP_DIR}" ]]; then
    echo "Usage: $0 /absolute/path/to/official_run_group" >&2
    exit 2
fi
if ! command -v tmux >/dev/null 2>&1; then
    echo "Error: tmux is not installed" >&2
    exit 1
fi

declare -A queues=(
    [3]="lift_bottle lift_can"
    [2]="pull_out_key put_bottle_in_shelf"
)

for task_name in lift_bottle lift_can pull_out_key put_bottle_in_shelf; do
    manifest="${PROJECT_DIR}/data_manifests/${task_name}_official_first50.json"
    if [[ ! -f "${manifest}" ]]; then
        echo "Error: validated manifest is missing: ${manifest}" >&2
        exit 1
    fi
done

timestamp=$(date +"%Y%m%d_%H%M%S")
session_prefix="vtla_official_more_${timestamp}"

# GPU 3 is intentionally launched first: it is free after insert_hole.
tmux new-session -d -s "${session_prefix}_gpu3" \
    "${LAUNCHER} --worker 3 ${GROUP_DIR} ${queues[3]}"

# GPU 2 may still be finishing insert_HDMI. Wait for a successful exit before
# assigning it more work, so runs never contend for the same physical device.
tmux new-session -d -s "${session_prefix}_gpu2_wait" \
    "bash -lc 'while [[ ! -f \"${GROUP_DIR}/insert_HDMI/exit_code\" ]]; do sleep 15; done; code=\$(<\"${GROUP_DIR}/insert_HDMI/exit_code\"); if [[ \"\${code}\" != 0 ]]; then echo \"insert_HDMI failed with exit code \${code}; continuation not started\"; exit \"\${code}\"; fi; exec \"${LAUNCHER}\" --worker 2 \"${GROUP_DIR}\" ${queues[2]}'"

cat > "${GROUP_DIR}/continuation_plan.txt" <<EOF
created_at=$(date --iso-8601=seconds)
git_commit=$(git -C "${PROJECT_DIR}" rev-parse HEAD 2>/dev/null || echo nogit)
gpu3=lift_bottle -> lift_can
gpu2=wait insert_HDMI exit_code=0 -> pull_out_key -> put_bottle_in_shelf
gpu1=existing grasp_classify -> insert_tube queue (unchanged)
session_prefix=${session_prefix}
EOF

echo "Started remaining official-task queues"
echo "  group: ${GROUP_DIR}"
echo "  GPU 3: lift_bottle -> lift_can"
echo "  GPU 2: wait for insert_HDMI -> pull_out_key -> put_bottle_in_shelf"
echo "  tmux prefix: ${session_prefix}"
