#!/usr/bin/env bash
# Unattended FlowWM training run + automatic morning report.
#
# Detached from any terminal (launch with setsid/nohup) so it survives logout.
# Whatever happens -- clean finish, crash, OOM -- make_report.py runs afterwards and
# writes OVERNIGHT_REPORT.md, so there is always something to read in the morning.
set -uo pipefail   # deliberately NOT -e: a training failure must still produce a report

REPO=/home/davi/Documents/Flow-World-Models
CFG=configs/dinov3/flow_matching_rae_waymo/local3060/384/local_overnight.yaml
EXP=$REPO/experiments/dinov3/flow_matching_rae_waymo/local3060/384/local_overnight
WRAP_LOG=$REPO/overnight_wrapper.log
REPORT=$REPO/OVERNIGHT_REPORT.md
GPU_LOG=$REPO/overnight_gpu.csv

cd "$REPO" || exit 1
export WANDB_MODE=disabled

{
  echo "=== FlowWM overnight run ==="
  echo "started : $(date '+%Y-%m-%d %H:%M:%S')"
  echo "config  : $CFG"
  echo "host    : $(hostname)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  echo "=== resolved config ==="
  cat "$CFG"
  echo "==========================="
} > "$WRAP_LOG" 2>&1

# Sample GPU temperature/memory every 2 min so the report can show thermals overnight.
( while true; do
    echo "$(date '+%H:%M:%S'),$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits)"
    sleep 120
  done ) > "$GPU_LOG" 2>&1 &
GPU_PID=$!

uv run torchrun --nproc_per_node=1 -m main --cfg_path "$CFG" >> "$WRAP_LOG" 2>&1
EXIT_CODE=$?

kill $GPU_PID 2>/dev/null

{
  echo "==========================="
  echo "finished: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "exit code: $EXIT_CODE"
} >> "$WRAP_LOG" 2>&1

uv run python make_report.py "$EXP" "$WRAP_LOG" "$EXIT_CODE" "$REPORT" >> "$WRAP_LOG" 2>&1

# Append the thermal summary; harmless if the sampler produced nothing.
if [ -s "$GPU_LOG" ]; then
  {
    echo ""
    echo "## Thermals overnight"
    echo ""
    echo '```'
    echo "samples: $(wc -l < "$GPU_LOG")  (time,temp_C,util_%,mem_MiB)"
    echo "max temp: $(cut -d, -f2 "$GPU_LOG" | sort -n | tail -1) C"
    echo "max mem : $(cut -d, -f4 "$GPU_LOG" | sort -n | tail -1) MiB"
    echo "first: $(head -1 "$GPU_LOG")"
    echo "last : $(tail -1 "$GPU_LOG")"
    echo '```'
  } >> "$REPORT"
fi

echo "DONE exit=$EXIT_CODE report=$REPORT" >> "$WRAP_LOG"
