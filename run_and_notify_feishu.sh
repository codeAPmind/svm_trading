#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [ -f "$HOME/.zshrc" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.zshrc"
fi

OPENCLAW_BIN="${OPENCLAW_BIN:-openclaw}"
TARGET_OPEN_ID="${FEISHU_TARGET_OPEN_ID:-}"
RUN_ARGS="${SVM_RUN_ARGS:---code OPEN --name Opendoor --start 2021-01-01 --end 2026-04-15 --mode full --no-optimize}"
PYTHON_BIN="${SVM_PYTHON_BIN:-python3}"
NOW="$(date '+%Y-%m-%d %H:%M:%S')"

if [ -z "$TARGET_OPEN_ID" ]; then
  echo "[ERROR] FEISHU_TARGET_OPEN_ID 未设置，无法通过 openclaw 发送飞书消息。"
  exit 2
fi

set +e
# shellcheck disable=SC2086
"$PYTHON_BIN" main.py ${RUN_ARGS}
RUN_EXIT=$?
set -e

LATEST_REPORT="$(ls -t output/ai_report_*.md 2>/dev/null | awk 'NR==1{print}')"
LATEST_BACKTEST="$(ls -t output/backtest_*.png 2>/dev/null | awk 'NR==1{print}')"
LATEST_METRICS="$(ls -t output/metrics_*.json 2>/dev/null | awk 'NR==1{print}')"

if [ "$RUN_EXIT" -eq 0 ]; then
  MSG="✅ SVM 任务完成（${NOW}）。参数: ${RUN_ARGS}"
else
  MSG="❌ SVM 任务失败（${NOW}）。退出码: ${RUN_EXIT}。参数: ${RUN_ARGS}"
fi

if [ -n "${LATEST_REPORT}" ] && [ "$RUN_EXIT" -eq 0 ]; then
  REPORT_CONTENT="$(cat "$LATEST_REPORT")"
  ATTACH_INFO=""
  if [ -n "${LATEST_BACKTEST}" ]; then
    ATTACH_INFO="${ATTACH_INFO}\n回测图文件: $(basename "$LATEST_BACKTEST")"
  fi
  if [ -n "${LATEST_METRICS}" ]; then
    ATTACH_INFO="${ATTACH_INFO}\n指标文件: $(basename "$LATEST_METRICS")"
  fi
  "$OPENCLAW_BIN" message send \
    --channel feishu \
    --target "$TARGET_OPEN_ID" \
    --message "${MSG}${ATTACH_INFO}\n\n【报告正文】\n${REPORT_CONTENT}" || true
else
  "$OPENCLAW_BIN" message send \
    --channel feishu \
    --target "$TARGET_OPEN_ID" \
    --message "$MSG" || true
fi

exit "$RUN_EXIT"
