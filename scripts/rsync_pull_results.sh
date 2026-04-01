#!/bin/bash
# =============================================================================
# rsync_pull_results.sh — 从远端服务器拉取实验结果（排除大文件）
# =============================================================================
#
# 用法:
#   bash scripts/rsync_pull_results.sh <ssh_host> <ssh_port> [condition]
#
# 示例:
#   bash scripts/rsync_pull_results.sh connect.westc.seetacloud.com 43499
#   bash scripts/rsync_pull_results.sh connect.westc.seetacloud.com 43499 frz
# =============================================================================

set -euo pipefail

REMOTE_HOST="${1:?用法: $0 <ssh_host> <ssh_port> [condition]}"
REMOTE_PORT="${2:?用法: $0 <ssh_host> <ssh_port> [condition]}"
CONDITION="${3:-}"  # 留空则拉全部

LOCAL_BASE="/data1/intern/research_society/forks/aira-cogito"
REMOTE_BASE="~/Research_Society/forks/aira-cogito"

SSH_CMD="ssh -p $REMOTE_PORT"

echo "=========================================="
echo "  拉取远端实验结果"
echo "  远端: root@$REMOTE_HOST:$REMOTE_PORT"
echo "  条件: ${CONDITION:-全部}"
echo "=========================================="

# ---- 1. 拉取 outputs（排除大文件）----
if [ -n "$CONDITION" ]; then
    REMOTE_OUTPUTS="$REMOTE_BASE/outputs/exp_a/$CONDITION/"
    LOCAL_OUTPUTS="$LOCAL_BASE/outputs/exp_a/$CONDITION/"
else
    REMOTE_OUTPUTS="$REMOTE_BASE/outputs/"
    LOCAL_OUTPUTS="$LOCAL_BASE/outputs/"
fi

mkdir -p "$LOCAL_OUTPUTS"

echo ""
echo "[1/2] 拉取 outputs..."
rsync -avz --progress -e "$SSH_CMD" \
    --exclude='workspace_agent/' \
    --exclude='task_data/' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='*.safetensors' \
    --exclude='*.bin' \
    --exclude='*.h5' \
    --exclude='*.ckpt' \
    --exclude='*.sif' \
    "root@$REMOTE_HOST:$REMOTE_OUTPUTS" \
    "$LOCAL_OUTPUTS"

# ---- 2. 拉取启动日志 ----
echo ""
echo "[2/2] 拉取 log/..."
mkdir -p "$LOCAL_BASE/log/"

rsync -avz --progress -e "$SSH_CMD" \
    "root@$REMOTE_HOST:$REMOTE_BASE/log/" \
    "$LOCAL_BASE/log/"

# ---- 统计 ----
echo ""
echo "=========================================="
echo "  完成!"
echo "  outputs → $LOCAL_OUTPUTS"
echo "  logs    → $LOCAL_BASE/log_remote/"
echo ""
du -sh "$LOCAL_OUTPUTS" 2>/dev/null
du -sh "$LOCAL_BASE/log_remote/" 2>/dev/null
echo "=========================================="
