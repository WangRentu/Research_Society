#!/bin/bash
# =============================================================================
# cleanup_workspace.sh — 边跑边清理 workspace_agent 中的模型文件
# =============================================================================
#
# 背景：
#   aira-cogito 的每个实验 run 共用一个 workspace_agent/ 目录。
#   Agent 在每步生成的代码会训练模型并保存 .pt/.safetensors/.bin 文件。
#   这些文件在该步评估完成后就不再被框架读取，但会持续累积（单 run 最大 90GB）。
#
# 安全性分析：
#   - 框架在每步只读 submission.csv，不读模型文件 → 删除安全
#   - workspace_agent/data 是符号链接指向任务数据 → 绝对不能删
#   - solution.py 是当前正在执行的代码 → 不删 .py 文件
#   - improve_op 的 LLM 可能生成 torch.load("old_model.pt") 的代码
#     → 极少发生（prompt 要求 SELF-CONTAINED），发生了也只是一个 buggy step
#
# 用法：
#   chmod +x cleanup_workspace.sh
#
#   # 手动执行一次
#   ./cleanup_workspace.sh /path/to/outputs
#
#   # 加入 cron，每 10 分钟执行
#   crontab -e
#   */10 * * * * /path/to/cleanup_workspace.sh /path/to/outputs >> /tmp/cleanup.log 2>&1
#
# =============================================================================

set -euo pipefail

# ─── 配置 ──────────────────────────────────────────────────────────────────
OUTPUTS_DIR="${1:?用法: $0 <outputs_dir>}"   # 必须传入 outputs 目录
AGE_MINUTES="${2:-30}"                        # 文件年龄阈值（分钟），默认 30
DRY_RUN="${3:-false}"                         # 设为 true 只打印不删除
# ───────────────────────────────────────────────────────────────────────────

if [ ! -d "$OUTPUTS_DIR" ]; then
    echo "[ERROR] 目录不存在: $OUTPUTS_DIR"
    exit 1
fi

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
TOTAL_FREED=0
FILES_DELETED=0
DIRS_DELETED=0

echo "[$TIMESTAMP] 开始清理 workspace_agent (阈值: ${AGE_MINUTES}min, dry_run: ${DRY_RUN})"
echo "  目标目录: $OUTPUTS_DIR"

# ─── 1. 删除散落的大模型文件 ──────────────────────────────────────────────
#    匹配: .pt .pth .safetensors .bin .h5 .ckpt .pkl（大于 1MB 且超过阈值）
#    排除: workspace_agent/data/ (符号链接到任务数据，绝对不碰)
while IFS= read -r -d '' file; do
    size=$(stat --format='%s' "$file" 2>/dev/null || echo 0)
    size_mb=$((size / 1024 / 1024))
    if [ "$DRY_RUN" = "true" ]; then
        echo "  [DRY] 会删除: $file (${size_mb}MB)"
    else
        rm -f "$file"
        FILES_DELETED=$((FILES_DELETED + 1))
        TOTAL_FREED=$((TOTAL_FREED + size_mb))
    fi
done < <(find "$OUTPUTS_DIR" -path "*/workspace_agent/*" \
    ! -path "*/workspace_agent/data/*" \
    ! -path "*/workspace_agent/data" \
    -type f \
    \( -name "*.pt" -o -name "*.pth" -o -name "*.safetensors" \
       -o -name "*.bin" -o -name "*.h5" -o -name "*.ckpt" \) \
    -size +1M \
    -mmin +"$AGE_MINUTES" \
    -print0 2>/dev/null)

# ─── 2. 删除 HuggingFace checkpoint 目录 ─────────────────────────────────
#    匹配: checkpoint-N/ 目录（HuggingFace Trainer 标准命名）
while IFS= read -r -d '' dir; do
    size=$(du -sm "$dir" 2>/dev/null | cut -f1 || echo 0)
    if [ "$DRY_RUN" = "true" ]; then
        echo "  [DRY] 会删除目录: $dir (${size}MB)"
    else
        rm -rf "$dir"
        DIRS_DELETED=$((DIRS_DELETED + 1))
        TOTAL_FREED=$((TOTAL_FREED + size))
    fi
done < <(find "$OUTPUTS_DIR" -path "*/workspace_agent/*" \
    ! -path "*/workspace_agent/data/*" \
    -type d \
    -name "checkpoint-*" \
    -mmin +"$AGE_MINUTES" \
    -print0 2>/dev/null)

# ─── 3. 删除空的模型子目录（清理残留） ────────────────────────────────────
#    上一步删完文件后，可能留下空目录
if [ "$DRY_RUN" != "true" ]; then
    find "$OUTPUTS_DIR" -path "*/workspace_agent/*" \
        ! -path "*/workspace_agent/data/*" \
        ! -path "*/workspace_agent/data" \
        ! -path "*/workspace_agent" \
        -type d -empty \
        -delete 2>/dev/null || true
fi

# ─── 4. 统计当前 workspace 占用 ──────────────────────────────────────────
CURRENT_USAGE=$(find "$OUTPUTS_DIR" -path "*/workspace_agent" -type d \
    -exec du -sm {} + 2>/dev/null | awk '{s+=$1} END {print s+0}')

echo "  删除文件: ${FILES_DELETED} 个"
echo "  删除目录: ${DIRS_DELETED} 个"
echo "  释放空间: ~${TOTAL_FREED} MB"
echo "  当前 workspace 总占用: ${CURRENT_USAGE} MB"
echo "[$TIMESTAMP] 清理完成"
echo ""
