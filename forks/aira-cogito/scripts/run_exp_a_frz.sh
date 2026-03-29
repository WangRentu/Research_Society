#!/bin/bash
# ================================================================
# Exp A - FRZ: Frozen cognitive state intervention
# Greedy (5 drafts) × 10 tasks × 3 seeds = 30 runs
#
# FRZ: 认知状态在 step N_f 后冻结，不再更新
# 测试持续演化的边际价值
#
# 数据隔离:
#   shared/logs/  — 框架默认日志 (LOGGING_DIR)
#   outputs/      — 实验数据 (checkpoint, results, workspace_agent)
#   log/          — 启动日志 (nohup stdout)
# ================================================================

# ---- 路径配置（自动检测）----
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

eval "$(conda shell.bash hook)"
conda activate aira-dojo

export PYTHONPATH="$BASE_DIR/src:$PYTHONPATH"
set -a && source "$BASE_DIR/.env" && set +a

# ---- 创建目录结构 ----
mkdir -p "$BASE_DIR/shared/logs"
mkdir -p "$BASE_DIR/outputs"

# ---- 实验参数 ----
EXP_ID="exp_a"
CONDITION="frz"
STEP_LIMIT=50
TIME_LIMIT=129600        # 36 小时
FREEZE_STEP="${FREEZE_STEP:-5}"  # 冻结步数，默认 step 5

# GPU 数量自动检测
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_GPUS=${NUM_GPUS:-1}

SEED_START="${SEED_START:-1}"
SEED_END="${SEED_END:-3}"

# 核心 10 任务
TASKS=(
    "TextualClassificationSickAccuracy"
    "TimeSeriesForecastingSolarWeeklyMAE"
    "SentimentAnalysisYelpReviewFullAccuracy"
    "ReadingComprehensionSquadExactMatch"
    "GraphRegressionZincMae"
    "CoreferenceResolutionWinograndeAccuracy"
    "CoreferenceResolutionSuperGLUEWSCAccuracy"
    "CodeRetrievalCodeXGlueMRR"
    "MathQuestionAnsweringSVAMPAccuracy"
    "CvMolecularPropertyPredictionQm9MeanAbsoluteError"
)

# ---- 启动日志目录（与 outputs 分离）----
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$BASE_DIR/log/${EXP_ID}_${CONDITION}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# ---- 启动 ----
echo "=========================================="
echo "  Exp A - FRZ (Frozen Cognitive State)"
echo "  freeze_step: $FREEZE_STEP"
echo "  tasks:       ${#TASKS[@]}"
echo "  seeds:       $SEED_START-$SEED_END"
echo "  step_limit:  $STEP_LIMIT"
echo "  GPUs:        $NUM_GPUS"
echo "  BASE_DIR:    $BASE_DIR"
echo "  outputs →    $BASE_DIR/outputs/"
echo "  shared →     $BASE_DIR/shared/logs/"
echo "  launch log → $LOG_DIR/"
echo "=========================================="

gpu_idx=0
run_count=0
PIDS=()

for task in "${TASKS[@]}"; do
    for seed in $(seq $SEED_START $SEED_END); do
        gpu=$(( gpu_idx % NUM_GPUS ))
        gpu_idx=$(( gpu_idx + 1 ))
        run_count=$(( run_count + 1 ))

        hydra_dir="outputs/${EXP_ID}/${CONDITION}/${task}/seed_${seed}"
        log_file="$LOG_DIR/${CONDITION}_${task}_s${seed}.log"

        echo "  [$run_count] $task seed=$seed GPU=$gpu"

        CUDA_VISIBLE_DEVICES=$gpu PYTORCH_JIT=0 nohup python -m dojo.main_run \
            +_exp=airsbench/aira_greedy_cogito_qwen_single \
            logger.use_wandb=False \
            task.name="$task" \
            metadata.git_issue_id="${EXP_ID}_${CONDITION}_${task}" \
            metadata.seed="$seed" \
            solver.step_limit="$STEP_LIMIT" \
            solver.time_limit_secs="$TIME_LIMIT" \
            solver.save_trajectory=true \
            solver.use_cognitive_state=true \
            solver.intervention_mode=frozen \
            solver.intervention_freeze_step="$FREEZE_STEP" \
            hydra.run.dir="$hydra_dir" \
            +logger.output_dir="$hydra_dir" \
            > "$log_file" 2>&1 &

        PIDS+=($!)

        # 每批暂停，避免同时启动过多
        if (( run_count % (NUM_GPUS * 2) == 0 )); then
            echo "  ... waiting 10s for batch to stabilize ..."
            sleep 10
        fi
    done
done

echo ""
echo "=========================================="
echo "  Launched $run_count runs"
echo "  PIDs: ${PIDS[*]:0:10}..."
echo ""
echo "  Monitor:"
echo "    tail -f $LOG_DIR/*.log"
echo "    ps aux | grep dojo.main_run | grep -v grep | wc -l"
echo ""
echo "  Stop all:"
echo "    kill $(echo ${PIDS[*]})"
echo "=========================================="

# 可选：等待所有完成
# for pid in "${PIDS[@]}"; do wait $pid 2>/dev/null; done
# echo "All FRZ runs completed."
