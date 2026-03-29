#!/bin/bash
# ================================================================
# Exp A: 分批运行 (每批 8 个进程，每 GPU 2 个)
# 避免 GPU 争抢导致的卡死问题
#
# 用法: bash scripts/run_exp_a_batch.sh nat
#       bash scripts/run_exp_a_batch.sh abl
# ================================================================

BASE_DIR="/data1/intern/research_society/forks/aira-cogito"
cd "$BASE_DIR"

eval "$(conda shell.bash hook)"
conda activate aira-dojo

export PYTHONPATH="$BASE_DIR/src:$PYTHONPATH"
set -a && source "$BASE_DIR/.env" && set +a

# ---- 参数 ----
EXP_ID="${EXP_ID:-exp_a}"
STEP_LIMIT=50
TIME_LIMIT=129600
NUM_GPUS=4
BATCH_SIZE=20  # 第一批 20 (10 tasks × 2 seeds)，第二批 10 (10 tasks × 1 seed)
SEED_START=1
SEED_END=3
CONDITION="${1:-nat}"

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

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$BASE_DIR/log/${EXP_ID}_batch_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# 设置 condition 参数
case "$CONDITION" in
    nat) COND_ARGS="solver.use_cognitive_state=true solver.intervention_mode=natural" ;;
    abl) COND_ARGS="solver.use_cognitive_state=true solver.intervention_mode=ablated" ;;
    *) echo "Usage: $0 [nat|abl]"; exit 1 ;;
esac

echo "=========================================="
echo "  Exp A Batch Runner"
echo "  Condition: $CONDITION"
echo "  Tasks: ${#TASKS[@]}  Seeds: $SEED_START-$SEED_END"
echo "  Batch size: $BATCH_SIZE (${NUM_GPUS} GPUs × 2)"
echo "  Log dir: $LOG_DIR"
echo "=========================================="

# 构建所有 (task, seed) 组合
# 先跑 seed 1,2（20个），再跑 seed 3（10个）
ALL_RUNS=()
for seed in 1 2; do
    for task in "${TASKS[@]}"; do
        ALL_RUNS+=("$task:$seed")
    done
done
for task in "${TASKS[@]}"; do
    ALL_RUNS+=("$task:3")
done

TOTAL=${#ALL_RUNS[@]}
echo "Total runs: $TOTAL"

# 分批执行
batch_start=0
batch_num=0

while [ $batch_start -lt $TOTAL ]; do
    batch_num=$((batch_num + 1))
    batch_end=$((batch_start + BATCH_SIZE))
    if [ $batch_end -gt $TOTAL ]; then
        batch_end=$TOTAL
    fi

    current_batch_size=$((batch_end - batch_start))
    echo ""
    echo "======== Batch $batch_num: runs $((batch_start+1))-$batch_end of $TOTAL ($current_batch_size runs) ========"

    gpu_idx=0
    PIDS=()

    for i in $(seq $batch_start $((batch_end - 1))); do
        run="${ALL_RUNS[$i]}"
        task="${run%%:*}"
        seed="${run##*:}"
        gpu=$((gpu_idx % NUM_GPUS))
        gpu_idx=$((gpu_idx + 1))

        issue_id="${EXP_ID}_${CONDITION}_${task}"
        hydra_dir="outputs/${EXP_ID}/${CONDITION}/${task}/seed_${seed}"
        log_file="$LOG_DIR/${CONDITION}_${task}_s${seed}.log"

        echo "  [$((i+1))/$TOTAL] $task seed=$seed GPU=$gpu"

        CUDA_VISIBLE_DEVICES=$gpu PYTORCH_JIT=0 nohup python -m dojo.main_run \
            +_exp=airsbench/aira_greedy_cogito_qwen_single \
            logger.use_wandb=False \
            task.name="$task" \
            metadata.git_issue_id="$issue_id" \
            metadata.seed="$seed" \
            solver.step_limit="$STEP_LIMIT" \
            solver.time_limit_secs="$TIME_LIMIT" \
            solver.save_trajectory=true \
            $COND_ARGS \
            hydra.run.dir="$hydra_dir" \
            +logger.output_dir="$hydra_dir" \
            > "$log_file" 2>&1 &

        PIDS+=($!)
        sleep 2
    done

    echo "  Waiting for batch $batch_num to complete (${#PIDS[@]} processes)..."
    echo "  PIDs: ${PIDS[*]}"

    # 等待这一批全部完成
    for pid in "${PIDS[@]}"; do
        wait $pid 2>/dev/null
    done

    echo "  Batch $batch_num completed."
    batch_start=$batch_end
done

echo ""
echo "=========================================="
echo "  All $TOTAL runs completed!"
echo "  Log dir: $LOG_DIR"
echo "=========================================="
   