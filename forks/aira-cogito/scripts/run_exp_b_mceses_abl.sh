#!/bin/bash
# ================================================================
# Exp B: MC-ESES ABL
# Usage:
#   bash scripts/run_exp_b_mceses_abl.sh 1    # run seed 1
#   bash scripts/run_exp_b_mceses_abl.sh 2    # run seed 2
#   bash scripts/run_exp_b_mceses_abl.sh 3    # run seed 3
# ================================================================

SEED="${1:?Usage: $0 <seed_number>}"

BASE_DIR="${BASE_DIR:-/data1/intern/research_society/forks/aira-cogito}"
cd "$BASE_DIR"

eval "$(conda shell.bash hook)"
conda activate aira-dojo

export PYTHONPATH="$BASE_DIR/src:$PYTHONPATH"
set -a && source "$BASE_DIR/.env" && set +a

# ---- 参数 ----
EXP_ID="exp_b"
STEP_LIMIT=50
TIME_LIMIT=129600
NUM_GPUS=4
BATCH_SIZE=10  # 10 tasks × 1 seed

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
LOG_DIR="$BASE_DIR/log/exp_b_mceses_abl_seed${SEED}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

CONDITION="mceses_abl_v1"

echo "=========================================="
echo "  Exp B: MC-ESES ABL — seed $SEED"
echo "  num_children=5, managed_evolution=true"
echo "  intervention_mode=ablated"
echo "  Tasks: ${#TASKS[@]}"
echo "  Base dir: $BASE_DIR"
echo "  Log dir: $LOG_DIR"
echo "=========================================="

ALL_RUNS=()
for task in "${TASKS[@]}"; do
    ALL_RUNS+=("$task:$SEED")
done

TOTAL=${#ALL_RUNS[@]}
echo "Total runs: $TOTAL"

batch_start=0
batch_num=0

while [ $batch_start -lt $TOTAL ]; do
    batch_num=$((batch_num + 1))
    batch_end=$((batch_start + BATCH_SIZE))
    if [ $batch_end -gt $TOTAL ]; then
        batch_end=$TOTAL
    fi

    echo ""
    echo "======== Batch $batch_num: runs $((batch_start+1))-$batch_end of $TOTAL ========"

    gpu_idx=0
    PIDS=()

    for i in $(seq $batch_start $((batch_end - 1))); do
        run="${ALL_RUNS[$i]}"
        task="${run%%:*}"
        seed="${run##*:}"
        gpu=$((gpu_idx % NUM_GPUS))
        gpu_idx=$((gpu_idx + 1))

        hydra_dir="outputs/${EXP_ID}/${CONDITION}/${task}/seed_${seed}"
        log_file="$LOG_DIR/${CONDITION}_${task}_s${seed}.log"

        echo "  [$((i+1))/$TOTAL] $task seed=$seed GPU=$gpu"

        CUDA_VISIBLE_DEVICES=$gpu PYTORCH_JIT=0 nohup python -m dojo.main_run \
            +_exp=airsbench/aira_mceses_qwen \
            logger.use_wandb=False \
            task.name="$task" \
            metadata.git_issue_id="${EXP_ID}_${CONDITION}_${task}" \
            metadata.seed="$seed" \
            solver.step_limit="$STEP_LIMIT" \
            solver.time_limit_secs="$TIME_LIMIT" \
            solver.num_children=5 \
            solver.adaptive_children=false \
            solver.intervention_mode=ablated \
            solver.save_trajectory=true \
            hydra.run.dir="$hydra_dir" \
            +logger.output_dir="$hydra_dir" \
            > "$log_file" 2>&1 &

        PIDS+=($!)
        sleep 2
    done

    echo "  Waiting for batch $batch_num to complete (${#PIDS[@]} processes)..."
    for pid in "${PIDS[@]}"; do
        wait $pid 2>/dev/null
    done

    echo "  Batch $batch_num completed."
    batch_start=$batch_end
done

echo ""
echo "=========================================="
echo "  All $TOTAL runs completed!"
echo "=========================================="
