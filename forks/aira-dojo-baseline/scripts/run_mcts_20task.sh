#!/bin/bash
# ================================================================
# Run 005: MCTS × 20 tasks × 2 seeds
# Baseline AIRA-dojo MCTS scaffold + Qwen 3.5-397B
# 30 step limit, 4× RTX 5090
# ================================================================

BASE_DIR="/data1/intern/research_society/forks/aira-dojo-baseline"
cd "$BASE_DIR"

eval "$(conda shell.bash hook)"
conda activate aira-dojo

export PYTHONPATH="$BASE_DIR/src:$PYTHONPATH"

RUN_ID="run005"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$BASE_DIR/log/${RUN_ID}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

SEED_START=1
SEED_END=2
STEP_LIMIT=30
TIME_LIMIT=43200
NUM_GPUS=4

TASKS=(
    "CodeGenerationAPPSPassAt5"
    "CodeRetrievalCodeXGlueMRR"
    "CoreferenceResolutionSuperGLUEWSCAccuracy"
    "CoreferenceResolutionWinograndeAccuracy"
    "CvMolecularPropertyPredictionQm9MeanAbsoluteError"
    "GMolecularPropertyPredictionQm9MeanAbsoluteError"
    "GraphRegressionZincMae"
    "MathQuestionAnsweringSVAMPAccuracy"
    "QuestionAnsweringDuoRCAccuracy"
    "QuestionAnsweringEli5Rouge1"
    "QuestionAnsweringFinqaAccuracy"
    "R2AbsMolecularPropertyPredictionQm9MeanAbsoluteError"
    "ReadingComprehensionSquadExactMatch"
    "SentimentAnalysisYelpReviewFullAccuracy"
    "TextualClassificationSickAccuracy"
    "TextualSimilaritySickSpearmanCorrelation"
    "TimeSeriesForecastingKaggleWebTrafficMASE"
    "TimeSeriesForecastingRideshareMAE"
    "TimeSeriesForecastingSolarWeeklyMAE"
    "U0MolecularPropertyPredictionQm9MeanAbsoluteError"
)

NUM_SEEDS=$(( SEED_END - SEED_START + 1 ))
TOTAL_RUNS=$(( ${#TASKS[@]} * NUM_SEEDS ))

echo "=========================================="
echo "  $RUN_ID: MCTS × ${#TASKS[@]} tasks × $NUM_SEEDS seeds"
echo "  Step limit: $STEP_LIMIT"
echo "  Log dir: $LOG_DIR"
echo "  Total runs: $TOTAL_RUNS"
echo "=========================================="

gpu_idx=0
run_count=0

for task in "${TASKS[@]}"; do
    for seed in $(seq $SEED_START $SEED_END); do
        gpu=$(( gpu_idx % NUM_GPUS ))
        gpu_idx=$(( gpu_idx + 1 ))
        run_count=$(( run_count + 1 ))

        issue_id="${RUN_ID}_mcts_${task}"
        hydra_dir="outputs/${RUN_ID}/${task}/seed_${seed}"

        echo "[$run_count/$TOTAL_RUNS] $task seed=$seed (GPU $gpu)"

        CUDA_VISIBLE_DEVICES=$gpu PYTORCH_JIT=0 nohup python -m dojo.main_run \
            +_exp=airsbench/aira_mcts_qwen_single \
            logger.use_wandb=False \
            task.name="$task" \
            metadata.git_issue_id="$issue_id" \
            metadata.seed="$seed" \
            solver.step_limit="$STEP_LIMIT" \
            solver.time_limit_secs="$TIME_LIMIT" \
            hydra.run.dir="$hydra_dir" \
            > "$LOG_DIR/${task}_seed${seed}.log" 2>&1 &

        # Throttle: don't launch more than NUM_GPUS*2 at once
        if (( run_count % (NUM_GPUS * 2) == 0 )); then
            echo "  ... waiting 10s for batch to start ..."
            sleep 10
        fi
    done
done

echo ""
echo "=========================================="
echo "  $TOTAL_RUNS runs launched!"
echo "  Monitor: tail -f $LOG_DIR/*.log"
echo "  Progress: ps aux | grep dojo.main_run | wc -l"
echo "  Stop all: kill \$(pgrep -f dojo.main_run)"
echo "=========================================="
