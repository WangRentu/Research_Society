#!/bin/bash
# ================================================================
# Experiment 3: Causal Intervention on Cognitive State z_t
# "When Does Thinking Help?" paper
#
# 4 conditions × 10 tasks × 3 seeds = 120 runs
# num_children=1 (linear chain, controlled experiment)
#
# Conditions:
#   natural   — full z_t evolution (control)
#   ablated   — z_t reset to z_0 after every step
#   frozen    — z_t frozen at step 5
#   scrambled — z_t from a different task
# ================================================================

BASE_DIR="/data1/intern/research_society/forks/aira-cogito"
cd "$BASE_DIR"

eval "$(conda shell.bash hook)"
conda activate aira-dojo

export PYTHONPATH="$BASE_DIR/src:$PYTHONPATH"

set -a
source "$BASE_DIR/.env"
set +a

RUN_ID="${RUN_ID:-exp3}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$BASE_DIR/log/${RUN_ID}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

SEED_START=1
SEED_END=3
STEP_LIMIT=30
TIME_LIMIT=43200
NUM_GPUS=4

# 10 representative tasks (Easy → Expert)
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

CONDITIONS=("natural" "ablated" "frozen" "scrambled")

# For scrambled condition: use z_t from TextualSimilarity for all tasks
# (a real z_t from a different domain — ensures it's task-irrelevant)
SCRAMBLE_SOURCE="/data1/intern/research_society/forks/aira-cogito/shared/logs/aira-cogito/user_jingyixi_issue_run008_mceses_TextualSimilaritySickSpearmanCorrelation/user_jingyixi_issue_run008_mceses_TextualSimilaritySickSpearmanCorrelation_seed_1_id_8092fb43a61ed6b9efe4958105766526519783294e6f2b54fccc5627/checkpoint/cognitive_state.json"

NUM_SEEDS=$(( SEED_END - SEED_START + 1 ))
TOTAL_RUNS=$(( ${#TASKS[@]} * ${#CONDITIONS[@]} * NUM_SEEDS ))

echo "=========================================="
echo "  Exp 3: Causal Intervention"
echo "  ${#TASKS[@]} tasks × ${#CONDITIONS[@]} conditions × $NUM_SEEDS seeds"
echo "  Total runs: $TOTAL_RUNS"
echo "  num_children=1 (linear chain)"
echo "  Log dir: $LOG_DIR"
echo "=========================================="

gpu_idx=0
run_count=0

for task in "${TASKS[@]}"; do
    for cond in "${CONDITIONS[@]}"; do
        for seed in $(seq $SEED_START $SEED_END); do
            gpu=$(( gpu_idx % NUM_GPUS ))
            gpu_idx=$(( gpu_idx + 1 ))
            run_count=$(( run_count + 1 ))

            issue_id="${RUN_ID}_${cond}_${task}"
            hydra_dir="outputs/${RUN_ID}/${cond}/${task}/seed_${seed}"

            # Build extra args for each condition
            EXTRA_ARGS=""
            if [ "$cond" = "scrambled" ]; then
                EXTRA_ARGS="solver.scramble_source_path=$SCRAMBLE_SOURCE"
            fi

            echo "[$run_count/$TOTAL_RUNS] $cond | $task | seed=$seed (GPU $gpu)"

            CUDA_VISIBLE_DEVICES=$gpu PYTORCH_JIT=0 nohup python -m dojo.main_run \
                +_exp=airsbench/exp3_intervention_single \
                logger.use_wandb=False \
                task.name="$task" \
                metadata.git_issue_id="$issue_id" \
                metadata.seed="$seed" \
                solver.step_limit="$STEP_LIMIT" \
                solver.time_limit_secs="$TIME_LIMIT" \
                solver.intervention_mode="$cond" \
                solver.num_children=1 \
                $EXTRA_ARGS \
                hydra.run.dir="$hydra_dir" \
                > "$LOG_DIR/${cond}_${task}_seed${seed}.log" 2>&1 &

            # Throttle: max NUM_GPUS*2 concurrent
            if (( run_count % (NUM_GPUS * 2) == 0 )); then
                echo "  ... waiting 15s for batch to start ..."
                sleep 15
            fi
        done
    done
done

echo ""
echo "=========================================="
echo "  $TOTAL_RUNS runs launched!"
echo "  Monitor: tail -f $LOG_DIR/*.log"
echo "  Progress: ps aux | grep exp3 | grep dojo | wc -l"
echo "  Stop all: pkill -f exp3"
echo "=========================================="
