#!/bin/bash
# ================================================================
# Exp A - NAT+: Greedy + CognitiveState + Managed Evolution
# Greedy (5 drafts) × 10 tasks × N seeds
#
# NAT+ = NAT + managed_evolution (triggered reflection + state decay)
#
# Usage:
#   bash scripts/run_exp_a_nat_plus.sh          # seed 1,2
#   SEED_START=3 SEED_END=3 bash scripts/run_exp_a_nat_plus.sh  # seed 3 only
# ================================================================

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

eval "$(conda shell.bash hook)"
conda activate aira-dojo

export PYTHONPATH="$BASE_DIR/src:$PYTHONPATH"
set -a && source "$BASE_DIR/.env" && set +a

# ---- 实验参数 ----
EXP_ID="exp_a"
CONDITION="nat_plus"
STEP_LIMIT=50
TIME_LIMIT=129600

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_GPUS=${NUM_GPUS:-1}

SEED_START="${SEED_START:-1}"
SEED_END="${SEED_END:-2}"

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
LOG_DIR="$BASE_DIR/log/${EXP_ID}_${CONDITION}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "  Exp A - NAT+ (Managed Evolution)"
echo "  tasks:       ${#TASKS[@]}"
echo "  seeds:       $SEED_START-$SEED_END"
echo "  step_limit:  $STEP_LIMIT"
echo "  GPUs:        $NUM_GPUS"
echo "  outputs →    outputs/${EXP_ID}/${CONDITION}/"
echo "  log →        $LOG_DIR/"
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
            solver.intervention_mode=natural \
            solver.managed_evolution=true \
            hydra.run.dir="$hydra_dir" \
            +logger.output_dir="$hydra_dir" \
            > "$log_file" 2>&1 &

        PIDS+=($!)

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
echo "    kill ${PIDS[*]}"
echo "=========================================="
