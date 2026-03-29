#!/bin/bash
BASE_DIR="/data1/intern/research_society/forks/aira-cogito"
cd "$BASE_DIR"
eval "$(conda shell.bash hook)"
conda activate aira-dojo
export PYTHONPATH="$BASE_DIR/src:$PYTHONPATH"
set -a && source "$BASE_DIR/.env" && set +a

EXP_ID="exp_a"
STEP_LIMIT=50
TIME_LIMIT=129600
NUM_GPUS=4
CONDITION="nat"
COND_ARGS="solver.use_cognitive_state=true solver.intervention_mode=natural"

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
LOG_DIR="$BASE_DIR/log/exp_a_seed3_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "Launching 10 seed_3 runs (1 per task, ~2-3 per GPU)"
gpu_idx=0
PIDS=()
for task in "${TASKS[@]}"; do
    gpu=$((gpu_idx % NUM_GPUS))
    gpu_idx=$((gpu_idx + 1))
    hydra_dir="outputs/${EXP_ID}/${CONDITION}/${task}/seed_3"
    log_file="$LOG_DIR/${CONDITION}_${task}_s3.log"
    echo "  $task seed=3 GPU=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu PYTORCH_JIT=0 nohup python -m dojo.main_run \
        +_exp=airsbench/aira_greedy_cogito_qwen_single \
        logger.use_wandb=False \
        task.name="$task" \
        metadata.git_issue_id="${EXP_ID}_${CONDITION}_${task}" \
        metadata.seed=3 \
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

echo "Launched ${#PIDS[@]} runs. PIDs: ${PIDS[*]}"
echo "Waiting for all to complete..."
for pid in "${PIDS[@]}"; do
    wait $pid 2>/dev/null
done
echo "All seed_3 runs completed."
