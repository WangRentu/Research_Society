#!/bin/bash
# ================================================================
# Exp A: 因果干预实验 (Causal Intervention)
# Greedy (5 drafts) × 4 conditions × 10 tasks × 3 seeds = 120 runs
#
# D1: G1-NAT + G1-ABL = 60 runs
# D2: G1-SCR + G1-FRZ = 60 runs
#
# 使用 PythonInterpreter (子进程模式)
# ================================================================

BASE_DIR="/data1/intern/research_society/forks/aira-cogito"
cd "$BASE_DIR"

eval "$(conda shell.bash hook)"
conda activate aira-dojo

export PYTHONPATH="$BASE_DIR/src:$PYTHONPATH"
set -a && source "$BASE_DIR/.env" && set +a

# ---- 实验参数 ----
EXP_ID="${EXP_ID:-exp_a}"
STEP_LIMIT=50
TIME_LIMIT=129600
NUM_GPUS=4
SEED_START=1
SEED_END=3

# 实验条件 (通过命令行参数选择)
# 用法: bash scripts/run_exp_a.sh [nat|abl|scr|frz|all]
CONDITION="${1:-all}"

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

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$BASE_DIR/log/${EXP_ID}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# ---- 启动函数 ----
launch_condition() {
    local cond_name="$1"    # nat, abl, scr, frz
    local cond_args="$2"    # solver override args
    local gpu_idx=0
    local run_count=0

    echo ""
    echo "========================================"
    echo "  Condition: $cond_name"
    echo "  Args: $cond_args"
    echo "========================================"

    for task in "${TASKS[@]}"; do
        for seed in $(seq $SEED_START $SEED_END); do
            gpu=$(( gpu_idx % NUM_GPUS ))
            gpu_idx=$(( gpu_idx + 1 ))
            run_count=$(( run_count + 1 ))

            issue_id="${EXP_ID}_${cond_name}_${task}"
            hydra_dir="outputs/${EXP_ID}/${cond_name}/${task}/seed_${seed}"
            log_file="$LOG_DIR/${cond_name}_${task}_s${seed}.log"

            echo "  [$run_count] $task seed=$seed GPU=$gpu ($cond_name)"

            CUDA_VISIBLE_DEVICES=$gpu PYTORCH_JIT=0 nohup python -m dojo.main_run \
                +_exp=airsbench/aira_greedy_cogito_qwen_single \
                logger.use_wandb=False \
                task.name="$task" \
                metadata.git_issue_id="$issue_id" \
                metadata.seed="$seed" \
                solver.step_limit="$STEP_LIMIT" \
                solver.time_limit_secs="$TIME_LIMIT" \
                solver.save_trajectory=true \
                $cond_args \
                hydra.run.dir="$hydra_dir" \
                +logger.output_dir="$hydra_dir" \
                > "$log_file" 2>&1 &

            # 每 GPU×2 个任务暂停一下，避免同时启动过多
            if (( run_count % (NUM_GPUS * 2) == 0 )); then
                echo "  ... waiting 15s for batch to stabilize ..."
                sleep 15
            fi
        done
    done
    echo "  → $run_count runs launched for condition $cond_name"
}

# ---- Scrambled state 准备 ----
# SCR 条件需要先跑 NAT，从其输出中提取 cognitive_state.json
# 交叉方式: task[i] 的 scrambled state 来自 task[(i+5) % 10]
get_scramble_source() {
    local task_idx="$1"
    local source_idx=$(( (task_idx + 5) % 10 ))
    local source_task="${TASKS[$source_idx]}"
    # 从 NAT seed 1 的输出中取
    echo "outputs/${EXP_ID}/nat/${source_task}/seed_1/cognitive_state.json"
}

# ---- 主逻辑 ----
echo "=========================================="
echo "  Exp A: Causal Intervention"
echo "  Tasks: ${#TASKS[@]}"
echo "  Seeds: $SEED_START-$SEED_END"
echo "  Steps: $STEP_LIMIT"
echo "  Interpreter: Apptainer (jupyter)"
echo "  Condition: $CONDITION"
echo "  Log dir: $LOG_DIR"
echo "=========================================="

case "$CONDITION" in
    nat)
        launch_condition "nat" \
            "solver.use_cognitive_state=true solver.intervention_mode=natural"
        ;;
    abl)
        launch_condition "abl" \
            "solver.use_cognitive_state=true solver.intervention_mode=ablated"
        ;;
    scr)
        # SCR 需要 NAT 先跑完产生 cognitive_state.json
        echo "NOTE: SCR condition requires NAT to have completed first!"
        echo "Launching SCR with cross-task scrambled states..."
        gpu_idx=0
        run_count=0
        for i in "${!TASKS[@]}"; do
            task="${TASKS[$i]}"
            scramble_src=$(get_scramble_source "$i")
            if [ ! -f "$BASE_DIR/$scramble_src" ]; then
                echo "  WARNING: $scramble_src not found, skipping $task SCR"
                continue
            fi
            for seed in $(seq $SEED_START $SEED_END); do
                gpu=$(( gpu_idx % NUM_GPUS ))
                gpu_idx=$(( gpu_idx + 1 ))
                run_count=$(( run_count + 1 ))

                issue_id="${EXP_ID}_scr_${task}"
                hydra_dir="outputs/${EXP_ID}/scr/${task}/seed_${seed}"
                log_file="$LOG_DIR/scr_${task}_s${seed}.log"

                echo "  [$run_count] $task seed=$seed GPU=$gpu (scr from ${TASKS[$(( (i+5) % 10 ))]})"

                CUDA_VISIBLE_DEVICES=$gpu PYTORCH_JIT=0 nohup python -m dojo.main_run \
                    +_exp=airsbench/aira_greedy_cogito_qwen_single \
                        logger.use_wandb=False \
                    task.name="$task" \
                    metadata.git_issue_id="$issue_id" \
                    metadata.seed="$seed" \
                    solver.step_limit="$STEP_LIMIT" \
                    solver.time_limit_secs="$TIME_LIMIT" \
                    solver.save_trajectory=true \
                    solver.use_cognitive_state=true \
                    solver.intervention_mode=scrambled \
                    solver.scramble_source_path="$BASE_DIR/$scramble_src" \
                    hydra.run.dir="$hydra_dir" \
                +logger.output_dir="$hydra_dir" \
                    > "$log_file" 2>&1 &

                if (( run_count % (NUM_GPUS * 2) == 0 )); then
                    sleep 15
                fi
            done
        done
        echo "  → $run_count runs launched for condition scr"
        ;;
    frz)
        launch_condition "frz" \
            "solver.use_cognitive_state=true solver.intervention_mode=frozen solver.intervention_freeze_step=5"
        ;;
    all)
        echo "Launching NAT + ABL (D1 batch, 60 runs)"
        launch_condition "nat" \
            "solver.use_cognitive_state=true solver.intervention_mode=natural"
        launch_condition "abl" \
            "solver.use_cognitive_state=true solver.intervention_mode=ablated"
        echo ""
        echo "NOTE: SCR + FRZ should be run after NAT completes (D2)."
        echo "  bash scripts/run_exp_a.sh scr"
        echo "  bash scripts/run_exp_a.sh frz"
        ;;
    *)
        echo "Usage: bash scripts/run_exp_a.sh [nat|abl|scr|frz|all]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "  Launched! Monitor:"
echo "    tail -f $LOG_DIR/*.log"
echo "    ps aux | grep dojo.main_run | wc -l"
echo "    kill \$(pgrep -f dojo.main_run)"
echo "=========================================="
