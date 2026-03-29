#!/bin/bash
# ================================================================
# Exp B: 跨策略迁移实验 (Cross-Strategy Transfer)
# 3 搜索结构 × ±z_t × 10 tasks × 3 seeds
#
# C0: Chain 无 z_t      (MC-ESES num_children=1, reflect off)
# C1: Chain + z_t        (MC-ESES num_children=1, reflect on)
# G0: Greedy 无 z_t      (Greedy 5 drafts, use_cognitive_state=false)
# G1: 复用 Exp A NAT     (不跑)
# M0: MCTS 无 z_t        (MC-ESES num_children=5, reflect off)
# M1: MC-ESES 完整版     (MC-ESES num_children=5, reflect on)
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
EXP_ID="${EXP_ID:-exp_b}"
STEP_LIMIT=50
TIME_LIMIT=43200
NUM_GPUS=4
SEED_START=1
SEED_END=3

CONDITION="${1:-all}"

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

# ---- 通用启动函数 ----
launch() {
    local cond_name="$1"
    local exp_config="$2"
    local extra_args="$3"
    local gpu_idx=0
    local run_count=0

    echo ""
    echo "========================================"
    echo "  Condition: $cond_name"
    echo "  Config: $exp_config"
    echo "  Extra: $extra_args"
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
                +_exp="$exp_config" \
                logger.use_wandb=False \
                task.name="$task" \
                metadata.git_issue_id="$issue_id" \
                metadata.seed="$seed" \
                solver.step_limit="$STEP_LIMIT" \
                solver.time_limit_secs="$TIME_LIMIT" \
                solver.save_trajectory=true \
                $extra_args \
                hydra.run.dir="$hydra_dir" \
                +logger.output_dir="$hydra_dir" \
                > "$log_file" 2>&1 &

            if (( run_count % (NUM_GPUS * 2) == 0 )); then
                echo "  ... waiting 15s ..."
                sleep 15
            fi
        done
    done
    echo "  → $run_count runs launched for $cond_name"
}

# ---- 主逻辑 ----
echo "=========================================="
echo "  Exp B: Cross-Strategy Transfer"
echo "  Tasks: ${#TASKS[@]}"
echo "  Seeds: $SEED_START-$SEED_END"
echo "  Steps: $STEP_LIMIT"
echo "  Condition: $CONDITION"
echo "  Log dir: $LOG_DIR"
echo "=========================================="

case "$CONDITION" in
    c0)
        # Chain 无 z_t: MC-ESES config, num_children=1, reflect off
        launch "c0" "airsbench/aira_mceses_qwen_single" \
            "solver.num_children=1 solver.reflect_after_every_step=false"
        ;;
    c1)
        # Chain + z_t: MC-ESES config, num_children=1, reflect on
        launch "c1" "airsbench/aira_mceses_qwen_single" \
            "solver.num_children=1 solver.reflect_after_every_step=true solver.intervention_mode=natural"
        ;;
    g0)
        # Greedy 无 z_t
        launch "g0" "airsbench/aira_greedy_cogito_qwen_single" \
            "solver.use_cognitive_state=false"
        ;;
    g1)
        echo "G1 = Exp A NAT. 复用 Exp A 数据，无需重跑。"
        echo "如果 Exp A NAT 尚未完成，先跑: bash scripts/run_exp_a.sh nat"
        ;;
    m0)
        # MCTS 无 z_t: MC-ESES config, num_children=5, reflect off
        launch "m0" "airsbench/aira_mceses_qwen_single" \
            "solver.num_children=5 solver.reflect_after_every_step=false"
        ;;
    m1)
        # MC-ESES 完整版: num_children=5, reflect on
        launch "m1" "airsbench/aira_mceses_qwen_single" \
            "solver.num_children=5 solver.reflect_after_every_step=true solver.intervention_mode=natural"
        ;;
    d3)
        # D3 批次: G0 + C0 = 60 runs
        echo "D3 batch: G0 + C0"
        launch "g0" "airsbench/aira_greedy_cogito_qwen_single" \
            "solver.use_cognitive_state=false"
        launch "c0" "airsbench/aira_mceses_qwen_single" \
            "solver.num_children=1 solver.reflect_after_every_step=false"
        ;;
    d4)
        # D4 批次: C1 + M0 = 60 runs
        echo "D4 batch: C1 + M0"
        launch "c1" "airsbench/aira_mceses_qwen_single" \
            "solver.num_children=1 solver.reflect_after_every_step=true solver.intervention_mode=natural"
        launch "m0" "airsbench/aira_mceses_qwen_single" \
            "solver.num_children=5 solver.reflect_after_every_step=false"
        ;;
    d5)
        # D5 批次: M1 = 30 runs
        echo "D5 batch: M1"
        launch "m1" "airsbench/aira_mceses_qwen_single" \
            "solver.num_children=5 solver.reflect_after_every_step=true solver.intervention_mode=natural"
        ;;
    all)
        echo "Launching all Exp B conditions (C0/C1/G0/M0/M1 = 150 runs)"
        echo "G1 复用 Exp A NAT，不重跑。"
        launch "c0" "airsbench/aira_mceses_qwen_single" \
            "solver.num_children=1 solver.reflect_after_every_step=false"
        launch "c1" "airsbench/aira_mceses_qwen_single" \
            "solver.num_children=1 solver.reflect_after_every_step=true solver.intervention_mode=natural"
        launch "g0" "airsbench/aira_greedy_cogito_qwen_single" \
            "solver.use_cognitive_state=false"
        launch "m0" "airsbench/aira_mceses_qwen_single" \
            "solver.num_children=5 solver.reflect_after_every_step=false"
        launch "m1" "airsbench/aira_mceses_qwen_single" \
            "solver.num_children=5 solver.reflect_after_every_step=true solver.intervention_mode=natural"
        ;;
    *)
        echo "Usage: bash scripts/run_exp_b.sh [c0|c1|g0|g1|m0|m1|d3|d4|d5|all]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "  Monitor: tail -f $LOG_DIR/*.log"
echo "  Progress: ps aux | grep dojo.main_run | wc -l"
echo "  Stop: kill \$(pgrep -f dojo.main_run)"
echo "=========================================="
