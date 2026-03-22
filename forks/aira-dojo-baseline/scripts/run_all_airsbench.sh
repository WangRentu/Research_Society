#!/bin/bash
# ================================================================
# AIRS-Bench Baseline 全量运行脚本
# 论文对齐配置：20 tasks × N seeds
# Greedy (AIDE) scaffold + Qwen 3.5-397B
# 24h/run, 4h/solution, 论文原版 prompt
# ================================================================

# 不用 set -e，避免子进程失败导致整个脚本退出
BASE_DIR="/data1/intern/research_society/forks/aira-dojo-baseline"
cd "$BASE_DIR"

eval "$(conda shell.bash hook)"
conda activate aira-dojo

# 确保使用 baseline 的代码
export PYTHONPATH="$BASE_DIR/src:$PYTHONPATH"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
# 使用绝对路径，防止 Hydra 改变 CWD 后找不到
LOG_DIR="$BASE_DIR/logs_baseline_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# ---------- 配置参数 ----------
SEED_START=${SEED_START:-1}
SEED_END=${SEED_END:-10}
STEP_LIMIT=${STEP_LIMIT:-200}
TIME_LIMIT=${TIME_LIMIT:-86400}
NUM_GPUS=${NUM_GPUS:-4}
NUM_SEEDS=$(( SEED_END - SEED_START + 1 ))

# 全部 20 个 airs-bench 任务
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

TOTAL_RUNS=$(( ${#TASKS[@]} * NUM_SEEDS ))

echo "=========================================="
echo "  AIRS-Bench Baseline (论文对齐)"
echo "  Tasks: ${#TASKS[@]}  Seeds: $SEED_START..$SEED_END ($NUM_SEEDS)"
echo "  Total runs: $TOTAL_RUNS"
echo "  GPUs: $NUM_GPUS  Step limit: $STEP_LIMIT"
echo "  Time limit: ${TIME_LIMIT}s ($(( TIME_LIMIT / 3600 ))h)"
echo "  Log dir: $LOG_DIR"
echo "=========================================="

run_idx=0
for task in "${TASKS[@]}"; do
    for seed in $(seq $SEED_START $SEED_END); do
        gpu=$(( run_idx % NUM_GPUS ))
        run_idx=$(( run_idx + 1 ))

        issue_id="baseline_$(echo "$task" | sed 's/\([A-Z]\)/_\L\1/g' | sed 's/^_//')"
        hydra_dir="$BASE_DIR/outputs/${TIMESTAMP}/${task}/seed_${seed}"

        echo "[$run_idx/$TOTAL_RUNS] $task seed=$seed (GPU $gpu)"

        CUDA_VISIBLE_DEVICES=$gpu nohup python -m dojo.main_run \
            +_exp=airsbench/aira_greedy_qwen_baseline \
            logger.use_wandb=False \
            task.name="$task" \
            metadata.git_issue_id="$issue_id" \
            metadata.seed="$seed" \
            solver.step_limit="$STEP_LIMIT" \
            solver.time_limit_secs="$TIME_LIMIT" \
            hydra.run.dir="$hydra_dir" \
            > "$LOG_DIR/${task}_seed${seed}.log" 2>&1 &

        # 启动间隔
        sleep 1
    done
done

echo ""
echo "=========================================="
echo "  全部 $TOTAL_RUNS 个 run 已启动"
echo "  监控: tail -f $LOG_DIR/*.log"
echo "  运行中: ps aux | grep dojo.main_run | grep -v grep | wc -l"
echo "  停止: kill \$(pgrep -f dojo.main_run)"
echo "=========================================="
