#!/bin/bash
# 8 个 airs-bench 任务并行启动脚本
# 4x RTX 5090, 8 tasks → 每卡 2 个任务
#
# 任务覆盖 7 个类别:
#   Math, Sentiment, ReadingComp, TimeSeries, Graph,
#   TextualClassification, Code
#
# 数据隔离：每个任务独立 hydra.run.dir + 独立 git_issue_id
# 确保 prepared_data_dir 互不干扰

set -e
cd /data1/intern/research_society/forks/aira-cogito

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate aira-dojo

RUN_ID="run004"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="./log/${RUN_ID}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# 8 个任务 — 覆盖 7 个类别
TASKS=(
    "MathQuestionAnsweringSVAMPAccuracy"
    "SentimentAnalysisYelpReviewFullAccuracy"
    "ReadingComprehensionSquadExactMatch"
    "TimeSeriesForecastingRideshareMAE"
    "GraphRegressionZincMae"
    "TextualClassificationSickAccuracy"
    "CodeRetrievalCodeXGlueMRR"
    "QuestionAnsweringDuoRCAccuracy"
)

# GPU 分配: 4 卡分 8 任务, 每卡 2 个
GPUS=(0 1 2 3 0 1 2 3)

NUM_TASKS=${#TASKS[@]}

echo "=========================================="
echo "  RUN ID: $RUN_ID"
echo "  启动 $NUM_TASKS 个 airs-bench 任务"
echo "  日志目录: $LOG_DIR"
echo "  步数限制: 30"
echo "  Interpreter: python (子进程)"
echo "  GPU 分配: 每卡 2 个任务"
echo "=========================================="

for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    gpu="${GPUS[$i]}"

    echo "[$((i+1))/$NUM_TASKS] 启动 $task (GPU $gpu)"

    # 每个任务独立的 issue_id → 独立的 shared/logs 目录
    issue_id="${RUN_ID}_${task}"

    # 每个任务独立的 Hydra output dir → 独立的 prepared_data_dir
    hydra_dir="outputs/${RUN_ID}/${task}"

    CUDA_VISIBLE_DEVICES=$gpu PYTORCH_JIT=0 nohup python -m dojo.main_run \
        +_exp=airsbench/run_textual_classification_sick_qwen \
        logger.use_wandb=False \
        task.name="$task" \
        metadata.git_issue_id="$issue_id" \
        solver.step_limit=30 \
        solver.time_limit_secs=43200 \
        hydra.run.dir="$hydra_dir" \
        > "$LOG_DIR/${task}.log" 2>&1 &

    echo "  PID: $! → $LOG_DIR/${task}.log"
done

echo ""
echo "=========================================="
echo "  全部启动完成！"
echo "  监控: tail -f $LOG_DIR/*.log"
echo "  查看进度: ps aux | grep dojo.main_run"
echo "  停止所有: kill \$(pgrep -f dojo.main_run)"
echo "=========================================="
