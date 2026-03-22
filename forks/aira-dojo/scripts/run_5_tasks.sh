#!/bin/bash
# 5个代表性 airs-bench 任务并行启动脚本
# 4x RTX 5090, 5 tasks → GPU 0-3 各一个, 第5个共享 GPU 0
# 基于之前 CodeRetrieval 实验经验优化参数
#
# 关键修复：每个任务使用独立的 hydra.run.dir，防止并行任务共享
# task_data/public/ 导致数据串台。

set -e
cd /data1/intern/research_society/forks/aira-dojo

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate aira-dojo

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="./logs_5tasks_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# 任务列表: (GPU, 任务名)
declare -A TASKS
TASKS[0]="MathQuestionAnsweringSVAMPAccuracy"
TASKS[1]="SentimentAnalysisYelpReviewFullAccuracy"
TASKS[2]="ReadingComprehensionSquadExactMatch"
TASKS[3]="TimeSeriesForecastingRideshareMAE"
TASKS[4]="GraphRegressionZincMae"

# GPU 分配: 4卡分5任务, 最后一个和第一个共享 GPU 0
GPUS=(0 1 2 3 0)

echo "=========================================="
echo "  启动 5 个 airs-bench 任务"
echo "  日志目录: $LOG_DIR"
echo "=========================================="

for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    gpu="${GPUS[$i]}"

    echo "[$((i+1))/5] 启动 $task (GPU $gpu)"

    # git_issue_id 决定 shared 日志目录名，每个任务用独立的 ID
    issue_id="airsbench_$(echo "$task" | sed 's/\([A-Z]\)/_\L\1/g' | sed 's/^_//')"

    # 每个任务独立的 Hydra output dir，避免并行任务共享 task_data/
    hydra_dir="outputs/${TIMESTAMP}/${task}"

    CUDA_VISIBLE_DEVICES=$gpu nohup python -m dojo.main_run \
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
