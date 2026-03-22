# Research Society — 项目背景

## 项目定位

这是一个**AI 研究智能体基准测试与开发平台**，目标是量化和比较不同 LLM 作为自主研究智能体（AI Research Agent）时的能力。

## 目录结构

```
research_society/
├── forks/
│   ├── MLGym/          # Facebook Research 的 ML 任务 Gym 环境（13 个 ML 任务，Docker 容器化执行）
│   ├── airs-bench/     # AI Research Science Benchmark（20 个跨领域任务的评测套件）
│   └── aira-dojo/      # 大规模并行智能体执行框架（支持 1000+ 并行 agent，Slurm 调度）
├── upstream/
│   └── MiroFish/       # 多智能体仿真引擎（群体智能预测框架）
├── docs/               # 文档（待完善）
├── env/                # 环境配置
└── experiments/        # 实验结果
```

## 三个核心子项目

### 1. MLGym（forks/MLGym）
- Facebook Research 开源的 ML 研究任务环境
- 包含 13 个多样化 ML 任务（CV、NLP、强化学习、博弈论）
- 使用 ReAct scaffold 驱动 agent 顺序推理
- 记录 agent 轨迹和模型交互历史

### 2. AIRS-Bench（forks/airs-bench）
- 综合评测套件，20 个任务覆盖：NLP、代码生成、数学、时间序列预测、生物化学
- 对比 agent 表现与 SOTA baseline
- 使用归一化评分跨任务比较
- 已有 14 个不同 agent 的完整 benchmark 结果

### 3. aira-dojo（forks/aira-dojo）
- 可扩展的并行 agent 执行框架
- 支持多种搜索策略：ReAct（顺序）、One-Shot、Greedy Search（并行探索）
- 集成隔离代码执行环境和 Slurm 作业调度

## 当前任务目标

1. **Benchmark AI Research Agent** — 在 20 个 ML 领域任务上量化自主研究能力
2. **对比 Agent Scaffold** — 评估不同决策架构（ReAct vs One-Shot vs Greedy Search）
3. **多模型评测** — 测试 Claude、GPT-4o、DeepSeek、Meta CWM 等模型
4. **追踪性能指标** — 归一化分数、有效提交率、Elo 排名
5. **扩展实验规模** — 支持大规模并行 agent 实验

## AIRS-Bench 归一化得分公式

Agent $a$ 在任务 $t$ 上的归一化得分（Normalized Score）：

```
NS_t^a = (φ_t(s_t^a) - φ_t(s_t^min)) / (φ_t(s_t^sota) - φ_t(s_t^min))
```

非线性变换 φ：

```
φ_t(s) = -log10(|s - s_t^opt|)
```

各符号含义：
- `s_t^a` — agent a 在任务 t 上的原始得分（test 集评估）
- `s_t^opt` — 任务 t 理论最优分（如 Accuracy 为 1.0，MAE 为 0.0），在 metadata.yaml 的 `optimal_score`
- `s_t^sota` — 任务 t 的 SOTA 得分（来自文献），在 metadata.yaml 的 `sota` 字段
- `s_t^min` — 所有 agent、所有 seed 中任务 t 的最差得分，在 metadata.yaml 的 `estimated_worst_score`

归一化后：0 = 最差 agent 水平，1 = SOTA 水平，>1 = 超越 SOTA。

参考论文：AIRS-Bench (arXiv:2602.06855)，公式定义在 forks/airs-bench/README.md 中。

## 技术栈

- Python（主要语言）
- Docker/Podman（任务隔离执行）
- Slurm（集群任务调度）
- LiteLLM（多模型统一接口）
- PyTorch / HuggingFace（ML 框架）
