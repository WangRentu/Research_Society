# Research Society — 项目背景

## 项目定位

这是一个以 **Cogito**（认知状态演化搜索框架）为核心的 AI 研究智能体平台。项目目标是构建具有**持久内生认知状态**的自主研究智能体，并在标准化 benchmark 上量化其能力。

核心创新：从传统的"解空间搜索"范式转向"认知状态空间搜索"——Agent 不再仅仅搜索代码方案，而是搜索最优的认知演化路径。

## 目录结构

```
research_society/
├── forks/
│   ├── aira-cogito/    # ★ 项目主体：Cogito 认知状态演化搜索框架
│   ├── MLGym/          # Facebook Research ML 任务 Gym 环境（13 个 ML 任务）
│   └── airs-bench/     # AI Research Science Benchmark（20 个跨领域评测任务）
├── upstream/
│   └── MiroFish/       # 多智能体仿真引擎（群体智能预测框架）
├── docs/
│   ├── idea/           # 设计方案与算法命名（MC-ESES 原理、认知状态演化系统设计）
│   ├── design/         # 分析报告
│   └── chat/           # 讨论记录
├── env/                # 环境配置
└── experiments/        # 实验结果
```

## Cogito — 核心框架（forks/aira-cogito）

### 设计理念

Cogito 源自笛卡尔 "Cogito, ergo sum"（我思故我在），核心思想是让 AI Agent 具备**显式的、持久的、自反演化的认知状态**。

**范式转换**：
- 传统（aira-dojo）：`Node_t → Operator → Node_t+1`（搜索代码方案）
- Cogito：`z_t → E(z_t) → a_t → r_t → U(z_t, r_t) → z_t+1`（搜索认知演化路径）

其中 Node 从"搜索对象"降级为"认知状态的外部投射"。

### 核心算法：MC-ESES（蒙特卡洛内生状态演化搜索）

Monte Carlo Endogenous State Evolution Search，四步流程：

1. **SELECTION** — UCB 选择演化潜力最高的认知状态 z_t
   ```
   z_t = argmax_i { Q_i + c√(ln N / n_i) }
   ```

2. **EXPANSION** — 外化器 E(z_t) 生成 K 个候选行动 a_t^(k)
   - 同一认知状态可产生不同风格的解决方案

3. **EVALUATION** — 执行 a_t^(k)，获取多维反馈 r_t^(k)
   - 反馈维度：metric + error patterns + stability + novelty
   - 认知状态演化：`z_t+1 = U(z_t, r_t)`（内生状态更新）

4. **BACKPROPAGATION** — 将轨迹质量回传至祖先节点
   - 多维回传权重：metric(0.6) + validity(0.2) + improvement(0.2)

**内生演化动力学**（自由能最小化）：
```
z_t+1 = z_t - η∇F(z_t; r_t, m_t)
```
- F = 变分自由能（认知状态与反馈之间的失配度）
- 通过 LLM reflect 调用实现：输入 z_t + r_t → 输出 z_t+1

### CognitiveState 结构

```python
class CognitiveState:
    task_understanding: str          # 对任务的结构化理解
    hypotheses: List[str]            # 活跃工作假设
    learned_patterns: List[str]      # 从反馈中发现的模式
    attempt_summaries: List[str]     # 压缩的尝试历史
    preferred_directions: List[str]  # 有前景的搜索方向
    avoided_directions: List[str]    # 已排除的死胡同
    confidence: float                # 置信度 (0-1)
```

### 四种搜索策略

| 策略 | 核心思想 | 状态特性 |
|------|---------|---------|
| **Greedy** | ε-greedy 探索，{draft, improve, debug, analyze} | 基础策略，带记忆算子 |
| **MCTS** | UCT 树搜索，探索-利用平衡 | UCT = norm_q + c√(ln N / n_i) |
| **EVO** | 多岛屿遗传算法 + 迁移 | 适应度归一化，温度退火选择 |
| **MC-ESES** | ★ MC-ESES 认知状态空间搜索 | 持久认知状态 + reflect_op + 多维回传 |

MC-ESES（Monte Carlo Endogenous State Evolution Search）是核心创新，分两阶段：
- **Phase A**：Greedy + 持久 CognitiveState + reflect_op（最小验证）
- **Phase B**：完整 MC-ESES，MCTS 搜索认知状态空间而非代码空间

### 算子体系

| 算子 | 功能 | 所属策略 |
|------|------|---------|
| draft_op | 从零生成方案 | 全部 |
| improve_op | 改进已有方案 | 全部 |
| debug_op | 修复错误方案 | Greedy/MCTS/MC-ESES |
| analyze_op | 评估方案质量 | 全部 |
| crossover_op | 交叉组合多个方案 | EVO |
| reflect_op | 更新认知状态 U(z_t, r_t) → z_t+1 | MC-ESES |

### 关键数据结构

- **Node**（Journal 元素）：code + plan + execution results + metric + tree structure
- **Journal**：所有探索节点的扁平集合，支持 JSONL checkpoint
- **CognitiveState**：持久认知状态，带 intrinsic_quality() 评估
- **CognitiveStateNode**：MCTS 节点在认知状态空间中的表示

### 相对 aira-dojo 的升级

| 维度 | aira-dojo | Cogito |
|------|-----------|--------|
| 搜索空间 | 代码方案空间 | 认知状态空间 |
| 状态表示 | 隐式（散布在 prompt 中） | 显式 CognitiveState 对象 |
| 搜索策略 | Greedy / MCTS / EVO | + MC-ESES（MC-ESES） |
| 反馈信号 | 单一 metric | 多维 r_t（metric + error + stability + novelty） |
| 回传机制 | 单一指标 | 多维回传（metric 0.6 + validity 0.2 + improvement 0.2） |
| 状态因果 | 无因果链接 | 内生演化因果 z_t → z_t+1 |
| UCT 评估 | 仅代码质量 | + intrinsic quality（认知状态丰富度） |
| reflect | 无 | reflect_op：结构化 JSON 状态更新 |

### 配置系统（Hydra）

```
src/dojo/configs/
├── solver/          # 求解器配置（mceses.yaml, evo.yaml, greedy.yaml, mcts.yaml）
├── solver/client/   # LLM 后端（litellm_o3, litellm_4o, gdm 等）
├── solver/memory/   # 记忆策略（simple, sibling, debug, no_memory）
├── benchmark/       # 基准测试配置
├── task/            # 任务定义
├── interpreter/     # 执行环境（Python / Jupyter）
└── _exp/            # 完整实验配置
```

### 运行方式

```bash
# 单次实验
python -m dojo.main_run +_exp=run_example logger.use_wandb=False

# SLURM 并行实验
python -m dojo.main_runner_job_array +_exp=runner_example launcher.debug=True
```

## Benchmark 子项目

### MLGym（forks/MLGym）
- Facebook Research 开源的 ML 研究任务环境
- 13 个多样化 ML 任务（CV、NLP、强化学习、博弈论）
- 使用 ReAct scaffold 驱动 agent 顺序推理

### AIRS-Bench（forks/airs-bench）
- 综合评测套件，20 个任务覆盖：NLP、代码生成、数学、时间序列预测、生物化学
- 归一化评分跨任务比较，已有 14 个 agent 的完整 benchmark 结果

### AIRS-Bench 归一化得分公式

```
NS_t^a = (φ_t(s_t^a) - φ_t(s_t^min)) / (φ_t(s_t^sota) - φ_t(s_t^min))
```

非线性变换 φ：`φ_t(s) = -log10(|s - s_t^opt|)`

- `s_t^opt` — 理论最优分，metadata.yaml 的 `optimal_score`
- `s_t^sota` — SOTA 得分，metadata.yaml 的 `sota`
- `s_t^min` — 最差得分，metadata.yaml 的 `estimated_worst_score`

归一化后：0 = 最差水平，1 = SOTA，>1 = 超越 SOTA。参考 arXiv:2602.06855。

## 技术栈

- Python（主要语言）
- Hydra / OmegaConf（配置管理）
- Docker/Podman + Apptainer（任务隔离执行）
- Slurm（集群任务调度）
- LiteLLM（多模型统一接口）
- PyTorch / HuggingFace（ML 框架）
- Weights & Biases（实验追踪）
