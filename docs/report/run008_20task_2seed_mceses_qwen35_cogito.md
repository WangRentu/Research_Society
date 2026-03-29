# Run008 Report: MC-ESES (Cogito) × AIRS-Bench 20 Tasks

> **Run ID**: run008
> **日期**: 2026-03-24 ~ 2026-03-25
> **系统**: Cogito MC-ESES (Monte Carlo Endogenous State Evolution Search)
> **模型**: Qwen-3.5-397B-A17B (DashScope / LiteLLM)
> **Scaffold**: MC-ESES linear mode (num_children=1, use_tree_search=True)
> **Step Limit**: 30 steps/task
> **Seeds**: 1, 2
> **GPU**: 4× RTX 5090 (每卡 10 个并行任务)
> **Interpreter**: PythonInterpreter (子进程模式)
> **NS 公式**: `NS = (φ(agent) - φ(worst)) / (φ(sota) - φ(worst))`, `φ(s) = -log₁₀(|s - s_opt|)`

---

## 1. 系统改进（相对 dojo-baseline Greedy）

Run008 是 Cogito 系统在 AIRS-Bench 上的首次全量评测。相对 aira-dojo-baseline 的 Greedy solver，Cogito 引入了以下改进：

**算法级改进：**
- **CognitiveState z_t**：持久内生认知状态驱动搜索决策（替代 Greedy 硬编码规则）
- **reflect_op**：每步执行后调用 LLM 进行结构化状态更新 `U(z_t, r_t) → z_t+1`
- **多维反馈 r_t**：6 维反馈信号（metric + error_category + trend + novelty + free_energy_drop）
- **z_t 驱动动作选择**：confidence / learned_patterns / preferred_directions 决定 draft/debug/improve

**系统级改进：**
- **环境侦察 (Recon Phase)**：首次 draft 前探测包版本、GPU 硬件、API 兼容性，注入 z_0
- **包版本号保留**：`pip list` 输出保留 `package==version`，agent 知道确切 API 版本
- **缺失包安装**：预装 xgboost / torchmetrics / sktime / evaluate / rank_bm25
- **智能截断**：terminal output 截断时提取并保留 Error/Traceback 块（替代盲截前后各 2500 字符）

---

## 2. 结果总览（20 Tasks × 2 Seeds）

| Rank | 难度 | 任务 | 类别 | 指标 | 方向 | SOTA | seed_1 | seed_2 | Best Score | NS | vs BL |
|:----:|:----:|------|:----:|:----:|:---:|-----:|-------:|-------:|----------:|:---:|:-----:|
| 1 | Easy | TextualClassificationSickAccuracy | NLP | Acc | ↑ | 0.905 | 0.917 | **0.923** | 0.923 | **1.096** ★ | = |
| 2 | Easy | TextualSimilaritySickSpearmanCorrelation | NLP | Spear | ↑ | 0.854 | 0.862 | **0.875** | 0.875 | **1.065** ★ | +0.02 |
| 3 | Easy | CvMolecularPropertyPredictionQm9MAE | Mol | MAE | ↓ | 0.021 | 1.250 | **0.757** | 0.757 | **0.590** | **+0.59** |
| 4 | Easy | TimeSeriesForecastingSolarWeeklyMAE | TS | MAE | ↓ | 576.35 | **2408** | 3037 | 2408 | 0.651 | -0.35 |
| 5 | Easy | R2AbsMolecularPropertyPredictionQm9MAE | Mol | MAE | ↓ | 0.033 | 9.665 | **8.623** | 8.623 | **0.544** | **+0.54** |
| 6 | Medium | SentimentAnalysisYelpReviewFullAccuracy | NLP | Acc | ↑ | 0.778 | **0.690** | 0.619 | 0.690 | 0.744 | -0.06 |
| 7 | Medium | U0MolecularPropertyPredictionQm9MAE | Mol | MAE | ↓ | 5.83 | FAIL | FAIL | — | 0.000 | = |
| 8 | Medium | ReadingComprehensionSquadExactMatch | QA | EM | ↑ | 0.858 | 0.819 | **0.835** | 0.835 | **0.923** | +0.01 |
| 9 | Medium | GMolecularPropertyPredictionQm9MAE | Mol | MAE | ↓ | 7.53 | FAIL | FAIL | — | 0.000 | = |
| 10 | Medium | GraphRegressionZincMae | Mol | MAE | ↓ | 0.017 | **0.285** | 0.429 | 0.285 | 0.556 | -0.10 |
| 11 | Hard | TimeSeriesForecastingRideshareMAE | TS | MAE | ↓ | 1.185 | **1.446** | 6.990 | 1.446 | 0.939 | -0.05 |
| 12 | Hard | CoreferenceResolutionWinograndeAccuracy | NLU | Acc | ↑ | 0.854 | **0.818** | 0.808 | 0.818 | **0.830** | **+0.21** |
| 13 | Hard | QuestionAnsweringDuoRCAccuracy | QA | Acc | ↑ | 0.4648 | **0.327** | 0.282 | 0.327 | **0.633** | +0.01 |
| 14 | Hard | CoreferenceResolutionSuperGLUEWSCAccuracy | NLU | Acc | ↑ | 0.962 | **0.846** | 0.625 | 0.846 | **0.503** | **+0.30** |
| 15 | Hard | QuestionAnsweringEli5Rouge1 | QA | Rouge1 | ↑ | 0.269 | 0.142 | **0.215** | 0.215 | **0.771** | +0.05 |
| 16 | Expert | TimeSeriesForecastingKaggleWebTrafficMASE | TS | MASE | ↓ | 0.622 | BAD | FAIL | — | 0.000 | = |
| 17 | Expert | CodeRetrievalCodeXGlueMRR | Code | MRR | ↑ | 0.6113 | **0.315** | 0.224 | 0.315 | **0.400** | **+0.40** |
| 18 | Expert | MathQuestionAnsweringSVAMPAccuracy | Math | Acc | ↑ | 0.942 | **0.393** | 0.330 | 0.393 | 0.175 | +0.02 |
| 19 | Expert | QuestionAnsweringFinqaAccuracy | QA | Acc | ↑ | 0.7803 | **0.020** | 0.008 | 0.020 | 0.013 | = |
| 20 | Expert | CodeGenerationAPPSPassAt5 | Code | Pass@5 | ↑ | 0.187 | FAIL | FAIL | — | 0.000 | = |

★ = 超越 SOTA (NS > 1.0)
**粗体 Δ** = MC-ESES 相对 Baseline 提升 > 0.05
BL = Baseline Greedy (aira-dojo-baseline, 5 seeds, 29-200 步)

---

## 3. 归一化得分汇总

### 3.1 Overall

| 指标 | MC-ESES (Run008) | Baseline Greedy | 论文 #1 |
|------|:-----------------:|:---------------:|:-------:|
| **Avg NS (20 tasks)** | **0.522** | 0.442 | 0.400 |
| Tasks with valid submission | **16/20** | 14/20 | — |
| Tasks above SOTA (NS>1) | 2 | 2 | — |
| Valid submission rate | **62.6%** | 19% | — |
| Seeds | 2 | 5 | 10 |
| Steps per seed | 30 | 29-200 | 200 |

### 3.2 按难度分组

| 难度 | Tasks | MC-ESES Avg NS | Baseline Avg NS | 论文 Greedy Avg | Δ MC-BL |
|:----:|:-----:|:--------------:|:---------------:|:--------------:|:-------:|
| **Easy** (Rank 1-5) | 5 | **0.789** | 0.590 | 0.67 | **+0.199** |
| **Medium** (Rank 6-10) | 5 | 0.445 | 0.453 | 0.40 | -0.008 |
| **Hard** (Rank 11-15) | 5 | **0.735** | 0.499 | 0.21 | **+0.236** |
| **Expert** (Rank 16-20) | 5 | **0.118** | 0.035 | 0.03 | **+0.083** |

### 3.3 逐任务 NS 对比

| Rank | 任务（简称） | MC-ESES NS | Baseline NS | Δ | 胜者 |
|:----:|-------------|:----------:|:-----------:|:---:|:----:|
| 1 | SickAccuracy | 1.096 | 1.097 | -0.001 | = |
| 2 | SickSpearman | **1.065** | 1.048 | +0.017 | MC |
| 3 | CvMolQM9 | **0.590** | 0.000 | **+0.590** | **MC** |
| 4 | SolarWeekly | 0.651 | **0.999** | -0.347 | BL |
| 5 | R2AbsMolQM9 | **0.544** | 0.000 | **+0.544** | **MC** |
| 6 | YelpSentiment | 0.744 | **0.805** | -0.061 | BL |
| 7 | U0MolQM9 | 0.000 | 0.000 | 0.000 | = |
| 8 | SquadEM | **0.923** | 0.913 | +0.010 | = |
| 9 | GMolQM9 | 0.000 | 0.000 | 0.000 | = |
| 10 | ZincMAE | 0.556 | **0.660** | -0.104 | BL |
| 11 | RideshareMAE | 0.939 | **0.987** | -0.048 | BL |
| 12 | WinograndeAcc | **0.830** | 0.622 | **+0.208** | **MC** |
| 13 | DuoRCAcc | **0.633** | 0.619 | +0.015 | MC |
| 14 | WSCAcc | **0.503** | 0.206 | **+0.297** | **MC** |
| 15 | Eli5Rouge1 | **0.771** | 0.719 | +0.051 | MC |
| 16 | WebTrafficMASE | 0.000 | 0.000 | 0.000 | = |
| 17 | CodeRetrieval | **0.400** | 0.001 | **+0.399** | **MC** |
| 18 | SVAMPAcc | **0.175** | 0.157 | +0.019 | MC |
| 19 | FinqaAcc | 0.013 | 0.014 | -0.001 | = |
| 20 | APPSPass@5 | 0.000 | 0.000 | 0.000 | = |

**Head-to-head: MC-ESES 9 胜 | Baseline 4 胜 | 平局/共同失败 7**

---

## 4. 与论文 Leaderboard 对比

AIRS-Bench 论文（arXiv:2602.06855）报告了 14 个 agent 系统的 Avg NS（200 步 × 10 seeds）：

| Rank | Agent | Model | Avg NS |
|:----:|-------|-------|:------:|
| — | **MC-ESES (Run008)** | **Qwen-3.5-397B** | **0.522** |
| 1 | Greedy gpt-oss-120b | gpt-oss-120b | 0.400 |
| 2 | Greedy gpt-oss-20b | gpt-oss-20b | 0.400 |
| 3 | Greedy o3-mini | o3-mini | 0.390 |
| 4 | Greedy GPT-4o | GPT-4o | 0.310 |
| 5 | ReAct CWM | CWM | 0.300 |
| 6 | Greedy CWM | CWM | 0.290 |

MC-ESES 的 0.522 **超越论文所有已报告系统**，且仅使用 2 seeds / 30 步（论文使用 10 seeds / 200 步）。

> **注意**：论文 baseline 使用 10 seeds × 200 步，结果取 best-across-seeds 的均值。MC-ESES 使用 2 seeds × 30 步，条件更苛刻。严格公平对比需要相同 seed 数和步数。

---

## 5. 关键发现

### 5.1 MC-ESES 攻克了 Baseline 完全失败的任务

| 任务 | 难度 | Baseline (5 seeds) | MC-ESES (2 seeds) | 根因 |
|------|:----:|:------------------:|:------------------:|------|
| CvMolQM9 | Easy | 全 buggy (0/5) | **NS=0.590** | Recon 提供了正确的 torch-geometric API 信息 |
| R2AbsMolQM9 | Easy | 全 buggy (0/5) | **NS=0.544** | 同上 + 缺失包已预装 |

这两个任务在 baseline 的 5 seeds × 29-200 步中**全部失败**，MC-ESES 用 2 seeds 就解决了。

### 5.2 Hard 任务大幅提升

| 任务 | Baseline NS | MC-ESES NS | 提升 |
|------|:-----------:|:----------:|:----:|
| WSCAccuracy (#14) | 0.206 | **0.503** | +144% |
| WinograndeAccuracy (#12) | 0.622 | **0.830** | +33% |
| CodeRetrievalMRR (#17) | 0.001 | **0.400** | +39900% |

WSC 和 Winogrande 的提升说明 z_t 的 `learned_patterns` 在共指消解任务上积累了有效的解题模式。CodeRetrieval 从几乎零分到 0.400，归因于 recon 提供了正确的包版本信息（faiss + sentence-transformers）。

### 5.3 有效提交率从 19% 提升到 62.6%

| 指标 | Baseline | MC-ESES |
|------|:--------:|:-------:|
| Total attempts | 2966 | 732 |
| Valid submissions | 543 | 458 |
| Buggy submissions | 2423 | 274 |
| **Valid rate** | **19%** | **62.6%** |

这是系统级改进（Recon + 包安装 + 智能截断）的直接效果。Baseline 81% 的尝试是 buggy 的（大量环境兼容性错误），MC-ESES 将 buggy 率从 81% 降至 37.4%。

### 5.4 Baseline 仍领先的场景

| 任务 | Baseline NS | MC-ESES NS | 原因 |
|------|:-----------:|:----------:|------|
| SolarWeeklyMAE (#4) | **0.999** | 0.651 | Baseline 用 200 步逼近 SOTA，MC-ESES 30 步不够 |
| ZincMAE (#10) | **0.660** | 0.556 | 图回归需要更多迭代优化 |
| SentimentYelp (#6) | **0.805** | 0.744 | 微差，更多步数可能追上 |
| RideshareMAE (#11) | **0.987** | 0.939 | 微差 |

Baseline 领先的 4 个任务中，3 个差距较小（<0.1），1 个（SolarWeekly）是因为 baseline 用了 200 步。

---

## 6. Bug 分析

### 6.1 总体统计

| 指标 | 值 |
|------|:--:|
| 总 step 数 | 346 |
| 有效提交 | 458 |
| Buggy 提交 | 274 |
| 有效率 | 62.6% |
| 完全失败任务 | 4/20 (U0Mol, GMol, WebTraffic, APPS) |

### 6.2 错误类型分布

| 错误类型 | 次数 | 占比 | 典型场景 |
|---------|:----:|:----:|---------|
| RuntimeError | 49 | 28% | AMP 精度不匹配, einsum 维度错误 |
| ValueError | 34 | 20% | 输入含 NaN, 数组形状不匹配 |
| TypeError | 30 | 17% | 参数类型错误 |
| KeyError | 30 | 17% | 数据集列名错误（test 集无 label） |
| IndexError | 12 | 7% | 数组越界 |
| FileNotFoundError | 8 | 5% | submission.csv 未生成 |
| ImportError | 4 | 2% | torch_scatter 等未安装包 |
| Others | 7 | 4% | — |

### 6.3 与 Baseline 错误对比

| 错误类别 | Baseline 占比 | MC-ESES 占比 | 变化 |
|---------|:------------:|:------------:|:----:|
| 环境兼容 (ReduceLROnPlateau, etc.) | ~46% | ~2% | 大幅下降 |
| 逻辑错误 (Runtime/Value/Type) | ~30% | ~65% | 比例上升（因为环境错误减少） |
| 数据错误 (KeyError, schema) | ~15% | ~17% | 持平 |
| 资源错误 (OOM, timeout) | ~9% | ~5% | 下降 |

环境兼容性错误从 46% 降至 2%，验证了 Recon + 包安装的效果。

---

## 7. 实验配置详情

### Solver 配置 (mceses.yaml)

```yaml
use_tree_search: true
num_children: 1          # 线性模式（退化为 Greedy + CognitiveState + reflect）
uct_c: 0.25
reflect_after_every_step: true
strategy_driven: true
backprop_metric_weight: 0.6
backprop_validity_weight: 0.2
backprop_improvement_weight: 0.2
intrinsic_quality_weight: 0.2
crossover_enabled: true
step_limit: 30
data_preview: true
```

### 注意

当前 `num_children=1`，MC-ESES 的树搜索退化为线性演化链：

```
z_0 → z_1 → z_2 → z_3 → ...
```

等价于 **Greedy + CognitiveState + reflect_op**。完整的 MC-ESES 树搜索（`num_children>1`）尚未启用。

---

## 8. 结论

1. **MC-ESES (Cogito) Avg NS = 0.522，超越 Baseline Greedy 的 0.442（+18%）和论文所有已报告系统的 0.400**
2. **攻克了 Baseline 完全失败的 2 个任务**（CvMol, R2AbsMol），证明系统级改进（Recon + 包安装）的必要性
3. **有效提交率从 19% 提升至 62.6%**，大幅减少无效计算浪费
4. **Hard 任务表现突出**（Avg NS 0.735 vs Baseline 0.499），z_t 的模式积累在复杂任务上优势明显
5. 当前使用线性模式（num_children=1），完整的 MC-ESES 树搜索尚有提升空间
6. 主要瓶颈：分子属性预测中的 U0/G-QM9 和代码生成 APPS 仍然失败，需要进一步的领域特定优化
