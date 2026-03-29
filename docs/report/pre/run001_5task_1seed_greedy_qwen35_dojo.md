# 5-Task Benchmark 运行结果分析报告

> **实验日期**: 2026-03-19
> **日志目录**: `forks/aira-dojo/logs_5tasks_20260319_042711/`

---

## 1. 实验配置

| 项目 | 值 |
|---|---|
| 执行框架 | aira-dojo (Greedy Search) |
| LLM 模型 | Qwen 3.5-397B-A17B (DashScope API) |
| 搜索策略 | Greedy Search (5 draft + 剩余 debug) |
| 步数上限 | 30 steps/task |
| 单步超时 | 4 小时 |
| 硬件 | 128 CPU cores, 1x NVIDIA RTX 5090 (33.7GB), ~252GB RAM |
| 随机种子 | 42 |
| 总运行时长 | ~7h 16m（04:27 ~ 11:43） |

---

## 2. 结果总览

| 任务 | 有效提交 | 最终得分 | 指标 | Buggy 率 | 首次有效步 | 任务耗时 |
|---|:---:|---|---|---|---|---|
| GraphRegressionZincMae | Yes | **0.1474** | MAE (越低越好) | 96.7% (29/30) | Step 28 | ~7h 16m |
| MathQuestionAnsweringSVAMPAccuracy | **No** | 无 | Accuracy (越高越好) | 100% (30/30) | 从未成功 | ~4h 22m |
| ReadingComprehensionSquadExactMatch | Yes | **0.6535** | Exact Match (越高越好) | ~78% | Step 28 | ~74min |
| SentimentAnalysisYelpReviewFullAccuracy | Yes | **0.6803** | Accuracy (越高越好) | 86.7% (26/30) | Step 24 | ~2h 13m |
| TimeSeriesForecastingRideshareMAE | Yes | **1.7425** | MAE (越低越好) | 83.3% (25/30) | Step 8 | ~5h |

**有效提交率: 4/5 (80%)**
**整体平均 Buggy 率: ~89%**

---

## 3. 逐任务详细分析

### 3.1 GraphRegressionZincMae

- **得分**: MAE = 0.1474
- **状态**: 成功 (Step 28)
- **日志大小**: 635MB / 75,599 行

#### 执行统计

| 类别 | 数量 |
|---|---|
| 代码执行总次数 | 30 |
| 成功执行 (exit code 0) | 5 |
| 失败执行 (exit code 1) | 25 |
| 通过 Grader 的有效提交 | 1 (Step 28) |
| 非 Buggy 节点 | 1 |

#### 得分进化

| 阶段 | 步骤 | 情况 |
|---|---|---|
| Steps 1-13 | 全部崩溃 | 数据格式混淆、import 错误、tensor 维度不匹配 |
| Steps 14-16 | 部分运行但无效 | 产出全零预测（156 条），Grader 拒绝 |
| Steps 17-26 | 全部崩溃 | 数据格式问题持续，Step 24 触发 4 小时超时 |
| Step 27 | 首次有效提交 | MAE = 1.542，Grader 接受但得分差 |
| **Step 28** | **最佳提交** | 从 HuggingFace Hub 加载 ZINC 数据，训练 GIN 模型 100 epochs，**MAE = 0.1474** |
| Step 29 | 崩溃 | TypeError，数据格式问题再次出现 |

#### 关键错误

1. **数据格式错乱（主因）**: 本地数据目录提供的是 Yelp 文本数据（`['label', 'text']` 列），而非 ZINC 图数据 `['node_feat', 'edge_index', 'edge_attr', 'y', 'num_nodes']`
2. CUDA device-side assert 错误（tensor 维度不匹配传播到 GPU）
3. Grader 失败：部分运行产出全零预测（156 条 vs 期望 5,000 条）
4. 一次 4 小时超时

#### 最终方案

GIN (Graph Isomorphism Network) 模型，175,238 参数，从 HuggingFace Hub 下载 ZINC 数据集（绕过错误的本地数据），训练 100 epochs / ~2 小时，最佳验证 MAE = 0.1548。

---

### 3.2 MathQuestionAnsweringSVAMPAccuracy

- **得分**: 无（完全失败）
- **状态**: 失败
- **日志大小**: 11MB / 69,439 行

#### 执行统计

| 类别 | 数量 |
|---|---|
| 代码执行总次数 | ~58 (draft + debug 分支) |
| 通过 Grader 的有效提交 | **0** |
| Draft 步骤 | 5 (Steps 1-5) |
| Debug 步骤 | 24 (Steps 6-29) |

#### 失败原因

1. **数据 Schema 严重不匹配（根本原因）**:
   - 任务描述期望: `{ID, Body, Question, Equation, Answer, Type, question_concat}`
   - 实际数据（早期）: `['target', 'label_target']`，仅 156 条
   - 实际数据（后期）: `['label', 'text']`，650,000 条
   - 评测器期望 **300 行**预测，代码产出 156 或 50,000 行，始终无法匹配
2. **submission.csv 路径错误**: 代码保存到 `./data/submission.csv`，Grader 期望 `./submission.csv`
3. **KeyError 系列**: `'Type'`, `'Equation'`, `'question_concat'` 等列在实际数据中不存在
4. 29 步全部为无效迭代，完全浪费计算资源

#### 结论

**此任务数据准备存在严重问题**，sandbox 中的数据文件与 SVAMP 任务描述完全不一致，agent 无论如何调试都无法解决这个根本性的外部问题。

---

### 3.3 ReadingComprehensionSquadExactMatch

- **得分**: Exact Match = 0.6535
- **状态**: 成功 (Step 28)
- **日志大小**: 14MB / 77,671 行

#### 执行统计

| 类别 | 数量 |
|---|---|
| 代码执行总次数 | ~60 |
| 成功执行并通过 Grader | 6 |
| 失败/Buggy 执行 | ~54 |
| LLM 判定为 Buggy | 113 次 |
| LLM 判定为非 Buggy | 32 次 |
| Bug 率 (LLM 评估) | ~78% |

#### 得分进化

| 步骤 | 最佳节点 | 得分 |
|---|---|---|
| Steps 1-27 | 0 (初始) | 无有效提交 |
| **Step 28** | **28** | **EM = 0.6535** |
| Step 29 | 28 | 未超越 |

#### 关键错误

1. **`datasets.map()` 多进程崩溃（主因）**: `num_proc=8` 导致 RuntimeError 子进程死亡，反复出现
2. 部分迭代尝试大模型 (`deberta-v3-large`) 导致预处理/训练崩溃
3. 本地数据列结构与预期不一致（test 数据缺少 `answers` 列）
4. Token 位置计算 bug

#### 最终方案

`distilbert-base-cased-distilled-squad` 预训练模型，2 epochs，batch size 16，lr=3e-5，AMP 启用，`num_proc=1`（修复多进程问题），训练约 16 分钟。数据从 HuggingFace 加载（87,599 train / 10,570 test）。

---

### 3.4 SentimentAnalysisYelpReviewFullAccuracy

- **得分**: Accuracy = 0.6803
- **状态**: 成功 (Step 26)
- **日志大小**: 10MB / 57,281 行

#### 执行统计

| 类别 | 数量 |
|---|---|
| 代码执行总次数 | 30 |
| 成功执行 (exit code 0) | 4 |
| 失败执行 (exit code 1) | 26 |
| 失败率 | 86.7% |

#### 得分进化

| 步骤 | 最佳节点 | 得分 | 方案 |
|---|---|---|---|
| Steps 1-23 | 0 (初始) | 无有效提交 | — |
| Step 24 | 24 | 0.568 | TF-IDF + LinearSVC |
| Step 25 | 24 | 0.568 | 未改善 |
| **Step 26** | **26** | **0.680** | **DistilBERT 手动训练** |
| Step 27 | 26 | 0.680 | Step 27 得分 0.678，未超越 |
| Steps 28-29 | 26 | 0.680 | 均失败 |

#### 关键错误

1. **`ImportError: transformers` 未安装（主因）**: 占据 Steps 3-23 共 ~20 步，agent 反复尝试 `from transformers import ...` 但始终失败
2. **列名不匹配**: 代码使用 `"text"` 列，实际数据列为 `['target', 'label_target']`
3. **`accelerate>=0.26.0` 缺失**: Steps 28-29 尝试 HuggingFace Trainer 失败

#### 最终方案

DistilBERT 手动 PyTorch 训练循环（绕过 `Trainer`/`accelerate`），3 epochs，验证 Accuracy 0.686，Grader Accuracy 0.680。TF-IDF + LinearSVC 作为回退方案得到 0.568。

---

### 3.5 TimeSeriesForecastingRideshareMAE

- **得分**: MAE = 1.7425
- **状态**: 成功 (Step 8)
- **日志大小**: 74MB / 85,846 行

#### 执行统计

| 类别 | 数量 |
|---|---|
| 代码执行总次数 | 30 |
| 成功执行 (exit code 0) | 5 |
| 失败执行 (exit code 1) | 25 |
| 非 Buggy 节点 | 2 |
| Buggy 节点 | 28 |

#### 所有通过 Grader 的提交

| 步骤 | Grader MAE | 验证 MAE | 说明 |
|---|---|---|---|
| **Step 8** | **1.742** | 0.625 | 最佳结果，使用真实数据集 |
| Step 26 | 8.027 | 未知 | 有效但远差于最佳 |
| Step 28 | 63.831 | 0.000 | 在合成数据上训练，验证 MAE 无意义 |
| Step 29 | 63.909 | 0.249 | 同上，合成数据 |

#### 关键错误

1. **`ReduceLROnPlateau.__init__()` 不支持 `verbose` 参数**: 新版 PyTorch 已移除此参数，反复出现
2. **`'Dataset' object has no attribute 'get'`**: HuggingFace Dataset 对象不支持 `.get()` 方法
3. **4 小时超时**: Step 13 训练了 93 epochs 但未能产出 submission
4. **数据列名错乱**: 实际列为 `['label', 'text']`，后期方案回退到生成合成数据（导致 MAE 暴涨到 63+）

#### 最终方案

Step 8 的方案正确处理了数据列名问题，在真实 Rideshare 数据上训练，得到 MAE = 1.7425。**此后 21 步的 debug 未能改善此结果**，大量计算被浪费。

---

## 4. 核心问题诊断

### 4.1 问题严重度排序

| 优先级 | 问题 | 严重度 | 影响范围 | 说明 |
|---|---|---|---|---|
| P0 | **Sandbox 数据格式错乱** | 致命 | 5/5 任务 | 本地数据目录中的列名和内容与任务描述严重不一致，是 buggy 率居高不下的首要原因 |
| P1 | **执行环境缺少依赖** | 严重 | Yelp 任务 | `transformers`, `accelerate` 等常用 ML 库未预装，导致前 23 步完全浪费 |
| P2 | **Agent 缺乏有效回退策略** | 中等 | 全部任务 | 反复尝试同一失败路径（如持续 import 不存在的包），debug 循环效率极低 |
| P3 | **Greedy Search 探索多样性不足** | 中等 | 全部任务 | 5 次 draft 后全部进入 debug 模式，缺乏新方向探索 |

### 4.2 Buggy 率分析

```
GraphRegressionZincMae:         ████████████████████████████████████████████████░  96.7%
MathQA SVAMP:                   ██████████████████████████████████████████████████ 100.0%
ReadingComprehension SQuAD:     ███████████████████████████████████████░░░░░░░░░░░  78.0%
SentimentAnalysis Yelp:         ███████████████████████████████████████████░░░░░░░  86.7%
TimeSeries Rideshare:           █████████████████████████████████████████░░░░░░░░░  83.3%
                                                                         平均: ~89%
```

### 4.3 计算效率分析

在成功的 4 个任务中，首次有效提交平均出现在 **Step 22**（中位数 Step 26），意味着 **~75% 的搜索步数在做无效尝试**。

| 任务 | 首次有效步 | 浪费步数占比 |
|---|---|---|
| GraphRegressionZincMae | Step 28 | 90% |
| ReadingComprehension SQuAD | Step 28 | 90% |
| SentimentAnalysis Yelp | Step 24 | 77% |
| TimeSeries Rideshare | Step 8 | 23% |

---

## 5. 建议与后续行动

### 5.1 短期修复（优先）

1. **修复数据准备流程**: 确保 sandbox 中每个任务的数据文件（列名、格式、行数）与任务描述严格一致。当前数据错乱是所有问题的根源。
2. **预装常用依赖**: 在 Docker 镜像中预装 `transformers`, `accelerate`, `datasets`, `torch-geometric` 等常用包，避免 agent 在环境配置上浪费步数。
3. **重新运行 SVAMP 任务**: 修复数据后单独重跑，验证 agent 是否能产出有效结果。

### 5.2 中期优化

4. **增加 Greedy Search 的探索多样性**: 考虑增加 draft 次数（当前为 5），或引入随机重启策略，避免过早陷入单一 debug 分支。
5. **加入 Agent 回退机制**: 当连续 N 步 debug 失败时，自动尝试全新方向（如换数据加载方式、换模型架构）。
6. **对比其他搜索策略**: 与 ReAct (顺序) 和 One-Shot 策略进行对比实验。

### 5.3 长期方向

7. **多模型对比**: 在相同 5 任务上测试 Claude、GPT-4o、DeepSeek 等模型，评估模型差异。
8. **归一化评分体系**: 引入跨任务归一化得分，便于横向比较。
9. **扩展到完整 20 任务**: 修复基础设施问题后扩展到 AIRS-Bench 全部 20 个任务。

---

## 附录

### A. 日志文件清单

| 文件 | 大小 | 行数 |
|---|---|---|
| `GraphRegressionZincMae.log` | 635 MB | 75,599 |
| `MathQuestionAnsweringSVAMPAccuracy.log` | 11 MB | 69,439 |
| `ReadingComprehensionSquadExactMatch.log` | 14 MB | 77,671 |
| `SentimentAnalysisYelpReviewFullAccuracy.log` | 10 MB | 57,281 |
| `TimeSeriesForecastingRideshareMAE.log` | 74 MB | 85,846 |
| **总计** | **744 MB** | **365,836** |

### B. 成功任务的最终方案汇总

| 任务 | 模型/方法 | 训练时长 | 关键 Trick |
|---|---|---|---|
| ZINC (Graph) | GIN, 175K params, 100 epochs | ~2h | 绕过本地数据，从 HuggingFace Hub 下载 |
| SQuAD (RC) | distilbert-base-cased-distilled-squad, 2 epochs | ~16min | 设置 `num_proc=1` 修复多进程 bug |
| Yelp (Sentiment) | DistilBERT 手动 PyTorch 训练, 3 epochs | ~30min | 绕过 Trainer/accelerate，手写训练循环 |
| Rideshare (TS) | 自定义模型 | 未知 | 正确处理数据列名映射 |
