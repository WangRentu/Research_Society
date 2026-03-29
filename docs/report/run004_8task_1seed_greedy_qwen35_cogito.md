# Run004 实验报告：8-Task × 1-Seed Greedy Qwen3.5 (aira-cogito)

> **Run ID**: run004
> **日期**: 2026-03-24
> **模型**: Qwen3.5-397B-A17B (DashScope / LiteLLM)
> **Scaffold**: Greedy Search (aira-cogito fork)
> **Step Limit**: 30 steps/task
> **Seed**: 42
> **GPU**: 4× RTX 5090 (每卡 2 任务)
> **Interpreter**: PythonInterpreter (子进程)

## 1. 结果总览

| Rank | 任务 | 类别 | 指标 | SOTA | Best Score | NS | Steps | Buggy率 | 有效？ |
|------|------|------|------|------|-----------|-----|-------|---------|--------|
| 1 | TextualClassificationSickAccuracy | Text Classification | Accuracy | 0.905 | 0.8983 | 0.969 | 30 | 28/29 (97%) | ✅ |
| 6 | SentimentAnalysisYelpReviewFullAccuracy | Text Classification | Accuracy | 0.778 | 0.7002 | 0.770 | 15 | 11/14 (79%) | ✅ |
| 8 | ReadingComprehensionSquadExactMatch | Question Answering | ExactMatch | 0.858 | — | — | 30 | 29/29 (100%) | ❌ |
| 10 | GraphRegressionZincMae | Molecules and Proteins ML | MAE | 0.017 | — | — | 30 | 29/29 (100%) | ❌ |
| 11 | TimeSeriesForecastingRideshareMAE | Time Series | MAE | 1.185 | 1.3427 | 0.961 | 26 | 23/25 (92%) | ✅ |
| 13 | QuestionAnsweringDuoRCAccuracy | Question Answering | Accuracy | 0.4648 | 0.1827 | 0.323 | 13 | 11/12 (92%) | ✅ |
| 17 | CodeRetrievalCodeXGlueMRR | Code | MRR | 0.6113 | 0.0617 | 0.067 | 3 | 1/2 (50%) | ✅ |
| 18 | MathQuestionAnsweringSVAMPAccuracy | Math | Accuracy | 0.942 | 0.2933 | 0.122 | 30 | 25/29 (86%) | ✅ |

**总览统计**:
- 有效提交率: 6/8 tasks (75%)
- 平均归一化得分 (全 8 任务，无效=0): **0.402**
- 平均归一化得分 (仅有效 6 任务): 0.535
- 总体 Buggy 率 (不含 Step 0): 157/169 (92.9%)

> **注**：按论文规则，无效提交（未产出 submission.csv）的任务 NS=0，纳入均值计算。

## 2. 与论文 Baseline 对比

| 系统 | 模型 | Avg NS | 有效提交率 |
|------|------|--------|----------|
| **本次 run004** | Qwen3.5-397B | **0.402** | 6/8 (75%) |
| Greedy gpt-oss-120b (论文最优) | gpt-oss-120b | 0.402 | ~84% |
| Greedy GPT-4o (论文) | GPT-4o | 0.309 | ~80% |
| One-Shot o3-mini (论文) | o3-mini | 0.178 | ~32% |

## 3. 错误分析

### 3.1 Top 错误类型（全局，157 次 bug）

| 错误 | 次数 | 占比 | 可修复？ |
|------|------|------|---------|
| CUDA nvrtc-builtins.so missing | 29 | 18.5% | ✅ prompt 约束（禁用 torch.compile/NVRTC） |
| truncated | 27 | 17.2% | ⚠️ agent 代码质量 |
| ReduceLROnPlateau(verbose) deprecated | 21 | 13.4% | ✅ prompt 约束 |
| datasets.map(num_proc) crash | 13 | 8.3% | ✅ prompt 约束 |
| CUDA OOM | 7 | 4.5% | ⚠️ 资源限制 / batch size 过大 |
| predict_with_generate removed | 7 | 4.5% | ✅ prompt 约束 |
| empty output | 5 | 3.2% | ⚠️ agent 代码质量 |
| ValueError: setting an array element with a sequence | 5 | 3.2% | ⚠️ agent 代码质量 |
| ValueError: The least populated class in y has only 1 member | 3 | 1.9% | ⚠️ agent 代码质量 |
| TypeError: Dataset.map() got an unexpected keyword argument 'num_workers' | 3 | 1.9% | ⚠️ agent 代码质量 |
| _pickle.PicklingError: Cannot pickle a prepared model with automatic mixed preci | 2 | 1.3% | ⚠️ agent 代码质量 |
| ValueError: too many values to unpack (expected 2) | 2 | 1.3% | ⚠️ agent 代码质量 |
| evaluate not installed | 2 | 1.3% | ✅ prompt 约束 |
| IndexError: piece id is out of range | 2 | 1.3% | ⚠️ agent 代码质量 |
| ValueError: Input contains NaN | 2 | 1.3% | ⚠️ agent 代码质量 |

### 3.2 重复错误（同一任务反复犯同一 bug）

**TextualClassificationSickAccuracy** (Rank 1):
- 14× CUDA nvrtc-builtins.so missing
- 2× truncated
- 2× _pickle.PicklingError: Cannot pickle a prepared model with automatic mixed preci

**SentimentAnalysisYelpReviewFullAccuracy** (Rank 6):
- 9× CUDA nvrtc-builtins.so missing
- 2× CUDA OOM

**ReadingComprehensionSquadExactMatch** (Rank 8):
- 12× datasets.map(num_proc) crash
- 5× truncated
- 3× CUDA OOM
- 2× evaluate not installed
- 2× TypeError: Dataset.map() got an unexpected keyword argument 'num_workers'

**GraphRegressionZincMae** (Rank 10):
- 13× truncated
- 11× ReduceLROnPlateau(verbose) deprecated

**TimeSeriesForecastingRideshareMAE** (Rank 11):
- 9× ReduceLROnPlateau(verbose) deprecated
- 1× ValueError: Input contains NaN

**QuestionAnsweringDuoRCAccuracy** (Rank 13):
- 4× CUDA nvrtc-builtins.so missing
- 2× CUDA OOM
- 2× truncated

**MathQuestionAnsweringSVAMPAccuracy** (Rank 18):
- 6× predict_with_generate removed
- 5× truncated
- 5× ValueError: setting an array element with a sequence. The requested array has an
- 3× ValueError: The least populated class in y has only 1 member, which is too few.
- 2× IndexError: piece id is out of range.

## 4. 逐任务运行记录

### TextualClassificationSickAccuracy (Rank 1, Text Classification)

- **指标**: Accuracy (↑)
- **SOTA**: 0.905
- **Best Score**: 0.8982878108438647
- **归一化得分 (NS)**: 0.969
- **Buggy 率**: 28/29 (97%)

| Step | 状态 | Metric | 错误/备注 |
|------|------|--------|----------|
| 0 | — | — | root node (空节点) |
| 1 | ❌ | — | NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implem |
| 2 | ✅ | 0.8983 | — |
| 3 | ❌ | — | (empty output) |
| 4 | ❌ | — | datasets.map(num_proc) crash |
| 5 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 6 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 7 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 8 | ❌ | — | TypeError: optimizer can only optimize Tensors, but one of the params is dict |
| 9 | ❌ | — | (truncated) |
| 10 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 11 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 12 | ❌ | — | TypeError: Dataset.map() got an unexpected keyword argument 'num_workers' |
| 13 | ❌ | — | _pickle.PicklingError: Cannot pickle a prepared model with automatic mixed preci |
| 14 | ❌ | — | TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'grad |
| 15 | ❌ | — | (truncated) |
| 16 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 17 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 18 | ❌ | — | KeyError: 'eval_accuracy' |
| 19 | ❌ | — | ValueError: Attempting to unscale FP16 gradients. |
| 20 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 21 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 22 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 23 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 24 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 25 | ❌ | — | ValueError: Found input variables with inconsistent numbers of samples: [0, 495] |
| 26 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 27 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 28 | ❌ | — | ValueError: too many values to unpack (expected 2) |
| 29 | ❌ | — | _pickle.PicklingError: Cannot pickle a prepared model with automatic mixed preci |

### SentimentAnalysisYelpReviewFullAccuracy (Rank 6, Text Classification)

- **指标**: Accuracy (↑)
- **SOTA**: 0.778
- **Best Score**: 0.70018
- **归一化得分 (NS)**: 0.770
- **Buggy 率**: 11/14 (79%)

| Step | 状态 | Metric | 错误/备注 |
|------|------|--------|----------|
| 0 | — | — | root node (空节点) |
| 1 | ✅ | 0.6799 | — |
| 2 | ✅ | 0.7002 | ★ best (RoBERTa fine-tuning) |
| 3 | ✅ | 0.5960 | hybrid RoBERTa+LightGBM |
| 4 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 5 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 6 | ❌ | — | CUDA OOM |
| 7 | ❌ | — | CUDA OOM |
| 8 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 9 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 10 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 11 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 12 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 13 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 14 | ❌ | — | CUDA nvrtc-builtins.so missing |

### ReadingComprehensionSquadExactMatch (Rank 8, Question Answering)

- **指标**: ExactMatch (↑)
- **SOTA**: 0.858
- **Best Score**: None
- **归一化得分 (NS)**: N/A
- **Buggy 率**: 29/29 (100%)

| Step | 状态 | Metric | 错误/备注 |
|------|------|--------|----------|
| 0 | — | — | root node (空节点) |
| 1 | ❌ | — | datasets.map(num_proc) crash |
| 2 | ❌ | — | datasets.map(num_proc) crash |
| 3 | ❌ | — | (empty output) |
| 4 | ❌ | — | evaluate not installed |
| 5 | ❌ | — | (truncated) |
| 6 | ❌ | — | NameError: name 'tqdm' is not defined |
| 7 | ❌ | — | datasets.map(num_proc) crash |
| 8 | ❌ | — | (truncated) |
| 9 | ❌ | — | datasets.map(num_proc) crash |
| 10 | ❌ | — | datasets.map(num_proc) crash |
| 11 | ❌ | — | (truncated) |
| 12 | ❌ | — | Exception: Truncation error: Sequence to truncate too short to respect the provi |
| 13 | ❌ | — | datasets.map(num_proc) crash |
| 14 | ❌ | — | (truncated) |
| 15 | ❌ | — | datasets.map(num_proc) crash |
| 16 | ❌ | — | evaluate not installed |
| 17 | ❌ | — | (truncated) |
| 18 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 19 | ❌ | — | TypeError: Dataset.map() got an unexpected keyword argument 'num_workers' |
| 20 | ❌ | — | CUDA OOM |
| 21 | ❌ | — | AttributeError: 'dict' object has no attribute 'sequence_ids' |
| 22 | ❌ | — | datasets.map(num_proc) crash |
| 23 | ❌ | — | CUDA OOM |
| 24 | ❌ | — | datasets.map(num_proc) crash |
| 25 | ❌ | — | CUDA OOM |
| 26 | ❌ | — | TypeError: Dataset.map() got an unexpected keyword argument 'num_workers' |
| 27 | ❌ | — | datasets.map(num_proc) crash |
| 28 | ❌ | — | datasets.map(num_proc) crash |
| 29 | ❌ | — | datasets.map(num_proc) crash |

### GraphRegressionZincMae (Rank 10, Molecules and Proteins ML)

- **指标**: MAE (↓)
- **SOTA**: 0.017
- **Best Score**: None
- **归一化得分 (NS)**: N/A
- **Buggy 率**: 29/29 (100%)

| Step | 状态 | Metric | 错误/备注 |
|------|------|--------|----------|
| 0 | — | — | root node (空节点) |
| 1 | ❌ | — | (truncated) |
| 2 | ❌ | — | (truncated) |
| 3 | ❌ | — | ReduceLROnPlateau(verbose) deprecated |
| 4 | ❌ | — | (empty output) |
| 5 | ❌ | — | (truncated) |
| 6 | ❌ | — | ReduceLROnPlateau(verbose) deprecated |
| 7 | ❌ | — | ReduceLROnPlateau(verbose) deprecated |
| 8 | ❌ | — | (truncated) |
| 9 | ❌ | — | (truncated) |
| 10 | ❌ | — | RuntimeError: mat1 and mat2 shapes cannot be multiplied (2994x1 and 21x128) |
| 11 | ❌ | — | TypeError: DataLoader.__init__() got an unexpected keyword argument 'follow_batc |
| 12 | ❌ | — | (truncated) |
| 13 | ❌ | — | (truncated) |
| 14 | ❌ | — | (truncated) |
| 15 | ❌ | — | (truncated) |
| 16 | ❌ | — | ReduceLROnPlateau(verbose) deprecated |
| 17 | ❌ | — | ReduceLROnPlateau(verbose) deprecated |
| 18 | ❌ | — | (truncated) |
| 19 | ❌ | — | (truncated) |
| 20 | ❌ | — | ReduceLROnPlateau(verbose) deprecated |
| 21 | ❌ | — | ReduceLROnPlateau(verbose) deprecated |
| 22 | ❌ | — | (truncated) |
| 23 | ❌ | — | ReduceLROnPlateau(verbose) deprecated |
| 24 | ❌ | — | (truncated) |
| 25 | ❌ | — | ReduceLROnPlateau(verbose) deprecated |
| 26 | ❌ | — | RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 4 |
| 27 | ❌ | — | ReduceLROnPlateau(verbose) deprecated |
| 28 | ❌ | — | ReduceLROnPlateau(verbose) deprecated |
| 29 | ❌ | — | AttributeError: 'BatchNorm' object has no attribute 'weight' |

### TimeSeriesForecastingRideshareMAE (Rank 11, Time Series)

- **指标**: MAE (↓)
- **SOTA**: 1.185
- **Best Score**: 1.3427457215276097
- **归一化得分 (NS)**: 0.961
- **Buggy 率**: 23/25 (92%)

| Step | 状态 | Metric | 错误/备注 |
|------|------|--------|----------|
| 0 | — | — | root node (空节点) |
| 1 | ❌ | — | ReduceLROnPlateau(verbose) deprecated |
| 2 | ❌ | — | ValueError: operands could not be broadcast together with shapes (79488,168) (45 |
| 3 | ❌ | — | (empty output) |
| 4 | ❌ | — | RuntimeError: Tensors must have same number of dimensions: got 2 and 3 |
| 5 | ❌ | — | ValueError: Input data must be 2 dimensional and non empty. |
| 6 | ❌ | — | ReduceLROnPlateau(verbose) deprecated |
| 7 | ❌ | — | ValueError: operands could not be broadcast together with shapes (460,168) (1844 |
| 8 | ❌ | — | ReduceLROnPlateau(verbose) deprecated |
| 9 | ❌ | — | ReduceLROnPlateau(verbose) deprecated |
| 10 | ❌ | — | ValueError: Input contains NaN. |
| 11 | ✅ | 1.3427 | ★ best |
| 12 | ❌ | — | ValueError: num_samples should be a positive integer value, but got num_samples= |
| 13 | ❌ | — | ReduceLROnPlateau(verbose) deprecated |
| 14 | ❌ | — | ValueError: operands could not be broadcast together with shapes (2304,168,1) (2 |
| 15 | ❌ | — | ReduceLROnPlateau(verbose) deprecated |
| 16 | ❌ | — | ReduceLROnPlateau(verbose) deprecated |
| 17 | ❌ | — | RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x168 and 1x128) |
| 18 | ❌ | — | RuntimeError: The size of tensor a (170) must match the size of tensor b (168) a |
| 19 | ❌ | — | ReduceLROnPlateau(verbose) deprecated |
| 20 | ❌ | — | RuntimeError: The size of tensor a (174) must match the size of tensor b (170) a |
| 21 | ❌ | — | ValueError: too many values to unpack (expected 2) |
| 22 | ❌ | — | RuntimeError: tensor size mismatch in TCNBlock residual connection |
| 23 | ❌ | — | ReduceLROnPlateau(verbose) deprecated |
| 24 | ✅ | 23.994 | TCN+Attention，远不及 best |
| 25 | ❌ | — | ValueError: Input contains NaN |

### QuestionAnsweringDuoRCAccuracy (Rank 13, Question Answering)

- **指标**: Accuracy (↑)
- **SOTA**: 0.4648
- **Best Score**: 0.18270
- **归一化得分 (NS)**: 0.323
- **Buggy 率**: 11/12 (92%)

| Step | 状态 | Metric | 错误/备注 |
|------|------|--------|----------|
| 0 | — | — | root node (空节点) |
| 1 | ❌ | — | (truncated) |
| 2 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 3 | ❌ | — | RuntimeError: 0D or 1D target tensor expected, multi-target not supported |
| 4 | ❌ | — | predict_with_generate removed |
| 5 | ❌ | — | (truncated) |
| 6 | ✅ | 0.1827 | ★ RoBERTa classifier + T5 generator pipeline |
| 7 | ❌ | — | CUDA OOM |
| 8 | ❌ | — | CUDA OOM (DeBERTa-v3-large batch=32) |
| 9 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 10 | ❌ | — | CUDA nvrtc-builtins.so missing |
| 11 | ❌ | — | CUDA nvrtc-builtins.so missing (CPU fallback 也失败) |
| 12 | ❌ | — | CUDA nvrtc-builtins.so missing |

### CodeRetrievalCodeXGlueMRR (Rank 17, Code)

- **指标**: MRR (↑)
- **SOTA**: 0.6113
- **Best Score**: 0.0617
- **归一化得分 (NS)**: 0.067
- **Buggy 率**: 1/2 (50%)

| Step | 状态 | Metric | 错误/备注 |
|------|------|--------|----------|
| 0 | — | — | root node (空节点) |
| 1 | ✅ | 0.0617 | — |
| 2 | ❌ | — | NameError: name 'torch' is not defined |

### MathQuestionAnsweringSVAMPAccuracy (Rank 18, Math)

- **指标**: Accuracy (↑)
- **SOTA**: 0.942
- **Best Score**: 0.29333333333333333
- **归一化得分 (NS)**: 0.122
- **Buggy 率**: 25/29 (86%)

| Step | 状态 | Metric | 错误/备注 |
|------|------|--------|----------|
| 0 | — | — | root node (空节点) |
| 1 | ❌ | — | (truncated) |
| 2 | ❌ | — | ValueError: The least populated class in y has only 1 member, which is too few.  |
| 3 | ❌ | — | ValueError: The least populated class in y has only 1 member, which is too few.  |
| 4 | ❌ | — | (truncated) |
| 5 | ❌ | — | (empty output) |
| 6 | ❌ | — | (truncated) |
| 7 | ❌ | — | ValueError: setting an array element with a sequence. The requested array has an |
| 8 | ❌ | — | (truncated) |
| 9 | ❌ | — | KeyError: 'Common-Divison' |
| 10 | ✅ | 0.2000 | — |
| 11 | ❌ | — | AttributeError: 'tuple' object has no attribute 'shape' |
| 12 | ✅ | 0.2700 | — |
| 13 | ❌ | — | predict_with_generate removed |
| 14 | ❌ | — | ValueError: setting an array element with a sequence. The requested array has an |
| 15 | ❌ | — | (truncated) |
| 16 | ❌ | — | ValueError: The least populated class in y has only 1 member, which is too few.  |
| 17 | ❌ | — | IndexError: piece id is out of range. |
| 18 | ❌ | — | ReduceLROnPlateau(verbose) deprecated |
| 19 | ❌ | — | IndexError: piece id is out of range. |
| 20 | ❌ | — | predict_with_generate removed |
| 21 | ❌ | — | predict_with_generate removed |
| 22 | ✅ | 0.2933 | — |
| 23 | ✅ | 0.0333 | — |
| 24 | ❌ | — | ValueError: setting an array element with a sequence. The requested array has an |
| 25 | ❌ | — | predict_with_generate removed |
| 26 | ❌ | — | predict_with_generate removed |
| 27 | ❌ | — | ValueError: setting an array element with a sequence. The requested array has an |
| 28 | ❌ | — | predict_with_generate removed |
| 29 | ❌ | — | ValueError: setting an array element with a sequence. The requested array has an |

## 5. 关键发现

### 5.1 环境兼容性问题（已在 prompt 中修复，本次运行未生效）

- 共 72/157 次 bug (46%) 由已知环境约束导致
- `nvrtc-builtins.so.13.0` CUDA toolkit 版本不匹配（29 次，最大单因）—— Agent 尝试 AMP/torch.compile 时触发
- `ReduceLROnPlateau(verbose=)` 已在 PyTorch ≥2.4 移除（21 次）
- `datasets.map(num_proc=N)` 在沙箱环境中多进程崩溃（13 次）
- `TrainingArguments(predict_with_generate)` 已在 transformers ≥4.46 移除（7 次）
- `evaluate` 包未安装（2 次）
- 已在 prompt CONSTRAINTS 中加入约束，下次运行预计消除

### 5.2 Agent 缺乏跨步记忆

- 大量 bug 是同一任务反复犯同一错误（如 SentimentAnalysis 9 次 nvrtc、DuoRC 4 次 nvrtc）
- Greedy solver 没有跨步记忆，每个新 draft/debug 不知道前面踩过哪些坑
- 这是 MC-ESES CognitiveState + reflect_op 要解决的核心问题

### 5.3 Avg NS 追平论文最优

- 本次 run004 Avg NS = **0.402**，与论文最优 Greedy gpt-oss-120b (0.402) 持平
- 仅用 8 个任务子集，且有 3 个任务未跑完（SentimentAnalysis 15/30 步、DuoRC 13/30 步、CodeRetrieval 3/30 步）
- 若这 3 个任务跑满 30 步，NS 有进一步提升空间
