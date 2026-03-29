# Baseline Run Report: Greedy Qwen-3.5-397B × AIRS-Bench 20 Tasks

> **日期**: 2026-03-23  
> **模型**: Qwen-3.5-397B-A17B (LiteLLM)  
> **Scaffold**: Greedy Search (aira-cogito baseline)  
> **数据来源**: `search_data.json` → `node.metric` (grader test score)  
> **NS 公式**: `NS = (φ(agent) - φ(worst)) / (φ(sota) - φ(worst))`, `φ(s) = -log₁₀(|s - s_opt|)`

---

## 实验配置

| 批次 | Run ID | Seeds | Step Limit | Time Limit | 状态 |
|:----:|--------|:-----:|:----------:|:----------:|:----:|
| Run 1 | 20260320_083614 | 1, 2 | 29 | 86400s | 全部完成 |
| Run 2 | 20260321_190135 | 3, 4, 5 | 200 | 86400s | 39 完成, 21 卡死已杀 |

---

## 1. Run 1 结果 (seed 1-2, step_limit=29)

| # | Task | Cat | Metric | seed_1 Raw | seed_1 NS | seed_2 Raw | seed_2 NS | Best NS |
|:-:|------|:---:|:------:|----------:|:---------:|----------:|:---------:|:-------:|
| 1 | CodeGenerationAPPSPassAt5 | Code | Pass@5 | buggy (29s) | 0.000 | buggy (29s) | 0.000 | 0.000 |
| 2 | CodeRetrievalCodeXGlueMRR | Code | MRR | buggy (29s) | 0.000 | 0.001400 | 0.001 | 0.001 |
| 3 | CoreferenceResolutionSuperGLUEWSCAccuracy | NLU | Accuracy | 0.634615 | 0.196 | 0.634615 | 0.196 | 0.196 |
| 4 | CoreferenceResolutionWinograndeAccuracy | NLU | Accuracy | 0.523283 | 0.087 | 0.761642 | 0.622 | 0.622 |
| 5 | CvMolecularPropertyPredictionQm9MeanAbsoluteError | Mol | MAE | buggy (29s) | 0.000 | buggy (29s) | 0.000 | 0.000 |
| 6 | GMolecularPropertyPredictionQm9MeanAbsoluteError | Mol | MAE | buggy (29s) | 0.000 | buggy (29s) | 0.000 | 0.000 |
| 7 | GraphRegressionZincMae | Mol | MAE | 0.147367 | 0.660 | 0.183976 | 0.625 | 0.660 |
| 8 | MathQuestionAnsweringSVAMPAccuracy | Math | Accuracy | 0.360000 | 0.157 | 0.300000 | 0.125 | 0.157 |
| 9 | QuestionAnsweringDuoRCAccuracy | QA | Accuracy | 0.206407 | 0.370 | 0.320679 | 0.619 | 0.619 |
| 10 | QuestionAnsweringEli5Rouge1 | QA | Rouge1 | 0.202388 | 0.719 | buggy (29s) | 0.000 | 0.719 |
| 11 | QuestionAnsweringFinqaAccuracy | QA | Accuracy | 0.010462 | 0.007 | 0.020924 | 0.014 | 0.014 |
| 12 | R2AbsMolecularPropertyPredictionQm9MeanAbsoluteError | Mol | MAE | buggy (29s) | 0.000 | buggy (29s) | 0.000 | 0.000 |
| 13 | ReadingComprehensionSquadExactMatch | QA | EM | 0.831788 | 0.913 | 0.764711 | 0.741 | 0.913 |
| 14 | SentimentAnalysisYelpReviewFullAccuracy | NLP | Accuracy | 0.713560 | 0.805 | 0.707200 | 0.788 | 0.805 |
| 15 | TextualClassificationSickAccuracy | NLP | Accuracy | 0.923155 | 1.097 ★ | 0.918060 | 1.067 ★ | **1.097** |
| 16 | TextualSimilaritySickSpearmanCorrelation | NLP | Spearman | 0.836544 | 0.953 | 0.814258 | 0.899 | 0.953 |
| 17 | TimeSeriesForecastingKaggleWebTrafficMASE | TS | MASE | buggy (29s) | 0.000 | buggy (29s) | 0.000 | 0.000 |
| 18 | TimeSeriesForecastingRideshareMAE | TS | MAE | 1.409451 | 0.946 | 1.445258 | 0.939 | 0.946 |
| 19 | TimeSeriesForecastingSolarWeeklyMAE | TS | MAE | 856.769174 | 0.903 | 2234.538386 | 0.669 | 0.903 |
| 20 | U0MolecularPropertyPredictionQm9MeanAbsoluteError | Mol | MAE | buggy (24s) | 0.000 | buggy (29s) | 0.000 | 0.000 |

**Run 1 平均 NS**: 0.615 (14 有效任务) | 0.430 (20 任务)

---

## 2. Run 2 结果 (seed 3-5, step_limit=200)

| # | Task | Cat | seed_3 Raw | seed_3 NS | seed_4 Raw | seed_4 NS | seed_5 Raw | seed_5 NS | Best NS |
|:-:|------|:---:|----------:|:---------:|----------:|:---------:|----------:|:---------:|:-------:|
| 1 | CodeGenerationAPPSPassAt5 | Code | 卡死 (0s) | — | 卡死 (0s) | — | buggy (43s) | 0.000 | 0.000 |
| 2 | CodeRetrievalCodeXGlueMRR | Code | 0.000300 | 0.000 | buggy (52s) | 0.000 | 0.001400 | 0.001 | 0.001 |
| 3 | CoreferenceResolutionSuperGLUEWSCAccuracy | NLU | 卡死 (36s) | — | 卡死 (0s) | — | 0.644231 | 0.206 | 0.206 |
| 4 | CoreferenceResolutionWinograndeAccuracy | NLU | 0.527230 | 0.093 | 0.455406 | -0.016 | 0.494081 | 0.041 | 0.093 |
| 5 | CvMolecularPropertyPredictionQm9MeanAbsoluteError | Mol | buggy (55s) | 0.000 | buggy (54s) | 0.000 | buggy (40s) | 0.000 | 0.000 |
| 6 | GMolecularPropertyPredictionQm9MeanAbsoluteError | Mol | buggy (36s) | 0.000 | 卡死 (0s) | — | 卡死 (0s) | — | 0.000 |
| 7 | GraphRegressionZincMae | Mol | 1.499603 | 0.294 | 1.495548 | 0.295 | buggy (22s) | 0.000 | 0.295 |
| 8 | MathQuestionAnsweringSVAMPAccuracy | Math | 0.316667 | 0.134 | 卡死 (0s) | — | 卡死 (0s) | — | 0.134 |
| 9 | QuestionAnsweringDuoRCAccuracy | QA | buggy (26s) | 0.000 | 0.186668 | 0.331 | 0.137920 | 0.237 | 0.331 |
| 10 | QuestionAnsweringEli5Rouge1 | QA | 0.165355 | 0.574 | buggy (15s) | 0.000 | buggy (20s) | 0.000 | 0.574 |
| 11 | QuestionAnsweringFinqaAccuracy | QA | 0.003487 | 0.002 | 0.019180 | 0.013 | 0.001744 | 0.001 | 0.013 |
| 12 | R2AbsMolecularPropertyPredictionQm9MeanAbsoluteError | Mol | buggy (33s) | 0.000 | 卡死 (0s) | — | buggy (93s) | 0.000 | 0.000 |
| 13 | ReadingComprehensionSquadExactMatch | QA | 0.828477 | 0.903 | 卡死 (0s) | — | 卡死 (0s) | — | 0.903 |
| 14 | SentimentAnalysisYelpReviewFullAccuracy | NLP | 卡死 (0s) | — | buggy (8s) | 0.000 | 卡死 (0s) | — | 0.000 |
| 15 | TextualClassificationSickAccuracy | NLP | 卡死 (0s) | — | 卡死 (0s) | — | 卡死 (0s) | — | 0.000 |
| 16 | TextualSimilaritySickSpearmanCorrelation | NLP | 卡死 (0s) | — | 0.869688 | 1.048 ★ | 卡死 (0s) | — | **1.048** |
| 17 | TimeSeriesForecastingKaggleWebTrafficMASE | TS | buggy (28s) | 0.000 | buggy (30s) | 0.000 | buggy (14s) | 0.000 | 0.000 |
| 18 | TimeSeriesForecastingRideshareMAE | TS | 卡死 (72s) | — | 1.237835 | 0.987 | 卡死 (32s) | — | 0.987 |
| 19 | TimeSeriesForecastingSolarWeeklyMAE | TS | 579.574829 | 0.999 | 卡死 (318s) | — | 627.379022 | 0.979 | 0.999 |
| 20 | U0MolecularPropertyPredictionQm9MeanAbsoluteError | Mol | 卡死 (0s) | — | buggy (26s) | 0.000 | buggy (32s) | 0.000 | 0.000 |

**Run 2 平均 NS**: 0.465 (12 有效任务) | 0.279 (20 任务)

---

## 3. 全 Seeds 最佳结果排名

| Rank | Task | Cat | Best Raw | Best NS | Run | SOTA | vs SOTA |
|:----:|------|:---:|--------:|:-------:|:---:|-----:|:-------:|
| 1 | TextualClassificationSickAccuracy | NLP | 0.923155 | **1.097** | R1(s1) | 0.905 | **>SOTA** |
| 2 | TextualSimilaritySickSpearmanCorrelation | NLP | 0.869688 | **1.048** | R2(s4) | 0.854 | **>SOTA** |
| 3 | TimeSeriesForecastingSolarWeeklyMAE | TS | 579.574829 | 0.999 | R2(s3) | 576.35 | ≈SOTA |
| 4 | TimeSeriesForecastingRideshareMAE | TS | 1.237835 | 0.987 | R2(s4) | 1.185 | ≈SOTA |
| 5 | ReadingComprehensionSquadExactMatch | QA | 0.831788 | 0.913 | R1(s1) | 0.858 | < |
| 6 | SentimentAnalysisYelpReviewFullAccuracy | NLP | 0.713560 | 0.805 | R1(s1) | 0.778 | < |
| 7 | QuestionAnsweringEli5Rouge1 | QA | 0.202388 | 0.719 | R1(s1) | 0.269 | < |
| 8 | GraphRegressionZincMae | Mol | 0.147367 | 0.660 | R1(s1) | 0.017 | < |
| 9 | CoreferenceResolutionWinograndeAccuracy | NLU | 0.761642 | 0.622 | R1(s2) | 0.854 | < |
| 10 | QuestionAnsweringDuoRCAccuracy | QA | 0.320679 | 0.619 | R1(s2) | 0.4648 | < |
| 11 | CoreferenceResolutionSuperGLUEWSCAccuracy | NLU | 0.644231 | 0.206 | R2(s5) | 0.962 | < |
| 12 | MathQuestionAnsweringSVAMPAccuracy | Math | 0.360000 | 0.157 | R1(s1) | 0.942 | < |
| 13 | QuestionAnsweringFinqaAccuracy | QA | 0.020924 | 0.014 | R1(s2) | 0.7803 | < |
| 14 | CodeRetrievalCodeXGlueMRR | Code | 0.001400 | 0.001 | R1(s2) | 0.6113 | < |
| 15 | CodeGenerationAPPSPassAt5 | Code | — | 0.000 | — | 0.187 | FAIL |
| 16 | CvMolecularPropertyPredictionQm9MeanAbsoluteError | Mol | — | 0.000 | — | 0.021 | FAIL |
| 17 | GMolecularPropertyPredictionQm9MeanAbsoluteError | Mol | — | 0.000 | — | 7.53 | FAIL |
| 18 | R2AbsMolecularPropertyPredictionQm9MeanAbsoluteError | Mol | — | 0.000 | — | 0.033 | FAIL |
| 19 | TimeSeriesForecastingKaggleWebTrafficMASE | TS | — | 0.000 | — | 0.622 | FAIL |
| 20 | U0MolecularPropertyPredictionQm9MeanAbsoluteError | Mol | — | 0.000 | — | 5.83 | FAIL |

**全 Seeds 最佳平均 NS**: 0.632 (14 有效任务) | **0.442** (20 任务)

---

## 4. Run 1 vs Run 2 对比 (step_limit=29 vs 200)

| Task | Cat | R1 Best NS | R2 Best NS | Δ | 更多步数有帮助？ |
|------|:---:|:----------:|:----------:|:-:|:--------------:|
| CodeGenerationAPPSPassAt5 | Code | 0.000 | 0.000 | +0.000 | — |
| CodeRetrievalCodeXGlueMRR | Code | 0.001 | 0.001 | +0.000 | ≈ 持平 |
| CoreferenceResolutionSuperGLUEWSCAccuracy | NLU | 0.196 | 0.206 | +0.009 | ≈ 持平 |
| CoreferenceResolutionWinograndeAccuracy | NLU | 0.622 | 0.093 | -0.528 | ❌ 否 (退化) |
| CvMolecularPropertyPredictionQm9MeanAbsoluteError | Mol | 0.000 | 0.000 | +0.000 | — |
| GMolecularPropertyPredictionQm9MeanAbsoluteError | Mol | 0.000 | 0.000 | +0.000 | — |
| GraphRegressionZincMae | Mol | 0.660 | 0.295 | -0.365 | ❌ 否 (退化) |
| MathQuestionAnsweringSVAMPAccuracy | Math | 0.157 | 0.134 | -0.023 | ❌ 否 (退化) |
| QuestionAnsweringDuoRCAccuracy | QA | 0.619 | 0.331 | -0.288 | ❌ 否 (退化) |
| QuestionAnsweringEli5Rouge1 | QA | 0.719 | 0.574 | -0.146 | ❌ 否 (退化) |
| QuestionAnsweringFinqaAccuracy | QA | 0.014 | 0.013 | -0.001 | ≈ 持平 |
| R2AbsMolecularPropertyPredictionQm9MeanAbsoluteError | Mol | 0.000 | 0.000 | +0.000 | — |
| ReadingComprehensionSquadExactMatch | QA | 0.913 | 0.903 | -0.010 | ≈ 持平 |
| SentimentAnalysisYelpReviewFullAccuracy | NLP | 0.805 | 0.000 | -0.805 | ❌ 否 (退化) |
| TextualClassificationSickAccuracy | NLP | 1.097 | 0.000 | -1.097 | ❌ 否 (退化) |
| TextualSimilaritySickSpearmanCorrelation | NLP | 0.953 | 1.048 | +0.095 | ✅ 是 |
| TimeSeriesForecastingKaggleWebTrafficMASE | TS | 0.000 | 0.000 | +0.000 | — |
| TimeSeriesForecastingRideshareMAE | TS | 0.946 | 0.987 | +0.040 | ✅ 是 |
| TimeSeriesForecastingSolarWeeklyMAE | TS | 0.903 | 0.999 | +0.095 | ✅ 是 |
| U0MolecularPropertyPredictionQm9MeanAbsoluteError | Mol | 0.000 | 0.000 | +0.000 | — |

**对比结论**: Run 2 更好 3 任务 | Run 1 更好 7 任务 | 持平/无数据 10 任务
- Run 1 avg NS (20t): 0.430
- Run 2 avg NS (20t): 0.279

---

## 5. AIRS-Bench 排行榜对比

| Rank | Agent | Scaffold | Avg NS (20t) |
|:----:|-------|----------|:------------:|
| **→1** | **Ours: Greedy Qwen-3.5-397B (best 5 seeds)** | **Greedy** | **0.442** |
| 2 | Greedy gpt-oss-120b | Greedy | 0.402 |
| 3 | Greedy gpt-oss-20b | Greedy | 0.400 |
| 4 | Greedy o3-mini | Greedy | 0.391 |
| 5 | Greedy GPT-4o | Greedy | 0.309 |
| 6 | MLGym CWM | ReAct | 0.302 |
| 7 | Greedy CWM | Greedy | 0.287 |
| 8 | Greedy Devstral | Greedy | 0.179 |
| 9 | MLGym GPT-4o | ReAct | 0.178 |
| 10 | One-Shot o3-mini | One-Shot | 0.171 |
| 11 | One-Shot gpt-oss-120b | One-Shot | 0.161 |
| 12 | One-Shot gpt-oss-20b | One-Shot | 0.077 |
| 13 | One-Shot GPT-4o | One-Shot | 0.057 |
| 14 | One-Shot CWM | One-Shot | 0.041 |
| 15 | One-Shot Devstral | One-Shot | 0.018 |

> **注意**: 排行榜使用 10-20 seeds 均值±标准差。我们取 best-across-seeds，variance 更大，排名仅供参考。

---

## 6. Buggy 率分析

### Run 1 (step_limit=29)

| Task | Steps | Buggy | Buggy% | Valid |
|------|:-----:|:-----:|:------:|:-----:|
| CodeGenerationAPPSPassAt5 | 58 | 58 | 100% | 0 |
| CodeRetrievalCodeXGlueMRR | 58 | 56 | 96% | 2 |
| CoreferenceResolutionSuperGLUEWSCAccuracy | 58 | 45 | 77% | 13 |
| CoreferenceResolutionWinograndeAccuracy | 33 | 19 | 57% | 14 |
| CvMolecularPropertyPredictionQm9MeanAbsoluteError | 58 | 58 | 100% | 0 |
| GMolecularPropertyPredictionQm9MeanAbsoluteError | 58 | 58 | 100% | 0 |
| GraphRegressionZincMae | 58 | 43 | 74% | 15 |
| MathQuestionAnsweringSVAMPAccuracy | 58 | 31 | 53% | 27 |
| QuestionAnsweringDuoRCAccuracy | 51 | 37 | 72% | 14 |
| QuestionAnsweringEli5Rouge1 | 58 | 56 | 96% | 2 |
| QuestionAnsweringFinqaAccuracy | 58 | 50 | 86% | 8 |
| R2AbsMolecularPropertyPredictionQm9MeanAbsoluteError | 58 | 58 | 100% | 0 |
| ReadingComprehensionSquadExactMatch | 58 | 38 | 65% | 20 |
| SentimentAnalysisYelpReviewFullAccuracy | 35 | 14 | 40% | 21 |
| TextualClassificationSickAccuracy | 58 | 26 | 44% | 32 |
| TextualSimilaritySickSpearmanCorrelation | 50 | 34 | 68% | 16 |
| TimeSeriesForecastingKaggleWebTrafficMASE | 58 | 58 | 100% | 0 |
| TimeSeriesForecastingRideshareMAE | 58 | 56 | 96% | 2 |
| TimeSeriesForecastingSolarWeeklyMAE | 58 | 55 | 94% | 3 |
| U0MolecularPropertyPredictionQm9MeanAbsoluteError | 53 | 53 | 100% | 0 |
| **Total** | **1092** | **903** | **82%** | **189** |

### Run 2 (step_limit=200)

| Task | Steps | Buggy | Buggy% | Valid |
|------|:-----:|:-----:|:------:|:-----:|
| CodeGenerationAPPSPassAt5 | 43 | 43 | 100% | 0 |
| CodeRetrievalCodeXGlueMRR | 174 | 169 | 97% | 5 |
| CoreferenceResolutionSuperGLUEWSCAccuracy | 97 | 69 | 71% | 28 |
| CoreferenceResolutionWinograndeAccuracy | 67 | 64 | 95% | 3 |
| CvMolecularPropertyPredictionQm9MeanAbsoluteError | 149 | 149 | 100% | 0 |
| GMolecularPropertyPredictionQm9MeanAbsoluteError | 36 | 36 | 100% | 0 |
| GraphRegressionZincMae | 75 | 73 | 97% | 2 |
| MathQuestionAnsweringSVAMPAccuracy | 24 | 17 | 70% | 7 |
| QuestionAnsweringDuoRCAccuracy | 72 | 70 | 97% | 2 |
| QuestionAnsweringEli5Rouge1 | 47 | 46 | 97% | 1 |
| QuestionAnsweringFinqaAccuracy | 217 | 200 | 92% | 17 |
| R2AbsMolecularPropertyPredictionQm9MeanAbsoluteError | 126 | 126 | 100% | 0 |
| ReadingComprehensionSquadExactMatch | 44 | 30 | 68% | 14 |
| SentimentAnalysisYelpReviewFullAccuracy | 8 | 8 | 100% | 0 |
| TextualSimilaritySickSpearmanCorrelation | 66 | 6 | 9% | 60 |
| TimeSeriesForecastingKaggleWebTrafficMASE | 72 | 72 | 100% | 0 |
| TimeSeriesForecastingRideshareMAE | 101 | 88 | 87% | 13 |
| TimeSeriesForecastingSolarWeeklyMAE | 398 | 196 | 49% | 202 |
| U0MolecularPropertyPredictionQm9MeanAbsoluteError | 58 | 58 | 100% | 0 |
| **Total** | **1874** | **1520** | **81%** | **354** |

---

## 7. 失败任务分析 (全 Seeds 均 buggy)

**6/20 任务无有效提交**:

- **CodeGenerationAPPSPassAt5** [Code] — 累计 ~101 步全部失败
- **CvMolecularPropertyPredictionQm9MeanAbsoluteError** [Mol] — 累计 ~207 步全部失败
- **GMolecularPropertyPredictionQm9MeanAbsoluteError** [Mol] — 累计 ~94 步全部失败
- **R2AbsMolecularPropertyPredictionQm9MeanAbsoluteError** [Mol] — 累计 ~184 步全部失败
- **TimeSeriesForecastingKaggleWebTrafficMASE** [TS] — 累计 ~130 步全部失败
- **U0MolecularPropertyPredictionQm9MeanAbsoluteError** [Mol] — 累计 ~111 步全部失败

**失败原因**:
- 4× QM9 分子预测: 需要 PyG/图神经网络，LLM 不熟悉化学数据格式
- CodeGeneration: APPS 评估器需要特殊格式的 pass@k 计算
- WebTraffic MASE: 数据量巨大 (50 万条时序)，处理超时或格式错误

---

*Report generated: 2026-03-23*