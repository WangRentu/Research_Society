# AIRS-Bench 任务难度排名与得分参考

> 来源：AIRS-Bench 论文 Table 6 (arXiv:2602.06855)，Figure 9-10
> 排名依据：March of 9s 归一化得分（所有 agent × 所有 seed 的均值），分数越高 = 越简单

## 任务难度排名（Rank 1 = 最简单，Rank 20 = 最难）

| Rank | 难度 | 任务名 | 类别 | 指标 | 方向 | SOTA | Avg NS | 最优系统 | 最优 NS |
|------|------|--------|------|------|------|------|--------|---------|--------|
| 1 | Easy | TextualClassificationSickAccuracy | Text Classification | Accuracy | ↑ | 0.905 | 0.48 | Greedy gpt-oss-120b | **>1.0** (0.93) |
| 2 | Easy | TextualSimilaritySickSpearmanCorrelation | Text Extraction | Spearman | ↑ | 0.854 | 0.48 | Greedy gpt-oss-120b | **>1.0** (0.89) |
| 3 | Easy | CvMolecularPropertyPredictionQm9MAE | Molecules & Proteins | MAE | ↓ | 0.021 | 0.34 | Greedy gpt-oss-120b | ~0.55 |
| 4 | Easy | TimeSeriesForecastingSolarWeeklyMAE | Time Series | MAE | ↓ | 576.35 | 0.34 | Greedy o3-mini | ~0.50 |
| 5 | Easy | R2AbsMolecularPropertyPredictionQm9MAE | Molecules & Proteins | MAE | ↓ | 0.033 | 0.32 | Greedy gpt-oss-120b | ~0.55 |
| 6 | Medium | SentimentAnalysisYelpReviewFullAccuracy | Text Classification | Accuracy | ↑ | 0.778 | 0.30 | Greedy o3-mini | ~0.50 |
| 7 | Medium | U0MolecularPropertyPredictionQm9MAE | Molecules & Proteins | MAE | ↓ | 5.83 | 0.29 | Greedy gpt-oss-120b | ~0.50 |
| 8 | Medium | ReadingComprehensionSquadExactMatch | Question Answering | ExactMatch | ↑ | 0.858 | 0.28 | Greedy gpt-oss-120b | ~0.50 |
| 9 | Medium | GMolecularPropertyPredictionQm9MAE | Molecules & Proteins | MAE | ↓ | 7.53 | 0.28 | Greedy gpt-oss-120b | ~0.48 |
| 10 | Medium | GraphRegressionZincMae | Molecules & Proteins | MAE | ↓ | 0.017 | 0.28 | Greedy gpt-oss-120b | ~0.50 |
| 11 | Hard | TimeSeriesForecastingRideshareMAE | Time Series | MAE | ↓ | 1.185 | 0.20 | Greedy CWM | **>1.0** (1.153) |
| 12 | Hard | CoreferenceResolutionWinograndeAccuracy | Text Extraction | Accuracy | ↑ | 0.854 | 0.19 | Greedy gpt-oss-20b | **>1.0** (0.88) |
| 13 | Hard | QuestionAnsweringDuoRCAccuracy | Question Answering | Accuracy | ↑ | 0.4648 | 0.14 | Greedy gpt-oss-120b | ~0.30 |
| 14 | Hard | CoreferenceResolutionSuperGLUEWSCAccuracy | Text Extraction | Accuracy | ↑ | 0.962 | 0.11 | Greedy gpt-oss-120b | ~0.25 |
| 15 | Hard | QuestionAnsweringEli5Rouge1 | Question Answering | Rouge1 | ↑ | 0.269 | 0.10 | ReAct CWM | ~0.25 |
| 16 | Expert | TimeSeriesForecastingKaggleWebTrafficMASE | Time Series | MASE | ↓ | 0.622 | 0.07 | Greedy gpt-oss-120b | ~0.15 |
| 17 | Expert | CodeRetrievalCodeXGlueMRR | Code | MRR | ↑ | 0.6113 | 0.04 | Greedy gpt-oss-120b | ~0.10 |
| 18 | Expert | MathQuestionAnsweringSVAMPAccuracy | Math | Accuracy | ↑ | 0.942 | 0.01 | Greedy gpt-oss-120b | ~0.05 |
| 19 | Expert | QuestionAnsweringFinqaAccuracy | Question Answering | Accuracy | ↑ | 0.7803 | 0.00 | — | ~0.00 |
| 20 | Expert | CodeGenerationAPPSPassAt5 | Code | Pass@5 | ↑ | 0.187 | 0.00 | — | ~0.00 |

## 难度分组统计

| 分组 | 任务 Rank | Overall Avg NS | Greedy Avg | ReAct Avg | One-Shot Avg |
|------|-----------|---------------|------------|-----------|-------------|
| **Easy** | 1-5 | 0.39 | 0.67 | 0.41 | 0.11 |
| **Medium** | 6-10 | 0.29 | 0.40 | 0.31 | 0.16 |
| **Hard** | 11-15 | 0.15 | 0.21 | 0.21 | 0.07 |
| **Expert** | 16-20 | 0.02 | 0.03 | 0.04 | 0.01 |

## 超越 SOTA 的案例（4 个任务）

| 任务 | SOTA | Agent 最佳 | Agent 方案 | 系统 |
|------|------|-----------|-----------|------|
| TextualClassificationSickAccuracy | 0.905 | **0.931** | RoBERTa-large + DeBERTa-v3-large 集成 + 逻辑回归 meta-learner | Greedy gpt-oss-120b |
| TextualSimilaritySickSpearmanCorrelation | 0.854 | **0.89** | RoBERTa-base + RoBERTa-large + SBERT 加权集成 | Greedy gpt-oss-120b |
| CoreferenceResolutionWinograndeAccuracy | 0.854 | **0.88** | DeBERTa-v3-large 微调 | Greedy gpt-oss-20b |
| TimeSeriesForecastingRideshareMAE | 1.185 | **1.153** | Bi-directional GRU | Greedy CWM |

## SOTA 元数据

| 任务 | SOTA Score | Optimal Score | Estimated Worst | Lower is Better |
|------|-----------|---------------|-----------------|-----------------|
| CodeGenerationAPPSPassAt5 | 0.187 | 1.0 | 0.0 | No |
| CodeRetrievalCodeXGlueMRR | 0.6113 | 1.0 | 0.0 | No |
| CoreferenceResolutionSuperGLUEWSCAccuracy | 0.962 | 1.0 | 0.365 | No |
| CoreferenceResolutionWinograndeAccuracy | 0.854 | 1.0 | 0.466 | No |
| CvMolecularPropertyPredictionQm9MAE | 0.021 | 0.0 | 132.633 | Yes |
| GMolecularPropertyPredictionQm9MAE | 7.53 | 0.0 | 11185110 | Yes |
| GraphRegressionZincMae | 0.017 | 0.0 | 9.700 | Yes |
| MathQuestionAnsweringSVAMPAccuracy | 0.942 | 1.0 | 0.0 | No |
| QuestionAnsweringDuoRCAccuracy | 0.4648 | 1.0 | 0.0 | No |
| QuestionAnsweringEli5Rouge1 | 0.269 | 1.0 | 0.00245 | No |
| QuestionAnsweringFinqaAccuracy | 0.7803 | 1.0 | 0.0 | No |
| R2AbsMolecularPropertyPredictionQm9MAE | 0.033 | 0.0 | 6536.567 | Yes |
| ReadingComprehensionSquadExactMatch | 0.858 | 1.0 | 0.0 | No |
| SentimentAnalysisYelpReviewFullAccuracy | 0.778 | 1.0 | 0.182 | No |
| TextualClassificationSickAccuracy | 0.905 | 1.0 | 0.145 | No |
| TextualSimilaritySickSpearmanCorrelation | 0.854 | 1.0 | -0.587 | No |
| TimeSeriesForecastingKaggleWebTrafficMASE | 0.622 | 0.0 | 5.03e14 | Yes |
| TimeSeriesForecastingRideshareMAE | 1.185 | 0.0 | 30.225 | Yes |
| TimeSeriesForecastingSolarWeeklyMAE | 576.35 | 0.0 | 34762.0 | Yes |
| U0MolecularPropertyPredictionQm9MAE | 5.83 | 0.0 | 24183970 | Yes |

## 归一化得分公式

```
NS_t^a = (φ(s_t^a) - φ(s_t^min)) / (φ(s_t^sota) - φ(s_t^min))

φ(s) = -log10(|s - s_opt|)    (March of 9s transform)
```

- NS = 0 → 最差 agent 水平
- NS = 1 → SOTA 水平
- NS > 1 → 超越 SOTA

## 论文最优系统排名

| Rank | Agent | Scaffold | Model | Avg NS | Elo |
|------|-------|----------|-------|--------|-----|
| 1 | Greedy gpt-oss-120b | Greedy (AIRA-dojo) | gpt-oss-120b | 0.40 | 1122 |
| 2 | Greedy gpt-oss-20b | Greedy (AIRA-dojo) | gpt-oss-20b | 0.40 | 1116 |
| 3 | Greedy o3-mini | Greedy (AIRA-dojo) | o3-mini | 0.39 | 1146 |
| 4 | Greedy GPT-4o | Greedy (AIRA-dojo) | GPT-4o | 0.31 | 1052 |
| 5 | ReAct CWM | ReAct (MLGym) | CWM | 0.30 | 1076 |
| 6 | Greedy CWM | Greedy (AIRA-dojo) | CWM | 0.29 | 1036 |
