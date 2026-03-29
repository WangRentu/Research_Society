# Exp A — NAT Condition Final Results Report

**Date**: 2026-03-27
**Condition**: NAT (Natural, no intervention)
**Solver**: Greedy (5 drafts) + CognitiveState (reflect_after_every_step)
**Model**: Qwen-3.5-397B-A17B (via DashScope)
**Step budget**: 50 per seed
**Seeds**: 3 per task
**Tasks**: 10 (from AIRS-Bench)

---

## Summary (per-task, cross-seed average)

| Task | Domain | Metric Dir | s1 NS | s2 NS | s3 NS | **Avg NS** | Buggy% | Steps |
|------|--------|-----------|-------|-------|-------|-----------|--------|-------|
| SICK Accuracy | NLP | higher better | 1.075 | 1.083 | 1.109 | **1.089** | 51% | 139/150 |
| Solar MAE | TimeSeries | lower better | 0.851 | 0.922 | 0.916 | **0.897** | 30% | 148/150 |
| SQuAD EM | NLP | higher better | 0.756 | 0.844 | 0.661 | **0.754** | 73% | 128/150 |
| Qm9 MAE | Chemistry | lower better | 0.689 | 0.908 | 0.642 | **0.747** | 70% | 77/150 |
| ZINC MAE | Chemistry | lower better | 0.589 | 0.797 | 0.707 | **0.698** | 88% | 107/150 |
| Yelp Accuracy | NLP | higher better | 0.767 | 0.503 | 0.760 | **0.677** | 62% | 99/150 |
| Winogrande | NLP | higher better | 0.755 | 0.277 | 0.819 | **0.617** | 79% | 129/150 |
| Code Retrieval | Code | higher better | 0.350 | 0.380 | 0.129 | **0.286** | 69% | 74/150 |
| SuperGLUE WSC | NLP | higher better | 0.225 | 0.215 | 0.215 | **0.219** | 54% | 146/150 |
| SVAMP | Math | higher better | 0.193 | 0.168 | 0.177 | **0.180** | 45% | 148/150 |

**Overall Avg NS: 0.616** (cross-seed avg per task, then avg across 10 tasks)

---

## Progress

- Total steps: 1195/1500 (79.7%)
- Seeds completed (50 steps): 7/30
- Overall buggy rate: 717/1195 (60.0%)

---

## Detailed Per-Seed Results

### SICK Accuracy (`TextualClassificationSickAccuracy`)
- SOTA: 0.905, Optimal: 1.0, Worst: 0.1451
- Direction: higher is better
- **3 seeds 全部超越 SOTA (NS > 1.0)**

| Seed | Steps | Valid | Buggy | Buggy% | Best Metric | NS |
|------|-------|-------|-------|--------|-------------|------|
| seed_1 | 39/50 | 15 | 24 | 62% | 0.9195 | 1.075 |
| seed_2 | 50/50 | 27 | 23 | 46% | 0.9209 | 1.083 |
| seed_3 | 50/50 | 26 | 24 | 48% | 0.9252 | 1.109 |
| **Avg** | | | | | | **1.089** |

### Solar MAE (`TimeSeriesForecastingSolarWeeklyMAE`)
- SOTA: 576.35, Optimal: 0.0, Worst: 34761.99
- Direction: lower is better

| Seed | Steps | Valid | Buggy | Buggy% | Best Metric | NS |
|------|-------|-------|-------|--------|-------------|------|
| seed_1 | 50/50 | 37 | 13 | 26% | 1059.6627 | 0.851 |
| seed_2 | 48/50 | 32 | 16 | 33% | 791.9685 | 0.922 |
| seed_3 | 50/50 | 34 | 16 | 32% | 812.6223 | 0.916 |
| **Avg** | | | | | | **0.897** |

### SQuAD EM (`ReadingComprehensionSquadExactMatch`)
- SOTA: 0.858, Optimal: 1.0, Worst: 0.0
- Direction: higher is better

| Seed | Steps | Valid | Buggy | Buggy% | Best Metric | NS |
|------|-------|-------|-------|--------|-------------|------|
| seed_1 | 48/50 | 12 | 36 | 75% | 0.7713 | 0.756 |
| seed_2 | 45/50 | 18 | 27 | 60% | 0.8075 | 0.844 |
| seed_3 | 35/50 | 4 | 31 | 89% | 0.7248 | 0.661 |
| **Avg** | | | | | | **0.754** |

### Qm9 MAE (`CvMolecularPropertyPredictionQm9MeanAbsoluteError`)
- SOTA: 0.021, Optimal: 0.0, Worst: 132.633
- Direction: lower is better

| Seed | Steps | Valid | Buggy | Buggy% | Best Metric | NS |
|------|-------|-------|-------|--------|-------------|------|
| seed_1 | 32/50 | 13 | 19 | 59% | 0.3184 | 0.689 |
| seed_2 | 21/50 | 4 | 17 | 81% | 0.0470 | 0.908 |
| seed_3 | 24/50 | 6 | 18 | 75% | 0.4809 | 0.642 |
| **Avg** | | | | | | **0.747** |

### ZINC MAE (`GraphRegressionZincMae`)
- SOTA: 0.017, Optimal: 0.0, Worst: 9.700
- Direction: lower is better

| Seed | Steps | Valid | Buggy | Buggy% | Best Metric | NS |
|------|-------|-------|-------|--------|-------------|------|
| seed_1 | 40/50 | 5 | 35 | 88% | 0.2301 | 0.589 |
| seed_2 | 36/50 | 7 | 29 | 81% | 0.0616 | 0.797 |
| seed_3 | 31/50 | 1 | 30 | 97% | 0.1088 | 0.707 |
| **Avg** | | | | | | **0.698** |

### Yelp Accuracy (`SentimentAnalysisYelpReviewFullAccuracy`)
- SOTA: 0.778, Optimal: 1.0, Worst: 0.182
- Direction: higher is better

| Seed | Steps | Valid | Buggy | Buggy% | Best Metric | NS |
|------|-------|-------|-------|--------|-------------|------|
| seed_1 | 40/50 | 14 | 26 | 65% | 0.6992 | 0.767 |
| seed_2 | 25/50 | 4 | 21 | 84% | 0.5753 | 0.503 |
| seed_3 | 34/50 | 20 | 14 | 41% | 0.6965 | 0.760 |
| **Avg** | | | | | | **0.677** |

### Winogrande (`CoreferenceResolutionWinograndeAccuracy`)
- SOTA: 0.854, Optimal: 1.0, Worst: 0.466
- Direction: higher is better

| Seed | Steps | Valid | Buggy | Buggy% | Best Metric | NS |
|------|-------|-------|-------|--------|-------------|------|
| seed_1 | 48/50 | 5 | 43 | 90% | 0.7995 | 0.755 |
| seed_2 | 46/50 | 11 | 35 | 76% | 0.6275 | 0.277 |
| seed_3 | 35/50 | 11 | 24 | 69% | 0.8153 | 0.819 |
| **Avg** | | | | | | **0.617** |

### Code Retrieval (`CodeRetrievalCodeXGlueMRR`)
- SOTA: 0.6113, Optimal: 1.0, Worst: 0.0
- Direction: higher is better

| Seed | Steps | Valid | Buggy | Buggy% | Best Metric | NS |
|------|-------|-------|-------|--------|-------------|------|
| seed_1 | 22/50 | 11 | 11 | 50% | 0.2816 | 0.350 |
| seed_2 | 22/50 | 7 | 15 | 68% | 0.3018 | 0.380 |
| seed_3 | 30/50 | 5 | 25 | 83% | 0.1144 | 0.129 |
| **Avg** | | | | | | **0.286** |

### SuperGLUE WSC (`CoreferenceResolutionSuperGLUEWSCAccuracy`)
- SOTA: 0.962, Optimal: 1.0, Worst: 0.365
- Direction: higher is better

| Seed | Steps | Valid | Buggy | Buggy% | Best Metric | NS |
|------|-------|-------|-------|--------|-------------|------|
| seed_1 | 50/50 | 27 | 23 | 46% | 0.6635 | 0.225 |
| seed_2 | 48/50 | 20 | 28 | 58% | 0.6538 | 0.215 |
| seed_3 | 48/50 | 20 | 28 | 58% | 0.6538 | 0.215 |
| **Avg** | | | | | | **0.219** |

### SVAMP (`MathQuestionAnsweringSVAMPAccuracy`)
- SOTA: 0.942, Optimal: 1.0, Worst: 0.0
- Direction: higher is better

| Seed | Steps | Valid | Buggy | Buggy% | Best Metric | NS |
|------|-------|-------|-------|--------|-------------|------|
| seed_1 | 50/50 | 27 | 23 | 46% | 0.4233 | 0.193 |
| seed_2 | 50/50 | 33 | 17 | 34% | 0.3800 | 0.168 |
| seed_3 | 48/50 | 22 | 26 | 54% | 0.3967 | 0.177 |
| **Avg** | | | | | | **0.180** |

---

## Tier Analysis

| Tier | Tasks | Avg NS | Avg Buggy% |
|------|-------|--------|-----------|
| Easy | 2 (SICK, Solar) | 0.993 | 41% |
| Medium | 3 (Yelp, Winogrande, SQuAD) | 0.682 | 71% |
| Hard | 3 (Qm9, ZINC, CodeRetrieval) | 0.577 | 76% |
| Expert | 2 (WSC, SVAMP) | 0.199 | 49% |

---

## Key Observations

1. **SICK 全部超越 SOTA**：3 个 seed 的 NS 均 > 1.0（best metric 0.92+，SOTA 为 0.905），是表现最强的任务。

2. **Buggy rate 普遍偏高（60%）**：ZINC（88%）和 Winogrande（79%）尤为严重。主要原因：
   - 30 个进程共享 4 块 GPU，频繁 CUDA OOM
   - DashScope API 间歇性超时导致代码生成质量下降
   - 环境兼容性问题（PyTorch/transformers deprecated API）

3. **Easy tier 表现优异（0.993）**：SICK 和 Solar 均接近或超越 SOTA，说明认知状态在这类任务上能有效积累知识。

4. **Expert tier 表现受限（0.199）**：WSC 和 SVAMP 的 SOTA 非常高（0.962、0.942），agent 难以接近。

5. **Seed 间方差较大**：Winogrande（0.277-0.819）、SQuAD（0.661-0.844）、Yelp（0.503-0.767），反映 GPU 资源竞争导致的不确定性。

6. **进度 79.7%**：7/30 seeds 完成（50步），大部分任务在 30-48 步，受 API/GPU 瓶颈影响推进缓慢。

---

## Comparison with AIRS-Bench Leaderboard

| Agent | Avg NS | Steps | Seeds |
|-------|--------|-------|-------|
| **DLE-Agent NAT (ours, 10 tasks)** | **0.616** | **50** | **3** |
| Greedy gpt-oss-120b (AIRS-Bench best) | 0.40 | 200 | 10 |
| MCTS gpt-oss-120b | 0.37 | 200 | 10 |
| EVO gpt-oss-120b | 0.35 | 200 | 10 |

注意：我们的 0.616 是在 10 个任务子集上的得分（含 Easy-Expert），AIRS-Bench 排行榜是 20 个任务的全集平均。直接比较需谨慎。
