# Exp A 中期报告 — Greedy(5 drafts) + CognitiveState (NAT)

- **日期**: 2026-03-26
- **Solver**: GreedySolverConfig (num_drafts=5, use_cognitive_state=true)
- **算法**: Greedy 5-draft + reflect_op 认知状态演化 (intervention_mode=natural)
- **LLM**: Qwen-3.5-397B-A17B (via LiteLLM)
- **Interpreter**: Python (subprocess, 非 Apptainer)
- **硬件**: 4x RTX 5090 (32GB each), 30 进程分布在 4 GPU 上
- **配置**: step_limit=50, time_limit=43200s, execution_timeout=14400s
- **任务**: AIRS-Bench 10 任务, 3 seeds each = 30 runs
- **状态**: 进程已停止, 进度 52.3% (784/1500 steps), 4/30 seeds 完成

> **注意**: ABL (ablated) 条件数据因实验重启被覆盖丢失, 本报告仅含 NAT 条件结果。

---

## 1. 总览

| 指标 | 值 |
|------|------|
| 总步数 | 784 / 1500 (52.3%) |
| 完成 seed | 4 / 30 |
| 有效提交 | 295 (37.6%) |
| Buggy 提交 | 489 (62.4%) |
| 有分数任务 | 10 / 10 |
| **Avg NS** | **0.668** |
| 超 SOTA 任务 | 1 (TextualClassificationSick, NS=1.074) |
| Reflect 总次数 | 711 |

---

## 2. 逐任务结果

| Task | Category | Metric | Dir | s1 | s2 | s3 | Valid | Buggy | VR | Best | SOTA | NS | Tier |
|------|----------|--------|-----|----|----|-------|-------|-------|------|------|------|----|------|
| TextualClassificationSick | Text Classification | Accuracy | max | 18 | 25 | 28 | 32 | 39 | 45% | 0.9193 | 0.905 | **1.074** | Med |
| TimeSeriesForecastingSolar | Time Series | MAE | min | **50**✓ | 26 | **50**✓ | 88 | 38 | 70% | 812.62 | 576.35 | 0.916 | Easy |
| CvMolecularPropertyQm9 | Molecules & Proteins | MAE | min | 12 | 9 | 13 | 8 | 26 | 24% | 0.0778 | 0.021 | 0.850 | Med |
| ReadingComprehensionSquad | Question Answering | ExactMatch | max | 30 | 17 | 15 | 10 | 52 | 16% | 0.8075 | 0.858 | 0.844 | Hard |
| CoreferenceResolutionWinogrande | Text Extraction | Accuracy | max | 40 | 39 | 25 | 19 | 85 | 18% | 0.8074 | 0.854 | 0.786 | Hard |
| SentimentAnalysisYelp | Text Classification | Accuracy | max | 17 | 17 | 9 | 5 | 38 | 12% | 0.6931 | 0.778 | 0.752 | Hard |
| GraphRegressionZinc | Molecules & Proteins | MAE | min | 24 | 20 | 20 | 3 | 61 | 5% | 0.1342 | 0.017 | 0.674 | V.Hard |
| CodeRetrievalCodeXGlue | Code | MRR | max | 10 | 11 | 18 | 8 | 31 | 21% | 0.3018 | 0.611 | 0.380 | Med |
| CoreferenceResolutionWSC | Text Extraction | Accuracy | max | **50**✓ | 29 | 31 | 46 | 64 | 42% | 0.6635 | 0.962 | 0.225 | Med |
| MathQuestionAnsweringSVAMP | Math | Accuracy | max | 39 | **50**✓ | 42 | 76 | 55 | 58% | 0.3967 | 0.942 | 0.177 | Easy |

### NS 分布

```
1.074 █████████████████████  TextualClassificationSick     ★ 超 SOTA
0.916 ██████████████████     TimeSeriesForecastingSolar
0.850 █████████████████      CvMolecularPropertyQm9
0.844 ████████████████       ReadingComprehensionSquad
0.786 ███████████████        CoreferenceResolutionWinogrande
0.752 ███████████████        SentimentAnalysisYelp
0.674 █████████████          GraphRegressionZinc
0.380 ███████                CodeRetrievalCodeXGlue
0.225 ████                   CoreferenceResolutionWSC
0.177 ███                    MathQuestionAnsweringSVAMP
──────
0.668 ▓▓▓▓▓▓▓▓▓▓▓▓▓          AVG (10 tasks)
```

---

## 3. Per-Seed 详情

| Task | Seed 1 (steps/best/NS) | Seed 2 (steps/best/NS) | Seed 3 (steps/best/NS) |
|------|------------------------|------------------------|------------------------|
| TextualClassificationSick | 18 / 0.9099 / 1.024 | 25 / 0.9107 / 1.028 | 28 / 0.9193 / 1.074 |
| TimeSeriesForecastingSolar | **50✓** / 1059.66 / 0.851 | 26 / 845.85 / 0.906 | **50✓** / 812.62 / 0.916 |
| CvMolecularPropertyQm9 | 12 / 0.4206 / 0.658 | 9 / 0.0778 / 0.850 | 13 / 0.5862 / 0.620 |
| ReadingComprehensionSquad | 30 / 0.7713 / 0.756 | 17 / 0.8075 / 0.844 | 15 / - / - |
| CoreferenceResolutionWinogrande | 40 / 0.7995 / 0.755 | 39 / 0.6275 / 0.277 | 25 / 0.8074 / 0.786 |
| SentimentAnalysisYelp | 17 / 0.6931 / 0.752 | 17 / - / - | 9 / 0.6538 / 0.659 |
| GraphRegressionZinc | 24 / - / - | 20 / 0.1342 / 0.674 | 20 / - / - |
| CodeRetrievalCodeXGlue | 10 / 0.2805 / 0.348 | 11 / 0.3018 / 0.380 | 18 / 0.1144 / 0.129 |
| CoreferenceResolutionWSC | **50✓** / 0.6635 / 0.225 | 29 / 0.6346 / 0.196 | 31 / 0.6538 / 0.215 |
| MathQuestionAnsweringSVAMP | 39 / 0.3933 / 0.176 | **50✓** / 0.3800 / 0.168 | 42 / 0.3967 / 0.177 |

---

## 4. 错误分析

### 4.1 错误类型分布

| Task | Code Bug | OOM | Timeout | CUDA Assert | Total Buggy |
|------|----------|-----|---------|-------------|-------------|
| CoreferenceResolutionWinogrande | 62 | 16 | 3 | 4 | 85 |
| CoreferenceResolutionWSC | 43 | 12 | 2 | 7 | 64 |
| GraphRegressionZinc | 49 | 0 | 7 | 5 | 61 |
| MathQuestionAnsweringSVAMP | 45 | 3 | 1 | 6 | 55 |
| ReadingComprehensionSquad | 36 | 9 | 3 | 4 | 52 |
| TextualClassificationSick | 22 | 16 | 1 | 0 | 39 |
| SentimentAnalysisYelp | 20 | 9 | 6 | 3 | 38 |
| TimeSeriesForecastingSolar | 33 | 0 | 1 | 4 | 38 |
| CodeRetrievalCodeXGlue | 20 | 2 | 5 | 4 | 31 |
| CvMolecularPropertyQm9 | 19 | 1 | 5 | 1 | 26 |
| **TOTAL** | **349 (71.4%)** | **68 (13.9%)** | **34 (7.0%)** | **38 (7.8%)** | **489** |

### 4.2 错误分析要点

- **Code Bug 占 71.4%**: LLM (Qwen 3.5) 的 ML coding 基础能力是主要瓶颈。常见错误: dtype mismatch, shape error, API misuse
- **OOM 13.9%**: 30 个 ML 训练进程共享 4 块 32GB GPU (每 GPU 7-8 进程), 显存争抢严重。TextualClassification (16x OOM) 和 Winogrande (16x OOM) 最受影响
- **Timeout 7.0%**: 4h execution_timeout, 主要影响 GNN 训练 (GraphRegression 7x) 和大数据集 (SentimentAnalysis 6x)
- **CUDA Assert 7.8%**: `TORCH_USE_CUDA_DSA` 错误, 根因是 embedding index 越界或 dtype 不匹配

---

## 5. 认知状态演化统计

| Task | Reflects | Avg Confidence | Avg IQ | VR |
|------|----------|----------------|--------|------|
| MathQuestionAnsweringSVAMP | 124 | 0.601 | 0.921 | 58% |
| TimeSeriesForecastingSolar | 121 | 0.875 | 0.917 | 70% |
| CoreferenceResolutionWSC | 103 | 0.654 | 0.905 | 42% |
| CoreferenceResolutionWinogrande | 97 | 0.791 | 0.842 | 18% |
| TextualClassificationSick | 64 | 0.815 | 0.905 | 45% |
| GraphRegressionZinc | 55 | 0.796 | 0.804 | 5% |
| ReadingComprehensionSquad | 54 | 0.816 | 0.817 | 16% |
| SentimentAnalysisYelp | 37 | 0.752 | 0.825 | 12% |
| CodeRetrievalCodeXGlue | 30 | 0.715 | 0.858 | 21% |
| CvMolecularPropertyQm9 | 26 | 0.732 | 0.863 | 24% |

### 5.1 认知状态观察

- **Confidence 与 VR 不对齐**: GraphRegressionZinc 平均 confidence=0.796 但 VR 仅 5%, 表明认知状态存在过度自信 (overconfidence)
- **IQ 普遍较高 (0.80-0.92)**: reflect_op 稳定产出结构丰富的认知状态, 但高 IQ 不等于高 VR
- **VR 最高的任务 confidence 分化明显**: TimeSeries (0.875) 高且准, Math (0.601) 低且诚实 — 后者 VR=58% 说明低 confidence 驱动了更多探索

---

## 6. 任务难度分级

| Tier | Tasks | VR 范围 | Avg NS | 特征 |
|------|-------|---------|--------|------|
| **Easy** (VR≥50%) | TimeSeries, Math | 58-70% | 0.547 | 反馈信号清晰, 迭代收敛快 |
| **Medium** (20-50%) | TextualClassification, WSC, Qm9, CodeRetrieval | 21-45% | 0.632 | 需要领域知识, 中等 debug 难度 |
| **Hard** (5-20%) | Winogrande, Squad, Yelp | 12-18% | 0.794 | 环境资源瓶颈 (OOM/timeout) |
| **V.Hard** (VR<5%) | GraphRegression | 5% | 0.674 | GNN + timeout + dtype 三重困难 |

**反直觉发现**: Hard 任务的 NS 反而高于 Easy 任务。原因是 Easy 任务 (Math 0.177, WSC 0.225) 离 SOTA 很远 (SOTA accuracy > 0.9), 而 Hard 任务虽然 valid 少, 一旦出分就接近 SOTA。

---

## 7. 与历史实验对比

| 指标 | run008 (MC-ESES, 20 tasks) | Exp A NAT (Greedy+z_t, 10 tasks) |
|------|---------------------------|----------------------------------|
| Solver | MC-ESES (tree search) | Greedy (5 drafts) |
| Avg NS | 0.522 (20 tasks) | 0.668 (10 tasks) |
| VR | 62.6% | 37.6% |
| 超 SOTA 任务 | 3 | 1 |
| LLM | Qwen-3.5-397B | Qwen-3.5-397B |

> 注: 不可直接比较。run008 是 20 任务完整跑完; Exp A 仅 10 任务且进度 52%. 但 Exp A 的 10 个任务 Avg NS=0.668 高于 run008 相同 10 个任务的子集。

---

## 8. 后续计划

1. **重启 NAT 剩余**: 26/30 seeds 未完成, 需跑完全部 50 步
2. **重跑 ABL**: ABL 数据已丢失, 需完整重跑 30 runs
3. **降低资源争抢**: 考虑分批运行 (先 NAT 再 ABL) 或减少并发数, 避免 OOM 混杂因素
4. **SCR + FRZ 条件**: 待 NAT 完成后, 提取 cognitive_state.json 用于 SCR 条件
5. **Exp B**: Chain/Greedy/MCTS × ±z_t 交叉策略实验

---

## 附录: 实验配置

```yaml
# _exp/airsbench/aira_greedy_cogito_qwen_single.yaml
defaults:
  - override /interpreter: python
  - override /solver: airsbench/greedy_cogito

solver:
  step_limit: 50
  time_limit_secs: 43200
  execution_timeout: 14400
  num_drafts: 5
  use_cognitive_state: true
  reflect_after_every_step: true
  intervention_mode: natural  # NAT condition
```

```
# 启动命令
bash scripts/run_exp_a.sh nat
```

## 附录: 归一化得分公式

```
NS = (phi(score) - phi(worst)) / (phi(SOTA) - phi(worst))
phi(s) = -log10(|s - optimal|)
```

NS=0 对应最差水平, NS=1 对应 SOTA, NS>1 超越 SOTA。
