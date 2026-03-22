# AIRS-Bench 5-Task Round 2 实验报告

> **实验标识**: 5task-run002
> **执行时间**: 2026-03-19 18:19 ~ 2026-03-20 01:18
> **模型**: Qwen3.5-397B-A17B (via DashScope / LiteLLM)
> **Solver**: Greedy Search (debug_prob=1.0, max_debug_depth=20, num_drafts=5)
> **Step Limit**: 30 steps / task
> **Execution Timeout**: 14400s (4h) / step
> **Seed**: 42

---

## 1. 任务概览

| # | 任务全称 | 简称 | 领域 | 指标 | 方向 | 论文难度排名 | 难度带 |
|---|---------|------|------|------|------|-------------|--------|
| 1 | GraphRegressionZincMae | ZINC | 分子图回归 | MAE | ↓越低越好 | Rank 10 / 20 | Medium |
| 2 | MathQuestionAnsweringSVAMPAccuracy | MathQA | 数学问答 | Accuracy | ↑越高越好 | Rank 18 / 20 | Expert |
| 3 | ReadingComprehensionSquadExactMatch | SQuAD | 阅读理解 | Exact Match | ↑越高越好 | Rank 8 / 20 | Medium |
| 4 | SentimentAnalysisYelpReviewFullAccuracy | Yelp | 情感分析 | Accuracy | ↑越高越好 | Rank 6 / 20 | Medium |
| 5 | TimeSeriesForecastingRideshareMAE | Rideshare | 时序预测 | MAE | ↓越低越好 | Rank 11 / 20 | Hard |

> **难度排名来源**: AIRS-Bench 论文 Figure 9 / Table 6，按所有 agent 平均归一化分从高到低排列，Rank 1 最容易、Rank 20 最难。
> **难度带划分**: Easy (1-5), Medium (6-10), Hard (11-15), Expert (16-20)

### SOTA 参考

| 任务 | SOTA 分数 | SOTA 方法 | 年份 | 发表 | Est. Worst | Optimal |
|------|----------|----------|------|------|-----------|---------|
| ZINC | 0.017 | End-to-end attention on graphs | 2024 | Nature Comm. | 9.6999 | 0.0 |
| MathQA | 0.942 | Deeply Understanding Problems | 2026 | Front. CS | 0.0 | 1.0 |
| SQuAD | 0.858 | SplaXBERT | 2024 | Preprint | 0.0 | 1.0 |
| Yelp | 0.778 | SplitEE | 2023 | AIMLSystems | 0.1821 | 1.0 |
| Rideshare | 1.185 | Prototype-Guided Normalization | 2025 | Preprint | 30.2249 | 0.0 |

---

## 2. Round 2 执行结果

| 任务 | Steps | Good | Buggy | BugRate | 最优分 | 最优步 | 耗时 |
|------|-------|------|-------|---------|--------|--------|------|
| ZINC | 30/30 | 5 | 25 | 83% | **0.1048** | Step 20 | 7.0h |
| MathQA | 30/30 | 15 | 15 | **50%** | **0.4133** | Step 22 | 1.3h |
| SQuAD | 16/30* | 1 | 15 | 94% | **0.3411** | Step 14 | 6.3h |
| Yelp | 27/30* | 3 | 24 | 89% | **0.7004** | Step 17 | 6.5h |
| Rideshare | 30/30 | 9 | 21 | **70%** | **1.1975** | Step 27 | 6.0h |

> *SQuAD 和 Yelp 因进程卡住未跑完 30 步，已手动终止。

---

## 3. March of 9's 归一化得分

### 公式

$$\varphi(s) = -\log_{10}(|s - s_{\text{optimal}}|)$$

$$\text{Normalized Score} = \frac{\varphi(s_{\text{agent}}) - \varphi(s_{\text{worst}})}{\varphi(s_{\text{SOTA}}) - \varphi(s_{\text{worst}})}$$

### R2 归一化计算

| 任务 | 原始分 | phi(agent) | phi(worst) | phi(SOTA) | **归一化分** |
|------|--------|-----------|-----------|----------|------------|
| ZINC | 0.1048 | 0.9796 | -0.9868 | 1.7696 | **0.7134** |
| MathQA | 0.4133 | 0.2316 | 0.0000 | 1.2366 | **0.1873** |
| SQuAD | 0.3411 | 0.1812 | 0.0000 | 0.8477 | **0.2137** |
| Yelp | 0.7004 | 0.5235 | 0.0872 | 0.6536 | **0.7703** |
| Rideshare | 1.1975 | -0.0783 | -1.4804 | -0.0738 | **0.9968** |
| **R2 平均** | | | | | **0.5763** |

---

## 4. R1 vs R2 对比

### 4.1 原始分数对比

| 任务 | R1 Best | R2 Best | 变化 | 取优 |
|------|---------|---------|------|------|
| ZINC (↓) | 0.1474 | **0.1048** | ↓0.0426 改善 | R2 |
| MathQA (↑) | N/A (100% buggy) | **0.4133** | 从零到有 | R2 |
| SQuAD (↑) | **0.6535** | 0.3411 | R2 更差（未跑完） | R1 |
| Yelp (↑) | 0.6803 | **0.7004** | ↑0.0201 改善 | R2 |
| Rideshare (↓) | 1.7425 | **1.1975** | ↓0.5450 改善 | R2 |

### 4.2 归一化分数对比

| 任务 | R1 归一化 | R2 归一化 | Best 归一化 | 变化 |
|------|----------|----------|-----------|------|
| ZINC | 0.6597 | **0.7134** | 0.7134 | +0.054 |
| MathQA | 0.0000 | **0.1873** | 0.1873 | +0.187 |
| SQuAD | **0.5431** | 0.2137 | 0.5431 | — |
| Yelp | 0.7203 | **0.7703** | 0.7703 | +0.050 |
| Rideshare | 0.8810 | **0.9968** | 0.9968 | +0.116 |
| **平均** | **0.5608** | **0.5763** | **0.6422** | +0.081 |

### 4.3 Buggy Rate 对比

| 任务 | R1 BugRate | R2 BugRate | 变化 |
|------|-----------|-----------|------|
| ZINC | 97% (29/30) | 83% (25/30) | ↓14pp |
| MathQA | 100% (30/30) | **50%** (15/30) | ↓50pp |
| SQuAD | 97% (29/30) | 94% (15/16) | ↓3pp |
| Yelp | 90% (27/30) | 89% (24/27) | ↓1pp |
| Rideshare | 93% (28/30) | **70%** (21/30) | ↓23pp |
| **平均** | **95%** | **77%** | **↓18pp** |

### 4.4 小结

- **4/5 任务原始分提升**（SQuAD 因未跑完导致 R2 更差，R1 已有优秀解 0.6535）
- **MathQA 是最大突破**：R1 全军覆没 → R2 50% 成功率 + 15 个有效步
- **Rideshare 接近 SOTA**：1.1975 vs SOTA 1.185，差距仅 0.0125
- **Buggy Rate 全面下降**：平均 95% → 77%，MathQA 改善最大 (↓50pp)
- **取两轮最优 (Best-of-Both) 平均归一化分 = 0.6422**

---

## 5. 与 AIRS-Bench 论文基准对比

### 5.1 论文排行榜 (20 tasks × 10 seeds)

| 排名 | Agent | Scaffold | Model | ANS (20-task avg) |
|------|-------|----------|-------|-------------------|
| 1 | Greedy gpt-oss-120b | Greedy | Meta gpt-oss-120b | **0.402** |
| 2 | Greedy gpt-oss-20b | Greedy | Meta gpt-oss-20b | 0.400 |
| 3 | Greedy o3-mini | Greedy | OpenAI o3-mini | 0.391 |
| 4 | Greedy GPT-4o | Greedy | OpenAI GPT-4o | 0.309 |
| 5 | ReAct CWM | ReAct | Meta CWM | 0.302 |
| 6 | Greedy CWM | Greedy | Meta CWM | 0.287 |
| 7 | Greedy Devstral | Greedy | Mistral Devstral | 0.179 |
| 8 | ReAct GPT-4o | ReAct | OpenAI GPT-4o | 0.163 |

> **最强策略: Greedy scaffold**，Top 4 全部是 Greedy。

### 5.2 论文 Table 6: 5 任务全 agent 平均分

| 任务 | Rank | 全 agent 归一化均值 | 我们 Best | 倍数 |
|------|------|-------------------|----------|------|
| Yelp | 6 | 0.30 | 0.7703 | **2.6x** |
| SQuAD | 8 | 0.28 | 0.5431 | **1.9x** |
| ZINC | 10 | 0.28 | 0.7134 | **2.5x** |
| Rideshare | 11 | 0.20 | 0.9968 | **5.0x** |
| MathQA | 18 | 0.01 | 0.1873 | **18.7x** |
| **5-task 均值** | | **0.214** | **0.6422** | **3.0x** |

### 5.3 与最强 Greedy Scaffold 估算对比

基于论文 Figure 13 难度带 Greedy scaffold 平均分估算：

| 任务 | 难度带 | Greedy 带均值 | 我们 Best | 对比 |
|------|-------|-------------|----------|------|
| Yelp | Medium | ~0.70 | 0.7703 | ↑高于 Greedy 均值 |
| SQuAD | Medium | ~0.70 | 0.5431 | ↓低于 Greedy 均值 |
| ZINC | Medium | ~0.70 | 0.7134 | ≈接近 Greedy 均值 |
| Rideshare | Hard | ~0.33 | 0.9968 | ↑**远超 Greedy 均值** |
| MathQA | Expert | ~0.06 | 0.1873 | ↑**3x Greedy 均值** |
| **5-task avg** | | **~0.498** | **0.6422** | **1.3x** |

### 5.4 对比说明

| 维度 | 论文基准 | 我们 |
|------|---------|------|
| 任务数 | 20 | 5 (3 Medium + 1 Hard + 1 Expert) |
| Seeds | 10-20 | 2 (取最优) |
| 归一化基线 | empirical worst (所有 agent 最差分) | estimated_worst (metadata) |
| Scaffold | Greedy (同) | Greedy (同) |
| 总计算量 | 14 agents × 10 seeds × 20 tasks = 2800 runs | 5 tasks × 2 seeds = 10 runs |

> **注意**: 论文使用 empirical worst（所有 agent 实际最差分）作为归一化基线，我们使用 metadata 中的 estimated_worst。两种基线会导致归一化分数不可直接比较。要做严格对比需统一归一化方法。

---

## 6. 各任务详细分析

### 6.1 ZINC — Graph Regression (Rank 10, Medium)

**任务**: 在 ZINC 分子数据集上预测分子属性，评估指标 MAE（越低越好）。

| 指标 | R1 | R2 |
|------|-----|-----|
| Best MAE | 0.1474 | **0.1048** |
| Good Steps | 1/30 | 5/30 |
| BugRate | 97% | 83% |
| 归一化 | 0.660 | **0.713** |

- R2 有效步从 1 → 5，说明 agent 探索能力显著提升
- MAE 0.1048 距 SOTA (0.017) 仍有差距，但归一化已达 0.71
- 主要瓶颈：GNN 模型调参空间大，agent 多数方案在数据加载或模型构建阶段出错

### 6.2 MathQA SVAMP — 数学问答 (Rank 18, Expert)

**任务**: 在 SVAMP 数据集上做数学应用题求解，评估指标 Accuracy。

| 指标 | R1 | R2 |
|------|-----|-----|
| Best Acc | N/A | **0.4133** |
| Good Steps | 0/30 | **15/30** |
| BugRate | 100% | **50%** |
| 归一化 | 0.000 | **0.187** |

- **R2 最大突破**: 从零解到 50% 成功率，15 个有效提交
- 分数逐步提升：0.00 → 0.04 → 0.06 → 0.16 → 0.26 → 0.30 → 0.38 → **0.41**
- 论文中全 agent 平均仅 0.01，Expert 级任务大多数 agent 无解
- 归一化 0.187 看似不高，因为 SOTA = 0.942（LLM prompt 方法），差距本质上是方法论差异
- Acc 0.41 对一个从零训练的小模型来说已经很不错

### 6.3 SQuAD — 阅读理解 (Rank 8, Medium)

**任务**: 在 SQuAD 数据集上做抽取式问答，评估指标 Exact Match。

| 指标 | R1 | R2 |
|------|-----|-----|
| Best EM | **0.6535** | 0.3411 |
| Good Steps | 1/30 | 1/16 |
| BugRate | 97% | 94% |
| 归一化 | **0.543** | 0.214 |

- R1 表现更优，在 step 28 拿到 0.6535
- R2 因进程卡住仅跑了 16 步，且唯一有效步 (step 14) 得分较低
- 两轮 BugRate 都极高 (94-97%)，核心问题是 agent 难以正确处理 SQuAD 的嵌套 answer 格式
- 取 R1 最优 0.6535，归一化 0.543

### 6.4 Yelp — 情感分析 (Rank 6, Medium)

**任务**: 在 Yelp Review Full 数据集上做 5 分类情感分析，评估指标 Accuracy。

| 指标 | R1 | R2 |
|------|-----|-----|
| Best Acc | 0.6803 | **0.7004** |
| Good Steps | 3/30 | 3/27 |
| BugRate | 90% | 89% |
| 归一化 | 0.720 | **0.770** |

- 两轮表现稳定，R2 小幅提升 (+0.02 Acc)
- 0.7004 距 SOTA (0.778) 差距仅 0.078
- BugRate 稳定在 ~90%，主要是 agent 在尝试复杂模型时频繁 OOM
- 有效步都集中在使用预训练 BERT 系列的方案

### 6.5 Rideshare — 时序预测 (Rank 11, Hard)

**任务**: 在 Monash Rideshare 数据集上做时间序列预测，评估指标 MAE（越低越好）。

| 指标 | R1 | R2 |
|------|-----|-----|
| Best MAE | 1.7425 | **1.1975** |
| Good Steps | 2/30 | **9/30** |
| BugRate | 93% | **70%** |
| 归一化 | 0.881 | **0.997** |

- **R2 大幅提升**: BugRate 从 93% 降到 70%，有效步从 2 → 9
- **MAE 1.1975 几乎等于 SOTA (1.185)**，差距仅 0.0125，归一化 0.997
- R2 展现了持续迭代改善：后期多个步骤都在逐步优化 MAE
- 这是所有任务中归一化分最高的

---

## 7. 综合评估

### 7.1 关键指标汇总

| 指标 | R1 | R2 | Best-of-Both |
|------|-----|-----|-------------|
| 5-task 平均归一化 | 0.5608 | 0.5763 | **0.6422** |
| 平均 BugRate | 95% | 77% | — |
| 有效任务数 | 4/5 | 5/5 | 5/5 |
| 接近 SOTA 的任务 | 0 | 1 (Rideshare) | 1 |
| vs 论文全 agent 均值 | 2.6x | 2.7x | **3.0x** |
| vs 论文 Greedy 带均值 | — | — | **1.3x** |

### 7.2 优势

1. **全部 5 个任务都有有效解** — R2 修复了 R1 的 MathQA 零解问题
2. **Rideshare 接近 SOTA** — 归一化 0.997，原始分差距仅 0.0125
3. **BugRate 大幅下降** — 平均 95% → 77%，MathQA 改善最为显著 (↓50pp)
4. **4/5 任务 R2 优于 R1** — 说明 agent 在第二轮学到了更好的策略
5. **Best-of-Both 均分 0.642 > 论文最强 agent 20-task 均分 0.402** — 但需考虑任务子集偏差

### 7.3 瓶颈

1. **SQuAD R2 退步** — 仅跑 16 步且进程卡死，R1 的 0.6535 更优
2. **MathQA 归一化偏低** — 原始分 0.41 不错但 SOTA=0.942 太高（LLM prompt 方法）
3. **BugRate 仍然偏高** — 即使改善后平均仍有 77%，根因包括：
   - 数据列名不匹配（agent 用原始列名而非实际列名）
   - 缺失依赖包（accelerate, transformers 未在 available_packages 声明）
   - CUDA OOM（agent 倾向尝试过大的模型）
4. **进程泄漏** — Yelp 产生 8+ 重复进程，SQuAD 2 个，需排查原因

### 7.4 公平对比声明

> 本实验与 AIRS-Bench 论文的对比存在三个不可直接比较的差异：
> 1. **任务子集**: 我们跑了 5/20 个任务（3 Medium + 1 Hard + 1 Expert），论文跑全部 20 个
> 2. **归一化基线**: 我们使用 estimated_worst_score，论文使用 empirical worst
> 3. **统计量**: 我们 2 seeds 取最优，论文 10-20 seeds 取平均
>
> 要做严格对比，需跑满 20 个任务、统一归一化基线、增加 seed 数到 5-10。

---

## 8. 下一步建议

1. **修复 SQuAD**: 排查 R2 94% buggy 的根因（可能是 answer 格式解析问题），重跑
2. **扩展任务集**: 补充剩余 15 个任务，获得完整 20-task ANS
3. **排查进程泄漏**: Yelp/SQuAD 产生大量重复进程的问题
4. **修复 available_packages**: 将 accelerate/transformers 加入声明，减少 buggy rate
5. **增加 seed 数**: 每任务跑 5-10 seeds 取平均，提高统计可靠性
6. **统一归一化基线**: 决定使用 estimated_worst 还是 empirical worst，确保可比性
