# MLE-Bench vs AIRS-Bench 对比分析报告

> **分析日期**: 2026-03-19
> **用途**: 汇报材料 — 两大 AI Research Agent Benchmark 的定位、机制、差异对比

---

## 1. 一句话区分

- **MLE-Bench**（OpenAI, 2024）：74 个真实 Kaggle 竞赛，测试 Agent 的 **ML 工程能力**
- **AIRS-Bench**（Meta FAIR, 2025）：20 个跨领域研究任务，测试 Agent 的 **AI 科研能力**

---

## 2. 基本信息

| 维度 | MLE-Bench | AIRS-Bench |
|------|-----------|------------|
| **发布方** | OpenAI | Meta FAIR + 合作者 |
| **发布时间** | 2024 | 2025 |
| **GitHub** | openai/mle-bench | facebookresearch/airs-bench |
| **论文** | NeurIPS 2024 | 2025 |
| **任务数量** | **74** 个 Kaggle 竞赛 | **20** 个研究任务 |
| **执行框架** | aira-dojo（Meta FAIR） | aira-dojo（Meta FAIR） |
| **任务来源** | 真实 Kaggle 竞赛历史数据 | 学术 benchmark 数据集 |
| **定位** | ML 工程 / 竞赛能力评测 | 多领域科研泛化能力评测 |

---

## 3. 任务领域覆盖

### MLE-Bench（74 个任务）

按难度分层：

| 难度 | 数量 | 示例 |
|------|------|------|
| Low | 20 | Digit Recognizer, Titanic, House Prices |
| Medium | 38 | Dogs vs Cats, Store Sales Forecasting, NLP with Disaster Tweets |
| High | 15 | Google AI4Code, Feedback Prize, BirdCLEF |

按领域分布：

| 领域 | 任务类型 |
|------|---------|
| 计算机视觉 | 图像分类、目标检测、语义分割、图像匹配 |
| NLP | 文本分类、情感分析、命名实体识别、文本摘要 |
| 时序预测 | 商店销售、能源消耗、股票走势 |
| 表格数据 | 房价回归、保险预测、信用评分 |
| 音频 | 鸟类声音识别、音乐分类 |
| 生物信息 | 分子翻译、疾病检测 |
| 推荐系统 | 协同过滤、点击率预测 |

预设子集：

| 子集 | 数量 | 用途 |
|------|------|------|
| `all.txt` | 74 | 完整 benchmark |
| `lite.yaml` | 27 | 轻量评测 |
| `dev.yaml` | 4 | 开发调试 |
| `debug.yaml` | 1 | 单任务测试 |
| `low.txt` / `medium.txt` / `high.txt` | 20 / 38 / 15 | 难度分层 |

### AIRS-Bench（20 个任务）

| 领域 | 任务 | 指标 |
|------|------|------|
| **代码生成** | APPS (Pass@5) | Pass@5 |
| **代码检索** | CodeXGlue (MRR) | MRR |
| **共指消解** | SuperGLUE, Winogrande | Accuracy |
| **分子属性** | QM9 | MAE |
| **图回归** | ZINC | MAE |
| **数学问答** | SVAMP | Accuracy |
| **阅读理解** | SQuAD, DuoRC, Eli5, FinQA | Exact Match / F1 |
| **情感分析** | Yelp Review Full | Accuracy |
| **文本分类** | SICK | Accuracy |
| **时序预测** | Rideshare, 多个 Monash TSF 数据集 | MAE |

---

## 4. 评分体系对比

### MLE-Bench：Kaggle Medal 排名

```
每个任务的评分流程:
  Agent 提交 submission.csv
    → 与 Kaggle 历史 leaderboard 排名对比
    → 判定 medal 等级

Medal 标准:
  Gold   = 前 10% 参赛者
  Silver = 前 5% (或 Top 100, 取较大值)
  Bronze = 前 40% (或 Top 500, 取较大值)
  None   = 未达到 Bronze 线

最终指标: medal 分布
  例: "12 Gold, 20 Silver, 15 Bronze, 27 None"
```

### AIRS-Bench：March of 9's 对数归一化

```
每个任务的评分流程:

  Step 1: 对数变换
    φ(s) = -log₁₀(|s - s_optimal|)

  Step 2: 线性缩放到 [0, 1]
    normalized = (φ(agent得分) - φ(最差分)) / (φ(SOTA分) - φ(最差分))

最终指标: ANS (Average Normalized Score)
  ANS = 所有任务归一化分数的平均值

辅助指标:
  - VSR (Valid Submission Rate): 有效提交率
  - Elo: 头对头 Elo 评分
```

### 评分哲学差异

| | MLE-Bench | AIRS-Bench |
|---|---|---|
| **参照物** | Kaggle 真实人类选手 | SOTA 论文结果 |
| **跨任务比较** | Medal 计数（离散） | 归一化分数（连续） |
| **难度感知** | 通过 leaderboard 人数隐含 | φ 对数变换显式处理 |
| **满分含义** | 击败所有 Kaggle 选手 | 达到 SOTA 论文水平 |
| **零分含义** | 未产出有效提交 | 等于或差于估计最差分 |

---

## 5. Agent Prompt 策略对比（关键差异）

两个 benchmark 共享 aira-dojo 执行框架，但使用**不同的 operator prompt**。

### Draft（初始方案生成）

| 策略项 | MLE-Bench | AIRS-Bench |
|--------|-----------|------------|
| **角色设定** | "Kaggle Grandmaster attending a high-stakes competition" | "Kaggle Grandmaster attending a high-stakes competition"（同） |
| **验证策略** | **强制 5-fold CV** | **单 validation split**（避免 CV 开销） |
| **GPU 使用** | **必须用 DataParallel**（≥2 GPU 时） | 多 GPU 可选，不强制 DataParallel |
| **资源最大化** | "MUST actively maximize compute" | "Use effectively when beneficial" |
| **复杂度梯度** | 第一个 idea 简单，逐步复杂 | 同 |

### Improve（方案改进）

| 策略项 | MLE-Bench | AIRS-Bench |
|--------|-----------|------------|
| **改进粒度** | 原子改进（single actionable change） | 同 |
| **CV 要求** | 保持 5-fold CV | 保持单 split |
| **核心方法** | 不改变核心方法论 | 同 |

### Debug（Bug 修复）

| 策略项 | MLE-Bench | AIRS-Bench |
|--------|-----------|------------|
| **修复范围** | 只修 bug，不改核心逻辑 | 同 |
| **记忆引用** | 参考之前的 debug 尝试记录 | 同 |

### 影响分析

```
MLE-Bench 的 5-fold CV 策略:
  优点: 评估更稳健，得分方差更小
  缺点: 每次执行时间 × 5，在有限步数内能尝试的方案更少

AIRS-Bench 的单 split 策略:
  优点: 迭代更快，同样步数内能探索更多方案
  缺点: 评估可能不稳定，有 overfitting validation split 的风险
```

---

## 6. Solver 配置对比

两个 benchmark 在 aira-dojo 中共享三种搜索策略，但参数配置有差异：

### Greedy Search

| 参数 | MLE-Bench | AIRS-Bench |
|------|-----------|------------|
| `num_drafts` | 5 | 5 |
| `debug_prob` | 1.0 | 1.0 |
| `max_debug_depth` | 20 | 20 |
| `data_preview` | true | true |
| `improvement_steps` | 5 | 5 |

### MCTS

| 参数 | MLE-Bench | AIRS-Bench |
|------|-----------|------------|
| `step_limit` | 10,000 | 10,000 |
| `num_children` | 5 | 5 |
| `uct_c` | 0.25 | 0.25 |
| `max_debug_depth` | 20 | 20 |

### Evolutionary

| 参数 | MLE-Bench | AIRS-Bench |
|------|-----------|------------|
| `num_generations` | 100 | 100 |
| `individuals_per_generation` | 5 | 5 |
| `crossover_prob` | 0.5 | — |
| `num_islands` | 1 | 1 |
| `max_island_size` | 500 | 500 |

> Solver 参数基本一致，**核心差异在 operator prompt**（5-fold CV vs 单 split）。

---

## 7. 数据准备与基础设施对比

### MLE-Bench

```
安装流程:
  1. git clone openai/mle-bench（含 Git LFS 大文件）
  2. pip install -e .
  3. 配置 Kaggle API Token
  4. 运行 prepare.py 下载各竞赛数据

数据存储:
  MLE_BENCH_DATA_DIR/
  └── {competition_name}/
      └── prepared/
          ├── public/      # Agent 可见的训练数据
          └── private/     # 隐藏的测试标签

数据量: 每个竞赛 几十 MB ~ 几 GB 不等，总计 ~100GB+
```

### AIRS-Bench

```
安装流程:
  1. 从 HuggingFace Hub 下载 16 个数据集（hf_datasets.csv）
  2. 各任务的 prepare.py 处理为统一格式

数据存储:
  AIRS_BENCH_DATA_DIR/
  └── datasets/airs_raw_datasets/
      └── {dataset_name}/

任务目录:
  airsbench/tasks/rad/
  └── {TaskName}/
      ├── metadata.yaml       # SOTA/worst/optimal 分数
      ├── prepare.py          # 数据准备脚本
      ├── evaluate.py         # 评测脚本
      └── sample_submission.csv

数据量: ~10-20GB 总计
```

---

## 8. 排行榜与已有结果

### MLE-Bench 排行榜（OpenAI 论文数据）

| Agent | Medals (Gold/Silver/Bronze) | 有效提交率 |
|-------|---------------------------|-----------|
| OpenAI o1-preview + MLAB scaffold | 16.9% Gold | ~75% |
| GPT-4o + AIDE scaffold | 8.7% Gold | ~60% |
| Claude 3.5 Sonnet + MLAB | 7.3% Gold | ~65% |

### AIRS-Bench 排行榜（aira-dojo 数据）

| Agent | ANS | Seeds |
|-------|-----|-------|
| Greedy gpt-oss-120b | 0.402 ± 0.031 | 10 |
| Greedy gpt-oss-20b | 0.400 ± 0.032 | 10 |
| Greedy o3-mini | 0.391 ± 0.022 | 10 |
| Greedy GPT-4o | 0.309 ± 0.028 | 10 |

### 本次实验结果（Greedy Qwen-3.5-397B, 5 tasks, 1 seed）

| 任务 | 原始分 | March of 9's 归一化 |
|------|--------|-------------------|
| GraphRegressionZincMae | MAE=0.1474 | 0.660 |
| MathQuestionAnsweringSVAMP | 无提交 | 0.000 |
| ReadingComprehensionSquad | EM=0.6535 | 0.543 |
| SentimentAnalysisYelp | Acc=0.6803 | 0.720 |
| TimeSeriesForecastingRideshare | MAE=1.7425 | 0.881 |
| **平均 ANS** | | **0.561** |

---

## 9. 适用场景建议

| 场景 | 推荐 Benchmark | 原因 |
|------|---------------|------|
| 测试 Agent 的 **Kaggle 竞赛实力** | MLE-Bench | 74 个真实竞赛，有人类 leaderboard 参照 |
| 测试 Agent 的 **科研泛化能力** | AIRS-Bench | 20 个跨领域任务，归一化评分跨任务可比 |
| **快速迭代**实验（资源有限） | AIRS-Bench | 20 任务 + 单 split，每轮实验 ~6h |
| **全面评测**（资源充足） | MLE-Bench | 74 任务 + 5-fold CV，每轮实验 ~24-48h |
| 对比已有 **工业 Agent**（GPT-4o, o1 等） | MLE-Bench | OpenAI 有官方 baseline 数据 |
| 对比已有 **开源 Agent** | AIRS-Bench | Meta FAIR 有 14 个 agent 配置的完整结果 |
| **发论文**引用 | 两者均可 | MLE-Bench(NeurIPS 2024) / AIRS-Bench(2025) |

---

## 10. 代码结构对比

### MLE-Bench 在 aira-dojo 中的文件

```
forks/aira-dojo/src/dojo/
├── tasks/mlebench/
│   ├── task.py                          # MLEBenchTask 任务类
│   ├── evaluate.py                      # 评测逻辑（对接 Kaggle leaderboard）
│   ├── instructions.txt                 # Agent 指令模板
│   ├── README.md                        # 安装指南
│   ├── splits/                          # 任务子集定义
│   │   ├── all.txt                      # 全部 74 个
│   │   ├── lite.yaml                    # 轻量 27 个
│   │   ├── dev.yaml                     # 开发 4 个
│   │   └── low.txt / medium.txt / high.txt
│   └── utils/
│       └── prepare.py                   # 数据下载准备
├── config_dataclasses/task/
│   └── mlebench.py                      # 配置 + operator prompt 定义
├── configs/
│   ├── benchmark/mlebench/              # Benchmark 配置
│   ├── solver/mlebench/                 # Solver 配置（greedy/evo/mcts）
│   ├── solver/operators/mlebench/       # 算子 prompt（draft/improve/debug/crossover/analyze）
│   ├── task/mlebench/                   # 任务默认配置
│   └── _exp/mlebench/                   # 实验预设（aide_greedy_o3, aira_greedy_o3...）
```

### AIRS-Bench 在 aira-dojo 中的文件

```
forks/aira-dojo/src/dojo/
├── tasks/airsbench/
│   └── task.py                          # AIRSBenchTask 任务类
├── config_dataclasses/task/
│   └── airsbench.py                     # 配置 + operator prompt 定义
├── configs/
│   ├── benchmark/airsbench/             # Benchmark 配置
│   ├── solver/airsbench/                # Solver 配置
│   ├── solver/operators/airsbench/      # 算子 prompt
│   └── _exp/airsbench/                  # 实验预设

forks/airs-bench/                        # 独立的 AIRS-Bench 仓库
├── airsbench/tasks/rad/
│   └── {TaskName}/
│       ├── metadata.yaml                # SOTA / worst / optimal
│       ├── prepare.py                   # 数据准备
│       ├── evaluate.py                  # 评测脚本
│       └── sample_submission.csv        # 提交格式示例
├── datasets/
│   └── hf_datasets.csv                  # HuggingFace 数据集列表
└── notebooks/
    └── create_summary_plots.ipynb       # 归一化计算 + 可视化
```

---

## 11. 关键结论

1. **MLE-Bench 更"工业"**：来自真实 Kaggle 竞赛，有人类选手排名参照，适合衡量 Agent 能否在实际 ML 工程中胜任。

2. **AIRS-Bench 更"学术"**：覆盖代码/数学/分子/图结构等多领域，归一化评分使跨领域比较成为可能。

3. **Prompt 差异是性能差异的重要来源**：5-fold CV vs 单 split 直接影响每步执行时间和探索效率。

4. **两者互补，不可替代**：
   - 在 AIRS-Bench 上表现好 ≠ 在 MLE-Bench 上表现好（反之亦然）
   - 全面评估 Agent 能力需要两者结合

5. **aira-dojo 是两者的统一执行框架**：同一套 Greedy/MCTS/Evo 搜索策略，切换 benchmark 只需更换任务配置和 operator prompt。
