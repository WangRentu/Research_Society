# Exp A 三方对比分析：Dojo Baseline vs ABL vs NAT

> 用于论文 Experiments / Discussion 部分的数据支撑。
> 截至 2026-03-28，NAT 79.7%、ABL 62.1% 完成。

---

## 1. 实验配置对齐

| 配置项 | Dojo Baseline | ABL | NAT |
|--------|--------------|-----|-----|
| 框架 | aira-dojo-baseline (Meta 原版) | aira-cogito (DLE-Agent) | aira-cogito (DLE-Agent) |
| Solver | GreedySolverConfig | GreedySolverConfig | GreedySolverConfig |
| 搜索策略 | Greedy 5 drafts | Greedy 5 drafts | Greedy 5 drafts |
| 算子模板 | **aide_operators**（原版 ~500 词） | **mceses_operators**（增强 ~1500 词） | **mceses_operators**（增强 ~1500 词） |
| 环境探测 (Recon) | **无** | **有** | **有** |
| 智能截断 | **无**（头尾截断，丢失 traceback） | **有**（保留错误块） | **有**（保留错误块） |
| 包版本信息 | 只有包名 | **包名+版本号** | **包名+版本号** |
| 认知状态 z_t | **无** | 每步 reflect → **清空**（只保留 recon） | 每步 reflect → **保留演化** |
| intervention_mode | N/A | ablated | natural |
| LLM | Qwen-3.5-397B-A17B | Qwen-3.5-397B-A17B | Qwen-3.5-397B-A17B |
| Step limit | 50 (截断自 200 step 运行) | 50 | 50 |
| Seeds | 3 (seed 3/4/5) | 3 (seed 1/2/3) | 3 (seed 1/2/3) |
| 硬件 | 4×RTX 5090 (32GB) | 4×RTX 5090 (32GB) | 4×RTX 5090 (32GB) |
| Interpreter | Python subprocess | Python subprocess | Python subprocess |
| Invalid seed 处理 | NS = 0 | NS = 0 | NS = 0 |

---

## 2. 聚合结果（10 任务，multi-seed avg NS，invalid=0）

| 系统 | Avg NS | Valid Rate | Scored Tasks | vs Baseline |
|------|--------|------------|--------------|-------------|
| **Dojo Baseline** | 0.319 | ~19% | 7/10 | — |
| **ABL** | 0.612 | 30% | 10/10 | +91.8% |
| **NAT** | 0.616 | 40% | 10/10 | +93.1% |

### 增益分解

```
Total gain (NAT vs Baseline):     +0.297 (93.1%)
├── System-level improvements:    +0.293 (91.8%)  ← ABL vs Baseline
│   ├── Enhanced operator prompts (API compatibility constraints)
│   ├── Environment reconnaissance (package versions, GPU info)
│   ├── Smart terminal output truncation (preserve tracebacks)
│   ├── Package version retention (name==version format)
│   └── Pre-installed missing dependencies
└── Cognitive state evolution:    +0.004 (0.8%)   ← NAT vs ABL
```

**结论**：总提升的 98.7% 来自系统级改进，1.3% 来自认知状态演化。

---

## 3. 逐任务对比

| Task | Tier | Baseline | ABL | NAT | NAT-Baseline | NAT-ABL |
|------|------|----------|-----|-----|--------------|---------|
| SQuAD Exact Match | Med | 0.737 | 0.363 | 0.754 | +0.017 | **+0.391** |
| Qm9 MAE | Hard | 0.000 | 0.733 | 0.747 | +0.747 | +0.014 |
| Zinc MAE | Hard | 0.196 | 0.681 | 0.698 | +0.502 | +0.017 |
| SVAMP Accuracy | Exp | 0.165 | 0.201 | 0.180 | +0.015 | -0.021 |
| SICK Accuracy | Easy | 1.010 | 1.111 | 1.089 | +0.079 | -0.022 |
| Code Retrieval MRR | Hard | 0.001 | 0.322 | 0.286 | +0.285 | -0.036 |
| Solar Weekly MAE | Easy | 0.837 | 0.937 | 0.897 | +0.060 | -0.040 |
| Yelp Accuracy | Med | 0.000 | 0.744 | 0.677 | +0.677 | -0.068 |
| SuperGLUE WSC | Exp | 0.206 | 0.313 | 0.219 | +0.013 | -0.094 |
| Winogrande Accuracy | Med | 0.040 | 0.711 | 0.617 | +0.577 | -0.094 |
| **Average** | | **0.319** | **0.612** | **0.616** | **+0.297** | **+0.005** |

### NAT vs ABL 胜负

- NAT wins: **3/10**（SQuAD +0.391, Qm9 +0.014, Zinc +0.017）
- ABL wins: **7/10**
- 最大正向：SQuAD +0.391（知识积累型推理任务）
- 最大负向：Winogrande -0.094, WSC -0.094（方向锁定效应）

---

## 4. 系统级改进详解（ABL vs Baseline 的 +91.8% 来源）

### 4.1 增强算子 Prompt（贡献最大）

Dojo 原版 aide_operators 是通用指令（~500 词）。mceses_operators 新增了 12 条精确约束：

1. **Resource utilization block**：强制 agent 在代码开头检测 CPU/GPU 并设置线程数
2. **`ReduceLROnPlateau` 移除 verbose**：PyTorch ≥2.4 移除了该参数
3. **`predict_with_generate` 废弃**：transformers ≥4.46 移除，需用 GenerationConfig
4. **`datasets.map()` 设 `num_proc=1`**：避免沙箱中子进程崩溃
5. **`evaluate` 库未安装**：引导用 sklearn.metrics 替代
6. **`fp16=True` 而非 `bf16=True`**：当前 GPU 环境不支持 BF16 AMP
7. **`torch.optim.AdamW`** 替代 transformers 废弃的 AdamW
8. **lightgbm `early_stopping` callback** 替代废弃的 `early_stopping_rounds`
9. **向量化优先**：避免 Python for 循环处理大数据集
10. **`compute_overview` 变量**：注入 GPU 型号、显存、CPU 数
11. **`packages` 含版本号**：agent 可判断 API 兼容性
12. **`cognitive_state` 占位符**：为 z_t 注入预留接口

**效果**：Baseline 的 81% buggy rate 中大量来自 API 版本不兼容（ReduceLROnPlateau verbose、import evaluate 等）。这些在增强 prompt 中被直接预防。

### 4.2 环境探测（Recon）

首步执行前运行探测脚本，输出结构化 JSON：

```json
{
  "python": "3.12",
  "torch": "2.10.0",
  "transformers": "4.48.0",
  "gpu_count": 1,
  "gpu_name": "NVIDIA RTX 5090",
  "gpu_memory_gb": 32,
  "compatibility": ["ReduceLROnPlateau verbose removed", ...],
  "data_files": ["train/", "test/", "validation/"]
}
```

注入 `z_0.environment_context`。**让 agent 在写代码前就知道环境配置**，而非盲写后 debug。

### 4.3 智能终端输出截断

Dojo 原版 `trim_long_string`：保留头 2500 + 尾 2500 字符，中间截断。

问题：错误 Traceback 往往在中间（训练日志之后、清理代码之前），被截断后 agent 看不到报错，debug 无从下手。

改进后：先提取所有 `Traceback` 和 `Error` 块完整保留，剩余预算分配给头尾上下文。

### 4.4 包版本保留

`parse_pip_list_output()` 从 `torch` 改为 `torch==2.10.0`。Agent 知道版本号才能判断 API 是否可用。

### 4.5 预装缺失依赖

在 conda 环境中预装了 xgboost、torchmetrics、sktime、evaluate、rank_bm25。Baseline 缺这些包直接 ImportError。

---

## 5. NAT vs ABL 深入分析

### 5.1 Valid Rate 差异

NAT VR **40%** vs ABL **30%**（9/10 任务 NAT VR 更高）。认知状态帮助 agent 避免重复错误，产出更多可执行代码。但更多的 valid 提交并未转化为更高的 NS。

### 5.2 方差差异（方向锁定效应）

| Task | NAT std | ABL std | 倍数 |
|------|---------|---------|------|
| Winogrande | 0.242 | 0.039 | 6.2× |
| CodeRetrieval | 0.112 | 0.020 | 5.6× |
| SentimentAnalysis | 0.123 | 0.023 | 5.3× |
| CvQm9 | 0.116 | 0.081 | 1.4× |

NAT seed 间方差普遍是 ABL 的 3-6 倍。原因：
- 认知状态在早期积累了正确假设 → seed 表现很好
- 认知状态在早期积累了错误假设 → 方向锁定，整个 seed 偏离
- multi-seed avg 惩罚高方差：一个好 seed + 一个差 seed 的平均不如两个中等 seed

ABL 每步清空状态，探索更随机但更稳定。

### 5.3 ABL 为什么"不该"这么好

ABL 保留了 `environment_context`（recon 信息），这是 z_t 中**最有实用价值**的部分。所以 ABL 不是"完全无状态"，而是"无认知积累但有环境感知"。如果做一个 ABL-strict（连 recon 也清空），差距可能会拉大。

另外 ABL 使用了 mceses_operators 模板，其中 `{% if cognitive_state %}...{% endif %}` 块在 ABL 中不渲染（因为 z_t 内容为空），但 prompt 的其他增强部分（12 条约束、compute_overview 等）ABL 全都享有。

---

## 6. 论文写作要点

### 应该诚实呈现的

1. **总提升的 98.7% 来自系统级改进**——这是事实，回避它反而让论文不可信
2. **NAT vs ABL 的 z_t 因果效应很小**（+0.005 Avg NS）——但 VR 差异显著（+10pp）
3. **NAT 的 direction-locking 问题**——这是发现，不是缺陷
4. **逐任务效果差异巨大**——SQuAD +0.391 vs Winogrande -0.094

### 可以 frame 的叙事

1. "System-level improvements are a **prerequisite** for cognitive evolution to be effective. Before an agent can benefit from *thinking*, it must first be able to *code*."
2. "Cognitive state evolution trades **consistency for direction**: it dramatically improves valid submission rate but introduces variance through direction-locking."
3. "The benefit is **task-dependent**: knowledge-accumulation tasks (SQuAD) show large gains, while diversity-dependent tasks (Winogrande) suffer from over-commitment."
4. 将系统级改进作为**独立贡献**呈现，而非 z_t 的附属品

### 下一步改进方向（提高 NAT vs ABL 差距）

| 改进 | 预期效果 | 不影响 ABL？ |
|------|----------|-------------|
| plateau 检测 + 状态衰减 | 降低方向锁定风险，减少 NAT 方差 | ✓ |
| z_t 参与 search_policy（confidence → draft/improve 选择） | 让认知状态真正影响行为 | ✓ |
| 选择性 reflect（只在有信息增量时触发） | 减少噪声 reflect，提高 z_t 质量 | ✓ |
| Greedy draft 多样性（每个 draft 用不同 focus_direction） | 打破同质化 | ✓ |

这些改动只走 `intervention_mode="natural"` 路径，不影响 ABL 对照。
