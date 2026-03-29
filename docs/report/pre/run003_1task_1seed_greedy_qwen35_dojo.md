# Run 003 实验报告：Code Retrieval (CodeXGlue MRR)

> **1 Task × 1 Seed 单次运行分析**
> 生成时间：2026-03-19

---

## 1. 实验配置

| 配置项 | 值 |
|--------|-----|
| **Run ID** | run_003 |
| **任务名称** | CodeRetrievalCodeXGlueMRR |
| **任务类别** | Code Retrieval（代码检索） |
| **数据集** | google/code_x_glue_tc_nl_code_search_adv |
| **评估指标** | MRR（Mean Reciprocal Rank，越高越好） |
| **测试集大小** | 19,210 条查询 |
| **Benchmark** | AIRS-Bench |
| **Seed** | 42 |
| **LLM** | Qwen 3.5-397B-A17B（via DashScope/LiteLLM） |
| **搜索策略** | Greedy Search |
| **步数上限** | 100 |
| **执行超时** | 14,400 秒（4 小时/步） |
| **初始 Draft 数** | 5 |
| **Debug 概率** | 1.0 |
| **最大 Debug 深度** | 20 |
| **启动时间** | 2026-03-18 09:06:09 |

### 可用包配置

```
numpy, pandas, scikit-learn, statsmodels, xgboost, lightgbm,
torch, torchvision, torch-geometric, bayesian-optimization, timm
```

> **注意**：实际容器内额外安装了 `datasets==4.0.0`（任务依赖），但 `transformers`、`accelerate` 等未在 available_packages 中声明。

---

## 2. 任务说明

**Code Retrieval** 任务要求：给定自然语言查询（如 "Downloads Sina videos by URL"），从 19,210 个代码片段的语料库中检索出最相关的代码，以 MRR 作为评估指标。

- **SOTA**：0.6113（UniXcoder, ACL 2022）
- **最差估计分数**：0.0
- **最优分数**：1.0

---

## 3. 运行总览

| 指标 | 值 |
|------|-----|
| **总步数** | 27 |
| **有效步数（非 Buggy）** | 11（40.7%） |
| **Buggy 步数** | 16（59.3%） |
| **最佳 MRR** | **0.3791**（Step 18） |
| **总运行时长** | ~28.4 小时 |
| **运行区间** | 2026-03-17 20:55 → 2026-03-19 01:18 |

---

## 4. 逐步执行记录

| Step | 算子 | 状态 | MRR | 父节点 | 执行时间 | 说明 |
|------|------|------|-----|--------|---------|------|
| 0 | - | Buggy | - | - | 0s | 初始状态 |
| 1 | draft | ✅ Good | 0.0011 | - | 44s | Bi-Encoder + CodeBERT 基础方案 |
| 2 | draft | ❌ Buggy | - | - | 4s | 两阶段检索，立即失败 |
| 3 | draft | ✅ Good | 0.2819 | - | 1,300s | Fine-tuned CodeBERT + 对比学习 |
| 4 | draft | ❌ Buggy | - | - | 500s | Hard Negative Mining 尝试 |
| 5 | draft | ❌ Buggy | - | - | 0s | 失败 |
| 6 | debug | ❌ Buggy | - | 4 | 928s | 设备不匹配修复 |
| 7 | debug | ❌ Buggy | - | 2 | 5s | Keras/Transformers 兼容性 |
| 8 | debug | ✅ Good | 0.0011 | 6 | 109s | 恢复方案（但无提升） |
| 9 | debug | ❌ Buggy | - | 5 | 6,987s | CUDA OOM 错误 |
| 10 | debug | ✅ Good | 0.0314 | 9 | 4,939s | 恢复运行 |
| 11 | debug | ✅ Good | 0.2755 | 7 | 6,531s | 数据对齐修复 |
| 12 | improve | ❌ Buggy | - | 3 | 7,865s | 扩展上下文尝试，超时 |
| 13 | debug | ✅ Good | 0.2561 | 12 | 11,745s | Stage 2 超时修复 |
| 14 | improve | ✅ Good | **0.3219** | 3 | 1,304s | GraphCodeBERT + 多粒度融合 |
| 15 | improve | ❌ Buggy | - | 14 | 10s | 可学习投影头，立即失败 |
| 16 | debug | ❌ Buggy | - | 15 | 1,261s | DataParallel 包装问题 |
| 17 | debug | ✅ Good | 0.3104 | 16 | 1,373s | DataParallel state_dict 修复 |
| 18 | improve | ✅ Good | **0.3791** | 14 | 3,126s | Learned Pooling + 扩展上下文 |
| 19 | improve | ❌ Buggy | - | 18 | 997s | Dense/Sparse 混合融合失败 |
| 20 | debug | ✅ Good | 0.3360 | 19 | 3,213s | bfloat16 张量类型修复 |
| 21 | improve | ❌ Buggy | - | 20 | 10s | 非对称温度缩放，立即失败 |
| 22 | debug | ❌ Buggy | - | 21 | 12s | 维度不匹配（28×768 vs 2312×768） |
| 23 | debug | ❌ Buggy | - | 22 | 1,086s | 维度不匹配（28×778 vs 776×768） |
| 24 | debug | ✅ Good | 0.3406 | 23 | 3,442s | 零维张量拼接修复 |
| 25 | improve | ❌ Buggy | - | 24 | 1,365s | 对比学习 + 动量编码器 |
| 26 | debug | ❌ Buggy | - | 25 | 3,770s | Memory Bank 出队问题 |

---

## 5. 分数演进

```
MRR
0.40 ┤
     │                                    ★ 0.3791 (Step 18, Best)
0.35 ┤                                          ● 0.3406 (Step 24)
     │                                    ● 0.3360 (Step 20)
     │                          ● 0.3219 (Step 14)
0.30 ┤                                ● 0.3104 (Step 17)
     │          ● 0.2819 (Step 3)
     │                    ● 0.2755 (Step 11)
0.25 ┤              ● 0.2561 (Step 13)
     │
0.20 ┤
     │
0.10 ┤
     │
0.05 ┤        ● 0.0314 (Step 10)
     │  ● 0.0011 (Step 1)    ● 0.0011 (Step 8)
0.00 ┼──────────────────────────────────────────────
     0    3    8   10  11  13  14  17  18  20  24   Step
```

### 关键改进路径

```
Draft(3): 0.2819  →  Improve(14): 0.3219  →  Improve(18): 0.3791 ★
  │ CodeBERT+         │ GraphCodeBERT+       │ Learned Pooling+
  │ 对比学习          │ 多粒度融合           │ 扩展上下文
  │                   │
  └→ Debug(13): 0.2561 (超时修复分支)
```

**最佳路径**：Draft → Improve → Improve（三级改进链）

---

## 6. 算子效果分析

| 算子 | 总次数 | 成功次数 | 成功率 | 最佳贡献 |
|------|--------|---------|--------|---------|
| **draft** | 5 | 2 | 40.0% | 0.2819（Step 3） |
| **debug** | 12 | 6 | 50.0% | 0.3406（Step 24） |
| **improve** | 9 | 3 | 33.3% | **0.3791**（Step 18）★ |

- **Draft** 阶段产出了 2 个可用基线（Step 1 和 Step 3），其中 Step 3 成为后续改进的种子节点
- **Debug** 是最高频算子（12 次），成功率 50%，主要用于修复代码错误和运行时问题
- **Improve** 虽然成功率最低（33.3%），但贡献了最佳分数（Step 18）

---

## 7. 错误分析

### 7.1 Buggy 步骤分类

| 错误类型 | 步骤 | 次数 |
|---------|------|------|
| **维度/形状不匹配** | 15, 21, 22, 23 | 4 |
| **CUDA OOM** | 9 | 1 |
| **设备不匹配** | 6 | 1 |
| **依赖/兼容性** | 7 | 1 |
| **运行时错误** | 2, 4, 5, 16, 19, 25, 26 | 7 |
| **初始状态** | 0 | 1 |

### 7.2 典型错误模式

1. **Improve 后立即 Buggy（Step 15, 21）**：Agent 在 Improve 时引入新架构组件（投影头/温度缩放），但未正确处理维度兼容性
2. **Debug 链修复失败（Step 22→23）**：连续 Debug 未能解决维度不匹配问题，说明 Agent 对张量形状推理能力有限
3. **高级方案失败（Step 19, 25）**：Dense/Sparse 混合融合和动量编码器等复杂方案均失败，表明 Agent 倾向尝试复杂方案但实现不稳定

---

## 8. March of 9's 归一化分数

### 计算公式

$$\varphi(s) = -\log_{10}(|s - s_{\text{optimal}}|)$$

$$\text{normalized} = \frac{\varphi(s_{\text{agent}}) - \varphi(s_{\text{worst}})}{\varphi(s_{\text{SOTA}}) - \varphi(s_{\text{worst}})}$$

### 计算过程

| 参数 | 值 | φ 值 |
|------|-----|------|
| s_optimal | 1.0 | - |
| s_worst | 0.0 | φ(0.0) = -log₁₀(1.0) = **0.000** |
| s_SOTA | 0.6113 | φ(0.6113) = -log₁₀(0.3887) = **0.4104** |
| s_agent（best） | 0.3791 | φ(0.3791) = -log₁₀(0.6209) = **0.2070** |

$$\text{normalized} = \frac{0.2070 - 0.000}{0.4104 - 0.000} = \frac{0.2070}{0.4104} \approx \boxed{0.504}$$

### 分数解读

- **0.504** 表示 Agent 在对数尺度上达到了 SOTA 的 50.4%
- 原始 MRR 0.3791 / SOTA 0.6113 = 62.0%（线性比例）
- 对数归一化后分数较低，因为 MRR 从 0.38 到 0.61 的提升在对数空间中代表显著的难度跳跃

---

## 9. 关键发现

### 9.1 Greedy Search 有效性
- 59.3% 的 Buggy 率（远低于 5-task 实验的 90%+），说明 Code Retrieval 任务对 Agent 更友好
- 27 步中产出了 11 个有效节点，提供了充足的改进基础

### 9.2 改进链条清晰
- 最佳路径为 **3 级改进链**：Draft(3) → Improve(14) → Improve(18)
- 每次 Improve 都带来了模型架构的实质升级：CodeBERT → GraphCodeBERT → Learned Pooling

### 9.3 模型选择能力
- Agent 展现了合理的模型演进路线：从 CodeBERT 基线出发，升级到 GraphCodeBERT，最终采用定制的 Pooling 策略
- 但高级融合方案（Dense+Sparse、动量编码器）均失败，说明复杂架构的实现超出了 Agent 的可靠能力范围

### 9.4 与 SOTA 差距分析
- Agent 最佳 MRR = 0.3791，SOTA = 0.6113，差距 0.2322
- SOTA 方法（UniXcoder）使用了跨模态预训练，而 Agent 在 4 小时限制内仅能 fine-tune 现有模型
- 提升空间：更长的训练时间、更大的 batch size、BM25+Neural 混合检索

---

## 10. 运行环境

| 环境项 | 值 |
|--------|-----|
| Git Commit | ac9619aab9c5e8e96f94b77ee436b67f802aeaff |
| PyTorch 版本 | 2.10.0+cu130 |
| 用户 | jingyixi |
| 工作目录 | shared/logs/aira-dojo/user_jingyixi_issue_airsbench_textual_classification_sick/run_003/ |
| Checkpoint | run_003/checkpoint/journal.jsonl |
