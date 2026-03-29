# aira-cogito 搜索策略与节点架构详解

> **用途**: 汇报材料 — aira-cogito 框架中三种 Solver 的算法机制、节点定义、对比分析
> **源码位置**: `forks/aira-cogito/src/dojo/solvers/`

---

## 1. 核心概念：节点（Node）

三种策略操作的基本单元都是 **Node**，定义在 `dojo/core/solvers/utils/journal.py`。

**一个节点 = 一次完整的"写代码 → 执行 → 评分"循环。**

```
Node
├── code: str                  # LLM 生成的完整 Python 解题代码
├── plan: str                  # 解题思路（自然语言）
├── metric: MetricValue        # 执行后的评分（如 MAE=0.147, Acc=0.68）
├── is_buggy: bool             # 代码执行是否报错
├── parents: List[Node]        # 父节点（从谁改进/debug 而来）
├── children: Set[Node]        # 子节点（被谁改进/debug）
├── operators_used: List[str]  # 用了哪个算子（draft / debug / improve / crossover）
├── term_out: str              # 终端输出（含错误信息、打印结果）
├── exec_time: float           # 代码执行耗时
└── exit_code: int             # 进程退出码（0=成功, 1=报错）
```

### MetricValue（评分对象）

```
MetricValue
├── value: float | None        # 实际分数
├── maximize: bool             # True = 越高越好（Accuracy）, False = 越低越好（MAE）
└── info: Dict                 # 附加信息

WorstMetricValue（特殊子类）
└── value = None               # 永远排最差，用于 buggy 节点
```

比较逻辑：`maximize=True` 时高值 > 低值；`maximize=False` 时低值 > 高值；`None` 永远最差。

### Journal（日志/搜索记录）

```
Journal
├── nodes: List[Node]          # 所有产生过的节点（按 step 排序）
├── draft_nodes                # 所有 draft 产生的初始节点
├── buggy_nodes                # 所有 is_buggy=True 的节点
├── good_nodes                 # 所有 is_buggy=False 的节点
└── get_best_node()            # 返回 good_nodes 中 metric 最优的节点
```

**最终得分 = `journal.get_best_node().metric`，即所有步中的全局最优解。**

---

## 2. 四种算子（Operators）

所有策略共享同一套算子，每个算子本质是一次 LLM 调用：

| 算子 | 输入 | 输出 | 作用 |
|------|------|------|------|
| **Draft** | 任务描述 + 数据预览 | 全新代码 | 从零生成一份解题方案 |
| **Debug** | 父节点代码 + 报错信息 | 修复后的代码 | 根据错误日志修 bug |
| **Improve** | 父节点代码 + 当前得分 | 改进后的代码 | 在已成功代码基础上提升性能 |
| **Analyze** | 执行输出 | buggy 判定 + metric 提取 | 判断代码是否正常运行并提取分数 |
| **Crossover** | 两个父节点的代码 | 融合后的代码 | 结合两份方案的优点（仅 Evolutionary 使用） |

算子调用链：

```
Draft/Debug/Improve → 生成代码 → 执行代码 → Analyze → 得到 (is_buggy, metric)
```

---

## 3. 策略一：Greedy Search（贪心搜索）

> **源码**: `dojo/solvers/greedy/greedy.py`（549 行）

### 搜索树结构

```
              Root (空节点, metric=worst)
            / |   |   \   \
          D1  D2  D3  D4  D5         ← Phase 1: 5 个 Draft（独立生成）
          |       |        \
         db1     db3       db5       ← Phase 2: Debug（修 bug）
          |       |
         db2     db4
          |
         db3 ✅ metric=0.68          ← 修对了，成为 best_node
```

### 算法流程

```
1. 创建 Root 节点（空代码, metric=worst, is_buggy=True）

2. for step in range(step_limit):

   3. search_policy() 选择下一步操作：
      ├── if 已产生节点数 < num_drafts:
      │     → 返回 None → 触发 Draft（从零写代码）
      ├── elif random() < debug_prob:
      │     → 返回一个随机 buggy 叶节点 → 触发 Debug
      │     （前提：该节点的 debug 深度 < max_debug_depth）
      └── elif 存在 good_nodes:
            → 返回 best good node → 触发 Improve

   4. 执行选中的算子，生成新节点

   5. 运行代码 → Analyze → 标记 is_buggy + 提取 metric

   6. 如果新节点 metric 优于当前 best → 更新 best_node

3. 返回 best_node 的代码和得分
```

### 关键参数（本次实验配置）

```yaml
num_drafts: 5            # 先写 5 份独立方案
debug_prob: 1.0          # Phase 2 中 100% 做 debug（不做 improve）
max_debug_depth: 20      # 一条 debug 链最多 20 层
step_limit: 30           # 总共最多 30 个节点
```

### 特点

- **选择机制**: 简单直接 — debug 时随机选 buggy 节点，improve 时选最好节点
- **无信息回传**: 不像 MCTS 那样将结果回传到祖先节点
- **适合场景**: 任务简单、draft 质量高时效率最好
- **风险**: 如果 5 个 draft 方向都错了，后面 25 步 debug 全在错误方向上修补

---

## 4. 策略二：MCTS（蒙特卡洛树搜索）

> **源码**: `dojo/solvers/mcts/mcts.py`（650 行）

### 搜索树结构

```
                    Root (N=20, Q=3.2)
                 /    |    \     \     \
               D1     D2    D3    D4    D5          ← Expansion: 5 个 Draft
          (N=8,Q=1.6) (N=5,Q=0.8)  ...
           / | \       / | \
         I1  I2  I3   I4  I5  I6                    ← Expansion: 5 个 Improve
       (N=3) (N=2)   (N=1)
        / \    |
      I7  I8   db→db→✅                              ← buggy 走 Debug 链
     (N=1)(N=1)

I = Improve 节点, D = Draft 节点, db = Debug 节点
N = 访问次数, Q = 累计得分
```

### MCTSNode（扩展节点）

在 Node 基础上增加：

```
MCTSNode extends Node
├── explore_count: int         # 被访问次数 N
├── node_value: float          # 累计奖励值 Q
└── q_value(): float           # 平均值 = Q / N
```

### 核心：UCT 选择公式

```
UCT(node) = Q̄(node) + c × √( ln(N_parent) / N_node )
             ───────           ──────────────────────
             利用项              探索项
```

- **利用项** `Q̄`：该节点的归一化平均得分（越高说明这条路越有前途）
- **探索项**：访问次数越少，值越大（鼓励探索未充分尝试的分支）
- **c = 0.25**：探索常数，控制探索 vs 利用的平衡

### 算法流程

```
1. 创建 Root MCTSNode

2. for step in range(step_limit):

   3. Selection（选择）:
      从 Root 出发，每一层选 UCT 值最高的子节点，
      一直走到叶节点（无子节点的节点）

   4. Expansion（展开）:
      在叶节点上生成 num_children=5 个子节点
      ├── 如果叶节点是 Root → 用 Draft 算子
      └── 否则 → 用 Improve 算子

   5. Evaluation（评估）:
      执行每个子节点的代码 → Analyze → 得到 metric
      如果子节点 is_buggy → 进入 Debug Cycle（最多 max_debug_depth 轮）

   6. Backpropagation（回传）:
      将子节点的 metric 值沿路径回传到 Root：
      for node in path_from_leaf_to_root:
          node.explore_count += 1
          node.node_value += metric_value

3. 返回 best_node
```

### Debug Cycle（MCTS 专属）

```
对于 buggy 的子节点：
  for depth in range(max_debug_depth):
      新代码 = Debug(buggy_node.code, error_message)
      执行 → Analyze
      if not is_buggy:
          回传 metric 到路径上所有祖先
          break
```

### 关键参数

```yaml
step_limit: 500          # 远多于 Greedy
num_children: 5          # 每次展开 5 个子节点
uct_c: 0.25             # 探索常数
max_debug_depth: 20      # Debug 链最大深度
```

### 特点

- **智能选择**: UCT 公式自动平衡探索与利用，不像 Greedy 随机选
- **有信息回传**: 好分支的得分会传播到祖先，让后续选择更精准
- **搜索效率更高**: 但单步开销大（每次展开 5 个节点）
- **适合场景**: 搜索空间大、需要精细探索分支质量差异时

---

## 5. 策略三：Evolutionary（进化搜索）

> **源码**: `dojo/solvers/evo/evo.py`（1093 行）

### 多岛种群结构

```
  Island 0 (强岛)      Island 1 (中岛)      Island 2 (弱岛)
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │ D1  ✅ f=0.85 │    │ D2    f=0.60 │    │ D3  ✅ f=0.72 │
  │ D4    f=0.40 │    │ D5  ✅ f=0.71 │    │ D6    f=0.30 │
  └──────────────┘    └──────────────┘    └──────────────┘
         │                   │                   │
      Generation 1: 选岛 → 选父本 → Improve(变异) 或 Crossover(交叉)
         │                   │                   │
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │ D1  ✅ f=0.85 │    │ D5  ✅ f=0.71 │  ←─ Migration ──┤
  │ I(D1) f=0.88 │    │ C(D2,D5) f=? │    │ D3  ✅ f=0.72 │
  │ D4    淘汰    │    │ ...          │    │ I(D3) f=0.65 │
  └──────────────┘    └──────────────┘    └──────────────┘
         │                   │                   │
      Generation 2 ... N: 多代进化，优胜劣汰
```

### 核心概念

| 概念 | 对应 | 说明 |
|------|------|------|
| **Individual（个体）** | Node | 一份解题代码 |
| **Island（岛/种群）** | 节点集合 | 独立进化的子种群，有容量上限 |
| **Fitness（适应度）** | metric | 代码的评分 |
| **Mutation（变异）** | Improve 算子 | 改进单个父本 |
| **Crossover（交叉）** | Crossover 算子 | 融合两个父本 |
| **Migration（迁移）** | 岛间转移 | 强岛个体移植到弱岛 |

### Island 数据结构

```
Island
├── nodes: List[Node]              # 岛内所有个体
├── max_island_size: int           # 容量上限
├── fittest_individual: Node       # 岛内最优个体
├── register_node_in_island(node)  # 添加新个体（若满则淘汰最差）
├── remove_lowest()                # 移除适应度最低的个体
└── migrate_node(node)             # 从其他岛接收个体

SolutionsDatabase
├── islands: List[Island]          # 所有岛
├── get_normalized_score(score)    # 归一化到 [0,1]
└── sample_island() / sample_individual()  # 加权采样
```

### 算法流程

```
1. 创建 Root 节点 + SolutionsDatabase（含 num_islands 个空岛）

2. for generation in range(num_generations):

   3. if generation == 0（初始化）:
      └── 用 Draft 算子生成 individuals_per_generation 个初始方案
          分配到各岛

   4. else（进化迭代）:
      for i in range(individuals_per_generation):

         5. 选岛: 按岛的平均 fitness 加权采样
            （好岛被选中概率更大）

         6. 选父本: 岛内按个体 fitness 加权采样
            （好个体被选中概率更大）

         7. 选算子:
            ├── generation < num_generations_till_crossover:
            │     → Draft 或 Improve（变异）
            └── generation >= num_generations_till_crossover:
                  → Draft / Improve / Crossover（交叉）
                  （Crossover 概率 = crossover_prob）

         8. 执行算子 → 运行代码 → Analyze

         9. 如果 is_buggy → Debug Cycle（最多 max_debug_depth 轮）

        10. 如果解有效 → 加入岛
            （岛满时淘汰适应度最低的个体）

      11. Migration（周期性迁移）:
          ├── 触发概率: migration_prob
          ├── 将岛按平均 fitness 分为强弱两半
          ├── 弱岛只保留最优个体，其余清空
          └── 从强岛采样优秀个体移植到弱岛

3. 返回全局 best_node
```

### 温度调度（采样选择性控制）

```
T_t = T_initial - (T_initial - T_final) × t / num_generations

高温（T 大）→ 均匀采样（更多探索）
低温（T 小）→ 偏向好个体（更多利用）
```

早期高温鼓励探索多样方案，后期低温集中优化最好的方案。

### 关键参数

```yaml
num_islands: 1                     # 岛数量（默认 1，可扩展）
max_island_size: 500               # 每岛容量
num_generations: 100               # 进化代数
individuals_per_generation: 5      # 每代产生个体数
num_generations_till_crossover: 2  # 第 2 代开始允许交叉
crossover_prob: 0.5                # 交叉 vs 变异的概率
migration_prob: 0.0                # 迁移触发概率（默认关闭）
initial_temp: 1.0                  # 初始温度
final_temp: 1.0                    # 最终温度
```

### 独有算子：Crossover（交叉）

```
输入: Parent_A.code + Parent_B.code
提示: "融合这两份方案的优点，生成一份更好的代码"
输出: 新代码（继承两个父本的特征）

示例:
  Parent_A: 用 LightGBM 做特征工程, Acc=0.72
  Parent_B: 用 DistilBERT 做文本分类, Acc=0.68
  Child:    用 DistilBERT + LightGBM 特征增强, Acc=0.75（理想情况）
```

### 特点

- **种群多样性**: 多个岛独立进化，避免过早收敛到单一方案
- **交叉创新**: Crossover 可以融合不同方案的优点，产生新思路
- **自然淘汰**: 差方案被自动移除，搜索资源集中在优质方案上
- **适合场景**: 需要多种方案融合、搜索空间极大时

---

## 6. 三种策略对比总结

### 算法机制对比

| 维度 | Greedy | MCTS | Evolutionary |
|------|--------|------|--------------|
| **搜索结构** | 线性链（DAG） | 树 | 多种群（岛模型） |
| **节点选择** | 随机 buggy / 最优 good | UCT 公式（探索-利用平衡） | 适应度加权采样 + 温度调度 |
| **信息回传** | 无 | 有（Backpropagation） | 无（靠种群竞争淘汰） |
| **多样性来源** | 多个 draft 的随机性 | UCT 探索项 | 多岛 + 交叉 + 迁移 |
| **核心算子** | Draft + Debug | Draft + Improve + Debug | Draft + Improve + **Crossover** + Debug |
| **单步开销** | 低（1 个节点/步） | 高（展开 5 个节点/步） | 中等（1 个节点/步） |
| **默认步数** | 500 | 500 | 10,000（= 100 代 × 100 岛容量） |

### 搜索行为对比

```
Greedy:       D D D D D → db db db db db db db db db db ...
              "写 5 份，然后闷头 debug"

MCTS:         D D D D D → [选最有前途的] → I I I I I → [选最有前途的] → I I I I I → ...
              "写 5 份，不断展开最有前途的分支"

Evolutionary: D D D D D → [选最优 + 交叉] → 淘汰劣解 → [变异 + 交叉] → 淘汰 → ...
              "写 5 份，优胜劣汰，杂交出新品种"
```

### 适用场景建议

| 场景 | 推荐策略 | 原因 |
|------|---------|------|
| 任务简单，draft 质量高 | **Greedy** | 快速出解，debug 即可达标 |
| 方案空间大，需要精细比较 | **MCTS** | UCT 智能分配探索资源 |
| 需要多种思路融合创新 | **Evolutionary** | Crossover 可组合不同方案优点 |
| 计算预算有限 | **Greedy** | 单步开销最低 |
| 计算预算充足 | **Evolutionary** | 多岛多代充分搜索 |

---

## 7. 本次实验 Greedy 策略的具体表现

### 配置

```yaml
策略: Greedy Search
num_drafts: 5
debug_prob: 1.0        # 100% debug，不做 improve
max_debug_depth: 20
step_limit: 30
```

### 实际搜索行为

```
Steps  1-5:   Draft × 5（全部 buggy）
Steps  6-30:  Debug × 25（随机选 buggy 叶节点修 bug）
               ├── 大部分继续 buggy（数据列名问题无法通过 debug 修复）
               └── 极少数步骤碰巧修对 → 产出有效 metric
```

### 问题诊断

1. **debug_prob=1.0 意味着完全不做 Improve** — 即使有成功的节点，也不会在此基础上提升
2. **5 个 Draft 全 buggy** → 后续 25 步全在修 bug，没有新方向探索
3. **随机选择 buggy 节点** → 可能反复修同一条死路

### 如果换用 MCTS 或 Evolutionary

- **MCTS**: 每次展开 5 个 Improve 子节点，成功的分支会获得更多探索资源
- **Evolutionary**: 多个 draft 在不同岛独立进化，Crossover 可以融合部分成功方案

---

## 附录：配置文件位置

| 文件 | 说明 |
|------|------|
| `src/dojo/solvers/greedy/greedy.py` | Greedy 算法实现 |
| `src/dojo/solvers/mcts/mcts.py` | MCTS 算法实现 |
| `src/dojo/solvers/evo/evo.py` | Evolutionary 算法实现 |
| `src/dojo/core/solvers/utils/journal.py` | Node / Journal 数据结构 |
| `src/dojo/core/solvers/utils/metric.py` | MetricValue 评分对象 |
| `src/dojo/core/solvers/operators/draft.py` | Draft 算子 |
| `src/dojo/core/solvers/operators/debug.py` | Debug 算子 |
| `src/dojo/core/solvers/operators/improve.py` | Improve 算子 |
| `src/dojo/core/solvers/operators/crossover.py` | Crossover 算子 |
| `src/dojo/core/solvers/operators/core.py` | 算子基类 |
| `src/dojo/configs/solver/greedy.yaml` | Greedy 默认配置 |
| `src/dojo/configs/solver/mcts.yaml` | MCTS 默认配置 |
| `src/dojo/configs/solver/evo.yaml` | Evolutionary 默认配置 |
| `src/dojo/configs/solver/airsbench/` | airsbench 任务专用配置 |
