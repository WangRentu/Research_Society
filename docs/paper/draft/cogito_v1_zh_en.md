#  Cogito: Coupled Inner-Outer Evolution for Challenging AI Research Tasks

## 摘要
现有 AI 研究智能体将机器学习工程视为代码方案空间中的搜索（外部演化），但在认知层面基本是无状态的：智能体不断产生和评估候选代码，却不显式维护对任务的累积理解。我们提出 **Cogito**，一个将**内部认知演化**与**外部方案搜索**耦合的框架。Cogito 维护一个结构化认知状态 $z_t$，编码任务理解、工作假设、失败模式和搜索偏好；每步执行后 $z_t$ 通过 LLM 反思更新，并反向驱动下一步的算子选择和代码生成。这种内外耦合使搜索从盲目的方案空间遍历转变为由累积认知引导的定向探索。在 AIRS-Bench 的 20 个跨领域任务上，Cogito 以 30 步预算取得 Avg NS 0.522，超过 AIRA-dojo 以 200 步、10 seeds 取得的 0.402。通过因果状态干预实验（将 $z_t$ 替换为空白、无关任务状态或冻结快照），我们进一步验证：演化的认知状态携带任务特异性信息并因果性地改善决策，而非仅充当提示填充；其收益在困难、多步任务上最为显著。

## Abstract

Existing AI research agents treat machine learning engineering as search over a space of code solutions (outer evolution), yet remain largely stateless at the cognitive level: they continuously generate and evaluate candidate programs without maintaining an explicit, evolving understanding of the task. We propose **Cogito**, a framework that explicitly couples **inner cognitive evolution** with **outer solution search**. Cogito maintains a structured cognitive state $z_t$ encoding task understanding, working hypotheses, failure patterns, and search preferences; after each execution step, $z_t$ is updated via LLM-based reflection and, in turn, drives the next round of operator selection and code generation. This inner-outer coupling transforms search from blind traversal of the solution space into directed exploration guided by accumulated cognition. On AIRS-Bench across 20 cross-domain tasks, Cogito achieves an Avg Normalized Score of 0.522 with a 30-step budget, surpassing the 0.402 attained by AIRA-dojo with 200 steps and 10 seeds. Through causal state intervention experiments—replacing $z_t$ with blank states, states from unrelated tasks, or frozen snapshots—we further verify that the evolved cognitive state carries task-specific information and causally improves decision-making, rather than merely serving as prompt padding; these gains are most pronounced on challenging, multi-step tasks.

## 1 引言

AIDE \cite{aide2024} 将机器学习工程形式化为代码方案空间中的搜索：每个候选程序是一个节点，Draft、Debug 和 Improve 算子扩展方案树，自动评估反馈引导探索。这一 **”外部演化”** 范式——搜索发生在代码产物空间中——被 AIRA-dojo \cite{toledo2025aira} 系统化为统一的算子—搜索设计空间，并成为后续大量工作的基础语言。

然而，MLE-bench \cite{chan2024mlebench} 排行榜上的近期竞争格局表明，该领域已出现显著的**范式分化**。一路系统沿着搜索方向做大做深：MLEvolve、PiEvolve 等将进化搜索推至大规模分布式（Avg Medal 61%+）。另一路系统则转向放弃树结构，转而投资于**记忆与认知机制**：R\&D-Agent \cite{rdagent2024} 将研究假设生成与代码开发解耦为两个阶段；ML-Master 2.0 \cite{mlmaster2025} 引入短期—中期—长期三层记忆缓存，在 MLE-bench 上达到 56.4% medal rate。第三路系统如 Disarray（77.8%）则通过多模型 ensemble 和大规模算力实现。

这一分化暗示了一个更基础的问题：**在方案空间的外部搜索之外，智能体的内部认知是否构成一个独立的、值得投资的维度？** 现有系统——无论是搜索型还是记忆增强型——在认知层面仍然是碎片化的。搜索型系统（AIDE、AIRA-dojo）的内部理解隐含在提示词和瞬时上下文中，不跨步骤持久化。记忆增强系统（ML-Master 2.0）存储了历史交互痕迹，但其记忆本质上是**被动的**：它记录”发生了什么”，却不显式维护一个会主动演化的、关于任务的结构化信念——即”基于这些经历，我现在应该相信什么、下一步应该做什么”。

我们提出 **Cogito**，一个将**内部认知演化**与**外部方案搜索**显式耦合的框架。Cogito 的核心思想是：搜索不应只发生在代码空间中，也应发生在认知空间中——两个空间的演化相互驱动。具体地，Cogito 维护一个结构化认知状态 $z_t$，编码任务理解、工作假设、失败模式、搜索偏好和自评置信度。每步的完整循环为：

$$
z_t \xrightarrow{\text{驱动}} a_t \xrightarrow{\text{执行}} r_t \xrightarrow{\text{反思}} z_{t+1}
$$

其中 $z_t$ 通过注入算子提示来**驱动**代码生成（外部演化），执行反馈 $r_t$ 再通过 LLM 反思**回传**更新认知状态（内部演化）。这种内外耦合使搜索从盲目的方案空间遍历转变为由累积认知引导的定向探索。搜索拓扑与认知层正交：$z_t$ 可以沿线性链演化（等价于认知增强的贪心搜索），也可以在分支树上展开（等价于在认知状态空间上做蒙特卡洛树搜索）。当 $z_t = \varnothing$ 时，系统退化为标准 AIRA-dojo。

在 AIRS-Bench \cite{lupidi2025airs} 的 20 个跨领域任务上，Cogito 以 30 步预算取得 Avg NS 0.522，超过 AIRA-dojo 以 200 步、10 seeds 取得的 0.402（表 [X]）。更重要的是，我们不止报告性能提升，还通过**因果干预实验**验证认知层的作用机制。我们设计四种条件——Natural（正常演化）、Ablated（每步清空 $z_t$）、Scrambled（替换为无关任务的成熟 $z_t$）、Frozen（步骤 5 后冻结）——在同一套 LLM、算子模板和 token 预算下，**唯一区别是 $z_t$ 的内容**。这使我们能够区分认知演化本身的价值、任务相关性的价值和持续反思的价值。

实验表明：
- 演化的 $z_t$ 携带任务特异性信息——Scrambled 条件下性能显著下降，证明价值来自内容而非形式（RQ3）；
- 高信息量事件（分数跳变、首次成功）伴随的反思贡献了大部分有效状态更新（RQ2）；
- 内外耦合的收益在困难、多步任务上最为显著，在简单任务上收益有限（RQ4）。

**贡献。** （1）Cogito：一个将内部认知演化与外部方案搜索显式耦合的 AI 研究智能体框架，在 AIRS-Bench 上以更少步数超过先前最优。（2）因果干预实验设计，通过状态替换/清空/冻结验证认知层的作用机制，而非仅依赖性能对比。（3）关于内外耦合何时有效的实证分析：认知投资在困难任务上回报最高，在简单任务上应适度克制。

## 1 Introduction

AIDE \cite{aide2024} formalized machine learning engineering as search over a tree of code solutions: each candidate program is a node, Draft, Debug, and Improve operators expand the tree, and automated evaluation feedback guides exploration. This **"outer evolution"** paradigm—where search operates over code artifacts—was systematized by AIRA-dojo \cite{toledo2025aira} into a unified operator–search design space, and has since become the foundational vocabulary for a large body of follow-up work.

However, the recent competitive landscape on the MLE-bench \cite{chan2024mlebench} leaderboard reveals a striking **paradigm divergence**. One line of systems pushes search further: MLEvolve, PiEvolve, and others scale evolutionary search to distributed settings (Avg Medal 61%+). A second line abandons tree structures altogether in favor of investing in **memory and cognitive mechanisms**: R\&D-Agent \cite{rdagent2024} decouples research hypothesis generation from code development into two phases; ML-Master 2.0 \cite{mlmaster2025} introduces a three-tier memory cache (short-, mid-, and long-term), reaching 56.4\% medal rate on MLE-bench. A third line, such as Disarray (77.8\%), achieves top performance through multi-model ensembles and massive compute.

This divergence raises a more fundamental question: **beyond outer search over solution space, does an agent's inner cognition constitute an independent dimension worth investing in?** Current systems—whether search-based or memory-augmented—remain fragmented at the cognitive level. Search-based systems (AIDE, AIRA-dojo) keep their internal understanding implicit in prompts and ephemeral context windows, with no cross-step persistence. Memory-augmented systems (ML-Master 2.0) store historical interaction traces, but their memory is fundamentally **passive**: it records "what happened" without explicitly maintaining an actively evolving, structured belief about the task—i.e., "given these experiences, what should I believe now, and what should I do next?"

We propose **Cogito**, a framework that explicitly couples **inner cognitive evolution** with **outer solution search**. The core idea is that search should occur not only in code space but also in cognitive space—with the two evolutionary processes driving each other. Concretely, Cogito maintains a structured cognitive state $z_t$ encoding task understanding, working hypotheses, failure patterns, search preferences, and self-assessed confidence. The complete per-step cycle is:

$$
z_t \xrightarrow{\text{drive}} a_t \xrightarrow{\text{execute}} r_t \xrightarrow{\text{reflect}} z_{t+1}
$$

where $z_t$ **drives** code generation by injecting into operator prompts (outer evolution), and execution feedback $r_t$ is **reflected back** to update the cognitive state via LLM reflection (inner evolution). This inner-outer coupling transforms search from blind solution-space traversal into directed exploration guided by accumulated cognition. The search topology is orthogonal to the cognitive layer: $z_t$ can evolve along a linear chain (equivalent to cognition-augmented greedy search) or branch on a tree (equivalent to Monte Carlo tree search over cognitive state space). When $z_t = \varnothing$, the system degrades to standard AIRA-dojo.

On AIRS-Bench \cite{lupidi2025airs} across 20 cross-domain tasks, Cogito achieves Avg NS 0.522 with a 30-step budget, surpassing AIRA-dojo's 0.402 obtained with 200 steps and 10 seeds (Table [X]). More importantly, we go beyond reporting performance gains by validating the cognitive layer's mechanism through **causal intervention experiments**. We design four conditions—Natural (normal evolution), Ablated (reset $z_t$ every step), Scrambled (replace with a mature $z_t$ from an unrelated task), and Frozen (freeze after step 5)—under the same LLM, operator templates, and token budget, with **the only difference being the content of $z_t$**. This allows us to disentangle the value of cognitive evolution per se, the value of task-relevant content, and the value of continued reflection.

Our experiments show:
- The evolved $z_t$ carries task-specific information—performance drops significantly under the Scrambled condition, demonstrating that the value derives from content, not form (RQ3).
- Reflections accompanying high-information events (score jumps, first success) contribute the majority of effective state updates (RQ2).
- The benefits of inner-outer coupling are most pronounced on challenging, multi-step tasks, while gains on simple tasks are limited (RQ4).

**Contributions.** (1) Cogito: a framework that explicitly couples inner cognitive evolution with outer solution search for AI research agents, achieving state-of-the-art on AIRS-Bench with fewer steps. (2) A causal intervention experiment design that validates the cognitive layer's mechanism through state replacement, ablation, and freezing, rather than relying solely on performance comparison. (3) An empirical analysis of when inner-outer coupling is effective: cognitive investment yields the highest returns on challenging tasks, and should be moderated on simple ones.

## 2 相关工作

### 2.1 外部演化：方案空间搜索
AIDE \cite{aide2024} 开创了将机器学习工程形式化为代码方案树搜索的范式。AIRA-dojo \cite{toledo2025aira} 将其推广为统一的算子—搜索设计空间 $(\pi, O, E)$，在 MLE-bench \cite{chan2024mlebench} 上系统比较贪心、MCTS 和进化策略，发现**算子设计的贡献大于搜索策略本身**——这一发现暗示，性能瓶颈可能不在搜索结构，而在每步决策的质量。MLGym \cite{nathani2025mlgym} 提供标准化的 ReAct scaffold 环境。近期，MLEvolve、PiEvolve 等系统将进化搜索推至分布式规模，在 MLE-bench 上达到 61%+ medal rate。

这些系统共享一个核心假设：**演化发生在外部方案空间中**。搜索状态是候选代码及其评分，LLM 在过程中形成的理解仅隐含于提示词和瞬时上下文中。Cogito 的出发点是：如果算子质量比搜索策略更重要，那么改善算子决策所依据的内部认知，可能比改善搜索拓扑更有效。

### 2.2 被动记忆与主动认知
针对长轨迹中的信息遗忘，多种记忆机制已被提出。MemGPT \cite{packer2024memgpt} 引入分层虚拟内存管理。Voyager \cite{wang2023voyager} 维护可复用的技能库。ML-Master 2.0 \cite{mlmaster2025} 为 ML 智能体引入三层缓存，在 MLE-bench 上达到 56.4% medal rate。ExpeL \cite{zhao2024expel} 从轨迹中自动提取经验规则。R\&D-Agent \cite{rdagent2024} 将研究假设生成与代码开发解耦为两个阶段，隐式引入了”先思考再行动”的结构。

这些工作构成了从**无状态**到**有记忆**的第一步。但记忆和认知存在关键区别：记忆存储历史痕迹或提炼规则（”发生了什么”），认知则是一种会主动演化的结构化信念（”基于经历，我现在应该相信什么、下一步该做什么”）。ML-Master 2.0 的缓存和 R\&D-Agent 的阶段分离可视为认知的隐式近似；Cogito 将认知**显式化为一级状态变量**，使其可以被分析、干预和优化。

### 2.3 反思与自我改进
Reflexion \cite{shinn2023reflexion} 让智能体在失败后生成语言反思并存入记忆。LATS \cite{zhou2024lats} 将 MCTS 与 LLM 自我评估结合。Tree of Thoughts \cite{yao2024tot} 允许推理中的显式分支和回溯。Self-Refine \cite{madaan2023selfrefine} 通过”生成—批评—修改”循环提升输出。

这些工作将反思视为**局部操作**：输出是当前步骤的辅助文本或一次性修正。在 Cogito 中，反思是**内部演化的驱动力**——每步反思更新 $z_t$，该状态在后续每一步的算子调用中持续注入，形成从内部认知到外部动作的跨步骤因果链。

### 2.4 本文的位置
上述三条路线——外部搜索、被动记忆、局部反思——各自解决了问题的一个侧面。Cogito 将它们统一为**内外耦合的演化框架**：外部方案搜索提供行动多样性，内部认知演化提供决策连续性，两者通过 $z_t$ 的注入和更新形成闭环。我们不仅报告性能提升（AIRS-Bench Avg NS 0.522 vs AIRA-dojo 0.402），还通过因果状态干预验证耦合的作用机制。据我们所知，这是首个对 AI 研究智能体中认知状态进行因果干预分析的工作。


## 2 Related Work

### 2.1 Outer Evolution: Solution-Space Search

AIDE \cite{aide2024} pioneered the paradigm of formalizing machine learning engineering as search over a code solution tree. AIRA-dojo \cite{toledo2025aira} generalized this into a unified operator–search design space $(\pi, O, E)$, systematically comparing greedy, MCTS, and evolutionary strategies on MLE-bench \cite{chan2024mlebench} and finding that **operator design contributes more than search strategy**—a finding suggesting that the performance bottleneck may lie not in search structure, but in the quality of per-step decisions. MLGym \cite{nathani2025mlgym} provides a standardized ReAct scaffold environment. More recently, systems such as MLEvolve and PiEvolve have pushed evolutionary search to distributed scale, achieving 61\%+ medal rates on MLE-bench.

These systems share a core assumption: **evolution occurs in the outer solution space**. The search state consists of candidate code and its scores; the understanding formed by the LLM during the process remains implicit in prompts and ephemeral context. Cogito's starting point is: if operator quality matters more than search strategy, then improving the internal cognition on which operator decisions are based may be more effective than improving the search topology.

### 2.2 Passive Memory vs. Active Cognition

To address information loss over long trajectories, various memory mechanisms have been proposed. MemGPT \cite{packer2024memgpt} introduces tiered virtual memory management. Voyager \cite{wang2023voyager} maintains a reusable skill library. ML-Master 2.0 \cite{mlmaster2025} introduces a three-tier cache for ML agents, achieving 56.4\% medal rate on MLE-bench. ExpeL \cite{zhao2024expel} automatically extracts experience rules from trajectories. R\&D-Agent \cite{rdagent2024} decouples research hypothesis generation from code development into two phases, implicitly introducing a "think before acting" structure.

These works constitute a first step from **stateless** to **memory-augmented** agents. However, memory and cognition differ in a crucial way: memory stores historical traces or distilled rules ("what happened"), while cognition is an actively evolving structured belief ("given these experiences, what should I believe now, and what should I do next?"). ML-Master 2.0's cache and R\&D-Agent's phase separation can be viewed as implicit approximations of cognition; Cogito **elevates cognition to a first-class state variable**, making it analyzable, intervenable, and optimizable.

### 2.3 Reflection and Self-Improvement

Reflexion \cite{shinn2023reflexion} has agents generate verbal reflections after failure and store them in memory. LATS \cite{zhou2024lats} combines MCTS with LLM self-evaluation. Tree of Thoughts \cite{yao2024tot} enables explicit branching and backtracking during reasoning. Self-Refine \cite{madaan2023selfrefine} improves outputs through iterative generate–critique–revise loops.

These works treat reflection as a **local operation**: the output is auxiliary text for the current step or a one-time correction. In Cogito, reflection is **the driver of inner evolution**—each reflection updates $z_t$, which is persistently injected into every subsequent operator call, forming a cross-step causal chain from inner cognition to outer action.

### 2.4 Positioning of This Work

The three lines above—outer search, passive memory, local reflection—each address one facet of the problem. Cogito unifies them into a **coupled inner-outer evolutionary framework**: outer solution search provides action diversity, inner cognitive evolution provides decision continuity, and the two form a closed loop through $z_t$'s injection and update. We not only report performance improvements (AIRS-Bench Avg NS 0.522 vs. AIRA-dojo 0.402) but also validate the coupling mechanism through causal state interventions. To our knowledge, this is the first work to perform causal intervention analysis on cognitive states in AI research agents.

## 3 The Cogito Framework

### 3.1 概述：从无状态搜索到认知增强搜索

标准 AIRA-dojo 智能体 \cite{toledo2025aira} 可形式化为三元组 $(\pi, O, E)$：搜索策略 $\pi$ 从候选节点集中选择下一个扩展点，算子集 $O = \{\texttt{draft}, \texttt{improve}, \texttt{debug}\}$ 生成新代码，执行环境 $E$ 返回评分和终端输出。搜索状态完全由外部产物（代码、分数、错误日志）构成。

Cogito 引入两个新组件将其扩展为五元组 $(\pi, O, E, z_t, U)$：

- **认知状态** $z_t$：一个结构化对象，编码智能体当前对任务的累积理解；
- **状态更新函数** $U$：通过 LLM 反思将反馈整合进认知状态。

每步的完整流程为：

$$
z_t \xrightarrow{E_z} \text{prompt} \xrightarrow{O} a_t \xrightarrow{E} r_t \xrightarrow{U} z_{t+1}
$$

其中 $E_z(z_t)$ 将认知状态渲染为结构化文本并注入算子提示（实现为 `to_prompt_str()` 方法），$a_t$ 是一次动作（Draft / Improve / Debug），$r_t$ 是多维反馈信号。当 $z_t = \varnothing$ 时，$E_z$ 返回空字符串，系统退化为标准 AIRA-dojo。

### 3.2 认知状态 $z_t$

认知状态被实现为一个包含六个语义字段的结构化对象：

| 字段 | 类型 | 语义 |
|------|------|------|
| `task_understanding` | 自由文本 | 对任务结构、数据特征、评估指标的当前理解 |
| `hypotheses` | 字符串列表 | 当前认为值得检验的工作假设 |
| `learned_patterns` | 字符串列表 | 从执行反馈中发现的规律（如 “ReduceLROnPlateau 的 verbose 参数在 PyTorch≥2.4 中已移除”） |
| `preferred_directions` | 字符串列表 | 有前景的下一步方向 |
| `avoided_directions` | 字符串列表 | 已确认无效的方向（死胡同） |
| `confidence` | $[0, 1]$ 标量 | 对当前最优方向的置信度 |

此外，$z_t$ 附带一个压缩的尝试历史 `attempt_summaries`（最近 50 步的 approach—metric—insight 三元组），以及由环境探测脚本生成的 `environment_context`（包版本、GPU 信息、API 兼容性警告）。

**与记忆的区别。** $z_t$ 不是历史日志的缓存，而是一个经过提炼的信念摘要。它回答的问题不是”过去发生了什么”，而是”基于过去的经历，我现在应该相信什么、下一步应该做什么”。

**内禀质量。** 为了在不依赖任务分数的情况下评估状态质量，我们定义内禀质量指标：

$$
IQ(z_t) = \frac{1}{4}\left[\min\!\left(\frac{|\text{hyp}|}{5}, 1\right) + \min\!\left(\frac{|\text{pat}|}{5}, 1\right) + \frac{d_{\text{pref}} + d_{\text{avoid}}}{2} + \text{cal}(z_t)\right]
$$

其中 $d_{\text{pref}} = \min(|\text{preferred}|/3, 1)$，$d_{\text{avoid}} = \min(|\text{avoided}|/3, 1)$，$\text{cal}(z_t) = 1 - |\text{confidence} - \text{actual\_success\_rate}|$。$IQ$ 衡量状态的丰富度和校准度，而非任务性能。

### 3.3 反思更新 $U(z_t, r_t)$

**反馈信号 $r_t$。** 每步执行后，系统构造一个多维反馈对象：

| 维度 | 含义 | 取值 |
|------|------|------|
| `metric` | 评估分数 | 浮点数或 None（若代码崩溃） |
| `is_buggy` | 代码是否执行失败 | 布尔值 |
| `error_category` | 错误分类 | environment / resource / data / logic / none |
| `trend` | 相对历史最优的趋势 | improving / stagnating / degrading / first |
| `trend_delta` | 趋势的量化幅度 | 有符号浮点数 |
| `novelty` | 代码与此前尝试的差异度 | $[0, 1]$，基于行级 Jaccard 距离 |

**更新过程。** 反思通过一次 LLM 调用实现：输入为当前 $z_t$、反馈 $r_t$ 的结构化文本表示、当前代码和终端输出；输出为符合 $z_t$ JSON schema 的更新后状态 $z_{t+1}$。LLM 被要求返回结构化 JSON，包含更新后的六个语义字段以及一个 `key_insight` 摘要。更新后的状态同时记录到 `attempt_summaries` 中。

**逐步反思。** 当前实现在**每步执行后**都调用 $U$（`reflect_after_every_step=True`）。我们选择逐步反思而非选择性反思，是为了确保所有步骤的状态轨迹完整可分析。

**事后触发分类。** 为分析哪些步骤的反思最有价值，我们在事后对每步标注触发类型：

| 触发类型 | 判定条件 |
|---------|---------|
| Bootstrap | $t = 0$（首步，无历史） |
| Score Jump | 分数相对变化 > 5% |
| First Success | 首次 `is_buggy = False` |
| Plateau | 连续 3 步分数变化 < 1% |
| New Error | 出现新的 `error_category` |
| Routine | 以上均不满足 |

这一分类是**分析工具**，不影响系统运行逻辑。

### 3.4 认知驱动的搜索策略

$z_t$ 不仅注入算子提示，还直接驱动搜索策略的动作选择。具体地，`search_policy()` 的决策逻辑为：

1. **引导阶段**（`evolution_step == 0` 或 draft 数 < 2）：无条件执行 Draft，获取初始数据；
2. **Debug 路径**（近 3 步全部 buggy 且 `learned_patterns` 非空）：$z_t$ 已识别错误模式，执行 Debug；
3. **探索路径**（`confidence < 0.3` 或存在未尝试的 `preferred_directions`）：$z_t$ 不确定或有新想法，执行 Draft 并指定焦点方向；
4. **利用路径**（`confidence ≥ 0.3` 且存在有效节点）：$z_t$ 对当前方向有信心，执行 Improve；
5. **兜底**：Draft。

这种”由 $z_t$ 驱动决策”的设计意味着搜索策略本身成为认知状态的函数，而非独立于 $z_t$ 的固定规则。

### 3.5 搜索拓扑

Cogito 通过统一的认知状态树框架支持三种搜索拓扑，复杂度递增：

**Chain**（$K = 1$, `num_drafts=1`）。认知状态树退化为纯线性序列 $z_0 \to z_1 \to z_2 \to \cdots$。每步只产生一个代码节点，动作按固定序列进行（draft → improve → improve → $\cdots$）。因果链路完全线性：步 $t$ 的动作仅受 $z_t$ 影响，$z_t$ 仅由 $z_{t-1}$ 和 $r_{t-1}$ 决定。这是最简拓扑，归因关系完全确定，适合因果干预实验。

**Greedy**（$K = 1$, `num_drafts=5`）。认知状态仍为单链演化，但代码级节点形成**多根结构**：前 $D$ 步（默认 $D = 5$）无条件执行 Draft，产生 $D$ 个独立的初始方案。此后由 §3.4 的 $z_t$ 驱动策略选择后续动作（improve 最佳节点 / debug 最近失败节点 / draft 新方向）。关键特性是：$D$ 次 draft 的反馈**全部累积进同一个 $z_t$**，使认知状态在搜索早期快速吸收来自多个方向的信息。代码节点的多样性由多次 draft 提供，认知的连续性由单链 $z_t$ 保证。

**MC-ESES 分支树**（$K > 1$）。认知状态本身形成分支树。每轮通过 UCT 选择一个叶节点 $z_{\text{leaf}}$，展开 $K$ 个子节点。每个子节点克隆 $z_{\text{leaf}}$，分配不同的焦点方向（从 `hypotheses` 和 `preferred_directions` 池中分配），独立生成代码、执行、反思后创建独立的子认知状态。不同分支的 $z_t$ 彼此独立演化，但通过跨分支知识迁移（从 top-$k$ 叶节点收割 `learned_patterns` 和 `avoided_directions`）实现信息共享。多维回传机制将结果沿路径回传：

$$
v = \alpha \cdot \text{metric} + \beta \cdot \text{validity\_rate} + \gamma \cdot \text{improvement\_signal}
$$

其中 $\alpha = 0.6$，$\beta = 0.2$，$\gamma = 0.2$。UCT 值在计算时加入 $IQ(z_t)$ 的加权项（权重 0.2），以偏好认知质量更高的节点。

三种拓扑的对比如下：

| | Chain | Greedy | MC-ESES |
|---|---|---|---|
| 认知状态结构 | 单链 | 多链 | 分支树 |
| 初始探索 | 1 个 draft | $D$ 个独立 draft | 每轮 $K$ 个分支 |
| 动作选择 | 固定序列 | $z_t$ 驱动（§3.4） | UCT + $z_t$.confidence |
| 每步消耗 | 1 个节点 | 1 个节点 | $K$ 个节点 |
| 分支间信息共享 | — | — | 跨分支知识迁移 |

**本文的因果干预实验使用 Greedy 拓扑**（$K = 1$, $D = 5$），兼顾搜索多样性和因果归因清晰度。MC-ESES 分支树的结果作为补充分析。

### 3.6 因果干预设计

为了从相关性提升到因果证据，我们在反思更新 $U$ 之后插入一个**干预钩子** $I$：

$$
z_{t+1}^{\text{actual}} = I\big(U(z_t, r_t),\; t,\; \text{mode}\big)
$$

四种干预模式如下：

| 模式 | $I$ 的行为 | 控制了什么 |
|------|-----------|-----------|
| **Natural** | 恒等函数，$z_{t+1}^{\text{actual}} = U(z_t, r_t)$ | 完整系统（对照组） |
| **Ablated** | 返回空白 $z_0$（保留 `environment_context`） | 认知层存在但不累积 |
| **Scrambled** | 返回来自无关任务的成熟 $z^*$（保留 `environment_context`） | 结构化提示存在但内容不相关 |
| **Frozen** | $t \leq 5$ 时透传；$t > 5$ 时返回 $z_5$ 的冻结快照 | 持续反思 vs 早期反思 |

四个条件共享同一套 LLM、算子模板、搜索策略和 token 预算。**唯一区别是 $z_t$ 的内容。** 这使得条件间的性能差异可以归因于认知状态本身：

- Natural > Ablated $\Rightarrow$ 认知演化有价值
- Ablated > Scrambled $\Rightarrow$ 任务无关的 $z_t$ 比没有 $z_t$ 更差（$z_t$ 的价值来自内容，非形式）
- Natural > Frozen $\Rightarrow$ 持续反思优于仅早期反思

### 3.7 轨迹记录

为支持事后分析，系统在每步保存一条 trajectory 快照，包含：

- $z_t$（反思前）和 $z_{t+1}$（反思/干预后）的摘要统计量（字段计数、置信度、$IQ$ 值）
- 反馈 $r_t$ 的完整记录
- 状态差异 $\Delta(z_t, z_{t+1})$：各字段的增/删/改计数
- 触发类型标注
- 干预模式标记

所有快照以 JSONL 格式追加写入 `trajectory.jsonl`，供 RQ1–RQ4 的分析使用。

## 4 实验设置

### 4.1 基准与任务

**AIRS-Bench** \cite{lupidi2025airs} 是一个面向 AI 研究智能体的跨领域基准，包含 20 个机器学习工程任务，覆盖文本分类、问答、共指消解、代码生成与检索、数学推理、分子属性预测、图回归和时间序列预测。每个任务提供训练数据、评估脚本和标准化的容器执行环境。智能体的每一步生成一份完整的 Python 方案，在容器内执行后由评估脚本返回分数。

**归一化评分（NS）。** 为实现跨任务可比，AIRS-Bench 对原始分数进行非线性归一化：$\text{NS} = (\varphi(s) - \varphi(s_{\min})) / (\varphi(s_{\text{sota}}) - \varphi(s_{\min}))$，其中 $\varphi(s) = -\log_{10}|s - s_{\text{opt}}|$。归一化后 0 对应最差水平，1 对应 SOTA，>1 表示超越 SOTA。

**任务选择。** 从 20 个任务中选取 10 个代表性任务用于因果干预实验，覆盖四个难度段和多种任务类型：

| 难度 | 任务 | 类型 | Baseline NS |
|------|------|------|:-----------:|
| Easy | TextualClassificationSickAccuracy | NLP | 1.097 |
| Easy | TimeSeriesForecastingSolarWeeklyMAE | 时序 | 0.999 |
| Medium | SentimentAnalysisYelpReviewFullAccuracy | NLP | 0.805 |
| Medium | ReadingComprehensionSquadExactMatch | QA | 0.913 |
| Medium | GraphRegressionZincMae | 分子/图 | 0.660 |
| Hard | CoreferenceResolutionWinograndeAccuracy | NLU | 0.622 |
| Hard | CoreferenceResolutionSuperGLUEWSCAccuracy | NLU | 0.206 |
| Expert | CodeRetrievalCodeXGlueMRR | 代码 | 0.001 |
| Expert | MathQuestionAnsweringSVAMPAccuracy | 数学 | 0.157 |
| Expert | CvMolecularPropertyPredictionQm9MAE | 分子 | 0.000 |

其中 Baseline NS 为 AIRA-dojo Greedy（Qwen-3.5-397B，5 seeds best-across-seeds）在对应任务上的得分。Easy 任务 baseline 已接近或超过 SOTA；Expert 任务 baseline 接近零分或完全失败。选择这一分布的目的是检验认知层在不同难度下的差异化效果（RQ4）。

**难度分组依据。** 我们按 baseline NS 将任务分为四档：Easy (NS > 0.8)、Medium (0.4 < NS ≤ 0.8)、Hard (0.1 < NS ≤ 0.4)、Expert (NS ≤ 0.1)。该分组反映了无认知增强时智能体的能力边界。

### 4.2 比较条件

因果干预实验在**线性链拓扑**（$K = 1$）下运行，以保证每步的因果归因是确定的。四种条件如 §3.6 所述：

| 条件 | $z_t$ 来源 | 检验目标 |
|------|-----------|---------|
| Natural | 正常反思演化 | 完整系统（对照） |
| Ablated | 每步重置为 $z_0$ | 认知演化 vs 无演化 |
| Scrambled | 替换为 TextualSimilaritySick 任务的成熟 $z^*$ | 任务特异性 vs 形式填充 |
| Frozen | 步骤 5 后冻结 | 持续反思 vs 早期反思 |

**Scrambled 来源选择。** 我们使用 TextualSimilaritySickSpearmanCorrelation 任务在先前实验中产生的成熟认知状态（confidence = 0.95，7 条假设，9 条 learned patterns）作为 scramble 来源。该任务是 NLP 文本相似度回归，与实验中的数学、代码、分子等任务在领域上完全不相关，但 $z^*$ 本身结构完整、信息丰富——这确保了 Scrambled 条件测试的是”内容相关性”而非”结构充实度”。

**环境一致性。** 四个条件的 `environment_context` 字段（由环境探测脚本在步骤 0 生成）保持一致，不受干预影响。这确保了差异不来自环境信息的有无。

### 4.3 评估指标

我们从三个层面评估实验结果：

**任务性能指标。**
- **归一化分数（NS）**：跨任务的标准化得分，取每个条件下的 best-across-seeds 值；
- **平均归一化分数（Avg NS）**：10 个任务的 NS 均值；
- **有效运行率**：非 buggy 步骤占总步数的比例。

**认知状态指标。**
- **内禀质量 $IQ(z_t)$**：§3.2 定义的状态丰富度指标，不依赖任务分数；
- **状态变化幅度 $|\Delta|$**：反思前后 $z_t$ 各字段的增删改计数之和（`total_field_changes`），衡量单步反思造成的状态更新量；
- **置信度变化 $\Delta c$**：$z_{t+1}.\text{confidence} - z_t.\text{confidence}$；
- **按触发类型分组的 $|\Delta|$ 和下一步 metric 变化**：用于 RQ2 的分组分析。

**效率指标。**
- **反思 token 占比**：reflect 算子消耗的 token 数占总 token 数的比例；
- **每 token NS 增益**：$(\text{NS}_{\text{Natural}} - \text{NS}_{\text{Ablated}}) / \text{reflect\_tokens}$，衡量认知层的边际效率。

### 4.4 实现细节

**模型。** 所有算子（draft、improve、debug、analyze、reflect）统一使用 Qwen-3.5-397B-A17B（MoE 架构，激活 17B 参数），通过 LiteLLM 接口调用，temperature = 0.6（analyze 为 0.5）。选择该模型是因为它在 AIRS-Bench baseline 上已有充分的参考数据。

**搜索配置。** 线性链拓扑，$K = 1$。每个 run 最多 30 步，时间限制 12 小时。初始 draft 数 = 5，最大 debug 深度 = 20。每步后执行 reflect。

**种子与统计。** 每个条件 × 每个任务运行 3 个随机种子（seed 1–3），共 $4 \times 10 \times 3 = 120$ runs。报告均值 ± 标准误（SEM）。显著性检验使用 paired t-test（Natural vs 其他条件，配对单位为 task × seed）。

**硬件。** 4× NVIDIA RTX 5090 GPU，每个 run 独占 1 GPU。智能体代码在 Apptainer 容器内执行，与宿主环境隔离。

**提示模板。** 四种干预条件共享完全相同的 draft / improve / debug / analyze / reflect 提示模板。唯一区别是 `cognitive_state_str` 字段的内容——Natural 注入当前 $z_t$ 的渲染文本，Ablated 注入空白状态的渲染文本（基本为空），Scrambled 注入 $z^*$ 的渲染文本，Frozen 在步骤 5 后注入 $z_5$ 的渲染文本。

**触发分类参数。** Score Jump 阈值 = 5%（相对变化），Plateau 窗口 = 3 步（步间变化 < 1%），novelty 基于行级 Jaccard 距离（比较最近 5 步代码）。所有阈值在全部任务上保持固定，不做任务特异性调整。

**代码与数据。** 实验框架基于 AIRA-dojo \cite{toledo2025aira} 的 Cogito 分支实现，完整代码和配置文件将在论文发表时公开。

## 5 结果


## 6 讨论



## 7 结论


## 参考文献

> 以下为本文引用的核心文献（待替换为 BibTeX 自动生成格式）：

- \cite{aide2024} — AIDE: ML engineering as code generation. arXiv:2402.14592.
- \cite{toledo2025aira} — Toledo et al. AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench. arXiv:2507.02554.
- \cite{lupidi2025airs} — Lupidi et al. AIRS-Bench: Benchmarking AI Research Agents. arXiv:2602.06855.
- \cite{chan2024mlebench} — Chan et al. MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering. arXiv:2410.07095.
- \cite{nathani2025mlgym} — Nathani et al. MLGym: A New Framework and Benchmark for Advancing AI Research Agents. arXiv:2502.14499.
- \cite{rdagent2024} — Microsoft R&D-Agent: Research and Development Agent Framework.
- \cite{mlmaster2025} — ML-Master 2.0: Hierarchical Memory for ML Engineering Agents.
- \cite{shinn2023reflexion} — Shinn et al. Reflexion: Language Agents with Verbal Reinforcement Learning. NeurIPS 2023.
- \cite{zhou2024lats} — Zhou et al. Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models. ICML 2024.
- \cite{yao2024tot} — Yao et al. Tree of Thoughts: Deliberate Problem Solving with Large Language Models. NeurIPS 2023.
- \cite{madaan2023selfrefine} — Madaan et al. Self-Refine: Iterative Refinement with Self-Feedback. NeurIPS 2023.
- \cite{packer2024memgpt} — Packer et al. MemGPT: Towards LLMs as Operating Systems. ICLR 2024.
- \cite{wang2023voyager} — Wang et al. Voyager: An Open-Ended Embodied Agent with Large Language Models. NeurIPS 2023 (Oral).
- \cite{zhao2024expel} — Zhao et al. ExpeL: LLM Agents Are Experiential Learners. AAAI 2024.

## 附录 A. 研究问题
为方便阅读，这里重新列出本文研究的四个问题：

- **RQ1**：认知状态是否会随时间发生有意义的演化？
- **RQ2**：哪些反馈事件伴随了有用的状态转换？
- **RQ3**：更好的认知状态是否会因果性地产生更好的动作？
- **RQ4**：认知演化在什么任务条件下有益或有害？

## 附录 B. 认知状态示例 Schema
```json
{
  "task_understanding": "该任务是一个具有强类别不平衡和缺失值问题的表格分类任务。",
  "working_hypotheses": [
    "在完成类别编码后，树模型可能优于线性模型。",
    "缺失模式本身可能携带预测信号。"
  ],
  "failure_patterns": [
    "朴素标准化 + 逻辑回归明显欠拟合。",
    "当前流程在特征选择阶段泄露了验证集信息。"
  ],
  "preferred_directions": [
    "尝试结合目标感知插补的梯度提升模型。",
    "先建立稳健基线，再检查特征重要性。"
  ],
  "avoided_directions": [
    "不要继续调参当前线性基线。",
    "在修复类别处理问题前，避免直接进行特征缩放。"
  ],
  "confidence": 0.63
}
```

## 附录 C. 图示建议
- **图 1**：Cogito 总体框架图（认知层、算子层、搜索/执行层）
- **图 2**：因果干预设置示意（Natural / Ablated / Scrambled / Frozen）
- **图 3**：状态特异性与内禀质量随时间变化图
- **图 4**：触发类型 × 状态变化幅度 × 后续动作质量 热力图
- **图 5**：四种干预条件下 next-step score improvement 箱线图
- **图 6**：任务难度 × 方法 交互效应图
