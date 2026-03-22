# MiroFish 项目深度分析报告

> **分析日期**: 2026-03-19
> **项目来源**: `upstream/MiroFish/`（GitHub: 666ghj/MiroFish，盛大集团孵化）
> **定位**: 基于多智能体技术的新一代 AI 预测引擎

---

## 1. 一句话定位

**MiroFish 是一个多 Agent 社会仿真预测引擎** — 输入种子材料（新闻/报告/小说），自动构建"平行数字世界"，让成百上千个 AI 角色在模拟社交媒体上自由互动，通过群体行为涌现推演未来走向。

口号：*"简洁通用的群体智能引擎，预测万物"*

---

## 2. 核心架构：5 步流水线

```
输入                       MiroFish Pipeline                              输出
──────     ───────────────────────────────────────────────────     ────────────
种子文档  →  ① 本体生成  →  ② 图谱构建  →  ③ 环境搭建  →  ④ 多Agent模拟  →  ⑤ 预测报告
(PDF/MD/     (LLM 抽取      (Zep Cloud     (Profile +      (OASIS引擎       + 可交互
 TXT)         实体关系)      GraphRAG)      参数自动配)     双平台并行)      数字世界
```

### 工作流总览

| 阶段 | 核心服务 | 输入 | 输出 |
|------|---------|------|------|
| ① 本体生成 | `ontology_generator.py` | 种子文档 + 预测需求 | 实体/关系类型 Schema |
| ② 图谱构建 | `graph_builder.py` | 文本 + 本体定义 | Zep 知识图谱 |
| ③ 环境搭建 | `simulation_manager.py` | 图谱实体 + 需求描述 | Agent Profile + 模拟参数 |
| ④ 模拟运行 | `simulation_runner.py` | 配置文件 + OASIS 引擎 | Agent 动作日志 + 图谱更新 |
| ⑤ 报告生成 | `report_agent.py` | 模拟结果 + 图谱 | 预测报告 + 交互对话 |

---

## 3. 各阶段详解

### 3.1 本体生成（Ontology Generation）

**源码**: `backend/app/services/ontology_generator.py`

LLM 分析种子文档，自动抽取适合社会仿真的实体和关系类型定义。

```
输入: 用户上传的种子文档 + 自然语言预测需求
     例："分析武大舆情事件未来一周走向"

LLM 输出 Schema:
{
  "entity_types": [
    { "name": "PublicFigure", "attributes": ["role", "influence_level", ...] },
    { "name": "MediaOrganization", "attributes": ["media_type", "reach", ...] },
    { "name": "GovernmentBody", ... },
    ...  // 固定 10 个实体类型
  ],
  "edge_types": [
    { "name": "CRITICIZES", "source_targets": [{"source": "PublicFigure", "target": "Organization"}] },
    { "name": "REPORTS_ON", ... },
    ...
  ]
}
```

**设计约束**:
- 实体必须是**真实可在社媒上发声的主体**（人/机构/媒体）
- 不允许抽象概念（"舆论"、"情绪"）作为实体
- 固定生成 10 个实体类型，确保覆盖度

### 3.2 知识图谱构建（Graph Building）

**源码**: `backend/app/services/graph_builder.py`

使用 Zep Cloud 的 GraphRAG 能力，从种子文本中自动构建知识图谱。

```
处理流程:
  1. 文本分块（500 字/块，50 字重叠）
  2. 分批发送到 Zep Cloud API
  3. Zep 自动执行:
     - NER（命名实体识别）
     - 关系抽取
     - 向量嵌入
  4. 构建 GraphRAG 知识图谱

输出示例:
  武大校长 ──[管理]──→ 武汉大学
  武汉大学 ──[被报道]──→ 澎湃新闻
  当事人A ──[批评]──→ 武大校长
```

**技术细节**:
- 使用 `EpisodeData` 批量发送文本到 Zep
- 异步等待 Zep 处理完成（轮询 episode.processed 状态）
- 支持动态 Ontology（基于 Pydantic v2 动态类创建）
- 处理 Zep 保留字段名冲突（`uuid`, `name` 等自动加 `entity_` 前缀）

### 3.3 环境搭建（Environment Setup）

**源码**: `simulation_manager.py` + `oasis_profile_generator.py` + `simulation_config_generator.py`

自动化完成三个子任务：

#### 3.3a 实体 → Agent Profile

从 Zep 图谱读取实体节点，用 LLM 生成详细人设。

```
图谱节点: "张三 (PublicFigure)"
    ↓ LLM 增强 + Zep 检索二次丰富
Agent Profile:
    name: 张三
    bio: "武汉大学教授，研究方向计算机科学"
    persona: "性格谨慎，关注学术公正，倾向理性表达..."
    mbti: INTJ
    age: 45
    gender: male
    profession: 教授
    interested_topics: ["学术诚信", "教育改革"]
    stance: neutral
    influence_weight: 0.8
    karma: 1500
```

**Profile 数据结构** (`OasisAgentProfile`):

| 字段 | 说明 | 生成方式 |
|------|------|---------|
| `persona` | 详细人格描述 | LLM 根据图谱信息生成 |
| `mbti` | 性格类型 | LLM 推断 |
| `stance` | 立场（supportive/opposing/neutral/observer） | LLM 分析 |
| `influence_weight` | 影响力权重 | LLM 评估 |
| `activity_level` | 活跃度 0.0-1.0 | LLM 配置 |
| `sentiment_bias` | 情感倾向 -1.0 到 1.0 | LLM 配置 |

**输出格式**:
- Twitter: CSV 格式（OASIS 要求）
- Reddit: JSON 格式

#### 3.3b LLM 智能生成模拟参数

```python
# 模拟参数自动配置项
SimulationParameters:
  - 模拟轮数（如舆情事件：72轮 = 72小时）
  - 每个 Agent 的活跃时段（遵循中国作息时间）
  - 发帖频率 / 评论频率
  - 事件注入时间点
  - 平台选择（Twitter / Reddit / 双平台）

# 中国作息时间模型（内置）
CHINA_TIMEZONE_CONFIG:
  dead_hours: [0,1,2,3,4,5]      → activity × 0.05
  morning_hours: [6,7,8]          → activity × 0.4
  work_hours: [9-18]              → activity × 0.7
  peak_hours: [19,20,21,22]       → activity × 1.5
  night_hours: [23]               → activity × 0.5
```

#### 3.3c 生成配置文件

```
模拟目录结构:
  sim_xxx/
  ├── twitter_profiles.csv       # Twitter Agent 配置
  ├── reddit_profiles.json       # Reddit Agent 配置
  ├── simulation_config.json     # 模拟参数
  └── state.json                 # 模拟状态
```

### 3.4 模拟运行（Simulation Execution）

**源码**: `simulation_runner.py` + `backend/scripts/run_parallel_simulation.py`

基于 **OASIS**（CAMEL-AI 开源）仿真引擎，双平台并行运行。

```
  Twitter 模拟 ──┐
                 ├── 并行运行 N 轮（每轮 = 1 小时模拟时间）
  Reddit 模拟 ──┘

  每轮每个 Agent 的决策循环:
    1. 感知: 查看当前 feed（帖子流/评论流）
    2. 思考: LLM 根据 persona + 长期记忆 + 当前 feed → 决定行为
    3. 行动: 执行一个动作
    4. 记忆: 动作结果写回 Zep 图谱
```

**可用动作**:

| 平台 | 动作列表 |
|------|---------|
| Twitter | `CREATE_POST`, `LIKE_POST`, `REPOST`, `FOLLOW`, `DO_NOTHING`, `QUOTE_POST` |
| Reddit | `CREATE_POST`, `CREATE_COMMENT`, `LIKE_POST`, `DISLIKE_POST`, `LIKE_COMMENT`, `DISLIKE_COMMENT`, `SEARCH_POSTS`, `SEARCH_USER`, `TREND`, `REFRESH`, `FOLLOW`, `MUTE`, `DO_NOTHING` |

**记忆动态更新**（`zep_graph_memory_updater.py`）:

```
Agent 动作 → 自然语言描述 → 写入 Zep Graph

例:
  "张三: 发布了一条帖子：「对此事深表关切，希望校方给出解释」"
  "李四: 点赞了张三的帖子：「对此事深表关切...」"
  "新闻媒体A: 转发了张三的帖子并评论：「事件持续发酵」"

→ Zep 自动更新图谱中的关系和节点属性
→ 下一轮 Agent 检索时能"记住"之前的互动
```

这是 MiroFish 实现**群体涌现**的核心机制：Agent A 的行为影响 Agent B 的 feed，B 的反应又影响 C，形成信息传播链和舆论演化。

### 3.5 报告生成与深度交互

**源码**: `report_agent.py` + `zep_tools.py`

#### Report Agent（ReACT 模式）

```
报告生成流程:
  1. 规划: LLM 生成报告大纲（章节结构）
  2. 逐章生成:
     for each section:
       a. Think: 思考需要什么信息
       b. Act: 调用检索工具查询 Zep 图谱
       c. Observe: 分析检索结果
       d. Reflect: 反思是否充分（最多 N 轮反思）
       e. Write: 生成该章节内容
  3. 输出: 完整预测报告（Markdown）
```

#### 检索工具集

| 工具 | 功能 | 说明 |
|------|------|------|
| **InsightForge** | 深度洞察检索 | 自动生成子问题，多维度混合检索，最强大 |
| **PanoramaSearch** | 广度搜索 | 获取全貌，包括已过期内容 |
| **QuickSearch** | 快速检索 | 轻量级关键词搜索 |
| **Interview** | Agent 采访 | 与模拟世界中的特定 Agent 对话 |

#### 深度交互

模拟完成后，用户可以：
- **与任意 Agent 对话**: 直接与模拟世界中的角色交谈，了解其"想法"
- **与 Report Agent 追问**: 基于报告内容进行深度问答
- **注入新变量**: 修改条件重新推演

---

## 4. 技术栈

| 层 | 技术 | 说明 |
|---|---|---|
| 前端 | Vue 3 + Vite | 5 步向导式 UI，实时展示模拟进度 |
| 后端 | Flask (Python 3.11+) | RESTful API，3 个蓝图（graph/simulation/report） |
| 知识图谱 | Zep Cloud | GraphRAG + 长期记忆 + 实体关系抽取 |
| 仿真引擎 | OASIS (CAMEL-AI) | 社交媒体仿真，Agent 行为驱动 |
| LLM | OpenAI SDK 兼容格式 | 推荐 qwen-plus（阿里百炼） |
| 包管理 | uv | Python 快速包管理 |
| 部署 | Docker / docker-compose | 前端 :3000 + 后端 :5001 |

### 关键外部依赖

| 依赖 | 角色 | 不可替代性 |
|------|------|-----------|
| **Zep Cloud** | 知识图谱存储 + GraphRAG 检索 + 记忆管理 | 核心，不可替代 |
| **OASIS** | 社交媒体仿真引擎 | 核心，可替代但成本高 |
| **LLM API** | 驱动所有智能决策 | 核心，消耗量大 |

---

## 5. 代码结构

```
MiroFish/
├── backend/
│   ├── run.py                                # Flask 启动入口
│   ├── app/
│   │   ├── __init__.py                       # Flask 应用工厂
│   │   ├── config.py                         # 配置管理（.env 加载）
│   │   ├── api/
│   │   │   ├── graph.py                      # 图谱构建 API
│   │   │   ├── simulation.py                 # 模拟管理 API
│   │   │   └── report.py                     # 报告生成 API
│   │   ├── services/
│   │   │   ├── ontology_generator.py         # ① 本体生成（LLM）
│   │   │   ├── graph_builder.py              # ② 图谱构建（Zep）
│   │   │   ├── text_processor.py             # 文本分块处理
│   │   │   ├── zep_entity_reader.py          # 从 Zep 读取实体
│   │   │   ├── oasis_profile_generator.py    # ③a Agent人设生成（LLM）
│   │   │   ├── simulation_config_generator.py# ③b 模拟参数生成（LLM）
│   │   │   ├── simulation_manager.py         # ③ 环境搭建总控
│   │   │   ├── simulation_runner.py          # ④ 模拟运行器
│   │   │   ├── simulation_ipc.py             # 模拟进程间通信
│   │   │   ├── zep_graph_memory_updater.py   # ④ 记忆动态回写
│   │   │   ├── report_agent.py               # ⑤ 报告生成（ReACT）
│   │   │   └── zep_tools.py                  # ⑤ 检索工具集
│   │   ├── models/
│   │   │   ├── project.py                    # 项目数据模型
│   │   │   └── task.py                       # 异步任务管理
│   │   └── utils/
│   │       ├── llm_client.py                 # LLM 调用封装
│   │       ├── file_parser.py                # 文件解析
│   │       ├── logger.py                     # 日志
│   │       ├── retry.py                      # 重试逻辑
│   │       └── zep_paging.py                 # Zep 分页查询
│   └── scripts/
│       ├── run_parallel_simulation.py        # 双平台并行模拟脚本
│       ├── run_twitter_simulation.py         # Twitter 单平台模拟
│       ├── run_reddit_simulation.py          # Reddit 单平台模拟
│       └── action_logger.py                  # Agent 动作日志记录
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── Step1GraphBuild.vue           # ① 图谱构建 UI
│       │   ├── Step2EnvSetup.vue             # ③ 环境搭建 UI
│       │   ├── Step3Simulation.vue           # ④ 模拟运行 UI
│       │   ├── Step4Report.vue               # ⑤ 报告查看 UI
│       │   ├── Step5Interaction.vue          # ⑤ 深度交互 UI
│       │   ├── GraphPanel.vue                # 图谱可视化
│       │   └── HistoryDatabase.vue           # 历史记录
│       └── views/
│           ├── Home.vue                      # 首页
│           ├── MainView.vue                  # 主流程视图
│           ├── SimulationRunView.vue         # 模拟运行实时视图
│           ├── ReportView.vue                # 报告视图
│           └── InteractionView.vue           # 交互视图
├── docker-compose.yml
├── Dockerfile
└── package.json                              # npm scripts（setup/dev/build）
```

---

## 6. 与 research_society 的关联分析

### 定位对比

| 维度 | MiroFish | research_society (aira-dojo + airs-bench) |
|------|----------|------------------------------------------|
| **核心目标** | 用多 Agent 仿真**预测现实事件走向** | 用单 Agent **自主完成 ML 研究任务** |
| **Agent 数量** | 成百上千个，各有人格/立场/记忆 | 1 个 Agent（研究员），写代码解 ML 问题 |
| **Agent 行为** | 社交行为（发帖/点赞/转发/评论） | 研究行为（写代码/调试/提交） |
| **LLM 用途** | 驱动 Agent 的社交行为决策 | 驱动 Agent 的代码生成和调试 |
| **搜索策略** | 无搜索（Agent 自由演化，观察涌现） | Greedy / MCTS / Evo 搜索最优方案 |
| **知识管理** | Zep GraphRAG（图谱 + 长期记忆） | 无持久化记忆（每步独立） |
| **评估方式** | 质性分析（预测报告） | 量化 benchmark（MAE/Accuracy/ANS） |
| **交互模式** | 用户可注入变量、采访 Agent | 全自动，无人干预 |

### 三种连接方案

#### 方案 A：记忆系统移植（工程层 — 最实际）

**问题**: aira-dojo 的 Agent 没有跨步记忆，导致反复犯同样的错误（buggy 率 ~90%）。

**方案**: 将 MiroFish 的 Zep 记忆机制引入 aira-dojo。

```
当前 aira-dojo:
  Step 5: KeyError 'question'（第 1 次）
  Step 6: KeyError 'question'（第 2 次）  ← 完全不记得之前失败过
  ...
  Step 20: KeyError 'question'（第 11 次）

引入 Zep 记忆后:
  Step 5: KeyError 'question'
    → 写入 Zep: "列名只有 target 和 label_target，不存在 question"
  Step 6: 检索 Zep → 召回失败经验 → 直接用正确列名
    → 不再重复犯错
```

**实现路径**: aira-dojo 已有 `memory_op` 接口（draft/debug/improve 算子均支持），只需接入 Zep GraphRAG 替换当前简单实现。

**预期收益**: buggy 率从 ~90% 降至 ~50%，有效步数翻倍。

#### 方案 B：多 Agent 协作研究（架构层 — 最有研究价值）

**思路**: 借鉴 MiroFish 的多 Agent 架构，构建 "AI Research Society"。

```
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │  数据工程师    │  │  模型专家      │  │  调参师        │
  │  Agent        │  │  Agent        │  │  Agent        │
  │  "擅长数据清洗 │  │  "深度学习架构  │  │  "超参数优化    │
  │   特征工程"   │  │   设计经验丰富" │  │   经验丰富"    │
  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
         │                 │                 │
         └────────┬────────┴────────┬────────┘
                  │   讨论 / 辩论     │
                  ▼                  ▼
         ┌────────────────────────────────┐
         │  共享知识图谱（Zep GraphRAG）    │
         └────────────────┬───────────────┘
                          ▼
                 aira-dojo 执行框架
                 (代码执行 + 评分)
```

**机制**:
1. **讨论阶段**: 多个 Agent 角色分析任务、数据、方案（借鉴 MiroFish 社交互动）
2. **执行阶段**: 将讨论共识转化为 Draft，用搜索策略优化（借鉴 aira-dojo）
3. **反思阶段**: 失败后多 Agent 联合诊断（借鉴 MiroFish Report Agent 的 ReACT 模式）

**学术价值**: "Society of Mind" 在 ML 研究场景的实现，相关工作包括 MIT Society of Mind、Google LLM Debate、Microsoft AutoGen。

#### 方案 C：科研社区演化仿真（应用层 — 最有想象力）

用 MiroFish 仿真一个 AI 科研社区：
- Agent = 不同方向的研究者（CV / NLP / RL / Theory）
- 社交平台 = 学术论坛 / ArXiv 讨论区
- 推演问题 = "Scaling Law 遇到瓶颈后研究方向如何转变？"

### 方案推荐

| 方案 | 难度 | 研究价值 | 工程价值 | 建议优先级 |
|------|------|---------|---------|-----------|
| A. 记忆系统移植 | 低 | 低 | 高 | **P0 — 先做** |
| B. 多Agent协作研究 | 中 | **高（可发论文）** | 中 | **P1 — 核心方向** |
| C. 科研社区仿真 | 低 | 中 | 低 | P2 — 演示场景 |

---

## 7. 演示案例

MiroFish 已有的演示场景：

| 场景 | 类型 | 说明 |
|------|------|------|
| 武汉大学舆情推演 | 舆情预测 | 输入舆情报告，推演事件走向和各方反应 |
| 《红楼梦》失传结局推演 | 文学创作 | 输入前 80 回，预测后续剧情发展 |
| 金融方向推演（规划中） | 金融预测 | 预测市场事件后的投资者行为和价格走势 |
| 时政要闻推演（规划中） | 政策分析 | 预测政策发布后的社会反应 |

---

## 附录：关键配置项

### 环境变量（.env）

```env
# LLM API（OpenAI SDK 格式）
LLM_API_KEY=your_key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL_NAME=qwen-plus

# Zep Cloud（知识图谱 + 记忆）
ZEP_API_KEY=your_zep_key

# 模拟参数
OASIS_DEFAULT_MAX_ROUNDS=10
REPORT_AGENT_MAX_TOOL_CALLS=5
REPORT_AGENT_MAX_REFLECTION_ROUNDS=2
```

### API 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/graph/build` | POST | 构建知识图谱 |
| `/api/graph/{id}/data` | GET | 获取图谱数据 |
| `/api/simulation/create` | POST | 创建模拟 |
| `/api/simulation/{id}/prepare` | POST | 准备模拟环境 |
| `/api/simulation/{id}/start` | POST | 启动模拟 |
| `/api/simulation/{id}/status` | GET | 查询模拟状态 |
| `/api/report/generate` | POST | 生成预测报告 |
| `/api/report/{id}/chat` | POST | 与 Report Agent 对话 |
