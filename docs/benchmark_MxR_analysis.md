# Benchmark × <F, E, Q> × R 框架适配性分析

## 框架定义

论文将 Memory Agent 形式化为 **A = <L, M, R>**，其中 M 进一步分解为 **<F, E, Q>**：

| 算子 | 含义 | 层级 |
|---|---|---|
| **F** (Formation) | 原始数据 → 记忆结构 | F1 切片 → F2 抽取事实 → F3 关系/图 → F4 多层异构 |
| **E** (Evolution) | 记忆随时间变化 | E1 只增不减 → E2 缓存置换 → E3 拓扑融合 → E4 显式修补 |
| **Q** (Retrieval) | 从记忆中找信息 | Q1 相似度 → Q2 图遍历 → Q3 生成/路由 |
| **R** (Reasoning) | 使用记忆的推理范式 | R_Direct → R_Iter → R_Plan |

核心目标：在 benchmark 上做因子分析，揭示 F, E, Q, R 各维度的独立贡献和交互效应。

---

## Benchmark 总览

共 5 个 benchmark：3 个已开源 clone + MAB 已跑 + OfficeMem-Eval（自建，基于 q2 数据）。

---

## 1. MemoryAgentBench (MAB)

**论文**：Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions (ICLR 2026 under review)
**数据位置**：`/Users/bytedance/proj/memoRaxis/MemoryAgentBench/`
**状态**：已跑完 9 个系统

### 数据结构

将长文档拆成多轮对话 chunks，逐步 ingest，最后回答问题。统一格式：
```
context (长文本) → chunks c1,c2,...,cn → questions q1,...,qm → answers a1,...,am
```

### 四类任务

| 任务 | 实例数 | Context 规模 | 问题数/实例 | 核心能力 |
|---|---|---|---|---|
| **AR** (Accurate Retrieval) | 22 | 200K~1.9M chars | 100 | 精确片段检索（SH-QA, MH-QA, LongMemEval, EventQA） |
| **TTL** (Test-Time Learning) | 6 | 400K~5.7M chars | 100-200 | 从示例学习分类/推荐（BANKING77, CLINC150, 电影推荐） |
| **LRU** (Long Range Understanding) | 110 | 380K~1.7M chars | 1 | 小说摘要 + 侦探QA，需全局理解 |
| **CR** (Conflict Resolution) | 8 | 26K~262K chars | 100 | 事实冲突判断（基于 MQUAKE 反事实编辑） |

### <F, E, Q> × R 覆盖分析

| 维度 | 覆盖 | 说明 |
|---|---|---|
| **F** | ✅ | 不同系统用不同 F（chunking vs 事实抽取 vs 图构建），AR 任务对 F 敏感 |
| **E** | ⚠️ 仅 CR | CR 任务用反事实编辑模拟冲突，测 E4（显式修补）。其余三类任务无状态变化，E1/E2/E3 无法区分 |
| **Q** | ✅ | AR 直接测 Q 质量；不同系统的 Q 差异在 AR 上表现明显 |
| **R** | ❌ 未设计 | MAB 原始框架不区分推理范式，统一用 one-shot QA |

### 关键特点
- **数据量大**：AR 单实例 100K+ tokens context + 100 questions，统计意义强
- **CR 是唯一有冲突数据的任务**，但仅限 E4（冲突覆盖），不测 E2（遗忘）/E3（融合）
- **TTL/LRU 本质上测长上下文能力**，对传统 RAG 系统不友好（因为需要全局理解）

### ⚠️ "增量 ingest" 名不副实

MAB 论文声称 chunks 逐个注入模拟增量积累，但**实际跑法是所有 chunks 一股脑灌入 memory，然后才问问题**：
- 对 E1 系统（RAPTOR/HippoRAG）：灌入顺序完全不影响最终 memory 状态
- 对 E3/E4 系统（Mem0/Zep）：AR/TTL/LRU 数据无冲突，顺序也无所谓
- CR 有冲突，但一次性灌入时系统看到的是"一堆矛盾文本"，**不是"先 A 后 B"的时间线演化**
- 实质退化为：**长文档 → chunk → 全量存入向量库 → QA**，和普通 RAG 无本质区别

### 在我们框架中的定位
- **AR**：主测 F×Q（chunk 策略 × 检索质量），可扩展 R 轴
- **CR**：唯一有冲突数据，但因 ingest 方式限制，仅测"面对矛盾文本能否选对"，不测真正的时间线演化
- **TTL/LRU**：测 long-context 能力，不适合 M×R 因子分析
- **E 覆盖实质为零**：论文声称的增量特性在实操中未体现

---

## 2. OfficeMem-Eval（自建，论文主力 benchmark）

**数据位置**：`/Users/bytedance/proj/memoRaxis/q2DataBase/`
**状态**：Task 1 数据基本就绪，Task 2/3 及多模态待扩展

### 已有数据基础（q2）

基于 ByteDance 内部真实办公文档（飞书文档 + 会议纪要）构建：
- **1899 篇文档**（技术设计文档、会议纪要、教程指南等）
- **1672 场会议**（含 ASR 转写）
- **341 道问答题**，分布在 17 个 topic cluster（T_0000 ~ T_0016）
- 中文 271 题 (79.5%)，英文 70 题 (20.5%)
- **无法回答 57 题 (16.7%)**：测试系统能否正确拒绝

### Topic Clusters（示例）

| Cluster | 典型问题 | 回答类型 |
|---|---|---|
| T_0000 (38题) | 客户端注册流程抢跑拉取主导航的技术方案 | 技术细节 |
| T_0001 (29题) | 无头浏览器的定义 | 概念解释 |
| T_0002 (27题) | Q2降本实现的比例 | 数值事实 |
| T_0005 (24题) | 消息信噪比治理 + 知识问答优化的核心方向 | 跨文档综合 |
| T_0006 (20题) | Aily平台的甜点客户画像 | 业务判断 |
| T_0007 (25题) | Lark GTM LLPP authentication parameters | 英文技术 |

### 三类任务设计（大纲 v2.1）

| 任务 | 主测维度 | 数据来源 | 状态 |
|---|---|---|---|
| **Task 1: 事实检索** | F + Q | q2 已有 341 题 | ✅ 基本就绪 |
| **Task 2: 冲突修正** | E | 需在 q2 文档基础上引入反事实编辑（MQuAKE 式"需求变更"） | 待构建 |
| **Task 3: 复杂决策** | R | 需构造跨文档约束推理题（如"考虑延期+预算重新规划 Sprint"） | 待构建 |

### <F, E, Q> × R 覆盖分析

| 维度 | 当前覆盖（q2 已有） | 扩展后覆盖（完整 OfficeMem-Eval） |
|---|---|---|
| **F** | ✅✅ 异构文档（技术文档 vs 会议 ASR vs 表格） | ✅✅ + 多模态（截图/表格） |
| **E** | ⚠️ 文档有版本但未显式测 | ✅ Task 2 引入时间线演化 + 级联更新 |
| **Q** | ✅ 跨文档综合 + 单文档精确 | ✅ |
| **R** | ✅ 跨文档综合题天然需要 R_Iter/R_Plan | ✅ Task 3 专测 R_Plan |

### 关键特点
- **真实办公数据**：不是合成的，文档来自真实飞书空间
- **异构模态**：技术设计文档、会议 ASR 转写、Release Note，信息密度差异大
- **"无法回答"题**：16.7% 的题需要系统正确拒绝，测试幻觉控制
- **Topic clustering**：17 个 cluster 对应不同业务线/技术栈，可按 cluster 分析
- **全维度覆盖（扩展后）**：F×E×Q×R 在一个 benchmark 内统一测试
- **独特性**：首个同时考查 Cross-Modal Consistency + Decision Faithfulness 的办公记忆 benchmark

---

## 3. memory-probe / LoCoMo

**论文**：Diagnosing Retrieval vs. Utilization Bottlenecks in LLM Agent Memory (ICLR 2026 Workshop)
**arXiv**：https://arxiv.org/abs/2603.02473
**数据位置**：`/Users/bytedance/proj/AgeMem/memory-probe/`

### <F, E, Q> × R 覆盖

| 维度 | 覆盖 | 说明 |
|---|---|---|
| **F** | ✅ | 3 种 write strategy（raw chunk / extracted facts / summarized） |
| **E** | ❌ | LoCoMo 是静态对话，无状态变化 |
| **Q** | ✅ | 3 种检索方法（cosine / BM25 / hybrid+rerank） |
| **R** | ✅ 可扩展 | 原生只有 retrieve+answer，但可接入 R_Direct/R_Iter/R_Plan |
| **诊断** | ✅✅ | 3 个 probe（Relevance/Utilization/Failure）做白盒归因 |

### 在我们框架中的定位
- **F×Q×R 因子分析的最佳平台**
- 诊断 probe 可回答"误差来自 M 还是 R"（对应论文 Gap 1）
- **不测 E**，需要配合 AMemGym 补充

---

## 4. AMemGym

**论文**：AMemGym: Interactive Memory Benchmarking for Assistants in Long-Horizon Conversations (ICLR 2026)
**arXiv**：https://arxiv.org/abs/2603.01966
**数据位置**：`/Users/bytedance/proj/AgeMem/amemgym/`

### <F, E, Q> × R 覆盖

| 维度 | 覆盖 | 说明 |
|---|---|---|
| **F** | ⚠️ | 状态通过对话隐式暴露，F 差异不明显 |
| **E** | ✅✅ | **强测 E**：20 维状态 × 10 periods，每 period 4 次变化，30 个月跨度 |
| **Q** | ⚠️ | 多选格式削弱 Q 精度差异 |
| **R** | ⚠️ | 多选格式削弱 R 范式差异 |

### 在我们框架中的定位
- **E 算子差异的最佳实验平台**
- E1（RAPTOR/HippoRAG）vs E4（Mem0）应在此有显著差异
- 不适合做 F×Q×R 因子分析

---

## 5. StructMemEval

**论文**：Evaluating Memory Structure in LLM Agents (Preprint)
**arXiv**：https://arxiv.org/abs/2602.11243
**数据位置**：`/Users/bytedance/proj/AgeMem/StructMemEval/`

### 四类子任务的 <F, E, Q> × R 覆盖

| 子任务 | F | E | Q | R | 说明 |
|---|---|---|---|---|---|
| **Location** (46 cases) | F1 足够 | ✅ 1-5 次状态变化 | Q1 足够 | ⚠️ | 测 E：状态追踪 |
| **Accounting** (3 datasets) | ✅ 需结构化 F | ⚠️ 累加非冲突 | ⚠️ 无 queries 字段 | ❌ | 测 F：是否能维护账本 |
| **Graph** (100+ configs) | ✅ 需 F3 | ❌ 静态 | ✅ 需 Q2 | ⚠️ | 测 F3+Q2：关系图遍历 |
| **Recommendations** (6 categories) | ✅ | ❌ | ✅ | ✅ | 最像 LoCoMo，可用 R_Iter |

### 在我们框架中的定位
- **Graph 子任务**专测 F3+Q2（图结构 + 图遍历），其他 benchmark 不覆盖
- **Location** 是轻量版 AMemGym，测 E
- **Accounting** 独特：需要精确计算，不是传统 QA
- 整体不是一个统一 benchmark，更像 4 个独立小实验

---

## 综合对比

| Benchmark | F | E | Q | R | 诊断 | 数据真实性 | 多模态 |
|---|---|---|---|---|---|---|---|
| **MAB** | ✅ | ❌ 实质为零（"增量ingest"名不副实） | ✅ | ❌ | ❌ 黑盒 | 中（改造自 NLP 数据集） | ❌ |
| **OfficeMem-Eval** | ✅✅ | ⚠️→✅（Task 2 待建） | ✅ | ✅ | ✅ 分层指标 | ✅✅ 真实飞书 | ⚠️→✅（待建） |
| **memory-probe** | ✅ | ❌ | ✅ | ✅ 可扩展 | ✅✅ 3 probe | 中（合成对话） | ❌ |
| **AMemGym** | ⚠️ | ✅✅ | ⚠️ | ⚠️ | ⚠️ Write/Read 分离 | 中（合成画像） | ❌ |
| **StructMemEval** | ✅ 按任务 | ⚠️ 仅 Location | ✅ 按任务 | ⚠️ | ❌ | 低（合成） | ❌ |

---

## 实验方案建议

| 实验目的 | Benchmark | 设计 | 论文 Section |
|---|---|---|---|
| **F×Q×R 因子分析 + 白盒归因** | memory-probe (LoCoMo) | M_Flat × M_Planar × M_Hier × R_Direct/R_Iter/R_Plan | Gap 1 + Gap 2 |
| **E 算子验证** | AMemGym | E1 vs E2 vs E3 vs E4 系统对比 | Gap 2 (E 维度) |
| **结构化 F 的必要性** | StructMemEval (Graph + Accounting) | F1 vs F3 对比 | Gap 2 (F 维度) |
| **真实场景 + 全维度** | OfficeMem-Eval | 3×3 矩阵 + 分层指标，Task 1 可先行 | Section 3（主实验） |
| **大规模 F×Q 补充** | MAB-AR | 已有 9 系统结果，可直接引用 | Empirical 补充 |

---

## MAB 已跑系统 vs 大纲 3×3 矩阵的覆盖度审计

### 已跑系统映射

| 已跑系统 | 实现 | M 分类 | F | E | Q | MAB 覆盖 | 有效性 |
|---|---|---|---|---|---|---|---|
| simpleMem | 自写 naive RAG (PostgreSQL) | M_Flat (baseline) | F1 | E1 | Q1 | AR22 CR8 TTL6 LR40 | ✅ |
| mem0 | mem0 flat (Qdrant) | **M_Flat** | F2 | E4 | Q1 | AR22 CR8 TTL5 LR110 | ✅ |
| mem0g | mem0 + Neo4j 图谱 | **M_Planar** | F2 | E3+E4 | Q2 | AR8 CR8 TTL5 LR16 | ⚠️ 覆盖不全 |
| memGPT | Letta 后端 (gaoang) | Agentic（不在 3×3） | F1 | E2 | Q3 | AR22 CR8 TTL6 LR40 | ✅ |
| A-MEM | A-MEM (ChromaDB) | M_Planar | F1+F2 | E3 | Q1+Q2 | AR22 CR8 TTL❌ LR40 | ❌ 上游 bug，evolution 完全失效 |
| HippoRAG cs8000 | HippoRAG (知识图谱) | M_Planar | F2 | E1 | Q2 | AR22 CR8 TTL5 LR40 | ❌ NER 失败率 20-89% |
| HippoRAG cs1000 | 修复 NER 版 | M_Planar | F2 | E1 | Q2 | 正在跑 | ⚠️ |
| MIRIX | 文本/图谱版 | 混合（不在 3×3） | F1 | E1 | Q3 | AR22 CR8 TTL✅ LR40 | ✅ |
| MIRIX-emb | 向量版 | M_Flat 变体 | F1 | E1 | Q1 | ✅ ✅ - LR40 | ✅ |
| MemOS | MemCube | **M_Hier** | F1 | E2 | Q1 | AR22 CR8 TTL6 LR10 | ⚠️ 部分 adaptor 缺失 + 连接错误需补跑 |

### 大纲 3×3 矩阵 vs 实际覆盖

大纲要求（Section 3.1）：

|  | R_Direct | R_Iter | R_Plan |
|---|---|---|---|
| **M_Flat**: mem0 / Standard RAG | 需要 | 需要 | 需要 |
| **M_Planar**: Zep / A-Mem | 需要 | 需要 | 需要 |
| **M_Hier**: MemOS / MemoRAG / RAPTOR | 需要 | 需要 | 需要 |

实际可用：

|  | R_Direct | R_Iter | R_Plan |
|---|---|---|---|
| **M_Flat**: mem0 ✅, simpleMem ✅ | ✅ | ✅ | ✅ |
| **M_Planar**: ??? | ❌ | ❌ | ❌ |
| **M_Hier**: MemOS ⚠️ | ⚠️ | ⚠️ | ⚠️ |

### ❌ 关键缺口

**1. M_Planar 一行全空（致命）**

大纲指定的两个实现：
- **Zep**（首选）：完全没跑，一行代码都没有
- **A-Mem**（备选）：上游两个严重 bug（strengthen 存 index 当 UUID + evolution JSON 不稳定），evolution 完全失效，结果无效

替代候选：
- mem0g：覆盖不全（AR 只有 8/22），且 mem0g 本质是 M_Flat + 图谱增强，不是纯 M_Planar
- HippoRAG cs8000：NER 失败率 20-89%，作废
- HippoRAG cs1000：正在重跑，但 HippoRAG 是 E1（静态图），不测 E3（拓扑融合），不是理想的 M_Planar 代表

→ **Zep 是最紧急的待办**。它有成熟 SDK、强调时序图谱、支持 E3（增量合并与消歧），是大纲里 M_Planar 的首选。

**2. M_Hier 不完整**

大纲列了 3 个实现：
- MemOS：⚠️ 部分 adaptor 缺失 + 补跑
- **MemoRAG**：完全没跑
- **RAPTOR**：在 memoRaxis 里跑过（72/75 trees），但未出现在当前表格中

→ 至少需要 1 个完整的 M_Hier 系统。MemOS 补完或换 MemoRAG/RAPTOR。

**3. R轴在 MAB 上未系统化**

memoRaxis 的 R1/R2/R3 adaptors 已有，部分系统跑过，但不是所有系统 × 所有 R 的完整矩阵。不过 MAB 不是主实验平台（OfficeMem-Eval 才是），所以这个优先级较低。

### 行动优先级

| 优先级 | 行动 | 原因 |
|---|---|---|
| 🔴 P0 | **接入 Zep** | M_Planar 全空，3×3 矩阵缺中间一行 |
| 🟡 P1 | **补完 MemOS** 或 **接入 MemoRAG/RAPTOR** | M_Hier 不完整 |
| 🟡 P1 | **确认 RAPTOR 结果是否可用** | 可能已有数据只是没列出 |
| 🟢 P2 | HippoRAG cs1000 跑完后评估 | M_Planar 的备选 |
| 🟢 P2 | mem0g 补全覆盖 | M_Planar 的另一个备选 |

---

## 待讨论

1. **Zep 接入**：Zep 有 Python SDK（`zep-python`），接口是否和我们的 MemoryInterface 兼容？预估接入工作量？
2. **MemoRAG vs RAPTOR**：M_Hier 选哪个？MemoRAG 接口更现代（`MemoRAG.memorize()` → `MemoRAG.query()`），但 RAPTOR 可能已有 MAB 结果。
3. **RAPTOR 状态确认**：memoRaxis 里 72/75 trees 的 infer 结果在哪？能直接用吗？
4. **OfficeMem-Eval Task 2 构建**：q2 文档有版本历史（Release Note），能否据此生成冲突修正题？还是需要 synthetic 反事实编辑？
5. **OfficeMem-Eval Task 3 构建**：跨文档约束推理题的具体数据构造流程？
6. **多模态数据源**：q2 的会议 ASR + 文档已有，看板截图从哪来？
