# AgeMem 实验日志

> 来者不拒的流水账。记录跑 benchmark 过程中遇到的问题、分析过程、解决方案和结论。以时间为轴，最后再统一整理。

---

## 2026-04-10｜mem0g 接入 + Docker 稳定性问题

### 背景

第六个系统。mem0g 是 mem0 + Neo4j graph store：entity extraction → relationship triplets → conflict resolution → 双路检索（vector + graph relations）。

论文（arXiv:2504.19413）配置：m=10 previous messages, s=10 similar memories, GPT-4o-mini。

**论文自身结论**：mem0g 并不稳定优于 mem0——single-hop / multi-hop 上 mem0 更好，只有 temporal 类 mem0g 领先。graph 增加了复杂度但收益有限。

### 依赖

- Neo4j（Docker）+ Qdrant（Docker）+ `neo4j` Python 驱动 + `langchain-neo4j` + `rank-bm25`
- 所有三个 pip 包都是首次遇到缺失时安装的

### 工程问题

#### 1. Neo4j CypherSyntaxError（空 relation type）

**现象**：`Invalid input ']': expected a node label/relationship type name` — LLM 提取 entity/relation 时有时返回空 relation type，导致 Neo4j 查询 `-[r:]->` 语法错误。

**修复**：`mem0g_bench_src.py` 的 `add_memory()` 和 `retrieve()` 加 try/except 兜底，error 记 warning 但不中断。

#### 2. Docker 容器反复被清除（系统级问题）

**现象**：每隔 1-2 小时，Neo4j 和 Qdrant 的 Docker 容器被删除（不是 stop，是 rm）。`--restart always` 和 `-v` volume 挂载都无法阻止——容器本身被移除。

**影响**：
- mem0g StructMemEval 第一轮：跑到 28/172 时 Docker 断连，144 cases 全部 error
- mem0g 4 路并行第一轮：跑了 ~1.5h 后全部失效（connection refused）
- on-policy 4 users 的结果丢失（无增量保存）

**根因**：疑似机器上有系统级 Docker 清理任务（crontab 或 CI 系统的 `docker system prune`），定期清除非关键容器。

**临时对策**：每次重启后重新 `docker run`。但无法根治——长时间运行（>1h）的 mem0g 评测存在随时被中断的风险。

#### 3. 增量保存的重要性（再次验证）

有增量保存的 StructMemEval（28 cases）和 memory-probe（2 convs）在 Docker 断连后通过 resume 恢复，零损失。无增量保存的 on-policy 丢了 4 users ~1.5h 的计算。

**教训**：所有 bench 脚本必须支持增量保存。on-policy 脚本目前仍未加增量保存，是已知风险。

### Smoke test 结果

| Category | mem0g | mem0 | 差异 |
|----------|-------|------|------|
| state_machine (2) | 2/2 (100%) | 64.3% | graph 帮助 |
| tree_based (2) | 0/3 (0%) | 0% | 通病 |
| recommendations (1) | 0/10 (0%) | 4.6% | 持平 |
| AMemGym (2 users) | 6/20 (30%) | 36.5% | 略低 |
| memory-probe (1 conv) | **8/25 (32%)** | **9.0%** | **+23pp** |

全量估算：~14.6h 串行，4 路并行 ~3.7h。

### 实现文件

- `mem0g_bench_src.py` — Mem0GMemory wrapper（Neo4j graph + Qdrant vector 双路检索）
- `bench_memory_probe_mem0g.py` / `bench_structmemeval_mem0g.py` — bench 脚本（含 resume + 增量保存）
- `bench_amemgym_onpolicy.py` 增加 `--system mem0g` 支持

---

## 2026-04-10｜RAPTOR 全量评测完成

### 最终结果

| Benchmark | RAPTOR |
|-----------|--------|
| AMemGym (off-policy) | 52/200 (26.0%) |
| AMemGym (on-policy) | 489/2200 (22.2%) |
| memory-probe | 1106/1986 (55.7%) |
| StructMemEval | 243/595 (40.8%) |

RAPTOR ingest 极快（纯 embedding + 少量 summarize），三 bench 串行只需 ~2h。

---

## 2026-04-10｜A-MEM ChromaDB 问题三次复发与最终修复

### 问题历程

1. **Ephemeral client singleton conflict**（4/8）：多 case 创建 `chromadb.Client()` 时 settings 冲突 → 改为 PersistentClient
2. **PersistentClient readonly database**（4/9）：SQLite WAL 锁在高频 create/delete 循环中累积 → A-MEM StructMemEval 只成功 29/172，on-policy 全部失败
3. **最终修复**（4/10）：`chromadb.EphemeralClient()` + 每次 `build_index` 前 `delete_collection("memories")`，显式清理保证隔离，不依赖文件系统

测试验证：5 轮快速 create-use-reset 循环，每轮 count=1，完美隔离。StructMemEval resume 后 143 remaining cases 零 error。

### 教训

ChromaDB 的 client 管理比预期复杂得多——ephemeral 有 singleton 问题、PersistentClient 有 WAL 锁问题。最终方案是"用 ephemeral 但手动管理 collection 生命周期"。

---

## 2026-04-09~10｜AMemGym on-policy 全系统评测

### on-policy 结果（5 系统已完成）

| System | On-policy | Off-policy | 差异 |
|--------|-----------|------------|------|
| SimpleMem | 22.1% | 25.0% | -2.9pp |
| RAPTOR | 22.2% | 26.0% | -3.8pp |
| HippoRAG | 22.4% | 27.0% | -4.6pp |
| mem0 | **23.0%** | **36.5%** | **-13.5pp** |
| A-MEM | 🔄 进行中 | 25.0% | — |
| mem0g | 🔄 进行中 | — | — |

**关键发现**：mem0 在 on-policy 下从 36.5% 骤降至 23.0%，跟 RAG 系统持平。off-policy 时 mem0 的优势（一次性看到所有 period 的数据，evolution 可以合并/覆盖冲突信息）在 on-policy 下消失——因为早期 period 的状态提取已经存入，后期新状态覆盖不一定成功。

### memory-probe 补充 F1/BLEU 指标

纯文本计算，无需 LLM 调用，基于已有 results JSON 中的 pred/reference 字段：

| System | LLM-J | F1 | BLEU-1 |
|--------|------:|----:|-------:|
| SimpleMem | 32.2% | **41.1%** | **37.7%** |
| mem0 | 9.0% | 10.7% | 11.6% |
| HippoRAG | **59.8%** | 23.4% | 18.2% |
| RAPTOR | 55.7% | 20.4% | 15.3% |

LLM-J 和 F1 反映不同特性：HippoRAG/RAPTOR 语义正确但措辞冗长（高 LLM-J 低 F1），SimpleMem 直接返回原文（低 LLM-J 高 F1）。

---

### 累积工程教训清单

| # | 教训 | 来源 |
|---|------|------|
| 1 | 读 benchmark 论文确认评测协议（on-policy vs off-policy） | AMemGym 事故 |
| 2 | 读系统论文确认核心配置（top_k, evolution 开关等） | A-MEM top_k, HippoRAG MCQ |
| 3 | chunk_size 必须对齐或明确区分 | A-MEM memoRaxis 灾难 |
| 4 | 依赖 LLM 的功能必须端到端验证（JSON 输出稳定性） | A-MEM evolution JSON |
| 5 | 每个 benchmark 的每个 category 单独 smoke test | mem0 tree_based 0% |
| 6 | 不要假设外部 API 的 default 参数合理 | mem0 get_all limit=100 |
| 7 | **所有 bench 脚本必须增量保存** | A-MEM memory-probe 丢结果、mem0g on-policy 丢结果 |
| 8 | ChromaDB 需要手动管理 collection 生命周期 | A-MEM 三次 ChromaDB 事故 |
| 9 | Docker 容器可能被系统清理，不能假设持久运行 | mem0g Docker 清除事故 |
| 10 | 成本审计 ingest/infer 分离 | 全系统 |
| 11 | on-policy 脚本需要增量保存（当前未实现，已知风险） | mem0g 丢 4 users |

---

## 2026-04-09｜AMemGym on-policy 评测协议遗漏（待修复）

### 现象

读 AMemGym 论文（ICLR 2026）后发现：论文要求 **on-policy 评测**——每个 period 结束后用当前 period 的 state 做 ground truth 评测所有 QA。我们做的是 **off-policy**——全量 ingest 11 个 period 后只对比最终 state。

论文 Section 3.3：
> "assistants are prompted to answer all evaluation questions **after each interaction period**"

### 影响

- **所有 5 个系统（SimpleMem / mem0 / HippoRAG / A-MEM / RAPTOR）的 AMemGym 结果都需要重跑**
- off-policy 模式下系统间对比仍然公平（同样的错误做法），但绝对数值和论文不可比
- on-policy 模式：每 user 从 10 次评测变为 110 次（11 periods × 10 QA）

### 重跑代价估算

| 系统 | 增量 ingest？ | 估算时间 |
|------|:---:|---:|
| SimpleMem | ✅ | ~30 min |
| mem0 | ✅ | ~5h |
| HippoRAG | ❌ 需重建 graph | ~4h |
| A-MEM | ✅ | ~8h |
| RAPTOR | ❌ 需重建 tree | ~1h |

### 教训

接入 benchmark 前**必须读论文的 Evaluation 章节**，确认评测时序（batch vs on-policy vs per-turn）。这是第二次犯"不读论文就写脚本"的错误。

---

## 2026-04-09｜论文配置全面审计

对所有系统和 benchmark 的配置做了一次全面审计（5 个子 agent 并行检查）。

### memory-probe (LoCoMo)

- 评测协议（batch ingest → eval）：✅ 正确
- top_k 不一致：SimpleMem/mem0/HippoRAG 用 5，A-MEM/RAPTOR 用 10。但这不是 benchmark 要求，是各系统自身配置。
- 指标：我们只用 LLM-J（yes/no），论文还用 F1 和 BLEU-1

**补充计算了 F1 / BLEU-1**（纯文本计算，无需重跑）：

| System | LLM-J | F1 | BLEU-1 |
|--------|------:|----:|-------:|
| SimpleMem | 32.2% | **41.1%** | **37.7%** |
| mem0 | 9.0% | 10.7% | 11.6% |
| HippoRAG | **59.8%** | 23.4% | 18.2% |
| RAPTOR | 55.7% | 20.4% | 15.3% |

**关键发现**：HippoRAG/RAPTOR 的 LLM-J 高但 F1 低——因为生成冗长但语义正确的回答（gold="7 May 2023"，pred="Caroline went to the group on 7 May 2023"），LLM 判对但 token overlap 低。SimpleMem F1 最高因为直接返回原文 chunk，token 天然重合。两组指标互补。

### StructMemEval

- 评测协议：✅ 正确（batch ingest → LLM judge yes/no）
- Judge 标准：✅ 正确

### HippoRAG

- chunk_size 1000 chars（论文用 1200 tokens）：✅ 已知差异，基于 MAB 实验选择
- embedding 用 DashScope（论文用 NV-Embed-v2）：✅ 已知限制（无 GPU），所有系统统一
- top_k=5：✅ 匹配论文 qa_top_k

### mem0

- infer=True：✅ 正确
- top_k=5：✅ 合理
- 缺论文 PDF（arXiv:2504.19413）：已补充

### A-MEM

- evolution=ON, top_k=10：✅ 已对齐论文
- JSON truncation 已修（max_tokens 1000→3000 + json_repair）

### RAPTOR

- num_layers=3（论文默认 5）、tb_max_tokens=200（默认 100）、tr_top_k=10（默认 5）：均为合理调整
- 缺论文 PDF（arXiv:2401.18059）：已补充

---

## 2026-04-09｜A-MEM JSON 截断修复 + ChromaDB 锁修复

### JSON 截断问题

A-MEM evolution prompt 返回的 JSON 在 DashScope/qwen 上频繁截断（"Unterminated string" 错误），修复前 46 chunks 中 9+ 次失败。

**根因**：`max_tokens=1000` 不够 A-MEM 复杂嵌套 JSON（6 个字段含多层数组）。

**修复**：
1. `max_tokens` 从 1000 增加到 3000
2. 加 `json_repair` 库做 fallback（`pip install json_repair`）
3. 在 `memory_system.py` 的两处 `json.loads()` 替换为 `_safe_json_loads()`

**效果**：修复后 JSON error 从 9+ 次降为 **0 次**，ingest 速度也提升 37%（1444s → 909s/user）。

### ChromaDB PersistentClient 锁冲突

多个 case 串行运行时，ChromaDB PersistentClient 的 SQLite WAL 锁导致第二个 case 报 "attempt to write a readonly database"。

**修复**：改用 ephemeral（in-memory）client，bench 场景不需要持久化。

### A-MEM top_k 对齐

论文要求 k=10，我们原来用 k=5。已修正。

---

## 2026-04-09｜RAPTOR 接入 + 全量评测

### 背景

第五个系统。RAPTOR（ICLR 2024）：递归摘要树——chunk → UMAP cluster → LLM summarize → 递归建树 → 检索时沿 collapsed tree 遍历。

### 实现

- `raptor_bench_src.py` — RaptorBenchMemory wrapper
- 依赖：`umap-learn`, `faiss-cpu`（sentence-transformers 已有）
- 绕过 `AgenticMemorySystem.__init__` 的问题后改为 `object.__new__` 手动构建
- Embedding: DashScope text-embedding-v3（跟其他系统统一）
- 配置：tb_num_layers=3, tb_max_tokens=200, tr_top_k=10, tr_threshold=0.5

### Smoke test 结果（全流程覆盖）

| Category | 结果 | 关键数据 |
|----------|------|---------|
| state_machine (2) | 2/2 (100%) | tree=2(0L)，太小没建层 |
| tree_based (2) | 2/3 (67%) | tree=4(0L) |
| recommendations (2) | 4/20 (20%) | tree=17(1L)，开始建层 |
| AMemGym (3 users) | **13/30 (43.3%)** | tree=25(1L)，最高 |
| memory-probe (1 conv) | **15/25 (60%)** | tree=85(2L)，2层树 |

**关键发现**：RAPTOR ingest 极快（79 chunks 23.8s，纯 embedding + 少量 summarize），比 A-MEM（1294s）快 54×。

### 全量结果（进行中）

- AMemGym：52/200 (26.0%)，6.4 min ✅ 已完成
- memory-probe：1106/1986 (55.7%) ✅ 已完成
- StructMemEval：🔄 进行中

### 关于 AMemGym 26% vs smoke 43%

不是 bug。AMemGym 测用户状态追踪，RAPTOR 作为 RAG 系统无法区分新旧状态——跟 SimpleMem（25%）、HippoRAG（27%）处于同一水平。Smoke test 3 users 样本偏差。

---

## 2026-04-09｜Git 仓库初始化 + 推送

- 仓库名：`memory-systems-eval`（GitHub: Netreee/memory-systems-eval，public）
- Submodule：StructMemEval / memory-probe / amemgym（锁定到实验时 commit）
- amemgym data.json（上游没有）单独存入 `data/amemgym/v1.base/data.json`
- `config.yaml` 排除（含 API key），提供 `config.yaml.example`
- 外部 repo（AriadneMem 等）gitignore

---

## 2026-04-08｜A-MEM 接入：upstream evolution bug 修复 + smoke test

### 背景

扩展评测系统：SimpleMem / mem0 / HippoRAG 之后，接入第四个系统 A-MEM（agiresearch/A-mem）。
A-MEM 核心特性：每条记忆 ingest 时做 2 次 LLM 调用——`analyze_content()`（提取 keywords/context/tags）+ `process_memory()`（找邻居，决定是否 evolve：更新链接和邻居的 tags/context）。

### Upstream evolution bug（已修复）

**现象**：之前在 memoRaxis 项目上跑 A-MEM 评测时，evolution 功能完全失效（"A-MEM disaster"：chunk_size 错误 + evolution bug 导致全量结果作废）。

**根因**：`memory_system.py` 的 `find_related_memories()` 返回 ChromaDB 搜索结果的**顺序号** `[0,1,2,3,4]`，但 `process_memory()` 将这些顺序号当作 `self.memories` dict 转 list 后的下标来索引。结果是 evolution 更新了 dict 里碰巧排在前面的记忆，**不是实际搜到的邻居**。

```python
# BUG: find_related_memories 返回
indices.append(i)          # i = 0,1,2... 顺序号

# BUG: process_memory 使用
noteslist = list(self.memories.values())
notetmp = noteslist[memorytmp_idx]  # 取 dict 第 N 个，而非搜到的第 N 个邻居
```

**确认是 upstream bug**：`git diff` 确认 `indices.append(i)` 行未被我们修改，是 `agiresearch/A-mem` 原始代码的问题。

**修复**（3 处改动，均在 `memoRaxis/external/amem_repo/agentic_memory/memory_system.py`）：
1. `find_related_memories()` 返回 `doc_id` 列表而非顺序号
2. `process_memory()` 的 `update_neighbor` 按 `doc_id` 查找 `self.memories` 而非按下标索引
3. memory_str 中的 `memory index:{i}` 改为 `memory id:{doc_id}`，让 LLM 返回的 `suggested_connections` 是有效的 UUID

### 依赖安装

```
pip3 install chromadb nltk sentence-transformers
```
注：`sentence-transformers` 会附带安装 `torch`。A-MEM 使用本地 SentenceTransformer embedding（`all-MiniLM-L6-v2`, 384 维），与其他系统的 DashScope `text-embedding-v3`（1024 维）不同，但这是 A-MEM 的默认配置。

### 实现文件

- `amem_bench_src.py` — AMemBenchMemory wrapper（buffer→build_index、LLM token tracking、evolution 开关）
- `smoke_test_amem.py` — per-category smoke test（5 个 category）

### Smoke test 结果

| Category | 状态 | 关键数据 |
|----------|------|---------|
| state_machine_location | ✅ | 3 chunks, 5 LLM calls (analyze×3 + evolve×2), 正确回答 |
| tree_based | ✅ | 5 chunks, 9 LLM calls, 正确推理间接同事关系 |
| recommendations | ✅ | 15 chunks, 29 LLM calls, retrieve + answer 正常 |
| AMemGym | ✅ | 46 chunks, 91 LLM calls, MCQ 回答正确格式 |
| memory-probe | 🔄 | ingest 中（70 chunks × 2 LLM = ~140 calls，耗时较长）|

**关键观察**：A-MEM 每 chunk 约 2 次 LLM 调用（analyze + evolve），比 HippoRAG 更重。memory-probe 的 70 chunks 对应 ~140 LLM calls，预计 ingest 耗时较长。全量评测的时间估算需要 bottom-up 计算。

---

## 2026-04-08｜HippoRAG AMemGym MCQ prompt bug 修复 + 重跑

### 重跑结果

修复 MCQ prompt（字母索引→数字索引 + 范围校验）后重跑：

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 准确率 | 29/200 (14.5%) | **54/200 (27.0%)** |
| 超出范围预测 | 71/200 | **0/200** |
| 预测分布 | 偏向 E (57.5%) | 合理分布 |

### HippoRAG 三 benchmark 最终对比

| Benchmark | SimpleMem | mem0 | HippoRAG |
|-----------|-----------|------|----------|
| AMemGym | 25.0% | **36.5%** | 27.0% |
| memory-probe | 32.2% | 9.0% | **59.8%** |
| StructMemEval | 28.9% | 7.4% | **39.5%** |

---

## 2026-04-08｜HippoRAG AMemGym MCQ prompt bug

### 现象

HippoRAG AMemGym 全量评测结果 29/200 (14.5%)，远低于 SimpleMem (25.0%) 和 mem0 (36.5%)。对预测分布做检查时发现：200 道题中 115 道（57.5%）预测为 E (idx=4)，严重偏斜。

### 排查

AMemGym 的 answer_choices 长度不固定：

| 选项数 | 题数 |
|--------|------|
| 4 | 132 |
| 5 | 56 |
| 6 | 10 |
| 7 | 2 |

`bench_amemgym_hipporag.py` 的 MCQ prompt 硬编码了 "Answer with just the letter (A/B/C/D/E):"，且用字母索引、无范围校验。对于 132 道只有 4 个选项的题，prompt 提示 E 是合法选项，模型被诱导选择不存在的 E。

对比 SimpleMem/mem0 的 bench 脚本（`bench_amemgym_full.py` / `bench_amemgym_mem0.py`）：
- 使用数字索引：`(0) xxx, (1) xxx, ...`
- 提示："Reply with ONLY the choice number (e.g., 0, 1, 2, ...)"
- 有范围校验：`if 0 <= idx < len(choices)`

**这个 bug 只存在于 HippoRAG 版**，是写 bench 脚本时引入的不一致。

### 影响

- 71/200 预测超出 choice 范围（选了不存在的 E），必然判错（35.5%）
- 14.5% 准确率被人为压低，不能与 SimpleMem/mem0 公平对比

### 修复

将 HippoRAG 版 MCQ prompt 统一为与 SimpleMem/mem0 相同的数字索引格式 + 范围校验，删除旧结果，重跑 AMemGym。

---

### 附：StructMemEval resume 重跑说明

首次 StructMemEval 全量评测时（2026-04-07），跑到中途因 git submodule 操作删除了 `StructMemEval/` 目录，导致 141/172 cases 报 "No such file" 错误。在 bench 脚本中加入 resume 逻辑（读取已有结果跳过已完成 case）后重跑，141 cases 全部正常完成。

最终 results JSON 中残留了首次失败的 141 条 error 记录（stale），不影响准确率统计。

---

## 2026-04-07｜HippoRAG 三 benchmark 接入 pre-flight 验证

### 背景

在 mem0 + SimpleMem 双系统对比基础上，扩展到 7 个系统全量对比。
首选 HippoRAG（不依赖本地 Neo4j/Letta/Mirix 服务），接入三个外部 benchmark。

---

### 依赖安装问题

HippoRAG `__init__.py` 在 import 时无条件 load 所有 backend，包括 torch-dependent 的 TransformersLLM、ContrieverModel 等。当前环境无 torch，导致 import 失败。

**解决**：修改 HippoRAG 内部四个文件，将非核心 backend 改为 lazy/try-except import：
- `llm/__init__.py`：TransformersLLM / BedrockLLM → lazy
- `embedding_model/__init__.py`：所有非 OpenAI 模型 → lazy
- `embedding_model/base.py`：`import torch` → try-except
- `embedding_model/OpenAI.py`：`import torch` → try-except

同时将 `utils/embed_utils.py`（原用 torch 做 KNN）重写为 numpy 实现。

`HippoRAG.py` 加了 `from __future__ import annotations`（Python 3.9 不支持 `X|Y` 联合类型语法）。

---

### Pre-flight Checklist 逐项验证结果

| Item | 状态 | 关键数据 |
|------|------|---------|
| 1. 默认参数 | ✅ | `openie_mode=online`（ON）, `embedding_batch_size=8`（DashScope limit 10）|
| 2. reset 隔离 | ✅ | `shutil.rmtree(save_dir)` + 新实例，仅 `buffer.clear()` 不够 |
| 3. audit 字段 | ✅ | patch `h.llm_model.openai_client.chat.completions.create`，6 calls/3 chunks |
| 4. per-category smoke | ✅ | 见下表 |
| 5. bottom-up 估算 | ✅ | 见下表 |
| 6. chunk_size | ✅ | 1000 char，与 MAB hipporag_cs1000 对齐 |

#### Smoke test 结果（各 1 case）

| Category | HippoRAG | mem0（历史数据）| 备注 |
|----------|----------|---------------|------|
| state_machine_location | 1/1 ✅ | 42.9% | - |
| **tree_based** | **1/1 (100%)** ✅ | **0%** ❌ | 关键风险点，HippoRAG triple extraction 完美处理 |
| recommendations | 1/2 | 4.6% | - |
| AMemGym | 1/3 | 36.5% | 5选1题，baseline=20% |
| memory-probe cat=1~4 | **4/4** ✅ | cat=2: 2.4% | cat=2 时间题：HippoRAG 保留原文日期，mem0 丢失 |

**关键发现**：memory-probe cat=2（时间题，如"When did Caroline go to the LGBTQ support group?"）：
- mem0：丢失时态信息 → "I don't know."
- HippoRAG：保留原文完整 chunk → 正确答出"7 May 2023"

#### Bottom-up 全量耗时估算

| Benchmark | Ingest | Infer | 合计 |
|-----------|--------|-------|------|
| AMemGym (20 users) | ~13 min | ~10 min | ~23 min |
| memory-probe (10 conv, 1461 QA) | ~14 min | ~97 min | ~111 min |
| StructMemEval (172 cases) | ~37 min | ~94 min | ~131 min |
| **总计** | | | **~4.4h** (串行，内部并行后实际 ~2h) |

注：HippoRAG 内部 NER/triple extraction 使用 ThreadPoolExecutor 并行，实际 ingest 比估算快。

---

### 实现文件

- `hipporag_bench_src.py` — HippoRAGMemory wrapper（patch token tracking / reset / audit）
- `bench_amemgym_hipporag.py` / `bench_memory_probe_hipporag.py` / `bench_structmemeval_hipporag.py`
- `smoke_test_hipporag.py` — pre-flight 验证脚本

---

*最后更新：2026-04-07*

---

## 2026-04-02 ~ 2026-04-05｜mem0 三 benchmark 评测

### 背景

将 mem0（`infer=True`，LLM 原子事实提取）接入三个 benchmark：
- **memory-probe**（LoCoMo-10，10 段对话，1461 QA）
- **AMemGym**（20 用户 × 11 periods × ~46 sessions）
- **StructMemEval**（172 cases：state_machine×42 + tree_based×100 + recommendations×30，595 QA）

对比基线：SimpleMem（朴素 RAG，chunk 存原文 + 向量检索）

实现文件：
- `mem0_bench_src.py` — Mem0RAGMemory wrapper（Qdrant in-memory, `infer=True`）
- `bench_amemgym_mem0.py` / `bench_memory_probe_mem0.py` / `bench_structmemeval_mem0.py`

---

### 运行配置

```
MAX_WORKERS = 1   # 串行，避免 DashScope API 并发限速
LLM: DashScope qwen 系列
Embedding: DashScope text-embedding-v3 (dim=1024)
```

串行原因：3 workers 时 API throttle 导致 ~9s/chunk → ~39s/chunk，退化严重。

---

### 实际耗时（过夜串行）

| Benchmark | 实际耗时 | 启动时间 | 完成时间 |
|-----------|---------|---------|---------|
| AMemGym | ~3.5h | 2026-04-02 21:58 | 2026-04-03 01:27 |
| memory-probe | ~5.5h | 01:27 | 06:56 |
| StructMemEval state_machine + tree_based | ~1h50m | 06:56 | 08:44 |
| StructMemEval recommendations | ~8h | 08:44 | 16:52 |
| **总计** | **~19h** | | |

> ⚠️ 事前估算严重偏低（说的是"几小时"），实际跑了 19 小时。根本原因是没有对每个 category 单独做 per-case latency 的 bottom-up 估算，直接拍脑袋。recommendations category 每 case 约 15 min（包含 ingest LLM + answer LLM + judge LLM × 多个 QA），远超预估。

---

### 最终结果

#### 总览对比

| Benchmark | SimpleMem | mem0 | Δ |
|-----------|-----------|------|---|
| AMemGym | 25.0% (50/200) | **36.5%** (73/200) | **+11.5% ✅** |
| memory-probe | **32.2%** (496/1542) | 9.0% (131/1461) | -23.2% ❌ |
| StructMemEval | **28.9%** (172/595) | 7.4% (44/595) | -21.5% ❌ |

#### memory-probe 按 category

| cat | 含义 | SimpleMem | mem0 |
|-----|------|-----------|------|
| 1 | 单跳事实回忆 | 15.2% | 7.7% |
| 2 | 时间类问题 | 18.4% | **2.4%** ⚠️ |
| 3 | 推理/关联 | 18.8% | 8.3% |
| 4 | 多跳事实 | 44.5% | **11.8%** |

#### StructMemEval 按 category

| category | SimpleMem | mem0 |
|----------|-----------|------|
| state_machine_location | 42.9% | **64.3%** ✅ |
| tree_based | **65.7%** | **0.0%** ⚠️ |
| recommendations | **9.4%** | 4.6% |

---

### 异常点分析

#### 异常1：tree_based 0/181 (0.0%)

**现象**：全部 100 个 cases 的 `ingest_chunks=1, atoms=0`，回答全部输出 "No."

**排查过程**：

1. 查 tree_based 数据结构：session 里 100 条消息全是 `role=user`，无 assistant 轮
2. 查 `bench_structmemeval_mem0.py` 的 `ingest_case()`：
   ```python
   if msg["role"] == "assistant" and len(buf) >= 2:
       mem.add_memory(...)  # ← 只在 assistant 轮时 flush
   ```
   全 user-role → 永不 flush → 100 条堆成 1 个 3000-token 巨型 chunk
3. mem0 收到该 chunk 后，`get_all()` 返回 `atoms=0`
4. 查 `mem0/configs/prompts.py` 的 `USER_MEMORY_EXTRACTION_PROMPT`：专为个人信息设计（偏好/健康/计划），不认识组织关系图数据 "X works with Y"，LLM 返回 `facts: []`

**关键对比**：SimpleMem 用同样的 ingest 逻辑（也是全量堆 1 chunk），但直接存原文做相似检索 → 65.7%。mem0 的 LLM 提取步骤是瓶颈。

**修复尝试**：

将 `ingest_case()` 改为固定 `CHUNK_SIZE=5` 分块（100 msgs → 20 chunks × 5 msgs），重跑中等规模验证（18 cases，34 queries）：

```
结果：0/34 (0.0%) —— 与修复前完全相同
atoms 分布：多数 case atoms=0，部分 case atoms=5~20（随机性极高）
```

**结论**：

- ingest 分块是 bench 代码问题（已修复 `CHUNK_SIZE=5`），但不影响最终准确率
- 根本原因是 **mem0 的 extraction prompt 与图关系数据的设计不兼容**
- mem0 对 tree_based 类数据的准确率：0%，这是真实能力边界，不是工程 bug
- **tree_based 0/181 是 mem0 能力的有效实验结论，不需要修正**

---

#### 异常2：mem_size 全部 = 100（统计截断 bug）

**现象**：memory-probe 所有对话的 `ingest_mem_size` 恰好都是 100

**根因**：

`mem0_bench_src.py` 的 `mem_size` 属性：
```python
results = self.memory.get_all(user_id=self.user_id)  # 未传 limit
```

`mem0/memory/main.py`：
```python
def get_all(self, ..., limit: int = 100):  # 硬编码默认值
```

实际存储的 atoms 可能 >100，但 `get_all()` 只返回前 100 条，导致统计数据失真。

**修复**：`mem0_bench_src.py` 改为 `limit=10000`（已修复）

**影响**：只影响 audit 统计字段，**不影响任何准确率结论**。

---

#### 异常3：memory-probe cat=2 时间类问题 2.4%（远低于 SimpleMem 18.4%）

**现象**：

```
Q: When did Caroline go to the LGBTQ support group?
gold: 7 may 2023
pred: I don't know.
```

**排查**：

- bench 代码确实传入了日期：`text = f"[{date_str}]\n" + ...`，metadata 也有 `{"date": date_str}`
- 问题出在 mem0 内部：`_add_to_vector_store()` 做 fact extraction 时 **metadata 完全被忽略**，只处理文本
- LLM 提取事实时，将 "Caroline attended LGBTQ support group on May 7, 2023" 简化为 "Caroline attended LGBTQ support group"——**日期被丢弃**
- SimpleMem 直接存含 `[2023-05-07]` 前缀的原文 chunk，检索命中后模型能读到日期

**结论**：

mem0 的 atomic fact extraction 会系统性地丢失时态信息（具体日期、相对时间表述）。这是 mem0 架构的设计约束，不是 bench 代码问题。

---

### 综合结论：mem0 的能力图谱

| 场景 | mem0 表现 | 原因 |
|------|----------|------|
| 用户状态追踪（AMemGym）| **优于** SimpleMem (+11.5%) | E3 evolution（原子事实更新）正是设计目标 |
| 状态机/决策追踪（state_machine）| **优于** SimpleMem (+21.4%) | 结构化状态变化 → 原子事实提取质量高 |
| 多跳事实/对话回忆（memory-probe）| **差于** SimpleMem (-23.2%) | 提取损失信息，特别是时态 |
| 图关系数据（tree_based）| **完全失效** (0%) | extraction prompt 不认识组织关系图 |
| 复杂推荐/偏好（recommendations）| **差于** SimpleMem (-4.8%) | 多轮偏好推断，提取后语义丢失 |

**核心 insight**：mem0 的 `infer=True` 是一把双刃剑。对"用户当前状态"类任务，原子事实提取+去重+合并能大幅提升信噪比；对"verbatim recall"、"temporal precision"、"graph traversal"类任务，信息损失是致命的。

---

### 工程教训

#### 1. Smoke test 必须覆盖每个 category

本次未对 StructMemEval 每个 category 做逐个 smoke test，导致 tree_based 跑了 100 cases 才发现 atoms=0。标准检查项：
- `ingest_chunks` 是否合理（不应该是 1 对应大量消息）
- `atoms` 是否 > 0 且不是固定值（截断 bug）
- 至少 1 个 QA 的 pred 人眼确认有意义（不是 "I don't know" 或 "No"）

#### 2. 不要假设外部系统 API 的 default 参数合理

`get_all(limit=100)` 这种截断 default 在 benchmark 统计场景下是 bug，使用任何外部 API 前都要检查 default 值是否符合预期。

#### 3. 时间估算要 bottom-up

每个 case 的 LLM 调用数要数清楚：
- ingest：N_chunks × 1 LLM（mem0 内部提取）
- infer：N_queries × 1 LLM（answer）+ N_queries × 1 LLM（judge）

recommendations 每 case 平均 ~12 个 QA × 2 = 24 次 LLM 调用 + ingest，实测约 15 min/case。

---

### 待跟进

- [ ] mem0 在 tree_based 上的根本失效机制，值得写成 case study（extraction prompt 设计的局限性）
- [ ] memory-probe 时间信息丢失：是否可以通过在 ingest 时显式重复日期（每条 turn 都加 `[date]` 前缀）来缓解？
- [ ] SimpleMem vs mem0 的 F×E×Q 因子分解：mem0 主要贡献是 E3（evolution），F2（LLM 提取）在多数 benchmark 上反而是负贡献
- [ ] 下一个系统：A-MEM 或 RAPTOR，接入前先读原文确认核心特性，做完整 smoke test checklist

---

*最后更新：2026-04-05*
