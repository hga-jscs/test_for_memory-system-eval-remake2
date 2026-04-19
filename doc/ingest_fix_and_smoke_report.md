# Ingest 修复与 Ingest Smoke Test 报告

## 1) 改动文件
- `memgpt_bench_src.py`
- `raptor_bench_src.py`
- `hipporag_bench_src.py`
- `lightrag_bench_src.py`
- `ingest_audit_utils.py`（新增）
- `ingest_smoke_dataset.py`（新增）
- `ingest_smoke_common.py`（新增）
- `run_ingest_smoke_tests.py`（新增）
- `smoke_ingest_memgpt.py`（新增）
- `smoke_ingest_raptor.py`（新增）
- `smoke_ingest_hipporag.py`（新增）
- `smoke_ingest_lightrag.py`（新增）

## 2) 四后端真实存储位置
- **MemGPT/Letta**：`/v1/agents/{agent_id}/archival-memory`（远端 Letta archival store），本地审计落盘在 `results/ingest_audit/memgpt/<run_id>/`。
- **RAPTOR**：RAPTOR 运行时 tree（内存结构），并在 `save_dir` 写 `tree_summary.json`；完整审计落盘在 `results/ingest_audit/raptor/<run_id>/`。
- **HippoRAG**：`save_dir` 图索引目录（OpenIE + graph/index artifacts）；审计落盘在 `results/ingest_audit/hipporag/<run_id>/`。
- **LightRAG**：`working_dir/save_dir` 下 kv/vector/graph 等存储；审计落盘在 `results/ingest_audit/lightrag/<run_id>/`。

## 3) 原主要 ingest 问题（按后端）
- **MemGPT/Letta**：chunk 写入重试与错误分类不足、时间元数据未映射、source 追踪弱。
- **RAPTOR**：整包 join 导致 provenance 弱，save_dir 缺乏可审计 ingest 产物。
- **HippoRAG**：split/join 压平文本结构，speaker/time/session 信息丢失。
- **LightRAG**：整包 join 来源不透明，usage 字段伪 0，审计文件不足。

## 4) 修复策略
- 统一增加 ingest 审计写盘：
  - `config_snapshot.json`
  - `ingest_chunks.jsonl`
  - `ingest_summary.json`
  - `storage_manifest.json`
  - `provenance_map.json`
  - `stdout.log` / `stderr.log`
- **MemGPT**：
  - 保持真实 Letta archival 路径，不引入 fallback。
  - chunk 写入前注入可检索前缀（chunk/source/case/user/conv/time/period/category）。
  - 尝试把 `session_time/time/period` 解析到 `created_at`，失败则前缀保留原值。
  - 新增 ingest 级重试与错误分类（4xx/5xx/model-embedding/timeout-proxy-connection）。
- **RAPTOR**：
  - 仍兼容上游 `add_documents(big_text)`，但 wrapper 建立 `chunk_join_offsets` provenance map。
  - 持久化 `tree_summary.json` 并落盘审计文件。
- **HippoRAG**：
  - chunker 改为“优先按换行、其次按空格”，避免结构压平。
  - 保留 session/speaker/time 原文本结构并写入审计 chunk 清单。
- **LightRAG**：
  - 保持 embedding.dim 预检。
  - ingest 改为按 chunk 逐条 `ainsert`（带 source 前缀）。
  - usage 字段改为 `unknown_not_exposed_by_upstream`，不再伪装 0。

## 5) ingest smoke 样本设计
使用 `ingest_smoke_dataset.py` 合成数据覆盖：
- speaker 边界（Alice/Bob/Alicia）
- time 边界（`[Time: 2023-07-01]` / `[Time: 2023-07-02]`）
- session 边界（`[Session: s1]` / `s2`）
- 易错实体（Alice vs Alicia）
- 时间锚点（last Friday / two weeks ago / 具体日期）
- 列表事实（pets / activities）
- 唯一事实（`blue ceramic bowl with white dots`、`Voyageurs National Park`、`sunflower tattoo`）
- 反例（`purple dragon`）

## 6) 四后端 ingest smoke 验收结果
本环境执行结果均为 **明确失败**（属于依赖/服务环境限制，不是静默通过）：
- MemGPT：缺少 `LETTA_BASE_URL`。
- RAPTOR：代理环境下 tiktoken 依赖下载失败（403）。
- HippoRAG：缺少 `igraph/python-igraph` 运行依赖。
- LightRAG：导入 LightRAG 依赖失败。

## 7) 残余限制
- 需要可用 Letta 服务与正确 model/embedding handle。
- RAPTOR 仍受上游 tree 构建方式限制，node->chunk 是近似 provenance，不是严格一一映射。
- HippoRAG/OpenIE 的信息抽取质量仍受上游模型表现影响。
- LightRAG 上游未暴露精确 ingest usage；当前已显式标记 unknown，而非伪零值。

## 8) 运行命令清单
```bash
python run_ingest_smoke_tests.py --backends memgpt,raptor,hipporag,lightrag
python smoke_ingest_memgpt.py
python smoke_ingest_raptor.py
python smoke_ingest_hipporag.py
python smoke_ingest_lightrag.py
```
