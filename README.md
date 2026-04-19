# test_for_memory-system-eval-remake2

本仓库当前维护的**核心四后端**：

- MemGPT（Letta）
- RAPTOR
- LightRAG
- HippoRAG

## Smoke Test（统一体系）

四后端都支持独立 smoke test，且覆盖：

- AMemGym
- memory-probe
- StructMemEval（`state_machine_location` / `tree_based` / `recommendations`）

命令：

```bash
python smoke_test_memgpt.py
python smoke_test_raptor.py
python smoke_test_lightrag.py
python smoke_test_hipporag.py
```

支持分项运行（`--only`）：

```bash
python smoke_test_memgpt.py --only amemgym
python smoke_test_memgpt.py --only memory_probe
python smoke_test_memgpt.py --only state_machine
python smoke_test_lightrag.py --only tree_based
python smoke_test_hipporag.py --only recommendations
```

统一入口：

```bash
python run_smoke_tests.py --backends memgpt
python run_smoke_tests.py --backends raptor,lightrag
python run_smoke_tests.py --backends memgpt,raptor,lightrag,hipporag
python run_smoke_tests.py --backends memgpt --only memory_probe
```

结果目录统一为：

```text
results/smoke/<backend>/<run_id>/
```

包含：`manifest.json`、`summary.json`、`summary.md`、`cases.jsonl`、`retrievals.jsonl`、`llm_calls.jsonl`、`stdout.log`、`stderr.log`、`config_snapshot.json`。

## Full Benchmark

一条命令顺序跑完四后端全量 benchmark：

```bash
python run_full_benchmark_all_backends.py
```

按“后端分开做”全量 benchmark：

```bash
python run_full_benchmark_split_backends.py
```

## 文档

- 运行与排障：`docs/backend_runtime_guide.md`
- 一键全量说明：`docs/full_benchmark_one_command.md`
- Anaconda 的 smoke + full 指南：`doc/anaconda_smoke_and_full_eval.md`

## mem0g 当前定位

`mem0g` 相关脚本保留为历史/兼容路径，不属于本仓库当前四后端 smoke 主目标集合。
