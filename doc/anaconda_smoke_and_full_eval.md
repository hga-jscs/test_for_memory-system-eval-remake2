# Anaconda 环境下的 Smoke Test 与全量测评指令

> 目标后端：MemGPT / RAPTOR / LightRAG / HippoRAG

## 1) 进入环境

```bash
conda activate <your_env>
cd /workspace/test_for_memory-system-eval-remake2
```

## 2) 四后端独立 Smoke Test

```bash
python smoke_test_memgpt.py
python smoke_test_raptor.py
python smoke_test_lightrag.py
python smoke_test_hipporag.py
```

## 3) 分项 Smoke Test（--only）

```bash
python smoke_test_memgpt.py --only memory_probe
python smoke_test_lightrag.py --only tree_based
python smoke_test_hipporag.py --only amemgym
python smoke_test_raptor.py --only recommendations
```

可选值：

- `amemgym`
- `memory_probe`
- `state_machine`
- `tree_based`
- `recommendations`

## 4) 统一入口 Smoke Test

```bash
python run_smoke_tests.py --backends memgpt,raptor,lightrag,hipporag
python run_smoke_tests.py --backends memgpt --only memory_probe
```

## 5) 全量测评（Full Benchmark）

### 一键按顺序跑四后端

```bash
python run_full_benchmark_all_backends.py
```

### 按后端拆分跑

```bash
python run_full_benchmark_split_backends.py
```

## 6) Smoke 结果查看

每次 smoke 会落盘到：

```text
results/smoke/<backend>/<run_id>/
```

重点文件：

- `summary.md`：人类可读摘要
- `summary.json`：程序可解析摘要
- `cases.jsonl`：case 级结果
- `retrievals.jsonl`：top-k 检索明细
- `llm_calls.jsonl`：调用级日志
- `stdout.log` / `stderr.log`：运行输出

## 7) Smoke vs Full 的区别

- **Smoke**：小样本、快反馈，目标是暴露接线、检索、回答、解析、依赖等关键问题。
- **Full**：全数据/大样本，目标是正式统计性能与稳定性。

## 8) mem0g 定位

`smoke_test_mem0g.py` 与 mem0g benchmark 脚本仅作为历史兼容路径，不纳入当前四目标后端 smoke 体系。
