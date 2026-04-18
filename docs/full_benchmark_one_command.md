# 四后端“全 benchmark”一条命令跑完（MemGPT / RAPTOR / LightRAG / HippoRAG）

> 更新时间：2026-04-18

这份文档是 **傻瓜式操作说明**：你只需要按顺序复制命令，即可一次跑完四个后端在三个基准上的全量评测。

---

## 0. 你将会跑什么

一次命令会顺序执行下列 12 个脚本：

- MemGPT（Letta）
  - `bench_amemgym_memgpt.py`
  - `bench_memory_probe_memgpt.py`
  - `bench_structmemeval_memgpt.py`
- RAPTOR
  - `bench_amemgym_raptor.py`
  - `bench_memory_probe_raptor.py`
  - `bench_structmemeval_raptor.py`
- LightRAG
  - `bench_amemgym_lightrag.py`
  - `bench_memory_probe_lightrag.py`
  - `bench_structmemeval_lightrag.py`
- HippoRAG
  - `bench_amemgym_hipporag.py`
  - `bench_memory_probe_hipporag.py`
  - `bench_structmemeval_hipporag.py`

---

## 1. 前置准备（严格按这个来）

在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

安装依赖（建议最小必需集）：

```bash
pip install -r third_party/raptor/requirements.txt
pip install -r third_party/HippoRAG/requirements.txt
pip install -e third_party/LightRAG
```

配置 `config.yaml`（从模板复制）：

```bash
cp config.yaml.example config.yaml
# 然后手工填入 llm 与 embedding 的 api_key/base_url/model
```

如果你跑 MemGPT（Letta），还需要 Letta 服务：

```bash
export LETTA_BASE_URL=http://127.0.0.1:8283
# 先确保 letta server 已启动且可访问
```

---

## 2. 一条命令跑完四后端全 benchmark

```bash
python run_full_benchmark_all_backends.py
```

你会看到这种可视化调试输出：

- 每一步都有大标题：`[x/12] backend=...`
- 每行子进程日志都有前缀：`[backend/script] ...`
- 每步都会输出日志路径：`[DEBUG] log file: ...`
- 最后有总汇总表（每个脚本耗时 + 状态）

所有原始日志会保存到：

- `logs/full_benchmark/`

---

## 3. 常用变体命令

只跑某几个后端（例如 memgpt + lightrag）：

```bash
python run_full_benchmark_all_backends.py --backends memgpt lightrag
```

自定义日志目录：

```bash
python run_full_benchmark_all_backends.py --logs-dir logs/full_benchmark_2026_04_18
```

---

## 4. 失败时怎么查（最短路径）

1. 先看终端最后一行失败脚本名。
2. 打开对应日志文件（`logs/full_benchmark/*.log`）。
3. 如果是导入报错：先补依赖。
4. 如果是网络/鉴权报错：检查 `config.yaml` 与 API key。
5. 如果是 MemGPT（Letta）报错：确认 `LETTA_BASE_URL` 和 Letta 服务状态。

---

## 5. 设计原则（为什么这样写）

- **正确性优先**：任一步失败立即停止，避免“半成功”污染实验结论。
- **可视化调试优先**：实时前缀日志 + 独立日志文件 + 总结表，便于定位问题。
- **一条命令可复现**：避免手动逐个脚本跑造成漏跑（尤其是 MemGPT）。
