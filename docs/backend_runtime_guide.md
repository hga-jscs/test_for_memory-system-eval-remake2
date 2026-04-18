# 四后端（MemGPT / RAPTOR / LightRAG / HippoRAG）运行与排障指南

> 更新时间：2026-04-18

本指南对应当前仓库的四后端全量评测流程，包含 **MemGPT（Letta）**，并提供一条命令跑完全部 benchmark 的方式。

---

## 1) 通用前置条件

1. Python 3.10+
2. `config.yaml` 已配置 LLM 与 embedding（`api_key/base_url/model`）
3. 建议使用虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

---

## 2) RAPTOR

安装：

```bash
pip install -r third_party/raptor/requirements.txt
```

检查：

```bash
python -c "from raptor_bench_src import RaptorBenchMemory; print('ok')"
```

---

## 3) HippoRAG

安装：

```bash
pip install -r third_party/HippoRAG/requirements.txt
```

检查：

```bash
python -c "from hipporag_bench_src import HippoRAGMemory; print('ok')"
```

---

## 4) LightRAG

安装：

```bash
pip install -e third_party/LightRAG
```

检查：

```bash
python -c "from lightrag_bench_src import LightRAGBenchMemory; print('ok')"
```

---

## 5) MemGPT（Letta）

必要条件：

1. 已安装 Letta SDK / 源码依赖
2. Letta server 已启动
3. 设置 `LETTA_BASE_URL`

```bash
export LETTA_BASE_URL=http://127.0.0.1:8283
python -c "from memgpt_bench_src import MemGPTBenchMemory; print('ok')"
```

---

## 6) 一条命令跑完四后端“全 benchmark”

```bash
python run_full_benchmark_all_backends.py
```

说明：
- 会顺序执行 12 个脚本（4 后端 × 3 benchmark）
- 每步都有可视化调试输出（阶段标题 + 前缀日志）
- 每步原始日志保存到 `logs/full_benchmark/`
- 任一步失败会立即停止，保证结果正确性

更多傻瓜式说明见：`docs/full_benchmark_one_command.md`。

---

## 7) 常见问题排查

- `ModuleNotFoundError`
  - 按上面对应后端重新安装依赖
- API 报错（鉴权/连通性）
  - 检查 `config.yaml` 与模型端点
- MemGPT（Letta）连接失败
  - 检查 Letta server 状态与 `LETTA_BASE_URL`
- 想定位某一步具体错误
  - 打开 `logs/full_benchmark/` 中对应 `.log` 文件，按时间倒序排查
