# 四个后端（Letta / RAPTOR / LightRAG / HippoRAG）运行与排障指南

> 更新时间：2026-04-18

你要求只保留 **letta、raptor、lightrag、hipporag**，并且不要静默降级。
当前代码已改成 **strict mode**：
- 不再自动降级到本地 TF-IDF
- 依赖缺失或配置错误会直接抛出明确错误
- 日志里会输出可视化调试信息（`[DEBUG] ...`）

---

## 1) 通用前置条件

1. Python 3.10+
2. `config.yaml` 已配置 LLM 与 embedding（`api_key/base_url/model`）
3. 建议在虚拟环境运行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

---

## 2) RAPTOR

### 安装建议

```bash
pip install -r third_party/raptor/requirements.txt
```

### 运行检查

```bash
python -c "from raptor_bench_src import RaptorBenchMemory; print('ok')"
```

### 典型报错与处理

- `No module named 'raptor'`
  - 检查 `third_party/raptor` 是否存在
  - 确认依赖安装成功
- OpenAI embedding/LLM 调用失败
  - 检查 `config.yaml` 中 `llm` / `embedding` 的 `base_url` 与 `api_key`

### 调试输出（可视化）

构建索引成功后会打印：
- chunk 数
- tree node 数

---

## 3) HippoRAG

### 安装建议

```bash
pip install -r third_party/HippoRAG/requirements.txt
```

### 运行检查

```bash
python -c "from hipporag_bench_src import HippoRAGMemory; print('ok')"
```

### 典型报错与处理

- `No module named 'hipporag'`
  - 检查 `third_party/HippoRAG/src` 路径是否存在
- OpenIE/LLM 报错
  - 确认 `config.yaml` 中 LLM 可用
  - 确认外网/代理对目标模型端点可达

### 调试输出（可视化）

构建索引成功后会打印：
- ingest chunk 数
- ingest LLM 调用次数

---

## 4) LightRAG

### 安装建议

```bash
pip install -e third_party/LightRAG
```

### 必需参数

- `config.yaml` 里 LLM/embedding 参数可用
- 可选环境变量：
  - `LIGHTRAG_EMBEDDING_DIM`（默认 1024）
  - `LIGHTRAG_MAX_EMBED_TOKENS`（默认 8192）

### 运行检查

```bash
python -c "from lightrag_bench_src import LightRAGBenchMemory; print('ok')"
```

### 典型报错与处理

- `无法导入 LightRAG`
  - 重新执行 `pip install -e third_party/LightRAG`
- `build_index 失败`
  - 大概率是模型端点配置问题或依赖版本冲突
  - 优先检查 `config.yaml`

### 调试输出（可视化）

会打印：
- indexed chunks
- ingest time(ms)
- retrieve 时响应字符长度

---

## 5) Letta

### 当前状态

`memgpt_bench_src.py` 已切换为 **Letta strict mode**：
- 会强校验 `LETTA_BASE_URL` 和 Letta 服务连通性
- 不再走 fallback

### 仍需你本地完成的步骤（否则无法运行）

1. 安装 Letta（本地源码或可用发行版）
2. 启动 Letta server
3. 设置：

```bash
export LETTA_BASE_URL=http://127.0.0.1:8283
```

4. 再执行脚本

### 为什么这里还不能“开箱即用”

因为当前仓库里的 benchmark 适配器还没最终绑定 Letta 的 archival/retrieval API（只做了运行时连通性硬校验）。
这部分已经在代码里用 `NotImplementedError` 明确提示，避免“看起来能跑、实际在退化”。

### 你可以怎么补齐（推荐）

在 `memgpt_bench_src.py` 的 `_LettaInMemoryAdapter.retrieve` 中接 Letta 官方检索 API：
- 写入：archival memory insert
- 检索：archival memory search
- 返回结构映射成 `Evidence(content, metadata)`

---

## 6) 为什么这次要“严格失败”而不是“自动降级”

你的目标是验证四个后端真实能力，不是跑通一个替代实现。
因此现在策略是：
- **依赖/配置不满足就显式报错**
- **报错信息中给出具体修复路径**
- **保留 `[DEBUG]` 输出方便排障**

这比 silent fallback 更适合研究与对比实验。
