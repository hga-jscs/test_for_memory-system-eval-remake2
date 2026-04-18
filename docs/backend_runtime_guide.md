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

### 1.1 代理与网络分层（必须区分）

这部分是高频踩坑点：**Docker Desktop 代理、Windows 系统代理、Conda 下载代理、localhost 直连是四个层次**，不要混配。

- Docker Desktop 代理：仅影响容器网络，不等于主机 Python/conda 请求。
- Windows 系统代理：可能影响浏览器与部分系统组件，不应默认让本地服务走代理。
- Conda/Pip 代理：应单独配置（例如 `conda config --set proxy_servers...`），不要假设自动继承系统代理就正确。
- 本地服务直连：访问本机 Letta 必须绕过代理。

建议在 shell 中显式设置：

```bash
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1
```

对于 Windows PowerShell：

```powershell
$env:NO_PROXY=\"localhost,127.0.0.1\"
$env:no_proxy=\"localhost,127.0.0.1\"
```

### 1.2 推荐环境隔离策略（四后端分离）

建议每个 backend 使用独立 conda/venv，避免 `openai/httpx`、`lightrag`、`hipporag` 依赖互相污染：

- `env-memgpt`: `letta-client` + benchmark 通用依赖
- `env-raptor`: RAPTOR + benchmark 通用依赖
- `env-lightrag`: LightRAG + benchmark 通用依赖
- `env-hipporag`: HippoRAG + benchmark 通用依赖

这样可以把“依赖冲突”与“后端自身错误”分开定位。

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

可选（仅当你明确需要把 HippoRAG 当作站点包安装时）：

```bash
pip install -e third_party/HippoRAG
```

说明（关键）：
- 当前仓库 HippoRAG backend 的默认策略是 **源码路径直导入**（优先 `memoRaxis/external/hipporag_repo/src`，其次 `third_party/HippoRAG/src`），并不依赖 editable install 才能运行。
- preflight 会在启动时区分并输出：
  - 源码路径缺失 / 不可见；
  - `hipporag` 包导入链缺失模块（例如 `igraph`）；
  - 上游 editable install 的硬钉依赖提示（例如 `openai==1.91.1`）。
- `pip install -e third_party/HippoRAG` 可能因上游 `setup.py/requirements.txt` 中 `openai==1.91.1` 等硬钉失败；这不应阻塞本仓库使用源码路径运行 benchmark。
- **源码路径直导入不会自动安装依赖**，因此即便能 `import hipporag`，运行时仍可能缺少 `python-igraph`。

检查：

```bash
python -c "from hipporag_bench_src import HippoRAGMemory; print('ok')"
python -c "import igraph; print('igraph ok')"
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

### 4.1 DashScope(OpenAI-compatible) 关键兼容点

- `lightrag_bench_src.py` 会优先读取 `config.yaml -> embedding.dim` 作为 LightRAG 的 `embedding_dim`，并在 build_index 前做一次 embedding 维度预检（日志会打印 `expected_dim/actual_dim/model/source`）。
- 若出现维度不匹配，错误会明确给出：
  - `expected_dim`
  - `actual_dim`
  - `model`
  - `config_source`
- LightRAG 上游 `chat.completions.parse` 在部分 OpenAI-compatible SDK/网关（含 DashScope 路线）不可用。当前仓库已对该路径做兼容：
  - 优先调用 `parse`（若客户端支持）
  - 不支持时回退到 `chat.completions.create` + JSON response_format 路径
  - 因此不会再以 `'AsyncCompletions' object has no attribute 'parse'` 作为主错误

---

## 5) MemGPT（Letta）

必要条件：

1. 已安装 Letta Python SDK（优先官方 SDK）
2. Letta server 已启动
3. 设置 `LETTA_BASE_URL`
4. 确保 `NO_PROXY=localhost,127.0.0.1`，避免本地请求被转发到代理

```bash
pip install letta-client
export LETTA_BASE_URL=http://127.0.0.1:8283
python -c "from letta_client import Letta; c=Letta(base_url='http://localhost:8283'); print(c.health())"
```

SDK 导入优先级：

1. `from letta_client import Letta`（推荐）
2. `from letta import create_client`（仅兼容旧代码）

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
  - 若返回 502/503，优先检查 `NO_PROXY=localhost,127.0.0.1`
- 想定位某一步具体错误
  - 打开 `logs/full_benchmark/` 中对应 `.log` 文件，按时间倒序排查

---

## 8) benchmark 成功 / 失败判定规则（强一致）

`bench_*.py` 统一执行健康判定，确保“可判定、可失败、可复现”：

- **PASS（真成功）**
  - 至少有 1 个非 skipped case 完成；
  - 至少有有效 query 被评估；
  - 不是“全部任务失败”。
- **FAIL（真失败）**
  - 没收集到任务；
  - 所有 case 都失败；
  - 没有任何有效 query；
  - 关键初始化失败（如 SDK 缺失、服务不可达、事件循环冲突）。

当健康检查失败时，脚本会返回非零退出码；`run_full_benchmark_split_backends.py` 会将该步骤记为 FAIL，而不是 PASS。
