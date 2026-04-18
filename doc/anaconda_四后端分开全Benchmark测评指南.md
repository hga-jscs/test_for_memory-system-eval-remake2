# 在 Anaconda 里“分开做”四后端全 Benchmark 测评（带可视化调试输出）

> 适用仓库：`/workspace/test_for_memory-system-eval-remake2`  
> 你的目标：按 **MemGPT / RAPTOR / LightRAG / HippoRAG** 四个后端 **分开执行** 三大 benchmark（AMemGym / memory-probe / StructMemEval），并且全过程有清晰日志与阶段输出。  
> 原则：**代码正确性 > 运行效率**，优先保证每一步可核验、可复现。

---

## 1. 建立 Anaconda 环境

```bash
conda create -n mem-bench python=3.10 -y
conda activate mem-bench
python -m pip install -U pip setuptools wheel
```

校验解释器：

```bash
which python
python -V
```

---

## 2. 进入项目并安装依赖

```bash
cd /workspace/test_for_memory-system-eval-remake2
```

### 2.1 主依赖

```bash
pip install \
  openai \
  pyyaml \
  numpy \
  pandas \
  scikit-learn \
  tqdm \
  tiktoken \
  json_repair \
  python-dotenv
```

### 2.2 后端相关依赖（建议全装，避免半路报错）

```bash
pip install -r third_party/HippoRAG/requirements.txt
pip install -r third_party/raptor/requirements.txt
pip install -r memory-probe/requirements.txt
pip install -e amemgym
```

---

## 3. 准备配置（必须）

在仓库根目录创建 `config.yaml`（如已存在可跳过）：

```yaml
llm:
  api_key: "你的API_KEY"
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o-mini"

embedding:
  api_key: "你的API_KEY"
  base_url: "https://api.openai.com/v1"
  model: "text-embedding-3-large"

database:
  provider: "local"
```

同时建议导出环境变量：

```bash
export OPENAI_API_KEY="你的API_KEY"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

---

## 4. 先做 smoke test（防止全量白跑）

```bash
python smoke_test_amem.py
python smoke_test_hipporag.py
python smoke_test_raptor.py
python smoke_test_mem0g.py
```

如果这里还不稳定，不建议直接全量 benchmark。

---

## 5. 按“后端分开”执行全 Benchmark（核心）

仓库已提供新脚本：`run_full_benchmark_split_backends.py`。

它的行为是：
1. **先选一个后端**；
2. 在该后端下按顺序执行：`AMemGym -> memory-probe -> StructMemEval`；
3. 打印实时前缀日志（例如 `[raptor/bench_memory_probe_raptor] ...`）；
4. 打印“后端级小结”+“最终总表”；
5. 默认严格失败（任何一步失败立即停，确保正确性）。

### 5.1 一次跑四后端（分阶段）

```bash
python run_full_benchmark_split_backends.py
```

### 5.2 只跑单个后端（推荐你分四次跑）

```bash
python run_full_benchmark_split_backends.py --backends memgpt
python run_full_benchmark_split_backends.py --backends raptor
python run_full_benchmark_split_backends.py --backends lightrag
python run_full_benchmark_split_backends.py --backends hipporag
```

### 5.3 失败后继续（仅用于收集全量失败面）

```bash
python run_full_benchmark_split_backends.py --continue-on-fail
```

> 正式交付时建议不要开该参数，优先“失败即停 + 修复后重跑”。

---

## 6. 可视化调试输出与日志排查

默认日志目录：`logs/full_benchmark_split/`

### 6.1 实时查看

```bash
tail -f logs/full_benchmark_split/01_memgpt_bench_amemgym_memgpt.log
```

### 6.2 检索关键字段

```bash
rg -n "ERROR|WARN|fallback|accuracy|correct|chunks|ingest|retrieve" logs/full_benchmark_split
```

### 6.3 结果文件检查

```bash
ls results_*.json
python -m json.tool results_amemgym_memgpt.json >/dev/null
python -m json.tool results_memory_probe_memgpt.json >/dev/null
python -m json.tool results_structmemeval_memgpt.json >/dev/null
```

---

## 7. 推荐执行策略（你这次需求的最佳实践）

你要的是“四个后端分开做全 benchmark”，建议严格按下面节奏：

1. `memgpt` 全量跑完并检查结果；
2. 修复问题后再跑 `raptor`；
3. 再跑 `lightrag`；
4. 最后 `hipporag`；
5. 每个后端跑完都先检查对应三份 `results_*.json` 是否有效 JSON、指标是否异常。

这样可显著减少“混跑导致定位困难”的问题。

---

## 8. 常见错误与处理

- `config.yaml` 缺失：先补配置再跑。
- 某后端依赖缺失：按第 2 节补装。
- 输出全是 fallback：检查 API、模型可达性、网络与 key。
- 指标异常（全 0 / 全 1）：先排查答案解析与提示词，再排查后端检索。

---

## 9. 一条可直接复制的完整命令（推荐）

```bash
conda activate mem-bench && \
cd /workspace/test_for_memory-system-eval-remake2 && \
python run_full_benchmark_split_backends.py --backends memgpt && \
python run_full_benchmark_split_backends.py --backends raptor && \
python run_full_benchmark_split_backends.py --backends lightrag && \
python run_full_benchmark_split_backends.py --backends hipporag
```

以上就是“分开做”四后端全 benchmark 的标准流程。你后续如果要，我可以再给你补一版“自动汇总四后端三基准总分对比表（CSV + Markdown）”模板。
