# 在 Anaconda 里从 0 到跑完整套测评的傻瓜式指南（含 Fallback 修复手册）

> 适用仓库：`test_for_memory-system-eval-remake2`  
> 目标：从创建环境开始，直到三大基准（AMemGym / memory-probe / StructMemEval）全部测评完成，并提供常见 fallback 场景的修复步骤。  
> 重点原则：**先保证正确性，再考虑效率**；每一步都带有**可视化调试输出**，方便你“看得见”进度和问题。

---

## 0. 你将得到什么

跑完本教程后你会得到：

1. 一个可复用的 conda 环境（统一依赖）。
2. 完整的配置文件（`config.yaml`）。
3. 分层验证流程：
   - 先 smoke test（快速验通路）；
   - 再跑三套 benchmark 正式评测。
4. 全部结果文件（`results_*.json`）。
5. 遇到 fallback 或异常时的“对照修复表”。

---

## 1. 创建 Anaconda 环境（从零开始）

> 建议 Python 版本：**3.10**（兼容性通常最稳）。

```bash
# 1) 创建环境
conda create -n mem-eval python=3.10 -y

# 2) 激活环境
conda activate mem-eval

# 3) 升级基础打包工具
python -m pip install -U pip setuptools wheel
```

建议先确认解释器路径没错：

```bash
which python
python -V
```

预期你应该看到当前 python 来自 `.../anaconda3/envs/mem-eval/...`。

---

## 2. 进入仓库并安装依赖

```bash
cd /workspace/test_for_memory-system-eval-remake2
```

### 2.1 安装主流程依赖（推荐一次性执行）

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

### 2.2 安装子模块依赖（按需）

```bash
# HippoRAG
pip install -r third_party/HippoRAG/requirements.txt

# RAPTOR
pip install -r third_party/raptor/requirements.txt

# memory-probe（如需其额外依赖）
pip install -r memory-probe/requirements.txt

# amemgym（本地可编辑安装）
pip install -e amemgym
```

> 如果你只先跑 smoke test，可以先不装全部 third_party 依赖；缺什么再补，节省初次安装时间。

---

## 3. 准备配置文件（必须）

本仓库默认会读取项目根目录下的 `config.yaml`。如果缺失会直接报错。

在仓库根目录创建 `config.yaml`：

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

同时建议导出环境变量（很多子模块也会读）：

```bash
export OPENAI_API_KEY="你的API_KEY"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

---

## 4. 可视化调试输出模板（强烈建议）

为每次执行都保留日志，便于追踪失败点：

```bash
mkdir -p logs
python smoke_test_hipporag.py 2>&1 | tee logs/smoke_hipporag_$(date +%Y%m%d_%H%M%S).log
```

推荐你统一使用以下“可视化观察点”：

- ingest 阶段：
  - `chunks` 是否 > 0
  - `ingest_llm_calls` 是否 > 0（依系统而定）
  - `ingest_time_ms` 是否异常飙高
- retrieve 阶段：
  - `retrieve` 返回数量是否 > 0
  - top_k 结果内容是否和 query 语义相关
- QA 阶段：
  - 预测是否总是同一个选项（典型提示词/解析 bug）
  - 是否大量出现 “I don't know”

辅助命令：

```bash
# 实时看最近日志
tail -f logs/xxx.log

# 提取关键统计行
rg -n "chunks=|ingest|retrieve|correct|accuracy|fallback|WARN|ERROR" logs/xxx.log
```

---

## 5. 先跑 Smoke Test（先通路，再全量）

> 顺序建议：从风险高的系统先验。

```bash
# A-MEM
python smoke_test_amem.py

# HippoRAG
python smoke_test_hipporag.py

# RAPTOR
python smoke_test_raptor.py

# Mem0G
python smoke_test_mem0g.py

# 其他变体（如使用）
python smoke_test_amem_v2.py
```

### Smoke 通过标准（最低）

- 脚本能完整跑完，不中途崩溃。
- 每类数据至少有非空检索结果。
- 日志里出现合理的 `chunks`、`ingest_*`、`correct/total`。

如果 smoke 不通过，不要直接跑全量（否则你大概率白跑几小时）。

---

## 6. 正式跑三大 benchmark（建议顺序）

### 6.1 AMemGym

```bash
# SimpleMem baseline
python bench_amemgym_full.py

# mem0
python bench_amemgym_mem0.py

# HippoRAG
python bench_amemgym_hipporag.py

# RAPTOR
python bench_amemgym_raptor.py

# A-MEM
python bench_amemgym_amem.py

# LightRAG（如需）
python bench_amemgym_lightrag.py
```

### 6.2 memory-probe

```bash
python bench_memory_probe_full.py
python bench_memory_probe_mem0.py
python bench_memory_probe_mem0g.py
python bench_memory_probe_hipporag.py
python bench_memory_probe_raptor.py
python bench_memory_probe_memgpt.py
python bench_memory_probe_amem.py
```

### 6.3 StructMemEval

```bash
python bench_structmemeval.py
python bench_structmemeval_mem0.py
python bench_structmemeval_mem0g.py
python bench_structmemeval_hipporag.py
python bench_structmemeval_raptor.py
python bench_structmemeval_memgpt.py
```

> 建议每个命令都 `| tee logs/xxx.log`，避免出问题后无从定位。

---

## 7. 全部测评完成后的收尾检查（必须做）

```bash
# 看结果文件是否生成
ls results_*.json

# 快速检查 JSON 格式
python -m json.tool results_amemgym_full.json >/dev/null
python -m json.tool results_memory_probe_full.json >/dev/null
python -m json.tool results_structmemeval_full.json >/dev/null
```

建议再做一次一致性检查：

- 是否存在某个系统结果缺失（跑漏了）。
- 是否存在异常“全 0/全 1”准确率（通常是评测逻辑或答案解析错误）。
- 是否存在大量 `error` 或 `fallback` 字段。

---

## 8. Fallback 场景修复手册（重点）

下面是最常见“跑着跑着退化到 fallback / 结果异常”的修复策略。

### 场景 A：后端不可用，自动降级 fallback 检索

**现象**
- 日志出现 `backend_mode=fallback` 或类似提示；
- 检索质量明显下降，但流程仍可跑完。

**原因**
- 上游后端依赖缺失、API 不兼容、网络问题等。

**修复步骤**
1. 先确认是否只是临时网络抖动：重试一次相同命令。  
2. 检查依赖是否完整：`pip install -r ...` 补齐第三方包。  
3. 检查 API 可达与 key 是否可用。  
4. 如果要先保证流程可交付：允许 fallback 跑完并在报告中标记。  
5. 计划复跑：后端恢复后删除旧结果并重跑对应 benchmark。

---

### 场景 B：JSON 解析失败 / LLM 输出不规范

**现象**
- 报错包含 `JSONDecodeError` 或结构化解析失败。

**修复步骤**
1. 安装修复库：
   ```bash
   pip install json_repair
   ```
2. 将解析逻辑改为“先正常 `json.loads`，失败后 `json_repair` 再解析”。
3. 将原始模型输出打印到日志（至少前 500 字符），便于复盘。
4. 对关键字段做 schema 校验（缺字段直接告警，不默默吞掉）。

---

### 场景 C：长跑中断（容器/服务断连）

**现象**
- 跑了 1 小时后突然 `connection refused` / 服务失联；
- 结果文件只写了一部分。

**修复步骤**
1. 确认是否已有“增量保存”；如果没有，先补增量保存再重跑。  
2. 对长任务拆分执行（按 category / user 范围分批跑）。  
3. 每批跑完即落盘，绝不只在最终一次性写结果。  
4. 重启依赖服务后，从上次完成位置继续（resume）。

---

### 场景 D：检索有结果但答案总是错误（或极端偏斜）

**现象**
- 模型总选同一个选项（如总是 E）；
- 准确率异常低，且不是随机波动级别。

**修复步骤**
1. 检查多选题 prompt 是否允许不存在选项。  
2. 检查 answer parser 是否有范围校验（例如只允许 `0..len(choices)-1`）。  
3. 打印每题 `question + choices + raw_pred + parsed_idx`。  
4. 小样本人工对照 20 题，确认是 prompt 问题还是 memory 问题。

---

### 场景 E：tree_based / 时序题表现离谱低

**现象**
- 某类任务接近 0 分，且明显偏离其它系统。

**修复步骤**
1. 先做该 category 的专项 smoke（每类 1~3 case）。  
2. 打印 ingest 后的关键统计（如 atoms/chunks/mem_count）。  
3. 确认输入序列化方式是否正确（是否误丢角色/时间信息）。  
4. 若确认是系统能力边界（非工程 bug），在报告中明确标注，不做“错误修复”。

---

## 9. 推荐执行节奏（最省时间）

1. 建环境 + 配置。  
2. 跑 smoke（每系统 5~15 分钟）。  
3. 修掉所有 smoke 阻塞问题。  
4. 再跑全量 benchmark。  
5. 对异常系统做 targeted rerun（而不是全量重跑）。

---

## 10. 最后给你的“防踩坑清单”

- [ ] 没有 `config.yaml` 不要开跑。  
- [ ] 没有 smoke 通过，不要开全量。  
- [ ] 每个命令必须保存日志。  
- [ ] 每个 benchmark 必须增量写盘。  
- [ ] 出现 fallback 要记录原因与影响范围。  
- [ ] 结果异常先查解析/提示词，再查模型能力。  
- [ ] 正确性优先：宁可慢一点，也不要“看起来跑完但结果不可信”。

---

如果你愿意，我可以下一步直接再给你一版：
1) `config.yaml` 可直接复制的模板（按 OpenAI / OpenRouter 双版本）；
2) 一键执行脚本 `run_all_with_logs.sh`（自动 tee 日志 + 失败重试 + 结果检查）。
