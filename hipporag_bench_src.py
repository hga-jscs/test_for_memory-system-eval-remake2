# -*- coding: utf-8 -*-
"""HippoRAG memory backend for AgeMem external benchmarks.

核心设计：
- openie_mode='online'：必须开启，这是 HippoRAG 的定义特性（OpenIE 三元组提取 + PPR 检索）
- chunk_size=1000：预分块到 ~1000 char，与 MAB hipporag_cs1000 对齐（cs1000 相比 default 有 32× AR 提升）
- 隔离：每个 user/case 必须 shutil.rmtree(save_dir) + 重建 HippoRAG 实例（仅清 buffer 不够）
- audit：patch openai_client.chat.completions.create 追踪 ingest LLM tokens
"""

import os
import sys
import shutil
import time
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from simpleMem_src import get_config, Evidence

# HippoRAG 源码路径（优先 memoRaxis/external，其次 third_party）
_REPO_ROOT = Path(__file__).resolve().parent
_HIPPO_SRC_CANDIDATES = [
    _REPO_ROOT / "memoRaxis" / "external" / "hipporag_repo" / "src",
    _REPO_ROOT / "third_party" / "HippoRAG" / "src",
]
for _hippo_src in _HIPPO_SRC_CANDIDATES:
    if _hippo_src.exists() and str(_hippo_src) not in sys.path:
        sys.path.insert(0, str(_hippo_src))
        break

CHUNK_SIZE = 1000       # 预分块目标字符数（与 MAB cs1000 对齐）
DEFAULT_TOP_K = 5


class HippoRAGDependencyError(RuntimeError):
    """HippoRAG 运行时依赖缺失。"""


def ensure_hipporag_runtime_dependencies(verbose: bool = True) -> None:
    """前置检查 HippoRAG 运行依赖，缺失时抛出聚焦且可执行的错误。

    检查项：
    1) `import hipporag`
    2) `from igraph import Graph`
    """
    checks = []

    try:
        importlib.import_module("hipporag")
        checks.append("hipporag=ok")
    except ModuleNotFoundError as exc:
        # `import hipporag` 过程中也可能因为其内部依赖缺失而抛出 ModuleNotFoundError。
        if exc.name in {"igraph", "python_igraph"}:
            checks.append(f"hipporag=blocked_by_{exc.name}")
            detail = (
                "缺少 igraph/python-igraph 依赖，HippoRAG StructMemEval 无法运行。\n"
                "请安装 python-igraph（PyPI 包名通常是 python-igraph / python_igraph）。\n"
                "示例：pip install python-igraph==0.11.8"
            )
            raise HippoRAGDependencyError(detail) from exc
        checks.append(f"hipporag=missing({exc})")
        detail = (
            "缺少 hipporag 包（或源码路径不可见），HippoRAG backend 无法运行。\n"
            "请确认 third_party/HippoRAG/src 已在 PYTHONPATH，或执行：\n"
            "  pip install -e third_party/HippoRAG"
        )
        raise HippoRAGDependencyError(detail) from exc
    except Exception as exc:
        checks.append(f"hipporag=error({exc})")
        detail = f"HippoRAG 导入失败：{exc}"
        raise HippoRAGDependencyError(detail) from exc

    try:
        from igraph import Graph  # type: ignore  # noqa: F401
        checks.append("igraph=ok")
    except Exception as exc:
        checks.append(f"igraph=missing({exc})")
        detail = (
            "缺少 igraph/python-igraph 依赖，HippoRAG StructMemEval 无法运行。\n"
            "请安装 python-igraph（PyPI 包名通常是 python-igraph / python_igraph）。\n"
            "示例：pip install python-igraph==0.11.8"
        )
        raise HippoRAGDependencyError(detail) from exc

    if verbose:
        print(f"[HippoRAG][Preflight] {' | '.join(checks)}")


def _text_to_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """按词边界将长文本切成 ~chunk_size char 的片段。"""
    words = text.split()
    chunks, cur = [], []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + 1 > chunk_size and cur:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(w)
        cur_len += len(w) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [text[:chunk_size]]


def _patch_llm_tracking(h) -> None:
    """Monkey-patch h.llm_model.openai_client 以追踪实际 API 调用的 token 数。

    Cache hit 不经过 openai_client（cost=0），不计入统计，符合成本审计语义。
    调用后可读 h._audit_prompt_tokens / h._audit_completion_tokens / h._audit_llm_calls。
    """
    h._audit_prompt_tokens = 0
    h._audit_completion_tokens = 0
    h._audit_llm_calls = 0

    orig_create = h.llm_model.openai_client.chat.completions.create

    def _tracked_create(**params):
        resp = orig_create(**params)
        h._audit_llm_calls += 1
        if resp.usage:
            h._audit_prompt_tokens += resp.usage.prompt_tokens
            h._audit_completion_tokens += resp.usage.completion_tokens
        return resp

    h.llm_model.openai_client.chat.completions.create = _tracked_create


def _build_hipporag(save_dir: str):
    """创建并返回一个 HippoRAG 实例（DashScope 直连，无需 embedding proxy）。"""
    ensure_hipporag_runtime_dependencies(verbose=False)
    from hipporag import HippoRAG  # type: ignore
    from hipporag.utils.config_utils import BaseConfig  # type: ignore

    conf = get_config()
    llm_conf = conf.llm
    emb_conf = conf.embedding

    os.environ["OPENAI_API_KEY"] = llm_conf.get("api_key", "")

    global_config = BaseConfig(
        llm_name=llm_conf.get("model", "qwen-plus"),
        llm_base_url=llm_conf.get("base_url"),
        embedding_model_name=emb_conf.get("model", "text-embedding-v3"),
        embedding_base_url=emb_conf.get("base_url"),
        openie_mode="online",           # 核心特性：必须开启
        save_dir=save_dir,
        force_index_from_scratch=True,
        force_openie_from_scratch=True,
        embedding_batch_size=8,         # DashScope text-embedding-v3 最大 batch=10，留 margin
        seed=42,
    )
    h = HippoRAG(global_config=global_config)
    _patch_llm_tracking(h)
    return h


class HippoRAGMemory:
    """HippoRAG 记忆体封装，接口与 Mem0RAGMemory / SimpleRAGMemory 兼容。

    使用方式：
        mem = HippoRAGMemory(save_dir="/tmp/hippo_user_001")
        # ingest
        for chunk in chunks:
            mem.add_memory(chunk)
        t0 = time.time()
        mem.build_index()       # 触发 OpenIE + 图构建（昂贵）
        ingest_ms = (time.time() - t0) * 1000
        # retrieve
        evidences = mem.retrieve("query text", top_k=5)
        # reset（下一个 user 前必须调用）
        mem.reset()
    """

    def __init__(self, save_dir: str, top_k_default: int = DEFAULT_TOP_K):
        self.save_dir = save_dir
        self.top_k_default = top_k_default
        self._buffer: List[str] = []
        self._h = None          # lazy init in build_index
        self.backend_mode = "hipporag"

        # audit 字段（build_index 后填充）
        self.ingest_time_ms: float = 0.0
        self.ingest_llm_prompt_tokens: int = 0
        self.ingest_llm_completion_tokens: int = 0
        self.ingest_llm_calls: int = 0
        self.ingest_chunks: int = 0
        self.retrieve_count: int = 0

    # ── pre-chunking helpers ──────────────────────────────────────────────

    def add_text(self, text: str) -> int:
        """将一段文本按 CHUNK_SIZE 预分块后加入 buffer。返回新增 chunk 数。"""
        chunks = _text_to_chunks(text)
        self._buffer.extend(chunks)
        return len(chunks)

    def add_memory(self, text: str, metadata: Optional[Dict] = None) -> None:
        """兼容旧接口：直接按 CHUNK_SIZE 拆分后缓存。"""
        self.add_text(text)

    # ── index build ──────────────────────────────────────────────────────

    def build_index(self) -> None:
        """构建 HippoRAG 图索引（昂贵操作，整个 user/case 所有 chunk 一次性传入）。"""
        ensure_hipporag_runtime_dependencies(verbose=False)

        if not self._buffer:
            raise ValueError("[HippoRAGMemory] buffer is empty, nothing to index.")

        # 清理上一次留下的索引目录（防污染）
        if Path(self.save_dir).exists():
            shutil.rmtree(self.save_dir)
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        self.ingest_chunks = len(self._buffer)
        self._h = _build_hipporag(self.save_dir)
        t0 = time.time()
        self._h.index(self._buffer)
        self.ingest_time_ms = (time.time() - t0) * 1000

        self.ingest_llm_calls = self._h._audit_llm_calls
        self.ingest_llm_prompt_tokens = self._h._audit_prompt_tokens
        self.ingest_llm_completion_tokens = self._h._audit_completion_tokens
        self.backend_mode = "hipporag"
        print(f"[HippoRAGMemory][DEBUG] indexed chunks={self.ingest_chunks} llm_calls={self.ingest_llm_calls}")

    # ── retrieve ─────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 0) -> List[Evidence]:
        k = top_k or self.top_k_default
        if self._h is None:
            return []

        out = self._h.retrieve([query], num_to_retrieve=k)

        # 解析返回值
        solutions = out[0] if (isinstance(out, tuple) and len(out) == 2) else out
        if not solutions:
            return []

        sol = solutions[0]
        docs = getattr(sol, "docs", None) or []
        scores = getattr(sol, "doc_scores", None)

        evidences: List[Evidence] = []
        for i, doc in enumerate(docs[:k]):
            score_val = 0.0
            if scores is not None:
                try:
                    score_val = float(scores[i])
                except Exception:
                    pass
            evidences.append(Evidence(
                content=doc,
                metadata={"source": "HippoRAG", "rank": i + 1, "score": score_val},
            ))
        self.retrieve_count += 1
        return evidences

    # ── reset ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """彻底清理：删除索引目录 + 重置 buffer + 丢弃 HippoRAG 实例。"""
        if Path(self.save_dir).exists():
            shutil.rmtree(self.save_dir)
        self._buffer = []
        self._h = None
        self.ingest_time_ms = 0.0
        self.ingest_llm_prompt_tokens = 0
        self.ingest_llm_completion_tokens = 0
        self.ingest_llm_calls = 0
        self.ingest_chunks = 0
        self.retrieve_count = 0

    # ── audit summary ────────────────────────────────────────────────────

    def audit_ingest(self) -> Dict[str, Any]:
        return {
            "backend_mode": self.backend_mode,
            "ingest_chunks": self.ingest_chunks,
            "ingest_time_ms": round(self.ingest_time_ms),
            "ingest_llm_calls": self.ingest_llm_calls,
            "ingest_llm_prompt_tokens": self.ingest_llm_prompt_tokens,
            "ingest_llm_completion_tokens": self.ingest_llm_completion_tokens,
        }
