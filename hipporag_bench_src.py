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
import importlib.util
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from simpleMem_src import get_config, Evidence
from ingest_audit_utils import IngestAuditWriter

# HippoRAG 源码路径（优先 memoRaxis/external，其次 third_party）
_REPO_ROOT = Path(__file__).resolve().parent
_HIPPO_SRC_CANDIDATES = [
    _REPO_ROOT / "memoRaxis" / "external" / "hipporag_repo" / "src",
    _REPO_ROOT / "third_party" / "HippoRAG" / "src",
]
_THIRD_PARTY_HIPPO_ROOT = _REPO_ROOT / "third_party" / "HippoRAG"
_THIRD_PARTY_HIPPO_SETUP = _THIRD_PARTY_HIPPO_ROOT / "setup.py"
_THIRD_PARTY_HIPPO_REQ = _THIRD_PARTY_HIPPO_ROOT / "requirements.txt"

CHUNK_SIZE = 1000       # 预分块目标字符数（与 MAB cs1000 对齐）
DEFAULT_TOP_K = 5


class HippoRAGDependencyError(RuntimeError):
    """HippoRAG 运行时依赖缺失。"""


def _discover_hipporag_source() -> Optional[Path]:
    for candidate in _HIPPO_SRC_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _ensure_hipporag_source_visibility() -> Dict[str, str]:
    """统一 HippoRAG 源码可见性策略：优先直连仓库内 third_party/memoRaxis 源码。"""
    source_root = _discover_hipporag_source()
    if source_root is None:
        return {"mode": "missing_source", "detail": "no source candidate exists"}

    source_root_str = str(source_root)
    if source_root_str not in sys.path:
        sys.path.insert(0, source_root_str)

    spec = importlib.util.find_spec("hipporag")
    if spec is None:
        return {
            "mode": "source_not_visible",
            "source_root": source_root_str,
            "detail": "find_spec('hipporag') returned None",
        }

    origin = str(spec.origin or "unknown")
    return {
        "mode": "source_visible",
        "source_root": source_root_str,
        "module_origin": origin,
    }


def _inspect_editable_install_constraints() -> Dict[str, str]:
    """检查上游 editable install 可能的硬钉版本风险（仅用于诊断提示，不作为运行前置）。"""
    out: Dict[str, str] = {}
    if _THIRD_PARTY_HIPPO_SETUP.exists():
        text = _THIRD_PARTY_HIPPO_SETUP.read_text(encoding="utf-8")
        m = re.search(r'openai==([0-9][^"\']*)', text)
        if m:
            out["setup_openai_pin"] = f"openai=={m.group(1)}"
    if _THIRD_PARTY_HIPPO_REQ.exists():
        req_text = _THIRD_PARTY_HIPPO_REQ.read_text(encoding="utf-8")
        m_req = re.search(r'^\s*openai==([0-9][^\s#]*)', req_text, flags=re.MULTILINE)
        if m_req:
            out["requirements_openai_pin"] = f"openai=={m_req.group(1)}"
    return out


def ensure_hipporag_runtime_dependencies(verbose: bool = True) -> None:
    """前置检查 HippoRAG 运行依赖，缺失时抛出聚焦且可执行的错误。

    检查项：
    1) 仓库内 HippoRAG 源码路径可见性（优先 third_party/memoRaxis 直导入）
    2) `import hipporag`
    3) `from igraph import Graph`
    4) 上游 editable install 风险提示（openai hard pin）
    """
    checks = []
    source_state = _ensure_hipporag_source_visibility()
    editable_constraints = _inspect_editable_install_constraints()

    def _raise(detail: str, *, cause: Optional[BaseException] = None) -> None:
        if verbose:
            print(f"[HippoRAG][Preflight] {' | '.join(checks) if checks else 'no-checks'}")
            if editable_constraints:
                print(f"[HippoRAG][Preflight][NOTE] editable constraints={editable_constraints}")
        if cause is None:
            raise HippoRAGDependencyError(detail)
        raise HippoRAGDependencyError(detail) from cause

    if source_state["mode"] == "missing_source":
        checks.append("source=missing")
        detail = (
            "HippoRAG 源码路径不存在：未找到 memoRaxis/external/hipporag_repo/src 或 "
            "third_party/HippoRAG/src。\n"
            "这是源码路径问题，不是 igraph 问题。\n"
            "请确认 third_party/HippoRAG 已拉取到仓库。"
        )
        _raise(detail)

    if source_state["mode"] == "source_not_visible":
        checks.append("source=not_visible")
        detail = (
            "已找到 HippoRAG 源码，但 Python 仍不可见。\n"
            f"source_root={source_state.get('source_root', '?')}\n"
            "这是源码路径可见性问题；请检查 PYTHONPATH 或启动脚本工作目录。"
        )
        _raise(detail)

    checks.append(f"source=ok({source_state.get('source_root', '?')})")
    checks.append(f"module_origin={source_state.get('module_origin', 'unknown')}")

    try:
        importlib.import_module("hipporag")
        checks.append("hipporag=ok")
    except ModuleNotFoundError as exc:
        # `import hipporag` 过程中也可能因为其内部依赖缺失而抛出 ModuleNotFoundError。
        if exc.name in {"igraph", "python_igraph"}:
            checks.append(f"hipporag=blocked_by_{exc.name}")
            detail = (
                f"hipporag 包已可见，但导入被底层依赖 {exc.name} 阻塞。\n"
                "这属于运行依赖缺失，不是源码路径问题。\n"
                "请安装 python-igraph（PyPI 包名通常是 python-igraph / python_igraph）。\n"
                "示例：pip install python-igraph==0.11.8"
            )
            _raise(detail, cause=exc)
        checks.append(f"hipporag=blocked_by_{exc.name}")
        detail = (
            f"hipporag 导入失败，底层缺少模块: {exc.name}\n"
            "这不是“包不可见”的笼统问题，而是导入链依赖缺失。\n"
            "请先补齐该模块，再重试 StructMemEval。"
        )
        _raise(detail, cause=exc)
    except Exception as exc:
        checks.append(f"hipporag=error({exc})")
        detail = f"HippoRAG 导入失败：{exc}"
        _raise(detail, cause=exc)

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
        _raise(detail, cause=exc)

    if editable_constraints:
        pin_msg = editable_constraints.get("setup_openai_pin") or editable_constraints.get("requirements_openai_pin")
        if pin_msg:
            checks.append(f"editable_install_pin={pin_msg}")

    if verbose:
        print(f"[HippoRAG][Preflight] {' | '.join(checks)}")
        if editable_constraints:
            print(
                "[HippoRAG][Preflight][NOTE] 检测到上游 editable install 依赖硬钉："
                f"{editable_constraints}。若 `pip install -e third_party/HippoRAG` 报 "
                "`No matching distribution found`，请优先使用当前仓库的源码路径直导入方案。"
            )


def _text_to_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """结构保真 chunker：优先按换行切，其次按空格切。"""
    if len(text) <= chunk_size:
        return [text]
    lines = text.splitlines(keepends=True)
    chunks: List[str] = []
    cur = ""
    for line in lines:
        if len(cur) + len(line) <= chunk_size:
            cur += line
            continue
        if cur:
            chunks.append(cur.rstrip("\n"))
            cur = ""
        if len(line) <= chunk_size:
            cur = line
            continue
        start = 0
        while start < len(line):
            end = min(start + chunk_size, len(line))
            if end < len(line):
                cut = line.rfind(" ", start, end)
                if cut > start:
                    end = cut
            piece = line[start:end].rstrip()
            if piece:
                chunks.append(piece)
            start = end + (1 if end < len(line) and line[end:end + 1] == " " else 0)
    if cur.strip():
        chunks.append(cur.rstrip("\n"))
    return chunks


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
        self._buffer_meta: List[Dict[str, Any]] = []
        self._h = None          # lazy init in build_index
        self.backend_mode = "hipporag"
        self._audit_writer = IngestAuditWriter(backend="hipporag", save_dir=save_dir)

        # audit 字段（build_index 后填充）
        self.ingest_time_ms: float = 0.0
        self.ingest_llm_prompt_tokens: int = 0
        self.ingest_llm_completion_tokens: int = 0
        self.ingest_llm_calls: int = 0
        self.ingest_chunks: int = 0
        self.retrieve_count: int = 0

    # ── pre-chunking helpers ──────────────────────────────────────────────

    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """将一段文本按 CHUNK_SIZE 预分块后加入 buffer。返回新增 chunk 数。"""
        chunks = _text_to_chunks(text)
        base_meta = dict(metadata or {})
        source_id = str(base_meta.get("source_id") or f"source-{len(self._buffer_meta):06d}")
        for i, chunk in enumerate(chunks):
            self._buffer.append(chunk)
            self._buffer_meta.append({**base_meta, "source_id": source_id, "chunk_offset": i})
        return len(chunks)

    def add_memory(self, text: str, metadata: Optional[Dict] = None) -> None:
        """兼容旧接口：直接按 CHUNK_SIZE 拆分后缓存。"""
        self.add_text(text, metadata=metadata)

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
        self._audit_writer.write_config(
            {"save_dir": self.save_dir, "chunk_size": CHUNK_SIZE, "openie_mode": "online"}
        )
        for idx, chunk in enumerate(self._buffer):
            meta = self._buffer_meta[idx] if idx < len(self._buffer_meta) else {}
            self._audit_writer.add_chunk(
                {
                    "chunk_id": f"hipporag-{idx:06d}",
                    "chunk_idx": idx,
                    "text": chunk,
                    "source_metadata": meta,
                    "storage_target": {"save_dir": self.save_dir},
                }
            )
        self._h = _build_hipporag(self.save_dir)
        t0 = time.time()
        self._h.index(self._buffer)
        self.ingest_time_ms = (time.time() - t0) * 1000

        self.ingest_llm_calls = self._h._audit_llm_calls
        self.ingest_llm_prompt_tokens = self._h._audit_prompt_tokens
        self.ingest_llm_completion_tokens = self._h._audit_completion_tokens
        self.backend_mode = "hipporag"
        graph_files = sorted(str(p) for p in Path(self.save_dir).rglob("*") if p.is_file())
        self._audit_writer.write_provenance(
            {
                "original_chunk_count": len(self._buffer),
                "indexed_chunk_count": self.ingest_chunks,
                "graph_file_count": len(graph_files),
                "graph_files": graph_files[:200],
            }
        )
        self._audit_writer.finalize(
            summary=self.audit_ingest(),
            storage_manifest={
                "backend": "hipporag",
                "save_dir": self.save_dir,
                "graph_file_count": len(graph_files),
            },
        )
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
            "chunk_boundary_preserved": True,
            "original_chunk_count": len(self._buffer_meta),
            "ingest_audit_run_id": self._audit_writer.run_id,
            "ingest_audit_dir": str(self._audit_writer.root),
        }
