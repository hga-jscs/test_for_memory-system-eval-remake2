#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LightRAG benchmark backend wrapper (strict mode, no fallback)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from functools import partial
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from simpleMem_src import Evidence, get_config
from ingest_audit_utils import IngestAuditWriter

_REPO_ROOT = Path(__file__).resolve().parent
_LIGHTRAG_CANDIDATES = [
    _REPO_ROOT / "memoRaxis" / "external" / "lightrag_repo",
    _REPO_ROOT / "third_party" / "LightRAG",
]
for _repo in _LIGHTRAG_CANDIDATES:
    if _repo.exists() and str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
        break


def _safe_console_text(value: Any, limit: int = 240) -> str:
    text = str(value)
    if len(text) > limit:
        text = text[:limit] + "..."
    return text.encode("ascii", "backslashreplace").decode("ascii")


def _safe_print(*parts: Any) -> None:
    line = " ".join(_safe_console_text(p) for p in parts)
    print(line, flush=True)


class LightRAGBenchMemory:
    def __init__(self, save_dir: str = "/tmp/lightrag_bench", chunk_size: int = 1000):
        self.save_dir = save_dir
        self.chunk_size = chunk_size
        self._buffer: List[str] = []
        self._buffer_meta: List[Dict[str, Any]] = []
        self._backend = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.backend_mode = "uninitialized"
        self._ingest_time_ms = 0
        self._embedding_meta: Dict[str, Any] = {}
        self._retrieve_calls = 0
        self._retrieve_time_ms_total = 0
        self._retrieve_last_count = 0
        self._audit_writer = IngestAuditWriter(backend="lightrag", save_dir=save_dir)
        self._usage: Dict[str, Any] = {
            "ingest_llm_calls": "unknown_not_exposed_by_upstream",
            "ingest_llm_prompt_tokens": "unknown_not_exposed_by_upstream",
            "ingest_llm_completion_tokens": "unknown_not_exposed_by_upstream",
        }

    @staticmethod
    def _check_runtime_deps() -> None:
        missing: list[str] = []
        if find_spec("lightrag") is None:
            missing.append("lightrag")
        if missing:
            missing_str = ", ".join(missing)
            raise RuntimeError(
                "[LightRAGBenchMemory] 缺少必需依赖模块："
                f"{missing_str}。请先按 docs/backend_runtime_guide.md 安装 LightRAG 依赖。"
            )

    def add_memory(self, text: str, metadata: Optional[Dict] = None) -> None:
        self.add_text(text, metadata=metadata)

    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        chunks = self._text_to_chunks(text)
        base_meta = dict(metadata or {})
        source_id = str(base_meta.get("source_id") or f"source-{len(self._buffer_meta):06d}")
        for i, chunk in enumerate(chunks):
            self._buffer.append(chunk)
            self._buffer_meta.append({**base_meta, "source_id": source_id, "chunk_offset": i})
        return len(chunks)

    def _text_to_chunks(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]
        chunks = []
        while text:
            if len(text) <= self.chunk_size:
                chunks.append(text)
                break
            cut = text.rfind("\n", 0, self.chunk_size)
            if cut <= 0:
                cut = text.rfind(" ", 0, self.chunk_size)
            if cut <= 0:
                cut = self.chunk_size
            chunks.append(text[:cut].rstrip())
            text = text[cut:].lstrip()
        return chunks

    def _build_runtime(self):
        try:
            from lightrag import LightRAG
            from lightrag.utils import EmbeddingFunc
            from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        except Exception as e:
            raise RuntimeError(
                "[LightRAGBenchMemory] 无法导入 LightRAG 依赖，请先完成 docs/backend_runtime_guide.md 的 LightRAG 安装步骤。"
            ) from e

        conf = get_config()
        llm_conf = conf.llm
        emb_conf = conf.embedding
        embedding_model = emb_conf.get("model", "text-embedding-v3")
        configured_dim_raw = emb_conf.get("dim", 1024)
        try:
            configured_dim = int(configured_dim_raw)
        except (TypeError, ValueError) as e:
            raise RuntimeError(
                "[LightRAGBenchMemory] embedding.dim 配置非法，必须是整数。"
                f"current_value={configured_dim_raw!r} source=config.yaml:embedding.dim model={embedding_model}"
            ) from e
        if configured_dim <= 0:
            raise RuntimeError(
                "[LightRAGBenchMemory] embedding.dim 配置非法，必须是正整数。"
                f"current_value={configured_dim} source=config.yaml:embedding.dim model={embedding_model}"
            )
        config_source = "config.yaml:embedding.dim"
        self._embedding_meta = {
            "model": embedding_model,
            "dim": configured_dim,
            "source": config_source,
        }

        async def _llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            return await openai_complete_if_cache(
                model=llm_conf.get("model"),
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key=llm_conf.get("api_key"),
                base_url=llm_conf.get("base_url"),
                **kwargs,
            )

        embed_callable = openai_embed
        if hasattr(openai_embed, "func") and callable(getattr(openai_embed, "func")):
            # Avoid nested EmbeddingFunc wrapping:
            # openai_embed itself is already an EmbeddingFunc(declared_dim=1536 by decorator).
            # If we partial(openai_embed, ...), inner wrapper can override our configured dim.
            embed_callable = openai_embed.func

        embedding_func = EmbeddingFunc(
            embedding_dim=configured_dim,
            max_token_size=int(os.getenv("LIGHTRAG_MAX_EMBED_TOKENS", "8192")),
            func=partial(
                embed_callable,
                model=embedding_model,
                api_key=emb_conf.get("api_key"),
                base_url=emb_conf.get("base_url"),
            ),
            send_dimensions=True,
            model_name=embedding_model,
        )

        return LightRAG, _llm_model_func, embedding_func

    def _validate_embedding_contract(self, embedding_func: Any) -> None:
        assert self._loop is not None
        probe_text = ["[LightRAGBenchMemory] embedding dimension probe"]
        model_name = self._embedding_meta.get("model", "<unknown>")
        source = self._embedding_meta.get("source", "config")
        expected_dim = int(self._embedding_meta.get("dim", embedding_func.embedding_dim))
        try:
            vectors = self._run_coro(embedding_func(probe_text))
        except Exception as e:
            raise RuntimeError(
                "[LightRAGBenchMemory] embedding 预检失败。"
                f" expected_dim={expected_dim} actual_dim=unknown model={model_name} config_source={source}"
            ) from e

        actual_dim = getattr(vectors, "shape", [0, 0])[1] if getattr(vectors, "ndim", 0) == 2 else None
        if actual_dim != expected_dim:
            raise RuntimeError(
                "[LightRAGBenchMemory] embedding 维度不匹配。"
                f" expected_dim={expected_dim} actual_dim={actual_dim} model={model_name} config_source={source}"
            )
        _safe_print(
            "[LightRAGBenchMemory][DEBUG]",
            f"embedding_probe_ok expected_dim={expected_dim}",
            f"actual_dim={actual_dim}",
            f"model={model_name}",
            f"source={source}",
        )

    def build_index(self) -> None:
        if not self._buffer:
            raise ValueError("[LightRAGBenchMemory] buffer is empty, nothing to index.")

        self._check_runtime_deps()
        LightRAG, llm_func, embedding_func = self._build_runtime()
        _safe_print(
            "[LightRAGBenchMemory][DEBUG]",
            f"embedding_config model={self._embedding_meta.get('model')}",
            f"dim={self._embedding_meta.get('dim')}",
            f"source={self._embedding_meta.get('source')}",
        )
        self._audit_writer.write_config(
            {
                "save_dir": self.save_dir,
                "chunk_size": self.chunk_size,
                "embedding_meta": self._embedding_meta,
            }
        )

        os.makedirs(self.save_dir, exist_ok=True)
        self._loop = asyncio.new_event_loop()
        self._validate_embedding_contract(embedding_func)
        rag = LightRAG(
            working_dir=self.save_dir,
            llm_model_func=llm_func,
            embedding_func=embedding_func,
        )

        t0 = time.time()
        try:
            self._run_coro(rag.initialize_storages())
            for idx, chunk in enumerate(self._buffer):
                meta = self._buffer_meta[idx] if idx < len(self._buffer_meta) else {}
                chunk_id = f"lightrag-{idx:06d}"
                payload = (
                    f"[chunk_id={chunk_id}]"
                    f"[source_id={meta.get('source_id', 'unknown')}]"
                    f"[case_id={meta.get('case_id', 'unknown')}]"
                    f"[user_id={meta.get('user_id', 'unknown')}]"
                    f"[conv_id={meta.get('conv_id', 'unknown')}]\n"
                    f"{chunk}"
                )
                self._run_coro(rag.ainsert(payload))
                self._audit_writer.add_chunk(
                    {
                        "chunk_id": chunk_id,
                        "chunk_idx": idx,
                        "text": chunk,
                        "source_metadata": meta,
                        "storage_target": {"working_dir": self.save_dir},
                    }
                )
        except Exception as e:
            err_text = str(e)
            if "Embedding dimension mismatch detected" in err_text:
                actual_dim = "unknown"
                matched = re.search(r"total elements \((\d+)\).*expected dimension \((\d+)\)", err_text)
                if matched:
                    total_elements = int(matched.group(1))
                    expected_dim = int(self._embedding_meta.get("dim") or 0)
                    if expected_dim > 0 and total_elements % expected_dim == 0:
                        actual_dim = str(total_elements // expected_dim)
                raise RuntimeError(
                    "[LightRAGBenchMemory] build_index 失败：embedding 维度不匹配。"
                    f" expected_dim={self._embedding_meta.get('dim')} actual_dim={actual_dim} "
                    f"model={self._embedding_meta.get('model')} config_source={self._embedding_meta.get('source')}"
                ) from e
            if "bound to a different event loop" in err_text:
                raise RuntimeError(
                    "[LightRAGBenchMemory] build_index 失败：检测到 asyncio.Lock 绑定不同事件循环。"
                    "请确保单进程串行执行，并为每个 case 使用唯一 working_dir。"
                ) from e
            raise RuntimeError(
                "[LightRAGBenchMemory] build_index 失败。请查看上游异常以确认是 embedding/LLM 兼容问题还是事件循环冲突。"
            ) from e

        self._backend = rag
        self.backend_mode = "real_lightrag_data_api_v1"
        self._ingest_time_ms = int((time.time() - t0) * 1000)
        graph_files = sorted(str(p) for p in Path(self.save_dir).rglob("*") if p.is_file())
        self._audit_writer.write_provenance(
            {
                "original_chunk_count": len(self._buffer),
                "working_dir_file_count": len(graph_files),
                "working_dir_files": graph_files[:200],
            }
        )
        self._audit_writer.finalize(
            summary=self.audit_ingest(),
            storage_manifest={
                "backend": "lightrag",
                "working_dir": self.save_dir,
                "file_count": len(graph_files),
            },
        )
        _safe_print("[LightRAGBenchMemory][DEBUG]", f"indexed chunks={len(self._buffer)}", f"time_ms={self._ingest_time_ms}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        if self._backend is None:
            raise RuntimeError("[LightRAGBenchMemory] backend is not ready, call build_index() first.")
        assert self._loop is not None
        t0 = time.time()
        try:
            from lightrag import QueryParam

            data = self._run_coro(
                self._backend.aquery_data(
                    query,
                    param=QueryParam(mode="hybrid", top_k=top_k, chunk_top_k=top_k, only_need_context=True),
                )
            )
        except Exception as e:
            raise RuntimeError(
                "[LightRAGBenchMemory] retrieve 失败，请根据 docs/backend_runtime_guide.md 排查。"
            ) from e

        if not isinstance(data, dict):
            raise RuntimeError(f"[LightRAGBenchMemory] retrieve 返回结构异常: {type(data)}")

        status = str(data.get("status", ""))
        if status and status.lower() != "success":
            raise RuntimeError(f"[LightRAGBenchMemory] retrieve 数据接口失败: {data.get('message', '<unknown>')}")

        payload = data.get("data", {}) if isinstance(data.get("data", {}), dict) else {}
        chunks = payload.get("chunks", []) if isinstance(payload.get("chunks", []), list) else []
        entities = payload.get("entities", []) if isinstance(payload.get("entities", []), list) else []
        relationships = payload.get("relationships", []) if isinstance(payload.get("relationships", []), list) else []

        evidences: List[Evidence] = []

        for chunk in chunks[:top_k]:
            content = str(chunk.get("content", "")).strip()
            if not content:
                continue
            evidences.append(
                Evidence(
                    content=content,
                    metadata={
                        "source": "LightRAGChunk",
                        "backend_mode": self.backend_mode,
                        "chunk_id": chunk.get("chunk_id"),
                        "file_path": chunk.get("file_path"),
                        "reference_id": chunk.get("reference_id"),
                        "score": None,
                    },
                )
            )

        for ent in entities:
            if len(evidences) >= top_k:
                break
            ent_name = str(ent.get("entity_name", "")).strip()
            ent_desc = str(ent.get("description", "")).strip()
            if not ent_name and not ent_desc:
                continue
            evidences.append(
                Evidence(
                    content=f"ENTITY {ent_name}: {ent_desc}".strip(),
                    metadata={
                        "source": "LightRAGEntity",
                        "backend_mode": self.backend_mode,
                        "entity_type": ent.get("entity_type"),
                        "reference_id": ent.get("reference_id"),
                        "score": None,
                    },
                )
            )

        for rel in relationships:
            if len(evidences) >= top_k:
                break
            src = str(rel.get("src_id", "")).strip()
            tgt = str(rel.get("tgt_id", "")).strip()
            desc = str(rel.get("description", "")).strip()
            if not (src or tgt or desc):
                continue
            evidences.append(
                Evidence(
                    content=f"REL {src} -> {tgt}: {desc}".strip(),
                    metadata={
                        "source": "LightRAGRelation",
                        "backend_mode": self.backend_mode,
                        "keywords": rel.get("keywords"),
                        "reference_id": rel.get("reference_id"),
                        "score": rel.get("weight"),
                    },
                )
            )

        elapsed_ms = int((time.time() - t0) * 1000)
        self._retrieve_calls += 1
        self._retrieve_time_ms_total += elapsed_ms
        self._retrieve_last_count = len(evidences)

        _safe_print(
            "[LightRAGBenchMemory][DEBUG]",
            f"retrieve query={query}",
            f"top_k={top_k}",
            f"chunks={len(chunks)}",
            f"entities={len(entities)}",
            f"relations={len(relationships)}",
            f"returned={len(evidences)}",
            f"elapsed_ms={elapsed_ms}",
        )
        return evidences

    def reset(self) -> None:
        self._buffer = []
        self._buffer_meta = []
        if self._backend is not None:
            self._shutdown_backend()
        self._backend = None
        if self._loop is not None:
            self._shutdown_loop()
            self._loop = None
        self._ingest_time_ms = 0
        self._retrieve_calls = 0
        self._retrieve_time_ms_total = 0
        self._retrieve_last_count = 0
        self.backend_mode = "uninitialized"

    def _run_coro(self, coro: Any) -> Any:
        """Run all async operations on one dedicated loop.

        Reusing multiple asyncio.run() calls for the same LightRAG instance can bind
        internal asyncio.Lock objects to different loops (notably doc_status:default_key).
        """
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def _shutdown_backend(self) -> None:
        if self._backend is None:
            return
        if self._loop is None:
            self._loop = asyncio.new_event_loop()

        try:
            finalize = getattr(self._backend, "finalize_storages", None)
            if callable(finalize):
                self._run_coro(finalize())
        except Exception as e:  # noqa: BLE001 - cleanup path best effort
            _safe_print(f"[LightRAGBenchMemory][WARN] finalize_storages failed: {e}")

        try:
            llm_func = getattr(self._backend, "llm_model_func", None)
            shutdown = getattr(llm_func, "shutdown", None)
            if callable(shutdown):
                self._run_coro(shutdown())
        except Exception as e:  # noqa: BLE001 - cleanup path best effort
            _safe_print(f"[LightRAGBenchMemory][WARN] llm_model_func.shutdown failed: {e}")

        try:
            from lightrag.kg.shared_storage import finalize_share_data

            finalize_share_data()
        except Exception as e:  # noqa: BLE001 - cleanup path best effort
            _safe_print(f"[LightRAGBenchMemory][WARN] finalize_share_data failed: {e}")

    def _shutdown_loop(self) -> None:
        assert self._loop is not None
        loop = self._loop
        try:
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        except Exception:
            pending = []
        if pending:
            _safe_print(f"[LightRAGBenchMemory][DEBUG] cancelling pending_tasks={len(pending)}")
            for task in pending:
                task.cancel()
            try:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception as e:  # noqa: BLE001 - cleanup path best effort
                _safe_print(f"[LightRAGBenchMemory][WARN] pending task cleanup failed: {e}")

        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

    def audit_ingest(self) -> Dict[str, Any]:
        return {
            "backend_mode": self.backend_mode,
            "ingest_chunks": len(self._buffer),
            "ingest_time_ms": self._ingest_time_ms,
            **self._usage,
            "embedding_model": self._embedding_meta.get("model"),
            "embedding_dim": self._embedding_meta.get("dim"),
            "embedding_config_source": self._embedding_meta.get("source"),
            "ingest_audit_run_id": self._audit_writer.run_id,
            "ingest_audit_dir": str(self._audit_writer.root),
        }

    def audit_retrieve(self) -> Dict[str, Any]:
        return {
            "backend_mode": self.backend_mode,
            "retrieve_calls": self._retrieve_calls,
            "retrieve_time_ms_total": self._retrieve_time_ms_total,
            "retrieve_last_count": self._retrieve_last_count,
            "retrieve_path": "LightRAG.aquery_data(mode=hybrid, only_need_context=True)",
            "retrieve_llm_calls": 0,
            "retrieve_llm_prompt_tokens": 0,
            "retrieve_llm_completion_tokens": 0,
        }
