#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LightRAG benchmark backend wrapper (strict mode, no fallback)."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from functools import partial
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from simpleMem_src import Evidence, get_config

_REPO_ROOT = Path(__file__).resolve().parent
_LIGHTRAG_CANDIDATES = [
    _REPO_ROOT / "memoRaxis" / "external" / "lightrag_repo",
    _REPO_ROOT / "third_party" / "LightRAG",
]
for _repo in _LIGHTRAG_CANDIDATES:
    if _repo.exists() and str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
        break


class LightRAGBenchMemory:
    def __init__(self, save_dir: str = "/tmp/lightrag_bench", chunk_size: int = 1000):
        self.save_dir = save_dir
        self.chunk_size = chunk_size
        self._buffer: List[str] = []
        self._backend = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.backend_mode = "uninitialized"
        self._ingest_time_ms = 0
        self._embedding_meta: Dict[str, Any] = {}

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
        self.add_text(text)

    def add_text(self, text: str) -> int:
        chunks = self._text_to_chunks(text)
        self._buffer.extend(chunks)
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

        embedding_func = EmbeddingFunc(
            embedding_dim=configured_dim,
            max_token_size=int(os.getenv("LIGHTRAG_MAX_EMBED_TOKENS", "8192")),
            func=partial(
                openai_embed,
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
        print(
            "[LightRAGBenchMemory][DEBUG] "
            f"embedding_probe_ok expected_dim={expected_dim} actual_dim={actual_dim} "
            f"model={model_name} source={source}"
        )

    def build_index(self) -> None:
        if not self._buffer:
            raise ValueError("[LightRAGBenchMemory] buffer is empty, nothing to index.")

        self._check_runtime_deps()
        LightRAG, llm_func, embedding_func = self._build_runtime()
        print(
            "[LightRAGBenchMemory][DEBUG] "
            f"embedding_config model={self._embedding_meta.get('model')} "
            f"dim={self._embedding_meta.get('dim')} source={self._embedding_meta.get('source')}"
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
            self._run_coro(rag.ainsert("\n\n".join(self._buffer)))
        except Exception as e:
            err_text = str(e)
            if "Embedding dimension mismatch detected" in err_text:
                raise RuntimeError(
                    "[LightRAGBenchMemory] build_index 失败：embedding 维度不匹配。"
                    f" expected_dim={self._embedding_meta.get('dim')} actual_dim=unknown "
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
        self.backend_mode = "lightrag"
        self._ingest_time_ms = int((time.time() - t0) * 1000)
        print(f"[LightRAGBenchMemory][DEBUG] indexed chunks={len(self._buffer)} time_ms={self._ingest_time_ms}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        if self._backend is None:
            raise RuntimeError("[LightRAGBenchMemory] backend is not ready, call build_index() first.")
        try:
            from lightrag import QueryParam
            resp = self._run_coro(self._backend.aquery(query, param=QueryParam(mode="hybrid", top_k=top_k)))
        except Exception as e:
            raise RuntimeError(
                "[LightRAGBenchMemory] retrieve 失败，请根据 docs/backend_runtime_guide.md 排查。"
            ) from e

        text = resp if isinstance(resp, str) else str(resp)
        print(f"[LightRAGBenchMemory][DEBUG] query={query[:48]!r} top_k={top_k} resp_chars={len(text)}")
        return [Evidence(content=text, metadata={"source": "LightRAG", "rank": 1, "score": 1.0})]

    def reset(self) -> None:
        self._buffer = []
        if self._backend is not None:
            self._shutdown_backend()
        self._backend = None
        if self._loop is not None:
            self._shutdown_loop()
            self._loop = None
        self._ingest_time_ms = 0
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
            print(f"[LightRAGBenchMemory][WARN] finalize_storages failed: {e}")

        try:
            llm_func = getattr(self._backend, "llm_model_func", None)
            shutdown = getattr(llm_func, "shutdown", None)
            if callable(shutdown):
                self._run_coro(shutdown())
        except Exception as e:  # noqa: BLE001 - cleanup path best effort
            print(f"[LightRAGBenchMemory][WARN] llm_model_func.shutdown failed: {e}")

        try:
            from lightrag.kg.shared_storage import finalize_share_data

            finalize_share_data()
        except Exception as e:  # noqa: BLE001 - cleanup path best effort
            print(f"[LightRAGBenchMemory][WARN] finalize_share_data failed: {e}")

    def _shutdown_loop(self) -> None:
        assert self._loop is not None
        loop = self._loop
        try:
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        except Exception:
            pending = []
        if pending:
            print(f"[LightRAGBenchMemory][DEBUG] cancelling pending_tasks={len(pending)}")
            for task in pending:
                task.cancel()
            try:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception as e:  # noqa: BLE001 - cleanup path best effort
                print(f"[LightRAGBenchMemory][WARN] pending task cleanup failed: {e}")

        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

    def audit_ingest(self) -> Dict[str, Any]:
        return {
            "backend_mode": self.backend_mode,
            "ingest_chunks": len(self._buffer),
            "ingest_time_ms": self._ingest_time_ms,
            "ingest_llm_calls": 0,
            "ingest_llm_prompt_tokens": 0,
            "ingest_llm_completion_tokens": 0,
        }
