#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LightRAG benchmark backend wrapper (strict mode, no fallback)."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from functools import partial
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
            embedding_dim=int(os.getenv("LIGHTRAG_EMBEDDING_DIM", "1024")),
            max_token_size=int(os.getenv("LIGHTRAG_MAX_EMBED_TOKENS", "8192")),
            func=partial(
                openai_embed,
                model=emb_conf.get("model", "text-embedding-v3"),
                api_key=emb_conf.get("api_key"),
                base_url=emb_conf.get("base_url"),
            ),
        )

        return LightRAG, _llm_model_func, embedding_func

    def build_index(self) -> None:
        if not self._buffer:
            raise ValueError("[LightRAGBenchMemory] buffer is empty, nothing to index.")

        LightRAG, llm_func, embedding_func = self._build_runtime()

        os.makedirs(self.save_dir, exist_ok=True)
        rag = LightRAG(
            working_dir=self.save_dir,
            llm_model_func=llm_func,
            embedding_func=embedding_func,
        )
        self._loop = asyncio.new_event_loop()

        t0 = time.time()
        try:
            self._run_coro(rag.initialize_storages())
            self._run_coro(rag.ainsert("\n\n".join(self._buffer)))
        except Exception as e:
            raise RuntimeError(
                "[LightRAGBenchMemory] build_index 失败。该问题通常不是 API 接口错误，而是并发/事件循环复用导致的异步锁冲突。"
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
        self._backend = None
        if self._loop is not None:
            self._loop.close()
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

    def audit_ingest(self) -> Dict[str, Any]:
        return {
            "backend_mode": self.backend_mode,
            "ingest_chunks": len(self._buffer),
            "ingest_time_ms": self._ingest_time_ms,
            "ingest_llm_calls": 0,
            "ingest_llm_prompt_tokens": 0,
            "ingest_llm_completion_tokens": 0,
        }
