#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Letta benchmark backend wrapper.

注意：本实现不再使用 TF-IDF fallback。
如果当前环境尚未完成 Letta 服务端配置，会在 build_index 阶段抛出带修复建议的错误。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from simpleMem_src import Evidence

_REPO_ROOT = Path(__file__).resolve().parent
_LETTA_CANDIDATES = [
    _REPO_ROOT / "memoRaxis" / "external" / "letta_repo",
    _REPO_ROOT / "third_party" / "letta",
]
for _repo in _LETTA_CANDIDATES:
    if _repo.exists() and str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
        break


class LettaBenchMemory:
    """Letta memory wrapper (strict mode, no fallback)."""

    def __init__(self, save_dir: str = "/tmp/letta_bench", chunk_size: int = 1000):
        self.save_dir = save_dir
        self.chunk_size = chunk_size
        self._buffer: List[str] = []
        self._backend = None
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

    def build_index(self) -> None:
        t0 = time.time()
        if not self._buffer:
            raise ValueError("[LettaBenchMemory] buffer is empty, nothing to index.")

        try:
            from letta import create_client
        except Exception as e:
            raise RuntimeError(
                "[LettaBenchMemory] 未检测到可用的 Letta SDK。"
                "请先按 docs/backend_runtime_guide.md 的 Letta 小节完成安装与环境变量配置。"
            ) from e

        base_url = os.getenv("LETTA_BASE_URL")
        if not base_url:
            raise RuntimeError(
                "[LettaBenchMemory] 缺少 LETTA_BASE_URL。"
                "请启动 Letta server 并设置 LETTA_BASE_URL，详见 docs/backend_runtime_guide.md。"
            )

        try:
            client = create_client(base_url=base_url)
            self._backend = _LettaInMemoryAdapter(client=client, docs=list(self._buffer))
            self.backend_mode = "letta"
            self._ingest_time_ms = int((time.time() - t0) * 1000)
            print(
                f"[LettaBenchMemory][DEBUG] indexed chunks={len(self._buffer)} "
                f"mode={self.backend_mode} base_url={base_url}"
            )
        except Exception as e:
            raise RuntimeError(
                "[LettaBenchMemory] Letta 索引构建失败。"
                "请检查 server 可达性、认证、模型配置；"
                "并参考 docs/backend_runtime_guide.md 的故障排查步骤。"
            ) from e

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        if self._backend is None:
            raise RuntimeError("[LettaBenchMemory] backend is not ready, call build_index() first.")
        out = self._backend.retrieve(query=query, top_k=top_k)
        print(f"[LettaBenchMemory][DEBUG] query={query[:48]!r} top_k={top_k} hit={len(out)}")
        return out

    def reset(self) -> None:
        self._buffer = []
        self._backend = None
        self._ingest_time_ms = 0
        self.backend_mode = "uninitialized"

    def audit_ingest(self) -> Dict[str, Any]:
        return {
            "backend_mode": self.backend_mode,
            "ingest_chunks": len(self._buffer),
            "ingest_time_ms": self._ingest_time_ms,
            "ingest_llm_calls": 0,
            "ingest_llm_prompt_tokens": 0,
            "ingest_llm_completion_tokens": 0,
        }


class _LettaInMemoryAdapter:
    """占位适配器：用于以 Letta server 可连接性作为硬门槛。

    当前仓库场景下我们只需要 benchmark 的 add/retrieve 契约。
    若后续接入 Letta archival memory API，可在该类内替换为真实写入/检索。
    """

    def __init__(self, client: Any, docs: List[str]):
        self._client = client
        self._docs = docs

        # 轻量连通性探测（失败则抛异常，避免静默退化）
        _ = self._client.list_models(limit=1)

    def retrieve(self, query: str, top_k: int) -> List[Evidence]:
        # 这里使用 Letta 环境可达性作为强约束；真实检索 API 接入前，直接抛出显式错误。
        raise NotImplementedError(
            "[LettaBenchMemory] 当前仅完成 Letta 运行时硬校验，"
            "尚未绑定 archival_memory_search API。"
            "请按 docs/backend_runtime_guide.md 的 'Letta API 绑定' 小节完成接线。"
        )


# Backward compatibility for existing scripts
MemGPTBenchMemory = LettaBenchMemory
