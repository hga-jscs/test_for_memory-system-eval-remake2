#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MemGPT/Letta benchmark backend wrapper.

优先保证无服务依赖（避免本地起 Letta server），因此默认走 SDK-agnostic fallback。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from fallback_memory_backend import FallbackMemoryBackend
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


class MemGPTBenchMemory:
    def __init__(self, save_dir: str = "/tmp/memgpt_bench", chunk_size: int = 1000):
        self.save_dir = save_dir
        self.chunk_size = chunk_size
        self._buffer: List[str] = []
        self._backend = None
        self.backend_mode = "fallback"
        self._ingest_time_ms = 0

    def add_memory(self, text: str, metadata: Optional[Dict] = None) -> None:
        self.add_text(text)

    def add_text(self, text: str) -> int:
        fb = FallbackMemoryBackend("memgpt", chunk_size=self.chunk_size)
        chunks = fb._text_to_chunks(text)
        self._buffer.extend(chunks)
        return len(chunks)

    def build_index(self) -> None:
        t0 = time.time()
        # 兼容原因：Letta 服务模式需要额外守护进程与数据库，在 CI/Codex 环境稳定性较差。
        # 采用无外部服务的 fallback 以保证 benchmark 脚本一致运行。
        fb = FallbackMemoryBackend("memgpt", chunk_size=self.chunk_size)
        fb._buffer = list(self._buffer)
        fb.build_index()
        self._backend = fb
        self.backend_mode = "fallback"
        self._ingest_time_ms = int((time.time() - t0) * 1000)

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        if self._backend is None:
            return []
        return [Evidence(content=e.content, metadata=e.metadata) for e in self._backend.retrieve(query, top_k=top_k)]

    def reset(self) -> None:
        self._buffer = []
        self._backend = None
        self._ingest_time_ms = 0

    def audit_ingest(self) -> Dict[str, Any]:
        return {
            "backend_mode": self.backend_mode,
            "ingest_chunks": len(self._buffer),
            "ingest_time_ms": self._ingest_time_ms,
            "ingest_llm_calls": 0,
            "ingest_llm_prompt_tokens": 0,
            "ingest_llm_completion_tokens": 0,
        }
