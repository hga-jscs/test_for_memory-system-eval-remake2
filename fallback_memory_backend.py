#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight fallback retrieval backend for benchmark continuity.

用于上游后端在当前环境不可用时（依赖缺失、代理阻断、API 变更）提供最小兼容层，
保证 benchmark 全链路可运行并产出结构化结果。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class Evidence:
    content: str
    metadata: dict


class FallbackMemoryBackend:
    def __init__(self, backend_name: str, chunk_size: int = 1000):
        self.backend_name = backend_name
        self.chunk_size = chunk_size
        self._buffer: List[str] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._ingest_time_ms = 0

    def add_memory(self, text: str, metadata: Dict[str, Any] | None = None) -> None:
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
        if not self._buffer:
            return
        t0 = time.time()
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._matrix = self._vectorizer.fit_transform(self._buffer)
        self._ingest_time_ms = int((time.time() - t0) * 1000)

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        if self._vectorizer is None or self._matrix is None:
            return []
        qv = self._vectorizer.transform([query])
        scores = (self._matrix @ qv.T).toarray().reshape(-1)
        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
        evidences = []
        for rank, idx in enumerate(ranked, start=1):
            evidences.append(Evidence(
                content=self._buffer[idx],
                metadata={"source": f"{self.backend_name}-fallback", "rank": rank, "score": float(scores[idx])},
            ))
        return evidences

    def reset(self) -> None:
        self._buffer = []
        self._vectorizer = None
        self._matrix = None
        self._ingest_time_ms = 0

    def audit_ingest(self) -> Dict[str, Any]:
        return {
            "ingest_chunks": len(self._buffer),
            "ingest_time_ms": self._ingest_time_ms,
            "ingest_llm_calls": 0,
            "ingest_llm_prompt_tokens": 0,
            "ingest_llm_completion_tokens": 0,
        }
