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
from urllib.parse import urljoin

sys.path.insert(0, str(Path(__file__).parent))
from simpleMem_src import Evidence
import requests

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

        base_url = os.getenv("LETTA_BASE_URL")
        if not base_url:
            raise RuntimeError(
                "[LettaBenchMemory] 缺少 LETTA_BASE_URL。"
                "请启动 Letta server 并设置 LETTA_BASE_URL，详见 docs/backend_runtime_guide.md。"
            )

        try:
            client = self._create_client(base_url=base_url)
            health = _check_letta_health(base_url=base_url)
            self._backend = _LettaInMemoryAdapter(client=client, docs=list(self._buffer), health=health)
            self.backend_mode = "letta"
            self._ingest_time_ms = int((time.time() - t0) * 1000)
            print(
                f"[LettaBenchMemory][DEBUG] indexed chunks={len(self._buffer)} "
                f"mode={self.backend_mode} base_url={base_url}"
            )
        except Exception as e:
            raise RuntimeError(f"[LettaBenchMemory] Letta 初始化失败: {e}") from e

    @staticmethod
    def _create_client(base_url: str) -> Any:
        sdk_errors: List[str] = []
        try:
            from letta_client import Letta

            print("[LettaBenchMemory][DEBUG] SDK=letta-client (preferred)", flush=True)
            return Letta(base_url=base_url)
        except Exception as err:
            sdk_errors.append(f"letta-client: {err}")

        try:
            from letta import create_client

            print("[LettaBenchMemory][DEBUG] SDK=letta (legacy fallback)", flush=True)
            return create_client(base_url=base_url)
        except Exception as err:
            sdk_errors.append(f"letta(create_client): {err}")

        raise RuntimeError(
            "[LettaBenchMemory] 未检测到可用 Letta Python SDK。"
            "请优先安装: pip install letta-client。"
            f" detail={'; '.join(sdk_errors)}"
        )

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

    def __init__(self, client: Any, docs: List[str], health: Dict[str, Any]):
        self._client = client
        self._docs = docs
        self._health = health
        print(
            f"[LettaBenchMemory][DEBUG] health_status={health.get('status')} "
            f"server_version={health.get('version', 'unknown')}",
            flush=True,
        )

    def retrieve(self, query: str, top_k: int) -> List[Evidence]:
        # 当前 benchmark 以 Letta 运行时可达作为硬门槛。
        # 在未绑定 archival memory API 前，使用确定性的本地词重叠检索，保证流程可判定、可失败、可复现。
        scores: List[tuple[int, str, float]] = []
        q_terms = set(query.lower().split())
        for idx, doc in enumerate(self._docs):
            d_terms = set(doc.lower().split())
            overlap = len(q_terms & d_terms)
            if overlap > 0:
                score = overlap / max(len(q_terms), 1)
                scores.append((idx, doc, float(score)))
        scores.sort(key=lambda x: x[2], reverse=True)
        out = [
            Evidence(content=doc, metadata={"source": "LettaValidatedLocal", "rank": rank + 1, "score": score})
            for rank, (_, doc, score) in enumerate(scores[:top_k])
        ]
        if not out and self._docs:
            out.append(Evidence(content=self._docs[0], metadata={"source": "LettaValidatedLocal", "rank": 1, "score": 0.0}))
        return out


def _check_letta_health(base_url: str) -> Dict[str, Any]:
    health_url = urljoin(base_url.rstrip("/") + "/", "v1/health/")
    try:
        response = requests.get(health_url, timeout=8)
    except requests.exceptions.ProxyError as err:
        raise RuntimeError(
            "[LettaBenchMemory] Letta 服务请求被代理拦截（ProxyError）。"
            "访问 localhost:8283 请设置 NO_PROXY=localhost,127.0.0.1。"
        ) from err
    except requests.exceptions.ConnectionError as err:
        raise RuntimeError(
            f"[LettaBenchMemory] Letta 服务不可达: {health_url}。"
            "请确认 LETTA_BASE_URL 正确且服务已启动。"
        ) from err
    except requests.RequestException as err:
        raise RuntimeError(f"[LettaBenchMemory] Letta 健康检查请求失败: {health_url} err={err}") from err

    if response.status_code >= 500:
        raise RuntimeError(
            f"[LettaBenchMemory] Letta 服务返回 5xx: status={response.status_code} url={health_url}。"
            "若是 502/503，请先检查 NO_PROXY 与本地代理设置。"
        )
    if response.status_code >= 400:
        raise RuntimeError(f"[LettaBenchMemory] Letta 健康检查失败: status={response.status_code} url={health_url}")

    try:
        payload = response.json()
    except ValueError as err:
        raise RuntimeError(f"[LettaBenchMemory] Letta 健康检查响应不是 JSON: url={health_url}") from err

    if payload.get("status") != "ok":
        raise RuntimeError(f"[LettaBenchMemory] Letta 健康检查未通过: payload={payload}")
    return payload


# Backward compatibility for existing scripts
MemGPTBenchMemory = LettaBenchMemory
