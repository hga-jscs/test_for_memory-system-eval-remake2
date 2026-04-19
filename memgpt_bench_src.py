#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Letta benchmark backend wrapper (REAL Letta archival memory only).

本实现强制走真实 Letta archival memory 写入/检索：
- 禁止本地 overlap fallback
- Letta 连接或 API 失败时直接抛错
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests
from requests import exceptions as req_exc

sys.path.insert(0, str(Path(__file__).parent))
from simpleMem_src import Evidence
from ingest_audit_utils import IngestAuditWriter, compact_error, parse_time_to_iso

_REPO_ROOT = Path(__file__).resolve().parent
_LETTA_CANDIDATES = [
    _REPO_ROOT / "memoRaxis" / "external" / "letta_repo",
    _REPO_ROOT / "third_party" / "letta",
]
for _repo in _LETTA_CANDIDATES:
    if _repo.exists() and str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
        break


def _safe_console_text(value: Any, limit: int = 240) -> str:
    text = str(value)
    if len(text) > limit:
        text = text[:limit] + "..."
    return text.encode("ascii", "backslashreplace").decode("ascii")


def _safe_print(*parts: Any) -> None:
    safe_line = " ".join(_safe_console_text(p) for p in parts)
    print(safe_line, flush=True)


class LettaBenchMemory:
    """Letta memory wrapper (strict REAL backend mode)."""

    def __init__(self, save_dir: str = "/tmp/letta_bench", chunk_size: int = 1000):
        self.save_dir = save_dir
        self.chunk_size = chunk_size
        self._buffer: List[str] = []
        self._buffer_meta: List[Dict[str, Any]] = []

        self._base_url = ""
        self._http: Optional[requests.Session] = None
        self._agent_id: Optional[str] = None
        self._created_passage_ids: List[str] = []
        self._model_handle: Optional[str] = None
        self._embedding_handle: Optional[str] = None
        self._llm_models: List[Dict[str, Any]] = []
        self._embedding_models: List[Dict[str, Any]] = []

        self.backend_mode = "uninitialized"
        self._ingest_time_ms = 0
        self._retrieve_calls = 0
        self._retrieve_time_ms_total = 0
        self._retrieve_last_count = 0

        # Letta archival writes/searches通常不经过 LLM；token usage 并未由这些 endpoint 返回。
        self._usage = {
            "ingest_llm_calls": "not_exposed_by_upstream",
            "ingest_llm_prompt_tokens": "not_exposed_by_upstream",
            "ingest_llm_completion_tokens": "not_exposed_by_upstream",
            "retrieve_llm_calls": 0,
            "retrieve_llm_prompt_tokens": 0,
            "retrieve_llm_completion_tokens": 0,
            "token_usage_note": "Letta archival-memory REST endpoints do not expose token usage in responses.",
        }
        self._ingest_errors: List[Dict[str, Any]] = []
        self._chunk_audit: List[Dict[str, Any]] = []
        self._audit_writer = IngestAuditWriter(backend="memgpt", save_dir=save_dir)

    def add_memory(self, text: str, metadata: Optional[Dict] = None) -> None:
        self.add_text(text, metadata=metadata)

    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        chunks = self._text_to_chunks(text)
        base_meta = dict(metadata or {})
        source_id = str(base_meta.get("source_id") or f"source-{len(self._buffer_meta):06d}")
        for offset, chunk in enumerate(chunks):
            self._buffer.append(chunk)
            self._buffer_meta.append(
                {
                    **base_meta,
                    "source_id": source_id,
                    "chunk_offset": offset,
                }
            )
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

        self._base_url = base_url.rstrip("/")
        self._http = requests.Session()

        health = _check_letta_health(base_url=self._base_url, http=self._http)
        self._llm_models = self._list_llm_models()
        self._embedding_models = self._list_embedding_models()
        self._model_handle = self._resolve_model_handle()
        self._embedding_handle = self._resolve_embedding_handle()
        self._validate_model_embedding_pair()
        agent_name = self._make_agent_name(self.save_dir)
        self._agent_id = self._create_agent(agent_name)
        self.backend_mode = "real_letta_archival_memory_v2"

        _safe_print(
            "[LettaBenchMemory][DEBUG]",
            f"health_status={health.get('status')}",
            f"server_version={health.get('version', 'unknown')}",
            f"backend_mode={self.backend_mode}",
            f"base_url={self._base_url}",
            f"agent_id={self._agent_id}",
            f"model={self._model_handle}",
            f"embedding={self._embedding_handle}",
            f"llm_models={len(self._llm_models)}",
            f"embedding_models={len(self._embedding_models)}",
        )

        self._audit_writer.write_config(
            {
                "backend_mode": self.backend_mode,
                "base_url": self._base_url,
                "agent_model": self._model_handle,
                "embedding_model": self._embedding_handle,
                "save_dir": self.save_dir,
                "chunk_size": self.chunk_size,
            }
        )

        for idx, chunk in enumerate(self._buffer):
            metadata = self._buffer_meta[idx] if idx < len(self._buffer_meta) else {}
            memory_id = self._insert_archival_memory(chunk=chunk, chunk_idx=idx, metadata=metadata)
            self._created_passage_ids.append(memory_id)
            row = {
                "chunk_id": f"memgpt-{idx:06d}",
                "chunk_idx": idx,
                "text": chunk,
                "source_metadata": metadata,
                "storage_target": {"agent_id": self._agent_id, "memory_id": memory_id},
            }
            self._chunk_audit.append(row)
            self._audit_writer.add_chunk(row)
            if idx < 3 or idx == len(self._buffer) - 1:
                _safe_print(
                    "[LettaBenchMemory][DEBUG]",
                    f"ingest chunk={idx + 1}/{len(self._buffer)}",
                    f"memory_id={memory_id}",
                )

        self._ingest_time_ms = int((time.time() - t0) * 1000)
        self._audit_writer.finalize(
            summary=self.audit_ingest(),
            storage_manifest={
                "backend": "memgpt",
                "run_id": self._audit_writer.run_id,
                "save_dir": self.save_dir,
                "agent_id": self._agent_id,
                "created_passage_count": len(self._created_passage_ids),
            },
        )

    @staticmethod
    def _make_agent_name(save_dir: str) -> str:
        digest = hashlib.sha1(save_dir.encode("utf-8")).hexdigest()[:10]
        return f"bench-{digest}-{uuid.uuid4().hex[:8]}"

    def _resolve_model_handle(self) -> str:
        configured = os.getenv("LETTA_AGENT_MODEL") or os.getenv("LETTA_MODEL")
        if configured:
            return configured
        models = self._llm_models
        if not models:
            raise RuntimeError(
                "[LettaBenchMemory] /v1/models/ 返回空，无法自动选择 agent model。"
                "请设置 LETTA_AGENT_MODEL (或 LETTA_MODEL)。"
            )
        first = models[0]
        handle = first.get("handle") or first.get("name")
        if not handle:
            raise RuntimeError(f"[LettaBenchMemory] /v1/models/ 返回缺少 handle/name: {first}")
        return str(handle)

    def _resolve_embedding_handle(self) -> Optional[str]:
        configured = os.getenv("LETTA_EMBEDDING_MODEL") or os.getenv("LETTA_EMBEDDING")
        if configured:
            return configured
        models = self._embedding_models
        if not models:
            return None
        first = models[0]
        handle = first.get("handle") or first.get("name")
        return str(handle) if handle else None

    def _list_llm_models(self) -> List[Dict[str, Any]]:
        data = self._request("GET", "/v1/models/")
        if not isinstance(data, list):
            raise RuntimeError(f"[LettaBenchMemory] /v1/models/ 返回结构异常: {type(data)}")
        return [m for m in data if isinstance(m, dict)]

    def _list_embedding_models(self) -> List[Dict[str, Any]]:
        data = self._request("GET", "/v1/models/embedding")
        if not isinstance(data, list):
            raise RuntimeError(f"[LettaBenchMemory] /v1/models/embedding 返回结构异常: {type(data)}")
        return [m for m in data if isinstance(m, dict)]

    @staticmethod
    def _model_map_by_handle(models: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        mapped: Dict[str, Dict[str, Any]] = {}
        for m in models:
            handle = m.get("handle") or m.get("name")
            if handle:
                mapped[str(handle)] = m
        return mapped

    def _validate_model_embedding_pair(self) -> None:
        if not self._model_handle:
            raise RuntimeError("[LettaBenchMemory] model handle 为空，无法创建 agent")

        llm_map = self._model_map_by_handle(self._llm_models)
        emb_map = self._model_map_by_handle(self._embedding_models)

        if self._model_handle not in llm_map:
            available = ", ".join(list(llm_map.keys())[:6]) or "<empty>"
            raise RuntimeError(
                "[LettaBenchMemory] 指定的 LETTA_AGENT_MODEL/LETTA_MODEL 不在 /v1/models/ 列表中。"
                f" model={self._model_handle} available={available}"
            )

        if self._embedding_handle:
            if self._embedding_handle not in emb_map:
                available = ", ".join(list(emb_map.keys())[:6]) or "<empty>"
                raise RuntimeError(
                    "[LettaBenchMemory] 指定的 LETTA_EMBEDDING_MODEL/LETTA_EMBEDDING 不在 /v1/models/embedding 列表中。"
                    f" embedding={self._embedding_handle} available={available}"
                )
            if self._embedding_handle == self._model_handle:
                raise RuntimeError(
                    "[LettaBenchMemory] model 与 embedding handle 相同。"
                    "这通常意味着把同一个句柄误同时用于 LLM 与 embedding，"
                    "会在 archival-memory 写入时触发服务端 embedding 错误。"
                    f" model=embedding={self._model_handle}。"
                    "请设置不同的 LETTA_AGENT_MODEL 与 LETTA_EMBEDDING_MODEL。"
                )

    def _create_agent(self, name: str) -> str:
        payload: Dict[str, Any] = {
            "name": name,
            "model": self._model_handle,
            "include_base_tools": True,
            "include_multi_agent_tools": False,
            "agent_type": "memgpt_v2_agent",
            "metadata": {
                "benchmark": "memgpt_bench",
                "save_dir": self.save_dir,
                "created_by": "LettaBenchMemory",
            },
            "tags": ["benchmark", "memgpt", "real-letta"],
        }
        if self._embedding_handle:
            payload["embedding"] = self._embedding_handle
        data = self._request("POST", "/v1/agents/", json=payload)
        agent_id = data.get("id")
        if not agent_id:
            raise RuntimeError(f"[LettaBenchMemory] create_agent 返回缺少 id: {data}")
        return str(agent_id)

    def _insert_archival_memory(self, chunk: str, chunk_idx: int, metadata: Dict[str, Any]) -> str:
        if not self._agent_id:
            raise RuntimeError("[LettaBenchMemory] agent not initialized before ingestion")
        created_at = (
            parse_time_to_iso(metadata.get("session_time"))
            or parse_time_to_iso(metadata.get("time"))
            or parse_time_to_iso(metadata.get("period"))
            or datetime.now(timezone.utc).isoformat()
        )
        prefix = (
            f"[chunk_idx={chunk_idx}]"
            f"[source_id={metadata.get('source_id', f'source-{chunk_idx:06d}')}]"
            f"[case_id={metadata.get('case_id', 'unknown')}]"
            f"[user_id={metadata.get('user_id', 'unknown')}]"
            f"[conv_id={metadata.get('conv_id', 'unknown')}]"
            f"[session_time={metadata.get('session_time') or metadata.get('time') or 'unknown'}]"
            f"[period={metadata.get('period', 'unknown')}]"
            f"[category={metadata.get('category', 'unknown')}]"
        )
        payload = {
            "text": prefix + "\n" + chunk,
            "tags": [
                "bench",
                "real-letta",
                f"chunk-{chunk_idx:06d}",
                f"source-{metadata.get('source_id', f'source-{chunk_idx:06d}')}",
                f"case-{metadata.get('case_id', 'unknown')}",
                f"user-{metadata.get('user_id', 'unknown')}",
                f"conv-{metadata.get('conv_id', 'unknown')}",
            ],
            "created_at": created_at,
        }
        data = self._request_with_retry(
            "POST",
            f"/v1/agents/{self._agent_id}/archival-memory",
            json=payload,
            op=f"insert_chunk_{chunk_idx}",
        )
        if not isinstance(data, list) or not data:
            raise RuntimeError(f"[LettaBenchMemory] archival-memory 写入返回异常: {data}")
        first = data[0]
        memory_id = first.get("id")
        if not memory_id:
            raise RuntimeError(f"[LettaBenchMemory] 写入结果缺少 memory_id: {first}")
        return str(memory_id)

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        if not self._agent_id:
            raise RuntimeError("[LettaBenchMemory] backend is not ready, call build_index() first.")

        t0 = time.time()
        data = self._request(
            "GET",
            f"/v1/agents/{self._agent_id}/archival-memory/search",
            params={"query": query, "top_k": max(1, int(top_k))},
        )
        results = data.get("results", []) if isinstance(data, dict) else []
        if not isinstance(results, list):
            raise RuntimeError(f"[LettaBenchMemory] search 返回结构异常: {data}")

        evidences: List[Evidence] = []
        for rank, item in enumerate(results[:top_k], start=1):
            content = str(item.get("content", ""))
            if not content.strip():
                continue
            evidences.append(
                Evidence(
                    content=content,
                    metadata={
                        "source": "LettaArchivalMemory",
                        "backend_mode": self.backend_mode,
                        "agent_id": self._agent_id,
                        "memory_id": item.get("id"),
                        "timestamp": item.get("timestamp"),
                        "rank": rank,
                        "score": None,
                    },
                )
            )

        self._retrieve_calls += 1
        elapsed_ms = int((time.time() - t0) * 1000)
        self._retrieve_time_ms_total += elapsed_ms
        self._retrieve_last_count = len(evidences)

        _safe_print(
            "[LettaBenchMemory][DEBUG]",
            "retrieve real-letta route=/v1/agents/{agent_id}/archival-memory/search",
            f"query={query}",
            f"top_k={top_k}",
            f"returned={len(evidences)}",
            f"elapsed_ms={elapsed_ms}",
            f"agent_id={self._agent_id}",
        )
        return evidences

    def reset(self) -> None:
        if self._agent_id:
            try:
                self._request("DELETE", f"/v1/agents/{self._agent_id}")
                _safe_print("[LettaBenchMemory][DEBUG]", f"deleted agent_id={self._agent_id}")
            except Exception as e:
                raise RuntimeError(f"[LettaBenchMemory] reset 删除 agent 失败 id={self._agent_id}: {e}") from e

        if self._http is not None:
            self._http.close()

        self._buffer = []
        self._buffer_meta = []
        self._http = None
        self._agent_id = None
        self._created_passage_ids = []
        self._ingest_time_ms = 0
        self._retrieve_calls = 0
        self._retrieve_time_ms_total = 0
        self._retrieve_last_count = 0
        self.backend_mode = "uninitialized"

    def audit_ingest(self) -> Dict[str, Any]:
        source_complete = 0
        for row in self._chunk_audit:
            meta = row.get("source_metadata", {})
            if all(meta.get(k) is not None for k in ["source_id"]):
                source_complete += 1
        return {
            "backend_mode": self.backend_mode,
            "ingest_path": "POST /v1/agents/ + POST /v1/agents/{agent_id}/archival-memory",
            "ingest_chunks": len(self._buffer),
            "ingest_time_ms": self._ingest_time_ms,
            **self._usage,
            "agent_id": self._agent_id,
            "created_passage_count": len(self._created_passage_ids),
            "agent_model": self._model_handle,
            "embedding_model": self._embedding_handle,
            "source_metadata_complete": source_complete,
            "ingest_error_count": len(self._ingest_errors),
            "ingest_audit_run_id": self._audit_writer.run_id,
            "ingest_audit_dir": str(self._audit_writer.root),
        }

    def audit_retrieve(self) -> Dict[str, Any]:
        return {
            "backend_mode": self.backend_mode,
            "retrieve_path": "GET /v1/agents/{agent_id}/archival-memory/search",
            "retrieve_calls": self._retrieve_calls,
            "retrieve_time_ms_total": self._retrieve_time_ms_total,
            "retrieve_last_count": self._retrieve_last_count,
            "agent_id": self._agent_id,
            **self._usage,
        }

    def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: Dict[str, Any] | None = None,
        op: str = "request",
        retries: int = 3,
    ) -> Any:
        for attempt in range(1, retries + 1):
            try:
                return self._request(method, path, json=json, params=params)
            except RuntimeError as exc:
                msg = str(exc)
                retryable = ("HTTP 5" in msg) or ("timeout" in msg.lower()) or ("connection" in msg.lower())
                if attempt >= retries or not retryable:
                    self._ingest_errors.append({"op": op, "attempt": attempt, **compact_error(exc)})
                    raise
                sleep_s = min(2.0 * attempt, 5.0)
                _safe_print(f"[LettaBenchMemory][WARN] {op} failed attempt={attempt}/{retries}, retrying in {sleep_s}s: {msg}")
                time.sleep(sleep_s)
        raise RuntimeError(f"[LettaBenchMemory] unreachable retry path op={op}")

    def _request(self, method: str, path: str, *, json: Any | None = None, params: Dict[str, Any] | None = None) -> Any:
        if self._http is None:
            raise RuntimeError("[LettaBenchMemory] HTTP session not initialized")
        url = urljoin(self._base_url.rstrip("/") + "/", path.lstrip("/"))
        try:
            resp = self._http.request(method=method.upper(), url=url, json=json, params=params, timeout=30)
        except req_exc.Timeout as e:
            raise RuntimeError(f"[LettaBenchMemory][timeout] 请求超时 {method} {url}: {e}") from e
        except req_exc.ProxyError as e:
            raise RuntimeError(f"[LettaBenchMemory][proxy] 代理错误 {method} {url}: {e}") from e
        except req_exc.ConnectionError as e:
            raise RuntimeError(f"[LettaBenchMemory][connection] 连接失败 {method} {url}: {e}") from e
        except requests.RequestException as e:
            raise RuntimeError(f"[LettaBenchMemory][request] 请求失败 {method} {url}: {e}") from e

        if resp.status_code >= 400:
            body = _safe_console_text(resp.text, limit=400)
            category = "4xx_config_or_payload" if 400 <= resp.status_code < 500 else "5xx_server"
            if "embedding" in body.lower() or "model" in body.lower():
                category = "model_or_embedding_handle_error"
            raise RuntimeError(f"[LettaBenchMemory][{category}] HTTP {resp.status_code} {method} {url} body={body}")

        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError as e:
            raise RuntimeError(f"[LettaBenchMemory] 非 JSON 响应 {method} {url}: {_safe_console_text(resp.text, limit=300)}") from e


def _check_letta_health(base_url: str, http: requests.Session) -> Dict[str, Any]:
    health_url = urljoin(base_url.rstrip("/") + "/", "v1/health/")
    try:
        response = http.get(health_url, timeout=8)
    except requests.exceptions.ProxyError as err:
        raise RuntimeError(
            "[LettaBenchMemory] Letta 服务请求被代理拦截（ProxyError）。"
            "访问 localhost:8283 请设置 NO_PROXY=localhost,127.0.0.1。"
        ) from err
    except requests.exceptions.ConnectionError as err:
        raise RuntimeError(
            f"[LettaBenchMemory] Letta 服务不可达: {health_url}."
            "请确认 LETTA_BASE_URL 正确且服务已启动。"
        ) from err
    except requests.RequestException as err:
        raise RuntimeError(f"[LettaBenchMemory] Letta 健康检查请求失败: {health_url} err={err}") from err

    if response.status_code >= 500:
        raise RuntimeError(
            f"[LettaBenchMemory] Letta 服务返回 5xx: status={response.status_code} url={health_url}."
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
