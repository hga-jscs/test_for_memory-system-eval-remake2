#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal Letta archival-memory diagnostic.

验证闭环：
1) create agent
2) insert archival memory
3) search archival memory

输出可直接用于定位 model/embedding 句柄与 provider 配置问题。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from typing import Any
from urllib.parse import urljoin

import requests


def safe_text(v: Any, limit: int = 400) -> str:
    s = str(v)
    if len(s) > limit:
        s = s[:limit] + "..."
    return s.encode("ascii", "backslashreplace").decode("ascii")


def p(*parts: Any) -> None:
    print(" ".join(safe_text(x) for x in parts), flush=True)


def request(session: requests.Session, method: str, base_url: str, path: str, **kwargs: Any) -> Any:
    url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    t0 = time.time()
    resp = session.request(method=method.upper(), url=url, timeout=30, **kwargs)
    dt = int((time.time() - t0) * 1000)
    p(f"[HTTP] {method.upper()} {path} status={resp.status_code} latency_ms={dt}")
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code} {method.upper()} {path} body={safe_text(resp.text)}")
    if not resp.content:
        return {}
    try:
        return resp.json()
    except ValueError as e:
        raise RuntimeError(f"non-json response for {method.upper()} {path}: {safe_text(resp.text)}") from e


def choose_handle(items: list[dict[str, Any]], explicit: str | None, kind: str) -> str:
    mapped = {str(i.get("handle") or i.get("name")): i for i in items if (i.get("handle") or i.get("name"))}
    if explicit:
        if explicit not in mapped:
            raise RuntimeError(f"{kind} handle not found: {explicit}; available={list(mapped)[:8]}")
        return explicit
    if not mapped:
        raise RuntimeError(f"no {kind} models available")
    return next(iter(mapped.keys()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.getenv("LETTA_BASE_URL", "http://localhost:8283"))
    ap.add_argument("--model", default=os.getenv("LETTA_AGENT_MODEL") or os.getenv("LETTA_MODEL"))
    ap.add_argument("--embedding", default=os.getenv("LETTA_EMBEDDING_MODEL") or os.getenv("LETTA_EMBEDDING"))
    args = ap.parse_args()

    s = requests.Session()
    agent_id = None

    try:
        health = request(s, "GET", args.base_url, "/v1/health/")
        p("[HEALTH]", health)

        llms = request(s, "GET", args.base_url, "/v1/models/")
        embs = request(s, "GET", args.base_url, "/v1/models/embedding")
        if not isinstance(llms, list) or not isinstance(embs, list):
            raise RuntimeError("/v1/models endpoints returned non-list")

        model = choose_handle([x for x in llms if isinstance(x, dict)], args.model, "llm")
        embedding = choose_handle([x for x in embs if isinstance(x, dict)], args.embedding, "embedding")
        p(f"[SELECTED] model={model} embedding={embedding}")

        if model == embedding:
            raise RuntimeError(
                "selected model and embedding handles are identical; this usually breaks archival-memory insertion"
            )

        payload = {
            "name": f"diag-{uuid.uuid4().hex[:8]}",
            "model": model,
            "embedding": embedding,
            "include_base_tools": True,
            "include_multi_agent_tools": False,
            "agent_type": "memgpt_v2_agent",
            "tags": ["diag", "archival"],
            "metadata": {"debug_script": "debug_letta_archival_insert.py"},
        }
        p("[CREATE_AGENT][payload]", json.dumps(payload, ensure_ascii=False))
        created = request(s, "POST", args.base_url, "/v1/agents/", json=payload)
        agent_id = str(created.get("id"))
        if not agent_id:
            raise RuntimeError(f"create agent missing id: {created}")
        p(f"[CREATE_AGENT][ok] agent_id={agent_id}")

        insert_payload = {"text": "diagnostic memory: sky is blue", "tags": ["diag", "insert"]}
        p("[INSERT][payload]", json.dumps(insert_payload, ensure_ascii=False))
        inserted = request(s, "POST", args.base_url, f"/v1/agents/{agent_id}/archival-memory", json=insert_payload)
        p("[INSERT][ok]", inserted)

        searched = request(
            s,
            "GET",
            args.base_url,
            f"/v1/agents/{agent_id}/archival-memory/search",
            params={"query": "sky color", "top_k": 3},
        )
        p("[SEARCH][ok]", searched)

        p("[RESULT] SUCCESS: create+insert+search loop works")
        return 0
    except Exception as e:  # noqa: BLE001
        p("[RESULT] FAILED", e)
        msg = str(e).lower()
        if "selected model and embedding handles are identical" in msg:
            p("[ROOT_CAUSE_HINT] model/embedding handle confusion")
        elif "embedding" in msg and "not found" in msg:
            p("[ROOT_CAUSE_HINT] embedding handle invalid")
        elif "model" in msg and "not found" in msg:
            p("[ROOT_CAUSE_HINT] llm model handle invalid")
        elif "http 500" in msg and "archival-memory" in msg:
            p("[ROOT_CAUSE_HINT] server-side embedding provider/config likely broken")
        return 2
    finally:
        if agent_id:
            try:
                request(s, "DELETE", args.base_url, f"/v1/agents/{agent_id}")
                p(f"[CLEANUP] deleted agent_id={agent_id}")
            except Exception as e:  # noqa: BLE001
                p(f"[CLEANUP][WARN] failed to delete agent_id={agent_id}: {e}")
        s.close()


if __name__ == "__main__":
    raise SystemExit(main())
