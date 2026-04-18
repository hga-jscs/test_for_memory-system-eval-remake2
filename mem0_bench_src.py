# -*- coding: utf-8 -*-
"""mem0 memory backend for AgeMem benchmarks.

Wraps mem0ai Memory with in-memory Qdrant (path=':memory:'), no Docker required.
Compatible interface with simpleMem_src.SimpleRAGMemory.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 必须在任何 mem0 import 之前设置：
# mem0 默认开启 telemetry，会在 ~/.mem0/migrations_qdrant 创建共享 Qdrant 实例，
# 多线程并发初始化时产生文件锁冲突。禁用 telemetry 彻底消除冲突。
os.environ["MEM0_TELEMETRY"] = "false"

sys.path.insert(0, str(Path(__file__).parent))

from simpleMem_src import get_config, Evidence


def build_mem0_config(collection_name: str) -> Dict[str, Any]:
    """Build mem0 config from config.yaml, using in-memory Qdrant."""
    conf = get_config()
    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "embedding_model_dims": conf.embedding.get("dim", 1024),
                "path": ":memory:",       # pure in-memory, no Docker
                "on_disk": False,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": conf.embedding.get("model"),
                "api_key": conf.embedding.get("api_key"),
                "openai_base_url": conf.embedding.get("base_url"),
                "embedding_dims": conf.embedding.get("dim", 1024),
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": conf.llm.get("model"),
                "api_key": conf.llm.get("api_key"),
                "openai_base_url": conf.llm.get("base_url"),
                "max_tokens": 2000,
                "temperature": 0.1,
            },
        },
        # Disable version history to save tokens/time
        "version": "v1.1",
    }


class Mem0RAGMemory:
    """Wrapper around mem0 Memory with same interface as SimpleRAGMemory."""

    def __init__(self, collection_name: str):
        from mem0 import Memory
        self._collection_name = collection_name
        config = build_mem0_config(collection_name)
        self.memory = Memory.from_config(config)
        self.user_id = "ingest_user"
        self.chunks_added = 0      # raw chunks sent to mem0

    @property
    def mem_size(self) -> int:
        """Number of atomic memories stored after LLM extraction."""
        try:
            results = self.memory.get_all(user_id=self.user_id, limit=10000)
            items = results.get("results", []) if isinstance(results, dict) else results
            return len(items)
        except Exception:
            return self.chunks_added

    def add_memory(self, text: str, metadata: Optional[Dict] = None) -> None:
        self.memory.add(
            text,
            user_id=self.user_id,
            metadata=metadata or {},
            infer=True,          # LLM extracts atomic facts
        )
        self.chunks_added += 1

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        results = self.memory.search(query, user_id=self.user_id, limit=top_k)
        hits = results.get("results", []) if isinstance(results, dict) else results
        evidences = []
        for h in hits:
            content = h.get("memory", "")
            meta = {"score": h.get("score", 0.0), "id": h.get("id", ""), "source": "mem0"}
            if h.get("metadata"):
                meta.update(h["metadata"])
            evidences.append(Evidence(content=content, metadata=meta))
        return evidences

    def reset(self) -> None:
        try:
            self.memory.reset()
        except Exception:
            pass
        self.chunks_added = 0
