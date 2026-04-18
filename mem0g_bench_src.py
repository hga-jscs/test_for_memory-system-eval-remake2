#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mem0g (graph-enhanced mem0) memory backend for AgeMem benchmarks.

mem0 + Neo4j graph: entity extraction → relationship triplets → conflict resolution.
核心特性: include_graph=True, 检索返回 vector hits + graph relations。

依赖: Neo4j server (Docker), Qdrant server (Docker), neo4j Python driver
论文: arXiv:2504.19413, Section 2.2

论文配置:
  - m=10 previous messages for context
  - s=10 similar memories for comparison
  - GPT-4o-mini (我们用 DashScope qwen-plus)
"""

import os
import sys
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["MEM0_TELEMETRY"] = "false"

sys.path.insert(0, str(Path(__file__).parent))
from simpleMem_src import get_config, Evidence


def build_mem0g_config(collection_name: str) -> Dict[str, Any]:
    """Build mem0 config with graph store (Neo4j) enabled."""
    conf = get_config()

    neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASSWORD", "password")

    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "embedding_model_dims": conf.embedding.get("dim", 1024),
                "host": "localhost",
                "port": 6333,
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
        "graph_store": {
            "provider": "neo4j",
            "config": {
                "url": neo4j_url,
                "username": neo4j_user,
                "password": neo4j_pass,
            },
        },
        "version": "v1.1",
    }


class Mem0GMemory:
    """mem0 with graph store — returns vector hits + graph relations."""

    def __init__(self, collection_name: str):
        from mem0 import Memory
        self._collection_name = collection_name
        config = build_mem0g_config(collection_name)
        self.memory = Memory.from_config(config)
        self.user_id = "bench_user"
        self.chunks_added = 0
        self._ingest_time_ms = 0

    @property
    def mem_size(self) -> int:
        try:
            results = self.memory.get_all(user_id=self.user_id, limit=10000)
            items = results.get("results", []) if isinstance(results, dict) else results
            return len(items)
        except Exception:
            return self.chunks_added

    def add_memory(self, text: str, metadata: Optional[Dict] = None) -> None:
        try:
            self.memory.add(
                text,
                user_id=self.user_id,
                metadata=metadata or {},
                infer=True,
            )
        except Exception as e:
            # Neo4j CypherSyntaxError etc. — LLM 可能生成空 relation type
            # 记录但不中断整个 benchmark
            import logging
            logging.warning(f"mem0g add_memory failed (will continue): {type(e).__name__}: {str(e)[:100]}")
        self.chunks_added += 1

    def retrieve(self, query: str, top_k: int = 10) -> List[Evidence]:
        """Retrieve from both vector store and graph store."""
        try:
            results = self.memory.search(query, user_id=self.user_id, limit=top_k)
        except Exception as e:
            import logging
            logging.warning(f"mem0g retrieve failed: {type(e).__name__}: {str(e)[:100]}")
            return []

        evidences: List[Evidence] = []

        if not isinstance(results, dict):
            return evidences

        # Vector hits
        for hit in results.get("results", []):
            content = hit.get("memory", "")
            meta = {
                "source": "mem0g",
                "type": "vector",
                "score": hit.get("score", 0.0),
                "id": hit.get("id", ""),
            }
            if hit.get("metadata"):
                meta.update(hit["metadata"])
            evidences.append(Evidence(content=content, metadata=meta))

        # Graph relations
        for rel in results.get("relations", []):
            source = rel.get("source", "")
            relation = rel.get("relation") or rel.get("relationship", "")
            target = rel.get("target") or rel.get("destination", "")
            relation_text = f"{source} --[{relation}]--> {target}"
            meta = {"source": "mem0g", "type": "graph_relation"}
            meta.update(rel)
            evidences.append(Evidence(content=relation_text, metadata=meta))

        return evidences

    def reset(self) -> None:
        try:
            self.memory.reset()
        except Exception:
            pass
        self.chunks_added = 0
