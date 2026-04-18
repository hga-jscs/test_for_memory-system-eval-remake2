# -*- coding: utf-8 -*-
"""基于内存向量存储的 Naive RAG 记忆系统（无外部数据库依赖）"""

import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from .config import get_config
from .llm_interface import get_embedding
from .logger import get_logger
from .memory_interface import BaseMemorySystem, Evidence


class SimpleRAGMemory(BaseMemorySystem):
    """
    纯内存的简单 RAG 记忆系统。
    使用 numpy 做余弦相似度检索，零外部依赖。
    """

    def __init__(self, collection_name: str = "default"):
        self._logger = get_logger()
        self._config = get_config()
        self._emb_config = self._config.embedding
        self._dim = self._emb_config.get("dim", 1536)
        self._collection_name = collection_name

        # 内存存储
        self._entries: List[Dict[str, Any]] = []  # {id, content, metadata}
        self._vectors: List[np.ndarray] = []

        self._logger.info(
            "SimpleRAGMemory 初始化: collection=%s, dim=%d",
            collection_name, self._dim,
        )

    @property
    def size(self) -> int:
        return len(self._entries)

    def add_memory(self, data: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        if metadata is None:
            metadata = {}

        vec = get_embedding(data, self._emb_config)
        entry_id = str(uuid.uuid4())[:8]

        self._entries.append({
            "id": entry_id,
            "content": data,
            "metadata": metadata,
        })
        self._vectors.append(np.array(vec, dtype=np.float32))

        self._logger.debug("添加记忆 [%s]: %s...", entry_id, data[:50])
        return entry_id

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        if not self._entries:
            return []

        query_vec = np.array(
            get_embedding(query, self._emb_config), dtype=np.float32
        )

        # 余弦相似度
        mat = np.stack(self._vectors)
        norms = np.linalg.norm(mat, axis=1, keepdims=False)
        query_norm = np.linalg.norm(query_vec)

        # 避免除零
        denom = norms * query_norm
        denom = np.where(denom > 1e-9, denom, 1.0)

        scores = (mat @ query_vec) / denom
        scores = np.nan_to_num(scores, nan=0.0)

        # 取 top_k
        k = min(top_k, len(self._entries))
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            entry = self._entries[idx]
            meta = dict(entry["metadata"])
            meta["score"] = float(scores[idx])
            meta["memory_id"] = entry["id"]
            results.append(Evidence(content=entry["content"], metadata=meta))

        self._logger.debug(
            "检索完成: query='%s...', 结果数=%d", query[:40], len(results)
        )
        return results

    def reset(self) -> None:
        self._entries.clear()
        self._vectors.clear()
        self._logger.info("SimpleRAGMemory 已重置 (collection=%s)", self._collection_name)

    def get_all_memories(self) -> List[Dict[str, Any]]:
        return [
            {"id": e["id"], "content": e["content"], "metadata": e["metadata"]}
            for e in self._entries
        ]
