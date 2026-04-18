# -*- coding: utf-8 -*-
"""A-MEM (Agentic Memory) 适配器

将 agiresearch/A-mem 的 AgenticMemorySystem 包装为 BaseMemorySystem 接口，
支持 ChromaDB 持久化、Ark LLM 端点和 response_format 降级。
"""
from __future__ import annotations

import json
import logging
import pickle
import threading
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.memory_interface import BaseMemorySystem, Evidence
from src.config import get_config

# A-MEM imports (via external/amem_repo)
from external.amem_repo.agentic_memory.memory_system import AgenticMemorySystem, MemoryNote
from external.amem_repo.agentic_memory.llm_controller import LLMController
from external.amem_repo.agentic_memory.retrievers import ChromaRetriever

logger = logging.getLogger(__name__)


class AMemMemory(BaseMemorySystem):
    """A-MEM 记忆系统适配器

    核心特性:
    - ChromaDB PersistentClient 持久化 (每个 instance 一个目录)
    - 本地 SentenceTransformer embedding (all-MiniLM-L6-v2, 384-dim)
    - 记忆演化 (evolution): 每条新记忆触发 LLM 分析 + 邻居更新
    - Ark LLM 端点兼容 (base_url + response_format 降级)
    """

    def __init__(
        self,
        chroma_dir: str,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        enable_evolution: bool = True,
        evo_threshold: int = 100,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        conf = get_config()
        self._model = llm_model or conf.llm.get("model")
        self._base_url = llm_base_url or conf.llm.get("base_url")
        self._api_key = llm_api_key or conf.llm.get("api_key")
        self._chroma_dir = str(chroma_dir)
        self._model_name = model_name
        self._enable_evolution = enable_evolution

        # LLM stats
        self._llm_calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._stats_lock = threading.Lock()

        # --- Init AgenticMemorySystem (ephemeral ChromaDB, will be replaced) ---
        self._amem = AgenticMemorySystem(
            model_name=model_name,
            llm_backend="openai",
            llm_model=self._model,
            api_key=self._api_key,
            evo_threshold=evo_threshold,
        )

        # Replace LLM controller with one that has base_url
        self._amem.llm_controller = LLMController(
            backend="openai",
            model=self._model,
            api_key=self._api_key,
            base_url=self._base_url,
        )

        # Wrap get_completion for response_format fallback + retry + stats
        self._patch_llm()

        # Replace ephemeral retriever with persistent ChromaDB
        self._setup_persistent_retriever()

        # Load existing memories dict from pickle (if resuming)
        self._load_memories()

        # Patch consolidate_memories to use persistent retriever
        self._patch_consolidate()

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _setup_persistent_retriever(self):
        """Replace ephemeral ChromaDB with PersistentClient."""
        Path(self._chroma_dir).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=self._chroma_dir)
        ef = SentenceTransformerEmbeddingFunction(model_name=self._model_name)
        collection = client.get_or_create_collection(
            name="memories", embedding_function=ef
        )

        # Build a ChromaRetriever-compatible object with persistent backend
        retriever = ChromaRetriever.__new__(ChromaRetriever)
        retriever.client = client
        retriever.embedding_function = ef
        retriever.collection = collection
        self._amem.retriever = retriever

    def _patch_llm(self):
        """Wrap get_completion for: retry on 429, json_schema→json_object fallback, stats."""
        original_fn = self._amem.llm_controller.llm.get_completion
        adapter = self

        def wrapped_get_completion(prompt, response_format=None, temperature=0.7):
            max_retries = 8
            wait = 30
            rf = response_format
            for attempt in range(max_retries):
                try:
                    result = original_fn(prompt, response_format=rf, temperature=temperature)
                    with adapter._stats_lock:
                        adapter._llm_calls += 1
                    return result
                except Exception as e:
                    err_str = str(e)
                    # Rate limit → exponential backoff
                    if "429" in err_str or "RateLimit" in err_str or "rate_limit" in err_str.lower():
                        if attempt < max_retries - 1:
                            logger.warning(
                                "A-MEM LLM 429, waiting %ds (attempt %d/%d)",
                                wait, attempt + 1, max_retries,
                            )
                            _time.sleep(wait)
                            wait = min(wait * 2, 300)
                            continue
                    # json_schema not supported → degrade to json_object
                    if rf and rf.get("type") == "json_schema":
                        logger.warning(
                            "json_schema not supported, falling back to json_object: %s",
                            err_str[:200],
                        )
                        rf = {"type": "json_object"}
                        continue
                    raise
            raise RuntimeError("A-MEM LLM call failed after max retries")

        self._amem.llm_controller.llm.get_completion = wrapped_get_completion

    def _patch_consolidate(self):
        """Override consolidate_memories to use persistent retriever."""
        amem = self._amem
        adapter = self

        def patched_consolidate():
            # Recreate persistent collection (clear + re-add)
            client = chromadb.PersistentClient(path=adapter._chroma_dir)
            try:
                client.delete_collection("memories")
            except Exception:
                pass
            ef = SentenceTransformerEmbeddingFunction(
                model_name=adapter._model_name
            )
            collection = client.get_or_create_collection(
                name="memories", embedding_function=ef
            )
            retriever = ChromaRetriever.__new__(ChromaRetriever)
            retriever.client = client
            retriever.embedding_function = ef
            retriever.collection = collection
            amem.retriever = retriever

            for memory in amem.memories.values():
                metadata = _note_to_metadata(memory)
                amem.retriever.add_document(memory.content, metadata, memory.id)

        amem.consolidate_memories = patched_consolidate

    @property
    def _memories_pkl_path(self) -> Path:
        return Path(self._chroma_dir) / "memories.pkl"

    def _load_memories(self):
        """Load pickled memories dict (for resume)."""
        if self._memories_pkl_path.exists():
            with open(self._memories_pkl_path, "rb") as f:
                self._amem.memories = pickle.load(f)
            logger.info(
                "Loaded %d memories from %s",
                len(self._amem.memories), self._memories_pkl_path,
            )

    # ------------------------------------------------------------------
    # Public API (BaseMemorySystem)
    # ------------------------------------------------------------------

    def add_memory(self, data: str, metadata: Dict[str, Any]) -> None:
        time_val = metadata.get("time")
        if self._enable_evolution:
            # 1) LLM call: analyze content → keywords / context / tags
            analysis = self._amem.analyze_content(data)
            if not isinstance(analysis, dict):
                analysis = {"keywords": [], "context": "General", "tags": []}
            # 2) add_note → process_memory (evolution LLM call if neighbors exist)
            self._amem.add_note(
                content=data,
                time=time_val,
                keywords=analysis.get("keywords", []),
                context=analysis.get("context", "General"),
                tags=analysis.get("tags", []),
            )
        else:
            # No LLM calls — raw storage only
            note = MemoryNote(content=data, timestamp=time_val)
            self._amem.memories[note.id] = note
            md = _note_to_metadata(note)
            self._amem.retriever.add_document(note.content, md, note.id)

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        results = self._amem.search_agentic(query, k=top_k)
        evidences = []
        for r in results:
            evidences.append(
                Evidence(
                    content=r.get("content", ""),
                    metadata={
                        "source": "A-MEM",
                        "id": r.get("id", ""),
                        "context": r.get("context", ""),
                        "keywords": r.get("keywords", []),
                        "tags": r.get("tags", []),
                        "score": r.get("score", 0.0),
                        "is_neighbor": r.get("is_neighbor", False),
                    },
                )
            )
        return evidences

    def reset(self) -> None:
        self._amem.memories.clear()
        self._setup_persistent_retriever()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self):
        """Persist memories dict to pickle (ChromaDB auto-persists)."""
        with open(self._memories_pkl_path, "wb") as f:
            pickle.dump(self._amem.memories, f)
        logger.info(
            "Saved %d memories to %s",
            len(self._amem.memories), self._memories_pkl_path,
        )

    def get_llm_stats(self) -> dict:
        return {
            "llm_calls": self._llm_calls,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._prompt_tokens + self._completion_tokens,
        }

    @property
    def memory_count(self) -> int:
        return len(self._amem.memories)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _note_to_metadata(note: MemoryNote) -> dict:
    return {
        "id": note.id,
        "content": note.content,
        "keywords": note.keywords,
        "links": note.links,
        "retrieval_count": note.retrieval_count,
        "timestamp": note.timestamp,
        "last_accessed": note.last_accessed,
        "context": note.context,
        "evolution_history": note.evolution_history,
        "category": note.category,
        "tags": note.tags,
    }
