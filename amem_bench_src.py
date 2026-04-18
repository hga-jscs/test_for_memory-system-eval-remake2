#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A-MEM (Agentic Memory) bench wrapper

将 agiresearch/A-mem 包装为 bench 脚本可用的接口。
核心特性: evolution（每条记忆触发 LLM 分析 + 邻居更新）。

依赖: chromadb, nltk, sentence-transformers (via chromadb)
A-MEM repo: memoRaxis/external/amem_repo
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import sys
import threading
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# A-MEM repo path
_AMEM_REPO = str(Path(__file__).resolve().parent.parent / "memoRaxis" / "external" / "amem_repo")
if _AMEM_REPO not in sys.path:
    sys.path.insert(0, _AMEM_REPO)

from agentic_memory.memory_system import AgenticMemorySystem, MemoryNote
from agentic_memory.llm_controller import LLMController
from agentic_memory.retrievers import ChromaRetriever

from simpleMem_src import get_config

logger = logging.getLogger(__name__)


@dataclass
class Evidence:
    content: str
    metadata: dict


class AMemBenchMemory:
    """A-MEM wrapper for benchmark evaluation.

    Pattern: add_memory() accumulates text → build_index() triggers A-MEM
    ingestion (analyze + evolution per chunk) → retrieve() searches.
    """

    def __init__(
        self,
        save_dir: str,
        enable_evolution: bool = True,
        evo_threshold: int = 100,
        chunk_size: int = 1000,
    ):
        self.save_dir = save_dir
        self._enable_evolution = enable_evolution
        self._evo_threshold = evo_threshold
        self._chunk_size = chunk_size
        self._buffer: List[str] = []
        self._amem: Optional[AgenticMemorySystem] = None

        # audit counters
        self._ingest_time_ms = 0
        self._llm_calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

    # ── Buffer / chunk ───────────────────────────────────────────────

    def add_memory(self, text: str) -> None:
        """Pre-chunk text and add to buffer."""
        for chunk in self._text_to_chunks(text):
            self._buffer.append(chunk)

    def add_text(self, text: str) -> None:
        """Alias for add_memory."""
        self.add_memory(text)

    def _text_to_chunks(self, text: str) -> List[str]:
        if len(text) <= self._chunk_size:
            return [text]
        chunks = []
        while text:
            if len(text) <= self._chunk_size:
                chunks.append(text)
                break
            cut = text.rfind('\n', 0, self._chunk_size)
            if cut <= 0:
                cut = text.rfind(' ', 0, self._chunk_size)
            if cut <= 0:
                cut = self._chunk_size
            chunks.append(text[:cut].rstrip())
            text = text[cut:].lstrip()
        return chunks

    # ── Build / Ingest ───────────────────────────────────────────────

    def build_index(self) -> None:
        """Ingest all buffered chunks into A-MEM."""
        if not self._buffer:
            return

        config = get_config()
        chroma_dir = str(Path(self.save_dir) / "chroma")
        Path(chroma_dir).mkdir(parents=True, exist_ok=True)

        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        # Build AgenticMemorySystem manually to avoid ephemeral ChromaDB in __init__
        amem = object.__new__(AgenticMemorySystem)
        amem.memories = {}
        amem.model_name = "all-MiniLM-L6-v2"
        amem.evo_cnt = 0
        amem.evo_threshold = self._evo_threshold

        # LLM controller with base_url
        amem.llm_controller = LLMController(
            backend="openai",
            model=config.llm["model"],
            api_key=config.llm["api_key"],
            base_url=config.llm["base_url"],
        )

        # Evolution system prompt (copied from upstream source)
        amem._evolution_system_prompt = '''
You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
Make decisions about its evolution.

The new memory context:
{context}
content: {content}
keywords: {keywords}

The nearest neighbors memories:
{nearest_neighbors_memories}

Based on this information, determine:
1. Should this memory be evolved? Consider its relationships with other memories.
2. What specific actions should be taken (strengthen, update_neighbor)?
   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
The number of neighbors is {neighbor_number}.
Return your decision in JSON format with the following structure:
{{
    "should_evolve": True or False,
    "actions": ["strengthen", "update_neighbor"],
    "suggested_connections": ["memory id from the neighbors above"],
    "tags_to_update": ["tag_1",..."tag_n"],
    "new_context_neighborhood": ["new context",...,"new context"],
    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
}}
'''

        # EphemeralClient + delete_collection: 无 SQLite 锁，显式清理保证隔离
        client = chromadb.EphemeralClient()
        try:
            client.delete_collection("memories")
        except Exception:
            pass
        ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        collection = client.get_or_create_collection(name="memories", embedding_function=ef)
        retriever = ChromaRetriever.__new__(ChromaRetriever)
        retriever.client = client
        retriever.embedding_function = ef
        retriever.collection = collection
        amem.retriever = retriever

        # Patch LLM for: token tracking + response_format fallback + retry
        self._patch_llm(amem)

        # Ingest chunks
        t0 = _time.time()
        for chunk in self._buffer:
            if self._enable_evolution:
                analysis = amem.analyze_content(chunk)
                if not isinstance(analysis, dict):
                    analysis = {"keywords": [], "context": "General", "tags": []}
                amem.add_note(
                    content=chunk,
                    keywords=analysis.get("keywords", []),
                    context=analysis.get("context", "General"),
                    tags=analysis.get("tags", []),
                )
            else:
                note = MemoryNote(content=chunk)
                amem.memories[note.id] = note
                metadata = _note_to_metadata(note)
                amem.retriever.add_document(note.content, metadata, note.id)

        self._ingest_time_ms = int((_time.time() - t0) * 1000)
        self._amem = amem

    def _patch_llm(self, amem: AgenticMemorySystem) -> None:
        """Wrap get_completion for token tracking, retry, response_format fallback."""
        original_fn = amem.llm_controller.llm.get_completion
        wrapper = self

        def wrapped(prompt, response_format=None, temperature=0.7):
            max_retries = 8
            wait = 30
            rf = response_format
            for attempt in range(max_retries):
                try:
                    client = amem.llm_controller.llm.client
                    kwargs = dict(
                        model=amem.llm_controller.llm.model,
                        messages=[
                            {"role": "system", "content": "You must respond with a JSON object."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=temperature,
                        max_tokens=3000,  # A-MEM evolution JSON needs more room
                    )
                    if rf:
                        kwargs["response_format"] = rf
                    resp = client.chat.completions.create(**kwargs)

                    if resp.usage:
                        wrapper._llm_calls += 1
                        wrapper._prompt_tokens += resp.usage.prompt_tokens or 0
                        wrapper._completion_tokens += resp.usage.completion_tokens or 0

                    return resp.choices[0].message.content

                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RateLimit" in err_str or "rate_limit" in err_str.lower():
                        if attempt < max_retries - 1:
                            logger.warning("A-MEM LLM 429, wait %ds (attempt %d/%d)", wait, attempt+1, max_retries)
                            _time.sleep(wait)
                            wait = min(wait * 2, 300)
                            continue
                    if rf and rf.get("type") == "json_schema":
                        logger.warning("json_schema not supported, fallback to json_object")
                        rf = {"type": "json_object"}
                        continue
                    raise
            raise RuntimeError("A-MEM LLM call failed after max retries")

        amem.llm_controller.llm.get_completion = wrapped

    # ── Retrieve ─────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        if not self._amem or not self._amem.memories:
            return []
        results = self._amem.search_agentic(query, k=top_k)
        return [
            Evidence(
                content=r.get("content", ""),
                metadata={
                    "id": r.get("id", ""),
                    "context": r.get("context", ""),
                    "keywords": r.get("keywords", []),
                    "tags": r.get("tags", []),
                    "is_neighbor": r.get("is_neighbor", False),
                },
            )
            for r in results
        ]

    # ── Reset ────────────────────────────────────────────────────────

    def reset(self) -> None:
        if self._amem is not None:
            try:
                self._amem.retriever.client.delete_collection("memories")
            except Exception:
                pass
        self._amem = None
        self._buffer.clear()
        self._ingest_time_ms = 0
        self._llm_calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

    # ── Audit ────────────────────────────────────────────────────────

    def audit_ingest(self) -> dict:
        return {
            "ingest_chunks": len(self._buffer),
            "ingest_time_ms": self._ingest_time_ms,
            "ingest_llm_calls": self._llm_calls,
            "ingest_llm_prompt_tokens": self._prompt_tokens,
            "ingest_llm_completion_tokens": self._completion_tokens,
            "ingest_mem_count": len(self._amem.memories) if self._amem else 0,
        }


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
