#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAPTOR bench wrapper

递归摘要树：chunk → cluster → LLM summarize → 递归建树 → 检索时沿树遍历。
核心特性：多层摘要（tb_num_layers=3），检索时同时返回叶子和摘要节点。

依赖: umap-learn, faiss-cpu, tiktoken, scipy
RAPTOR repo: memoRaxis/external/raptor_repo
"""
from __future__ import annotations

import sys
import threading
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from scipy.spatial import distance

# RAPTOR repo path
_RAPTOR_REPO = str(Path(__file__).resolve().parent.parent / "memoRaxis" / "external" / "raptor_repo")
if _RAPTOR_REPO not in sys.path:
    sys.path.insert(0, _RAPTOR_REPO)

from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.EmbeddingModels import BaseEmbeddingModel
from raptor.SummarizationModels import BaseSummarizationModel
from raptor.QAModels import BaseQAModel

from simpleMem_src import get_config

import logging
logger = logging.getLogger(__name__)


@dataclass
class Evidence:
    content: str
    metadata: dict


class _NoQAModel(BaseQAModel):
    def answer_question(self, *args, **kwargs):
        return ""


class _CompatEmbeddingModel(BaseEmbeddingModel):
    """OpenAI-compat embedding with retry."""
    def __init__(self):
        conf = get_config().embedding
        self.base_url = conf.get("base_url")
        self.api_key = conf.get("api_key")
        self.model = conf.get("model", "text-embedding-v3")
        from openai import OpenAI
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def create_embedding(self, text: str):
        text = text.replace("\n", " ")
        max_retries = 8
        wait = 30
        for attempt in range(max_retries):
            try:
                resp = self._client.embeddings.create(input=text, model=self.model)
                return resp.data[0].embedding
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "RateLimit" in err_str or "rate_limit" in err_str.lower()) \
                        and attempt < max_retries - 1:
                    logger.warning("RAPTOR embedding 429, wait %ds (%d/%d)", wait, attempt+1, max_retries)
                    _time.sleep(wait)
                    wait = min(wait * 2, 300)
                    continue
                raise


class _CompatSummarizationModel(BaseSummarizationModel):
    """OpenAI-compat summarization with token tracking."""
    def __init__(self):
        conf = get_config().llm
        from openai import OpenAI
        self._client = OpenAI(api_key=conf.get("api_key"), base_url=conf.get("base_url"))
        self.model = conf.get("model")
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._stats_lock = threading.Lock()

    def summarize(self, context, max_tokens=180):
        try:
            max_tokens = int(max_tokens)
        except Exception:
            max_tokens = 180
        if max_tokens < 32:
            max_tokens = 32

        max_retries = 8
        wait = 30
        resp = None
        for attempt in range(max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a careful summarizer."},
                        {"role": "user", "content": f"Summarize the following. Keep key entities, dates, decisions:\n{context}"},
                    ],
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                break
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "RateLimit" in err_str or "rate_limit" in err_str.lower()) \
                        and attempt < max_retries - 1:
                    logger.warning("RAPTOR summarize 429, wait %ds (%d/%d)", wait, attempt+1, max_retries)
                    _time.sleep(wait)
                    wait = min(wait * 2, 300)
                    continue
                raise

        if resp is None:
            return (context or "")[:2000]

        content = ""
        try:
            content = resp.choices[0].message.content or ""
        except Exception:
            pass

        # Fallback for non-standard response formats
        if not content.strip():
            try:
                raw = resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()
                msg = raw.get("choices", [{}])[0].get("message", {})
                c = msg.get("content", "")
                if isinstance(c, str):
                    content = c
                elif isinstance(c, list):
                    content = "".join(
                        p.get("text", "") if isinstance(p, dict) else str(p) for p in c
                    )
            except Exception:
                pass

        if not content.strip():
            content = (context or "")[:2000]

        with self._stats_lock:
            self.llm_calls += 1
            try:
                self.prompt_tokens += resp.usage.prompt_tokens or 0
                self.completion_tokens += resp.usage.completion_tokens or 0
            except Exception:
                pass

        return content


class RaptorBenchMemory:
    """RAPTOR wrapper for benchmark evaluation.

    Pattern: add_memory() accumulates text → build_index() constructs RAPTOR tree
    (cluster + LLM summarize per layer) → retrieve() queries tree.
    """

    def __init__(
        self,
        save_dir: str = "/tmp/raptor_bench",
        tb_num_layers: int = 3,
        tb_max_tokens: int = 200,
        tb_summarization_length: int = 120,
        tr_threshold: float = 0.5,
        tr_top_k: int = 10,
        chunk_size: int = 1000,
    ):
        self.save_dir = save_dir
        self._chunk_size = chunk_size
        self._buffer: List[str] = []

        # Save params for reset
        self._tb_num_layers = tb_num_layers
        self._tb_max_tokens = tb_max_tokens
        self._tb_summarization_length = tb_summarization_length
        self._tr_threshold = tr_threshold
        self._tr_top_k = tr_top_k

        self._emb = _CompatEmbeddingModel()
        self._summ = _CompatSummarizationModel()
        self._ra: Optional[RetrievalAugmentation] = None
        self._ingest_time_ms = 0

    # ── Buffer / chunk ───────────────────────────────────────────────

    def add_memory(self, text: str) -> None:
        for chunk in self._text_to_chunks(text):
            self._buffer.append(chunk)

    def add_text(self, text: str) -> None:
        self.add_memory(text)

    def _text_to_chunks(self, text: str) -> List[str]:
        if len(text) <= self._chunk_size:
            return [text]
        chunks = []
        while text:
            if len(text) <= self._chunk_size:
                chunks.append(text)
                break
            cut = text.rfind("\n", 0, self._chunk_size)
            if cut <= 0:
                cut = text.rfind(" ", 0, self._chunk_size)
            if cut <= 0:
                cut = self._chunk_size
            chunks.append(text[:cut].rstrip())
            text = text[cut:].lstrip()
        return chunks

    # ── Build / Ingest ───────────────────────────────────────────────

    def _make_config(self):
        return RetrievalAugmentationConfig(
            embedding_model=self._emb,
            summarization_model=self._summ,
            qa_model=_NoQAModel(),
            tb_num_layers=self._tb_num_layers,
            tb_max_tokens=self._tb_max_tokens,
            tb_summarization_length=self._tb_summarization_length,
            tr_threshold=self._tr_threshold,
            tr_top_k=self._tr_top_k,
        )

    def build_index(self) -> None:
        """Build RAPTOR tree from buffered chunks."""
        if not self._buffer:
            return

        config = self._make_config()
        ra = RetrievalAugmentation(config=config, tree=None)
        text = "\n\n".join(self._buffer)

        t0 = _time.time()
        ra.add_documents(text)
        self._ingest_time_ms = int((_time.time() - t0) * 1000)
        self._ra = ra

    # ── Retrieve ─────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 10) -> List[Evidence]:
        if self._ra is None or self._ra.tree is None:
            return []

        context, layer_info = self._ra.retrieve(
            question=query,
            top_k=top_k,
            collapse_tree=True,
            return_layer_information=True,
        )

        q_emb = self._ra.retriever.create_embedding(query)
        emb_key = getattr(self._ra.retriever, "context_embedding_model", None) or "EMB"

        evidences: List[Evidence] = []
        for item in layer_info:
            idx = int(item["node_index"])
            layer = int(item["layer_number"])
            node = self._ra.tree.all_nodes[idx]

            # Get node embedding for scoring
            node_emb = None
            try:
                if isinstance(node.embeddings, dict):
                    node_emb = node.embeddings.get(emb_key)
                    if node_emb is None and node.embeddings:
                        node_emb = next(iter(node.embeddings.values()))
                else:
                    node_emb = node.embeddings
            except Exception:
                pass

            if node_emb is not None:
                d = float(distance.cosine(q_emb, node_emb))
                score = float(1.0 - d / 2.0)
            else:
                d = None
                score = 0.0

            evidences.append(Evidence(
                content=node.text,
                metadata={"source": "RAPTOR", "node_index": idx, "layer": layer, "score": score},
            ))

        # Fallback if layer_info was empty
        if not evidences and isinstance(context, str) and context.strip():
            evidences.append(Evidence(
                content=context,
                metadata={"source": "RAPTOR", "node_index": -1, "layer": -1, "score": 0.0},
            ))

        return evidences

    # ── Reset ────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._ra = None
        self._buffer.clear()
        self._ingest_time_ms = 0
        # Recreate summarization model to reset stats
        self._summ = _CompatSummarizationModel()

    # ── Audit ────────────────────────────────────────────────────────

    def audit_ingest(self) -> dict:
        stats = {
            "llm_calls": self._summ.llm_calls,
            "prompt_tokens": self._summ.prompt_tokens,
            "completion_tokens": self._summ.completion_tokens,
        }
        n_nodes = 0
        n_layers = 0
        if self._ra and self._ra.tree:
            n_nodes = len(self._ra.tree.all_nodes)
            n_layers = self._ra.tree.num_layers
        return {
            "ingest_chunks": len(self._buffer),
            "ingest_time_ms": self._ingest_time_ms,
            "ingest_llm_calls": stats["llm_calls"],
            "ingest_llm_prompt_tokens": stats["prompt_tokens"],
            "ingest_llm_completion_tokens": stats["completion_tokens"],
            "tree_nodes": n_nodes,
            "tree_layers": n_layers,
        }
