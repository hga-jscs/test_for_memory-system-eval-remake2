#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAPTOR bench wrapper with robust path resolution (strict backend mode)."""
from __future__ import annotations

import sys
import threading
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from scipy.spatial import distance

# RAPTOR repo path（优先 memoRaxis/external，其次 third_party）
_REPO_ROOT = Path(__file__).resolve().parent
_RAPTOR_REPO_CANDIDATES = [
    _REPO_ROOT / "memoRaxis" / "external" / "raptor_repo",
    _REPO_ROOT / "third_party" / "raptor",
]
for _raptor_repo in _RAPTOR_REPO_CANDIDATES:
    if _raptor_repo.exists() and str(_raptor_repo) not in sys.path:
        sys.path.insert(0, str(_raptor_repo))
        break

from simpleMem_src import create_openai_client_compatible, get_config

import logging
logger = logging.getLogger(__name__)


@dataclass
class Evidence:
    content: str
    metadata: dict


class RaptorBenchMemory:
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
        self._tb_num_layers = tb_num_layers
        self._tb_max_tokens = tb_max_tokens
        self._tb_summarization_length = tb_summarization_length
        self._tr_threshold = tr_threshold
        self._tr_top_k = tr_top_k

        self._ra = None
        self._emb = None
        self._summ = None
        self._ingest_time_ms = 0
        self.backend_mode = "raptor"

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

    def _make_runtime(self):
        from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
        from raptor.EmbeddingModels import BaseEmbeddingModel
        from raptor.SummarizationModels import BaseSummarizationModel
        from raptor.QAModels import BaseQAModel

        class _NoQAModel(BaseQAModel):
            def answer_question(self, *args, **kwargs):
                return ""

        class _CompatEmbeddingModel(BaseEmbeddingModel):
            def __init__(self):
                conf = get_config().embedding
                self.base_url = conf.get("base_url")
                self.api_key = conf.get("api_key")
                self.model = conf.get("model", "text-embedding-v3")
                self._client = create_openai_client_compatible(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    logger=logger,
                )

            def create_embedding(self, text: str):
                text = text.replace("\n", " ")
                resp = self._client.embeddings.create(input=text, model=self.model)
                return resp.data[0].embedding

        class _CompatSummarizationModel(BaseSummarizationModel):
            def __init__(self):
                conf = get_config().llm
                self._client = create_openai_client_compatible(
                    api_key=conf.get("api_key"),
                    base_url=conf.get("base_url"),
                    logger=logger,
                )
                self.model = conf.get("model")
                self.llm_calls = 0
                self.prompt_tokens = 0
                self.completion_tokens = 0
                self._stats_lock = threading.Lock()

            def summarize(self, context, max_tokens=180):
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a careful summarizer."},
                        {"role": "user", "content": f"Summarize the following. Keep key entities, dates, decisions:\n{context}"},
                    ],
                    temperature=0.2,
                    max_tokens=max(32, int(max_tokens)),
                )
                content = resp.choices[0].message.content or (context or "")[:2000]
                with self._stats_lock:
                    self.llm_calls += 1
                    try:
                        self.prompt_tokens += resp.usage.prompt_tokens or 0
                        self.completion_tokens += resp.usage.completion_tokens or 0
                    except Exception:
                        pass
                return content

        self._emb = _CompatEmbeddingModel()
        self._summ = _CompatSummarizationModel()
        config = RetrievalAugmentationConfig(
            embedding_model=self._emb,
            summarization_model=self._summ,
            qa_model=_NoQAModel(),
            tb_num_layers=self._tb_num_layers,
            tb_max_tokens=self._tb_max_tokens,
            tb_summarization_length=self._tb_summarization_length,
            tr_threshold=self._tr_threshold,
            tr_top_k=self._tr_top_k,
        )
        return RetrievalAugmentation, config

    def build_index(self) -> None:
        if not self._buffer:
            return
        RetrievalAugmentation, config = self._make_runtime()
        ra = RetrievalAugmentation(config=config, tree=None)
        text = "\n\n".join(self._buffer)
        t0 = _time.time()
        ra.add_documents(text)
        self._ingest_time_ms = int((_time.time() - t0) * 1000)
        self._ra = ra
        self.backend_mode = "raptor"
        print(f"[RaptorBenchMemory][DEBUG] indexed chunks={len(self._buffer)} tree_nodes={len(self._ra.tree.all_nodes) if self._ra and self._ra.tree else 0}")

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

            score = 0.0
            if node_emb is not None:
                d = float(distance.cosine(q_emb, node_emb))
                score = float(1.0 - d / 2.0)
            evidences.append(Evidence(content=node.text, metadata={"source": "RAPTOR", "node_index": idx, "layer": layer, "score": score}))

        if not evidences and isinstance(context, str) and context.strip():
            evidences.append(Evidence(content=context, metadata={"source": "RAPTOR", "node_index": -1, "layer": -1, "score": 0.0}))
        return evidences

    def reset(self) -> None:
        self._ra = None
        self._buffer.clear()
        self._ingest_time_ms = 0
        self._emb = None
        self._summ = None

    def audit_ingest(self) -> dict:
        llm_calls = 0
        prompt_tokens = 0
        completion_tokens = 0
        n_nodes = 0
        n_layers = 0
        if self.backend_mode == "raptor" and self._summ is not None:
            llm_calls = self._summ.llm_calls
            prompt_tokens = self._summ.prompt_tokens
            completion_tokens = self._summ.completion_tokens
        if self._ra and self._ra.tree:
            n_nodes = len(self._ra.tree.all_nodes)
            n_layers = self._ra.tree.num_layers
        return {
            "backend_mode": self.backend_mode,
            "ingest_chunks": len(self._buffer),
            "ingest_time_ms": self._ingest_time_ms,
            "ingest_llm_calls": llm_calls,
            "ingest_llm_prompt_tokens": prompt_tokens,
            "ingest_llm_completion_tokens": completion_tokens,
            "tree_nodes": n_nodes,
            "tree_layers": n_layers,
        }
