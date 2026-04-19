#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified observability helpers for benchmark/smoke runs.

Prioritizes auditability over performance:
- run-level manifest/summary
- case-level JSONL
- call-level JSONL
- retrieval trace JSONL
- deterministic error/quality classification
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_human() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_version(pkg_name: str) -> str:
    try:
        from importlib.metadata import version

        return version(pkg_name)
    except Exception:
        return "unknown"


def collect_runtime_metadata() -> dict[str, Any]:
    commit = "unknown"
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
            .strip()
        )
    except Exception:
        pass

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "git_commit": commit,
        "dependency_versions": {
            "openai": _safe_version("openai"),
            "numpy": _safe_version("numpy"),
            "pandas": _safe_version("pandas"),
            "scipy": _safe_version("scipy"),
            "torch": _safe_version("torch"),
        },
    }


def stable_run_id(*, backend: str, benchmark: str, salt: str = "") -> str:
    raw = f"{utc_ts()}::{backend}::{benchmark}::{salt or os.getpid()}"
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{utc_ts()}_{backend}_{benchmark}_{h}"


@dataclass
class ObservableRun:
    backend: str
    benchmark: str
    run_id: str
    base_dir: Path = Path("results")
    strict_fail: bool = True
    max_workers: int = 1
    model_config: dict[str, Any] = field(default_factory=dict)

    started_at: str = field(default_factory=utc_human)
    finished_at: str | None = None
    counts: dict[str, int] = field(default_factory=lambda: {"total": 0, "completed": 0, "failed": 0, "skipped": 0})
    failures: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.run_dir = self.base_dir / self.backend / self.benchmark / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.paths = {
            "manifest": self.run_dir / "manifest.json",
            "summary": self.run_dir / "summary.json",
            "summary_md": self.run_dir / "summary.md",
            "cases": self.run_dir / "cases.jsonl",
            "llm_calls": self.run_dir / "llm_calls.jsonl",
            "retrievals": self.run_dir / "retrievals.jsonl",
            "metrics": self.run_dir / "metrics.json",
            "config_snapshot": self.run_dir / "config_snapshot.json",
            "stdout": self.run_dir / "stdout.log",
            "stderr": self.run_dir / "stderr.log",
        }

        self.runtime = collect_runtime_metadata()
        self._write_json(
            self.paths["config_snapshot"],
            {
                "backend": self.backend,
                "benchmark": self.benchmark,
                "run_id": self.run_id,
                "started_at": self.started_at,
                "strict_fail": self.strict_fail,
                "max_workers": self.max_workers,
                "model_config": self.model_config,
                "runtime": self.runtime,
            },
        )
        self.flush_manifest(ok=None, reason="running")

    def _write_json(self, path: Path, obj: dict[str, Any]) -> None:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_jsonl(self, path: Path, obj: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def log_case(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.paths["cases"], record)

    def log_call(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.paths["llm_calls"], record)

    def log_retrieval(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.paths["retrievals"], record)

    def add_failure(self, *, case_id: str, error_type: str, message: str, stage: str = "unknown") -> None:
        self.failures.append(
            {
                "at": utc_human(),
                "case_id": case_id,
                "error_type": error_type,
                "message": message,
                "stage": stage,
            }
        )

    def flush_manifest(self, *, ok: bool | None, reason: str) -> None:
        manifest = {
            "backend": self.backend,
            "benchmark": self.benchmark,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "strict_fail": self.strict_fail,
            "max_workers": self.max_workers,
            "counts": self.counts,
            "ok": ok,
            "reason": reason,
            "runtime": self.runtime,
            "model_config": self.model_config,
            "artifacts": {k: str(v) for k, v in self.paths.items()},
            "failures": self.failures,
        }
        self._write_json(self.paths["manifest"], manifest)

    def finalize(self, *, ok: bool, reason: str, extra_summary: dict[str, Any] | None = None) -> None:
        self.finished_at = utc_human()
        self.flush_manifest(ok=ok, reason=reason)

        summary = {
            "backend": self.backend,
            "benchmark": self.benchmark,
            "run_id": self.run_id,
            "ok": ok,
            "reason": reason,
            "counts": self.counts,
            "failures": self.failures,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if extra_summary:
            summary.update(extra_summary)
        self._write_json(self.paths["summary"], summary)
        self._write_json(self.paths["metrics"], extra_summary or {})

        md = [
            f"# Summary: {self.backend}/{self.benchmark}",
            "",
            f"- run_id: `{self.run_id}`",
            f"- ok: `{ok}`",
            f"- reason: `{reason}`",
            f"- total/completed/failed/skipped: {self.counts['total']}/{self.counts['completed']}/{self.counts['failed']}/{self.counts['skipped']}",
            f"- started_at: {self.started_at}",
            f"- finished_at: {self.finished_at}",
        ]
        if self.failures:
            md.append("\n## Failures")
            for f in self.failures[:20]:
                md.append(f"- `{f['case_id']}` [{f['stage']}/{f['error_type']}] {f['message']}")
        self.paths["summary_md"].write_text("\n".join(md) + "\n", encoding="utf-8")


def classify_failure(exc: Exception) -> str:
    msg = str(exc).lower()
    if "timeout" in msg:
        return "timeout"
    if "connection" in msg or "refused" in msg:
        return "connection_error"
    if "json" in msg and "parse" in msg:
        return "parse_error"
    if "import" in msg or "module" in msg:
        return "dependency_missing"
    if "embedding" in msg and "dim" in msg:
        return "embedding_dimension_mismatch"
    return "runtime_error"


def classify_answer_quality(
    *,
    question: str,
    answer: str,
    gold: str,
    retrieval_texts: list[str],
    retrieval_scores: list[float] | None = None,
) -> dict[str, Any]:
    q = (question or "").lower()
    a = (answer or "").lower().strip()
    g = (gold or "").lower().strip()
    texts = [t.lower() for t in retrieval_texts if t]
    scores = retrieval_scores or [0.0 for _ in texts]

    has_gold_evidence = any(g and g[:24] in t for t in texts)
    denied = ("未提到" in answer) or ("not mention" in a) or ("不知道" in answer)
    hallucination = bool(a) and not any(a[:24] in t for t in texts) and not has_gold_evidence

    tags: list[str] = []
    if denied and has_gold_evidence:
        tags.append("denial_conflicts_with_topk")
    if "when" in q or "时间" in q or "哪天" in q:
        if any(tok in a for tok in ["yesterday", "tomorrow", "今天", "明天"]) and not any(ch.isdigit() for ch in a):
            tags.append("time_anchor_weak")
    if any(w in q for w in ["谁", "who", "person", "人物"]) and any(w in a for w in ["他", "她", "they"]):
        tags.append("entity_binding_risk")
    if hallucination:
        tags.append("unsupported_by_topk")
    if has_gold_evidence and g and g not in a:
        tags.append("retrieved_but_answer_missed_key_evidence")

    score_hint = max(scores) if scores else 0.0
    guessed = (g and g in a) and (score_hint < 0.15)
    if guessed:
        tags.append("possibly_lucky_guess")

    return {
        "quality_tags": tags,
        "has_gold_evidence": has_gold_evidence,
        "possible_guess": guessed,
    }
