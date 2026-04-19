#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"


@dataclass
class IngestAuditWriter:
    backend: str
    save_dir: str
    run_id: str = field(default_factory=utc_run_id)
    root: Path = field(init=False)
    chunks_path: Path = field(init=False)
    summary_path: Path = field(init=False)
    config_path: Path = field(init=False)
    manifest_path: Path = field(init=False)
    provenance_path: Path = field(init=False)
    stdout_path: Path = field(init=False)
    stderr_path: Path = field(init=False)
    _chunks: List[Dict[str, Any]] = field(default_factory=list)
    _stdout: List[str] = field(default_factory=list)
    _stderr: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path("results") / "ingest_audit" / self.backend / self.run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.chunks_path = self.root / "ingest_chunks.jsonl"
        self.summary_path = self.root / "ingest_summary.json"
        self.config_path = self.root / "config_snapshot.json"
        self.manifest_path = self.root / "storage_manifest.json"
        self.provenance_path = self.root / "provenance_map.json"
        self.stdout_path = self.root / "stdout.log"
        self.stderr_path = self.root / "stderr.log"

    def log_stdout(self, msg: str) -> None:
        self._stdout.append(msg)

    def log_stderr(self, msg: str) -> None:
        self._stderr.append(msg)

    def add_chunk(self, row: Dict[str, Any]) -> None:
        self._chunks.append(row)

    def write_config(self, config: Dict[str, Any]) -> None:
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_provenance(self, data: Dict[str, Any]) -> None:
        self.provenance_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def finalize(self, summary: Dict[str, Any], storage_manifest: Dict[str, Any]) -> None:
        with self.chunks_path.open("w", encoding="utf-8") as f:
            for row in self._chunks:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        self.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self.manifest_path.write_text(json.dumps(storage_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self.stdout_path.write_text("\n".join(self._stdout) + ("\n" if self._stdout else ""), encoding="utf-8")
        self.stderr_path.write_text("\n".join(self._stderr) + ("\n" if self._stderr else ""), encoding="utf-8")


def parse_time_to_iso(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def compact_error(exc: BaseException) -> Dict[str, Any]:
    return {
        "error": str(exc),
        "type": exc.__class__.__name__,
        "traceback": traceback.format_exc(limit=6),
    }
