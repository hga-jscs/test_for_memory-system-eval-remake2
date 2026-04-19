#!/usr/bin/env python3
from __future__ import annotations

from ingest_smoke_common import run_ingest_smoke
from lightrag_bench_src import LightRAGBenchMemory


if __name__ == "__main__":
    raise SystemExit(run_ingest_smoke("lightrag", lambda save_dir: LightRAGBenchMemory(save_dir=save_dir)))
