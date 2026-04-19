#!/usr/bin/env python3
from __future__ import annotations

from ingest_smoke_common import run_ingest_smoke
from memgpt_bench_src import MemGPTBenchMemory


if __name__ == "__main__":
    raise SystemExit(run_ingest_smoke("memgpt", lambda save_dir: MemGPTBenchMemory(save_dir=save_dir)))
