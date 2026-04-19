#!/usr/bin/env python3
from __future__ import annotations

from ingest_smoke_common import run_ingest_smoke
from hipporag_bench_src import HippoRAGMemory


if __name__ == "__main__":
    raise SystemExit(run_ingest_smoke("hipporag", lambda save_dir: HippoRAGMemory(save_dir=save_dir)))
