#!/usr/bin/env python3
from __future__ import annotations

from ingest_smoke_common import run_ingest_smoke
from raptor_bench_src import RaptorBenchMemory


if __name__ == "__main__":
    raise SystemExit(run_ingest_smoke("raptor", lambda save_dir: RaptorBenchMemory(save_dir=save_dir)))
