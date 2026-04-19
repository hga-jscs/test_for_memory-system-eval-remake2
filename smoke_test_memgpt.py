#!/usr/bin/env python3
from __future__ import annotations

from memgpt_bench_src import MemGPTBenchMemory
from smoke_common import memgpt_preflight, parse_args, run_backend_smoke


def main() -> int:
    args = parse_args()
    return run_backend_smoke(
        backend="memgpt",
        memory_factory=lambda save_dir: MemGPTBenchMemory(save_dir=save_dir),
        only=args.only,
        preflight=memgpt_preflight,
    )


if __name__ == "__main__":
    raise SystemExit(main())
