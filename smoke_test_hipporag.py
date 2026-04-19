#!/usr/bin/env python3
from __future__ import annotations

from hipporag_bench_src import HippoRAGMemory
from smoke_common import default_preflight, parse_args, run_backend_smoke


def main() -> int:
    args = parse_args()
    return run_backend_smoke(
        backend="hipporag",
        memory_factory=lambda save_dir: HippoRAGMemory(save_dir=save_dir),
        only=args.only,
        preflight=default_preflight,
    )


if __name__ == "__main__":
    raise SystemExit(main())
