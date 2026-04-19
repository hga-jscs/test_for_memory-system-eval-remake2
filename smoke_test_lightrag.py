#!/usr/bin/env python3
from __future__ import annotations

from lightrag_bench_src import LightRAGBenchMemory
from smoke_common import default_preflight, parse_args, run_backend_smoke


def main() -> int:
    args = parse_args()
    return run_backend_smoke(
        backend="lightrag",
        memory_factory=lambda save_dir: LightRAGBenchMemory(save_dir=save_dir),
        only=args.only,
        preflight=default_preflight,
        require_response_format_check=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
