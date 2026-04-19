#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = {
    "memgpt": "smoke_ingest_memgpt.py",
    "raptor": "smoke_ingest_raptor.py",
    "hipporag": "smoke_ingest_hipporag.py",
    "lightrag": "smoke_ingest_lightrag.py",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="run ingest smoke tests only")
    p.add_argument("--backends", type=str, default="memgpt,raptor,hipporag,lightrag")
    p.add_argument("--strict-fail", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    backends = [x.strip() for x in args.backends.split(",") if x.strip()]
    rows = []
    any_fail = False
    for b in backends:
        if b not in SCRIPTS:
            print(f"Unknown backend: {b}")
            return 2
        cmd = [sys.executable, SCRIPTS[b]]
        print(f"=== RUN {b}: {' '.join(cmd)}")
        p = subprocess.run(cmd)
        row = {"backend": b, "exit_code": p.returncode, "ok": p.returncode == 0}
        rows.append(row)
        any_fail = any_fail or p.returncode != 0

    out = Path("results") / "ingest_smoke" / "ingest_smoke_table.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": rows}, ensure_ascii=False, indent=2))
    if args.strict_fail and any_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
