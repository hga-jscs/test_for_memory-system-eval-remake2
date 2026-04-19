#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = {
    "memgpt": "smoke_test_memgpt.py",
    "raptor": "smoke_test_raptor.py",
    "lightrag": "smoke_test_lightrag.py",
    "hipporag": "smoke_test_hipporag.py",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run smoke tests with unified report")
    p.add_argument("--backends", type=str, default="memgpt,raptor,lightrag,hipporag")
    p.add_argument("--only", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    selected = [b.strip() for b in args.backends.split(",") if b.strip()]
    invalid = [b for b in selected if b not in SCRIPTS]
    if invalid:
        print(f"Unknown backends: {invalid}")
        return 2

    rows: list[dict] = []
    any_fail = False

    for backend in selected:
        cmd = [sys.executable, SCRIPTS[backend]]
        if args.only:
            cmd.extend(["--only", args.only])
        print(f"\n=== Running {backend}: {' '.join(cmd)} ===")
        proc = subprocess.run(cmd)
        ok = proc.returncode == 0
        any_fail = any_fail or (not ok)

        latest = _latest_run_dir(backend)
        summary = _read_summary(latest / "summary.json") if latest else {}

        row = {
            "backend": backend,
            "exit_code": proc.returncode,
            "ok": ok,
            "run_dir": str(latest) if latest else "",
            "passed": summary.get("counts", {}).get("passed"),
            "failed": summary.get("counts", {}).get("failed"),
            "reason": summary.get("reason", ""),
        }
        rows.append(row)

    print("\n\n==== SMOKE TOTAL ====")
    print(f"{'backend':<12} {'ok':<5} {'exit':<5} {'passed':<6} {'failed':<6} run_dir")
    for r in rows:
        print(f"{r['backend']:<12} {str(r['ok']):<5} {r['exit_code']:<5} {str(r['passed']):<6} {str(r['failed']):<6} {r['run_dir']}")

    table_path = Path("results") / "smoke" / "smoke_run_table.json"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote table: {table_path}")

    return 1 if any_fail else 0


def _latest_run_dir(backend: str) -> Path | None:
    p = Path("results") / "smoke" / backend
    if not p.exists():
        return None
    dirs = [d for d in p.iterdir() if d.is_dir()]
    if not dirs:
        return None
    return sorted(dirs)[-1]


def _read_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


if __name__ == "__main__":
    raise SystemExit(main())
