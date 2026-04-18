#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一条指令跑完四后端全 benchmark。

覆盖后端: memgpt(letta) / raptor / lightrag / hipporag
覆盖数据集: AMemGym / memory-probe / StructMemEval

特性:
- 可视化调试输出（清晰阶段头、实时子进程日志前缀、最终汇总表）
- 严格失败（某一步失败立即退出，避免悄悄跳过）
- 自动落盘日志到 logs/full_benchmark/
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

BACKEND_TO_SCRIPTS = {
    "memgpt": [
        "bench_amemgym_memgpt.py",
        "bench_memory_probe_memgpt.py",
        "bench_structmemeval_memgpt.py",
    ],
    "raptor": [
        "bench_amemgym_raptor.py",
        "bench_memory_probe_raptor.py",
        "bench_structmemeval_raptor.py",
    ],
    "lightrag": [
        "bench_amemgym_lightrag.py",
        "bench_memory_probe_lightrag.py",
        "bench_structmemeval_lightrag.py",
    ],
    "hipporag": [
        "bench_amemgym_hipporag.py",
        "bench_memory_probe_hipporag.py",
        "bench_structmemeval_hipporag.py",
    ],
}


@dataclass
class RunRecord:
    backend: str
    script: str
    elapsed_sec: float
    status: str
    log_path: str


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def print_banner(title: str) -> None:
    line = "=" * 92
    print(f"\n{line}\n{title}\n{line}", flush=True)


def build_plan(backends: Iterable[str]) -> list[tuple[str, str]]:
    plan: list[tuple[str, str]] = []
    for b in backends:
        scripts = BACKEND_TO_SCRIPTS[b]
        for s in scripts:
            plan.append((b, s))
    return plan


def run_one(step_idx: int, total_steps: int, backend: str, script: str, logs_dir: Path) -> RunRecord:
    script_path = Path(script)
    if not script_path.exists():
        raise FileNotFoundError(f"脚本不存在: {script}")

    log_path = logs_dir / f"{step_idx:02d}_{backend}_{script_path.stem}.log"
    start = time.time()

    print_banner(f"[{step_idx}/{total_steps}] backend={backend} | script={script} | start={utc_now()}")
    print(f"[DEBUG] log file: {log_path}", flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"# start: {utc_now()}\n")
        lf.write(f"# backend: {backend}\n")
        lf.write(f"# script: {script}\n\n")

        proc = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert proc.stdout is not None

        for line in proc.stdout:
            lf.write(line)
            print(f"[{backend}/{script_path.stem}] {line.rstrip()}" , flush=True)

        return_code = proc.wait()

    elapsed = time.time() - start
    status = "PASS" if return_code == 0 else f"FAIL({return_code})"
    print(f"[DEBUG] finished backend={backend} script={script} status={status} elapsed={elapsed:.1f}s", flush=True)
    if return_code != 0:
        raise RuntimeError(f"{script} 执行失败，退出码={return_code}，日志={log_path}")

    return RunRecord(
        backend=backend,
        script=script,
        elapsed_sec=elapsed,
        status=status,
        log_path=str(log_path),
    )


def print_summary(records: list[RunRecord], started_at: float, logs_dir: Path) -> None:
    total = time.time() - started_at
    print_banner("FULL BENCHMARK SUMMARY")
    print(f"[DEBUG] finished_at={utc_now()}")
    print(f"[DEBUG] total_elapsed={total:.1f}s ({total / 60:.1f} min)")
    print(f"[DEBUG] logs_dir={logs_dir}")
    print("-" * 92)
    print(f"{'#':<4}{'backend':<12}{'script':<36}{'status':<12}{'elapsed(s)':<12}")
    print("-" * 92)
    for i, r in enumerate(records, start=1):
        print(f"{i:<4}{r.backend:<12}{r.script:<36}{r.status:<12}{r.elapsed_sec:<12.1f}")
    print("-" * 92)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full benchmark for selected backends in one command.")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=list(BACKEND_TO_SCRIPTS.keys()) + ["all"],
        default=["all"],
        help="要跑的后端。默认 all。",
    )
    parser.add_argument(
        "--logs-dir",
        default="logs/full_benchmark",
        help="日志输出目录。默认 logs/full_benchmark",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backends = list(BACKEND_TO_SCRIPTS.keys()) if "all" in args.backends else args.backends
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    plan = build_plan(backends)
    print_banner("ONE-COMMAND FULL BENCHMARK RUNNER")
    print(f"[DEBUG] started_at={utc_now()}")
    print(f"[DEBUG] selected_backends={backends}")
    print(f"[DEBUG] total_steps={len(plan)}")
    print(f"[DEBUG] python={sys.executable}")

    started_at = time.time()
    records: list[RunRecord] = []

    for idx, (backend, script) in enumerate(plan, start=1):
        rec = run_one(idx, len(plan), backend, script, logs_dir)
        records.append(rec)

    print_summary(records, started_at, logs_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - 需要统一打印错误摘要
        print_banner("FULL BENCHMARK FAILED")
        print(f"[ERROR] {exc}")
        raise
