#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四后端“分开做”全 benchmark 运行器。

设计目标：
1) 先按后端分组，再在组内跑 AMemGym / memory-probe / StructMemEval。
2) 可视化调试输出清晰：阶段横幅、实时日志前缀、每后端小结、最终总表。
3) 正确性优先：默认严格失败（某脚本失败立即停止）；可用 --continue-on-fail 放宽。
4) 每一步都持久化日志，便于定位问题和复跑。
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


def banner(title: str) -> None:
    line = "=" * 100
    print(f"\n{line}\n{title}\n{line}", flush=True)


def sub_banner(title: str) -> None:
    line = "-" * 100
    print(f"\n{line}\n{title}\n{line}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full benchmarks backend-by-backend (split mode) with strong debug output."
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=list(BACKEND_TO_SCRIPTS.keys()) + ["all"],
        default=["all"],
        help="要跑的后端列表，默认 all。",
    )
    parser.add_argument(
        "--logs-dir",
        default="logs/full_benchmark_split",
        help="日志输出目录，默认 logs/full_benchmark_split。",
    )
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="遇到失败后继续跑后续项（默认关闭，默认策略为正确性优先的严格失败）。",
    )
    return parser.parse_args()


def run_script(
    *,
    backend: str,
    script: str,
    backend_step: int,
    backend_total: int,
    global_step: int,
    global_total: int,
    logs_dir: Path,
) -> RunRecord:
    script_path = Path(script)
    if not script_path.exists():
        raise FileNotFoundError(f"脚本不存在: {script}")

    log_path = logs_dir / f"{global_step:02d}_{backend}_{script_path.stem}.log"
    start = time.time()

    sub_banner(
        f"[GLOBAL {global_step}/{global_total}] [BACKEND {backend_step}/{backend_total}] "
        f"backend={backend} script={script} start={utc_now()}"
    )
    print(f"[DEBUG] log_file={log_path}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"# start: {utc_now()}\n")
        log_file.write(f"# backend: {backend}\n")
        log_file.write(f"# script: {script}\n\n")

        process = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert process.stdout is not None

        for line in process.stdout:
            log_file.write(line)
            print(f"[{backend}/{script_path.stem}] {line.rstrip()}", flush=True)

        return_code = process.wait()

    elapsed = time.time() - start
    status = "PASS" if return_code == 0 else f"FAIL({return_code})"
    print(
        f"[DEBUG] done backend={backend} script={script} status={status} elapsed={elapsed:.1f}s",
        flush=True,
    )

    return RunRecord(
        backend=backend,
        script=script,
        elapsed_sec=elapsed,
        status=status,
        log_path=str(log_path),
    )


def print_backend_summary(backend: str, records: list[RunRecord]) -> None:
    sub_banner(f"BACKEND SUMMARY: {backend}")
    print(f"{'#':<4}{'script':<40}{'status':<14}{'elapsed(s)':<12}")
    print("-" * 100)
    for idx, rec in enumerate(records, start=1):
        print(f"{idx:<4}{rec.script:<40}{rec.status:<14}{rec.elapsed_sec:<12.1f}")
    print("-" * 100)


def print_final_summary(all_records: list[RunRecord], started_at: float, logs_dir: Path) -> None:
    elapsed = time.time() - started_at
    banner("SPLIT FULL BENCHMARK SUMMARY")
    print(f"[DEBUG] finished_at={utc_now()}")
    print(f"[DEBUG] total_elapsed={elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"[DEBUG] logs_dir={logs_dir}")
    print("-" * 100)
    print(f"{'#':<4}{'backend':<12}{'script':<40}{'status':<14}{'elapsed(s)':<12}")
    print("-" * 100)
    for idx, rec in enumerate(all_records, start=1):
        print(f"{idx:<4}{rec.backend:<12}{rec.script:<40}{rec.status:<14}{rec.elapsed_sec:<12.1f}")
    print("-" * 100)


def backend_order(backends_arg: Iterable[str]) -> list[str]:
    if "all" in backends_arg:
        return list(BACKEND_TO_SCRIPTS.keys())
    return list(backends_arg)


def main() -> int:
    args = parse_args()
    selected_backends = backend_order(args.backends)
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    total_steps = sum(len(BACKEND_TO_SCRIPTS[b]) for b in selected_backends)
    strict_fail = not args.continue_on_fail

    banner("SPLIT RUNNER: 四后端分开做全 benchmark")
    print(f"[DEBUG] started_at={utc_now()}")
    print(f"[DEBUG] selected_backends={selected_backends}")
    print(f"[DEBUG] total_steps={total_steps}")
    print(f"[DEBUG] strict_fail={strict_fail}")
    print(f"[DEBUG] python={sys.executable}")

    started_at = time.time()
    all_records: list[RunRecord] = []
    global_step = 0

    for backend in selected_backends:
        scripts = BACKEND_TO_SCRIPTS[backend]
        backend_records: list[RunRecord] = []
        banner(f"BACKEND PHASE START: {backend} ({len(scripts)} scripts)")

        for idx, script in enumerate(scripts, start=1):
            global_step += 1
            rec = run_script(
                backend=backend,
                script=script,
                backend_step=idx,
                backend_total=len(scripts),
                global_step=global_step,
                global_total=total_steps,
                logs_dir=logs_dir,
            )
            backend_records.append(rec)
            all_records.append(rec)

            if rec.status != "PASS" and strict_fail:
                print_backend_summary(backend, backend_records)
                print_final_summary(all_records, started_at, logs_dir)
                raise RuntimeError(
                    f"backend={backend}, script={script} 失败，日志={rec.log_path}。"
                    "当前为严格失败模式，已停止后续任务。"
                )

        print_backend_summary(backend, backend_records)

    print_final_summary(all_records, started_at, logs_dir)

    any_failed = any(r.status != "PASS" for r in all_records)
    return 1 if any_failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - 统一输出失败摘要
        banner("SPLIT FULL BENCHMARK FAILED")
        print(f"[ERROR] {exc}")
        raise
