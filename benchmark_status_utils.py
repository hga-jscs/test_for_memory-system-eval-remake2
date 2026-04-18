#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared benchmark status checks.

Correctness rule: a benchmark run is only considered successful when at least one
non-skipped case has been evaluated successfully and not every case errored out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutionHealth:
    total_tasks: int
    completed_cases: int
    skipped_cases: int
    error_cases: int
    evaluated_queries: int
    ok: bool
    reason: str


def evaluate_execution_health(results: list[dict[str, Any]], errors: list[dict[str, Any]], total_tasks: int) -> ExecutionHealth:
    completed = [r for r in results if not r.get("skipped")]
    skipped_cases = len(results) - len(completed)
    evaluated_queries = sum(int(r.get("n_queries", 0)) for r in completed)
    error_cases = len(errors)

    if total_tasks <= 0:
        return ExecutionHealth(total_tasks, 0, skipped_cases, error_cases, evaluated_queries, False, "no tasks collected")
    if len(completed) <= 0:
        return ExecutionHealth(total_tasks, 0, skipped_cases, error_cases, evaluated_queries, False, "no successful case execution")
    if error_cases >= total_tasks:
        return ExecutionHealth(total_tasks, len(completed), skipped_cases, error_cases, evaluated_queries, False, "all tasks failed")
    if evaluated_queries <= 0:
        return ExecutionHealth(total_tasks, len(completed), skipped_cases, error_cases, evaluated_queries, False, "no evaluated queries")

    return ExecutionHealth(total_tasks, len(completed), skipped_cases, error_cases, evaluated_queries, True, "ok")
