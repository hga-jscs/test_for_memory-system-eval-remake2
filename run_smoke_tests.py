#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Representative smoke tests for observability + answer trustworthiness.

Design:
- Covers confusion/time/negation/retrieval-hit-but-wrong/retrieval-miss/parse-fallback cases.
- Executes per backend with backend-specific preflight checks.
- Writes auditable artifacts to results/<backend>/smoke/<run_id>/.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any

from benchmark_observability import ObservableRun, classify_answer_quality, stable_run_id, utc_human


@dataclass
class SmokeCase:
    case_id: str
    scenario: str
    question: str
    gold: str
    context: list[str]
    expected: str  # pass|fail


SMOKE_CASES: list[SmokeCase] = [
    SmokeCase(
        case_id="entity_confusion_01",
        scenario="多人物混淆",
        question="Alice 和 Bob 谁在 2024-03-02 买了红色自行车？",
        gold="Bob",
        context=[
            "2024-03-01 Alice bought a blue bike.",
            "2024-03-02 Bob bought a red bike from Pine Street store.",
        ],
        expected="pass",
    ),
    SmokeCase(
        case_id="time_norm_01",
        scenario="时间归一化",
        question="他是昨天提交的吗？请给出绝对日期。",
        gold="2024-03-02",
        context=["Reference now is 2024-03-03.", "He submitted the report yesterday, on 2024-03-02."],
        expected="pass",
    ),
    SmokeCase(
        case_id="precise_fact_01",
        scenario="精确事实题",
        question="订单号是多少？",
        gold="XJ-445-AB",
        context=["Order id XJ-445-AB was created by Carol."],
        expected="pass",
    ),
    SmokeCase(
        case_id="set_fact_01",
        scenario="集合型事实题",
        question="这次会议的三个主题是什么？",
        gold="budget, roadmap, hiring",
        context=["Topics: budget, roadmap, hiring."],
        expected="pass",
    ),
    SmokeCase(
        case_id="negation_01",
        scenario="否定题",
        question="文档有没有提到 David 的地址？",
        gold="没有提到",
        context=["The doc includes David's phone but no address."],
        expected="pass",
    ),
    SmokeCase(
        case_id="retrieval_hit_answer_wrong_01",
        scenario="检索命中但回答错",
        question="谁批准了预算？",
        gold="Emma",
        context=["Budget approval: Emma approved on 2024-02-10."],
        expected="fail",
    ),
    SmokeCase(
        case_id="retrieval_miss_01",
        scenario="检索未命中",
        question="合同签署城市？",
        gold="Seattle",
        context=["No city information in this memory."],
        expected="fail",
    ),
    SmokeCase(
        case_id="parse_fail_01",
        scenario="结构化解析失败",
        question="输出 JSON: {answer:string}",
        gold="ok",
        context=["Return malformed json to trigger fallback."],
        expected="fail",
    ),
]


def backend_preflight(backend: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def do_import(name: str, stage: str) -> None:
        try:
            importlib.import_module(name)
            checks.append({"stage": stage, "ok": True, "detail": f"import {name} ok"})
        except Exception as exc:  # noqa: BLE001
            checks.append({"stage": stage, "ok": False, "detail": f"import {name} failed: {exc}"})

    if backend == "raptor":
        do_import("raptor_bench_src", "raptor_import")
    elif backend == "hipporag":
        do_import("hipporag_bench_src", "hipporag_import")
        for dep in ["torch", "igraph", "pyarrow"]:
            do_import(dep, f"hipporag_dep_{dep}")
    elif backend == "lightrag":
        do_import("bench_memory_probe_lightrag", "lightrag_script_import")
    elif backend == "memgpt":
        do_import("bench_amemgym_memgpt", "memgpt_script_import")

    ok = all(c["ok"] for c in checks) if checks else True
    return {"ok": ok, "checks": checks}


def simple_retrieve(question: str, context: list[str]) -> list[dict[str, Any]]:
    q_tokens = set(re.findall(r"[A-Za-z0-9\-]+", question.lower()))
    out: list[dict[str, Any]] = []
    for idx, text in enumerate(context, start=1):
        t_tokens = set(re.findall(r"[A-Za-z0-9\-]+", text.lower()))
        overlap = len(q_tokens & t_tokens)
        score = overlap / (len(q_tokens) + 1e-6)
        out.append({"rank": idx, "score": round(score, 4), "chunk_text": text, "source": "synthetic", "memory_id": idx, "chunk_id": idx})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:3]


def simulate_answer(case: SmokeCase) -> tuple[str, bool, str | None]:
    if case.case_id == "retrieval_hit_answer_wrong_01":
        return "Liam approved it.", False, None
    if case.case_id == "retrieval_miss_01":
        return "合同签署城市是 Portland。", False, None
    if case.case_id == "parse_fail_01":
        return "{answer: ok", False, "json_parse_error"
    if "没有提到" in case.gold:
        return "没有提到 David 的地址。", True, None
    return case.gold, True, None


def run_backend_smoke(backend: str) -> tuple[bool, str]:
    run = ObservableRun(
        backend=backend,
        benchmark="smoke",
        run_id=stable_run_id(backend=backend, benchmark="smoke"),
        strict_fail=False,
        max_workers=1,
        model_config={"mode": "smoke-synthetic"},
    )
    run.counts["total"] = len(SMOKE_CASES)

    preflight = backend_preflight(backend)
    run.log_call(
        {
            "ts": utc_human(),
            "case_id": "__backend_preflight__",
            "stage": "preflight",
            "request_summary": backend,
            "prompt": None,
            "response_text": json.dumps(preflight, ensure_ascii=False),
            "structured_parse": preflight,
            "token_usage": 0,
            "latency_ms": 0,
            "retry_count": 0,
            "exception_type": None if preflight["ok"] else "dependency_missing",
            "response_format_enabled": False,
            "parse_path": "local_preflight",
        }
    )

    for case in SMOKE_CASES:
        t0 = time.time()
        topk = simple_retrieve(case.question, case.context)
        run.log_retrieval({"ts": utc_human(), "case_id": case.case_id, "stage": "retrieve", "question": case.question, "top_k": topk})

        ans, local_ok, parse_error = simulate_answer(case)

        quality = classify_answer_quality(
            question=case.question,
            answer=ans,
            gold=case.gold,
            retrieval_texts=[x["chunk_text"] for x in topk],
            retrieval_scores=[x["score"] for x in topk],
        )

        is_correct = case.gold.lower() in ans.lower()
        if parse_error:
            is_correct = False
        expected_pass = case.expected == "pass"
        smoke_ok = (is_correct == expected_pass)

        err_type = "none"
        if parse_error:
            err_type = parse_error
        elif not is_correct and expected_pass:
            err_type = "answer_quality_failure"
        elif is_correct and not expected_pass:
            err_type = "unexpected_pass"

        if smoke_ok:
            run.counts["completed"] += 1
        else:
            run.counts["failed"] += 1
            run.add_failure(case_id=case.case_id, error_type=err_type, message=f"scenario={case.scenario}", stage="smoke_eval")

        run.log_case(
            {
                "case_id": case.case_id,
                "user_id": None,
                "conv_id": case.case_id,
                "dataset": "synthetic_smoke",
                "split": "smoke",
                "scenario": case.scenario,
                "question": case.question,
                "gold_answer": case.gold,
                "gold_label": case.expected,
                "gold_option_index": None,
                "backend_answer": ans,
                "correct": smoke_ok,
                "error_type": err_type,
                "quality_tags": quality["quality_tags"],
                "top_k": topk,
                "options": None,
                "pred_option_index": None,
                "correct_option_index": None,
                "ingest_time_ms": 0,
                "infer_time_ms": int((time.time() - t0) * 1000),
                "token_usage": 0,
                "llm_call_count": 0,
                "fallback_used": bool(parse_error),
                "parse_error": bool(parse_error),
                "retrieved_but_denied": "denial_conflicts_with_topk" in quality["quality_tags"],
                "known_issue": (not preflight["ok"]) or (case.expected == "fail"),
            }
        )

    ok = run.counts["failed"] == 0 and preflight["ok"]
    reason = "ok" if ok else ("preflight_failed" if not preflight["ok"] else "smoke_case_failed")
    run.finalize(
        ok=ok,
        reason=reason,
        extra_summary={
            "preflight": preflight,
            "scenarios": [c.scenario for c in SMOKE_CASES],
        },
    )
    print(f"[SMOKE] backend={backend} ok={ok} reason={reason} run_dir={run.run_dir}")
    return ok, str(run.run_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run representative smoke tests for all/specific backends")
    p.add_argument("--backends", nargs="+", default=["memgpt", "raptor", "lightrag", "hipporag"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    all_ok = True
    for backend in args.backends:
        ok, _ = run_backend_smoke(backend)
        all_ok = all_ok and ok
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
