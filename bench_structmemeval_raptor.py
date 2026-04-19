#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StructMemEval benchmark for RAPTOR with auditable artifacts.

Core change: produce run/case/call/retrieval artifacts under
results/<backend>/<benchmark>/<run_id>/
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmark_io_utils import load_json_with_fallback
from benchmark_observability import ObservableRun, classify_answer_quality, classify_failure, stable_run_id, utc_human
from benchmark_status_utils import evaluate_execution_health
from raptor_bench_src import RaptorBenchMemory
from simpleMem_src import OpenAIClient, get_config

BASE = Path("StructMemEval/benchmark")
CATEGORIES = {
    "state_machine_location": BASE / "data" / "state_machine_location",
    "tree_based": BASE / "tree_based" / "graph_configs",
    "recommendations": BASE / "recommendations" / "data",
}
SAVE_BASE = Path("/tmp/bench_raptor_sme")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="StructMemEval benchmark (RAPTOR) with observability outputs")
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="Limit number of cases, 0 means all")
    p.add_argument("--strict-fail", action="store_true", help="Stop on first hard error")
    p.add_argument("--benchmark-name", default="structmemeval")
    return p.parse_args()


def collect_cases(limit: int = 0) -> list[dict]:
    tasks = []
    for cat, cat_dir in CATEGORIES.items():
        if not cat_dir.exists():
            print(f"[WARN] path missing: {cat_dir}")
            continue
        for fp in sorted(cat_dir.rglob("*.json")):
            tasks.append({"category": cat, "path": fp})
    if limit > 0:
        tasks = tasks[:limit]
    return tasks


def ingest_case(mem: RaptorBenchMemory, case: dict, category: str) -> None:
    for session in case.get("sessions", []):
        sid = session.get("session_id", "?")
        topic = session.get("topic", "")
        msgs = session.get("messages", [])
        header = f"[Session: {sid}" + (f", Topic: {topic}" if topic else "") + "]"
        if category == "tree_based":
            body = "\n".join(m.get("content", "") for m in msgs)
        else:
            body = "\n".join(f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in msgs)
        mem.add_text(header + "\n" + body)


def answer_with_memory(llm: OpenAIClient, mem: RaptorBenchMemory, question: str, run: ObservableRun, case_key: str) -> tuple[str, list[dict], int]:
    t0 = time.time()
    evidences = mem.retrieve(question, top_k=10)
    latency_ms = int((time.time() - t0) * 1000)

    topk = []
    for i, e in enumerate(evidences, start=1):
        topk.append(
            {
                "rank": i,
                "score": float(e.metadata.get("score", 0.0) or 0.0),
                "source": e.metadata.get("source", "RAPTOR"),
                "memory_id": e.metadata.get("node_index"),
                "chunk_id": e.metadata.get("node_index"),
                "chunk_text": e.content,
            }
        )

    run.log_retrieval(
        {
            "ts": utc_human(),
            "case_id": case_key,
            "stage": "retrieve",
            "question": question,
            "top_k": topk,
            "latency_ms": latency_ms,
        }
    )

    ctx = "\n\n".join(f"[{i+1}] {e.content}" for i, e in enumerate(evidences))
    prompt = (
        "Based on the conversation memories below, answer the question concisely. "
        "If evidence is insufficient, explicitly say evidence is insufficient.\n\n"
        f"## Memories\n{ctx}\n\n## Question\n{question}\n\n## Answer:"
    )

    call_start_tokens = llm.total_tokens
    t1 = time.time()
    answer = llm.generate(prompt, temperature=0.0, max_tokens=300)
    call_latency = int((time.time() - t1) * 1000)
    used_tokens = llm.total_tokens - call_start_tokens

    run.log_call(
        {
            "ts": utc_human(),
            "case_id": case_key,
            "stage": "answer",
            "request_summary": question[:200],
            "prompt": prompt[:4000],
            "response_text": answer,
            "structured_parse": None,
            "token_usage": used_tokens,
            "latency_ms": call_latency,
            "retry_count": 0,
            "exception_type": None,
            "response_format_enabled": False,
            "parse_path": "plain_text",
        }
    )
    return answer, topk, used_tokens


def judge_answer(llm: OpenAIClient, question: str, pred: str, reference: dict, category: str, run: ObservableRun, case_key: str) -> tuple[bool, int]:
    ref_text = reference.get("text", "")
    if category == "recommendations":
        criteria = reference.get("evaluation_criteria", [])
        criteria_str = "\n".join(f"- {c}" for c in criteria) if criteria else ""
        prompt = (
            "You are a strict judge evaluating a memory recall answer.\n\n"
            f"Question: {question}\nReference answer: {ref_text}\n"
            + (f"Evaluation criteria:\n{criteria_str}\n" if criteria_str else "")
            + f"Predicted answer: {pred}\n\n"
            "Does the predicted answer satisfy the criteria and semantically match the reference?\n"
            "Answer ONLY 'yes' or 'no'."
        )
    else:
        prompt = (
            "Does the predicted answer semantically match the reference?\n"
            "Answer ONLY 'yes' or 'no'.\n\n"
            f"Question: {question}\nReference: {ref_text}\nPredicted: {pred}\n\nMatch:"
        )

    before = llm.total_tokens
    t0 = time.time()
    result = llm.generate(prompt, temperature=0.0, max_tokens=10)
    latency_ms = int((time.time() - t0) * 1000)
    used = llm.total_tokens - before

    run.log_call(
        {
            "ts": utc_human(),
            "case_id": case_key,
            "stage": "judge",
            "request_summary": question[:200],
            "prompt": prompt[:3000],
            "response_text": result,
            "structured_parse": None,
            "token_usage": used,
            "latency_ms": latency_ms,
            "retry_count": 0,
            "exception_type": None,
            "response_format_enabled": False,
            "parse_path": "plain_text",
        }
    )
    return ("yes" in result.lower()), used


def eval_case(task: dict, run: ObservableRun) -> dict:
    category = task["category"]
    path = task["path"]
    case = load_json_with_fallback(path)
    case_id = case.get("case_id", path.stem)
    case_key = f"{category}/{case_id}"
    queries = case.get("queries", [])

    if not queries:
        rec = {
            "case_id": case_id,
            "category": category,
            "dataset": "StructMemEval",
            "split": "all",
            "question": None,
            "gold_answer": None,
            "backend_answer": None,
            "correct": None,
            "error_type": "skipped",
            "top_k": [],
            "options": None,
            "pred_option_idx": None,
            "correct_option_idx": None,
            "ingest_time_ms": 0,
            "infer_time_ms": 0,
            "token_usage": 0,
            "llm_call_count": 0,
            "fallback_used": False,
            "parse_error": False,
            "retrieved_but_denied": False,
            "reason": "no queries",
        }
        run.log_case(rec)
        return {"case_id": case_id, "category": category, "skipped": True, "reason": "no queries"}

    config = get_config()
    save_dir = str(SAVE_BASE / f"{category[:12]}_{case_id[:16]}")

    mem = RaptorBenchMemory(save_dir=save_dir)
    llm = OpenAIClient(api_key=config.llm["api_key"], base_url=config.llm["base_url"], model=config.llm["model"])

    ingest_case(mem, case, category)
    t0 = time.time()
    mem.build_index()
    ingest_time_ms = int((time.time() - t0) * 1000)
    audit = mem.audit_ingest()

    t1 = time.time()
    correct = 0
    qa_results = []
    total_tokens = 0

    for q_idx, q in enumerate(queries):
        question = q.get("question", "")
        reference = q.get("reference_answer", {})
        gold = reference.get("text", "")

        pred, topk, answer_tokens = answer_with_memory(llm, mem, question, run, f"{case_key}#{q_idx}")
        ok, judge_tokens = judge_answer(llm, question, pred, reference, category, run, f"{case_key}#{q_idx}")

        total_tokens += answer_tokens + judge_tokens
        if ok:
            correct += 1

        quality = classify_answer_quality(
            question=question,
            answer=pred,
            gold=gold,
            retrieval_texts=[t["chunk_text"] for t in topk],
            retrieval_scores=[float(t.get("score", 0.0) or 0.0) for t in topk],
        )

        err_type = "none" if ok else "wrong_answer"
        if "unsupported_by_topk" in quality["quality_tags"]:
            err_type = "hallucination"
        elif "retrieved_but_answer_missed_key_evidence" in quality["quality_tags"]:
            err_type = "retrieval_answer_disconnect"

        case_record = {
            "case_id": case_id,
            "user_id": None,
            "conv_id": case_id,
            "dataset": "StructMemEval",
            "split": "all",
            "category": category,
            "question": question,
            "gold_answer": gold,
            "gold_label": None,
            "gold_option_index": None,
            "backend_answer": pred,
            "correct": ok,
            "error_type": err_type,
            "quality_tags": quality["quality_tags"],
            "top_k": topk,
            "options": None,
            "pred_option_index": None,
            "correct_option_index": None,
            "ingest_time_ms": ingest_time_ms,
            "infer_time_ms": None,
            "token_usage": answer_tokens + judge_tokens,
            "llm_call_count": 2,
            "fallback_used": False,
            "parse_error": False,
            "retrieved_but_denied": "denial_conflicts_with_topk" in quality["quality_tags"],
            "possible_guess": quality["possible_guess"],
        }
        run.log_case(case_record)

        qa_results.append(
            {
                "question": question[:120],
                "correct": ok,
                "pred": pred[:160],
                "quality_tags": quality["quality_tags"],
            }
        )

    infer_time_ms = int((time.time() - t1) * 1000)
    mem.reset()

    return {
        "case_id": case_id,
        "category": category,
        "n_queries": len(queries),
        "correct": correct,
        "ingest_chunks": audit["ingest_chunks"],
        "ingest_time_ms": ingest_time_ms,
        "ingest_llm_calls": audit["ingest_llm_calls"],
        "ingest_llm_prompt": audit["ingest_llm_prompt_tokens"],
        "ingest_llm_completion": audit["ingest_llm_completion_tokens"],
        "tree_nodes": audit["tree_nodes"],
        "tree_layers": audit["tree_layers"],
        "infer_time_ms": infer_time_ms,
        "infer_llm_tokens": total_tokens,
        "qa_results": qa_results,
    }


def main() -> int:
    args = parse_args()
    SAVE_BASE.mkdir(parents=True, exist_ok=True)

    run = ObservableRun(
        backend="raptor",
        benchmark=args.benchmark_name,
        run_id=stable_run_id(backend="raptor", benchmark=args.benchmark_name),
        strict_fail=args.strict_fail,
        max_workers=args.max_workers,
        model_config={},
    )
    try:
        cfg = get_config()
        run.model_config = {
            "llm_model": cfg.llm.get("model"),
            "embedding_model": cfg.embedding.get("model"),
            "embedding_dim": cfg.embedding.get("dim"),
            "base_url": cfg.llm.get("base_url"),
        }
        run.flush_manifest(ok=None, reason="running")
    except Exception as exc:  # noqa: BLE001
        et = classify_failure(exc)
        run.counts["total"] = 0
        run.counts["failed"] = 1
        run.add_failure(case_id="__init__", error_type=et, message=str(exc), stage="config")
        run.finalize(ok=False, reason=f"config_error:{et}", extra_summary={"errors": [str(exc)]})
        print(f"[BENCH-STATUS] ok=False reason=config_error:{et} total=0 completed=0 skipped=0 errors=1 evaluated_queries=0")
        print(f"run_dir={run.run_dir}")
        return 2

    print("=" * 70)
    print("StructMemEval benchmark (RAPTOR, observable mode)")
    print("=" * 70)

    tasks = collect_cases(limit=args.limit)
    run.counts["total"] = len(tasks)
    cat_cnt = Counter(t["category"] for t in tasks)
    for cat, cnt in sorted(cat_cnt.items()):
        print(f"  {cat}: {cnt} cases")
    print(f"  total: {len(tasks)} | MAX_WORKERS={args.max_workers} | run_id={run.run_id}\n")

    results, errors = [], []
    t_wall_start = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(eval_case, t, run): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            done += 1
            task = futures[future]
            try:
                r = future.result()
                results.append(r)
                if r.get("skipped"):
                    run.counts["skipped"] += 1
                    tag = "SKIP"
                    detail = r["reason"]
                else:
                    run.counts["completed"] += 1
                    tag = f"{r['correct']}/{r['n_queries']}"
                    detail = (
                        f"chunks={r['ingest_chunks']} tree={r['tree_nodes']}nodes/{r['tree_layers']}L "
                        f"ingest={r['ingest_time_ms']}ms infer={r['infer_time_ms']}ms"
                    )
                print(f"[{done:3d}/{len(tasks)}] {r['category'][:22]:22s} {r['case_id'][:28]:28s} {tag} {detail}")
            except Exception as exc:  # noqa: BLE001
                et = classify_failure(exc)
                run.counts["failed"] += 1
                run.add_failure(case_id=str(task["path"]), error_type=et, message=str(exc), stage="eval_case")
                errors.append({"path": str(task["path"]), "error": str(exc), "error_type": et})
                print(f"[{done:3d}/{len(tasks)}] ERROR {task['path'].name}: {exc}")
                if args.strict_fail:
                    break

    wall_ms = int((time.time() - t_wall_start) * 1000)

    agg = defaultdict(lambda: dict(correct=0, total=0, ingest_ms=0, infer_ms=0, chunks=0, llm_calls=0, tree_nodes=0, tree_layers=0))
    for r in results:
        if r.get("skipped"):
            continue
        a = agg[r["category"]]
        a["correct"] += r["correct"]
        a["total"] += r["n_queries"]
        a["ingest_ms"] += r["ingest_time_ms"]
        a["infer_ms"] += r["infer_time_ms"]
        a["chunks"] += r["ingest_chunks"]
        a["llm_calls"] += r["ingest_llm_calls"]
        a["tree_nodes"] += r["tree_nodes"]
        a["tree_layers"] = max(a["tree_layers"], r["tree_layers"])

    g = defaultdict(int)
    for cat, a in sorted(agg.items()):
        for k, v in a.items():
            if k == "tree_layers":
                g[k] = max(g[k], v)
            else:
                g[k] += v

    overall = (g["correct"] / g["total"] * 100) if g["total"] else 0.0

    health = evaluate_execution_health(results=results, errors=errors, total_tasks=len(tasks))
    ok = health.ok and len(errors) == 0
    reason = health.reason if health.ok else f"{health.reason}; errors={len(errors)}"
    run.finalize(
        ok=ok,
        reason=reason,
        extra_summary={
            "overall_accuracy": round(overall, 3),
            "wall_ms": wall_ms,
            "errors": errors,
            "categories": agg,
            "health": health.__dict__,
        },
    )

    legacy_path = Path("results_structmemeval_raptor.json")
    legacy_path.write_text(json.dumps({"results": results, "errors": errors, "run_dir": str(run.run_dir)}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"run_dir={run.run_dir}")
    print(f"legacy_result={legacy_path}")
    print(
        "[BENCH-STATUS] "
        f"ok={ok} reason={reason} total={health.total_tasks} "
        f"completed={health.completed_cases} skipped={health.skipped_cases} "
        f"errors={health.error_cases} evaluated_queries={health.evaluated_queries}"
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
