#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StructMemEval 全量评测 - HippoRAG 版
覆盖: state_machine_location (42) / tree_based (100) / recommendations (30)

注意 tree_based：全部 role=user 的关系图数据（"X works with Y"），
HippoRAG 的 OpenIE triple extraction 正是为此类数据设计，预期比 mem0 (0%) 大幅改善。
"""

import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from simpleMem_src import get_config, OpenAIClient
from hipporag_bench_src import (
    HippoRAGMemory,
    HippoRAGDependencyError,
    ensure_hipporag_runtime_dependencies,
)
from benchmark_io_utils import load_json_with_fallback
from benchmark_status_utils import evaluate_execution_health

BASE = Path("StructMemEval/benchmark")
CATEGORIES = {
    "state_machine_location": BASE / "data" / "state_machine_location",
    "tree_based":             BASE / "tree_based" / "graph_configs",
    "recommendations":        BASE / "recommendations" / "data",
}

MAX_WORKERS  = 1
RESULTS_PATH = Path("results_structmemeval_hipporag.json")
SAVE_BASE    = Path("/tmp/bench_hipporag_sme")


# ── 数据加载 ──────────────────────────────────────────────────────────────

def collect_cases():
    tasks = []
    for cat, cat_dir in CATEGORIES.items():
        if not cat_dir.exists():
            print(f"[WARN] 路径不存在: {cat_dir}")
            continue
        for fp in sorted(cat_dir.rglob("*.json")):
            tasks.append({"category": cat, "path": fp})
    return tasks


# ── Ingest ───────────────────────────────────────────────────────────────

def ingest_case(mem: HippoRAGMemory, case: dict, category: str) -> None:
    """将 case 的 sessions 序列化为文本，预分块写入 buffer。

    tree_based：全 user-role 消息直接拼接后按 chunk_size 分块（避免 mem0 的巨型 chunk 问题）。
    其余：每 session 格式化为 header + 对话体后预分块。
    """
    for session in case.get("sessions", []):
        sid   = session.get("session_id", "?")
        topic = session.get("topic", "")
        msgs  = session.get("messages", [])
        header = f"[Session: {sid}" + (f", Topic: {topic}" if topic else "") + "]"

        if category == "tree_based":
            # 全 user-role：忽略 role 前缀，直接拼接事实语句
            body = "\n".join(m["content"] for m in msgs)
        else:
            body = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)

        mem.add_text(header + "\n" + body)


# ── Answer & Judge ────────────────────────────────────────────────────────

def answer_with_memory(llm: OpenAIClient, mem: HippoRAGMemory, question: str) -> str:
    evidences = mem.retrieve(question, top_k=5)
    ctx = "\n\n".join(f"[{i+1}] {e.content}" for i, e in enumerate(evidences))
    prompt = (
        "Based on the conversation memories below, answer the question concisely.\n\n"
        f"## Memories\n{ctx}\n\n## Question\n{question}\n\n## Answer:"
    )
    return llm.generate(prompt, temperature=0.0, max_tokens=300)


def judge_answer(llm: OpenAIClient, question: str, pred: str,
                 reference: dict, category: str) -> bool:
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
    result = llm.generate(prompt, temperature=0.0, max_tokens=10)
    return "yes" in result.lower()


# ── 单 case 评测 ──────────────────────────────────────────────────────────

def eval_case(task: dict) -> dict:
    category = task["category"]
    path     = task["path"]
    case     = load_json_with_fallback(path)
    case_id  = case.get("case_id", path.stem)
    queries  = case.get("queries", [])

    if not queries:
        return {"case_id": case_id, "category": category,
                "skipped": True, "reason": "no queries"}

    config   = get_config()
    save_dir = str(SAVE_BASE / f"{category[:12]}_{case_id[:16]}")

    mem = HippoRAGMemory(save_dir=save_dir)
    llm = OpenAIClient(
        api_key=config.llm["api_key"],
        base_url=config.llm["base_url"],
        model=config.llm["model"],
    )

    # Ingest
    ingest_case(mem, case, category)
    t0 = time.time()
    mem.build_index()
    ingest_time_ms = (time.time() - t0) * 1000
    audit = mem.audit_ingest()

    # Infer
    t1 = time.time()
    tokens_before = llm.total_tokens
    correct = 0
    qa_results = []

    for q in queries:
        question  = q.get("question", "")
        reference = q.get("reference_answer", {})
        pred = answer_with_memory(llm, mem, question)
        ok   = judge_answer(llm, question, pred, reference, category)
        if ok:
            correct += 1
        qa_results.append({
            "question": question[:80],
            "correct":  ok,
            "pred":     pred[:120],
        })

    infer_time_ms    = (time.time() - t1) * 1000
    infer_llm_tokens = llm.total_tokens - tokens_before

    mem.reset()

    return {
        "case_id":    case_id,
        "category":   category,
        "n_queries":  len(queries),
        "correct":    correct,
        # audit ingest
        "ingest_chunks":         audit["ingest_chunks"],
        "ingest_time_ms":        audit["ingest_time_ms"],
        "ingest_llm_calls":      audit["ingest_llm_calls"],
        "ingest_llm_prompt":     audit["ingest_llm_prompt_tokens"],
        "ingest_llm_completion": audit["ingest_llm_completion_tokens"],
        # audit infer
        "infer_time_ms":         round(infer_time_ms),
        "infer_emb_calls":       len(queries),
        "infer_llm_tokens":      infer_llm_tokens,
        "qa_results":            qa_results,
    }


# ── main ──────────────────────────────────────────────────────────────────

def main() -> int:
    SAVE_BASE.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("StructMemEval 全量评测 (HippoRAG, openie_mode=online, chunk_size=1000)")
    print("=" * 70)
    print("[Preflight] 检查 HippoRAG 运行依赖（源码路径直导入 + 关键运行依赖）...")
    try:
        ensure_hipporag_runtime_dependencies(verbose=True)
    except HippoRAGDependencyError as exc:
        print(f"[BENCH-STATUS] ok=False reason=preflight dependency check failed: {exc}")
        return 2

    tasks = collect_cases()
    from collections import Counter

    # ── Resume: 跳过已完成的 case ─────────────────────────────────────────
    prior_results, prior_errors = [], []
    if RESULTS_PATH.exists():
        prior = load_json_with_fallback(RESULTS_PATH)
        prior_results = prior.get("results", [])
        prior_errors  = prior.get("errors", [])
        done_keys = {(r["category"], r["case_id"]) for r in prior_results if not r.get("skipped")}
        tasks = [t for t in tasks
                 if (t["category"], t["path"].stem) not in done_keys]
        print(f"  Resume: 跳过 {len(done_keys)} 个已完成 case，剩余 {len(tasks)} 个\n")

    cat_cnt = Counter(t["category"] for t in tasks)
    for cat, cnt in sorted(cat_cnt.items()):
        print(f"  {cat}: {cnt} cases")
    print(f"  合计: {len(tasks)} cases | MAX_WORKERS={MAX_WORKERS}\n")

    results = list(prior_results)
    errors  = list(prior_errors)
    t_wall_start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(eval_case, t): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            done += 1
            task = futures[future]
            try:
                r = future.result()
                results.append(r)
                if r.get("skipped"):
                    tag    = "SKIP"
                    detail = r["reason"]
                else:
                    tag    = f"{r['correct']}/{r['n_queries']}"
                    detail = (
                        f"chunks={r['ingest_chunks']}  "
                        f"ingest={r['ingest_time_ms']}ms  infer={r['infer_time_ms']}ms  "
                        f"llm_calls={r['ingest_llm_calls']}"
                    )
                print(f"[{done:3d}/{len(tasks)}] {r['category'][:22]:22s} {r['case_id'][:28]:28s} {tag}  {detail}")
            except Exception as e:
                errors.append({"path": str(task["path"]), "error": str(e)})
                print(f"[{done:3d}/{len(tasks)}] ERROR {task['path'].name}: {e}")
                import traceback; traceback.print_exc()

    wall_ms = (time.time() - t_wall_start) * 1000

    # ── 汇总 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    agg = defaultdict(lambda: dict(correct=0, total=0, ingest_ms=0,
                                   infer_ms=0, chunks=0,
                                   ingest_llm_calls=0, ingest_prompt=0, ingest_compl=0,
                                   infer_emb=0, llm_tokens=0))
    for r in results:
        if r.get("skipped"):
            continue
        a = agg[r["category"]]
        a["correct"]       += r["correct"]
        a["total"]         += r["n_queries"]
        a["ingest_ms"]     += r["ingest_time_ms"]
        a["infer_ms"]      += r["infer_time_ms"]
        a["chunks"]        += r["ingest_chunks"]
        a["ingest_llm_calls"] += r["ingest_llm_calls"]
        a["ingest_prompt"] += r["ingest_llm_prompt"]
        a["ingest_compl"]  += r["ingest_llm_completion"]
        a["infer_emb"]     += r["infer_emb_calls"]
        a["llm_tokens"]    += r["infer_llm_tokens"]

    g = defaultdict(int)
    for cat, a in sorted(agg.items()):
        acc = a["correct"] / a["total"] * 100 if a["total"] else 0
        print(f"\n  [{cat}]  {a['correct']}/{a['total']}  ({acc:.1f}%)")
        print(f"    Ingest | time(serial): {a['ingest_ms']:.0f}ms | chunks: {a['chunks']}")
        print(f"           | llm_calls: {a['ingest_llm_calls']} | prompt: {a['ingest_prompt']} | compl: {a['ingest_compl']}")
        print(f"    Infer  | time(serial): {a['infer_ms']:.0f}ms | emb_calls: {a['infer_emb']} | llm_tokens: {a['llm_tokens']}")
        for k in a:
            g[k] += a[k]

    overall = g["correct"] / g["total"] * 100 if g["total"] else 0
    print("\n" + "─" * 70)
    print(f"  总准确率  : {g['correct']}/{g['total']} ({overall:.1f}%)")
    print(f"  Ingest    | time: {g['ingest_ms']:.0f}ms | chunks: {g['chunks']}")
    print(f"            | llm_calls: {g['ingest_llm_calls']} | prompt: {g['ingest_prompt']} | compl: {g['ingest_compl']}")
    print(f"  Infer     | time: {g['infer_ms']:.0f}ms | llm_tokens: {g['llm_tokens']}")
    print(f"  Wall      : {wall_ms:.0f}ms ({wall_ms/1000/60:.1f}min)")
    print(f"  Errors    : {len(errors)}")
    print("─" * 70)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": results, "errors": errors}, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {RESULTS_PATH}")
    health = evaluate_execution_health(results=results, errors=errors, total_tasks=len(tasks))
    print(
        "[BENCH-STATUS] "
        f"ok={health.ok} reason={health.reason} total={health.total_tasks} "
        f"completed={health.completed_cases} skipped={health.skipped_cases} "
        f"errors={health.error_cases} evaluated_queries={health.evaluated_queries}"
    )
    return 0 if health.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
