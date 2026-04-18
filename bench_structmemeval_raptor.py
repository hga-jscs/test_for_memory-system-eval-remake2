#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StructMemEval 全量评测 - RAPTOR 版
覆盖: state_machine_location (42) / tree_based (100) / recommendations (30)
RAPTOR: tree=3L, top_k=10
"""

import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from simpleMem_src import get_config, OpenAIClient
from raptor_bench_src import RaptorBenchMemory
from benchmark_io_utils import load_json_with_fallback

BASE = Path("StructMemEval/benchmark")
CATEGORIES = {
    "state_machine_location": BASE / "data" / "state_machine_location",
    "tree_based":             BASE / "tree_based" / "graph_configs",
    "recommendations":        BASE / "recommendations" / "data",
}

MAX_WORKERS  = 1
RESULTS_PATH = Path("results_structmemeval_raptor.json")
SAVE_BASE    = Path("/tmp/bench_raptor_sme")


def collect_cases():
    tasks = []
    for cat, cat_dir in CATEGORIES.items():
        if not cat_dir.exists():
            print(f"[WARN] 路径不存在: {cat_dir}")
            continue
        for fp in sorted(cat_dir.rglob("*.json")):
            tasks.append({"category": cat, "path": fp})
    return tasks


def ingest_case(mem, case, category):
    for session in case.get("sessions", []):
        sid   = session.get("session_id", "?")
        topic = session.get("topic", "")
        msgs  = session.get("messages", [])
        header = f"[Session: {sid}" + (f", Topic: {topic}" if topic else "") + "]"

        if category == "tree_based":
            body = "\n".join(m["content"] for m in msgs)
        else:
            body = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)

        mem.add_text(header + "\n" + body)


def answer_with_memory(llm, mem, question):
    evidences = mem.retrieve(question, top_k=10)
    ctx = "\n\n".join(f"[{i+1}] {e.content}" for i, e in enumerate(evidences))
    prompt = (
        "Based on the conversation memories below, answer the question concisely.\n\n"
        f"## Memories\n{ctx}\n\n## Question\n{question}\n\n## Answer:"
    )
    return llm.generate(prompt, temperature=0.0, max_tokens=300)


def judge_answer(llm, question, pred, reference, category):
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


def eval_case(task):
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

    mem = RaptorBenchMemory(save_dir=save_dir)
    llm = OpenAIClient(
        api_key=config.llm["api_key"],
        base_url=config.llm["base_url"],
        model=config.llm["model"],
    )

    ingest_case(mem, case, category)
    t0 = time.time()
    mem.build_index()
    ingest_time_ms = (time.time() - t0) * 1000
    audit = mem.audit_ingest()

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
        "ingest_chunks":         audit["ingest_chunks"],
        "ingest_time_ms":        audit["ingest_time_ms"],
        "ingest_llm_calls":      audit["ingest_llm_calls"],
        "ingest_llm_prompt":     audit["ingest_llm_prompt_tokens"],
        "ingest_llm_completion": audit["ingest_llm_completion_tokens"],
        "tree_nodes":            audit["tree_nodes"],
        "tree_layers":           audit["tree_layers"],
        "infer_time_ms":         round(infer_time_ms),
        "infer_llm_tokens":      infer_llm_tokens,
        "qa_results":            qa_results,
    }


def main():
    SAVE_BASE.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("StructMemEval 全量评测 (RAPTOR, tree=3L, top_k=10)")
    print("=" * 70)

    tasks = collect_cases()
    from collections import Counter
    cat_cnt = Counter(t["category"] for t in tasks)
    for cat, cnt in sorted(cat_cnt.items()):
        print(f"  {cat}: {cnt} cases")
    print(f"  合计: {len(tasks)} cases | MAX_WORKERS={MAX_WORKERS}\n")

    results, errors = [], []
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
                        f"tree={r['tree_nodes']}nodes/{r['tree_layers']}L  "
                        f"ingest={r['ingest_time_ms']}ms  infer={r['infer_time_ms']}ms  "
                        f"llm_calls={r['ingest_llm_calls']}"
                    )
                print(f"[{done:3d}/{len(tasks)}] {r['category'][:22]:22s} {r['case_id'][:28]:28s} {tag}  {detail}")
            except Exception as e:
                errors.append({"path": str(task["path"]), "error": str(e)})
                print(f"[{done:3d}/{len(tasks)}] ERROR {task['path'].name}: {e}")
                import traceback; traceback.print_exc()

    wall_ms = (time.time() - t_wall_start) * 1000

    print("\n" + "=" * 70)
    agg = defaultdict(lambda: dict(correct=0, total=0, ingest_ms=0,
                                   infer_ms=0, chunks=0, llm_calls=0,
                                   tree_nodes=0, tree_layers=0))
    for r in results:
        if r.get("skipped"):
            continue
        a = agg[r["category"]]
        a["correct"]     += r["correct"]
        a["total"]       += r["n_queries"]
        a["ingest_ms"]   += r["ingest_time_ms"]
        a["infer_ms"]    += r["infer_time_ms"]
        a["chunks"]      += r["ingest_chunks"]
        a["llm_calls"]   += r["ingest_llm_calls"]
        a["tree_nodes"]  += r["tree_nodes"]
        a["tree_layers"]  = max(a["tree_layers"], r["tree_layers"])

    g = defaultdict(int)
    for cat, a in sorted(agg.items()):
        acc = a["correct"] / a["total"] * 100 if a["total"] else 0
        print(f"\n  [{cat}]  {a['correct']}/{a['total']}  ({acc:.1f}%)")
        print(f"    Ingest | time: {a['ingest_ms']:.0f}ms | chunks: {a['chunks']} | llm_calls: {a['llm_calls']}")
        print(f"           | tree_nodes: {a['tree_nodes']} | tree_layers: {a['tree_layers']}")
        print(f"    Infer  | time: {a['infer_ms']:.0f}ms")
        for k in a:
            if k != "tree_layers":
                g[k] += a[k]
            else:
                g[k] = max(g[k], a[k])

    overall = g["correct"] / g["total"] * 100 if g["total"] else 0
    print("\n" + "\u2500" * 70)
    print(f"  总准确率  : {g['correct']}/{g['total']} ({overall:.1f}%)")
    print(f"  Wall      : {wall_ms:.0f}ms ({wall_ms/1000/60:.1f}min)")
    print(f"  Errors    : {len(errors)}")
    print("\u2500" * 70)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": results, "errors": errors}, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {RESULTS_PATH}")


if __name__ == "__main__":
    main()
