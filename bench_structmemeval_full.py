#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StructMemEval 全量评测 - 并行版
覆盖: state_machine_location (42) / tree_based (100) / recommendations (30)
跳过: accounting (无 query 字段)
"""

import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from simpleMem_src import SimpleRAGMemory, get_config, OpenAIClient
from benchmark_io_utils import load_json_with_fallback

# ── 数据路径 ──────────────────────────────────────────────────────────────
BASE = Path("StructMemEval/benchmark")
CATEGORIES = {
    "state_machine_location": BASE / "data" / "state_machine_location",
    "tree_based":             BASE / "tree_based" / "graph_configs",
    "recommendations":        BASE / "recommendations" / "data",
}

MAX_WORKERS = 8   # 并发 case 数，可根据 API 限速调整
RESULTS_PATH = Path("results_structmemeval_full.json")


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


def load_case(path: Path) -> dict:
    return load_json_with_fallback(path)


# ── Ingest ───────────────────────────────────────────────────────────────

def ingest_case(mem: SimpleRAGMemory, case: dict) -> int:
    ingested = 0
    for session in case.get("sessions", []):
        sid = session.get("session_id", "?")
        topic = session.get("topic", "")
        messages = session.get("messages", [])

        buf = []
        for msg in messages:
            buf.append(f"{msg['role']}: {msg['content']}")
            if msg["role"] == "assistant" and len(buf) >= 2:
                header = f"[Session: {sid}" + (f", Topic: {topic}" if topic else "") + "]"
                mem.add_memory(header + "\n" + "\n".join(buf),
                               {"session": sid, "topic": topic})
                ingested += 1
                buf = []
        if buf:
            header = f"[Session: {sid}" + (f", Topic: {topic}" if topic else "") + "]"
            mem.add_memory(header + "\n" + "\n".join(buf),
                           {"session": sid, "topic": topic})
            ingested += 1
    return ingested


# ── Infer & Judge ─────────────────────────────────────────────────────────

def answer_with_memory(llm: OpenAIClient, mem: SimpleRAGMemory, question: str) -> str:
    evidences = mem.retrieve(question, top_k=5)
    context = "\n\n".join(f"[Memory {i+1}] {e.content}" for i, e in enumerate(evidences))
    prompt = f"""Based on the conversation memories below, answer the question.
Be specific and concise.

## Memories
{context}

## Question
{question}

## Answer:"""
    return llm.generate(prompt, temperature=0.0, max_tokens=300)


def judge_answer(llm: OpenAIClient, question: str, pred: str,
                 reference: dict, category: str) -> bool:
    ref_text = reference.get("text", "")
    if category == "recommendations":
        criteria = reference.get("evaluation_criteria", [])
        criteria_str = "\n".join(f"- {c}" for c in criteria) if criteria else ""
        prompt = f"""You are a strict judge evaluating a memory recall answer.

Question: {question}
Reference answer: {ref_text}
{"Evaluation criteria:" + chr(10) + criteria_str if criteria_str else ""}
Predicted answer: {pred}

Does the predicted answer satisfy the criteria and semantically match the reference?
Answer ONLY "yes" or "no"."""
    else:
        prompt = f"""You are a judge. Does the predicted answer semantically match the reference?
Answer ONLY "yes" or "no".

Question: {question}
Reference: {ref_text}
Predicted: {pred}

Match (yes/no):"""
    result = llm.generate(prompt, temperature=0.0, max_tokens=10)
    return "yes" in result.lower()


# ── 单 case 评测 ──────────────────────────────────────────────────────────

def eval_case(task: dict) -> dict:
    category = task["category"]
    path = task["path"]
    case = load_case(path)
    case_id = case.get("case_id", path.stem)
    queries = case.get("queries", [])

    if not queries:
        return {"case_id": case_id, "category": category,
                "skipped": True, "reason": "no queries"}

    config = get_config()
    mem = SimpleRAGMemory(collection_name=f"{category}_{case_id}")
    llm = OpenAIClient(
        api_key=config.llm["api_key"],
        base_url=config.llm["base_url"],
        model=config.llm["model"],
    )

    # Ingest
    t0 = time.time()
    ingest_case(mem, case)
    ingest_ms = (time.time() - t0) * 1000
    ingest_emb_calls = mem.size

    # Infer
    t1 = time.time()
    tokens_before = llm.total_tokens
    correct = 0
    qa_results = []

    for q in queries:
        question = q.get("question", "")
        reference = q.get("reference_answer", {})
        pred = answer_with_memory(llm, mem, question)
        ok = judge_answer(llm, question, pred, reference, category)
        if ok:
            correct += 1
        qa_results.append({
            "question": question[:80],
            "correct": ok,
            "pred": pred[:120],
        })

    infer_ms = (time.time() - t1) * 1000
    infer_llm_tokens = llm.total_tokens - tokens_before
    mem.reset()

    return {
        "case_id": case_id,
        "category": category,
        "n_queries": len(queries),
        "correct": correct,
        "ingest_ms": ingest_ms,
        "ingest_emb_calls": ingest_emb_calls,
        "infer_ms": infer_ms,
        "infer_emb_calls": len(queries),
        "infer_llm_tokens": infer_llm_tokens,
        "qa_results": qa_results,
    }


# ── main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("StructMemEval 全量评测 (SimpleMem, 并行)")
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
                    tag = "SKIP"
                    detail = r["reason"]
                else:
                    tag = f"{r['correct']}/{r['n_queries']}"
                    detail = f"ingest={r['ingest_ms']:.0f}ms infer={r['infer_ms']:.0f}ms tok={r['infer_llm_tokens']}"
                print(f"[{done:3d}/{len(tasks)}] {r['category'][:22]:22s} {r['case_id'][:28]:28s} {tag}  {detail}")
            except Exception as e:
                errors.append({"path": str(task["path"]), "error": str(e)})
                print(f"[{done:3d}/{len(tasks)}] ERROR {task['path'].name}: {e}")

    wall_ms = (time.time() - t_wall_start) * 1000

    # ── 汇总 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    agg = defaultdict(lambda: dict(correct=0, total=0, ingest_ms=0,
                                   infer_ms=0, ingest_emb=0,
                                   infer_emb=0, llm_tokens=0))
    for r in results:
        if r.get("skipped"):
            continue
        a = agg[r["category"]]
        a["correct"]    += r["correct"]
        a["total"]      += r["n_queries"]
        a["ingest_ms"]  += r["ingest_ms"]
        a["infer_ms"]   += r["infer_ms"]
        a["ingest_emb"] += r["ingest_emb_calls"]
        a["infer_emb"]  += r["infer_emb_calls"]
        a["llm_tokens"] += r["infer_llm_tokens"]

    g = dict(correct=0, total=0, ingest_ms=0, infer_ms=0,
             ingest_emb=0, infer_emb=0, llm_tokens=0)
    for cat, a in sorted(agg.items()):
        acc = a["correct"] / a["total"] * 100 if a["total"] else 0
        print(f"\n  [{cat}]  {a['correct']}/{a['total']}  ({acc:.1f}%)")
        print(f"    Ingest | time(serial): {a['ingest_ms']:.0f} ms | emb_calls: {a['ingest_emb']} | llm_tokens: 0")
        print(f"    Infer  | time(serial): {a['infer_ms']:.0f} ms  | emb_calls: {a['infer_emb']}  | llm_tokens: {a['llm_tokens']}")
        for k in g:
            g[k] += a[k]

    overall = g["correct"] / g["total"] * 100 if g["total"] else 0
    print("\n" + "─" * 70)
    print("── 总 Audit ──────────────────────────────────────────────────────")
    print(f"  总准确率  : {g['correct']}/{g['total']} ({overall:.1f}%)")
    print(f"  Ingest    | time(串行累计): {g['ingest_ms']:.0f} ms | emb_calls: {g['ingest_emb']} | llm_tokens: 0")
    print(f"  Infer     | time(串行累计): {g['infer_ms']:.0f} ms  | emb_calls: {g['infer_emb']}  | llm_tokens: {g['llm_tokens']}")
    print(f"  Wall time : {wall_ms:.0f} ms  ({wall_ms/1000/60:.1f} min, 实际并行耗时)")
    print(f"  Errors    : {len(errors)}")
    print("─" * 70)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": results, "errors": errors}, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {RESULTS_PATH}")


if __name__ == "__main__":
    main()
