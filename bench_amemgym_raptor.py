#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AMemGym 全量评测 - RAPTOR 版
20 个用户各自独立：ingest 全部 11 periods → 回答全部 10 QA（5选1）
RAPTOR: tree=3L, top_k=10
"""

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from simpleMem_src import get_config, OpenAIClient
from raptor_bench_src import RaptorBenchMemory
from benchmark_io_utils import load_json_with_fallback

DATA_PATH    = Path("data/amemgym/v1.base/data.json")
MAX_WORKERS  = 1
RESULTS_PATH = Path("results_amemgym_raptor.json")
SAVE_BASE    = Path("/tmp/bench_raptor_amemgym")


def ingest_all_periods(mem, user):
    for period in user.get("periods", []):
        period_start = period.get("period_start", "?")
        period_end   = period.get("period_end", "?")
        for session in period.get("sessions", []):
            query        = session.get("query", "")
            exposed      = session.get("exposed_states", {})
            session_time = session.get("session_time", "")
            if not query:
                continue
            parts = [f"[Time: {session_time}, Period: {period_start}~{period_end}]",
                     f"User query: {query}"]
            if exposed:
                parts.append("States: " + ", ".join(f"{k}={v}" for k, v in exposed.items()))
            mem.add_memory("\n".join(parts))


def find_correct_index(qa, user):
    periods = user.get("periods", [])
    if not periods:
        return 0
    current_state = periods[-1].get("state", {})
    required_info = qa.get("required_info", [])
    correct_state = [current_state.get(k, "") for k in required_info]
    for i, choice in enumerate(qa.get("answer_choices", [])):
        if choice.get("state") == correct_state:
            return i
    return 0


def answer_multichoice(llm, mem, qa):
    question = qa["query"]
    choices  = qa.get("answer_choices", [])
    evidences = mem.retrieve(question, top_k=10)
    ctx = "\n\n".join(f"[{i+1}] {e.content}" for i, e in enumerate(evidences))
    choices_str = "\n".join(f"  ({i}) {c.get('text','')}" for i, c in enumerate(choices))
    prompt = (
        f"Memory context:\n{ctx}\n\n"
        f"Question: {question}\n\n"
        f"Choices:\n{choices_str}\n\n"
        "Reply with ONLY the choice number (e.g., 0, 1, 2, ...):"
    )
    result = llm.generate(prompt, temperature=0.0, max_tokens=10)
    m = re.search(r'\d+', result)
    if m:
        idx = int(m.group())
        if 0 <= idx < len(choices):
            return idx
    return 0


def eval_user(user, uid):
    config  = get_config()
    save_dir = str(SAVE_BASE / f"user_{uid:03d}")

    mem = RaptorBenchMemory(save_dir=save_dir)
    llm = OpenAIClient(
        api_key=config.llm["api_key"],
        base_url=config.llm["base_url"],
        model=config.llm["model"],
    )

    ingest_all_periods(mem, user)
    t0 = time.time()
    mem.build_index()
    ingest_time_ms = (time.time() - t0) * 1000
    audit = mem.audit_ingest()

    qas = user.get("qas", [])
    t1 = time.time()
    tokens_before = llm.total_tokens
    correct = 0
    qa_results = []

    for qa in qas:
        correct_idx = find_correct_index(qa, user)
        pred_idx    = answer_multichoice(llm, mem, qa)
        ok = (pred_idx == correct_idx)
        if ok:
            correct += 1
        qa_results.append({
            "question": qa.get("query", "")[:80],
            "correct": ok,
            "pred_idx": pred_idx,
            "correct_idx": correct_idx,
        })

    infer_time_ms    = (time.time() - t1) * 1000
    infer_llm_tokens = llm.total_tokens - tokens_before

    mem.reset()

    return {
        "user_id":    user.get("id", str(uid)),
        "n_queries":  len(qas),
        "correct":    correct,
        "ingest_chunks":           audit["ingest_chunks"],
        "ingest_time_ms":          audit["ingest_time_ms"],
        "ingest_llm_calls":        audit["ingest_llm_calls"],
        "ingest_llm_prompt":       audit["ingest_llm_prompt_tokens"],
        "ingest_llm_completion":   audit["ingest_llm_completion_tokens"],
        "tree_nodes":              audit["tree_nodes"],
        "tree_layers":             audit["tree_layers"],
        "infer_time_ms":           round(infer_time_ms),
        "infer_llm_tokens":        infer_llm_tokens,
        "qa_results":              qa_results,
    }


def main():
    SAVE_BASE.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AMemGym 全量评测 (RAPTOR, tree=3L, top_k=10)")
    print("=" * 70)

    data = load_json_with_fallback(DATA_PATH)
    users = data if isinstance(data, list) else list(data.values())
    print(f"  用户数: {len(users)} | MAX_WORKERS={MAX_WORKERS}\n")

    results, errors = [], []
    t_wall_start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(eval_user, u, i): (i, u) for i, u in enumerate(users)}
        done = 0
        for future in as_completed(futures):
            done += 1
            uid, user = futures[future]
            try:
                r = future.result()
                results.append(r)
                print(
                    f"[{done:2d}/{len(users)}] user={uid:2d}  "
                    f"{r['correct']}/{r['n_queries']}  "
                    f"chunks={r['ingest_chunks']}  "
                    f"ingest={r['ingest_time_ms']}ms  infer={r['infer_time_ms']}ms  "
                    f"llm_calls={r['ingest_llm_calls']}  "
                    f"tree={r['tree_nodes']}nodes/{r['tree_layers']}L"
                )
            except Exception as e:
                errors.append({"user_id": uid, "error": str(e)})
                print(f"[{done:2d}/{len(users)}] ERROR user={uid}: {e}")
                import traceback; traceback.print_exc()

    wall_ms = (time.time() - t_wall_start) * 1000

    print("\n" + "=" * 70)
    total_q = sum(r["n_queries"] for r in results)
    total_c = sum(r["correct"] for r in results)
    acc = total_c / total_q * 100 if total_q else 0
    total_ingest_ms = sum(r["ingest_time_ms"] for r in results)
    total_infer_ms  = sum(r["infer_time_ms"] for r in results)

    print(f"\n  总准确率  : {total_c}/{total_q} ({acc:.1f}%)")
    print(f"  Ingest | time: {total_ingest_ms:.0f}ms | chunks: {sum(r['ingest_chunks'] for r in results)}")
    print(f"         | llm_calls: {sum(r['ingest_llm_calls'] for r in results)}")
    print(f"         | tree_nodes: {sum(r['tree_nodes'] for r in results)} | tree_layers: {max((r['tree_layers'] for r in results), default=0)}")
    print(f"  Infer  | time: {total_infer_ms:.0f}ms")
    print(f"  Wall   : {wall_ms:.0f}ms ({wall_ms/1000/60:.1f}min)")
    print(f"  Errors : {len(errors)}")
    print("=" * 70)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": results, "errors": errors}, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {RESULTS_PATH}")


if __name__ == "__main__":
    main()
