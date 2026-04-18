#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AMemGym 全量评测 - 并行版
20 个用户各自独立：ingest 全部 11 periods → 回答全部 10 QA
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from simpleMem_src import SimpleRAGMemory, get_config, OpenAIClient

DATA_PATH    = Path("data/amemgym/v1.base/data.json")
MAX_WORKERS  = 5    # 20 用户，5 并发
RESULTS_PATH = Path("results_amemgym_full.json")


# ── Ingest ───────────────────────────────────────────────────────────────

def ingest_all_periods(mem: SimpleRAGMemory, user: dict) -> int:
    ingested = 0
    for pi, period in enumerate(user.get("periods", [])):
        period_start = period.get("period_start", "?")
        period_end   = period.get("period_end", "?")
        for session in period.get("sessions", []):
            query        = session.get("query", "")
            exposed      = session.get("exposed_states", {})
            session_time = session.get("session_time", "")
            if not query:
                continue
            parts = [f"[Time: {session_time}, Period: {period_start} ~ {period_end}]",
                     f"User asked: {query}"]
            if exposed:
                parts.append("Known states: " + ", ".join(f"{k}={v}" for k, v in exposed.items()))
            mem.add_memory("\n".join(parts), {"period": pi, "time": session_time})
            ingested += 1
    return ingested


# ── Infer ─────────────────────────────────────────────────────────────────

def find_correct_index(qa: dict, user: dict) -> int:
    """以最后一个 period 的 state 作为 ground truth"""
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


def answer_multichoice(llm: OpenAIClient, mem: SimpleRAGMemory, qa: dict) -> int:
    question = qa["query"]
    choices  = qa["answer_choices"]
    evidences = mem.retrieve(question, top_k=5)
    context   = "\n".join(f"[Memory {i+1}] {e.content}" for i, e in enumerate(evidences))
    choices_text = "\n".join(f"  ({i}) {c['answer'][:150]}" for i, c in enumerate(choices))
    prompt = f"""Based on the memories below, select the best answer choice for the question.
Reply with ONLY the choice number (e.g., 0, 1, 2, ...).

## Memories
{context}

## Question
{question}

## Choices
{choices_text}

## Your choice (number only):"""
    result = llm.generate(prompt, temperature=0.0, max_tokens=10)
    for ch in result.strip():
        if ch.isdigit():
            idx = int(ch)
            if 0 <= idx < len(choices):
                return idx
    return 0


# ── 单用户评测 ────────────────────────────────────────────────────────────

def eval_user(user: dict) -> dict:
    user_id = user.get("id", "unknown")

    config = get_config()
    mem = SimpleRAGMemory(collection_name=f"amemgym_{user_id[:8]}")
    llm = OpenAIClient(
        api_key=config.llm["api_key"],
        base_url=config.llm["base_url"],
        model=config.llm["model"],
    )

    # Ingest
    t0 = time.time()
    n_entries = ingest_all_periods(mem, user)
    ingest_ms = (time.time() - t0) * 1000
    ingest_emb_calls = mem.size

    # Infer
    qas = user.get("qas", [])
    t1 = time.time()
    tokens_before = llm.total_tokens
    correct = 0
    qa_results = []

    for qa in qas:
        correct_idx = find_correct_index(qa, user)
        pred_idx    = answer_multichoice(llm, mem, qa)
        match       = pred_idx == correct_idx
        if match:
            correct += 1
        choices = qa["answer_choices"]
        qa_results.append({
            "question":    qa["query"][:80],
            "correct_idx": correct_idx,
            "pred_idx":    pred_idx,
            "match":       match,
            "gold_ans":    choices[correct_idx]["answer"][:60] if correct_idx < len(choices) else "?",
            "pred_ans":    choices[pred_idx]["answer"][:60]    if pred_idx    < len(choices) else "?",
        })

    infer_ms = (time.time() - t1) * 1000
    infer_llm_tokens = llm.total_tokens - tokens_before
    mem.reset()

    return {
        "user_id": user_id,
        "n_periods": len(user.get("periods", [])),
        "n_sessions": n_entries,
        "n_qa": len(qas),
        "correct": correct,
        "ingest_ms": ingest_ms,
        "ingest_emb_calls": ingest_emb_calls,
        "infer_ms": infer_ms,
        "infer_emb_calls": len(qas),
        "infer_llm_tokens": infer_llm_tokens,
        "qa_results": qa_results,
    }


# ── main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("AMemGym 全量评测 (SimpleMem, 并行)")
    print("=" * 70)

    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"总用户数: {len(data)} | MAX_WORKERS={MAX_WORKERS}")
    total_qa = sum(len(u.get("qas", [])) for u in data)
    print(f"总 QA 数: {total_qa}\n")

    results, errors = [], []
    t_wall_start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(eval_user, user): user for user in data}
        done = 0
        for future in as_completed(futures):
            done += 1
            user = futures[future]
            uid  = user.get("id", "?")[:12]
            try:
                r = future.result()
                results.append(r)
                acc = r["correct"] / r["n_qa"] * 100 if r["n_qa"] else 0
                print(f"[{done:2d}/{len(data)}] {uid:12s}"
                      f"  {r['correct']}/{r['n_qa']} ({acc:5.1f}%)"
                      f"  periods={r['n_periods']} sessions={r['n_sessions']}"
                      f"  ingest={r['ingest_ms']:.0f}ms infer={r['infer_ms']:.0f}ms"
                      f"  tok={r['infer_llm_tokens']}")
            except Exception as e:
                errors.append({"user_id": uid, "error": str(e)})
                print(f"[{done:2d}/{len(data)}] ERROR {uid}: {e}")

    wall_ms = (time.time() - t_wall_start) * 1000

    # ── 汇总 ──────────────────────────────────────────────────────────────
    valid = [r for r in results if "correct" in r]
    g_correct    = sum(r["correct"]          for r in valid)
    g_total      = sum(r["n_qa"]             for r in valid)
    g_ingest_ms  = sum(r["ingest_ms"]        for r in valid)
    g_infer_ms   = sum(r["infer_ms"]         for r in valid)
    g_ingest_emb = sum(r["ingest_emb_calls"] for r in valid)
    g_infer_emb  = sum(r["infer_emb_calls"]  for r in valid)
    g_llm_tokens = sum(r["infer_llm_tokens"] for r in valid)

    overall = g_correct / g_total * 100 if g_total else 0
    print("\n" + "─" * 70)
    print("── 总 Audit ──────────────────────────────────────────────────────")
    print(f"  总准确率  : {g_correct}/{g_total} ({overall:.1f}%)")
    print(f"  Ingest    | time(串行累计): {g_ingest_ms:.0f} ms | emb_calls: {g_ingest_emb} | llm_tokens: 0")
    print(f"  Infer     | time(串行累计): {g_infer_ms:.0f} ms  | emb_calls: {g_infer_emb}  | llm_tokens: {g_llm_tokens}")
    print(f"  Wall time : {wall_ms:.0f} ms  ({wall_ms/1000/60:.1f} min, 实际并行耗时)")
    print(f"  Errors    : {len(errors)}")
    print("─" * 70)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": results, "errors": errors}, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {RESULTS_PATH}")


if __name__ == "__main__":
    main()
