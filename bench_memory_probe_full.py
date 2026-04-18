#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""memory-probe (LoCoMo) 全量评测 - 并行版
10 段对话各自独立，每段对话 ingest 全部 session → 回答全部 QA
"""

import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from simpleMem_src import SimpleRAGMemory, get_config, OpenAIClient

DATA_PATH   = Path("memory-probe/data/locomo10.json")
MAX_WORKERS = 5    # 10 对话，5 并发
CHUNK_SIZE  = 3    # turn 分块粒度
RESULTS_PATH = Path("results_memory_probe_full.json")


# ── Ingest ───────────────────────────────────────────────────────────────

def ingest_all_sessions(mem: SimpleRAGMemory, conv: dict) -> int:
    conversation = conv["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]
    ingested = 0
    i = 1
    while True:
        session_key = f"session_{i}"
        date_key    = f"session_{i}_date_time"
        turns = conversation.get(session_key)
        if not turns or not isinstance(turns, list):
            break
        date_str = conversation.get(date_key, "")
        for start in range(0, len(turns), CHUNK_SIZE):
            chunk = turns[start: start + CHUNK_SIZE]
            parts = [f"{t.get('speaker', '?')}: {t.get('text', '')}" for t in chunk]
            text = f"[{date_str}]\n" + "\n".join(parts)
            mem.add_memory(text, {"session": i, "date": date_str,
                                   "speakers": f"{speaker_a},{speaker_b}"})
            ingested += 1
        i += 1
    return ingested


# ── Infer ─────────────────────────────────────────────────────────────────

def answer_with_memory(llm: OpenAIClient, mem: SimpleRAGMemory, question: str) -> str:
    evidences = mem.retrieve(question, top_k=5)
    context = "\n\n".join(f"[Memory {i+1}] {e.content}" for i, e in enumerate(evidences))
    prompt = f"""Based on the following conversation memories, answer the question concisely.
If the answer is not found in the memories, say "I don't know".

## Memories
{context}

## Question
{question}

## Answer (be concise, just the key fact):"""
    return llm.generate(prompt, temperature=0.0, max_tokens=200)


# ── 单对话评测 ────────────────────────────────────────────────────────────

def eval_conversation(idx: int, conv: dict) -> dict:
    conversation = conv["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]
    conv_label = f"{speaker_a}&{speaker_b}"

    config = get_config()
    mem = SimpleRAGMemory(collection_name=f"locomo_conv{idx}")
    llm = OpenAIClient(
        api_key=config.llm["api_key"],
        base_url=config.llm["base_url"],
        model=config.llm["model"],
    )

    # Ingest
    t0 = time.time()
    n_chunks = ingest_all_sessions(mem, conv)
    ingest_ms = (time.time() - t0) * 1000
    ingest_emb_calls = mem.size

    # Infer（跳过 category=5 对抗样本，无 answer 字段）
    qas = [qa for qa in conv["qa"] if "answer" in qa]
    t1 = time.time()
    tokens_before = llm.total_tokens
    correct = 0
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    qa_results = []

    for qa in qas:
        question = qa["question"]
        gold     = str(qa["answer"]).lower().strip()
        category = qa.get("category", "?")

        pred = answer_with_memory(llm, mem, question)
        pred_l = pred.lower().strip()
        match = gold in pred_l or pred_l in gold

        cat_stats[category]["total"]  += 1
        if match:
            correct += 1
            cat_stats[category]["correct"] += 1

        qa_results.append({
            "question": question[:80],
            "gold": gold[:60],
            "pred": pred.strip()[:80],
            "correct": match,
            "category": category,
        })

    infer_ms = (time.time() - t1) * 1000
    infer_llm_tokens = llm.total_tokens - tokens_before
    mem.reset()

    return {
        "conv_idx": idx,
        "conv_label": conv_label,
        "n_qa": len(qas),
        "correct": correct,
        "cat_stats": dict(cat_stats),
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
    print("memory-probe (LoCoMo) 全量评测 (SimpleMem, 并行)")
    print("=" * 70)

    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"总对话数: {len(data)} | MAX_WORKERS={MAX_WORKERS}")
    total_qa = sum(len(c["qa"]) for c in data)
    print(f"总 QA 数: {total_qa}\n")

    results, errors = [], []
    t_wall_start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(eval_conversation, i, conv): i
                   for i, conv in enumerate(data)}
        done = 0
        for future in as_completed(futures):
            done += 1
            i = futures[future]
            try:
                r = future.result()
                results.append(r)
                acc = r["correct"] / r["n_qa"] * 100 if r["n_qa"] else 0
                print(f"[{done:2d}/{len(data)}] conv{r['conv_idx']:02d} {r['conv_label'][:30]:30s}"
                      f"  {r['correct']:3d}/{r['n_qa']:3d} ({acc:5.1f}%)"
                      f"  ingest={r['ingest_ms']:.0f}ms infer={r['infer_ms']:.0f}ms"
                      f"  tok={r['infer_llm_tokens']}")
            except Exception as e:
                errors.append({"conv_idx": i, "error": str(e)})
                print(f"[{done:2d}/{len(data)}] ERROR conv{i}: {e}")

    wall_ms = (time.time() - t_wall_start) * 1000
    results.sort(key=lambda r: r.get("conv_idx", 0))

    # ── 汇总 ──────────────────────────────────────────────────────────────
    valid = [r for r in results if "correct" in r]
    g_correct      = sum(r["correct"]          for r in valid)
    g_total        = sum(r["n_qa"]             for r in valid)
    g_ingest_ms    = sum(r["ingest_ms"]        for r in valid)
    g_infer_ms     = sum(r["infer_ms"]         for r in valid)
    g_ingest_emb   = sum(r["ingest_emb_calls"] for r in valid)
    g_infer_emb    = sum(r["infer_emb_calls"]  for r in valid)
    g_llm_tokens   = sum(r["infer_llm_tokens"] for r in valid)

    # 按 category 汇总
    cat_agg = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in valid:
        for cat, s in r["cat_stats"].items():
            cat_agg[cat]["correct"] += s["correct"]
            cat_agg[cat]["total"]   += s["total"]

    overall = g_correct / g_total * 100 if g_total else 0
    print("\n" + "─" * 70)
    print("── 按 Category 准确率 ────────────────────────────────────────────")
    for cat in sorted(cat_agg):
        s = cat_agg[cat]
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"  cat={cat}: {s['correct']}/{s['total']} ({acc:.1f}%)")

    print("\n── 总 Audit ──────────────────────────────────────────────────────")
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
