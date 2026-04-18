#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""memory-probe 全量评测 - LightRAG 版
LoCoMo-10: 10 段对话，每段 ~199 QA（总计 ~1461 QA），category 1-5
LightRAG: tree=3L, top_k=5
"""

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from simpleMem_src import get_config, OpenAIClient
from lightrag_bench_src import LightRAGBenchMemory

DATA_PATH    = Path("memory-probe/data/locomo10.json")
MAX_WORKERS  = 1
RESULTS_PATH = Path("results_memory_probe_lightrag.json")
SAVE_BASE    = Path("/tmp/bench_lightrag_probe")


def ingest_conversation(mem, conv):
    conversation = conv["conversation"]
    session_keys = sorted(
        [k for k in conversation.keys() if re.match(r"session_\d+$", k)],
        key=lambda x: int(x.split("_")[1])
    )
    for sk in session_keys:
        date = conversation.get(sk + "_date_time", "")
        turns = conversation[sk]
        header = f"[{date}]" if date else ""
        body = "\n".join(f"{t.get('speaker','')}: {t.get('text','')}" for t in turns)
        mem.add_memory((header + "\n" + body).strip())


def answer_with_memory(llm, mem, question):
    evidences = mem.retrieve(question, top_k=5)
    ctx = "\n\n".join(f"[{i+1}] {e.content}" for i, e in enumerate(evidences))
    prompt = (
        "Based on the conversation memories below, answer the question concisely.\n\n"
        f"## Memories\n{ctx}\n\n## Question\n{question}\n\n## Answer:"
    )
    return llm.generate(prompt, temperature=0.0, max_tokens=200)


def judge_answer(llm, question, pred, reference):
    prompt = (
        "Does the predicted answer semantically match the reference?\n"
        "Answer ONLY 'yes' or 'no'.\n\n"
        f"Question: {question}\nReference: {reference}\nPredicted: {pred}\n\nMatch:"
    )
    r = llm.generate(prompt, temperature=0.0, max_tokens=10)
    return "yes" in r.lower()


def eval_conv(conv, conv_idx):
    config   = get_config()
    save_dir = str(SAVE_BASE / f"conv_{conv_idx:02d}")

    mem = LightRAGBenchMemory(save_dir=save_dir)
    llm = OpenAIClient(
        api_key=config.llm["api_key"],
        base_url=config.llm["base_url"],
        model=config.llm["model"],
    )

    ingest_conversation(mem, conv)
    t0 = time.time()
    mem.build_index()
    ingest_time_ms = (time.time() - t0) * 1000
    audit = mem.audit_ingest()

    qas = conv.get("qa", [])
    t1 = time.time()
    tokens_before = llm.total_tokens
    correct = 0
    qa_results = []

    for qa in qas:
        question  = qa.get("question", "")
        reference = str(qa.get("answer", ""))
        category  = qa.get("category", 0)

        pred = answer_with_memory(llm, mem, question)
        ok   = judge_answer(llm, question, pred, reference)
        if ok:
            correct += 1
        qa_results.append({
            "question": question[:80],
            "category": category,
            "correct":  ok,
            "pred":     pred[:120],
            "reference": reference[:80],
        })

    infer_time_ms    = (time.time() - t1) * 1000
    infer_llm_tokens = llm.total_tokens - tokens_before

    mem.reset()

    return {
        "conv_id":    conv.get("sample_id", conv_idx),
        "n_queries":  len(qas),
        "correct":    correct,
        "ingest_chunks":         audit["ingest_chunks"],
        "ingest_time_ms":        audit["ingest_time_ms"],
        "ingest_llm_calls":      audit["ingest_llm_calls"],
        "ingest_llm_prompt":     audit["ingest_llm_prompt_tokens"],
        "ingest_llm_completion": audit["ingest_llm_completion_tokens"],
        "infer_time_ms":         round(infer_time_ms),
        "infer_llm_tokens":      infer_llm_tokens,
        "qa_results":            qa_results,
    }


def main():
    SAVE_BASE.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("memory-probe 全量评测 (LightRAG, tree=3L, top_k=5)")
    print("=" * 70)

    data  = json.load(open(DATA_PATH))
    convs = data if isinstance(data, list) else data.get("conversations", [data])
    print(f"  对话数: {len(convs)} | MAX_WORKERS={MAX_WORKERS}\n")

    results, errors = [], []
    t_wall_start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(eval_conv, c, i): i for i, c in enumerate(convs)}
        done = 0
        for future in as_completed(futures):
            done += 1
            conv_idx = futures[future]
            try:
                r = future.result()
                results.append(r)
                print(
                    f"[{done:2d}/{len(convs)}] conv={conv_idx}  "
                    f"{r['correct']}/{r['n_queries']}  "
                    f"chunks={r['ingest_chunks']}  "
                                        f"ingest={r['ingest_time_ms']}ms  infer={r['infer_time_ms']}ms"
                )
            except Exception as e:
                errors.append({"conv_idx": conv_idx, "error": str(e)})
                print(f"[{done:2d}/{len(convs)}] ERROR conv={conv_idx}: {e}")
                import traceback; traceback.print_exc()

    wall_ms = (time.time() - t_wall_start) * 1000

    print("\n" + "=" * 70)
    cat_agg = defaultdict(lambda: dict(correct=0, total=0))
    for r in results:
        for qa in r.get("qa_results", []):
            cat = qa.get("category", "unknown")
            cat_agg[cat]["correct"] += int(qa["correct"])
            cat_agg[cat]["total"]   += 1

    for cat in sorted(cat_agg.keys()):
        a = cat_agg[cat]
        acc = a["correct"] / a["total"] * 100 if a["total"] else 0
        print(f"  cat={cat}: {a['correct']}/{a['total']} ({acc:.1f}%)")

    total_q = sum(r["n_queries"] for r in results)
    total_c = sum(r["correct"] for r in results)
    acc = total_c / total_q * 100 if total_q else 0

    print(f"\n  总准确率  : {total_c}/{total_q} ({acc:.1f}%)")
    print(f"  Wall   : {wall_ms:.0f}ms ({wall_ms/1000/60:.1f}min)")
    print(f"  Errors : {len(errors)}")
    print("=" * 70)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": results, "errors": errors}, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {RESULTS_PATH}")


if __name__ == "__main__":
    main()
