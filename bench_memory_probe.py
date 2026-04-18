#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SimpleMem × memory-probe (LoCoMo) 小规模验证
用 1 段对话的前 3 个 session，回答 5 个问题。
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from simpleMem_src import SimpleRAGMemory, get_config, OpenAIClient, get_logger

DATA_PATH = Path("memory-probe/data/locomo10.json")
MAX_SESSIONS = 3
MAX_QUESTIONS = 5

logger = get_logger()


def load_locomo():
    with open(DATA_PATH) as f:
        return json.load(f)


def ingest_sessions(mem: SimpleRAGMemory, conv: dict, max_sessions: int):
    """将对话 session 按 3-turn chunk 写入记忆"""
    conversation = conv["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]
    ingested = 0

    for i in range(1, max_sessions + 1):
        session_key = f"session_{i}"
        date_key = f"session_{i}_date_time"
        turns = conversation.get(session_key)
        date_str = conversation.get(date_key, "")

        if not turns or not isinstance(turns, list):
            continue

        # 3-turn chunking
        chunk_size = 3
        for start in range(0, len(turns), chunk_size):
            chunk_turns = turns[start : start + chunk_size]
            text_parts = []
            for t in chunk_turns:
                speaker = t.get("speaker", "?")
                text = t.get("text", "")
                text_parts.append(f"{speaker}: {text}")

            chunk_text = f"[{date_str}]\n" + "\n".join(text_parts)
            mem.add_memory(chunk_text, {
                "session": i,
                "date": date_str,
                "speakers": f"{speaker_a},{speaker_b}",
            })
            ingested += 1

    return ingested


def answer_with_memory(llm, mem: SimpleRAGMemory, question: str, top_k: int = 5) -> str:
    """检索记忆 + LLM 生成答案"""
    evidences = mem.retrieve(question, top_k=top_k)
    context = "\n\n".join(
        f"[Memory {i+1}] {e.content}" for i, e in enumerate(evidences)
    )

    prompt = f"""Based on the following conversation memories, answer the question concisely.
If the answer is not found in the memories, say "I don't know".

## Memories
{context}

## Question
{question}

## Answer (be concise, just the key fact):"""

    return llm.generate(prompt, temperature=0.0, max_tokens=200)


def main():
    print("=" * 60)
    print("SimpleMem × memory-probe (LoCoMo) 验证")
    print("=" * 60)

    data = load_locomo()
    conv = data[0]  # 第一段对话
    speakers = f"{conv['conversation']['speaker_a']} & {conv['conversation']['speaker_b']}"
    print(f"对话: {speakers}")
    print(f"总 QA 数: {len(conv['qa'])}")

    # 初始化
    config = get_config()
    mem = SimpleRAGMemory(collection_name="locomo_test")
    llm = OpenAIClient(
        api_key=config.llm["api_key"],
        base_url=config.llm["base_url"],
        model=config.llm["model"],
    )

    # ── Ingest ──────────────────────────────────────────────
    t_ingest_start = time.time()
    llm_tokens_before_ingest = llm.total_tokens
    n = ingest_sessions(mem, conv, MAX_SESSIONS)
    ingest_ms = (time.time() - t_ingest_start) * 1000
    ingest_llm_tokens = llm.total_tokens - llm_tokens_before_ingest
    ingest_emb_calls = mem.size  # 每次 add_memory = 1 次 embedding 调用

    print(f"写入记忆: {n} chunks (前 {MAX_SESSIONS} 个 sessions)")

    # ── Infer ───────────────────────────────────────────────
    qas = conv["qa"][:MAX_QUESTIONS]
    correct = 0
    total = len(qas)

    print(f"\n--- 回答 {total} 个问题 ---\n")
    t_infer_start = time.time()
    llm_tokens_before_infer = llm.total_tokens

    for i, qa in enumerate(qas):
        question = qa["question"]
        gold = qa["answer"]
        category = qa.get("category", "?")

        pred = answer_with_memory(llm, mem, question)
        # 简单的包含匹配
        gold_lower = str(gold).lower().strip()
        pred_lower = str(pred).lower().strip()
        match = gold_lower in pred_lower or pred_lower in gold_lower

        if match:
            correct += 1
            status = "✓"
        else:
            status = "✗"

        print(f"  [{status}] Q{i+1} (cat={category}): {question}")
        print(f"       Gold: {gold}")
        print(f"       Pred: {pred.strip()[:100]}")
        print()

    infer_ms = (time.time() - t_infer_start) * 1000
    infer_llm_tokens = llm.total_tokens - llm_tokens_before_infer
    infer_emb_calls = total  # 每个问题 retrieve 1 次 embedding

    # ── 结果 & Audit ────────────────────────────────────────
    print(f"结果: {correct}/{total} ({correct/total*100:.0f}%)")
    print()
    print("── Audit ──────────────────────────────────────")
    print(f"  Ingest | time: {ingest_ms:.0f} ms | emb_calls: {ingest_emb_calls} | llm_tokens: {ingest_llm_tokens}")
    print(f"  Infer  | time: {infer_ms:.0f} ms | emb_calls: {infer_emb_calls} | llm_tokens: {infer_llm_tokens}")
    print(f"  Total  | time: {ingest_ms + infer_ms:.0f} ms | llm_tokens: {llm.total_tokens}")
    print("────────────────────────────────────────────────")

    mem.reset()


if __name__ == "__main__":
    main()
