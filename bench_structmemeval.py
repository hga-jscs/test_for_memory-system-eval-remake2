#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SimpleMem × StructMemEval 小规模验证
用 2 个 static case 测试：加载对话 → 检索记忆 → 回答问题 → LLM judge。
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from simpleMem_src import SimpleRAGMemory, get_config, OpenAIClient, get_logger
from benchmark_io_utils import load_json_with_fallback

DATA_DIR = Path("StructMemEval/benchmark/data/state_machine_location")
CASES = ["static_001.json", "static_002.json"]

logger = get_logger()


def load_case(filename: str) -> dict:
    return load_json_with_fallback(DATA_DIR / filename)


def ingest_case(mem: SimpleRAGMemory, case: dict):
    """将 case 的所有 session messages 写入记忆"""
    ingested = 0
    for session in case["sessions"]:
        session_id = session.get("session_id", "?")
        topic = session.get("topic", "")
        messages = session.get("messages", [])

        # 按 user-assistant 对分组
        buf = []
        for msg in messages:
            buf.append(f"{msg['role']}: {msg['content']}")
            if msg["role"] == "assistant" and len(buf) >= 2:
                text = f"[Session: {session_id}, Topic: {topic}]\n" + "\n".join(buf)
                mem.add_memory(text, {"session": session_id, "topic": topic})
                ingested += 1
                buf = []

        # 剩余
        if buf:
            text = f"[Session: {session_id}, Topic: {topic}]\n" + "\n".join(buf)
            mem.add_memory(text, {"session": session_id, "topic": topic})
            ingested += 1

    return ingested


def answer_with_memory(llm, mem: SimpleRAGMemory, question: str) -> str:
    evidences = mem.retrieve(question, top_k=5)
    context = "\n\n".join(
        f"[Memory {i+1}] {e.content}" for i, e in enumerate(evidences)
    )

    prompt = f"""Based on the conversation memories below, answer the question.
Be specific and concise. Answer with the exact information from the memories.

## Memories
{context}

## Question
{question}

## Answer:"""

    return llm.generate(prompt, temperature=0.0, max_tokens=200)


def judge_answer(llm, question: str, pred: str, reference: str) -> bool:
    """LLM judge: 判断预测答案是否语义匹配参考答案"""
    prompt = f"""You are a judge. Determine if the predicted answer semantically matches the reference answer.
Answer ONLY "yes" or "no".

Question: {question}
Reference answer: {reference}
Predicted answer: {pred}

Match (yes/no):"""

    result = llm.generate(prompt, temperature=0.0, max_tokens=10)
    return "yes" in result.lower()


def main():
    print("=" * 60)
    print("SimpleMem × StructMemEval 验证")
    print("=" * 60)

    config = get_config()
    llm = OpenAIClient(
        api_key=config.llm["api_key"],
        base_url=config.llm["base_url"],
        model=config.llm["model"],
    )

    total_correct = 0
    total_questions = 0
    total_ingest_ms = 0.0
    total_ingest_emb_calls = 0
    total_ingest_llm_tokens = 0
    total_infer_ms = 0.0
    total_infer_emb_calls = 0
    total_infer_llm_tokens = 0

    for case_file in CASES:
        case = load_case(case_file)
        case_id = case.get("case_id", case_file)
        print(f"\n--- Case: {case_id} ---")

        mem = SimpleRAGMemory(collection_name=f"struct_{case_id}")

        # ── Ingest ──────────────────────────────────────────
        t_ingest_start = time.time()
        llm_tokens_before_ingest = llm.total_tokens
        n = ingest_case(mem, case)
        ingest_ms = (time.time() - t_ingest_start) * 1000
        ingest_llm_tokens = llm.total_tokens - llm_tokens_before_ingest
        ingest_emb_calls = mem.size

        print(f"写入记忆: {n} chunks ({len(case['sessions'])} sessions)")

        # ── Infer ────────────────────────────────────────────
        n_q = len(case["queries"])
        t_infer_start = time.time()
        llm_tokens_before_infer = llm.total_tokens

        for qi, q in enumerate(case["queries"]):
            question = q["question"]
            reference = q["reference_answer"]["text"]

            pred = answer_with_memory(llm, mem, question)
            match = judge_answer(llm, question, pred, reference)

            if match:
                total_correct += 1
                status = "✓"
            else:
                status = "✗"
            total_questions += 1

            print(f"  [{status}] {question}")
            print(f"       Ref:  {reference[:80]}")
            print(f"       Pred: {pred.strip()[:80]}")

        infer_ms = (time.time() - t_infer_start) * 1000
        infer_llm_tokens = llm.total_tokens - llm_tokens_before_infer
        infer_emb_calls = n_q  # retrieve 1次/问题，judge不用embedding

        print(f"  Ingest | time: {ingest_ms:.0f} ms | emb_calls: {ingest_emb_calls} | llm_tokens: {ingest_llm_tokens}")
        print(f"  Infer  | time: {infer_ms:.0f} ms | emb_calls: {infer_emb_calls} | llm_tokens: {infer_llm_tokens}")

        total_ingest_ms += ingest_ms
        total_ingest_emb_calls += ingest_emb_calls
        total_ingest_llm_tokens += ingest_llm_tokens
        total_infer_ms += infer_ms
        total_infer_emb_calls += infer_emb_calls
        total_infer_llm_tokens += infer_llm_tokens

        mem.reset()

    print(f"\n总结: {total_correct}/{total_questions} ({total_correct/total_questions*100:.0f}%)")
    print()
    print("── Audit ──────────────────────────────────────")
    print(f"  Ingest | time: {total_ingest_ms:.0f} ms | emb_calls: {total_ingest_emb_calls} | llm_tokens: {total_ingest_llm_tokens}")
    print(f"  Infer  | time: {total_infer_ms:.0f} ms | emb_calls: {total_infer_emb_calls} | llm_tokens: {total_infer_llm_tokens}")
    print(f"  Total  | time: {total_ingest_ms + total_infer_ms:.0f} ms | llm_tokens: {llm.total_tokens}")
    print("────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
