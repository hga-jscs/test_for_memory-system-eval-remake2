#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SimpleMem × AMemGym 小规模验证
用 1 个用户的前 3 个 period，回答前 3 个多选题。
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from simpleMem_src import SimpleRAGMemory, get_config, OpenAIClient, get_logger

DATA_PATH = Path("data/amemgym/v1.base/data.json")
MAX_PERIODS = 3
MAX_QAS = 3

logger = get_logger()


def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)


def ingest_periods(mem: SimpleRAGMemory, user: dict, max_periods: int):
    """将用户的对话 sessions 写入记忆"""
    ingested = 0
    periods = user["periods"][:max_periods]

    for pi, period in enumerate(periods):
        period_start = period.get("period_start", "?")
        period_end = period.get("period_end", "?")

        for session in period.get("sessions", []):
            query = session.get("query", "")
            exposed = session.get("exposed_states", {})
            session_time = session.get("session_time", "")

            if not query:
                continue

            # 记录用户的问题和暴露的状态信息
            parts = [f"[Time: {session_time}, Period: {period_start} ~ {period_end}]"]
            parts.append(f"User asked: {query}")

            if exposed:
                state_info = ", ".join(f"{k}={v}" for k, v in exposed.items())
                parts.append(f"Known states: {state_info}")

            text = "\n".join(parts)
            mem.add_memory(text, {
                "period": pi,
                "time": session_time,
            })
            ingested += 1

    return ingested


def answer_multichoice(llm, mem: SimpleRAGMemory, qa: dict) -> int:
    """检索记忆 + LLM 从多选项中选择答案，返回选项 index"""
    question = qa["query"]
    choices = qa["answer_choices"]

    evidences = mem.retrieve(question, top_k=5)
    context = "\n".join(
        f"[Memory {i+1}] {e.content}" for i, e in enumerate(evidences)
    )

    choices_text = "\n".join(
        f"  ({i}) {c['answer'][:150]}" for i, c in enumerate(choices)
    )

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

    # 解析数字
    for ch in result.strip():
        if ch.isdigit():
            idx = int(ch)
            if 0 <= idx < len(choices):
                return idx
    return 0  # fallback


def find_correct_index(qa: dict, user: dict, period_idx: int) -> int:
    """找到与当前 period 状态匹配的正确选项"""
    periods = user["periods"]
    if period_idx >= len(periods):
        period_idx = len(periods) - 1

    current_state = periods[period_idx].get("state", {})
    required_info = qa.get("required_info", [])

    # 收集当前状态中 required_info 对应的值
    correct_state = []
    for info_key in required_info:
        correct_state.append(current_state.get(info_key, ""))

    # 找匹配的 choice
    for i, choice in enumerate(qa["answer_choices"]):
        if choice.get("state") == correct_state:
            return i

    return 0


def main():
    print("=" * 60)
    print("SimpleMem × AMemGym 验证")
    print("=" * 60)

    data = load_data()
    user = data[0]
    user_id = user.get("id", "user_0")
    print(f"用户: {user_id}")
    print(f"Periods: {len(user['periods'])}, QAs: {len(user['qas'])}")

    config = get_config()
    mem = SimpleRAGMemory(collection_name="amemgym_test")
    llm = OpenAIClient(
        api_key=config.llm["api_key"],
        base_url=config.llm["base_url"],
        model=config.llm["model"],
    )

    # ── Ingest ──────────────────────────────────────────────
    t_ingest_start = time.time()
    llm_tokens_before_ingest = llm.total_tokens
    n = ingest_periods(mem, user, MAX_PERIODS)
    ingest_ms = (time.time() - t_ingest_start) * 1000
    ingest_llm_tokens = llm.total_tokens - llm_tokens_before_ingest
    ingest_emb_calls = mem.size

    print(f"写入记忆: {n} entries (前 {MAX_PERIODS} 个 periods)")

    # ── Infer ───────────────────────────────────────────────
    qas = user["qas"][:MAX_QAS]
    correct = 0

    print(f"\n--- 回答 {len(qas)} 个多选题 ---\n")
    t_infer_start = time.time()
    llm_tokens_before_infer = llm.total_tokens

    for i, qa in enumerate(qas):
        question = qa["query"]
        choices = qa["answer_choices"]
        correct_idx = find_correct_index(qa, user, MAX_PERIODS - 1)

        pred_idx = answer_multichoice(llm, mem, qa)
        match = pred_idx == correct_idx

        if match:
            correct += 1
            status = "✓"
        else:
            status = "✗"

        print(f"  [{status}] Q{i+1}: {question[:80]}")
        print(f"       正确: ({correct_idx}) {choices[correct_idx]['answer'][:60]}")
        print(f"       预测: ({pred_idx}) {choices[pred_idx]['answer'][:60]}")
        print()

    infer_ms = (time.time() - t_infer_start) * 1000
    infer_llm_tokens = llm.total_tokens - llm_tokens_before_infer
    infer_emb_calls = len(qas)  # 每个问题 retrieve 1 次 embedding

    # ── 结果 & Audit ────────────────────────────────────────
    print(f"结果: {correct}/{len(qas)} ({correct/len(qas)*100:.0f}%)")
    print()
    print("── Audit ──────────────────────────────────────")
    print(f"  Ingest | time: {ingest_ms:.0f} ms | emb_calls: {ingest_emb_calls} | llm_tokens: {ingest_llm_tokens}")
    print(f"  Infer  | time: {infer_ms:.0f} ms | emb_calls: {infer_emb_calls} | llm_tokens: {infer_llm_tokens}")
    print(f"  Total  | time: {ingest_ms + infer_ms:.0f} ms | llm_tokens: {llm.total_tokens}")
    print("────────────────────────────────────────────────")

    mem.reset()


if __name__ == "__main__":
    main()
