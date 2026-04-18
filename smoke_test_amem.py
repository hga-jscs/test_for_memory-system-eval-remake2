#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A-MEM pre-flight smoke test

逐 category 验证：
1. import 正常
2. ingest 产生合理数量的 chunks 和 LLM calls
3. evolution 被触发 (mem_count > 0, llm_calls > 0)
4. retrieve 返回有意义的内容
5. 至少 1 个 QA 的 pred 非空且有意义
6. audit 字段完整

用法:
  python smoke_test_amem.py                # 全部 category
  python smoke_test_amem.py --only amemgym # 单个 category
"""
import argparse
import json
from benchmark_io_utils import load_json_with_fallback
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from simpleMem_src import get_config, OpenAIClient
from amem_bench_src import AMemBenchMemory

SAVE_BASE = Path("/tmp/smoke_amem")


def check(name: str, ok: bool, detail: str = ""):
    tag = "✅" if ok else "❌"
    print(f"  {tag} {name}" + (f" — {detail}" if detail else ""))
    return ok


# ── 1. StructMemEval: state_machine_location ────────────────────

def test_state_machine():
    print("\n" + "=" * 60)
    print("state_machine_location (StructMemEval)")
    print("=" * 60)

    data_dir = Path("StructMemEval/benchmark/data/state_machine_location")
    fp = sorted(data_dir.glob("*.json"))[0]
    case = load_json_with_fallback(fp)
    print(f"  case: {fp.name}, sessions={len(case.get('sessions', []))}, queries={len(case.get('queries', []))}")

    mem = AMemBenchMemory(save_dir=str(SAVE_BASE / "sm"))
    config = get_config()
    llm = OpenAIClient(api_key=config.llm["api_key"], base_url=config.llm["base_url"], model=config.llm["model"])

    # Ingest
    for session in case.get("sessions", []):
        body = "\n".join(f"{m['role']}: {m['content']}" for m in session.get("messages", []))
        mem.add_text(f"[Session: {session.get('session_id', '?')}]\n{body}")

    print(f"  buffer size: {len(mem._buffer)} chunks")
    check("buffer > 0", len(mem._buffer) > 0)

    t0 = time.time()
    mem.build_index()
    dt = time.time() - t0
    audit = mem.audit_ingest()
    print(f"  ingest: {dt:.1f}s, chunks={audit['ingest_chunks']}, llm_calls={audit['ingest_llm_calls']}, mem_count={audit['ingest_mem_count']}")
    print(f"  tokens: prompt={audit['ingest_llm_prompt_tokens']}, completion={audit['ingest_llm_completion_tokens']}")

    check("ingest_chunks > 0", audit["ingest_chunks"] > 0)
    check("llm_calls > 0 (evolution ON)", audit["ingest_llm_calls"] > 0)
    check("mem_count > 0", audit["ingest_mem_count"] > 0)

    # Retrieve + Answer
    q = case["queries"][0]
    evidences = mem.retrieve(q["question"], top_k=5)
    print(f"  retrieve: {len(evidences)} evidences for '{q['question'][:60]}...'")
    check("retrieve > 0", len(evidences) > 0)

    if evidences:
        print(f"    [1] {evidences[0].content[:100]}...")

    ctx = "\n".join(f"[{i+1}] {e.content}" for i, e in enumerate(evidences))
    pred = llm.generate(
        f"Based on memories:\n{ctx}\n\nQuestion: {q['question']}\nAnswer concisely:",
        temperature=0.0, max_tokens=200
    )
    ref = q["reference_answer"]["text"]
    print(f"  Q: {q['question'][:80]}")
    print(f"  Gold: {ref[:80]}")
    print(f"  Pred: {pred[:80]}")
    check("pred non-empty", len(pred.strip()) > 0)
    check("pred != 'I don't know'", "don't know" not in pred.lower())

    mem.reset()
    return True


# ── 2. StructMemEval: tree_based ─────────────────────────────────

def test_tree_based():
    print("\n" + "=" * 60)
    print("tree_based (StructMemEval)")
    print("=" * 60)

    data_dir = Path("StructMemEval/benchmark/tree_based/graph_configs")
    fp = sorted(data_dir.glob("*.json"))[0]
    case = load_json_with_fallback(fp)
    print(f"  case: {fp.name}, sessions={len(case.get('sessions', []))}, queries={len(case.get('queries', []))}")

    mem = AMemBenchMemory(save_dir=str(SAVE_BASE / "tb"))
    config = get_config()
    llm = OpenAIClient(api_key=config.llm["api_key"], base_url=config.llm["base_url"], model=config.llm["model"])

    # Ingest (tree_based: all user-role messages)
    for session in case.get("sessions", []):
        body = "\n".join(m["content"] for m in session.get("messages", []))
        mem.add_text(f"[Session: {session.get('session_id', '?')}]\n{body}")

    print(f"  buffer size: {len(mem._buffer)} chunks")
    check("buffer > 0", len(mem._buffer) > 0)

    mem.build_index()
    audit = mem.audit_ingest()
    print(f"  ingest: chunks={audit['ingest_chunks']}, llm_calls={audit['ingest_llm_calls']}, mem_count={audit['ingest_mem_count']}")

    check("ingest_chunks > 0", audit["ingest_chunks"] > 0)
    check("mem_count > 0", audit["ingest_mem_count"] > 0)

    # Retrieve + Answer
    q = case["queries"][0]
    evidences = mem.retrieve(q["question"], top_k=5)
    print(f"  retrieve: {len(evidences)} evidences")
    check("retrieve > 0", len(evidences) > 0)

    ctx = "\n".join(f"[{i+1}] {e.content}" for i, e in enumerate(evidences))
    pred = llm.generate(
        f"Based on memories:\n{ctx}\n\nQuestion: {q['question']}\nAnswer concisely:",
        temperature=0.0, max_tokens=200
    )
    ref = q["reference_answer"]["text"]
    print(f"  Q: {q['question'][:80]}")
    print(f"  Gold: {ref[:80]}")
    print(f"  Pred: {pred[:80]}")
    check("pred non-empty", len(pred.strip()) > 0)

    mem.reset()
    return True


# ── 3. StructMemEval: recommendations ────────────────────────────

def test_recommendations():
    print("\n" + "=" * 60)
    print("recommendations (StructMemEval)")
    print("=" * 60)

    data_dir = Path("StructMemEval/benchmark/recommendations/data")
    fps = sorted(data_dir.rglob("*.json"))
    fp = fps[0]
    case = load_json_with_fallback(fp)
    print(f"  case: {fp.name}, sessions={len(case.get('sessions', []))}, queries={len(case.get('queries', []))}")

    mem = AMemBenchMemory(save_dir=str(SAVE_BASE / "rec"))
    config = get_config()
    llm = OpenAIClient(api_key=config.llm["api_key"], base_url=config.llm["base_url"], model=config.llm["model"])

    for session in case.get("sessions", []):
        body = "\n".join(f"{m['role']}: {m['content']}" for m in session.get("messages", []))
        mem.add_text(f"[Session: {session.get('session_id', '?')}]\n{body}")

    print(f"  buffer size: {len(mem._buffer)} chunks")
    mem.build_index()
    audit = mem.audit_ingest()
    print(f"  ingest: chunks={audit['ingest_chunks']}, llm_calls={audit['ingest_llm_calls']}, mem_count={audit['ingest_mem_count']}")

    check("chunks > 0", audit["ingest_chunks"] > 0)
    check("mem_count > 0", audit["ingest_mem_count"] > 0)

    q = case["queries"][0]
    evidences = mem.retrieve(q["question"], top_k=5)
    print(f"  retrieve: {len(evidences)} evidences")
    check("retrieve > 0", len(evidences) > 0)

    ctx = "\n".join(f"[{i+1}] {e.content}" for i, e in enumerate(evidences))
    pred = llm.generate(
        f"Based on memories:\n{ctx}\n\nQuestion: {q['question']}\nAnswer concisely:",
        temperature=0.0, max_tokens=200
    )
    print(f"  Q: {q['question'][:80]}")
    print(f"  Pred: {pred[:80]}")
    check("pred non-empty", len(pred.strip()) > 0)

    mem.reset()
    return True


# ── 4. AMemGym ───────────────────────────────────────────────────

def test_amemgym():
    print("\n" + "=" * 60)
    print("AMemGym")
    print("=" * 60)

    data = load_json_with_fallback("data/amemgym/v1.base/data.json")
    users = data if isinstance(data, list) else list(data.values())
    user = users[0]
    print(f"  user: {user.get('id', '0')}, periods={len(user.get('periods', []))}, qas={len(user.get('qas', []))}")

    mem = AMemBenchMemory(save_dir=str(SAVE_BASE / "ag"))
    config = get_config()
    llm = OpenAIClient(api_key=config.llm["api_key"], base_url=config.llm["base_url"], model=config.llm["model"])

    for period in user.get("periods", []):
        for session in period.get("sessions", []):
            query = session.get("query", "")
            exposed = session.get("exposed_states", {})
            session_time = session.get("session_time", "")
            if not query:
                continue
            parts = [f"[Time: {session_time}]", f"User query: {query}"]
            if exposed:
                parts.append("States: " + ", ".join(f"{k}={v}" for k, v in exposed.items()))
            mem.add_memory("\n".join(parts))

    print(f"  buffer size: {len(mem._buffer)} chunks")
    mem.build_index()
    audit = mem.audit_ingest()
    print(f"  ingest: chunks={audit['ingest_chunks']}, llm_calls={audit['ingest_llm_calls']}, mem_count={audit['ingest_mem_count']}")
    print(f"  tokens: prompt={audit['ingest_llm_prompt_tokens']}, completion={audit['ingest_llm_completion_tokens']}")

    check("chunks > 0", audit["ingest_chunks"] > 0)
    check("llm_calls > 0", audit["ingest_llm_calls"] > 0)
    check("mem_count > 0", audit["ingest_mem_count"] > 0)

    qa = user["qas"][0]
    evidences = mem.retrieve(qa["query"], top_k=5)
    print(f"  retrieve: {len(evidences)} evidences for '{qa['query'][:60]}...'")
    check("retrieve > 0", len(evidences) > 0)

    choices = qa.get("answer_choices", [])
    choices_str = "\n".join(f"  ({i}) {c.get('text','')}" for i, c in enumerate(choices))
    ctx = "\n".join(f"[{i+1}] {e.content}" for i, e in enumerate(evidences))
    pred = llm.generate(
        f"Memory context:\n{ctx}\n\nQuestion: {qa['query']}\n\nChoices:\n{choices_str}\n\nReply with ONLY the choice number:",
        temperature=0.0, max_tokens=10,
    )
    print(f"  Q: {qa['query'][:80]}")
    print(f"  Pred: {pred.strip()}")
    check("pred is a number", any(c.isdigit() for c in pred))

    mem.reset()
    return True


# ── 5. memory-probe ──────────────────────────────────────────────

def test_memory_probe():
    print("\n" + "=" * 60)
    print("memory-probe (conv 0, first 3 QA)")
    print("=" * 60)

    import re as _re
    data = load_json_with_fallback("memory-probe/data/locomo10.json")
    convs = data if isinstance(data, list) else data.get("conversations", [data])
    conv = convs[0]
    conversation = conv["conversation"]
    session_keys = sorted(
        [k for k in conversation.keys() if _re.match(r"session_\d+$", k)],
        key=lambda x: int(x.split("_")[1])
    )

    mem = AMemBenchMemory(save_dir=str(SAVE_BASE / "mp"))
    config = get_config()
    llm = OpenAIClient(api_key=config.llm["api_key"], base_url=config.llm["base_url"], model=config.llm["model"])

    for sk in session_keys:
        date = conversation.get(sk + "_date_time", "")
        turns = conversation[sk]
        header = f"[{date}]" if date else ""
        body = "\n".join(f"{t.get('speaker','')}: {t.get('text','')}" for t in turns)
        mem.add_memory((header + "\n" + body).strip())

    print(f"  buffer size: {len(mem._buffer)} chunks")
    mem.build_index()
    audit = mem.audit_ingest()
    print(f"  ingest: chunks={audit['ingest_chunks']}, llm_calls={audit['ingest_llm_calls']}, mem_count={audit['ingest_mem_count']}")

    check("chunks > 0", audit["ingest_chunks"] > 0)
    check("mem_count > 0", audit["ingest_mem_count"] > 0)

    qas = conv.get("qa", [])[:3]
    correct = 0
    for qa in qas:
        question = qa["question"]
        reference = str(qa.get("answer", ""))
        evidences = mem.retrieve(question, top_k=5)
        ctx = "\n".join(f"[{i+1}] {e.content}" for i, e in enumerate(evidences))
        pred = llm.generate(
            f"Based on memories:\n{ctx}\n\nQuestion: {question}\nAnswer concisely:",
            temperature=0.0, max_tokens=200,
        )
        ok_str = llm.generate(
            f"Does the predicted answer match the reference?\nQuestion: {question}\nReference: {reference}\nPredicted: {pred}\nAnswer ONLY yes or no.",
            temperature=0.0, max_tokens=10,
        )
        ok = "yes" in ok_str.lower()
        if ok:
            correct += 1
        print(f"  Q: {question[:60]}...")
        print(f"    Gold: {reference[:60]}, Pred: {pred[:60]}, Judge: {'✅' if ok else '❌'}")

    check(f"at least 1/3 correct", correct >= 1, f"{correct}/3")

    mem.reset()
    return True


# ── main ─────────────────────────────────────────────────────────

TESTS = {
    "state_machine": test_state_machine,
    "tree_based": test_tree_based,
    "recommendations": test_recommendations,
    "amemgym": test_amemgym,
    "memory_probe": test_memory_probe,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None, help="Run only this test")
    args = parser.parse_args()

    SAVE_BASE.mkdir(parents=True, exist_ok=True)

    if args.only:
        if args.only in TESTS:
            TESTS[args.only]()
        else:
            print(f"Unknown test: {args.only}. Available: {list(TESTS.keys())}")
    else:
        for name, fn in TESTS.items():
            try:
                fn()
            except Exception as e:
                print(f"\n❌ {name} FAILED: {e}")
                import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
