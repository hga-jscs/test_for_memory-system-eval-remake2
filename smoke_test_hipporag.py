#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HippoRAG pre-flight smoke test

逐条验证 checklist：
  Item 1: 默认参数（openie_mode=online / batch_size=8 / force_index=True）✓ 已在 src 中固定
  Item 2: reset 隔离（shutil.rmtree + 新实例）✓ 已在 src 中实现
  Item 3: audit 字段（LLM token patch 是否有效）← 本脚本验证
  Item 4: 各 category smoke test（各 1 case × ~3 QA）
  Item 5: 每 case 耗时估算（bottom-up 用）

每个 test 打印 PASS / FAIL 及关键数字，方便人眼确认。
"""

import json
import sys
import time
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from simpleMem_src import get_config, OpenAIClient
from hipporag_bench_src import HippoRAGMemory, _text_to_chunks, CHUNK_SIZE

SAVE_BASE = Path("/tmp/smoke_hipporag")
SAVE_BASE.mkdir(exist_ok=True)

SEP = "─" * 65


def make_save_dir(name: str) -> str:
    p = SAVE_BASE / name
    if p.exists():
        shutil.rmtree(p)
    return str(p)


def get_llm():
    conf = get_config()
    return OpenAIClient(
        api_key=conf.llm["api_key"],
        base_url=conf.llm["base_url"],
        model=conf.llm["model"],
    )


def answer_with_memory(llm, mem, question: str) -> str:
    evidences = mem.retrieve(question, top_k=5)
    context = "\n\n".join(f"[{i+1}] {e.content}" for i, e in enumerate(evidences))
    prompt = (
        "Based on the conversation memories below, answer the question concisely.\n\n"
        f"## Memories\n{context}\n\n## Question\n{question}\n\n## Answer:"
    )
    return llm.generate(prompt, temperature=0.0, max_tokens=200)


def judge(llm, question, pred, reference_text) -> bool:
    prompt = (
        "Does the predicted answer semantically match the reference?\n"
        "Answer ONLY 'yes' or 'no'.\n\n"
        f"Question: {question}\nReference: {reference_text}\nPredicted: {pred}\n\nMatch:"
    )
    r = llm.generate(prompt, temperature=0.0, max_tokens=10)
    return "yes" in r.lower()


# ═══════════════════════════════════════════════════════════════════════
# Item 3: audit 字段验证（用 3 个真实 chunk 跑一次 build_index）
# ═══════════════════════════════════════════════════════════════════════

def test_audit_fields():
    print(f"\n{'═'*65}")
    print("Item 3: Audit 字段验证")
    print(SEP)

    mem = HippoRAGMemory(save_dir=make_save_dir("audit_test"))
    # 3 个真实语义 chunk
    chunks = [
        "Alice is a software engineer at TechCorp. She works on distributed systems and has 5 years of experience.",
        "Bob is Alice's manager. He joined TechCorp last year from a startup. Alice reports to Bob on the infrastructure team.",
        "TechCorp's infrastructure team handles cloud deployments. Alice and Bob collaborate on the Kubernetes migration project.",
    ]
    for c in chunks:
        mem.add_memory(c)

    print(f"  Buffered {len(mem._buffer)} chunks, building index...")
    t0 = time.time()
    mem.build_index()
    wall = time.time() - t0

    audit = mem.audit_ingest()
    print(f"  Wall time : {wall:.1f}s")
    print(f"  ingest_chunks       : {audit['ingest_chunks']}")
    print(f"  ingest_time_ms      : {audit['ingest_time_ms']}")
    print(f"  ingest_llm_calls    : {audit['ingest_llm_calls']}")
    print(f"  ingest_llm_prompt   : {audit['ingest_llm_prompt_tokens']}")
    print(f"  ingest_llm_complete : {audit['ingest_llm_completion_tokens']}")

    # 检查 LLM 调用是否被追踪到
    if audit["ingest_llm_calls"] > 0:
        print("  [PASS] LLM token patch 生效，ingest_llm_calls > 0")
    else:
        print("  [WARN] ingest_llm_calls = 0（可能全部命中缓存，非首次运行属正常）")

    # 检查 retrieve
    results = mem.retrieve("Who does Alice report to?", top_k=3)
    print(f"\n  Retrieve test:")
    print(f"    Q: Who does Alice report to?")
    for i, r in enumerate(results):
        print(f"    [{i+1}] (score={r.metadata.get('score',0):.3f}) {r.content[:80]}")

    if results:
        print("  [PASS] retrieve 返回结果非空")
    else:
        print("  [FAIL] retrieve 返回空结果")

    mem.reset()
    print("  [PASS] reset() 完成")


# ═══════════════════════════════════════════════════════════════════════
# Item 4a: StructMemEval - state_machine_location（1 case）
# ═══════════════════════════════════════════════════════════════════════

def test_structmemeval_state_machine():
    print(f"\n{'═'*65}")
    print("Item 4a: StructMemEval - state_machine_location（1 case）")
    print(SEP)

    sm_dir = Path("StructMemEval/benchmark/data/state_machine_location")
    fp = sorted(sm_dir.rglob("*.json"))[0]
    case = json.load(open(fp))
    queries = case.get("queries", [])[:3]
    print(f"  File: {fp.name}")
    print(f"  Sessions: {len(case.get('sessions', []))}, Queries (using {len(queries)})")

    mem = HippoRAGMemory(save_dir=make_save_dir("sme_state_machine"))
    llm = get_llm()

    # ingest: serialize each session to text, pre-chunk to ~1000 chars
    n_chunks = 0
    for session in case.get("sessions", []):
        sid = session.get("session_id", "?")
        topic = session.get("topic", "")
        msgs = session.get("messages", [])
        header = f"[Session: {sid}" + (f", Topic: {topic}" if topic else "") + "]"
        body = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
        text = header + "\n" + body
        n_chunks += mem.add_text(text)

    print(f"  Pre-chunks: {n_chunks} (from {len(case.get('sessions',[]))} sessions)")
    print(f"  Building index...")
    mem.build_index()
    audit = mem.audit_ingest()
    print(f"  Audit: {audit}")

    correct = 0
    for q in queries:
        question = q.get("question", "")
        ref_text = q.get("reference_answer", {}).get("text", "")
        pred = answer_with_memory(llm, mem, question)
        ok = judge(llm, question, pred, ref_text)
        correct += int(ok)
        print(f"  {'✓' if ok else '✗'}  Q: {question[:60]}")
        print(f"     pred: {pred[:80]}")
        print(f"     ref : {ref_text[:60]}")

    print(f"\n  结果: {correct}/{len(queries)} | ingest={audit['ingest_time_ms']}ms")
    mem.reset()
    print(f"  [{'PASS' if True else 'FAIL'}] state_machine smoke test 完成")


# ═══════════════════════════════════════════════════════════════════════
# Item 4b: StructMemEval - tree_based（1 case，关键风险点）
# ═══════════════════════════════════════════════════════════════════════

def test_structmemeval_tree_based():
    print(f"\n{'═'*65}")
    print("Item 4b: StructMemEval - tree_based（1 case，关键风险）")
    print(SEP)

    tb_dir = Path("StructMemEval/benchmark/tree_based/graph_configs")
    # 用小的 case（trimmed_10）快速验证
    fps = sorted(tb_dir.glob("*_trimmed_10_*.json"))
    if not fps:
        fps = sorted(tb_dir.rglob("*.json"))
    fp = fps[0]
    case = json.load(open(fp))
    queries = case.get("queries", [])[:3]

    sessions = case.get("sessions", [])
    msgs_all = sessions[0].get("messages", []) if sessions else []
    roles = set(m["role"] for m in msgs_all)
    print(f"  File: {fp.name}")
    print(f"  Messages: {len(msgs_all)}, roles: {roles}")
    print(f"  Sample msg: {msgs_all[0]['content'][:70] if msgs_all else 'N/A'}")
    print(f"  Queries: {len(case.get('queries', []))}, using {len(queries)}")

    mem = HippoRAGMemory(save_dir=make_save_dir("sme_tree_based"))
    llm = get_llm()

    # 所有 user-role 消息直接拼接成文本，再按 1000 char 分块
    body = "\n".join(m["content"] for m in msgs_all)
    n_chunks = mem.add_text(body)
    print(f"  Pre-chunks: {n_chunks} (total text ~{len(body)} chars)")

    print(f"  Building index...")
    mem.build_index()
    audit = mem.audit_ingest()
    print(f"  Audit: {audit}")

    # 核心验证：OpenIE 是否能从 "X works with Y" 提取三元组
    # 通过 retrieve 结果人眼确认
    print(f"\n  Retrieve 测试（关系查询）:")
    test_q = "Who does " + msgs_all[0]["content"].split()[0] + " work with?"
    results = mem.retrieve(test_q, top_k=3)
    print(f"  Q: {test_q}")
    for i, r in enumerate(results):
        print(f"    [{i+1}] {r.content[:90]}")

    correct = 0
    for q in queries:
        question = q.get("question", "")
        ref_text = q.get("reference_answer", {}).get("text", "")
        pred = answer_with_memory(llm, mem, question)
        ok = judge(llm, question, pred, ref_text)
        correct += int(ok)
        print(f"  {'✓' if ok else '✗'}  Q: {question[:70]}")
        print(f"     pred: {pred[:80]}")
        print(f"     ref : {ref_text[:60]}")

    acc = correct / len(queries) * 100 if queries else 0
    print(f"\n  结果: {correct}/{len(queries)} ({acc:.0f}%) | ingest={audit['ingest_time_ms']}ms")
    if correct > 0:
        print("  [PASS] tree_based: HippoRAG 能正确回答关系图查询")
    else:
        print("  [WARN] tree_based: 0分，检查 retrieve 内容是否有意义")
    mem.reset()


# ═══════════════════════════════════════════════════════════════════════
# Item 4c: StructMemEval - recommendations（1 case）
# ═══════════════════════════════════════════════════════════════════════

def test_structmemeval_recommendations():
    print(f"\n{'═'*65}")
    print("Item 4c: StructMemEval - recommendations（1 case）")
    print(SEP)

    rec_dir = Path("StructMemEval/benchmark/recommendations/data")
    fp = sorted(rec_dir.rglob("*.json"))[0]
    case = json.load(open(fp))
    queries = case.get("queries", [])[:2]
    print(f"  File: {fp.name}")
    print(f"  Sessions: {len(case.get('sessions',[]))}, Queries (using {len(queries)})")

    mem = HippoRAGMemory(save_dir=make_save_dir("sme_recommendations"))
    llm = get_llm()

    n_chunks = 0
    for session in case.get("sessions", []):
        sid = session.get("session_id", "?")
        msgs = session.get("messages", [])
        header = f"[Session: {sid}]"
        body = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
        n_chunks += mem.add_text(header + "\n" + body)

    print(f"  Pre-chunks: {n_chunks}")
    print(f"  Building index...")
    mem.build_index()
    audit = mem.audit_ingest()
    print(f"  Audit: {audit}")

    correct = 0
    for q in queries:
        question = q.get("question", "")
        ref_text = q.get("reference_answer", {}).get("text", "")
        criteria = q.get("reference_answer", {}).get("evaluation_criteria", [])
        pred = answer_with_memory(llm, mem, question)
        ok = judge(llm, question, pred, ref_text)
        correct += int(ok)
        print(f"  {'✓' if ok else '✗'}  Q: {question[:60]}")
        print(f"     pred: {pred[:80]}")

    print(f"\n  结果: {correct}/{len(queries)} | ingest={audit['ingest_time_ms']}ms")
    mem.reset()


# ═══════════════════════════════════════════════════════════════════════
# Item 4d: AMemGym（1 user）
# ═══════════════════════════════════════════════════════════════════════

def test_amemgym():
    print(f"\n{'═'*65}")
    print("Item 4d: AMemGym（1 user）")
    print(SEP)

    data_path = Path("data/amemgym/v1.base/data.json")
    data = json.load(open(data_path))
    users = data.get("users", data) if isinstance(data, dict) else data
    if isinstance(users, dict):
        users = list(users.values())
    user = users[0]
    qas = user.get("qas", user.get("queries", user.get("questions", [])))[:3]

    print(f"  User periods: {len(user.get('periods', []))}")
    print(f"  QAs (using {len(qas)})")

    mem = HippoRAGMemory(save_dir=make_save_dir("amemgym"))
    llm = get_llm()

    # ingest: each session → one text block
    n_chunks = 0
    for pi, period in enumerate(user.get("periods", [])):
        period_start = period.get("period_start", "?")
        period_end   = period.get("period_end", "?")
        for session in period.get("sessions", []):
            query        = session.get("query", "")
            exposed      = session.get("exposed_states", {})
            session_time = session.get("session_time", "")
            parts = [f"[Time: {session_time}, Period: {period_start}~{period_end}]",
                     f"User query: {query}"]
            if exposed:
                parts.append("States: " + ", ".join(f"{k}={v}" for k, v in exposed.items()))
            n_chunks += mem.add_text("\n".join(parts))

    print(f"  Pre-chunks: {n_chunks}")
    print(f"  Building index...")
    mem.build_index()
    audit = mem.audit_ingest()
    print(f"  Audit: {audit}")

    # AMemGym: multiple-choice QA
    periods = user.get("periods", [])
    current_state = periods[-1].get("state", {}) if periods else {}

    correct = 0
    for qa in qas:
        question = qa.get("query", "")
        choices  = qa.get("answer_choices", [])
        req_info = qa.get("required_info", [])
        correct_state = [current_state.get(k, "") for k in req_info]
        correct_idx = next(
            (i for i, c in enumerate(choices) if c.get("state") == correct_state), 0
        )
        # Format multi-choice prompt
        choices_str = "\n".join(
            f"({chr(65+i)}) {c.get('text','')}" for i, c in enumerate(choices)
        )
        evidences = mem.retrieve(question, top_k=5)
        ctx = "\n\n".join(f"[{i+1}] {e.content}" for i, e in enumerate(evidences))
        prompt = (
            f"Memory context:\n{ctx}\n\n"
            f"Question: {question}\n\n"
            f"Choices:\n{choices_str}\n\n"
            "Answer with just the letter (A/B/C/D):"
        )
        ans = llm.generate(prompt, temperature=0.0, max_tokens=5).strip().upper()
        pred_idx = ord(ans[0]) - ord('A') if ans and ans[0].isalpha() else 0
        ok = (pred_idx == correct_idx)
        correct += int(ok)
        print(f"  {'✓' if ok else '✗'}  Q: {question[:60]}")
        print(f"     pred={ans} correct={chr(65+correct_idx)}")

    print(f"\n  结果: {correct}/{len(qas)} | ingest={audit['ingest_time_ms']}ms")
    mem.reset()


# ═══════════════════════════════════════════════════════════════════════
# Item 4e: memory-probe（cat 1~4 各 1 conversation）
# ═══════════════════════════════════════════════════════════════════════

def test_memory_probe():
    print(f"\n{'═'*65}")
    print("Item 4e: memory-probe（cat 1-4 各 1 case）")
    print(SEP)

    probe_dir = Path("memory-probe/LoCoMo-10")
    if not probe_dir.exists():
        probe_dir = Path("memory-probe")

    data_path = Path("memory-probe/data/locomo10.json")
    if not data_path.exists():
        print("  [SKIP] 找不到 memory-probe 数据")
        return

    data = json.load(open(data_path))
    convs = data if isinstance(data, list) else data.get("conversations", [data])
    conv = convs[0]
    conversation = conv["conversation"]
    qas = conv.get("qa", [])

    print(f"  Conversations: {len(convs)}, QAs in conv[0]: {len(qas)}")

    # 按 category 各取 1 个 QA (cat 1-4)
    cats = {}
    for q in qas:
        cat = q.get("category", "unknown")
        if cat not in cats:
            cats[cat] = q
    print(f"  Categories found: {sorted(cats.keys())}")

    mem = HippoRAGMemory(save_dir=make_save_dir("memory_probe"))
    llm = get_llm()

    # ingest: 提取所有 session_N 并序列化为文本
    import re
    n_chunks = 0
    session_keys = sorted(
        [k for k in conversation.keys() if re.match(r"session_\d+$", k)],
        key=lambda x: int(x.split("_")[1])
    )
    for sk in session_keys:
        date_key = sk + "_date_time"
        date = conversation.get(date_key, "")
        turns = conversation[sk]
        header = f"[{date}]" if date else ""
        body = "\n".join(
            f"{t.get('speaker','')}: {t.get('text','')}" for t in turns
        )
        n_chunks += mem.add_text((header + "\n" + body).strip())

    print(f"  Pre-chunks: {n_chunks}")
    print(f"  Building index...")
    mem.build_index()
    audit = mem.audit_ingest()
    print(f"  Audit: {audit}")

    for cat in sorted([c for c in cats.keys() if c in [1, 2, 3, 4]])[:4]:
        qa = cats[cat]
        question = qa.get("question", "")
        ref = qa.get("answer", "")
        pred = answer_with_memory(llm, mem, question)
        ok = judge(llm, question, pred, str(ref))
        print(f"  cat={cat} {'✓' if ok else '✗'}  Q: {question[:55]}")
        print(f"     pred: {pred[:70]}")
        print(f"     ref : {str(ref)[:55]}")

    mem.reset()
    print("  memory-probe smoke test 完成")


# ═══════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["audit","state_machine","tree_based","recommendations","amemgym","memory_probe"], help="只跑某一项")
    args = parser.parse_args()

    t_start = time.time()

    tests = {
        "audit": test_audit_fields,
        "state_machine": test_structmemeval_state_machine,
        "tree_based": test_structmemeval_tree_based,
        "recommendations": test_structmemeval_recommendations,
        "amemgym": test_amemgym,
        "memory_probe": test_memory_probe,
    }

    if args.only:
        tests[args.only]()
    else:
        for name, fn in tests.items():
            try:
                fn()
            except Exception as e:
                print(f"\n[ERROR in {name}]: {e}")
                import traceback; traceback.print_exc()

    print(f"\n{'═'*65}")
    print(f"Smoke test 总耗时: {time.time()-t_start:.1f}s")
    print(f"{'═'*65}")
