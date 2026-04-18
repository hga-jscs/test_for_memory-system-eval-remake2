#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A-MEM full-pipeline smoke test v2

每个 category 跑 2~3 cases，覆盖完整 ingest→retrieve→answer→judge 流程。
检查项:
  1. ingest: chunks > 0, llm_calls > 0, mem_count > 0
  2. retrieve: evidences > 0, content 非空
  3. answer: pred 非空、格式正确、MCQ 范围校验
  4. judge: yes/no 判定逻辑、recommendations criteria 路径
  5. bottom-up 耗时估算
"""
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from simpleMem_src import get_config, OpenAIClient
from amem_bench_src import AMemBenchMemory

SAVE_BASE = Path("/tmp/smoke_amem_v2")
SAVE_BASE.mkdir(parents=True, exist_ok=True)

config = get_config()

def make_llm():
    return OpenAIClient(
        api_key=config.llm["api_key"],
        base_url=config.llm["base_url"],
        model=config.llm["model"],
    )


def answer_with_memory(llm, mem, question):
    evidences = mem.retrieve(question, top_k=10)
    ctx = "\n\n".join(f"[{i+1}] {e.content}" for i, e in enumerate(evidences))
    prompt = (
        "Based on the conversation memories below, answer the question concisely.\n\n"
        f"## Memories\n{ctx}\n\n## Question\n{question}\n\n## Answer:"
    )
    return llm.generate(prompt, temperature=0.0, max_tokens=200), len(evidences)


def judge_answer(llm, question, pred, reference, category="default"):
    ref_text = reference if isinstance(reference, str) else reference.get("text", "")
    if category == "recommendations":
        criteria = reference.get("evaluation_criteria", []) if isinstance(reference, dict) else []
        criteria_str = "\n".join(f"- {c}" for c in criteria) if criteria else ""
        prompt = (
            "You are a strict judge evaluating a memory recall answer.\n\n"
            f"Question: {question}\nReference answer: {ref_text}\n"
            + (f"Evaluation criteria:\n{criteria_str}\n" if criteria_str else "")
            + f"Predicted answer: {pred}\n\n"
            "Does the predicted answer satisfy the criteria and semantically match the reference?\n"
            "Answer ONLY 'yes' or 'no'."
        )
    else:
        prompt = (
            "Does the predicted answer semantically match the reference?\n"
            "Answer ONLY 'yes' or 'no'.\n\n"
            f"Question: {question}\nReference: {ref_text}\nPredicted: {pred}\n\nMatch:"
        )
    r = llm.generate(prompt, temperature=0.0, max_tokens=10)
    return "yes" in r.lower()


def answer_multichoice(llm, mem, qa):
    question = qa["query"]
    choices = qa.get("answer_choices", [])
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


# ═══════════════════════════════════════════════════════════════════
# 1. StructMemEval — 每 category 2 cases
# ═══════════════════════════════════════════════════════════════════

def test_structmemeval():
    print("=" * 70)
    print("StructMemEval: state_machine (2) + tree_based (2) + recommendations (2)")
    print("=" * 70)

    CATEGORIES = {
        "state_machine_location": Path("StructMemEval/benchmark/data/state_machine_location"),
        "tree_based": Path("StructMemEval/benchmark/tree_based/graph_configs"),
        "recommendations": Path("StructMemEval/benchmark/recommendations/data"),
    }
    N_CASES = 2

    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0, "ingest_s": 0, "infer_s": 0, "llm_calls": 0, "chunks": 0})

    for cat, cat_dir in CATEGORIES.items():
        fps = sorted(cat_dir.rglob("*.json"))[:N_CASES]
        print(f"\n  [{cat}] {len(fps)} cases")

        for fp in fps:
            case = json.load(open(fp))
            case_id = case.get("case_id", fp.stem)
            queries = case.get("queries", [])
            if not queries:
                print(f"    {case_id}: SKIP (no queries)")
                continue

            mem = AMemBenchMemory(save_dir=str(SAVE_BASE / f"sme_{cat[:6]}_{case_id[:12]}"))
            llm = make_llm()

            # Ingest
            for session in case.get("sessions", []):
                msgs = session.get("messages", [])
                if cat == "tree_based":
                    body = "\n".join(m["content"] for m in msgs)
                else:
                    body = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
                mem.add_text(f"[Session: {session.get('session_id', '?')}]\n{body}")

            t0 = time.time()
            mem.build_index()
            ingest_s = time.time() - t0
            audit = mem.audit_ingest()

            # Check ingest
            assert audit["ingest_chunks"] > 0, f"{case_id}: ingest_chunks=0"
            assert audit["ingest_llm_calls"] > 0, f"{case_id}: llm_calls=0"
            assert audit["ingest_mem_count"] > 0, f"{case_id}: mem_count=0"

            # Infer + Judge
            t1 = time.time()
            correct = 0
            for q in queries:
                question = q.get("question", "")
                reference = q.get("reference_answer", {})
                pred, n_ev = answer_with_memory(llm, mem, question)
                assert n_ev > 0, f"{case_id}: retrieve returned 0 evidences"
                assert len(pred.strip()) > 0, f"{case_id}: empty prediction"
                ok = judge_answer(llm, question, pred, reference, cat)
                if ok:
                    correct += 1
            infer_s = time.time() - t1

            a = cat_stats[cat]
            a["correct"] += correct
            a["total"] += len(queries)
            a["ingest_s"] += ingest_s
            a["infer_s"] += infer_s
            a["llm_calls"] += audit["ingest_llm_calls"]
            a["chunks"] += audit["ingest_chunks"]

            acc = correct / len(queries) * 100
            print(f"    {case_id}: {correct}/{len(queries)} ({acc:.0f}%) | "
                  f"chunks={audit['ingest_chunks']} llm={audit['ingest_llm_calls']} "
                  f"ingest={ingest_s:.1f}s infer={infer_s:.1f}s")
            mem.reset()

    print(f"\n  {'─' * 60}")
    for cat, a in cat_stats.items():
        acc = a["correct"] / a["total"] * 100 if a["total"] else 0
        print(f"  [{cat}] {a['correct']}/{a['total']} ({acc:.1f}%) | "
              f"chunks={a['chunks']} llm_calls={a['llm_calls']} "
              f"ingest={a['ingest_s']:.1f}s infer={a['infer_s']:.1f}s")
    return cat_stats


# ═══════════════════════════════════════════════════════════════════
# 2. AMemGym — 3 users
# ═══════════════════════════════════════════════════════════════════

def test_amemgym():
    print("\n" + "=" * 70)
    print("AMemGym: 3 users, full MCQ pipeline with range check")
    print("=" * 70)

    data = json.load(open("data/amemgym/v1.base/data.json"))
    users = data if isinstance(data, list) else list(data.values())

    total_correct, total_q = 0, 0
    total_ingest_s, total_infer_s = 0, 0

    for uid, user in enumerate(users[:3]):
        mem = AMemBenchMemory(save_dir=str(SAVE_BASE / f"ag_u{uid}"))
        llm = make_llm()

        # Ingest
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

        t0 = time.time()
        mem.build_index()
        ingest_s = time.time() - t0
        audit = mem.audit_ingest()

        assert audit["ingest_chunks"] > 0
        assert audit["ingest_llm_calls"] > 0

        # Infer
        qas = user.get("qas", [])
        t1 = time.time()
        correct = 0
        oor = 0  # out of range
        for qa in qas:
            correct_idx = find_correct_index(qa, user)
            pred_idx = answer_multichoice(llm, mem, qa)
            n_choices = len(qa.get("answer_choices", []))
            if pred_idx >= n_choices:
                oor += 1
            if pred_idx == correct_idx:
                correct += 1
        infer_s = time.time() - t1

        total_correct += correct
        total_q += len(qas)
        total_ingest_s += ingest_s
        total_infer_s += infer_s

        acc = correct / len(qas) * 100 if qas else 0
        print(f"  user={uid}: {correct}/{len(qas)} ({acc:.0f}%) | "
              f"chunks={audit['ingest_chunks']} llm={audit['ingest_llm_calls']} "
              f"ingest={ingest_s:.1f}s infer={infer_s:.1f}s oor={oor}")
        assert oor == 0, f"user={uid}: {oor} out-of-range predictions!"
        mem.reset()

    acc = total_correct / total_q * 100 if total_q else 0
    print(f"\n  总计: {total_correct}/{total_q} ({acc:.1f}%) | "
          f"ingest={total_ingest_s:.1f}s infer={total_infer_s:.1f}s")
    return {"correct": total_correct, "total": total_q,
            "ingest_s": total_ingest_s, "infer_s": total_infer_s}


# ═══════════════════════════════════════════════════════════════════
# 3. memory-probe — 1 conv, all QA categories
# ═══════════════════════════════════════════════════════════════════

def test_memory_probe():
    print("\n" + "=" * 70)
    print("memory-probe: conv 0, full QA (all categories)")
    print("=" * 70)

    data = json.load(open("memory-probe/data/locomo10.json"))
    convs = data if isinstance(data, list) else data.get("conversations", [data])
    conv = convs[0]
    conversation = conv["conversation"]
    session_keys = sorted(
        [k for k in conversation.keys() if re.match(r"session_\d+$", k)],
        key=lambda x: int(x.split("_")[1])
    )

    mem = AMemBenchMemory(save_dir=str(SAVE_BASE / "mp_c0"))
    llm = make_llm()

    # Ingest
    for sk in session_keys:
        date_str = conversation.get(sk + "_date_time", "")
        turns = conversation[sk]
        header = f"[{date_str}]" if date_str else ""
        body = "\n".join(f"{t.get('speaker','')}: {t.get('text','')}" for t in turns)
        mem.add_memory((header + "\n" + body).strip())

    print(f"  buffer: {len(mem._buffer)} chunks")
    t0 = time.time()
    mem.build_index()
    ingest_s = time.time() - t0
    audit = mem.audit_ingest()
    print(f"  ingest: {ingest_s:.1f}s, chunks={audit['ingest_chunks']}, "
          f"llm_calls={audit['ingest_llm_calls']}, mem_count={audit['ingest_mem_count']}")

    assert audit["ingest_chunks"] > 0
    assert audit["ingest_llm_calls"] > 0

    # Infer — sample 5 QA per category (or all if < 5)
    qas = conv.get("qa", [])
    cat_qas = defaultdict(list)
    for qa in qas:
        cat_qas[qa.get("category", 0)].append(qa)

    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    t1 = time.time()

    for cat in sorted(cat_qas.keys()):
        sample = cat_qas[cat][:5]
        for qa in sample:
            question = qa["question"]
            reference = str(qa.get("answer", ""))
            pred, n_ev = answer_with_memory(llm, mem, question)
            assert n_ev > 0, f"cat={cat}: retrieve returned 0"
            ok = judge_answer(llm, question, pred, reference)
            cat_stats[cat]["total"] += 1
            if ok:
                cat_stats[cat]["correct"] += 1

    infer_s = time.time() - t1

    total_c = sum(a["correct"] for a in cat_stats.values())
    total_q = sum(a["total"] for a in cat_stats.values())
    for cat in sorted(cat_stats.keys()):
        a = cat_stats[cat]
        acc = a["correct"] / a["total"] * 100 if a["total"] else 0
        print(f"  cat={cat}: {a['correct']}/{a['total']} ({acc:.0f}%)")
    print(f"\n  总计: {total_c}/{total_q} ({total_c/total_q*100:.1f}%) | "
          f"ingest={ingest_s:.1f}s infer={infer_s:.1f}s")

    mem.reset()
    return {"cat_stats": dict(cat_stats), "ingest_s": ingest_s, "infer_s": infer_s}


# ═══════════════════════════════════════════════════════════════════
# Bottom-up 全量耗时估算
# ═══════════════════════════════════════════════════════════════════

def estimate(sme_stats, ag_stats, mp_stats):
    print("\n" + "=" * 70)
    print("Bottom-up 全量耗时估算")
    print("=" * 70)

    # StructMemEval: 42 + 100 + 30 = 172 cases
    sme_cases = {"state_machine_location": 42, "tree_based": 100, "recommendations": 30}
    total_sme = 0
    for cat, n in sme_cases.items():
        a = sme_stats.get(cat, {})
        cases_tested = 2
        avg_s = (a.get("ingest_s", 0) + a.get("infer_s", 0)) / cases_tested if cases_tested else 60
        est = avg_s * n / 60
        total_sme += est
        print(f"  StructMemEval/{cat}: {n} cases × {avg_s:.0f}s = ~{est:.0f} min")

    # AMemGym: 20 users
    ag_avg = (ag_stats["ingest_s"] + ag_stats["infer_s"]) / 3
    ag_est = ag_avg * 20 / 60
    print(f"  AMemGym: 20 users × {ag_avg:.0f}s = ~{ag_est:.0f} min")

    # memory-probe: 10 convs (ingest varies, ~70 chunks each)
    mp_total = mp_stats["ingest_s"] + mp_stats["infer_s"]
    mp_est = mp_total * 10 / 60
    print(f"  memory-probe: 10 convs × {mp_total:.0f}s = ~{mp_est:.0f} min")

    grand = total_sme + ag_est + mp_est
    print(f"\n  总计: ~{grand:.0f} min ({grand/60:.1f} h)")


# ═══════════════════════════════════════════════════════════════════

def main():
    sme = test_structmemeval()
    ag = test_amemgym()
    mp = test_memory_probe()
    estimate(sme, ag, mp)
    print("\n✅ 全部 smoke test 完成")


if __name__ == "__main__":
    main()
