#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一 R1/R2/R3 评测脚本

对指定系统 × 指定 benchmark，ingest 一次后依次跑 R1/R2/R3 infer。
结果分别保存到 results_{bench}_{system}_r{1,2,3}.json

用法:
  python bench_r123.py --system simple --bench memory_probe
  python bench_r123.py --system hipporag --bench structmemeval
  python bench_r123.py --system mem0 --bench amemgym
  python bench_r123.py --system simple --bench all   # 跑全部 3 bench
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from simpleMem_src import get_config, OpenAIClient
from adaptors import (
    SingleTurnAdaptor, IterativeAdaptor, PlanAndActAdaptor,
    AdaptorResult,
)


# ── System factory ───────────────────────────────────────────

def create_memory(system: str, save_dir: str):
    if system == "simple":
        from simpleMem_src import SimpleRAGMemory
        return SimpleRAGMemory()
    elif system == "mem0":
        from mem0_bench_src import Mem0RAGMemory
        return Mem0RAGMemory(collection_name=Path(save_dir).name)
    elif system == "hipporag":
        from hipporag_bench_src import HippoRAGMemory
        return HippoRAGMemory(save_dir=save_dir)
    elif system == "amem":
        from amem_bench_src import AMemBenchMemory
        return AMemBenchMemory(save_dir=save_dir)
    elif system == "raptor":
        from raptor_bench_src import RaptorBenchMemory
        return RaptorBenchMemory(save_dir=save_dir)
    else:
        raise ValueError(f"Unknown system: {system}")


def needs_build(system: str) -> bool:
    return system in ("hipporag", "amem", "raptor")


def get_top_k(system: str) -> int:
    return {"simple": 5, "mem0": 5, "hipporag": 5, "amem": 10, "raptor": 10}.get(system, 10)


# ── Adaptor factory ──────────────────────────────────────────

def make_adaptor(r_mode: str, llm, mem):
    if r_mode == "r1":
        return SingleTurnAdaptor(llm, mem)
    elif r_mode == "r2":
        return IterativeAdaptor(llm, mem)
    elif r_mode == "r3":
        return PlanAndActAdaptor(llm, mem, max_expansion_steps=6)
    else:
        raise ValueError(f"Unknown r_mode: {r_mode}")


# ── Judge ────────────────────────────────────────────────────

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


# ═══════════════════════════════════════════════════════════════
# memory-probe
# ═══════════════════════════════════════════════════════════════

def run_memory_probe(system, r_modes):
    data = json.load(open("memory-probe/data/locomo10.json"))
    convs = data if isinstance(data, list) else data.get("conversations", [data])
    top_k = get_top_k(system)
    config = get_config()

    for r_mode in r_modes:
        results_path = Path(f"results_memory_probe_{system}_r{r_mode[-1]}.json")

        # Resume
        prior = []
        done_ids = set()
        if results_path.exists():
            prior = json.load(open(results_path)).get("results", [])
            done_ids = {r["conv_id"] for r in prior}

        results = list(prior)
        print(f"\n{'='*60}\nmemory-probe | {system} | {r_mode.upper()} (resume={len(done_ids)})\n{'='*60}")

        for ci, conv in enumerate(convs):
            conv_id = conv.get("sample_id", ci)
            if conv_id in done_ids:
                continue

            mem = create_memory(system, f"/tmp/bench_r123/{system}_mp_{ci}")
            llm = OpenAIClient(api_key=config.llm["api_key"], base_url=config.llm["base_url"], model=config.llm["model"])

            # Ingest
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

            if needs_build(system):
                mem.build_index()

            # Infer with adaptor
            adaptor = make_adaptor(r_mode, llm, mem)
            qas = conv.get("qa", [])
            correct = 0
            qa_results = []

            for qa in qas:
                question = qa["question"]
                reference = str(qa.get("answer", ""))
                category = qa.get("category", 0)

                result = adaptor.run(question, top_k=top_k)
                pred = result.answer
                ok = judge_answer(llm, question, pred, reference)
                if ok:
                    correct += 1
                qa_results.append({
                    "question": question[:80],
                    "category": category,
                    "correct": ok,
                    "pred": pred[:120],
                    "reference": reference[:80],
                    "steps": result.steps_taken,
                })

            results.append({
                "conv_id": conv_id,
                "n_queries": len(qas),
                "correct": correct,
                "qa_results": qa_results,
            })
            print(f"  [{len(results):2d}/{len(convs)}] conv={ci}  {correct}/{len(qas)}  steps_avg={sum(q['steps'] for q in qa_results)/len(qa_results):.1f}")

            # Incremental save
            with open(results_path, "w") as f:
                json.dump({"results": results, "system": system, "r_mode": r_mode}, f, indent=2, ensure_ascii=False)

            mem.reset()

        total_c = sum(r["correct"] for r in results)
        total_q = sum(r["n_queries"] for r in results)
        print(f"  总准确率: {total_c}/{total_q} ({total_c/total_q*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════
# StructMemEval
# ═══════════════════════════════════════════════════════════════

def run_structmemeval(system, r_modes):
    BASE = Path("StructMemEval/benchmark")
    CATEGORIES = {
        "state_machine_location": BASE / "data" / "state_machine_location",
        "tree_based": BASE / "tree_based" / "graph_configs",
        "recommendations": BASE / "recommendations" / "data",
    }
    top_k = get_top_k(system)
    config = get_config()

    tasks = []
    for cat, cat_dir in CATEGORIES.items():
        if cat_dir.exists():
            for fp in sorted(cat_dir.rglob("*.json")):
                tasks.append({"category": cat, "path": fp})

    for r_mode in r_modes:
        results_path = Path(f"results_structmemeval_{system}_r{r_mode[-1]}.json")

        prior = []
        done_keys = set()
        if results_path.exists():
            prior = json.load(open(results_path)).get("results", [])
            done_keys = {(r["category"], r["case_id"]) for r in prior if not r.get("skipped")}

        results = list(prior)
        print(f"\n{'='*60}\nStructMemEval | {system} | {r_mode.upper()} (resume={len(done_keys)})\n{'='*60}")

        for ti, task in enumerate(tasks):
            cat = task["category"]
            fp = task["path"]
            case = json.load(open(fp))
            case_id = case.get("case_id", fp.stem)
            queries = case.get("queries", [])

            if (cat, case_id) in done_keys or not queries:
                continue

            mem = create_memory(system, f"/tmp/bench_r123/{system}_sme_{ti}")
            llm = OpenAIClient(api_key=config.llm["api_key"], base_url=config.llm["base_url"], model=config.llm["model"])

            # Ingest
            for session in case.get("sessions", []):
                msgs = session.get("messages", [])
                if cat == "tree_based":
                    body = "\n".join(m["content"] for m in msgs)
                else:
                    body = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
                mem.add_memory(f"[Session: {session.get('session_id', '?')}]\n{body}")

            if needs_build(system):
                mem.build_index()

            adaptor = make_adaptor(r_mode, llm, mem)
            correct = 0
            qa_results = []

            for q in queries:
                result = adaptor.run(q["question"], top_k=top_k)
                pred = result.answer
                ok = judge_answer(llm, q["question"], pred, q.get("reference_answer", {}), cat)
                if ok:
                    correct += 1
                qa_results.append({"question": q["question"][:80], "correct": ok, "pred": pred[:120], "steps": result.steps_taken})

            results.append({
                "case_id": case_id, "category": cat,
                "n_queries": len(queries), "correct": correct,
                "qa_results": qa_results,
            })
            print(f"  [{len(results):3d}/{len(tasks)}] {cat[:20]:20s} {case_id[:25]:25s} {correct}/{len(queries)}")

            with open(results_path, "w") as f:
                json.dump({"results": results, "system": system, "r_mode": r_mode}, f, indent=2, ensure_ascii=False)

            mem.reset()

        valid = [r for r in results if not r.get("skipped")]
        total_c = sum(r["correct"] for r in valid)
        total_q = sum(r["n_queries"] for r in valid)
        if total_q:
            print(f"  总准确率: {total_c}/{total_q} ({total_c/total_q*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════
# AMemGym (on-policy)
# ═══════════════════════════════════════════════════════════════

def find_correct_index(qa, period_state):
    required_info = qa.get("required_info", [])
    correct_state = [period_state.get(k, "") for k in required_info]
    for i, choice in enumerate(qa.get("answer_choices", [])):
        if choice.get("state") == correct_state:
            return i
    return 0


def run_amemgym(system, r_modes, start=0, end=None):
    data = json.load(open("data/amemgym/v1.base/data.json"))
    all_users = data if isinstance(data, list) else list(data.values())
    end = end or len(all_users)
    users = list(enumerate(all_users))[start:end]
    top_k = get_top_k(system)
    config = get_config()
    is_incremental = system in ("simple", "mem0")

    for r_mode in r_modes:
        suffix = f"_{start}_{end}" if start > 0 or end < len(all_users) else ""
        results_path = Path(f"results_amemgym_{system}_r{r_mode[-1]}{suffix}.json")

        print(f"\n{'='*60}\nAMemGym on-policy | {system} | {r_mode.upper()} [{start},{end})\n{'='*60}")

        results = []

        for uid, user in users:
            save_dir = f"/tmp/bench_r123/{system}_ag_{uid}"
            periods = user.get("periods", [])
            qas = user.get("qas", [])

            total_correct, total_evals = 0, 0
            period_results = []

            if is_incremental:
                mem = create_memory(system, save_dir)

            for pi in range(len(periods)):
                period_state = periods[pi]["state"]

                if is_incremental:
                    # Add only this period
                    for session in periods[pi].get("sessions", []):
                        query = session.get("query", "")
                        exposed = session.get("exposed_states", {})
                        session_time = session.get("session_time", "")
                        if not query: continue
                        parts = [f"[Time: {session_time}]", f"User query: {query}"]
                        if exposed:
                            parts.append("States: " + ", ".join(f"{k}={v}" for k, v in exposed.items()))
                        mem.add_memory("\n".join(parts))
                else:
                    # Cumulative rebuild
                    mem = create_memory(system, save_dir)
                    for p in periods[:pi+1]:
                        for session in p.get("sessions", []):
                            query = session.get("query", "")
                            exposed = session.get("exposed_states", {})
                            session_time = session.get("session_time", "")
                            if not query: continue
                            parts = [f"[Time: {session_time}]", f"User query: {query}"]
                            if exposed:
                                parts.append("States: " + ", ".join(f"{k}={v}" for k, v in exposed.items()))
                            mem.add_memory("\n".join(parts))
                    if needs_build(system):
                        mem.build_index()

                # Eval with adaptor
                llm = OpenAIClient(api_key=config.llm["api_key"], base_url=config.llm["base_url"], model=config.llm["model"])
                adaptor = make_adaptor(r_mode, llm, mem)
                period_correct = 0

                for qa in qas:
                    correct_idx = find_correct_index(qa, period_state)
                    choices = qa.get("answer_choices", [])
                    choices_str = "\n".join(f"  ({i}) {c.get('text','')}" for i, c in enumerate(choices))
                    task = f"Question: {qa['query']}\n\nChoices:\n{choices_str}\n\nReply with ONLY the choice number."

                    result = adaptor.run(task, top_k=top_k)
                    m = re.search(r'\d+', result.answer)
                    pred_idx = int(m.group()) if m and 0 <= int(m.group()) < len(choices) else 0
                    if pred_idx == correct_idx:
                        period_correct += 1

                total_correct += period_correct
                total_evals += len(qas)
                period_results.append({"period_idx": pi, "correct": period_correct})

                if not is_incremental:
                    mem.reset()

            if is_incremental:
                mem.reset()

            acc = total_correct / total_evals * 100 if total_evals else 0
            per_p = " ".join(str(pr["correct"]) for pr in period_results)
            print(f"  [{uid+1:2d}] user={uid}  {total_correct}/{total_evals} ({acc:.1f}%)  [{per_p}]")

            results.append({
                "user_id": user.get("id", str(uid)),
                "total_correct": total_correct,
                "total_evals": total_evals,
                "period_results": period_results,
            })

            # Incremental save
            with open(results_path, "w") as f:
                json.dump({"results": results, "system": system, "r_mode": r_mode}, f, indent=2, ensure_ascii=False)

        total_c = sum(r["total_correct"] for r in results)
        total_e = sum(r["total_evals"] for r in results)
        if total_e:
            print(f"  总准确率: {total_c}/{total_e} ({total_c/total_e*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════

BENCH_MAP = {
    "memory_probe": run_memory_probe,
    "structmemeval": run_structmemeval,
    "amemgym": run_amemgym,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True, choices=["simple", "mem0", "hipporag", "amem", "raptor"])
    parser.add_argument("--bench", required=True, help="memory_probe|structmemeval|amemgym|all")
    parser.add_argument("--r", default="r1,r2,r3", help="Comma-separated: r1,r2,r3")
    parser.add_argument("--start", type=int, default=0, help="AMemGym start user")
    parser.add_argument("--end", type=int, default=None, help="AMemGym end user")
    args = parser.parse_args()

    r_modes = [r.strip() for r in args.r.split(",")]

    if args.bench == "all":
        benches = ["memory_probe", "structmemeval", "amemgym"]
    else:
        benches = [args.bench]

    for bench in benches:
        fn = BENCH_MAP[bench]
        if bench == "amemgym":
            fn(args.system, r_modes, start=args.start, end=args.end)
        else:
            fn(args.system, r_modes)


if __name__ == "__main__":
    main()
