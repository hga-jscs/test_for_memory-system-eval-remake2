#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AMemGym on-policy 全量评测（通用版）

论文要求：每个 period 结束后用该 period 的 state 评测全部 QA。
11 periods × 10 QAs = 110 次评测/user。

用法:
  python bench_amemgym_onpolicy.py --system raptor
  python bench_amemgym_onpolicy.py --system simple
  python bench_amemgym_onpolicy.py --system mem0
  python bench_amemgym_onpolicy.py --system hipporag
  python bench_amemgym_onpolicy.py --system amem

两种 ingest 模式:
  - incremental (simple/mem0): 每 period 追加新数据，不重建
  - cumulative  (hipporag/raptor/amem): 每 period 重建全部数据
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

DATA_PATH = Path("data/amemgym/v1.base/data.json")

# ── System registry ──────────────────────────────────────────────

SYSTEM_CONFIG = {
    "simple": {
        "label":   "SimpleMem",
        "mode":    "incremental",  # add_memory 立即生效
        "top_k":   5,
        "results": "results_amemgym_onpolicy_simple.json",
        "save_base": "/tmp/bench_onpolicy_simple",
    },
    "mem0": {
        "label":   "mem0",
        "mode":    "incremental",
        "top_k":   5,
        "results": "results_amemgym_onpolicy_mem0.json",
        "save_base": "/tmp/bench_onpolicy_mem0",
    },
    "hipporag": {
        "label":   "HippoRAG",
        "mode":    "cumulative",   # 每 period 需要重建 graph
        "top_k":   5,
        "results": "results_amemgym_onpolicy_hipporag.json",
        "save_base": "/tmp/bench_onpolicy_hipporag",
    },
    "amem": {
        "label":   "A-MEM",
        "mode":    "cumulative",   # 每 period 需要重建（evolution 依赖全局 memory）
        "top_k":   10,
        "results": "results_amemgym_onpolicy_amem.json",
        "save_base": "/tmp/bench_onpolicy_amem",
    },
    "raptor": {
        "label":   "RAPTOR",
        "mode":    "cumulative",   # 每 period 需要重建 tree
        "top_k":   10,
        "results": "results_amemgym_onpolicy_raptor.json",
        "save_base": "/tmp/bench_onpolicy_raptor",
    },
    "mem0g": {
        "label":   "mem0g",
        "mode":    "incremental",  # add_memory 立即生效（含 graph extraction）
        "top_k":   10,
        "results": "results_amemgym_onpolicy_mem0g.json",
        "save_base": "/tmp/bench_onpolicy_mem0g",
    },
}


def create_memory(system: str, save_dir: str):
    """Factory: create memory instance for the given system."""
    if system == "simple":
        from simpleMem_src import SimpleRAGMemory
        return SimpleRAGMemory()
    elif system == "mem0":
        from mem0_bench_src import Mem0RAGMemory
        # collection_name 用 save_dir 的 basename 保证唯一
        cname = Path(save_dir).name
        return Mem0RAGMemory(collection_name=cname)
    elif system == "hipporag":
        from hipporag_bench_src import HippoRAGMemory
        return HippoRAGMemory(save_dir=save_dir)
    elif system == "amem":
        from amem_bench_src import AMemBenchMemory
        return AMemBenchMemory(save_dir=save_dir)
    elif system == "raptor":
        from raptor_bench_src import RaptorBenchMemory
        return RaptorBenchMemory(save_dir=save_dir)
    elif system == "mem0g":
        from mem0g_bench_src import Mem0GMemory
        cname = Path(save_dir).name
        return Mem0GMemory(collection_name=cname)
    else:
        raise ValueError(f"Unknown system: {system}")


def needs_build(system: str) -> bool:
    """Does this system require explicit build_index() after adding data?"""
    return system in ("hipporag", "amem", "raptor")


def ingest_periods(mem, user, up_to_period: int):
    """Ingest periods 0..up_to_period (inclusive) into memory."""
    for pi in range(up_to_period + 1):
        period = user["periods"][pi]
        period_start = period.get("period_start", "?")
        period_end   = period.get("period_end", "?")
        for session in period.get("sessions", []):
            query        = session.get("query", "")
            exposed      = session.get("exposed_states", {})
            session_time = session.get("session_time", "")
            if not query:
                continue
            parts = [f"[Time: {session_time}, Period: {period_start}~{period_end}]",
                     f"User query: {query}"]
            if exposed:
                parts.append("States: " + ", ".join(f"{k}={v}" for k, v in exposed.items()))
            text = "\n".join(parts)
            if hasattr(mem, 'add_memory'):
                mem.add_memory(text)
            elif hasattr(mem, 'add_text'):
                mem.add_text(text)


def ingest_single_period(mem, user, period_idx: int):
    """Ingest only one period's sessions (for incremental mode)."""
    period = user["periods"][period_idx]
    period_start = period.get("period_start", "?")
    period_end   = period.get("period_end", "?")
    for session in period.get("sessions", []):
        query        = session.get("query", "")
        exposed      = session.get("exposed_states", {})
        session_time = session.get("session_time", "")
        if not query:
            continue
        parts = [f"[Time: {session_time}, Period: {period_start}~{period_end}]",
                 f"User query: {query}"]
        if exposed:
            parts.append("States: " + ", ".join(f"{k}={v}" for k, v in exposed.items()))
        text = "\n".join(parts)
        if hasattr(mem, 'add_memory'):
            mem.add_memory(text)
        elif hasattr(mem, 'add_text'):
            mem.add_text(text)


def find_correct_index(qa, period_state):
    """Find correct answer index using the given period's state."""
    required_info = qa.get("required_info", [])
    correct_state = [period_state.get(k, "") for k in required_info]
    for i, choice in enumerate(qa.get("answer_choices", [])):
        if choice.get("state") == correct_state:
            return i
    return 0


def answer_multichoice(llm, mem, qa, top_k):
    question = qa["query"]
    choices  = qa.get("answer_choices", [])
    evidences = mem.retrieve(question, top_k=top_k)
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


def eval_user(user, uid, system, cfg):
    config  = get_config()
    save_dir = str(Path(cfg["save_base"]) / f"user_{uid:03d}")
    top_k   = cfg["top_k"]
    mode    = cfg["mode"]

    llm = OpenAIClient(
        api_key=config.llm["api_key"],
        base_url=config.llm["base_url"],
        model=config.llm["model"],
    )

    periods = user.get("periods", [])
    qas     = user.get("qas", [])
    n_periods = len(periods)

    all_period_results = []
    total_correct = 0
    total_evals   = 0
    t_user_start  = time.time()

    if mode == "incremental":
        # Create memory once, add data incrementally
        mem = create_memory(system, save_dir)

    for pi in range(n_periods):
        period = periods[pi]
        period_state = period["state"]

        if mode == "incremental":
            # Add only this period's data
            ingest_single_period(mem, user, pi)
        else:
            # Cumulative: reset and rebuild from scratch
            mem = create_memory(system, save_dir)
            ingest_periods(mem, user, pi)
            if needs_build(system):
                mem.build_index()

        # Eval all QAs against this period's state
        period_correct = 0
        for qa in qas:
            correct_idx = find_correct_index(qa, period_state)
            pred_idx    = answer_multichoice(llm, mem, qa, top_k)
            if pred_idx == correct_idx:
                period_correct += 1

        total_correct += period_correct
        total_evals   += len(qas)
        all_period_results.append({
            "period_idx": pi,
            "correct":    period_correct,
            "n_queries":  len(qas),
        })

        if mode == "cumulative":
            mem.reset()

    if mode == "incremental":
        mem.reset()

    wall_s = time.time() - t_user_start
    acc = total_correct / total_evals * 100 if total_evals else 0

    return {
        "user_id":       user.get("id", str(uid)),
        "n_periods":     n_periods,
        "n_queries_per_period": len(qas),
        "total_evals":   total_evals,
        "total_correct": total_correct,
        "accuracy":      round(acc, 2),
        "wall_s":        round(wall_s, 1),
        "period_results": all_period_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True, choices=list(SYSTEM_CONFIG.keys()))
    parser.add_argument("--start", type=int, default=0, help="Start user index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End user index (exclusive)")
    args = parser.parse_args()

    system = args.system
    cfg    = SYSTEM_CONFIG[system]
    suffix = f"_{args.start}_{args.end}" if args.start > 0 or args.end else ""
    results_path = Path(cfg["results"].replace(".json", f"{suffix}.json"))

    Path(cfg["save_base"]).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"AMemGym ON-POLICY 评测 ({cfg['label']}, top_k={cfg['top_k']})")
    print("=" * 70)

    data  = json.load(open(DATA_PATH))
    all_users = data if isinstance(data, list) else list(data.values())
    start = args.start
    end = args.end if args.end is not None else len(all_users)
    users = list(enumerate(all_users))[start:end]
    print(f"  用户范围: [{start}, {end}) = {len(users)} users | mode={cfg['mode']}")
    print(f"  每 user: {len(all_users[0]['periods'])} periods × {len(all_users[0]['qas'])} QA "
          f"= {len(all_users[0]['periods']) * len(all_users[0]['qas'])} evals\n")

    results, errors = [], []
    t_wall_start = time.time()

    for uid, user in users:
        try:
            r = eval_user(user, uid, system, cfg)
            results.append(r)

            # Per-period breakdown (compact)
            per_p = " ".join(f"{pr['correct']}" for pr in r["period_results"])
            print(
                f"[{uid+1:2d}/{len(users)}] user={uid:2d}  "
                f"{r['total_correct']}/{r['total_evals']} ({r['accuracy']:.1f}%)  "
                f"wall={r['wall_s']:.0f}s  "
                f"per_period=[{per_p}]"
            )
        except Exception as e:
            errors.append({"user_id": uid, "error": str(e)})
            print(f"[{uid+1:2d}/{len(users)}] ERROR user={uid}: {e}")
            import traceback; traceback.print_exc()

    wall_ms = (time.time() - t_wall_start) * 1000

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    total_evals   = sum(r["total_evals"] for r in results)
    total_correct = sum(r["total_correct"] for r in results)
    acc = total_correct / total_evals * 100 if total_evals else 0

    # Per-period aggregate
    n_periods = results[0]["n_periods"] if results else 0
    for pi in range(n_periods):
        pc = sum(r["period_results"][pi]["correct"] for r in results)
        pt = sum(r["period_results"][pi]["n_queries"] for r in results)
        print(f"  Period {pi:2d}: {pc}/{pt} ({pc/pt*100:.1f}%)")

    print(f"\n  总准确率 : {total_correct}/{total_evals} ({acc:.1f}%)")
    print(f"  Wall    : {wall_ms:.0f}ms ({wall_ms/1000/60:.1f}min)")
    print(f"  Errors  : {len(errors)}")
    print("=" * 70)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "errors": errors,
                   "system": cfg["label"], "mode": "on-policy"},
                  f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {results_path}")


if __name__ == "__main__":
    main()
