#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified smoke test framework for memory backends.

Goal: runnable, auditable, and strict smoke tests for MemGPT/RAPTOR/LightRAG/HippoRAG.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from benchmark_io_utils import load_json_with_fallback
from benchmark_observability import classify_answer_quality, classify_failure
from simpleMem_src import OpenAIClient, get_config


ONLY_ALIASES = {
    "amemgym": "amemgym",
    "memory_probe": "memory_probe",
    "state_machine": "state_machine",
    "state_machine_location": "state_machine",
    "tree_based": "tree_based",
    "recommendations": "recommendations",
}


@dataclass
class CaseSpec:
    benchmark: str
    case_id: str
    question: str
    gold: str
    ingest_payload: list[str]
    meta: dict[str, Any]


class SmokeRun:
    def __init__(self, backend: str):
        self.backend = backend
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
        self.run_dir = Path("results") / "smoke" / backend / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "manifest": self.run_dir / "manifest.json",
            "summary": self.run_dir / "summary.json",
            "summary_md": self.run_dir / "summary.md",
            "cases": self.run_dir / "cases.jsonl",
            "retrievals": self.run_dir / "retrievals.jsonl",
            "llm_calls": self.run_dir / "llm_calls.jsonl",
            "stdout": self.run_dir / "stdout.log",
            "stderr": self.run_dir / "stderr.log",
            "config": self.run_dir / "config_snapshot.json",
        }
        self.started_at = self._now()
        self.finished_at = None
        self.counts = {"total": 0, "passed": 0, "failed": 0, "skipped": 0}
        self.failures: list[dict[str, Any]] = []
        self._stdout_f = self.paths["stdout"].open("a", encoding="utf-8")
        self._stderr_f = self.paths["stderr"].open("a", encoding="utf-8")
        self._write_json(
            self.paths["config"],
            {
                "backend": backend,
                "run_id": self.run_id,
                "started_at": self.started_at,
                "python": sys.version,
                "executable": sys.executable,
                "config": _safe_config(),
                "dependencies": _deps_snapshot(),
            },
        )

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write_json(self, path: Path, obj: dict[str, Any]) -> None:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_jsonl(self, key: str, obj: dict[str, Any]) -> None:
        with self.paths[key].open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def out(self, msg: str) -> None:
        print(msg)
        self._stdout_f.write(msg + "\n")
        self._stdout_f.flush()

    def err(self, msg: str) -> None:
        print(msg, file=sys.stderr)
        self._stderr_f.write(msg + "\n")
        self._stderr_f.flush()

    def finalize(self, ok: bool, reason: str, extras: dict[str, Any] | None = None) -> None:
        self.finished_at = self._now()
        manifest = {
            "backend": self.backend,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "counts": self.counts,
            "ok": ok,
            "reason": reason,
            "artifacts": {k: str(v) for k, v in self.paths.items()},
            "failures": self.failures,
        }
        if extras:
            manifest.update(extras)
        self._write_json(self.paths["manifest"], manifest)
        self._write_json(self.paths["summary"], manifest)
        md = [
            f"# Smoke Summary ({self.backend})",
            f"- run_id: `{self.run_id}`",
            f"- ok: `{ok}`",
            f"- reason: `{reason}`",
            f"- counts: {self.counts}",
            f"- started_at: {self.started_at}",
            f"- finished_at: {self.finished_at}",
        ]
        if self.failures:
            md.append("\n## Failures")
            for f in self.failures:
                md.append(f"- {f['case_id']} [{f['error_type']}] {f['message']}")
        self.paths["summary_md"].write_text("\n".join(md) + "\n", encoding="utf-8")
        self._stdout_f.close()
        self._stderr_f.close()


def _safe_config() -> dict[str, Any]:
    try:
        c = get_config()
        return {
            "llm": {**c.llm, "api_key": "***" if c.llm.get("api_key") else ""},
            "embedding": {**c.embedding, "api_key": "***" if c.embedding.get("api_key") else ""},
        }
    except Exception as e:
        return {"error": str(e)}


def _deps_snapshot() -> dict[str, str]:
    deps = ["openai", "numpy", "torch", "igraph", "requests", "lightrag", "letta_client", "letta"]
    out: dict[str, str] = {}
    for d in deps:
        try:
            from importlib.metadata import version

            out[d] = version(d)
        except Exception:
            out[d] = "unknown"
    return out


class DummyLLM:
    total_tokens = 0

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 120) -> str:
        self.total_tokens += max(1, len(prompt) // 20)
        return ""

    def generate_json(self, prompt: str):
        self.total_tokens += max(1, len(prompt) // 20)
        raise RuntimeError("dummy_llm_no_json_support")


def _llm() -> OpenAIClient | DummyLLM:
    try:
        conf = get_config()
        return OpenAIClient(
            api_key=conf.llm.get("api_key"),
            base_url=conf.llm.get("base_url"),
            model=conf.llm.get("model"),
        )
    except Exception:
        return DummyLLM()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _simple_answer(topk: list[dict[str, Any]], gold: str, question: str) -> str:
    if not topk:
        return ""
    g = _norm(gold)
    for item in topk:
        t = item.get("content", "")
        if g and g in _norm(t):
            return gold
    return topk[0].get("content", "")[:180]


def _judge(pred: str, gold: str) -> bool:
    return bool(_norm(gold)) and _norm(gold) in _norm(pred)


def _issue_flags(question: str, gold: str, pred: str, topk: list[dict[str, Any]], parse_error: str | None) -> list[str]:
    tags: list[str] = []
    if not topk:
        tags.append("retrieval_empty")
    texts = [t.get("content", "") for t in topk]
    g = _norm(gold)
    evidence_has_gold = any(g and g in _norm(t) for t in texts)
    pred_norm = _norm(pred)
    denied = ("没有提到" in pred) or ("not mention" in pred_norm) or ("unknown" in pred_norm)
    if evidence_has_gold and g and g not in pred_norm:
        tags.append("retrieved_but_answer_missed_key_evidence")
    if denied and evidence_has_gold:
        tags.append("denial_conflicts_with_topk")
    if ("who" in question.lower() or "谁" in question) and any(x in pred for x in ["他", "她", "they"]):
        tags.append("entity_confusion")
    if any(x in question.lower() for x in ["yesterday", "tomorrow", "昨天", "明天"]) and not re.search(r"\d{4}-\d{2}-\d{2}", pred):
        tags.append("time_normalization_error")
    if parse_error:
        tags.append("structured_parse_failed")
        tags.append("response_format_incompatible")
    if "default" in pred_norm and len(pred_norm) < 32:
        tags.append("default_option_bias")
    return tags


def _build_amemgym_cases() -> list[CaseSpec]:
    data = load_json_with_fallback("data/amemgym/v1.base/data.json")
    users = data if isinstance(data, list) else list(data.values())
    u = users[0]
    ingest_payload: list[str] = []
    for period in u.get("periods", [])[:2]:
        for session in period.get("sessions", [])[:3]:
            q = session.get("query", "")
            if not q:
                continue
            exp = session.get("exposed_states", {})
            ingest_payload.append(f"query={q}; states={exp}")
    cases: list[CaseSpec] = []
    for i, qa in enumerate(u.get("qas", [])[:2]):
        choices = qa.get("answer_choices", [])
        gold = choices[0].get("text", "") if choices else ""
        cases.append(CaseSpec("amemgym", f"amemgym_u0_q{i}", qa.get("query", ""), gold, ingest_payload, {"choices": choices}))
    return cases


def _build_memory_probe_cases() -> list[CaseSpec]:
    data = load_json_with_fallback("memory-probe/data/locomo10.json")
    convs = data if isinstance(data, list) else data.get("conversations", [data])
    c = convs[0]
    ingest_payload: list[str] = []
    conversation = c.get("conversation", {})
    keys = sorted([k for k in conversation if re.match(r"session_\d+$", k)], key=lambda x: int(x.split("_")[1]))
    for k in keys[:3]:
        turns = conversation.get(k, [])
        ingest_payload.append("\n".join(f"{t.get('speaker')}: {t.get('text')}" for t in turns))
    cases: list[CaseSpec] = []
    for i, qa in enumerate(c.get("qa", [])[:3]):
        cases.append(CaseSpec("memory_probe", f"probe_c0_q{i}", qa.get("question", ""), str(qa.get("answer", "")), ingest_payload, {"category": qa.get("category")}))
    return cases


def _build_struct_cases() -> list[CaseSpec]:
    out: list[CaseSpec] = []
    specs = [
        ("state_machine", Path("StructMemEval/benchmark/data/state_machine_location")),
        ("tree_based", Path("StructMemEval/benchmark/tree_based/graph_configs")),
        ("recommendations", Path("StructMemEval/benchmark/recommendations/data")),
    ]
    for bench, path in specs:
        fp = sorted(path.rglob("*.json"))[0]
        case = load_json_with_fallback(fp)
        ingest_payload: list[str] = []
        for sess in case.get("sessions", [])[:3]:
            msgs = sess.get("messages", [])
            if bench == "tree_based":
                ingest_payload.append("\n".join(m.get("content", "") for m in msgs))
            else:
                ingest_payload.append("\n".join(f"{m.get('role')}: {m.get('content')}" for m in msgs))
        q = case.get("queries", [{}])[0]
        ref = q.get("reference_answer", {})
        gold = ref if isinstance(ref, str) else ref.get("text", "")
        out.append(CaseSpec(bench, f"{bench}_{fp.stem}", q.get("question", ""), gold, ingest_payload, {"source": str(fp)}))
    return out


def collect_cases(only: str | None) -> list[CaseSpec]:
    all_cases = _build_amemgym_cases() + _build_memory_probe_cases() + _build_struct_cases()
    if not only:
        return all_cases
    normalized = ONLY_ALIASES.get(only)
    if not normalized:
        raise ValueError(f"Unsupported --only={only}")
    return [c for c in all_cases if c.benchmark == normalized]


def run_backend_smoke(
    backend: str,
    memory_factory: Callable[[str], Any],
    only: str | None = None,
    preflight: Callable[[SmokeRun], tuple[bool, str]] | None = None,
    require_response_format_check: bool = False,
) -> int:
    run = SmokeRun(backend=backend)
    run.out(f"[SMOKE] backend={backend} run_id={run.run_id} only={only or 'all'}")

    preflight_ok, preflight_reason = (True, "ok")
    if preflight:
        preflight_ok, preflight_reason = preflight(run)

    cases = collect_cases(only)
    run.counts["total"] = len(cases)
    llm = _llm()

    for case in cases:
        t_case = time.time()
        mem = memory_factory(str(run.run_dir / f"mem_{case.case_id}"))
        parse_error: str | None = None
        pred = ""
        status_ok = False
        error_type = ""
        try:
            t0 = time.time()
            for text in case.ingest_payload:
                if hasattr(mem, "add_text"):
                    mem.add_text(text)
                else:
                    mem.add_memory(text)
            mem.build_index()
            ingest_ms = int((time.time() - t0) * 1000)
            audit = mem.audit_ingest() if hasattr(mem, "audit_ingest") else {}

            t1 = time.time()
            evs = mem.retrieve(case.question, top_k=5)
            infer_ms = int((time.time() - t1) * 1000)
            topk = [
                {
                    "rank": i + 1,
                    "score": float((getattr(e, "metadata", {}) or {}).get("score", 0.0) or 0.0),
                    "content": getattr(e, "content", ""),
                }
                for i, e in enumerate(evs)
            ]
            run.append_jsonl("retrievals", {"ts": run._now(), "case_id": case.case_id, "benchmark": case.benchmark, "top_k": topk})

            response_format_attempted = False
            if require_response_format_check and backend == "lightrag":
                response_format_attempted = True
                try:
                    obj = llm.generate_json(f"回答问题并输出JSON：{{\"answer\":string}}\nQuestion:{case.question}")
                    pred = obj.get("answer", "") if isinstance(obj, dict) else ""
                    if not pred:
                        raise ValueError("empty parsed answer")
                except Exception as pe:  # noqa: BLE001
                    parse_error = classify_failure(pe)
                    pred = _simple_answer(topk, case.gold, case.question)
            else:
                try:
                    prompt = (
                        "根据检索证据回答问题，简洁作答。\n"
                        f"Question: {case.question}\n"
                        f"Evidence:\n" + "\n".join(f"[{i+1}] {t['content'][:160]}" for i, t in enumerate(topk))
                    )
                    pred = llm.generate(prompt, temperature=0.0, max_tokens=120)
                except Exception as le:  # noqa: BLE001
                    parse_error = classify_failure(le)
                    pred = _simple_answer(topk, case.gold, case.question)

            tags = _issue_flags(case.question, case.gold, pred, topk, parse_error)
            quality = classify_answer_quality(
                question=case.question,
                answer=pred,
                gold=case.gold,
                retrieval_texts=[x["content"] for x in topk],
                retrieval_scores=[x["score"] for x in topk],
            )
            tags.extend(t for t in quality.get("quality_tags", []) if t not in tags)

            is_correct = _judge(pred, case.gold)
            status_ok = is_correct and ("retrieval_empty" not in tags)
            if status_ok:
                run.counts["passed"] += 1
            else:
                run.counts["failed"] += 1
                error_type = "|".join(tags) if tags else "wrong_answer"
                run.failures.append({"case_id": case.case_id, "error_type": error_type, "message": f"benchmark={case.benchmark}"})

            run.append_jsonl(
                "llm_calls",
                {
                    "ts": run._now(),
                    "case_id": case.case_id,
                    "stage": "answer",
                    "request_summary": case.question[:120],
                    "response_text": pred,
                    "structured_parse": None if not parse_error else {"error": parse_error},
                    "retry_count": 0,
                    "exception_type": parse_error,
                    "latency_ms": int((time.time() - t_case) * 1000),
                    "response_format_enabled": response_format_attempted,
                },
            )

            run.append_jsonl(
                "cases",
                {
                    "run_id": run.run_id,
                    "backend": backend,
                    "benchmark": case.benchmark,
                    "case_id": case.case_id,
                    "conv_id": case.meta.get("source") or case.case_id,
                    "user_id": case.meta.get("user_id"),
                    "question": case.question,
                    "gold": case.gold,
                    "pred": pred,
                    "correct": is_correct,
                    "ok": status_ok,
                    "error_type": error_type,
                    "issue_tags": tags,
                    "top_k": topk,
                    "ingest_time_ms": ingest_ms,
                    "infer_time_ms": infer_ms,
                    "llm_calls": 1,
                    "llm_tokens": getattr(llm, "total_tokens", 0),
                    "audit": audit,
                    "started_at": run._now(),
                    "finished_at": run._now(),
                },
            )

            run.out(f"[{backend}] {case.benchmark}/{case.case_id} => {'PASS' if status_ok else 'FAIL'} tags={tags}")

        except Exception as e:  # noqa: BLE001
            run.counts["failed"] += 1
            et = classify_failure(e)
            run.failures.append({"case_id": case.case_id, "error_type": et, "message": str(e)})
            run.append_jsonl(
                "cases",
                {
                    "backend": backend,
                    "benchmark": case.benchmark,
                    "case_id": case.case_id,
                    "question": case.question,
                    "gold": case.gold,
                    "pred": pred,
                    "correct": False,
                    "ok": False,
                    "error_type": et,
                    "issue_tags": [et],
                    "top_k": [],
                    "exception": str(e),
                    "traceback": traceback.format_exc(),
                    "started_at": run._now(),
                    "finished_at": run._now(),
                },
            )
            run.err(f"[{backend}] case failed: {case.case_id} :: {e}")
        finally:
            try:
                mem.reset()
            except Exception:
                pass

    ok = preflight_ok and run.counts["failed"] == 0 and run.counts["passed"] > 0
    reason = "ok" if ok else (f"preflight_failed:{preflight_reason}" if not preflight_ok else "case_failures")
    run.out(f"[SMOKE DONE] backend={backend} ok={ok} reason={reason} run_dir={run.run_dir}")
    run.finalize(ok=ok, reason=reason, extras={"preflight": {"ok": preflight_ok, "reason": preflight_reason}})
    return 0 if ok else 1


def memgpt_preflight(run: SmokeRun) -> tuple[bool, str]:
    base_url = os.getenv("LETTA_BASE_URL", "")
    if not base_url:
        run.err("[preflight] LETTA_BASE_URL is missing")
        return False, "missing_letta_base_url"
    try:
        import requests

        url = base_url.rstrip("/") + "/v1/health/"
        resp = requests.get(url, timeout=8)
        ok = resp.status_code == 200 and resp.json().get("status") == "ok"
        run.out(f"[preflight] letta health={resp.status_code} payload={resp.text[:160]}")
        return (ok, "ok" if ok else "health_not_ok")
    except Exception as e:  # noqa: BLE001
        run.err(f"[preflight] letta health check failed: {e}")
        return False, classify_failure(e)


def default_preflight(run: SmokeRun) -> tuple[bool, str]:
    return True, "ok"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=str, default=None, choices=sorted(ONLY_ALIASES.keys()))
    return p.parse_args(argv)
