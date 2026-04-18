# m_axis/src/bootstrap_loader.py
from __future__ import annotations
import json
from pathlib import Path

def load_bootstrap_state(path: str | None = None) -> dict:
    p = Path(path) if path else Path("m_axis/state/bootstrap_state.json")
    if not p.exists():
        raise FileNotFoundError(
            f"bootstrap_state.json not found at {p.resolve()}\n"
            f"请把第三步输出的 agent_id 写入该文件。"
        )
    return json.loads(p.read_text(encoding="utf-8"))
