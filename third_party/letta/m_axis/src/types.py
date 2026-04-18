# m_axis/src/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

MemorySource = Literal["core", "recall", "archival"]


@dataclass(frozen=True)
class MemoryHit:
    """
    结构化检索结果（白盒评测/归因用）

    - source: 来自 core/recall/archival 哪一层
    - text: 证据文本（建议已经做过截断，避免爆上下文）
    - score: 检索分数（None 代表该层不提供或未计算）
    - created_at_iso: ISO 时间字符串（冲突修正任务常用：越新越可信）
    - meta: 额外字段（tags、id、role、distance、doc_id、chunk_id...）
    """
    source: MemorySource
    text: str
    score: Optional[float] = None
    created_at_iso: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
