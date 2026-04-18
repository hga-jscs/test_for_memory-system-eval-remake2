# m_axis/src/interface.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .types import MemoryHit


@runtime_checkable
class MemoryInterface(Protocol):
    """
    统一 M 轴接口（Adapter 层对外契约）

    最小必需：
      - retrieve(query)->str : 给 R 轴 workflow 直接拼 prompt 用
      - memorize(text, kind, tags)->Any : 注入/更新记忆（benchmark 造数据、冲突修正任务会用）

    强烈建议：
      - retrieve_structured(query)->List[MemoryHit] : 白盒归因 / 记忆层指标用
    """

    async def retrieve(self, query: str) -> str:
        ...

    async def memorize(
        self,
        text: str,
        *,
        kind: str = "archival",
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Any:
        ...

    async def retrieve_structured(
        self,
        query: str,
        *,
        k: int = 8,
        include_core: bool = True,
    ) -> List[MemoryHit]:
        ...
