# m_axis/src/letta_adapter.py
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore
    except Exception:
        pass

from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import TextContent
from letta.schemas.message import MessageCreate
from letta.schemas.memory import Memory
from letta.schemas.block import Block, BlockUpdate

try:
    from letta.schemas.agent import UpdateAgent  # 你这版是 UpdateAgent
except Exception:
    from letta.schemas.agent import AgentUpdate as UpdateAgent  # type: ignore

from letta.helpers.message_helper import convert_message_creates_to_messages
from letta.services.agent_manager import AgentManager
from letta.services.message_manager import MessageManager
from letta.services.passage_manager import PassageManager
from letta.services.block_manager import BlockManager


MemorySource = Literal["core", "recall", "archival"]


@dataclass(frozen=True)
class MemoryHit:
    source: MemorySource
    text: str
    score: Optional[float] = None
    created_at_iso: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class LettaMemoryAdapter:
    """
    第四步：最稳妥 Letta/MemGPT M轴适配器（轻量封装）
    - 只做存/取
    - core 写入：直接更新 block.value（不走 update_memory_if_changed_async，避免 message_ids=None）
    - recall 检索：先走 search_messages_async；若命中为空（常见于中文FTS），fallback 到最近消息 + 子串匹配
    - archival 检索：先走 search_agent_archival_memory_async；若命中为空，fallback 到 query_agent_passages_async + 子串匹配
    """

    def __init__(
        self,
        *,
        actor: Any,
        agent_id: str,
        k_recall: int = 8,
        k_archival: int = 8,
        recall_search_mode: str = "fts",  # 建议默认 fts；但中文可能空，会自动 fallback
        include_core_snapshot: bool = True,
    ):
        self.actor = actor
        self.agent_id = agent_id
        self.k_recall = k_recall
        self.k_archival = k_archival
        self.recall_search_mode = recall_search_mode
        self.include_core_snapshot = include_core_snapshot

        self.agent_manager = AgentManager()
        self.message_manager = MessageManager()
        self.passage_manager = PassageManager()
        self.block_manager = BlockManager()

    async def memorize(
        self,
        text: str,
        *,
        kind: str = "archival",
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Any:
        kind = kind.strip().lower()
        tags = tags or []
        meta = meta or {}

        if kind == "recall":
            return await self._memorize_recall(text)
        if kind == "archival":
            return await self._memorize_archival(text, tags=tags)
        if kind == "core":
            return await self._memorize_core(text)

        raise ValueError(f"Unknown kind={kind}. Use recall|archival|core")

    async def retrieve(self, query: str) -> str:
        core_txt = ""
        if self.include_core_snapshot:
            core_txt = await self._get_core_snapshot()

        recall_hits = await self._retrieve_recall_hits(query, limit=self.k_recall)
        archival_hits = await self._retrieve_archival_hits(query, limit=self.k_archival)

        out: List[str] = []
        if core_txt.strip():
            out.append("[CORE]")
            out.append(core_txt.strip())
            out.append("")

        out.append("[RECALL TOP-K]")
        if recall_hits:
            for i, h in enumerate(recall_hits, 1):
                out.append(f"{i}. {h.text}")
        else:
            out.append("(empty)")
        out.append("")

        out.append("[ARCHIVAL TOP-K]")
        if archival_hits:
            for i, h in enumerate(archival_hits, 1):
                out.append(f"{i}. {h.text}")
        else:
            out.append("(empty)")

        return "\n".join(out).strip()

    async def retrieve_structured(self, query: str) -> List[MemoryHit]:
        hits: List[MemoryHit] = []

        if self.include_core_snapshot:
            core_txt = await self._get_core_snapshot()
            if core_txt.strip():
                hits.append(MemoryHit(source="core", text=core_txt.strip(), score=None, created_at_iso=None, meta=None))

        hits.extend(await self._retrieve_recall_hits(query, limit=self.k_recall))
        hits.extend(await self._retrieve_archival_hits(query, limit=self.k_archival))
        return hits

    async def _get_agent_state(self):
        return await self.agent_manager.get_agent_by_id_async(
            agent_id=self.agent_id,
            actor=self.actor,
            include_relationships=["memory", "sources"],
        )

    async def _memorize_recall(self, text: str):
        agent_state = await self._get_agent_state()
        run_id = None  # 避免 run_id 外键 warning

        msg_create = MessageCreate(
            role=MessageRole.user,
            content=[TextContent(text=text)],
        )

        msgs = await convert_message_creates_to_messages(
            message_creates=[msg_create],
            agent_id=agent_state.id,
            timezone=agent_state.timezone,
            run_id=run_id,
            wrap_user_message=False,
            wrap_system_message=False,
        )

        created = await self.message_manager.create_many_messages_async(
            pydantic_msgs=msgs,
            actor=self.actor,
            run_id=run_id,
            project_id=agent_state.project_id,
            template_id=agent_state.template_id,
            allow_partial=True,
        )
        return created

    async def _memorize_archival(self, text: str, tags: List[str]):
        agent_state = await self._get_agent_state()
        passages = await self.passage_manager.insert_passage(
            agent_state=agent_state,
            text=text,
            actor=self.actor,
            tags=tags,
            created_at=datetime.utcnow(),
            strict_mode=False,
        )
        return passages

    def _labels_from_memory(self, mem: Memory) -> List[str]:
        blocks = mem.get_blocks()
        return [b.label for b in blocks] if blocks else []

    def _pick_existing_core_block_label(self, mem: Memory) -> str:
        labels = self._labels_from_memory(mem)
        if "human" in labels:
            return "human"
        if "persona" in labels:
            return "persona"
        if labels:
            return labels[0]
        raise RuntimeError("Agent has no memory blocks")

    async def _ensure_minimum_blocks(self) -> None:
        agent_state = await self._get_agent_state()
        mem: Memory = agent_state.memory
        if mem.get_blocks():
            return

        created_block = await self.block_manager.create_or_update_block_async(
            block=Block(label="human", value="", limit=4000),
            actor=self.actor,
        )

        upd = UpdateAgent(block_ids=[created_block.id])
        await self.agent_manager.update_agent_async(
            agent_id=self.agent_id,
            agent_update=upd,
            actor=self.actor,
        )

    async def _memorize_core(self, text: str):
        await self._ensure_minimum_blocks()
        agent_state = await self._get_agent_state()
        mem: Memory = agent_state.memory

        target_label = self._pick_existing_core_block_label(mem)
        block_obj = mem.get_block(target_label)
        if not getattr(block_obj, "id", None):
            raise RuntimeError(f"Core block '{target_label}' has no id; cannot persist update.")

        old = block_obj.value or ""
        new_val = (old + "\n" + text).strip() if old.strip() else text.strip()

        updated_block = await self.block_manager.update_block_async(
            block_id=block_obj.id,
            block_update=BlockUpdate(value=new_val),
            actor=self.actor,
        )
        return updated_block

    async def _get_core_snapshot(self) -> str:
        agent_state = await self._get_agent_state()
        if not agent_state.memory.get_blocks():
            return ""
        max_files_open = getattr(agent_state, "max_files_open", None)
        txt = agent_state.memory.compile(
            tool_usage_rules=None,
            sources=getattr(agent_state, "sources", None),
            max_files_open=max_files_open,
            llm_config=getattr(agent_state, "llm_config", None),
        )
        return txt

    def _extract_message_text(self, msg) -> str:
        text_parts: List[str] = []
        if getattr(msg, "content", None):
            for c in msg.content:
                if getattr(c, "type", None) == "text":
                    t = getattr(c, "text", "")
                    if t:
                        text_parts.append(t)
        return "\n".join(text_parts).strip()

    def _tokenize_query(self, query: str) -> List[str]:
        # 最稳：按空白切分；再额外把原 query（去空格）也作为一个 token
        q = (query or "").strip()
        if not q:
            return []
        parts = [p.strip() for p in q.split() if p.strip()]
        compact = q.replace(" ", "")
        if compact and compact not in parts:
            parts.append(compact)
        return parts

    def _naive_match_score(self, text: str, tokens: List[str]) -> int:
        # 朴素评分：命中的 token 数 + 额外出现次数加分
        if not text:
            return 0
        score = 0
        for t in tokens:
            if not t:
                continue
            if t in text:
                score += 2
                score += min(text.count(t), 3)  # 最多加 3 次
        return score

    async def _retrieve_recall_hits(self, query: str, limit: int) -> List[MemoryHit]:
        if not query or not query.strip():
            return []

        # 1) 先走官方 search（fts/hybrid/vector…）
        results: List[Tuple[Any, dict]] = []
        try:
            results = await self.message_manager.search_messages_async(
                agent_id=self.agent_id,
                actor=self.actor,
                query_text=query,
                search_mode=self.recall_search_mode,
                roles=[MessageRole.user, MessageRole.assistant, MessageRole.system],
                limit=limit,
            )
        except Exception:
            results = []

        hits: List[MemoryHit] = []
        for msg, meta in results:
            txt = self._extract_message_text(msg)
            if not txt:
                continue
            score = None
            if isinstance(meta, dict):
                score = meta.get("score") or meta.get("relevance") or meta.get("distance")
            created_iso = None
            try:
                created_iso = msg.created_at.isoformat()
            except Exception:
                pass
            hits.append(MemoryHit(source="recall", text=txt, score=score, created_at_iso=created_iso, meta=meta if isinstance(meta, dict) else None))

        if hits:
            return hits

        # 2) Fallback：中文/FTS 不命中时，用“最近 user messages + 子串匹配”
        tokens = self._tokenize_query(query)

        try:
            recent = await self.message_manager.list_user_messages_for_agent_async(
                agent_id=self.agent_id,
                actor=self.actor,
                limit=max(50, limit * 10),
            )
        except Exception:
            recent = []

        scored: List[Tuple[int, str, Any]] = []
        for m in recent:
            txt = self._extract_message_text(m)
            s = self._naive_match_score(txt, tokens)
            if s > 0:
                created_iso = None
                try:
                    created_iso = m.created_at.isoformat()
                except Exception:
                    pass
                scored.append((s, created_iso or "", txt))

        # 分数高优先；同分按时间新优先
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        out: List[MemoryHit] = []
        for s, created_iso, txt in scored[:limit]:
            out.append(MemoryHit(source="recall", text=txt, score=float(s), created_at_iso=created_iso or None, meta={"fallback": "substring"}))
        return out

    async def _retrieve_archival_hits(self, query: str, limit: int) -> List[MemoryHit]:
        if not query or not query.strip():
            return []

        # 1) 先走官方 semantic/hybrid search（可能依赖 embedding）
        hits: List[MemoryHit] = []
        try:
            formatted = await self.agent_manager.search_agent_archival_memory_async(
                agent_id=self.agent_id,
                actor=self.actor,
                query=query,
                top_k=limit,
                tags=None,
                tag_match_mode="any",
                start_datetime=None,
                end_datetime=None,
            )
            for item in formatted:
                txt = (item.get("text") or item.get("passage") or "").strip()
                if not txt:
                    continue
                score = item.get("score")
                created_iso = item.get("timestamp") or item.get("created_at")
                meta = {k: v for k, v in item.items() if k not in ("text", "score", "timestamp", "created_at")}
                hits.append(MemoryHit(source="archival", text=txt, score=score, created_at_iso=created_iso, meta=meta))
        except Exception:
            hits = []

        if hits:
            return hits

        # 2) fallback：SQL 列表/过滤（返回 tuple: (passage, score, meta)）
        tokens = self._tokenize_query(query)

        try:
            tuples = await self.agent_manager.query_agent_passages_async(
                actor=self.actor,
                agent_id=self.agent_id,
                query_text=None,     # 先全取再本地过滤（最稳，不依赖 SQL/分词）
                limit=200,
                embed_query=False,
                ascending=False,
            )
        except Exception:
            tuples = []

        scored: List[Tuple[int, str, str]] = []
        for t in tuples:
            try:
                passage, score0, meta0 = t  # type: ignore
            except Exception:
                continue
            txt = (getattr(passage, "text", "") or "").strip()
            if not txt:
                continue
            s = self._naive_match_score(txt, tokens)
            if s > 0:
                created_iso = None
                try:
                    created_iso = passage.created_at.isoformat()
                except Exception:
                    created_iso = ""
                scored.append((s, created_iso or "", txt))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        out: List[MemoryHit] = []
        for s, created_iso, txt in scored[:limit]:
            out.append(MemoryHit(source="archival", text=txt, score=float(s), created_at_iso=created_iso or None, meta={"fallback": "substring"}))
        return out
