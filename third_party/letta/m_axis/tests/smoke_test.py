# m_axis/tests/smoke_test.py
from __future__ import annotations

import asyncio
import os

if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore
    except Exception:
        pass

from m_axis.src.bootstrap_loader import load_bootstrap_state
from m_axis.src.letta_adapter import LettaMemoryAdapter

from letta.services.user_manager import UserManager
from letta.services.organization_manager import OrganizationManager
from letta.services.message_manager import MessageManager
from letta.services.agent_manager import AgentManager


async def get_default_actor():
    org_mgr = OrganizationManager()
    user_mgr = UserManager()
    org = await org_mgr.create_default_organization_async()
    actor = await user_mgr.create_default_actor_async(org_id=org.id)
    return actor


def _msg_text(m) -> str:
    txt = ""
    if getattr(m, "content", None):
        for c in m.content:
            if getattr(c, "type", None) == "text":
                txt += (c.text or "")
    return txt.strip()


async def main():
    state = load_bootstrap_state()
    agent_id = state["agent_id"]
    actor = await get_default_actor()

    mem = LettaMemoryAdapter(
        actor=actor,
        agent_id=agent_id,
        k_recall=5,
        k_archival=5,
        recall_search_mode="fts",  # 仍然保留，但 recall 会自动 fallback 子串匹配
        include_core_snapshot=True,
    )

    # 写入
    await mem.memorize("【会议纪要修正】UI 设计截止从周五改为周三（最新）", kind="recall")
    await mem.memorize("【预算表v3】本月市场费用上限 20 万；超过需要 CTO 审批。", kind="archival", tags=["budget", "v3"])
    await mem.memorize("项目代号：Orion；负责人：小王；当前阶段：Sprint-3", kind="core")

    # DEBUG: recent user messages
    mm = MessageManager()
    recent = await mm.list_user_messages_for_agent_async(
        agent_id=agent_id,
        actor=actor,
        limit=30,
    )
    print("\n==================== DEBUG: RECENT USER MESSAGES ====================")
    for m in recent:
        print(f"- role={getattr(m,'role',None)} created_at={getattr(m,'created_at',None)} text={_msg_text(m)[:80]}")

    # DEBUG: recent passages (tuple: passage, score, meta)
    am = AgentManager()
    print("\n==================== DEBUG: RECENT PASSAGES ====================")
    try:
        tuples = await am.query_agent_passages_async(
            actor=actor,
            agent_id=agent_id,
            query_text=None,
            limit=10,
            embed_query=False,
            ascending=False,
        )
        for passage, score, meta in tuples:
            txt = (getattr(passage, "text", "") or "").strip()
            created = getattr(passage, "created_at", None)
            print(f"- created_at={created} score={score} text={txt[:80]}")
    except Exception as e:
        print(f"(passage debug skipped) error={e}")

    # Queries
    print("\n==================== QUERY 1: UI 截止 ====================")
    print(await mem.retrieve("UI 设计 截止"))

    print("\n==================== QUERY 2: 市场费用上限 ====================")
    print(await mem.retrieve("市场费用上限"))

    hits = await mem.retrieve_structured("预算 上限")
    print("\n==================== STRUCTURED HITS ====================")
    for h in hits[:10]:
        print(f"- source={h.source} score={h.score} time={h.created_at_iso} text={h.text[:80]}")


if __name__ == "__main__":
    asyncio.run(main())
