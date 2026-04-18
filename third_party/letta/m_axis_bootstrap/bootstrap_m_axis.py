import asyncio
import json

from letta.server.server import SyncServer
from letta.schemas.agent import CreateAgent
from letta.schemas.llm_config import LLMConfig
from letta.schemas.enums import AgentType

async def main():
    server = SyncServer()
    await server.init_async(init_with_default_org_and_user=True)

    actor = server.default_user
    org = server.default_org

    # 只为创建 agent 记录而给的最小 llm_config（不会真正调用）
    llm_cfg = LLMConfig(
        model="gpt-4o-mini",
        model_endpoint_type="openai",
        context_window=8192,
        model_endpoint=None,
    )

    agent_create = CreateAgent(
        name="m_axis_bootstrap_agent",
        agent_type=AgentType.letta_v1_agent,
        llm_config=llm_cfg,
        embedding_config=None,          # 第三步先不碰 embedding
        include_base_tools=False,       # 不拉工具系统
        include_default_source=False,
    )

    agent_state = await server.agent_manager.create_agent_async(
        agent_create=agent_create,
        actor=actor,
        _init_with_no_messages=True,
    )

    out = {"org_id": org.id, "user_id": actor.id, "agent_id": agent_state.id}
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
