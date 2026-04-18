# M 轴源点清单（Step 2 产物）

目标：锁定“存储记忆/提取记忆”的最终执行点（底座），为后续 Adapter 封装服务。
语义对齐：MemGPT 的 working context / recall storage / archival storage（对应 blocks/messages/passages）。

源代码根：D:\MemGPT\letta
包根：D:\MemGPT\letta\letta

--------------------------------
[Core / Blocks / Working Context]

(写入-底座1：块落库)
- file: D:\MemGPT\letta\letta\services\block_manager.py
- func: BlockManager.update_block_async(...)
- line: 255

(写入-底座2：对外入口，更新 memory 并驱动 block 更新)
- file: D:\MemGPT\letta\letta\services\agent_manager.py
- func: AgentManager.update_memory_if_changed_async(...)
- line: 1629
- notes: 这里可能包含 rebuild system prompt（后续提取 M 轴时建议拆分成“纯落库更新”与“prompt 重建策略”）

(读取/渲染-底座：把 blocks 编译成可注入上下文的文本)
- file: D:\MemGPT\letta\letta\schemas\memory.py
- func: Memory.compile(...)
- line: 271

--------------------------------
[Recall / Messages]

(写入-底座)
- file: D:\MemGPT\letta\letta\services\message_manager.py
- func: MessageManager.create_many_messages_async(...)
- line: 467

(检索-底座)
- file: D:\MemGPT\letta\letta\services\message_manager.py
- func: MessageManager.search_messages_async(...)
- line: 1125

--------------------------------
[Archival / Passages]

(写入-底座)
- file: D:\MemGPT\letta\letta\services\passage_manager.py
- func: PassageManager.insert_passage(...)
- line: 543

(检索-对外入口)
- file: D:\MemGPT\letta\letta\services\agent_manager.py
- func: AgentManager.search_agent_archival_memory_async(...)
- line: 2405

--------------------------------
[工具层壳（非底座，但要识别）]
- file: D:\MemGPT\letta\letta\services\tool_executor\core_tool_executor.py
- def conversation_search : line 83
- def archival_memory_search : line 280
- def archival_memory_insert : line 309
- notes: 这些是“给 LLM 的工具执行入口”，真正的存取底座在 managers（MessageManager/PassageManager/BlockManager/AgentManager）

[PATCH] Archival 区块修正版（请删除旧 Archival 区块，保留这一段）

(写入-底座)
- file: D:\MemGPT\letta\letta\services\passage_manager.py
- func: PassageManager.insert_passage(...)
- line: 543

(检索-底座核心执行)
- file: D:\MemGPT\letta\letta\services\agent_manager.py
- func: AgentManager.query_agent_passages_async(...)
- line: 2287

(检索-对外入口/包装)
- file: D:\MemGPT\letta\letta\services\agent_manager.py
- func: AgentManager.search_agent_archival_memory_async(...)
- line: 2405

