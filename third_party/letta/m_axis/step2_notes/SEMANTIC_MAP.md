MemGPT 论文语义 -> Letta 工程落点（Step 2 只做存取底座定位）

1) Working Context（可写块） -> Core Memory Blocks（blocks）
2) Recall Storage（消息数据库） -> Messages（messages）
3) Archival Storage（长文本/段落库） -> Passages（passages）

排除项（不是本步要抽的 M 底座）：
- agents step-loop / tool 决策 / request_heartbeat（R）
- REST routers / streaming（接入层）
- compaction/summarizer（策略层，接近 R）
