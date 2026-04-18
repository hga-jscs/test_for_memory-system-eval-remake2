#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SimpleMem 快速冒烟测试"""

import sys
from pathlib import Path

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).parent))

from simpleMem_src import get_config, SimpleRAGMemory, get_logger

logger = get_logger()


def test_config():
    """测试配置加载"""
    config = get_config()
    print(f"[OK] LLM model:       {config.llm.get('model')}")
    print(f"[OK] LLM base_url:    {config.llm.get('base_url')}")
    print(f"[OK] Embedding model: {config.embedding.get('model')}")
    print(f"[OK] Embedding dim:   {config.embedding.get('dim')}")
    return config


def test_memory():
    """测试记忆系统的 add / retrieve / reset"""
    mem = SimpleRAGMemory(collection_name="test")
    print(f"\n--- 测试记忆系统 ---")

    # 添加几条记忆
    test_data = [
        ("Alice 喜欢在周末去公园跑步", {"speaker": "Alice", "topic": "hobby"}),
        ("Bob 是一名软件工程师，专注于分布式系统", {"speaker": "Bob", "topic": "work"}),
        ("Alice 最近开始学习弹钢琴", {"speaker": "Alice", "topic": "hobby"}),
        ("Bob 计划下个月去日本旅行", {"speaker": "Bob", "topic": "travel"}),
        ("Alice 和 Bob 上周一起看了一部科幻电影", {"speaker": "Alice,Bob", "topic": "activity"}),
    ]

    for content, meta in test_data:
        entry_id = mem.add_memory(content, meta)
        print(f"  添加: [{entry_id}] {content[:40]}")

    print(f"\n记忆总数: {mem.size}")

    # 检索测试
    queries = [
        "Alice 的兴趣爱好是什么？",
        "Bob 的工作是什么？",
        "他们最近一起做了什么？",
    ]

    for q in queries:
        print(f"\n查询: {q}")
        results = mem.retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            score = r.metadata.get("score", 0)
            print(f"  [{i}] (score={score:.4f}) {r.content[:60]}")

    # 重置测试
    mem.reset()
    print(f"\n重置后记忆数: {mem.size}")
    assert mem.size == 0, "重置后应为空"
    print("[OK] 记忆系统测试通过")


if __name__ == "__main__":
    print("=" * 50)
    print("SimpleMem 冒烟测试")
    print("=" * 50)

    try:
        test_config()
        test_memory()
        print("\n" + "=" * 50)
        print("全部测试通过!")
        print("=" * 50)
    except Exception as e:
        print(f"\n[FAIL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
