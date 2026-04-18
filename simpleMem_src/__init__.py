# -*- coding: utf-8 -*-
"""SimpleMem — Naive RAG baseline for agent memory benchmarks"""

from .config import Config, get_config, reset_config
from .logger import get_logger
from .memory_interface import BaseMemorySystem, Evidence
from .llm_interface import BaseLLMClient, OpenAIClient, create_openai_client_compatible, get_embedding
from .simple_memory import SimpleRAGMemory

__all__ = [
    "Config",
    "get_config",
    "reset_config",
    "get_logger",
    "BaseMemorySystem",
    "Evidence",
    "BaseLLMClient",
    "OpenAIClient",
    "create_openai_client_compatible",
    "get_embedding",
    "SimpleRAGMemory",
]
