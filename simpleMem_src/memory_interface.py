# -*- coding: utf-8 -*-
"""记忆系统抽象接口"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel


class Evidence(BaseModel):
    """检索到的证据"""
    content: str
    metadata: Dict[str, Any] = {}


class BaseMemorySystem(ABC):
    """抽象记忆系统接口"""

    @abstractmethod
    def add_memory(self, data: str, metadata: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        pass

    def reset(self) -> None:
        pass
