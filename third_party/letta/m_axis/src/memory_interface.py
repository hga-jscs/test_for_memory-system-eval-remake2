from abc import ABC, abstractmethod

class MemoryInterface(ABC):
    """
    Unified adapter interface (from memo大纲v2.1):
    - retrieve(query) -> str
    Optional but recommended for benchmark ingestion:
    - memorize(text, **kwargs) -> None
    """

    @abstractmethod
    def retrieve(self, query: str) -> str:
        raise NotImplementedError

    def memorize(self, text: str, **kwargs) -> None:
        # optional
        return None
