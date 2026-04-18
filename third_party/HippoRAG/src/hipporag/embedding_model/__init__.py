"""
HippoRAG EmbeddingModel 工厂（做了懒加载处理）

目的：
- 避免在没有安装 torch / vllm / cohere 等可选依赖时 import 直接失败
- 只有当 embedding_model_name 真的选择了对应后端时才导入
"""

from ..utils.logging_utils import get_logger

from .base import EmbeddingConfig, BaseEmbeddingModel

logger = get_logger(__name__)


# 兼容老代码：尽量保留这些名字，但用 try/except 防止可选依赖导致 import 崩溃
try:
    from .Contriever import ContrieverModel  # type: ignore
except Exception:  # pragma: no cover
    ContrieverModel = None  # type: ignore

try:
    from .GritLM import GritLMEmbeddingModel  # type: ignore
except Exception:  # pragma: no cover
    GritLMEmbeddingModel = None  # type: ignore

try:
    from .NVEmbedV2 import NVEmbedV2EmbeddingModel  # type: ignore
except Exception:  # pragma: no cover
    NVEmbedV2EmbeddingModel = None  # type: ignore

try:
    from .OpenAI import OpenAIEmbeddingModel  # type: ignore
except Exception:  # pragma: no cover
    OpenAIEmbeddingModel = None  # type: ignore

try:
    from .Cohere import CohereEmbeddingModel  # type: ignore
except Exception:  # pragma: no cover
    CohereEmbeddingModel = None  # type: ignore

try:
    from .Transformers import TransformersEmbeddingModel  # type: ignore
except Exception:  # pragma: no cover
    TransformersEmbeddingModel = None  # type: ignore

try:
    from .VLLM import VLLMEmbeddingModel  # type: ignore
except Exception:  # pragma: no cover
    VLLMEmbeddingModel = None  # type: ignore


def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    if "GritLM" in embedding_model_name:
        if GritLMEmbeddingModel is None:
            raise ImportError("GritLMEmbeddingModel 导入失败：请安装对应依赖。")
        return GritLMEmbeddingModel

    if "NV-Embed-v2" in embedding_model_name:
        if NVEmbedV2EmbeddingModel is None:
            raise ImportError("NVEmbedV2EmbeddingModel 导入失败：请安装对应依赖。")
        return NVEmbedV2EmbeddingModel

    if "contriever" in embedding_model_name:
        if ContrieverModel is None:
            raise ImportError("ContrieverModel 导入失败：请安装对应依赖。")
        return ContrieverModel

    if "text-embedding" in embedding_model_name:
        if OpenAIEmbeddingModel is None:
            raise ImportError("OpenAIEmbeddingModel 导入失败：请安装 openai 及其依赖。")
        return OpenAIEmbeddingModel

    if "cohere" in embedding_model_name:
        if CohereEmbeddingModel is None:
            raise ImportError("CohereEmbeddingModel 导入失败：请安装 cohere 及其依赖。")
        return CohereEmbeddingModel

    if embedding_model_name.startswith("Transformers/"):
        if TransformersEmbeddingModel is None:
            raise ImportError("TransformersEmbeddingModel 导入失败：请安装 transformers/torch 等依赖。")
        return TransformersEmbeddingModel

    if embedding_model_name.startswith("VLLM/"):
        if VLLMEmbeddingModel is None:
            raise ImportError("VLLMEmbeddingModel 导入失败：请安装 vllm 等依赖。")
        return VLLMEmbeddingModel

    raise AssertionError(f"Unknown embedding model name: {embedding_model_name}")