import os

from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig

from .openai_gpt import CacheOpenAI
from .base import BaseLLM


logger = get_logger(__name__)


def _get_llm_class(config: BaseConfig):
    """
    HippoRAG LLM 工厂函数（做了懒加载处理）

    目的：
    - 避免在 Windows / 精简环境下因为未安装 bedrock / transformers 相关依赖而导致 import 失败
    - 只有当 llm_name 真的选择了 bedrock / Transformers/ 时，才导入对应模块
    """
    if config.llm_base_url is not None and "localhost" in config.llm_base_url and os.getenv("OPENAI_API_KEY") is None:
        os.environ["OPENAI_API_KEY"] = "sk-"

    if config.llm_name.startswith("bedrock"):
        try:
            from .bedrock_llm import BedrockLLM  # type: ignore
        except Exception as e:
            raise ImportError(
                "你选择了 bedrock LLM，但当前环境缺少 bedrock 依赖。"
                "请安装 HippoRAG 的 bedrock 相关依赖，或将 llm_name 改成 OpenAI 模型。"
            ) from e
        return BedrockLLM(config)

    if config.llm_name.startswith("Transformers/"):
        try:
            from .transformers_llm import TransformersLLM  # type: ignore
        except Exception as e:
            raise ImportError(
                "你选择了 Transformers/ 本地模型，但当前环境缺少 transformers/torch 等依赖。"
                "请安装对应依赖，或将 llm_name 改成 OpenAI 模型。"
            ) from e
        return TransformersLLM(config)

    return CacheOpenAI.from_experiment_config(config)