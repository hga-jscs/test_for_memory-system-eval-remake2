# -*- coding: utf-8 -*-
"""配置管理 — 从项目根目录 config.yaml 加载 LLM / Embedding 配置"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from benchmark_io_utils import read_text_with_fallback
from .logger import get_logger

_config: Optional["Config"] = None


class Config:
    def __init__(self, config_path: Optional[Path] = None):
        self._logger = get_logger()
        self._data: Dict[str, Any] = {}

        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        self._data = yaml.safe_load(read_text_with_fallback(config_path)) or {}

        self._logger.info("配置加载完成: %s", config_path)

    @property
    def llm(self) -> Dict[str, Any]:
        return self._data.get("llm", {})

    @property
    def embedding(self) -> Dict[str, Any]:
        return self._data.get("embedding", {})

    @property
    def database(self) -> Dict[str, Any]:
        return self._data.get("database", {})

    @property
    def raw(self) -> Dict[str, Any]:
        return self._data

    def get_prompt(self, adaptor: str, template_name: str) -> str:
        """获取推理范式的 Prompt 模板（从 prompts.yaml）"""
        if not hasattr(self, "_prompts"):
            prompts_path = Path(__file__).parent.parent / "prompts.yaml"
            if prompts_path.exists():
                self._prompts = yaml.safe_load(read_text_with_fallback(prompts_path)) or {}
            else:
                self._prompts = {}
        if adaptor not in self._prompts:
            raise KeyError(f"未知的适配器: {adaptor}")
        if template_name not in self._prompts[adaptor]:
            raise KeyError(f"未知的模板: {adaptor}.{template_name}")
        return self._prompts[adaptor][template_name]


def get_config(config_path: Optional[Path] = None) -> Config:
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def reset_config():
    global _config
    _config = None
